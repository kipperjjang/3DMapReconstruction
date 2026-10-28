"""Generative sparse U-Net terrain reconstruction with predictive pruning.

Paper backbone: arXiv:2206.08077v1, Figure 2 and Section III.B.
User adaptation: scans and aligned previous predictions are merged at t=0.
"""

from dataclasses import dataclass
from typing import List, Tuple

import torch
from torch import nn
import MinkowskiEngine as ME

from .blocks import ConvBlock, SpatialDownsample, SpatialUpsample, union_concat
from .inputs import ReconstructionInput
from .loss import match_keys, spatial_keys
from .voxel import VoxelGrid


@dataclass(frozen=True)
class UNetConfig:
    channels: Tuple[int, int, int, int, int] = (32, 64, 128, 256, 256)
    pruning_threshold: float = 0.5

    def __post_init__(self):
        if len(self.channels) != 5 or any(c <= 0 or int(c) != c for c in self.channels):
            raise ValueError("channels must contain five positive integers")
        if not 0 <= self.pruning_threshold <= 1:
            raise ValueError("pruning_threshold must be in [0, 1]")


@dataclass
class OccupancyStage:
    logits: object  # SparseTensor, before pruning
    keep: torch.Tensor  # model probabilities only; no GT injection


@dataclass
class EmptyOffsets:
    """Empty output rows without constructing a MinkowskiEngine coordinate map.

    ME 0.5.4 launches cuda_copy_n with zero blocks when registering an empty
    CUDA coordinate map. Empty predictions need only tensors for loss, metrics
    and feedback; they never pass through another sparse convolution.
    """

    F: torch.Tensor
    C: torch.Tensor

    @property
    def tensor_stride(self):
        """Return the spatial/temporal stride of the final output."""
        return [1, 1, 1, 1]

    def __len__(self):
        """Return the number of output feature rows."""
        return len(self.F)


@dataclass
class ReconstructionResult:
    offsets: object  # SparseTensor or EmptyOffsets [N,3], stride one and t=0
    confidence: torch.Tensor  # same row order as offsets
    stages: List[OccupancyStage]
    batch_size: int
    grid: VoxelGrid

    def points(self):
        """Return one [N_b,3] robot-frame cloud per batch member, in metres."""
        coords = self.offsets.C.to(self.offsets.F.device)
        positions = (coords[:, 1:4].to(self.offsets.F) + self.offsets.F) * self.grid.voxel_size + self.offsets.F.new_tensor(self.grid.origin)
        return [positions[coords[:, 0] == batch] for batch in range(self.batch_size)]


class ReconstructionUNet4D(nn.Module):
    """Four spatial down/up stages with two ordinary convolutions per block.

    Input is [batch,x,y,z,0] with three centroid offsets. All down/up strides are
    [2,2,2,1]. Generated coordinates survive only if occupancy >= alpha. No
    training-time GT retention or minimum-number-of-voxels fallback is used.
    """

    def __init__(self, config=UNetConfig()):
        super().__init__()
        self.config = config
        channels = config.channels
        self.encoders = nn.ModuleList([ConvBlock(3 if i == 0 else channels[i], channels[i]) for i in range(5)])
        self.downsamples = nn.ModuleList([SpatialDownsample(channels[i], channels[i+1]) for i in range(4)])
        self.upsamples = nn.ModuleList([SpatialUpsample(channels[i+1], channels[i]) for i in range(3, -1, -1)])
        self.decoders = nn.ModuleList([ConvBlock(2 * channels[i], channels[i]) for i in range(3, -1, -1)])
        self.occupancy_heads = nn.ModuleList([ME.MinkowskiConvolution(channels[i], 1, kernel_size=1, bias=True, dimension=4) for i in range(3, -1, -1)])
        self.offset_head = ME.MinkowskiConvolution(channels[0], 3, kernel_size=1, bias=True, dimension=4)
        self.sigmoid = ME.MinkowskiSigmoid()
        self.prune = ME.MinkowskiPruning()

    def _empty_result(self, packed, stages):
        # A graph-connected empty tensor allows a zero loss on an empty input.
        zero = self.offset_head.kernel.sum() * 0
        features = self.offset_head.kernel.new_empty((0, 3)) + zero
        coordinates = torch.empty((0, 5), dtype=torch.int32, device=features.device)
        offsets = EmptyOffsets(F=features, C=coordinates)
        return ReconstructionResult(offsets, features.new_empty(0), stages, packed.batch_size, packed.grid)

    def forward(self, packed: ReconstructionInput):
        if not isinstance(packed, ReconstructionInput):
            raise TypeError("Use merged_input() to construct a ReconstructionInput")
        if any(size % 16 != 0 for size in packed.grid.shape):
            raise ValueError("Grid dimensions must be divisible by 16")
        if len(packed.coordinates) and (packed.coordinates[:, -1] != 0).any():
            raise ValueError("This model expects merged input with all time coordinates t=0")
        if packed.features.device != self.offset_head.kernel.device:
            raise ValueError("Input features and model must be on the same device")
        if not len(packed.coordinates):
            return self._empty_result(packed, [])
        feature = packed.sparse()
        skips = []
        for level in range(5):
            feature = self.encoders[level](feature)
            if level < 4:
                skips.append(feature)
                feature = self.downsamples[level](feature)
        stages = []
        for level in range(4):
            feature = self.upsamples[level](feature)
            feature = union_concat(feature, skips[3-level])
            feature = self.decoders[level](feature)
            logits = self.occupancy_heads[level](feature)
            keep = logits.F[:, 0].sigmoid() >= self.config.pruning_threshold
            stages.append(OccupancyStage(logits, keep))
            if not keep.any():
                return self._empty_result(packed, stages)
            feature = self.prune(feature, keep)
        offsets = self.sigmoid(self.offset_head(feature))
        # Pruning may reorder rows: match by coordinates to align confidences.
        final_logits = stages[-1].logits
        device = offsets.F.device
        matched, rows = match_keys(spatial_keys(offsets.C.to(device), packed.grid.shape), spatial_keys(final_logits.C.to(device), packed.grid.shape))
        if not matched.all():
            raise RuntimeError("Final points must have an occupancy prediction")
        confidence = final_logits.F[rows, 0].sigmoid()
        return ReconstructionResult(offsets, confidence, stages, packed.batch_size, packed.grid)
