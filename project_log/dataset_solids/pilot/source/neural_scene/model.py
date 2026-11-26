"""Generative sparse U-Net terrain reconstruction with predictive pruning.

Paper backbone: arXiv:2206.08077v1, Figure 2 and Section III.B.
Current observations use t=0; aligned previous predictions use t=1.
"""

from dataclasses import dataclass
from typing import List, Tuple

import torch
from torch import nn
import MinkowskiEngine as ME

from .blocks import ConvBlock, SpatialDownsample, SpatialUpsample, union_concat
from .inputs import INPUT_POLICIES, MERGED_POLICY, TEMPORAL_POLICY, ReconstructionInput
from .loss import match_keys, spatial_keys
from .voxel import VoxelGrid


@dataclass(frozen=True)
class UNetConfig:
    channels: Tuple[int, int, int, int, int] = (32, 64, 128, 256, 256)
    pruning_threshold: float = 0.5
    input_policy: str = TEMPORAL_POLICY

    def __post_init__(self):
        if len(self.channels) != 5 or any(c <= 0 or int(c) != c for c in self.channels):
            raise ValueError("channels must contain five positive integers")
        if not 0 <= self.pruning_threshold <= 1:
            raise ValueError("pruning_threshold must be in [0, 1]")
        if self.input_policy not in INPUT_POLICIES:
            raise ValueError("Unsupported input policy: " + str(self.input_policy))


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
    stride: Tuple[int, int, int, int] = (1, 1, 1, 1)

    @property
    def tensor_stride(self):
        """Return the spatial/temporal stride of the final output."""
        return list(self.stride)

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

    Input is [batch,x,y,z,time] with three centroid offsets. All spatial down/up
    strides are [2,2,2,1]. The temporal policy collapses t=0/1 at the offset head.
    Generated coordinates survive only if occupancy >= alpha. No
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
        temporal = config.input_policy == TEMPORAL_POLICY
        self.offset_head = ME.MinkowskiConvolution(channels[0], 3,
            kernel_size=[1, 1, 1, 2] if temporal else 1,
            stride=[1, 1, 1, 2] if temporal else 1, bias=True, dimension=4)
        self.sigmoid = ME.MinkowskiSigmoid()
        self.prune = ME.MinkowskiPruning()

    def _empty_result(self, packed, stages):
        # A graph-connected empty tensor allows a zero loss on an empty input.
        zero = self.offset_head.kernel.sum() * 0
        features = self.offset_head.kernel.new_empty((0, 3)) + zero
        coordinates = torch.empty((0, 5), dtype=torch.int32, device=features.device)
        stride = (1, 1, 1, 2) if self.config.input_policy == TEMPORAL_POLICY else (1, 1, 1, 1)
        offsets = EmptyOffsets(F=features, C=coordinates, stride=stride)
        return ReconstructionResult(offsets, features.new_empty(0), stages, packed.batch_size, packed.grid)

    def forward(self, packed: ReconstructionInput):
        if not isinstance(packed, ReconstructionInput):
            raise TypeError("Use merged_input() to construct a ReconstructionInput")
        if any(size % 16 != 0 for size in packed.grid.shape):
            raise ValueError("Grid dimensions must be divisible by 16")
        if packed.input_policy != self.config.input_policy:
            raise ValueError("Packed input and model input policies differ")
        if len(packed.coordinates):
            times = packed.coordinates[:, -1]
            maximum = 0 if self.config.input_policy == MERGED_POLICY else 1
            if ((times < 0) | (times > maximum)).any():
                raise ValueError("Input time coordinates are outside the configured policy")
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
        keys, inverse = torch.unique(spatial_keys(final_logits.C.to(device), packed.grid.shape), return_inverse=True)
        probabilities = final_logits.F[:, 0].sigmoid()
        # The two temporal rows can collapse onto one spatial output voxel.
        confidence_by_key = probabilities.new_zeros(len(keys)).scatter_reduce_(
            0, inverse, probabilities, reduce="amax", include_self=False)
        matched, rows = match_keys(spatial_keys(offsets.C.to(device), packed.grid.shape), keys)
        if not matched.all():
            raise RuntimeError("Final points must have an occupancy prediction")
        confidence = confidence_by_key[rows]
        return ReconstructionResult(offsets, confidence, stages, packed.batch_size, packed.grid)
