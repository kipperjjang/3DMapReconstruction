"""Multi-scale occupancy BCE and sub-voxel Euclidean reconstruction loss."""

from dataclasses import dataclass
from typing import Sequence

import torch
from torch.nn import functional as F

from .inputs import point_tensor
from .voxel import voxelize


def spatial_keys(coordinates, shape):
    """Encode [batch,x,y,z,(time)] without mixing batch members."""
    c = coordinates[:, :4].to(torch.long)
    return ((c[:, 0] * shape[0] + c[:, 1]) * shape[1] + c[:, 2]) * shape[2] + c[:, 3]


def match_keys(query, reference):
    """Return membership and original reference row indices, in query order."""
    if not len(reference):
        return torch.zeros_like(query, dtype=torch.bool), torch.zeros_like(query)
    sorted_keys, order = torch.sort(reference)
    positions = torch.searchsorted(sorted_keys, query).clamp(max=len(reference)-1)
    return sorted_keys[positions] == query, order[positions]


def target_voxels(targets: Sequence[torch.Tensor], grid, device):
    coords, offsets = [], []
    for batch, points in enumerate(targets):
        cloud = voxelize(point_tensor(points, device), grid)
        batch_ids = torch.full((len(cloud.coordinates), 1), batch, dtype=torch.int32, device=device)
        coords.append(torch.cat((batch_ids, cloud.coordinates), 1))
        offsets.append(cloud.offsets)
    if not targets:
        raise ValueError("At least one target batch member is required")
    return torch.cat(coords), torch.cat(offsets)


def occupancy_labels(candidate_coordinates, target_coordinates, stride, grid):
    """Supervise coarse spatial occupancy with the current scene.

    Each occupied target cell maps to its enclosing coarse spatial cell. Time
    does not enter the occupancy label; the merged input policy fixes it at zero.
    """
    targets = target_coordinates.clone()
    stride_tensor = targets.new_tensor(stride[:3])
    targets[:, 1:4] = torch.div(targets[:, 1:4], stride_tensor, rounding_mode="floor") * stride_tensor
    labels, _ = match_keys(spatial_keys(candidate_coordinates, grid.shape), spatial_keys(targets, grid.shape))
    return labels


@dataclass
class ReconstructionLoss:
    total: torch.Tensor
    occupancy: torch.Tensor
    offset: torch.Tensor
    matched_voxels: int


def reconstruction_loss(result, targets, occupancy_weight=1.0, offset_weight=1.0):
    """BCE averaged over decoder stages plus mean Euclidean offset error.

    Stage reduction and relative loss weights are unspecified in the PDF. The
    weights are explicit arguments; defaults are equal. Offset distance is in
    voxel units, and multiplying it by voxel_size converts it to metres.
    Predicted support and GT support are matched by coordinates, never row order.
    """
    if len(targets) != result.batch_size:
        raise ValueError("One target point cloud is required per batch member")
    if occupancy_weight < 0 or offset_weight < 0:
        raise ValueError("Loss weights must be nonnegative")
    device = result.offsets.F.device
    target_coordinates, target_offsets = target_voxels(targets, result.grid, device)
    zero = result.offsets.F.sum() * 0
    stage_losses = []
    for stage in result.stages:
        logits = stage.logits.F[:, 0]
        if not len(logits):
            continue
        coordinates = stage.logits.C.to(device)
        labels = occupancy_labels(coordinates, target_coordinates, stage.logits.tensor_stride, result.grid)
        stage_losses.append(F.binary_cross_entropy_with_logits(logits, labels.to(logits.dtype)))
    occupancy = torch.stack(stage_losses).mean() if stage_losses else zero
    matched, target_rows = match_keys(spatial_keys(result.offsets.C.to(device), result.grid.shape), spatial_keys(target_coordinates, result.grid.shape))
    if matched.any():
        error = result.offsets.F[matched] - target_offsets[target_rows[matched]]
        offset = torch.linalg.vector_norm(error, dim=1).mean()
    else:
        offset = zero
    return ReconstructionLoss(occupancy_weight * occupancy + offset_weight * offset, occupancy, offset, int(matched.sum().item()))
