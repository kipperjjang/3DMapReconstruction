"""Merge aligned point clouds and pack 4D COO with a single t=0 slice."""

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import torch

from .voxel import VoxelGrid, voxelize


def point_tensor(points, device=None):
    """Accept torch or DynaMapping's possibly read-only NumPy point arrays."""
    if isinstance(points, torch.Tensor):
        return points.to(device=device, dtype=torch.float32)
    return torch.as_tensor(np.array(points, dtype=np.float32, copy=True), device=device)


@dataclass
class ReconstructionInput:
    coordinates: torch.Tensor  # [N, 5], batch, x, y, z, time
    features: torch.Tensor  # [N, 3], fractional centroid offset
    batch_size: int
    grid: VoxelGrid

    def sparse(self):
        try:
            import MinkowskiEngine as ME
        except ImportError as error:
            raise ImportError("Build MinkowskiEngine with scripts/build_minkowski.py and set PYTHONPATH=.deps") from error
        # Place both tensors on the target device to select the correct backend.
        return ME.SparseTensor(features=self.features, coordinates=self.coordinates.to(self.features.device))


def merged_input(
    measurements: Sequence[torch.Tensor],
    aligned_previous: Optional[Sequence[torch.Tensor]] = None,
    grid: VoxelGrid = VoxelGrid(),
) -> ReconstructionInput:
    """Merge current and aligned previous predictions BEFORE voxelization.

    All input sites have time coordinate zero, per the user's chosen adaptation.
    Previous points must already be in the current robot frame, using
    geometry.previous_to_current(). Empty scans and empty batch members are
    allowed. Detachment of recurrent predictions is handled by the caller.
    """
    if not len(measurements):
        raise ValueError("At least one batch member is required")
    if aligned_previous is not None and len(aligned_previous) != len(measurements):
        raise ValueError("measurements and aligned_previous must have equal batch sizes")
    reference = point_tensor(measurements[0])
    coordinates: List[torch.Tensor] = []
    features: List[torch.Tensor] = []
    for batch, measurement in enumerate(measurements):
        current = point_tensor(measurement, reference.device)
        previous = current.new_empty((0, 3)) if aligned_previous is None else point_tensor(aligned_previous[batch], reference.device)
        cloud = voxelize(torch.cat((current, previous)), grid)
        batch_column = torch.full((len(cloud.coordinates), 1), batch, dtype=torch.int32, device=reference.device)
        time_column = torch.zeros_like(batch_column)
        coordinates.append(torch.cat((batch_column, cloud.coordinates, time_column), dim=1))
        features.append(cloud.offsets)
    return ReconstructionInput(torch.cat(coordinates), torch.cat(features), len(measurements), grid)
