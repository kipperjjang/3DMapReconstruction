"""Pack aligned current/previous clouds into sparse space-time coordinates."""

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import torch

from .voxel import VoxelGrid, voxelize

MERGED_POLICY = "aligned_merge_t0"
TEMPORAL_POLICY = "aligned_temporal_t01"
INPUT_POLICIES = (MERGED_POLICY, TEMPORAL_POLICY)


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
    input_policy: str = TEMPORAL_POLICY

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
    input_policy: str = TEMPORAL_POLICY,
) -> ReconstructionInput:
    """Voxelize current (t=0) and aligned previous predictions (t=1) separately.

    The legacy aligned_merge_t0 policy averages both sources in the same voxel.
    Previous points must already be in the current robot frame, using
    geometry.previous_to_current(). Empty scans and empty batch members are
    allowed. Detachment of recurrent predictions is handled by the caller.
    """
    if not len(measurements):
        raise ValueError("At least one batch member is required")
    if aligned_previous is not None and len(aligned_previous) != len(measurements):
        raise ValueError("measurements and aligned_previous must have equal batch sizes")
    if input_policy not in INPUT_POLICIES:
        raise ValueError("Unsupported input policy: " + str(input_policy))
    reference = point_tensor(measurements[0])
    coordinates: List[torch.Tensor] = []
    features: List[torch.Tensor] = []
    for batch, measurement in enumerate(measurements):
        current = point_tensor(measurement, reference.device)
        previous = current.new_empty((0, 3)) if aligned_previous is None else point_tensor(aligned_previous[batch], reference.device)
        sources = [(0, torch.cat((current, previous)))] if input_policy == MERGED_POLICY else [(0, current), (1, previous)]
        for time, points in sources:
            cloud = voxelize(points, grid)
            batch_column = torch.full((len(cloud.coordinates), 1), batch, dtype=torch.int32, device=reference.device)
            time_column = torch.full_like(batch_column, time)
            coordinates.append(torch.cat((batch_column, cloud.coordinates, time_column), dim=1))
            features.append(cloud.offsets)
    return ReconstructionInput(torch.cat(coordinates), torch.cat(features), len(measurements), grid, input_policy)
