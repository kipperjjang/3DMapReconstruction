"""Equation (1) of Hoeller et al., arXiv:2206.08077v1.

Coordinates are integer cell indices; features are the centroid's fractional
position inside the cell. Coordinates include no batch or temporal column here.
"""

from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass(frozen=True)
class VoxelGrid:
    voxel_size: float = 0.05
    shape: Tuple[int, int, int] = (64, 64, 64)
    # The paper does not specify the grid's translation relative to the robot.
    # Supply the lower corner explicitly; this default centers all three axes.
    origin: Tuple[float, float, float] = (-1.6, -1.6, -1.6)

    def __post_init__(self):
        if self.voxel_size <= 0:
            raise ValueError("voxel_size must be positive")
        if len(self.shape) != 3 or any(s <= 0 or int(s) != s for s in self.shape):
            raise ValueError("shape must contain three positive integers")
        if len(self.origin) != 3:
            raise ValueError("origin must contain three coordinates")


@dataclass
class VoxelizedCloud:
    coordinates: torch.Tensor
    offsets: torch.Tensor
    counts: torch.Tensor

    def points(self, grid: VoxelGrid) -> torch.Tensor:
        """Decode one centroid per occupied voxel, in metres."""
        origin = self.offsets.new_tensor(grid.origin)
        return (self.coordinates.to(self.offsets) + self.offsets) * grid.voxel_size + origin


def _check_points(points: torch.Tensor):
    if not isinstance(points, torch.Tensor) or points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must be a floating point torch.Tensor with shape [N, 3]")
    if not points.is_floating_point():
        raise ValueError("points must use a floating point dtype")
        
    if not torch.isfinite(points).all():
        raise ValueError("points must contain only finite coordinates")


def voxelize(points: torch.Tensor, grid: VoxelGrid = VoxelGrid()) -> VoxelizedCloud:
    """Crop to a half-open grid and average points within each occupied voxel.

    Grouping uses floor(point coordinates) before averaging, including for
    negative world coordinates. Features retain gradients to the point positions.
    """
    _check_points(points)
    normalized = (points - points.new_tensor(grid.origin)) / grid.voxel_size
    inside = ((normalized >= 0) & (normalized < normalized.new_tensor(grid.shape))).all(1)
    normalized = normalized[inside]
    cells = normalized.floor().to(torch.int32)
    if not len(cells):
        return VoxelizedCloud(cells, normalized, torch.empty(0, dtype=torch.long, device=points.device))
    coordinates, inverse, counts = torch.unique(cells, dim=0, return_inverse=True, return_counts=True)
    sums = normalized.new_zeros((len(coordinates), 3)).index_add(0, inverse, normalized)
    offsets = sums / counts[:, None].to(sums) - coordinates.to(sums)
    return VoxelizedCloud(coordinates, offsets, counts)


def transform_points(points: torch.Tensor, previous_to_current: torch.Tensor) -> torch.Tensor:
    """Apply a rigid 4x4 transform mapping previous-frame points to this frame.

    If robot poses map local points into the world, this transform is
    inverse(world_from_current) @ world_from_previous.
    """
    _check_points(points)
    transform = torch.as_tensor(previous_to_current, dtype=points.dtype, device=points.device)
    if transform.shape != (4, 4) or not torch.isfinite(transform).all():
        raise ValueError("previous_to_current must be a finite 4x4 matrix")
    if not torch.allclose(transform[3], transform.new_tensor([0, 0, 0, 1])):
        raise ValueError("transform must have homogeneous last row [0, 0, 0, 1]")
    rotation = transform[:3, :3]
    if not torch.allclose(rotation.T @ rotation, torch.eye(3, device=points.device, dtype=points.dtype), atol=1e-4):
        raise ValueError("transform rotation must be orthonormal")
    if not torch.allclose(torch.det(rotation), rotation.new_tensor(1), atol=1e-4):
        raise ValueError("transform rotation must have determinant +1")
    return points @ rotation.T + transform[:3, 3]
