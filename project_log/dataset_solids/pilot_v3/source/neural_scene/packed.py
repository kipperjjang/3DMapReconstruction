"""Compact point storage retaining voxel identities and observation multiplicity.

Each point stores three uint8 cell coordinates and three uint8 within-cell
offsets. Midpoint decoding has <=0.05/512 m error per axis (0.170 mm in 3D).
"""

import numpy as np

ORIGIN = np.array([-1.6, -1.6, -1.6], dtype=np.float32)
SIZE = 0.05


def cells(points):
    """Return model-compatible cell coordinates for robot-frame metre points."""
    return np.floor((np.asarray(points, dtype=np.float32) - ORIGIN) / np.float32(SIZE)).astype(np.int32)


def crop(points):
    """Keep finite points in the half-open 64-cubed training grid."""
    points = np.asarray(points, dtype=np.float32)
    valid = np.isfinite(points).all(axis=1)
    points = points[valid]
    c = cells(points)
    return points[((c >= 0) & (c < 64)).all(axis=1)]


def keys(points):
    """Return integer occupied-cell keys, without deduplicating observations."""
    c = cells(points)
    return (c[:, 0] * 64 + c[:, 1]) * 64 + c[:, 2]


def centroids(points):
    """Reduce surface samples to one float32 centroid per training voxel."""
    if not len(points):
        return np.empty((0, 3), dtype=np.float32)
    _, inv, count = np.unique(keys(points), return_inverse=True, return_counts=True)
    return np.stack([np.bincount(inv, weights=points[:, axis]) / count for axis in range(3)], axis=1).astype(np.float32)


def encode(points):
    """Encode in spatial order; keep every duplicate measurement and cell identity."""
    points = np.asarray(points, dtype=np.float32)
    c = cells(points)
    if not np.isfinite(points).all() or not ((c >= 0) & (c < 64)).all():
        raise ValueError('Packed points must be finite and inside the grid')
    offset = (points.astype(np.float64) - ORIGIN.astype(np.float64)) / SIZE - c
    q = np.clip(np.floor(offset * 256), 0, 255).astype(np.uint8)
    order = np.lexsort((q[:, 2], q[:, 1], q[:, 0], c[:, 2], c[:, 1], c[:, 0]))
    return np.concatenate((c.astype(np.uint8), q), axis=1)[order]


def decode(packed):
    """Decode into robot-frame metres at within-cell bin midpoints."""
    packed = np.asarray(packed)
    if packed.ndim != 2 or packed.shape[1] != 6 or packed.dtype != np.uint8:
        raise ValueError('Expected uint8 point records with six columns')
    if np.any(packed[:, :3] >= 64):
        raise ValueError('Invalid packed grid coordinate')
    return ((packed[:, :3].astype(np.float64) + (packed[:, 3:] + 0.5) / 256) * SIZE + ORIGIN).astype(np.float32)


def read_frame(path, index):
    """Read one frame from a packed HDF5 sequence without loading the full shard."""
    import h5py
    with h5py.File(str(path), 'r') as file:
        result = {}
        for kind, field in (('current', 'current_points'), ('target', 'target_points')):
            begin, end = file[kind + '_offsets'][index:index + 2]
            result[field] = decode(file[kind][:, int(begin):int(end)].T)
        for field in ('robot_pose', 'sensor_pose', 'transformation', 'timestamp_ns'):
            result[field] = file[field][index]
        return result
