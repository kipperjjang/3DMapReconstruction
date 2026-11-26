"""DynaMapping-compatible robot pose conventions, usable with Python 3.7.

T_A_B maps coordinates in frame B into frame A. Poses are T_world_robot;
seven-element poses store XYZ translation followed by XYZW quaternion.
"""

import numpy as np


def pose_matrix(pose):
    """Validate and copy a right-handed rigid pose or normalize its quaternion."""
    value = np.asarray(pose, dtype=np.float64)
    if not np.isfinite(value).all():
        raise ValueError("Pose must be finite")
    if value.shape == (4, 4):
        rotation = value[:3, :3]
        if (not np.allclose(value[3], [0, 0, 0, 1], atol=1e-8)
                or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
                or not np.isclose(np.linalg.det(rotation), 1, atol=1e-6)):
            raise ValueError("Pose matrix must be a rigid right-handed transform")
        return value.copy()
    if value.shape != (7,):
        raise ValueError("Pose must be a 4x4 matrix or XYZ + XYZW quaternion")
    norm = np.linalg.norm(value[3:])
    if norm < 1e-12:
        raise ValueError("Quaternion must not be zero")
    x, y, z, w = value[3:] / norm
    result = np.eye(4)
    result[:3, :3] = [
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ]
    result[:3, 3] = value[:3]
    return result


def inverse(transform):
    """Rigid inverse, with DynaMapping's transpose-based calculation."""
    transform = pose_matrix(transform)
    result = np.eye(4)
    result[:3, :3] = transform[:3, :3].T
    result[:3, 3] = -result[:3, :3] @ transform[:3, 3]
    return result


def previous_to_current(current_pose, previous_pose, initial_pose=None):
    """Align the previous robot frame directly to the current robot frame.

    Like DynaMapping.update(), optionally express both poses relative to the
    initial robot pose first. The shared world origin cancels in the result.
    Missing poses raise an error; no interpolation or estimation is performed.
    """
    current = pose_matrix(current_pose)
    previous = pose_matrix(previous_pose)
    if initial_pose is not None:
        to_initial_world = inverse(initial_pose)
        current = to_initial_world @ current
        previous = to_initial_world @ previous
    rotation = current[:3, :3].T
    result = np.eye(4)
    result[:3, :3] = rotation @ previous[:3, :3]
    result[:3, 3] = rotation @ (previous[:3, 3] - current[:3, 3])
    return result
