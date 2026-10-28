import unittest

import numpy as np

from neural_scene.geometry import inverse, pose_matrix, previous_to_current


class GeometryTests(unittest.TestCase):
    def test_alignment_preserves_world_point_with_translation_and_rotation(self):
        previous = pose_matrix([4, -1, 2, 0, 0, np.sin(0.3), np.cos(0.3)])
        current = pose_matrix([5, 2, 1, 0, 0, np.sin(-0.5), np.cos(-0.5)])
        point = np.array([0.2, -0.7, 0.5, 1])
        aligned = previous_to_current(current, previous) @ point
        np.testing.assert_allclose(current @ aligned, previous @ point, atol=1e-12)

    def test_initial_robot_world_origin_cancels(self):
        previous = pose_matrix([4, -1, 2, 0, 0, np.sin(0.3), np.cos(0.3)])
        current = pose_matrix([5, 2, 1, 0, 0, np.sin(-0.5), np.cos(-0.5)])
        initial = pose_matrix([-3, 1, -2, np.sin(0.4), 0, 0, np.cos(0.4)])
        np.testing.assert_allclose(previous_to_current(current, previous, initial), inverse(current) @ previous, atol=1e-12)

    def test_xyzw_quaternion_normalization(self):
        np.testing.assert_allclose(pose_matrix([1, 2, 3, 0, 0, 0, 2]), pose_matrix([1, 2, 3, 0, 0, 0, 1]))

    def test_missing_or_invalid_pose_is_rejected(self):
        for pose in (None, [0]*7, np.zeros((4, 4))):
            with self.assertRaises(ValueError):
                pose_matrix(pose)


if __name__ == "__main__":
    unittest.main()
