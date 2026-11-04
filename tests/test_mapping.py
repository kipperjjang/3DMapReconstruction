import unittest

import numpy as np
import torch
from torch import nn

from neural_scene.geometry import previous_to_current
from neural_scene.mapping import NeuralSceneMapper, PointCloudFrame
from neural_scene.voxel import VoxelGrid


class FixedPrediction(nn.Module):
    """Controlled predictions isolate frame/state behavior from learned weights."""

    def __init__(self):
        super().__init__()
        self.prediction = nn.Parameter(torch.tensor([[0.4, 0.3, 0.2]]))
        self.inputs = []

    def forward(self, packed):
        self.inputs.append(packed)
        prediction = self.prediction * 1
        class Output:
            def points(self):
                return [prediction]
        return Output()


class MappingTests(unittest.TestCase):
    def test_feedback_is_prediction_aligned_with_dynamapping_convention(self):
        model = FixedPrediction()
        mapper = NeuralSceneMapper(model, 10)
        previous = PointCloudFrame([[1., 1., 1.]], [4, 2, 0, 0, 0, 0, 1], 0)
        current = PointCloudFrame([[0., 0., 0.]], [4.1, 2, 0, 0, 0, np.sin(0.3), np.cos(0.3)], 100000000)
        first = mapper.update(previous, track_grad=True)
        self.assertTrue(first.points.requires_grad)
        self.assertFalse(mapper.previous_prediction.requires_grad)
        step = mapper.update(current)
        expected_transform = previous_to_current(current.robot_pose, previous.robot_pose)
        expected_points = np.array([[0.4, 0.3, 0.2]]) @ expected_transform[:3, :3].T + expected_transform[:3, 3]
        np.testing.assert_allclose(step.aligned_previous.numpy(), expected_points, atol=1e-6)
        self.assertAlmostEqual(step.dt_s, 0.1)
        self.assertEqual(model.inputs[0].coordinates[:, -1].tolist(), [0])
        self.assertEqual(model.inputs[1].coordinates[:, -1].tolist(), [0, 1])

    def test_order_rejection_and_episode_reset(self):
        mapper = NeuralSceneMapper(FixedPrediction(), 10, VoxelGrid())
        frame = PointCloudFrame([[0., 0., 0.]], np.eye(4), 100)
        mapper.update(frame)
        with self.assertRaises(ValueError):
            mapper.update(frame)
        mapper.reset()
        self.assertEqual(len(mapper.update(frame).aligned_previous), 0)


if __name__ == "__main__":
    unittest.main()
