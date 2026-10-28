import unittest

import torch

from neural_scene.voxel import VoxelGrid, transform_points, voxelize


class VoxelTests(unittest.TestCase):
    def test_centroid_offsets_and_world_round_trip(self):
        grid = VoxelGrid(voxel_size=0.5, shape=(4, 4, 4), origin=(-1, -1, -1))
        points = torch.tensor([[-0.9, -0.8, -0.7], [-0.6, -0.7, -0.8], [0.2, 0.3, 0.4]])
        cloud = voxelize(points, grid)
        self.assertTrue(torch.equal(cloud.coordinates, torch.tensor([[0, 0, 0], [2, 2, 2]], dtype=torch.int32)))
        self.assertEqual(cloud.counts.tolist(), [2, 1])
        self.assertTrue(torch.allclose(cloud.offsets, torch.tensor([[0.5, 0.5, 0.5], [0.4, 0.6, 0.8]])))
        self.assertTrue(torch.allclose(cloud.points(grid), torch.stack((points[:2].mean(0), points[2]))))

    def test_lower_inclusive_upper_exclusive_crop(self):
        grid = VoxelGrid(voxel_size=1, shape=(2, 2, 2), origin=(0, 0, 0))
        cloud = voxelize(torch.tensor([[0., 0., 0.], [2., 0., 0.], [-0.1, 0., 0.]]), grid)
        self.assertEqual(len(cloud.coordinates), 1)
        self.assertEqual(cloud.coordinates.tolist(), [[0, 0, 0]])

    def test_empty_cloud(self):
        cloud = voxelize(torch.empty(0, 3))
        self.assertEqual(cloud.coordinates.shape, (0, 3))
        self.assertEqual(cloud.offsets.shape, (0, 3))

    def test_previous_to_current_transform_direction(self):
        points = torch.tensor([[1., 0., 0.]])
        transform = torch.tensor([[0., -1., 0., 2.], [1., 0., 0., 3.], [0., 0., 1., 0.], [0., 0., 0., 1.]])
        self.assertTrue(torch.equal(transform_points(points, transform), torch.tensor([[2., 4., 0.]])))

    def test_centroid_gradient(self):
        points = torch.tensor([[0.011, 0.012, 0.013], [0.021, 0.022, 0.023]], requires_grad=True)
        grid = VoxelGrid(origin=(0, 0, 0))
        voxelize(points, grid).points(grid).sum().backward()
        self.assertTrue(torch.allclose(points.grad, torch.full_like(points, 0.5)))

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            voxelize(torch.tensor([[float("nan"), 0., 0.]]))
        with self.assertRaises(ValueError):
            transform_points(torch.zeros(1, 3), torch.diag(torch.tensor([2., 1., 1., 1.])))


if __name__ == "__main__":
    unittest.main()
