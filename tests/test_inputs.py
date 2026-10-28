import unittest

import torch

from neural_scene.inputs import merged_input
from neural_scene.voxel import VoxelGrid


class InputTests(unittest.TestCase):
    def test_sources_are_merged_to_one_centroid_at_t_zero(self):
        grid = VoxelGrid(origin=(0, 0, 0))
        packed = merged_input([torch.tensor([[0.01, 0.02, 0.03]])], [torch.tensor([[0.04, 0.03, 0.02]])], grid)
        self.assertEqual(packed.coordinates.tolist(), [[0, 0, 0, 0, 0]])
        self.assertTrue(torch.allclose(packed.features, torch.tensor([[0.5, 0.5, 0.5]])))

    def test_empty_member_keeps_batch_identity(self):
        packed = merged_input([torch.empty(0, 3), torch.zeros(1, 3)])
        self.assertEqual(packed.batch_size, 2)
        self.assertEqual(packed.coordinates[:, 0].tolist(), [1])


if __name__ == "__main__":
    unittest.main()
