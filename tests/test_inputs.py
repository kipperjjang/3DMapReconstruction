import unittest

import torch

from neural_scene.inputs import MERGED_POLICY, TEMPORAL_POLICY, merged_input
from neural_scene.voxel import VoxelGrid


class InputTests(unittest.TestCase):
    def test_legacy_policy_still_merges_sources(self):
        packed = merged_input([torch.tensor([[.01, .02, .03]])],
            [torch.tensor([[.04, .03, .02]])], VoxelGrid(origin=(0, 0, 0)), input_policy=MERGED_POLICY)
        self.assertEqual(packed.coordinates.tolist(), [[0, 0, 0, 0, 0]])
        torch.testing.assert_close(packed.features, torch.tensor([[.5, .5, .5]]))
        self.assertEqual(packed.input_policy, MERGED_POLICY)

    def test_sources_keep_separate_centroids_at_each_time(self):
        grid = VoxelGrid(origin=(0, 0, 0))
        packed = merged_input([torch.tensor([[0.01, 0.02, 0.03]])], [torch.tensor([[0.04, 0.03, 0.02]])], grid)
        self.assertEqual(packed.coordinates.tolist(), [[0, 0, 0, 0, 0], [0, 0, 0, 0, 1]])
        self.assertTrue(torch.allclose(packed.features, torch.tensor([[0.2, 0.4, 0.6], [0.8, 0.6, 0.4]])))

    def test_empty_member_keeps_batch_identity(self):
        packed = merged_input([torch.empty(0, 3), torch.zeros(1, 3)])
        self.assertEqual(packed.batch_size, 2)
        self.assertEqual(packed.coordinates[:, 0].tolist(), [1])


if __name__ == "__main__":
    unittest.main()
