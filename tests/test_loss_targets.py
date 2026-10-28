import unittest

import torch

from neural_scene.loss import match_keys, occupancy_labels
from neural_scene.voxel import VoxelGrid


class TargetTests(unittest.TestCase):
    def test_coarse_targets_keep_batch_identity_and_cover_both_times(self):
        candidates = torch.tensor([[0, 0, 0, 0, 0], [0, 0, 0, 0, 1], [1, 0, 0, 0, 0], [0, 8, 0, 0, 0]])
        targets = torch.tensor([[0, 3, 1, 2]])
        self.assertEqual(occupancy_labels(candidates, targets, [4, 4, 4, 1], VoxelGrid()).tolist(), [True, True, False, False])

    def test_matching_uses_coordinates_instead_of_row_order(self):
        matched, rows = match_keys(torch.tensor([3, 7, 1]), torch.tensor([1, 3]))
        self.assertEqual(matched.tolist(), [True, False, True])
        self.assertEqual(rows[matched].tolist(), [1, 0])


if __name__ == "__main__":
    unittest.main()
