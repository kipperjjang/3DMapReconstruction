import unittest

from neural_scene.training import epoch_updates


class TrainingDurationTests(unittest.TestCase):
    def test_partition_frame_budget_rounds_up(self):
        self.assertEqual(epoch_updates(160000, 2, 12), 6667)
        self.assertEqual(epoch_updates(320000, 2, 12), 13334)

    def test_exact_and_small_dataset(self):
        self.assertEqual(epoch_updates(48, 2, 12), 2)
        self.assertEqual(epoch_updates(1, 2, 12), 1)

    def test_invalid_counts(self):
        for counts in ((0, 2, 12), (1, 0, 12), (1, 2, 0)):
            with self.assertRaises(ValueError):
                epoch_updates(*counts)
