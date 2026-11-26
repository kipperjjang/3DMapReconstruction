import argparse
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from neural_scene.evaluation import comparison_snapshot, export_evaluation, load_export
from neural_scene.experiment import SequenceDataset, evaluate
from neural_scene.model import ReconstructionUNet4D, UNetConfig
from neural_scene.training import save_checkpoint, load_checkpoint
from test_experiment import fixture_dataset


class EvaluationTests(unittest.TestCase):
    def test_export_matches_training_evaluator_and_preserves_clouds(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture_dataset(root)
            checkpoint = root / 'trained.pt'
            save_checkpoint(checkpoint, ReconstructionUNet4D(UNetConfig(channels=(2, 4, 4, 4, 4), pruning_threshold=0)))
            args = argparse.Namespace(checkpoint=checkpoint, dataset=root, output=root / 'export',
                split='validation', scene=None, device='cpu', limit=2, no_feedback=False)
            with redirect_stdout(io.StringIO()):
                exported = export_evaluation(args)
            expected = evaluate(load_checkpoint(checkpoint), SequenceDataset(root), 'validation', 'cpu', limit=2)
            self.assertEqual(exported, expected)
            manifest, sequences = load_export(args.output)
            self.assertTrue(manifest['feedback'])
            self.assertEqual(len(sequences), 1)
            self.assertEqual(len(sequences[0]['frames']), 2)
            timing = json.loads((args.output / 'timing.json').read_text())
            self.assertEqual(timing['frames'], 2)
            self.assertGreater(timing['total_inference_s'], 0)
            self.assertAlmostEqual(timing['mean_inference_ms'], timing['total_inference_s'] * 1000 / 2)
            self.assertAlmostEqual(sum(f['inference_ms'] for f in sequences[0]['frames']),
                                   timing['total_inference_s'] * 1000)
            self.assertGreaterEqual(timing['evaluation_total_s'], timing['total_inference_s'])
            self.assertEqual(manifest['timing'], timing)
            self.assertEqual(json.loads((args.output / 'report.json').read_text())['timing'], timing)
            for metadata in sequences[0]['frames']:
                with np.load(args.output / metadata['file'], allow_pickle=False) as data:
                    np.testing.assert_array_equal(data['current_points'], [[np.float32(.021), np.float32(.022), np.float32(.023)]])
                    self.assertEqual(len(data['confidence']), len(data['prediction_points']))
                    snapshot = comparison_snapshot(data, 'prediction', 'target', manifest['grid'])
                    np.testing.assert_array_equal(snapshot.current.points, data['prediction_points'])
                    np.testing.assert_array_equal(snapshot.history[0].aligned_point_cloud, data['target_points'])
                    self.assertEqual(snapshot.current.timestamp_ns, metadata['timestamp_ns'])
            with self.assertRaises(FileExistsError):
                export_evaluation(args)

    def test_comparison_voxels_crop_deduplicate_and_keep_pose(self):
        pose = np.eye(4)
        pose[0, 3] = 5
        data = dict(prediction_points=np.array([[.01, .02, .03], [.02, .02, .03], [1., 0, 0]]),
                    target_points=np.empty((0, 3)), robot_pose=pose, timestamp_ns=7, frame_id=2)
        snapshot = comparison_snapshot(data, 'prediction', 'target',
            dict(origin=[0, 0, 0], voxel_size=.05, shape=[16, 16, 16]), True)
        np.testing.assert_allclose(snapshot.current.voxel_centers, [[.025, .025, .025]])
        np.testing.assert_array_equal(snapshot.current.voxel_grid_pose_world, pose)
        self.assertEqual(snapshot.history[0].aligned_voxels.shape, (0, 3))

    def test_partial_exports_and_escaping_paths_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / 'evaluation.json'
            value = dict(format='neural-scene-evaluation-v1', status='running')
            path.write_text(json.dumps(value))
            with self.assertRaises(ValueError):
                load_export(root)
            value.update(status='completed', sequences=[dict(scene='bad', frames=[dict(file='../outside.npz')])])
            path.write_text(json.dumps(value))
            with self.assertRaises(ValueError):
                load_export(root)


if __name__ == '__main__':
    unittest.main()
