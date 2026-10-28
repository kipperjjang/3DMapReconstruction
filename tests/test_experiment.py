from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from neural_scene import experiment
from neural_scene.model import ReconstructionUNet4D, UNetConfig
from neural_scene.geometry import previous_to_current
from neural_scene.training import tensorboard_writer
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def fixture_dataset(directory):
    sequences = []
    for split in ("train", "validation", "test"):
        scene = split + "_fixture"
        folder = directory / "sequences" / scene
        folder.mkdir(parents=True)
        frames = []
        for index in range(12):
            name = "frame_{:05d}.npz".format(index)
            points = np.array([[0.021, 0.022, 0.023]], dtype=np.float32)
            np.savez(folder / name, current_points=points, target_points=points, robot_pose=np.eye(4), transformation=np.eye(4), timestamp_ns=np.int64(index*100000000))
            frames.append({"file": name})
        sequences.append({"scene": scene, "split": split, "frames": frames})
    (directory / "dataset.json").write_text(json.dumps({"format": "neural-scene-dynamapping-v1", "input_policy": "aligned_merge_t0", "grid": {"voxel_size": 0.05, "shape": [16, 16, 16], "origin": [0, 0, 0]}, "sequences": sequences}))


class ExperimentTests(unittest.TestCase):
    def test_bn_calibration_uses_only_train_and_does_not_change_parameters(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            fixture_dataset(directory)
            dataset = experiment.SequenceDataset(directory)
            frame = dataset.frame
            def training_frame_only(sequence, index):
                self.assertEqual(sequence['split'], 'train')
                return frame(sequence, index)
            dataset.frame = training_frame_only
            model = ReconstructionUNet4D(UNetConfig(channels=(2, 4, 4, 4, 4), pruning_threshold=0))
            parameters = {name: value.detach().clone() for name, value in model.named_parameters()}
            result = experiment.calibrate_batch_norm(model, dataset, 'cpu', 1, 1, 2, 1)
            self.assertFalse(result['parameters_updated'])
            self.assertEqual(result['scene_episode_counts'], {'train_fixture': 1})
            self.assertFalse(model.training)
            for name, value in model.named_parameters():
                torch.testing.assert_close(value, parameters[name], rtol=0, atol=0)
                self.assertIsNone(value.grad)
            self.assertTrue(any(int(module.num_batches_tracked) > 0 for module in model.modules() if isinstance(module, torch.nn.BatchNorm1d)))

    def test_candidate_training_defers_test_and_records_rates_and_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            fixture_dataset(directory)
            output = directory / 'candidate'
            arguments = ['experiment', '--dataset', str(directory), '--output', str(output),
                '--device', 'cpu', '--updates', '1', '--rollout', '2', '--skip-test',
                '--initial-lr', '0.001', '--final-lr', '0.00001', '--no-mirror']
            actual_evaluate = experiment.evaluate
            splits = []
            def checked_evaluate(model, dataset, split, device, **kwargs):
                splits.append(split)
                if split == 'test':
                    raise AssertionError('Candidate selection must not evaluate test')
                return actual_evaluate(model, dataset, split, device, **kwargs)
            def small_model():
                return ReconstructionUNet4D(UNetConfig(channels=(2, 4, 4, 4, 4), pruning_threshold=0))
            with patch.object(sys, 'argv', arguments), patch.object(experiment, 'ReconstructionUNet4D', small_model), patch.object(experiment, 'evaluate', checked_evaluate), redirect_stdout(io.StringIO()):
                experiment.main()
            report = json.loads((output / 'report.json').read_text())
            self.assertIsNone(report['held_out_test'])
            self.assertEqual(report['configuration']['initial_lr'], 0.001)
            self.assertEqual(set(splits), {'validation'})
            self.assertTrue((output / 'source' / 'neural_scene' / 'experiment.py').is_file())
            state = torch.load(output / 'latest.pt', map_location='cpu')
            self.assertAlmostEqual(state['optimizer']['param_groups'][0]['lr'], 0.00001)
            evaluated = directory / 'evaluation'
            eval_arguments = [value for value in arguments if value != '--skip-test'] + [
                '--evaluate-only', '--initial-checkpoint', str(output / 'best.pt'), '--output', str(evaluated)]
            with patch.object(sys, 'argv', eval_arguments), patch.object(experiment, 'ReconstructionUNet4D', small_model), redirect_stdout(io.StringIO()):
                experiment.main()
            evaluation = json.loads((evaluated / 'report.json').read_text())
            self.assertEqual(evaluation['mode'], 'evaluation')
            self.assertEqual(evaluation['held_out_test']['model']['frames'], 12)
            self.assertFalse((evaluated / 'training.jsonl').exists())
            self.assertIsNotNone(evaluation['configuration']['initial_checkpoint_sha256'])
            events = EventAccumulator(str(evaluated)).Reload()
            self.assertEqual(events.Scalars('test/model/f1')[0].step, 0)

    def test_full_training_validation_and_test_path_on_fixture(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            fixture_dataset(directory)
            output = directory / "run"
            arguments = ["experiment", "--dataset", str(directory), "--output", str(output), "--device", "cpu", "--updates", "1", "--rollout", "2", "--eval-every", "1", "--position-noise", "0", "--no-mirror"]
            def small_model():
                return ReconstructionUNet4D(UNetConfig(channels=(2, 4, 4, 4, 4), pruning_threshold=0))
            with patch.object(sys, "argv", arguments), patch.object(experiment, "ReconstructionUNet4D", small_model), redirect_stdout(io.StringIO()):
                experiment.main()
            report = json.loads((output / "report.json").read_text())
            self.assertEqual(report["status"], "completed")
            self.assertEqual(report["configuration"]["input_policy"], "aligned_merge_t0")
            self.assertEqual(report["held_out_test"]["model"]["frames"], 12)
            self.assertGreater(report["held_out_test"]["model"]["recall"], 0)
            for name in ("best.pt", "last.pt", "latest.pt", "training.jsonl"):
                self.assertTrue((output / name).is_file())
            state = torch.load(output / "latest.pt", map_location="cpu")
            self.assertEqual(state["step"], 1)
            self.assertAlmostEqual(state["optimizer"]["param_groups"][0]["lr"], 0.0001)
            self.assertTrue(list(output.glob("events.out.tfevents.*")))
            self.assertFalse((output / "tensorboard").exists())
            events = EventAccumulator(str(output)).Reload()
            self.assertEqual([event.step for event in events.Scalars('validation/model/f1')], [0, 1])
            record = json.loads((output / 'training.jsonl').read_text())
            for key in ('loss', 'occupancy', 'offset', 'lr', 'output_voxels', 'elapsed_s'):
                scalar = events.Scalars('train/' + key)[0]
                self.assertEqual(scalar.step, 1)
                self.assertAlmostEqual(scalar.value, record[key], delta=max(abs(record[key]) * 1e-6, 1e-8))
            self.assertEqual(events.Scalars('test/model/f1')[0].step, report['best_validation']['step'])
            # Simulate an interrupted final evaluation plus an uncommitted log
            # record: resume must discard records beyond the saved update.
            (output / "report.json").unlink()
            with (output / "training.jsonl").open("a") as stream:
                stream.write(json.dumps({"step": 2, "loss": 999}) + "\n")
            with tensorboard_writer(output) as writer:
                writer.add_scalar('train/loss', 999, 2)
            with patch.object(sys, "argv", arguments + ["--resume"]), patch.object(experiment, "ReconstructionUNet4D", small_model), redirect_stdout(io.StringIO()):
                experiment.main()
            committed = [json.loads(line)["step"] for line in (output / "training.jsonl").read_text().splitlines()]
            self.assertEqual(committed, [1])
            events = EventAccumulator(str(output)).Reload()
            self.assertEqual([event.step for event in events.Scalars('train/loss')], [1])
            self.assertEqual(json.loads((output / "report.json").read_text())["status"], "completed")

    def test_duplicate_scene_across_splits_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            fixture_dataset(directory)
            path = directory / "dataset.json"
            manifest = json.loads(path.read_text())
            manifest["sequences"][1]["scene"] = manifest["sequences"][0]["scene"]
            path.write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                experiment.SequenceDataset(directory)

    def test_mirror_conjugates_pose_and_preserves_feedback_alignment(self):
        previous_pose = np.eye(4)
        current_pose = np.eye(4)
        current_pose[:3, 3] = [0.3, 0.2, 0.1]
        frame = {"current_points": np.array([[0.2, 0.4, 0.5]], dtype=np.float32), "target_points": np.array([[0.1, 0.3, 0.6]], dtype=np.float32), "transformation": previous_to_current(current_pose, previous_pose), "robot_pose": current_pose}
        augmented = experiment.augmented_frame(frame, [-1, 1], np.random.default_rng(3), 0)
        mirror = np.diag([-1, 1, 1, 1])
        np.testing.assert_allclose(augmented["transformation"], previous_to_current(augmented["robot_pose"], mirror @ previous_pose @ mirror))
        point = np.array([0.2, 0.4, 0.5, 1.])
        np.testing.assert_allclose(augmented["transformation"] @ mirror @ point, mirror @ frame["transformation"] @ point)

    def test_rollout_starts_require_evidence_but_preserve_blind_followup(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            fixture_dataset(directory)
            dataset = experiment.SequenceDataset(directory)
            sequence = dataset.splits["train"][0]
            for index, frame in enumerate(sequence["frames"]):
                frame["current_points"] = 1 if index == 0 else 0
            self.assertEqual(dataset.observable_starts(sequence, 12), [0])
            self.assertEqual(dataset.observable_starts(sequence, 2), [0])


if __name__ == "__main__":
    unittest.main()
