import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

from neural_scene.inputs import MERGED_POLICY, TEMPORAL_POLICY, merged_input
from neural_scene.model import ReconstructionUNet4D, UNetConfig
from neural_scene.training import adam_with_decay, load_checkpoint, save_checkpoint


class TrainingIOTests(unittest.TestCase):
    def test_old_checkpoint_without_policy_in_config_keeps_legacy_behavior(self):
        model = ReconstructionUNet4D(UNetConfig(channels=(2, 4, 4, 4, 4),
            pruning_threshold=0, input_policy=MERGED_POLICY)).eval()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'legacy.pt'
            save_checkpoint(path, model)
            state = torch.load(path, map_location='cpu')
            state['config'].pop('input_policy')
            torch.save(state, path)
            loaded = load_checkpoint(path).eval()
            self.assertEqual(loaded.config.input_policy, MERGED_POLICY)
            self.assertEqual(loaded.offset_head.kernel.shape, model.offset_head.kernel.shape)
            packed = merged_input([torch.zeros(1, 3)], [torch.ones(1, 3)*.02], input_policy=MERGED_POLICY)
            with torch.no_grad():
                original, restored = model(packed), loaded(packed)
            original_rows = {tuple(c): f.tolist() for c,f in zip(original.offsets.C.tolist(), original.offsets.F)}
            restored_rows = {tuple(c): f.tolist() for c,f in zip(restored.offsets.C.tolist(), restored.offsets.F)}
            self.assertEqual(original_rows, restored_rows)

    def test_learning_rate_reaches_paper_endpoint(self):
        model = torch.nn.Linear(1, 1)
        optimizer, scheduler = adam_with_decay(model, total_updates=5)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.01)
        for _ in range(5):
            optimizer.zero_grad()
            model(torch.ones(1, 1)).sum().backward()
            optimizer.step()
            scheduler.step()
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.0001)

    def test_checkpoint_preserves_architecture_and_prediction(self):
        torch.manual_seed(19)
        model = ReconstructionUNet4D(UNetConfig(channels=(2, 4, 4, 4, 4), pruning_threshold=0)).eval()
        packed = merged_input([torch.zeros(1, 3)])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            save_checkpoint(path, model)
            restored = load_checkpoint(path).eval()
            self.assertEqual(restored.config, model.config)
            with torch.no_grad():
                original = model(packed)
                reloaded = restored(packed)
            original_rows = {tuple(c): f.tolist() for c, f in zip(original.offsets.C.tolist(), original.offsets.F)}
            reloaded_rows = {tuple(c): f.tolist() for c, f in zip(reloaded.offsets.C.tolist(), reloaded.offsets.F)}
            self.assertEqual(original_rows, reloaded_rows)

    def test_npz_cli_aligns_previous_prediction_when_current_scan_is_empty(self):
        model = ReconstructionUNet4D(UNetConfig(channels=(2, 4, 4, 4, 4), pruning_threshold=0)).eval()
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            checkpoint = directory / "model.pt"
            save_checkpoint(checkpoint, model)
            transform = np.eye(4)
            transform[0, 3] = 1
            np.savez(directory / "current.npz", current_points=np.empty((0, 3), dtype=np.float32), transformation=transform)
            np.savez(directory / "previous.npz", points=np.zeros((1, 3), dtype=np.float32))
            output = directory / "result.npz"
            command = [sys.executable, "-m", "neural_scene", "infer", "--checkpoint", str(checkpoint), "--input", str(directory / "current.npz"), "--previous-prediction", str(directory / "previous.npz"), "--output", str(output)]
            completed = subprocess.run(command, cwd=str(Path(__file__).resolve().parents[1]), capture_output=True, text=True, check=True)
            self.assertEqual(json.loads(completed.stdout)["points"], 4096)
            with np.load(output, allow_pickle=False) as result:
                self.assertEqual(result["point_frame"].item(), "current_robot")
                self.assertEqual(result["input_policy"].item(), "aligned_temporal_t01")
                self.assertGreaterEqual(result["points"][:, 0].min(), 0.8)
                self.assertLess(result["points"][:, 0].max(), 1.6)
                self.assertEqual(result["confidence"].shape, (4096,))


if __name__ == "__main__":
    unittest.main()
