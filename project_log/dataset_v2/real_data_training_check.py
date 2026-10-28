"""Exercise one full batch-2/12-step update on fresh data; not a trained model."""

import json
from pathlib import Path
import time

import numpy as np
import torch

from neural_scene.experiment import augmented_frame, predict
from neural_scene.model import ReconstructionUNet4D
from neural_scene.loss import reconstruction_loss
from neural_scene.training import adam_with_decay
from neural_scene.voxel import VoxelGrid

root = Path(__file__).resolve().parents[2]
folder = root / 'outputs/training_dataset_v2/sequences/01_gymnasium_pedestrians'
out = Path(__file__).resolve().parent
torch.manual_seed(20261020)
rng = np.random.default_rng(20261020)
model = ReconstructionUNet4D().train()
optimizer, scheduler = adam_with_decay(model, 1500, 0.01, 0.0001)
grid = VoxelGrid()
before = next(model.parameters()).detach().clone()
previous = [None, None]
begun = time.monotonic()
steps = []
optimizer.zero_grad()
for age in range(12):
    frames = []
    for start in (0, 12):
        with np.load(folder / 'frame_{:05d}.npz'.format(start + age), allow_pickle=False) as saved:
            frame = {key: saved[key].copy() for key in ('current_points', 'target_points', 'robot_pose', 'transformation', 'timestamp_ns')}
        frames.append(augmented_frame(frame, [1, 1], rng, 0.005))
    prediction = predict(model, frames, previous, grid, 'cpu')
    loss = reconstruction_loss(prediction, [frame['target_points'] for frame in frames])
    assert torch.isfinite(loss.total)
    (loss.total / 12).backward()
    previous = [points.detach() for points in prediction.points()]
    steps.append({'age': age, 'loss': float(loss.total.detach()), 'output_voxels': len(prediction.offsets)})
    print(json.dumps(steps[-1]), flush=True)
gradients = [p.grad for p in model.parameters() if p.grad is not None]
assert gradients and all(torch.isfinite(g).all() for g in gradients)
optimizer.step()
scheduler.step()
assert not torch.equal(before, next(model.parameters()).detach())
report = {'status': 'passed', 'purpose': 'single optimizer update execution check; not a completed training run',
          'device': 'cpu', 'batch_size': 2, 'rollout': 12, 'optimizer_updates': 1,
          'channels': list(model.config.channels), 'elapsed_s': time.monotonic() - begun,
          'mean_loss': float(np.mean([item['loss'] for item in steps])), 'steps': steps}
(out / 'real_data_training_check.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report), flush=True)
