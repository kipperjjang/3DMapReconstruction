# Empty CUDA output correction

The user reported `CUDA error: invalid configuration argument` during initial
validation, surfaced by `torch.isfinite` in target voxelization. They authorized
fixing the error and launching the full training job.

## Evidence and change

- DynaRaySim commit `26ed03c` changes live simulator occupancy ordering. Training
  reads packed HDF5 point arrays and uses the local `neural_scene.voxel` module;
  it does not call DynaRaySim's voxelization code. Loss matching uses coordinate
  keys and does not assume a shared ordering.
- With seed 20261028 the full model's first three validation frames in each of
  the four validation scenes all produce empty predictions before training.
- `_empty_result` previously constructed an empty CUDA `ME.SparseTensor`.
  ME 0.5.4's `src/coordinate_map_manager.cu` lines 89–96 launches `cuda_copy_n`
  with `(mapping.size() + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS` blocks,
  without checking whether `mapping.size()` is zero. Zero rows mean zero blocks.
  CUDA can report that launch error on a subsequent PyTorch operation.
- Empty predictions now use `EmptyOffsets`, storing the same empty feature and
  coordinate tensors without registering an ME coordinate map. The empty
  features remain connected to the model for backward. Nonempty outputs still
  use ME. Pruning thresholds, GT supervision, and point/coordinate order are
  unchanged.
- The local finite-point validation already present in `voxel.py` was preserved.

## Verification and training attempt

- Model, voxel and experiment regression checks: 17 passed.
  Log: `empty_output_regression.log`.
- Explicit empty-input and forced-all-pruned forward/loss/metrics/backward
  regression: passed on CPU. The same test includes CUDA when available.
  Log: `empty_output_regression_targeted.log`.
- Full-channel model, actual v3 validation data, 12 frames including detached
  feedback: passed on CPU. All 12 empty predictions were scored successfully.
  Log: `empty_output_real_data_check.log`.
- Ran `bash scripts/train_v3.sh`. It exited at GPU preflight: `/dev/nvidia*` is
  absent, `torch.cuda.is_available()` is false, and nvidia-smi returns 9 in the
  agent's execution environment. No full GPU updates were started. The specific
  CUDA correction must still be confirmed on the user's GPU host.
  Log: `training_launch_after_empty_fix.log`.

The existing run has no `latest.pt`; restart it with `bash scripts/train_v3.sh`
from this workspace on the GPU host. `--resume` requires an existing checkpoint.
Data collection and final dataset checks are complete: 200,000 frames,
approximately 2.89 GB, and all 120 independent reconstruction samples passed.
