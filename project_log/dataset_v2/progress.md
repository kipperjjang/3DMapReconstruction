# Progress

- Inspected current GT and sensor-coordinate DynaMapping APIs.
- Baseline: 630 frames, 701 MiB including full archives.
- User added a 5 GB data cap; interpreted conservatively as decimal GB.
- GPU check: nvidia-smi fails, /dev/nvidia* absent, torch.cuda.is_available=False.
- Original preparation/launcher source copied here before changes.

## Implementation and checks

- Added `scripts/prepare_dataset_v2.py`: current Torch APIs, full sequences,
  actual DynaMapping robot-frame alignment, per-frame hashes, atomic NPZ writes,
  explicit source provenance, 5,000,000,000 byte cap with 256 MiB reserve.
- Current ray-pattern fingerprints differ from historical auto-resolution audits.
  Collection explicitly uses numeric 0.05 m boundary / 0.03 m sampling and records
  that the old resolution audit does not validate the current source fingerprint.
- DynaRaySim HEAD: ac2419f543394a5ce886d7bcb5541b3ea1dd854d.
  DynaMapping HEAD: 3cf0a5ab724a299bfd8dcde7d38704b0531835f4.
  Actual Python source SHA256 values are in dataset provenance (includes worktree).
- Fixed `train_gpu.sh` to validate the actual `--dataset` argument. Added
  `train_v2.sh` for 1,500 updates with the existing full architecture.
- Network regression suite: 32 passed, 2 preparation-only tests skipped in
  Python 3.7. Actual DynaMapping preparation tests: 2 passed under simulator
  Python (nonidentity extrinsic + rotating robot; storage-cap rejection).
- `bash scripts/train_v2.sh` attempted: preflight failed because the execution
  environment exposes no NVIDIA devices. Console report records host RTX 4090
  kernel metadata but CUDA is unavailable. No GPU optimizer updates occurred.
- Started a separate CPU diagnostic with the real fresh dataset, full network,
  batch 2 / rollout 12, one optimizer update. This is a readiness check only.
- Collecting office/lounge independently alongside the coordinator; per-scene
  file locks prevent duplicate collection when coordinator reaches those scenes.
- Fresh gymnasium sequence completed: 200/200 frames with nonempty observations.
- Real-data CPU check passed: full `[32,64,128,256,256]` network, batch 2,
  rollout 12, finite loss/gradients and an actual parameter-changing Adam update.
  Mean loss 1.2473247, elapsed 69.87 s. No diagnostic weights are advertised as a
  trained model. A simple 1,500-update extrapolation is ~29 hours before validation
  and does not justify silently replacing the planned GPU run with CPU training.
- Outdoor scenarios and trapezoid collection also run independently under scene
  locks; the coordinator assembles and validates the final manifest.

## Dataset completed and independent audit

- Coordinator completed all 1,420 frames: train 1,040 / validation 180 / test 200.
  All-frame validation passed: hashes, finite/in-grid points, unique GT voxels,
  timestamps and pose-derived relative transforms. Maximum transform error = 0.
- Data storage before the additional sample audit: 804,184,064 bytes (~0.804 GB).
- Valid observed rollout starts increased from 332 to 769 (rollout 12). All 216
  blind training frames remain for recurrent feedback; no rollout starts blind.
- The first independent audit used NumPy's general matrix inverse. On an exactly
  grid-aligned wall in partition frame 0, a matrix difference of 8.88e-16 and
  point difference of 1.78e-15 changed half-open voxel assignment (2,337 cells per
  direction), producing misleading row-wise centroid differences up to 3.08 m.
  Mutual nearest centroid distance was only 1.27 mm, and stored targets exactly
  matched a reread using the current DynaMapping transform.
- Corrected the independent NumPy audit to use the rigid inverse R.T, -R.T@t
  required by the pose contract. No dataset points or comparison tolerance were
  changed. That frame now reconstructs exactly. The initial failed audit log is
  preserved as `sample_audit_general_inverse.log`; checking all 21 sample frames.

## Final state

- Independent reconstruction audit: all 21 frames passed. The 19 samples with
  observations have 100% of points within 5 cm of fresh GT; maximum 0.0271921 m.
- Final data including audit reports: 804,196,352 bytes, below 5,000,000,000.
- External simulator/mapping Python source hashes remained unchanged throughout
  collection. Pipeline source snapshots and hashes are retained in this log.
- Dataset is ready; full GPU training remains blocked by device exposure, with
  zero GPU optimizer updates and no new trained checkpoint. The separate CPU
  diagnostic completed one update successfully. No background CPU training job
  was substituted for the planned GPU experiment.
- Reproduce/continue with `bash scripts/train_v2.sh` in a GPU-enabled environment.
  See `DATASET_V2_RESULTS.md` and this folder's `summary.json` for the final state.
