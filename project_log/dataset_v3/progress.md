# Progress

- Read current simulator, mapper and paper from local sources.
- Confirmed existing training_run_v2 is now completed (1,500 GPU updates); preserve.
- User selected 200,000 unique frames with an approximately 10 GB cap.
- dyna CUDA preflight fails: no /dev/nvidia nodes. SciPy import, spatial tree,
  ndimage and sparse graph imports work through the workspace compatibility link.
- Initial pip install attempt failed due DNS/network restrictions; no environment
  packages were modified.

## Pilot, codec and environment

- dyna CPU MinkowskiEngine 0.5.4 built against torch 2.6; full-network smoke passed.
- SciPy 1.15.3 is linked from a compatible existing Python 3.10 installation.
  TensorBoard 2.11.2 is copied locally with NumPy 2 string alias replacements;
  historical dependency directories remain unchanged. Full suite: 41 passed.
- Compact codec: 6 uint8 values per point (cell xyz + 256-bin offset xyz), gzip.
  Raw observation multiplicity is preserved; maximum 3D quantization error ~0.17mm.
  Added bounded 512-frame LRU cache and packed HDF5 support to SequenceDataset.
- First 16-channel pilot acquired all 100 frames, but the wrapper mistakenly
  treated the simulator's [0] rejection-count list as a rejection. Fixed with
  any(counts); retained the failed pilot and log. It does not count toward 200k.
- Three separate 32-channel pilot scenes: 300 unique poses, 0 sensor-up error,
  all-frame validation passed, mean dense GT-to-measurement <=5cm coverage 0.45417,
  occupied GT voxel recall 0.20496. These are distinct metrics and neither is
  claimed as an exact reproduction of the paper's undefined visibility metric.
- Pilot shards average ~14 KB/frame; total 16.7 MB includes shared dense surfaces.
  40 unique scene geometries * 5,000 freshly acquired frames = 200,000 planned;
  train 160,000 / validation 20,000 / test 20,000. Production seeds do not overlap
  the pilots. 12 CPU workers, 2 threads each, on a 32-logical-core machine.
- Profiling: ~50s shared GT setup, then ~0.15s/frame in a single-worker pilot.
  200,000-frame production collection launched with a 10,000,000,000-byte cap and
  512 MiB global reserve. No repeated existing frames count toward the target.
- CUDA backend build started using the existing CUDA 11.7 toolkit, torch cu118,
  8.6+PTX for the host RTX4090; actual CUDA execution remains unavailable here.
- A one-update full-network pilot train/evaluate run is executing under dyna CPU.

## Training and independent reconstruction checks

- Packed-shard training reader and bounded-cache regression passed (1 additional
  test; 42 total tests now verified across the full suite plus this new test).
- Pilot CPU training completed in dyna: one actual full-model Adam update with
  batch 2 / rollout 12, followed by validation and held-out test. The checkpoint
  is only a pipeline check, not a model trained on the 200,000-frame dataset.
- Pilot independent audit re-ray-cast nine stored poses from the authored meshes
  and reconstructed GT using full dense-cloud filtering instead of the ROI tree.
  Point counts match; mutual errors stay below 0.167mm, within quantization bound.
- dyna CUDA backend built successfully with 8.6+PTX. Import reports
  ME.is_cuda_available=True but torch.cuda.is_available=False in this execution
  environment. Toolkit 11.7 / torch CUDA 11.8 minor version warning is recorded.
- `bash scripts/train_v3.sh` was attempted and stopped at GPU preflight. This is
  the actual 20,000-update launcher (batch 2, rollout 12, eval every 1,000,
  position noise 0.005 m, LR 0.01->0.0001, seed 20261028). 480,000 sampled train
  frame presentations correspond to about three passes over 160,000 train frames;
  they are not counted as newly collected timesteps.
- Current full collection continues; global data guard remains 10 GB with reserve.

## User training handoff and first production validation

- User requested the exact launch command and will start full training on the
  host. Supplied `conda activate dyna; bash scripts/train_v3.sh` from this project,
  and `--resume`. Continue collection and final checks before signalling readiness;
  do not start the user's full training job automatically.
- First 12 closed production scenes (60,000 frames) passed complete shard checks.
  Independent reraycast/dense-GT audits passed for 36 production frames.
- Commands and all data/coverage/quantization conventions are documented in
  `../../docs/DATASET_GUIDE`. Partial reports are explicitly distinguished from final
  200,000-frame readiness reports.
