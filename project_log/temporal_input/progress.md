# Progress

- Reviewed `.gitignore`; added local environment/credential and profiling ignores.
- Committed all existing source, guides and project logs as baseline `7dee8a4`.
  Dataset/checkpoint outputs and dependencies/build products are excluded.
- Implemented `aligned_temporal_t01` as the default model/input policy. Current
  and previous clouds are voxelized independently; decoder support keeps t=0/1.
  The final offset convolution uses kernel [1,1,1,2], stride [1,1,1,2], producing
  unique spatial outputs at t=0. Final confidence takes the maximum probability
  across matching spatial voxels from the two times.
- Preserved the legacy input and head via checkpoint policy inference, including
  configurations saved before the new policy field existed. Prediction,
  stateful mapping, inference/export, warm start and resume use model policy.
- Existing raw v3 data can be reused; no data/checkpoint files were rewritten.
  New temporal runs require a new output directory. Existing runs resume with
  their saved policy; explicit policy mismatches are rejected.
- Updated README, PAPER_NOTES, v3 data guide and inference guide. Historical
  source snapshots and training results remain descriptions of their own runs.

## Validation

- `bash scripts/run_dyna.sh -m pytest -q tests`: **47 passed**, 18.31 seconds.
  Log: `tests_final.log`. Covers temporal overlap without centroid mixing,
  gradients from both sources through the learned temporal head, one final
  voxel per spatial position, pruning/empty output, legacy checkpoint loading,
  train/evaluate/resume, inference CLI and export playback data contracts.
- `bash scripts/run_dyna.sh -m neural_scene smoke --device cpu`: full channels,
  both t=0 and t=1 present, 47,537,479 parameters, 3,611 output points,
  loss 1.168555; forward/loss/backward passed. Log: `full_channel_smoke.log`.
- Loaded actual `outputs/training_run_v3_cp/best.pt` on CPU with the inferred
  `aligned_merge_t0` policy and evaluated one real frame from each validation
  scene (4 total). Passed; this is a compatibility check, not full validation.
  Log: `legacy_v3_checkpoint.log`.
- `git diff --check` passed. New temporal training has not been launched;
  GPU execution remains unverified in this agent environment.

## Run a new temporal model

```bash
conda activate dyna
bash scripts/train_v3.sh --output outputs/training_run_v3_temporal \
  --input-policy aligned_temporal_t01
```

The baseline is committed (`7dee8a4`); temporal implementation and validation
are subsequent working-tree changes for review.
