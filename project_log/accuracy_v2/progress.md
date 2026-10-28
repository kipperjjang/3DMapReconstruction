# Accuracy study progress

- Preserved original training_run_v1 and saved baseline source snapshot + SHA256 hashes for its 20 artifacts.
- Training-only noise diagnostic (every fifth frame, 90 frames): augmented observation voxel precision
  0m=0.90673, 0.005m=0.74523, 0.05m=0.50980. This measures augmentation support corruption, not model accuracy.
- Added configurable initial/final LR, optional warm-start checkpoint, validation-only candidate runs,
  frozen checkpoint evaluation, and per-run source snapshot/configuration/logs.
- Added append-only experiment ledger and regenerated comparison CSV/JSON/Markdown with report hashes.
- Full 31 tests passed; additional evaluation-only fixture path tested separately after adding that mode.
- First candidate: same 500 updates/seed/alpha=.5 as v1, position noise reduced to .005m.
  Validation F1 at update 100 = .2058, above v1 best .1691; run continues.
- Candidate selection uses validation only. Existing test scene will be evaluated only for the selected candidate.
- noise005 completed: best validation F1 .2058178 at update 100 (original .1691477).
- noise005_lr001 running: lower initial LR .001, same seed/data/noise/500 updates.
- Added one train-only BN recalibration candidate (32 rollouts, clean observations, no learned weight updates).
  The parent will be the better of the two trained candidates on validation.
- Full 32 tests passed, including no test access during candidate training, immutable parameters and
  train-only frame access during recalibration, singleton BN with cumulative statistics and checkpoint evaluation.
- Completed candidates: noise005 val .2058178, noise005_lr001 val .1757968,
  clean train BN calibration of noise005 val .1936563. Calibration did not improve selection.
- Locked noise005 / update 100 / checkpoint SHA256 in selection.json before final test evaluation.
- Selected test F1 .3131510 vs v1 .2535819 (+23.49% relative), precision .3455934 vs .2074922,
  recall .2862769 vs .3259943. FP 297471 vs 683263 (-56.46%).
- All 20 original v1 artifact hashes remain unchanged. Immutable ledger contains all four completed experiments.
- Final evaluation checkpoint hash matches validation selection. Test was not used to choose among candidates.
- Detailed results/tradeoffs/reproducibility in ACCURACY_RESULTS.md, comparison CSV/JSON/Markdown retained.
- Final artifact checks: both training histories have exactly updates 1..500 with finite loss;
  all source snapshots match recorded hashes; all four ledger entries are unique.
- Actual calibrated checkpoint differs from its parent only in BN running-statistics buffers.
- Selected checkpoint CPU inference exports first two test frames with 1186/1302 points;
  second frame consumes the aligned first prediction. Confidence/finite-coordinate checks pass.
