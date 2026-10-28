# Accuracy improvement study

User requests autonomous accuracy improvements and persistent comparisons with all previous experiments.

- Freeze v1 artifacts, keep source snapshot and artifact hashes in outputs/accuracy_study_v2.
- Preserve 64³ grid, t=0 merge, channels/two convs, predictive pruning alpha=0.5, no GT forcing.
- Keep the original scene split and exact evaluation metrics for comparison.
- Diagnose observation noise relative to the 5cm grid using training frames only.
- Compare controlled training runs with smaller observation noise and lower initial LR.
- Compare one additional candidate with learned weights fixed and BN running statistics re-estimated
  on 32 clean training rollouts. Parent is selected by validation F1 among the two trained candidates.
  No validation/test inputs or labels are used to estimate these statistics.
- Select settings and checkpoint using validation only. Candidate trials defer test evaluation.
- Record source, configuration, training/validation history, checkpoint and lineage for every run.
- Evaluate the selected final candidate on the already-used test scene once. This is a repeated
  benchmark, not a newly unseen dataset; report that limitation and compare v1 and measurements.
- Persist comparison JSON/CSV/Markdown including unsuccessful trials and exact artifact locations.
