# Progress

- Found 27 source meshes; native dimensions span 0.05–1 m diameter, with capsules up to 3 m high.
- Existing v4 data collection is complete. This dataset uses a separate output directory and validation split.
- Preparing a 54-instance scene and deterministic aisle trajectory; existing user changes are preserved.
- Scene generation passed all 54 swept AABB and acquired-route approach checks. The floor is 30 × 24 m, objects form six rows of nine, and rounded aisle circuits shift 3.5 cm between passes.
- Current simulator rejects the legacy v4 `ego.terrain_alignment` key; omitted it and retained explicit zero roll/pitch poses. A Path-valued provenance setting was normalized before acquisition.
- Source scripts and original assets have been copied for reproducibility. Collection and automatic validation/audit are running in a managed command session; log: `collection.log`.
- GT generation completed in 78.4 seconds, retaining 6,561,866 surface samples. All 54 object instances have above-floor GT samples (minimum 96), including every 5 cm solid.
- Completed 10,000 unique frames, all assigned to validation, in 793 seconds including core validation and independent audits. Dataset storage is approximately 120 MB.
- Every packed record passed integrity, timestamp, pose uniqueness, upright sensor and feedback transform validation. No motion rejections or partial shards remain.
- Independent fresh raycasts and full retained-GT reconstruction passed for frames 0, 5000 and 9999; maximum mutual reconstruction distance <0.170 mm.
- The per-instance representation audit passed: all 54 objects have both above-floor GT samples and actual acquired returns in stored local crops. See `outputs/validation_basic_solids/asset_coverage_report.json`.
- Final source provenance checks passed. Collection status is `ready_for_validation`. Full results are recorded in `RESULTS.md`; no training or model evaluation was started.
