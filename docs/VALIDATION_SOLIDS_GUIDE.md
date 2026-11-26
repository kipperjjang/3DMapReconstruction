# Basic solid validation dataset

Dataset: `../outputs/validation_basic_solids/dataset.json`.
This independent scene is reserved entirely for validation and is not part of
the v4 training split.

Collection is complete: all 10,000 frames passed validation and independent
sample audits. All 54 placed instances have GT samples and acquired returns in
the stored local crops. The completed dataset occupies approximately 120 MB.

- 10,000 freshly raycast frames, 10 Hz, 1,000 seconds.
- Flat 30 × 24 m floor at Z=0; 54 stationary instances, two of each of the
  27 original `DynaRaySim/resources/assets/basic_solid/*.obj` meshes.
- Native metre dimensions are preserved, including 5 cm solids and 3 m capsules.
- Ego is scripted at Z=0.7 m and 0.65 m/s. Rounded aisle waypoints are sampled
  every 0.25 seconds; DynaRaySim interpolates position and quaternion rotation.
  Successive circuits shift slightly to avoid repeating acquired poses.
- Every asset instance is approached within 1.25 m during acquisition. All
  continuous waypoint segments clear inflated object bounds by the ego radius
  and clearance. These are geometric approach checks, not a claim that each
  tiny object receives a LiDAR return in every nearby frame.
- LiDAR matches v4: 32 channels, 3° azimuth spacing, 360° horizontal field,
  elevation −85° to +45°, 0.05–40 m range, upright mounting.
- The existing packed dataset pipeline stores current measurements, full local
  surface GT, poses, timestamps and DynaMapping feedback. The model's 3.2 m
  local crop and 5 cm voxels are unchanged.

Scene files under `sequences/scene_0000/` include `scene.obj`, `scenario.yaml`,
`terrain_metadata.json`, `terrain_preview.svg`, `surface_gt.npz`, `frames.h5`
and `sequence.json`. Placement metadata records per-asset counts, checksums,
native transformed bounds and closest acquired approach. Source asset copies
are retained under `source_assets/`.

Run `bash scripts/run_dyna.sh scripts/status_validation_solids.py` for progress.
Monitor `../project_log/validation_basic_solids/collection.log` and
`outputs/validation_basic_solids/status.json`. Completion is reported as
`ready_for_validation` only after every packed record passes validation and
first/middle/last frames pass independent raycast and GT reconstruction audits.
Final reports are `result.json`, `validation_report.json`, `sample_audit.json`.
`asset_coverage_report.json` additionally counts above-floor GT samples and
stored LiDAR returns within every instance's bounding box. Generate it after
collection with:

```bash
bash scripts/run_dyna.sh scripts/audit_validation_solids.py \
  --dataset outputs/validation_basic_solids
```

To create the same dataset in a **fresh** directory:

```bash
bash scripts/run_dyna.sh -u scripts/collect_validation_solids.py \
  --output outputs/validation_basic_solids_new
```

Frozen collection scripts and the acquisition log live in
`../project_log/validation_basic_solids`. Current DynaRaySim does not accept the
older v4 `ego.terrain_alignment` option; the scripted poses explicitly keep
roll and pitch zero, which is also checked for every recorded frame.
