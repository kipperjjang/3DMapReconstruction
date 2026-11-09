# Progress

- Existing v3 uses a 10x8 m corridor scene and 500 s per scene. The new plan
  expands area 20-fold and duration two-fold, with 400,000 total frames.
- User explicitly requested continuously moving, turning routes instead of
  back-and-forth motion. Box placement will preserve a clear swept route.
- Approximately 243 GB free disk space and 121 GiB available RAM observed.
- Preparing a full-sized pilot and verifying detached-process survival before
  launching the collection/validation supervisor.

## Pilot and launch results

- Initial route guard rejected a near-edge control point; relaxed the allowed
  control-point turn while retaining continuous forward motion and spline yaw.
  Retained the diagnostic log. All 40 production routes subsequently passed
  nonstationary/unique-pose and >20m X/Y exploration checks (`route_checks.log`).
- Changed floor depth to 2.5m so its underside stays below the training crop.
  The initial shallow-floor diagnostic is separate and not production data.
- Final full-sized pilot: 120 unique frames collected; whole-shard validation
  and all three independent reraycast/GT audits passed. Sensor up error=0;
  maximum mapper error ~2.51e-15. Dense GT coverage within 5cm ~52.37%, strict
  GT voxel recall ~26.27%; these are pilot metrics, not full-dataset results.
- Pilot GT generated in ~439.5s, retaining 9,319,785 surface samples; all scans
  completed in ~451.6s. Sample audit maximum distance <0.145mm.
- Pilot HDF5 ~826KB / 120 frames; static assets ~42.6MB per scene. Rough
  extrapolation ~4.45GB for 400,000 frames, subject to variation along the route.
  A hard 10GB cap/reserve remains active during collection.
- Supervisor `51497` (managed terminal session ID, NOT a host PID) passed the
  pilot gate and started scenes 0,1,2,3. Each entered full-size GT generation.
  Heartbeat remains current. Collection, full validation, 120-frame independent
  audit and final result writing proceed without any further model calls.
- Simple detached children are cleaned up in the restricted command environment;
  tmux sockets are unavailable there. Therefore the actual job uses a persistent
  managed command session. For a normal host terminal, `start_dataset_v4.sh`
  also supports the standard nohup background launch. Server/environment teardown
  is outside the persistence guarantee. No automatic full training was started.
- Frozen worker/validator/package source copies and external source hashes are
  recorded under the production log directory and dataset provenance.
- Added native SVG preview, v4 guide, progress helper and README link. Updated
  training launcher frame-count validation to support v4 (400k) and v3 (200k).
- Python/shell syntax, documentation links/fences and `git diff --check` passed.
