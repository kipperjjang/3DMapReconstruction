# Basic solid validation dataset

- Collect 10,000 fresh frames at 10 Hz, reserved entirely for validation.
- Place all 27 meshes from DynaRaySim/resources/assets/basic_solid twice on a flat floor, preserving native metre dimensions.
- Script an upright ego trajectory through the object rows using timed interpolated waypoints. Protect the complete swept path and bring every instance into the local training crop.
- Reuse v4 LiDAR settings, current simulator surface GT, DynaMapping transforms and packed frame encoder.
- Save asset hashes, placement metadata, route checks and source snapshots. Validate every frame and independently raycast first/middle/last scans.
