# 2026-10-28: upright LiDAR / 200,000 unique timesteps

- Use DynaRaySim/torch and DynaMapping/torch under conda dyna, without editing
  external repositories or existing trained models/data.
- User revised the cap to 10 GB and prioritized 200,000 genuinely acquired
  timesteps. Distinct randomized scenes/trajectories, grouped scene splits.
- All recorded sensor rotations must satisfy R_world_sensor[:,2] = [0,0,1].
  Disable terrain tilt and rotational calibration noise; yaw remains variable.
- Measure GT-to-measurement coverage (GT denominator), including explicit 5 cm
  nearest-measurement and occupied-voxel recall. The paper does not define a
  matching tolerance and uses four depth cameras; do not claim exact replication.
- Pilot storage/time/coverage first. Store shared GT geometry and compressed
  training shards, not duplicated full-world GT in every frame. Record any
  quantization bound and preserve measurement multiplicities before merging.
- Keep finite source stream counts distinct from training sampled timesteps.
- Current dyna: Python 3.10, torch 2.6/cu118, NumPy 2.2.6, missing SciPy and ME;
  no exposed GPU. Network download failed. Compatible SciPy 1.15.3 is available
  read-only from capstone and linked through workspace .deps-dyna.
- Prepare a dyna-compatible network backend and run actual training checks;
  report GPU/device or execution blockers truthfully.
