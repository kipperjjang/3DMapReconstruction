# Wide flat box terrain, continuous turning trajectories

- Preserve v3 data and all existing training runs. Use DynaRaySim/torch and
  DynaMapping/torch under conda dyna, with sensor Z aligned to world Z.
- New terrain: 40 m x 40 m flat floor, 160 static boxes per scene, independently
  sampled footprint sizes 0.2–3 m and heights 0.1–2.2 m.
- Replace corridor oscillation with smooth, forward-moving roaming paths across
  the floor. Yaw follows the path tangent; continuous segments must clear every
  box's inflated footprint. No rotation-only stops or exact pose replay.
- Planned production: 40 scenes x 10,000 unique frames at 10 Hz = 400,000 frames,
  1,000 simulated seconds per scene (twice v3's per-scene duration).
- Reuse the packed 64-cubed, 5 cm robot-frame training grid and actual mapper
  relative-pose calculations. A wider world does not change the network grid.
- Keep the previously authorized 10,000,000,000-byte cap for this new dataset.
- Pilot full-sized geometry before launch; measure storage and visibility
  proxies honestly. Do not filter GT to match the paper's 43% reference.
- Launch an independent, detached supervisor that owns collection, full shard
  validation and independent representative reraycast audits. Persist status,
  logs, completion/failure reports, PID information and source snapshots.
