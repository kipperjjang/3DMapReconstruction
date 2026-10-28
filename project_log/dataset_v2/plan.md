# Fresh GT dataset and training (2026-10-20)

- Generate the full seven current DynaRaySim scenarios (expected 1,420 frames;
  train 1,040 / validation 180 / test 200), preserving scenario-level splits.
- Keep the entire new dataset, including archives and NPZ, below 5,000,000,000
  bytes. Monitor size during collection with a conservative reserve.
- Use the current PyTorch GT implementation, composed mode to keep full-scene
  storage bounded. This retains the documented dynamic-contact approximation.
- Prefer matching auto-resolution audit; if the migrated pattern fingerprint
  invalidates the old audit, explicitly record numeric 5 cm as unaudited.
- Use /home/intern/DynaMapping, convert recorded sensor points into robot
  coordinates before its actual DynaMapping alignment. Retain raw scan counts.
- Validate every frame, independent pose transforms, unique GT cells and hashes.
- Train the existing full network with noise 0.005 m, batch 2, rollout 12,
  1,500 updates, LR 0.01 to 0.0001, validation selection, then held-out test.
- Preserve old data/checkpoints and external repositories. GPU is currently
  unavailable (no /dev/nvidia devices); record actual execution without claiming
  a trained model if the environment prevents the planned GPU run.
