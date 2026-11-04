# Preserve current and previous time coordinates

- Baseline committed as `7dee8a4` before implementation.
- Pack current observations at t=0 and aligned previous predictions at t=1,
  voxelizing each source independently so overlapping points remain distinct.
- Preserve both times through the spatial encoder/decoder. Collapse them with
  a learned 1x1x1x2, temporal-stride-two final offset convolution to one t=0 output.
- Keep spatial occupancy supervision at both times, and aggregate final
  confidence by spatial voxel. Detached feedback and pose alignment stay intact.
- Persist the policy in model configuration and checkpoints. Old checkpoints
  select their original merged policy automatically; raw v3 frame data is reusable.
- Check temporal input, both-source gradients, collapsed output uniqueness,
  complete pruning, checkpoint compatibility, training/resume and export.
- Update current documentation; preserve historical results and data manifests.
