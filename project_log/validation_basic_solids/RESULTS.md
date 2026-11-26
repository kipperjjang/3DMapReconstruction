# Completed validation dataset

- Output: `outputs/validation_basic_solids/dataset.json`
- Frames: 10,000, all validation; 10,000 unique acquired poses.
- All 27 native assets placed twice: 54 instances, each with GT samples and actual stored LiDAR returns.
- Flat floor: 30 × 24 m. Scripted interpolated aisle route, 0.65 m/s, ego Z=0.7 m.
- Full record validation and first/middle/last independent raycast and GT audits passed. Maximum reconstruction error <0.170 mm.
- No motion rejections or partial shards. External simulator/mapping sources unchanged.
- Dataset size: approximately 120 MB. Collection plus core verification: 793 seconds.
- Mean GT point coverage within 5 cm: 44.56%; mean strict GT voxel recall: 20.24%. These are measurement coverage metrics, not model accuracy.

```json
{
  "status": "ready_for_validation",
  "total_frames": 10000,
  "unique_geometry_scenes": 1,
  "split_frames": {
    "validation": 10000
  },
  "dataset_bytes": 119746560,
  "max_sensor_up_error": 0.0,
  "max_feedback_transform_error": 5.009881398621019e-15,
  "mean_gt_point_coverage_5cm": 0.4455678311584896,
  "gt_point_coverage_p05_p50_p95": [
    0.306742936890886,
    0.47009633155825764,
    0.5025216495058238
  ],
  "mean_gt_voxel_recall": 0.20241070429933017,
  "audited_frames": 3,
  "elapsed_s": 792.9813916683197
}
```

Asset representation audit:

```json
{
  "status": "passed",
  "method": "World AABB membership above Z=1mm, tolerance 0.2mm; counts use stored local-crop measurements",
  "placed_instances": 54,
  "unique_assets": 27,
  "minimum_copies_per_asset": 2,
  "instances_with_gt_samples": 54,
  "instances_with_acquired_returns": 54
}
```
