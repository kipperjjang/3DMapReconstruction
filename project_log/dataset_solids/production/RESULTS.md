# Solids dataset verified after viewer-only source drift

```json
{
  "status": "ready_for_training",
  "total_frames": 200000,
  "unique_acquired_poses": 200000,
  "split_frames": {
    "train": 160000,
    "validation": 20000,
    "test": 20000
  },
  "world_size_m": [
    40.0,
    40.0
  ],
  "duration_s_per_scene": 1000.0,
  "dynamic_objects": false,
  "geometry": "native basic solids and randomized boxes",
  "objects_per_scene": 160,
  "asset_types": 27,
  "dataset_bytes": 2982866944,
  "validation": {
    "status": "validated",
    "total_frames": 200000,
    "unique_geometry_scenes": 20,
    "split_frames": {
      "validation": 20000,
      "test": 20000,
      "train": 160000
    },
    "dataset_bytes": 2982866944,
    "max_sensor_up_error": 0.0,
    "max_feedback_transform_error": 9.797718192317006e-15,
    "mean_gt_point_coverage_5cm": 0.4945866279170573,
    "gt_point_coverage_p05_p50_p95": [
      0.40879460145370455,
      0.5007357460837819,
      0.528954868031367
    ],
    "mean_gt_voxel_recall": 0.2601622838029707
  },
  "audited_frames": 60,
  "recovery_evidence": "/home/hoon/ResearchProjects/3DMapReconstruction/project_log/dataset_solids/production/recovery_evidence.json"
}
```
