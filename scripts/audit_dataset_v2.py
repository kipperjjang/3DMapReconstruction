"""Independently compare prepared frames with recorded observations and fresh GT."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prepare_dataset_v2 as prep
import numpy as np
from scipy.spatial import cKDTree


def main():
    """Audit first/middle/last frames without relying on the stored transformations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=prep.ROOT / 'outputs/training_dataset_v2')
    args = parser.parse_args()
    manifest = json.loads((args.dataset / 'dataset.json').read_text())
    origin = np.asarray(manifest['grid']['origin'], dtype=np.float32)
    shape = np.asarray(manifest['grid']['shape'])
    reports = []
    for sequence in manifest['sequences']:
        dataset = prep.Dataset(sequence['archive'], verify=False)
        reader = prep.DynaRaySimReader(sequence['archive'], device='cpu')
        for index in sorted({0, len(reader) // 2, len(reader) - 1}):
            raw = reader.load_frame(index)
            world_sensor = raw.sensor_pose.numpy()
            world_robot = raw.robot_pose.numpy()
            # Use the rigid-transform convention of the data contract. A
            # general matrix inverse introduces ~1e-15 differences that can
            # move an exactly grid-aligned wall across a half-open boundary.
            robot_world = np.eye(4)
            robot_world[:3, :3] = world_robot[:3, :3].T
            robot_world[:3, 3] = -world_robot[:3, :3].T @ world_robot[:3, 3]
            sensor_robot = robot_world @ world_sensor
            points = raw.points.numpy().astype(np.float64)
            current = points @ sensor_robot[:3, :3].T + sensor_robot[:3, 3]
            current = prep.crop(current, origin, shape, 0.05)
            gt = dataset.read_gt_window(0, 'lidar', index, 1, mode='composed', frame='world', radius_m=4.0)[0]
            points = gt['points'].numpy().astype(np.float64)
            target = points @ robot_world[:3, :3].T + robot_world[:3, 3]
            target = prep.crop(target, origin, shape, 0.05)
            expected = prep.centroid_cloud(target, origin, 0.05)
            path = args.dataset / 'sequences' / sequence['scene'] / sequence['frames'][index]['file']
            with np.load(path, allow_pickle=False) as frame:
                np.testing.assert_allclose(frame['current_points'], current, atol=2e-6, rtol=0)
                np.testing.assert_allclose(frame['target_points'], expected, atol=2e-6, rtol=0)
            distances = cKDTree(target).query(current)[0] if len(current) and len(target) else np.empty(0)
            reports.append({'scene': sequence['scene'], 'index': index, 'observations': len(current),
                'surface_gt_points_in_crop': len(target), 'target_voxels': len(expected),
                'observation_surface_fraction_within_5cm': float(np.mean(distances <= 0.05)) if len(distances) else None,
                'observation_surface_p95_m': float(np.quantile(distances, .95)) if len(distances) else None,
                'observation_surface_max_m': float(distances.max()) if len(distances) else None})
            print(json.dumps(reports[-1]), flush=True)
    prep.atomic_json(args.dataset / 'sample_audit.json', {
        'status': 'passed_coordinate_and_target_reconstruction', 'sample_frames': len(reports),
        'scope': 'first/middle/last frame of each scene; nearest-surface coverage is diagnostic, not a boundary-resolution audit',
        'frames': reports})


if __name__ == '__main__':
    main()
