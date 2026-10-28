"""Measure augmentation support corruption on training frames only."""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def keys(points, origin, size, shape):
    cells = np.floor((points - origin) / size).astype(np.int64)
    cells = cells[np.all((cells >= 0) & (cells < shape), axis=1)]
    return np.unique((cells[:, 0] * shape[1] + cells[:, 1]) * shape[2] + cells[:, 2])


def main():
    directory = ROOT / 'outputs' / 'training_dataset_v1'
    manifest = json.loads((directory / 'dataset.json').read_text())
    grid = manifest['grid']
    origin, size, shape = np.asarray(grid['origin'], dtype=np.float32), np.float32(grid['voxel_size']), np.asarray(grid['shape'])
    rng = np.random.default_rng(20261012)
    totals = {str(bound): {'true_positive_voxels': 0, 'observation_voxels': 0, 'frames': 0}
              for bound in (0., .005, .05)}
    for sequence in manifest['sequences']:
        if sequence['split'] != 'train':
            continue
        for metadata in sequence['frames'][::5]:
            with np.load(directory / 'sequences' / sequence['scene'] / metadata['file'], allow_pickle=False) as data:
                current = data['current_points']
                target = keys(data['target_points'], origin, size, shape)
            unit_noise = rng.uniform(-1, 1, current.shape).astype(np.float32)
            for bound, totals_for_bound in totals.items():
                observed = keys(current + float(bound) * unit_noise, origin, size, shape)
                totals_for_bound['true_positive_voxels'] += int(np.isin(observed, target).sum())
                totals_for_bound['observation_voxels'] += len(observed)
                totals_for_bound['frames'] += 1
    for value in totals.values():
        value['precision_of_augmented_observation_support'] = value['true_positive_voxels'] / max(value['observation_voxels'], 1)
    report = {'split': 'train only', 'sampling': 'every fifth frame from all five training scenes',
              'noise_uniform_bound_m': totals}
    output = ROOT / 'outputs' / 'accuracy_study_v2' / 'training_noise_diagnostic.json'
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
