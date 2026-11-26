"""Check exported arrays and independently sum per-frame metric counts."""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/evaluation_audit_20261111'
results = {}
for name in ('validation', 'test', 'train', 'no_feedback', 'cpu', 'v3_temporal', 'v3_legacy', 'no_limit'):
    directory = OUT / name
    manifest = json.loads((directory / 'evaluation.json').read_text())
    report = json.loads((directory / 'report.json').read_text())
    assert manifest['status'] == 'completed'
    total = 0
    for sequence in manifest['sequences']:
        for metadata in sequence['frames']:
            with np.load(directory / metadata['file'], allow_pickle=False) as data:
                for key in ('prediction_points', 'current_points', 'target_points'):
                    assert data[key].ndim == 2 and data[key].shape[1] == 3
                    assert np.isfinite(data[key]).all()
                assert len(data['confidence']) == len(data['prediction_points'])
                assert np.isfinite(data['confidence']).all()
                assert data['robot_pose'].shape == (4, 4)
                assert int(data['frame_id']) == metadata['frame_id']
            total += 1
    for section in ('model', 'measurement_baseline'):
        for key in ('tp', 'fp', 'fn', 'frames'):
            summed = sum(f[section][key] for s in manifest['sequences'] for f in s['frames'])
            assert summed == report[section][key]
    results[name] = dict(frames=total, model_f1=report['model']['f1'],
                         baseline_f1=report['measurement_baseline']['f1'])

# Compare the first CPU frames with their GPU equivalents independent of sparse row order.
errors = []
for i in range(2):
    clouds = []
    for name in ('cpu', 'validation'):
        with np.load(OUT / name / 'scene_000' / f'frame_{i:05d}.npz') as data:
            points = data['prediction_points']
            cells = np.floor((points + 1.6) / .05).astype(int)
            clouds.append(points[np.lexsort(cells.T[::-1])])
    assert clouds[0].shape == clouds[1].shape
    errors.append(float(np.max(np.abs(clouds[0] - clouds[1]))))
assert max(errors) < 1e-4
results['cpu_gpu_max_coordinate_difference_m'] = max(errors)
(Path(__file__).resolve().parent / 'export_checks.json').write_text(json.dumps(results, indent=2) + '\n')
print(json.dumps(results, indent=2))
