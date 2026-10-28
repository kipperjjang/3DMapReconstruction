"""Monitor the launched collector/validators and publish a persistent handoff."""

import hashlib
import json
from pathlib import Path
import time
import traceback


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'outputs/training_dataset_v3'
LOG = ROOT / 'project_log/dataset_v3'


def read(path):
    """Read JSON committed by a completed stage."""
    return json.loads(path.read_text())


def publish(value):
    """Atomically expose the final status to humans and other processes."""
    temporary = LOG / 'final_status.json.tmp'
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(LOG / 'final_status.json')
    print(json.dumps(value, indent=2, ensure_ascii=False), flush=True)


def main():
    """Wait for existing jobs; never label partial validation as complete."""
    deadline = time.monotonic() + 7200
    required = [DATA / name for name in (
        'collection_summary.json', 'validation_report.json', 'sample_audit.json')]
    publish({'status': 'waiting', 'message': 'Collection and both previously launched validation jobs are running.'})
    while not all(path.exists() for path in required):
        for name in ('collection.log', 'final_validation.log', 'final_audit.log',
                     'tail_scene_36.log', 'tail_scene_37.log', 'tail_scene_38.log', 'tail_scene_39.log'):
            path = LOG / name
            if path.exists():
                tail = path.read_text(errors='replace')[-16000:]
                if 'Traceback (most recent call last)' in tail:
                    raise RuntimeError('A pipeline job failed; inspect ' + str(path))
        if time.monotonic() > deadline:
            raise TimeoutError('Pipeline did not finish within two hours; inspect collection and validation logs.')
        time.sleep(15)
    collected, validated, audit = map(read, required)
    assert collected['status'] == 'collected' and collected['total_frames'] == 200000
    assert validated['status'] == 'validated' and validated['total_frames'] == 200000
    assert validated['unique_geometry_scenes'] == 40
    assert validated['split_frames'] == {'train': 160000, 'validation': 20000, 'test': 20000}
    assert audit['status'] == 'passed' and audit['frames'] == 120
    provenance = read(DATA / 'provenance.json')
    changed = [name for name, digest in provenance['sources'].items()
               if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest]
    assert not changed, 'Simulator/mapper source changed during collection: ' + str(changed)
    size = sum(max(p.stat().st_size, p.stat().st_blocks * 512)
               for p in DATA.rglob('*') if p.is_file())
    assert size < 10_000_000_000
    assert not list(DATA.rglob('*.partial'))
    max_error = max(row[key] for row in audit['samples']
                    for key in ('current_points_max_distance_m', 'target_points_max_distance_m'))
    result = {
        'status': 'ready_for_user_training', 'total_frames': 200000,
        'scenes': 40, 'split_frames': validated['split_frames'],
        'dataset_bytes': size, 'dynamic_objects': False,
        'max_sensor_up_error': validated['max_sensor_up_error'],
        'max_feedback_transform_error': validated['max_feedback_transform_error'],
        'mean_gt_point_coverage_5cm': validated['mean_gt_point_coverage_5cm'],
        'mean_gt_voxel_recall': validated['mean_gt_voxel_recall'],
        'independent_audit_frames': audit['frames'],
        'independent_audit_max_distance_m': max_error,
        'external_source_files_verified': len(provenance['sources']),
        'full_training_started_by_agent': False,
        'training_command': 'conda activate dyna && bash scripts/train_v3.sh',
    }
    report = f'''# Dataset v3 results

Status: **ready_for_user_training**. Collection and final validation completed.

- 200,000 acquired frames; 40 unique static scenes.
- Train / validation / test: 160,000 / 20,000 / 20,000 frames.
- Final dataset size: {size:,} bytes ({size / 1e9:.3f} GB), below 10 GB.
- Dynamic objects: **none**; only the LiDAR moves and rotates.
- Maximum sensor Z-axis error: {validated['max_sensor_up_error']:.3g}.
- Maximum DynaMapping feedback transform error: {validated['max_feedback_transform_error']:.3g}.
- Mean GT coverage within 5 cm: {validated['mean_gt_point_coverage_5cm']:.2%}.
- Mean strict GT voxel recall: {validated['mean_gt_voxel_recall']:.2%}.
- Independent reraycast / GT reconstruction: all {audit['frames']} sampled frames passed;
  maximum point error {max_error * 1000:.4f} mm.
- Source hashes: all {len(provenance['sources'])} simulator/mapper Python files unchanged.

The 5 cm coverage is an explicit distance-based proxy, not an exact reproduction
of the paper's unspecified 43% visibility measure. Pilot frames are excluded.
Full GPU training has not been started by the agent; the user will launch it.

```bash
cd "/home/intern/Neural Scene unet"
conda activate dyna
bash scripts/train_v3.sh
# Resume an interrupted run:
# bash scripts/train_v3.sh --resume
```

The launcher checks GPU access, validates all 200,000 frames, runs a CUDA smoke
test, then starts 20,000 updates. Outputs: `outputs/training_run_v3`.
Actual CUDA execution is checked on the user's host; GPU devices were unavailable
in the agent's execution environment. See `DATASET_V3_GUIDE.md` for details.

Evidence: `outputs/training_dataset_v3/validation_report.json`,
`outputs/training_dataset_v3/sample_audit.json`, and
`project_log/dataset_v3/final_status.json`.
'''
    (ROOT / 'DATASET_V3_RESULTS.md').write_text(report)
    with (LOG / 'progress.md').open('a') as stream:
        stream.write('\n## Final completion\n\n' + report + '\n')
    publish(result)
    print('\n검증 완료. 학습 실행: conda activate dyna && bash scripts/train_v3.sh', flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        publish({'status': 'failed', 'error': str(error),
                 'details': 'project_log/dataset_v3/finalize.log'})
        traceback.print_exc()
        raise SystemExit(1)
