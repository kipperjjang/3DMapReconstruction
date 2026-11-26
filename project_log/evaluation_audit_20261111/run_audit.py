"""Run from the repository root using scripts/run_dyna.sh (GPU and display required)."""
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/evaluation_audit_20261111'
LOG = Path(__file__).resolve().parent
RESULTS = json.loads((LOG / 'commands.json').read_text()) if (LOG / 'commands.json').exists() else []


def run(name, arguments, expected=0, timeout=240):
    if any(row['name'] == name and row['passed'] for row in RESULTS):
        return
    command = [sys.executable, '-m', 'neural_scene.evaluation', *map(str, arguments)]
    start = time.perf_counter()
    with (LOG / (name + '.log')).open('w') as stream:
        result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                timeout=timeout)
    row = dict(name=name, command=command, returncode=result.returncode,
               expected=expected, passed=result.returncode == expected,
               seconds=round(time.perf_counter() - start, 2))
    RESULTS.append(row)
    (LOG / 'commands.json').write_text(json.dumps(RESULTS, indent=2) + '\n')
    print(name, row['passed'], row['seconds'], flush=True)


def export(name, *extra, checkpoint='training_run_v4_temporal', dataset='training_dataset_v4'):
    run(name, ['export', '--checkpoint', ROOT / 'outputs' / checkpoint / 'best.pt',
               '--dataset', ROOT / 'outputs' / dataset, '--output', OUT / name, *extra])


export('test', '--split', 'test', '--limit', '100', '--device', 'cuda:0')
export('train', '--split', 'train', '--limit', '5', '--device', 'cuda:0')
export('no_feedback', '--split', 'validation', '--limit', '100', '--no-feedback', '--device', 'cuda:0')
export('cpu', '--scene', 'scene_0000', '--limit', '2', '--device', 'cpu')
export('v3_temporal', '--scene', 'scene_0000', '--limit', '5', '--device', 'cuda:0',
       checkpoint='training_run_v3_temporal', dataset='training_dataset_v3')
export('v3_legacy', '--scene', 'scene_0000', '--limit', '5', '--device', 'cuda:0',
       checkpoint='training_run_v3', dataset='training_dataset_v3')

source = OUT / 'validation'
common = ['play', '--input', source, '--no-hold', '--fps', '1000', '--max-points', '1000']
run('vispy_layers_mesh', common + ['--layers', 'prediction', 'observation', 'target', '--mesh',
    '--prediction-opacity', '.8', '--observation-opacity', '.5', '--gt-opacity', '.2',
    '--mesh-opacity', '.15', '--point-size', '4', '--distance', '5'])
run('vispy_voxels', ['play', '--input', OUT / 'cpu', '--voxels', '--mesh', '--no-hold', '--fps', '1000'])
for foreground, reference in [('prediction', 'none'), ('observation', 'target'), ('target', 'observation')]:
    name = 'rerun_' + foreground
    run(name, common + ['--renderer', 'rerun', '--headless', '--recording', OUT / (name + '.rrd'),
                       '--foreground', foreground, '--reference', reference])
run('rerun_voxels_mesh', common + ['--renderer', 'rerun', '--headless', '--voxels', '--mesh',
    '--layers', 'prediction', 'observation', 'target', '--recording', OUT / 'voxels.rrd',
    '--prediction-opacity', '.8', '--observation-opacity', '.4', '--gt-opacity', '0', '--mesh-opacity', '.1'])
run('explicit_mesh', ['play', '--input', OUT / 'cpu', '--scene', 'scene_0000', '--mesh',
    OUT / 'cpu/scene_000/source_mesh.npz', '--renderer', 'rerun', '--headless',
    '--recording', OUT / 'explicit.rrd'])
for name, arguments, expected in [
    ('invalid_fps', common + ['--fps', '0'], 2),
    ('invalid_opacity', common + ['--gt-opacity', 'nan'], 2),
    ('invalid_headless', common + ['--headless'], 2),
    ('invalid_loop', common + ['--renderer', 'rerun', '--headless', '--loop', '--recording', OUT / 'unused.rrd'], 2),
    ('existing_recording', common + ['--renderer', 'rerun', '--headless', '--recording', OUT / 'explicit.rrd'], 1),
    ('missing_scene', common + ['--scene', 'does_not_exist'], 1),
    ('ambiguous_mesh', common + ['--mesh', OUT / 'cpu/scene_000/source_mesh.npz'], 1),
]:
    run(name, arguments, expected)

run('missing_mesh_after_fix', ['play', '--input', OUT / 'cpu', '--mesh', OUT / 'missing.obj',
    '--renderer', 'rerun', '--headless', '--recording', OUT / 'must_not_exist.rrd'], 1)
assert not (OUT / 'must_not_exist.rrd').exists()

run('rerun_gui', ['play', '--input', OUT / 'cpu', '--renderer', 'rerun', '--mesh',
    '--layers', 'prediction', 'observation', 'target'])

subprocess.run([sys.executable, str(LOG / 'check_lifecycle.py')], cwd=ROOT, check=True, timeout=120)

raise SystemExit(0 if all(row['passed'] for row in RESULTS) else 1)
