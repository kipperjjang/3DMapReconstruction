"""Exercise real VisPy lifecycle, uncapped export, and legacy mesh relocation."""
import copy
import faulthandler
import json
import os
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
from unittest.mock import patch

from neural_scene.evaluation import main
from neural_scene.evaluation_viewer import ComparisonViewer

faulthandler.dump_traceback_later(20, repeat=False)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/evaluation_audit_20261111'
LOG = Path(__file__).resolve().parent
results = {}


def cli(*args):
    with patch.object(sys, 'argv', ['evaluation', *map(str, args)]):
        main()


# Preserve the real frame contents while shortening only the audit manifest.
clip = OUT / 'twelve_frame_dataset'
if not clip.exists():
    clip.mkdir()
    source = ROOT / 'outputs/training_dataset_v4'
    manifest = json.loads((source / 'dataset.json').read_text())
    selected = []
    for split in ('train', 'validation', 'test'):
        sequence = copy.deepcopy(next(s for s in manifest['sequences'] if s['split'] == split))
        sequence['frames'] = sequence['frames'][:12]
        selected.append(sequence)
        shard = Path(sequence['shard'])
        (clip / shard).parent.mkdir(parents=True, exist_ok=True)
        os.link(source / shard, clip / shard)
        os.link(source / shard.parent / 'scene.obj', clip / shard.parent / 'scene.obj')
    manifest['sequences'] = selected
    (clip / 'dataset.json').write_text(json.dumps(manifest))
if not (OUT / 'no_limit').exists():
    cli('export', '--checkpoint', ROOT / 'outputs/training_run_v4_temporal/best.pt',
        '--dataset', clip, '--output', OUT / 'no_limit', '--device', 'cuda:0')
export = json.loads((OUT / 'no_limit/evaluation.json').read_text())
assert export['limit_per_scene'] is None
assert len(export['sequences'][0]['frames']) == 12
results['no_limit_real_frames'] = 12


class LifecycleViewer(ComparisonViewer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.updates = 0
        self.held = False
        windows.append(self)

    def update(self, snapshot):
        super().update(snapshot)
        self.updates += 1
        print('Lifecycle update', mode, self.updates, flush=True)
        if self.updates == 1:
            for key, char in [('1', '1'), ('+', '+'), ('-', '-'), ('0', '0'), ('0', '0')]:
                self._key_press(SimpleNamespace(key=SimpleNamespace(name=key), text=char))
            self.window._key_press(SimpleNamespace(key=SimpleNamespace(name='R')))
            from vispy.io import write_png
            write_png(str(LOG / 'vispy.png'), self.canvas.render())

    def wait_until(self, deadline):
        if mode == 'loop' and self.updates == 4:
            self.window._key_press(SimpleNamespace(key=SimpleNamespace(name='Escape')))
            return False
        return self.window.wait_until(deadline)

    def hold(self):
        from vispy import app
        self.held = True
        self.timer = app.Timer(app=self.window.app, interval=.15, iterations=1,
                               connect=lambda event: self.window.close(), start=True)
        return self.window.hold()


for mode in ('loop', 'hold'):
    print('Starting lifecycle', mode, flush=True)
    windows = []
    with patch('neural_scene.evaluation_viewer.ComparisonViewer', LifecycleViewer):
        cli('play', '--input', OUT / 'cpu', '--fps', '1000', '--mesh',
            *(['--loop', '--no-hold'] if mode == 'loop' else []))
    window = windows[0]
    assert window.closed
    assert window.updates == (4 if mode == 'loop' else 2)
    assert window.held == (mode == 'hold')
    results[mode] = dict(updates=window.updates, held=window.held, closed=window.closed)
    print('Finished lifecycle', mode, flush=True)

legacy = OUT / 'legacy_export'
if not legacy.exists():
    shutil.copytree(OUT / 'cpu', legacy)
    manifest = json.loads((legacy / 'evaluation.json').read_text())
    manifest['dataset'] = '/missing/original/dataset'
    for sequence in manifest['sequences']:
        sequence.pop('mesh', None)
    (legacy / 'evaluation.json').write_text(json.dumps(manifest))
recording = OUT / 'relocated.rrd'
if not recording.exists():
    cli('play', '--input', legacy, '--mesh', '--dataset', ROOT / 'outputs/training_dataset_v4',
        '--renderer', 'rerun', '--headless', '--recording', recording)
assert recording.stat().st_size > 0
results['legacy_relocated_mesh'] = True
(LOG / 'lifecycle.json').write_text(json.dumps(results, indent=2) + '\n')
print(json.dumps(results), flush=True)
faulthandler.cancel_dump_traceback_later()
