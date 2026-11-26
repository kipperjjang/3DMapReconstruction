"""Playback geometry checks that do not require the model backend."""
import argparse
import json
import os
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from neural_scene.evaluation import comparison_snapshot, opacity, playback
from neural_scene.evaluation_mesh import (load_mesh, mesh_in_robot, export_source_mesh,
                                          resolve_mesh)


class VisualizationTests(unittest.TestCase):
    def test_realtime_camera_stays_at_ego_after_pan_and_pose_change(self):
        from neural_scene.evaluation_viewer import ComparisonViewer
        viewer = ComparisonViewer.__new__(ComparisonViewer)
        viewer.follow_ego = True
        viewer._mesh = None
        viewer.window = SimpleNamespace(
            camera=SimpleNamespace(center=(4., -2., 1.), azimuth=27., distance=3.),
            canvas=SimpleNamespace(update=lambda: None),
            app=SimpleNamespace(process_events=lambda: None))
        for translation in ((10., 20., 1.), (-30., 5., 2.)):
            pose = np.eye(4)
            pose[:3, 3] = translation
            viewer._update_vispy(SimpleNamespace(layers={}, current=SimpleNamespace(sensor_pose=pose)))
            self.assertEqual(viewer.window.camera.center, (0., 0., 0.))
            self.assertEqual(viewer.window.camera.azimuth, 27.)
            self.assertEqual(viewer.window.camera.distance, 3.)

    def test_missing_or_invalid_mesh_fails_before_creating_recording(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'evaluation.json').write_text(json.dumps(dict(
                format='neural-scene-evaluation-v1', status='completed',
                sequences=[dict(scene='sample', frames=[])])))
            args = SimpleNamespace(input=root, scene=None, layers=None,
                foreground='prediction', reference='target', dataset=None)
            for filename, error in [('missing.obj', FileNotFoundError), ('invalid.obj', ValueError)]:
                args.mesh = str(root / filename)
                if filename == 'invalid.obj':
                    (root / filename).write_text('v 0 0 0\n')
                with patch('neural_scene.evaluation_viewer.ComparisonViewer') as viewer:
                    with self.assertRaises(error):
                        playback(args)
                    viewer.assert_not_called()

    @unittest.skipUnless(os.environ.get('RUN_VIEWER_TESTS'), 'requires installed renderers and a display')
    def test_renderers_draw_mesh_clouds_voxels_and_apply_live_opacity(self):
        from neural_scene.evaluation_viewer import ComparisonViewer
        data = dict(prediction_points=np.array([[0., 0., 0.], [.2, .2, .2]], dtype=np.float32),
                    current_points=np.array([[.1, 0., 0.]], dtype=np.float32),
                    target_points=np.empty((0, 3), dtype=np.float32),
                    robot_pose=np.eye(4), timestamp_ns=7, frame_id=2)
        names = ['prediction', 'observation', 'target']
        grid = dict(origin=[-1, -1, -1], voxel_size=.05, shape=[40, 40, 40])
        mesh = (np.array([[-1, -1, -.2], [1, -1, -.2], [0, 1, -.2]], dtype=np.float32),
                np.array([[0, 1, 2]], dtype=np.uint32))
        with tempfile.TemporaryDirectory() as tmp:
            recording = Path(tmp) / 'test.rrd'
            for renderer in ('vispy', 'rerun'):
                options = dict(show=False, point_size=15, distance=3)
                if renderer == 'rerun':
                    options['recording_path'] = str(recording)
                window = ComparisonViewer(renderer, names,
                    dict(prediction=.8, observation=.5, target=.2, mesh=.25), **options)
                try:
                    window.set_mesh(mesh)
                    for voxels in (False, True):
                        window.update(comparison_snapshot(data, 'prediction', 'target', grid, voxels, names))
                        if renderer == 'vispy':
                            image = window.canvas.render()
                            self.assertTrue(image.any())
                            # Control hints and layer opacities must remain inside the canvas.
                            white = (image[8:120, 8:500, :3] > 200).all(axis=2)
                            self.assertGreater(np.count_nonzero(white.any(axis=1)), 25)
                            window._key_press(SimpleNamespace(key=SimpleNamespace(name='1'), text='1'))
                            window._key_press(SimpleNamespace(key=SimpleNamespace(name='0'), text='0'))
                            self.assertEqual(window.opacities['prediction'], 0)
                            self.assertTrue(all(v.opacity == 0 for v in window._visuals['prediction']))
                            hidden = window.canvas.render()
                            self.assertFalse(np.array_equal(image, hidden))
                            window._key_press(SimpleNamespace(key=SimpleNamespace(name='0'), text='0'))
                            self.assertAlmostEqual(window.opacities['prediction'], .8)
                    window.set_mesh(None)
                    window.update(comparison_snapshot(data, 'prediction', 'target', grid, False, names))
                finally:
                    window.close()
            self.assertGreater(recording.stat().st_size, 0)

    def test_world_mesh_matches_robot_cloud_under_rotation_and_translation(self):
        pose = np.array([[0, -1, 0, 4], [1, 0, 0, 5], [0, 0, 1, 2], [0, 0, 0, 1.]])
        local = np.array([[1., 2., 3.], [-2, .5, 0]])
        world = local @ pose[:3, :3].T + pose[:3, 3]
        np.testing.assert_allclose(mesh_in_robot(world, pose), local)

    def test_bundled_mesh_survives_removing_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scene = root / 'dataset' / 'scenes' / 'sample'
            scene.mkdir(parents=True)
            obj = scene / 'scene.obj'
            obj.write_text('v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nf -4 -3 -2 -1\n')
            vertices, faces = load_mesh(obj)
            self.assertEqual(faces.shape, (2, 3))
            export = root / 'export'
            entry = dict(scene='sample', directory='scene_000')
            export_source_mesh(root / 'dataset', dict(shard='scenes/sample/frames.h5'), export, entry)
            obj.unlink()
            path = resolve_mesh('auto', export, {}, entry)
            actual = load_mesh(path)
            np.testing.assert_array_equal(actual[0], vertices)
            np.testing.assert_array_equal(actual[1], faces)
            entry['mesh']['file'] = '../escape.npz'
            with self.assertRaises(ValueError):
                resolve_mesh('auto', export, {}, entry)

    def test_three_clouds_and_voxels_keep_distinct_geometry(self):
        data = dict(prediction_points=np.array([[.01, .02, .03]]),
                    current_points=np.array([[.11, .02, .03]]),
                    target_points=np.empty((0, 3)), robot_pose=np.eye(4), timestamp_ns=7, frame_id=2)
        names = ['prediction', 'observation', 'target']
        snapshot = comparison_snapshot(data, 'prediction', 'target',
            dict(origin=[0, 0, 0], voxel_size=.05, shape=[16, 16, 16]), True, names)
        self.assertEqual(list(snapshot.layers), names)
        np.testing.assert_allclose(snapshot.layers['observation'].voxel_centers, [[.125, .025, .025]])
        self.assertEqual(snapshot.layers['target'].voxel_centers.shape, (0, 3))

    def test_opacity_rejects_invalid_values(self):
        for value in ('-0.1', '1.1', 'nan', 'inf'):
            with self.assertRaises(argparse.ArgumentTypeError):
                opacity(value)
        self.assertEqual(opacity('0'), 0)
        self.assertEqual(opacity('1'), 1)

    def test_legacy_export_finds_relocated_dataset_mesh(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / 'scenes' / 'sample'
            folder.mkdir(parents=True)
            source = folder / 'scene.obj'
            source.write_text('v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n')
            (root / 'dataset.json').write_text(json.dumps(dict(sequences=[
                dict(scene='sample', shard='scenes/sample/frames.h5')])))
            self.assertEqual(resolve_mesh('auto', root, dict(dataset='/missing/old/path'),
                dict(scene='sample'), dataset=root), source)

    @unittest.skipUnless(os.environ.get('RUN_VIEWER_TESTS'), 'requires installed renderers and a display')
    def test_playback_command_with_bundled_mesh(self):
        import subprocess
        import sys
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            np.savez(root / 'frame.npz', prediction_points=np.array([[0., 0., 0.]], dtype=np.float32),
                current_points=np.array([[0., .1, 0.]], dtype=np.float32), target_points=np.empty((0, 3)),
                robot_pose=np.eye(4), timestamp_ns=0, frame_id=0)
            np.savez(root / 'mesh.npz', vertices=np.array([[0., 0., -.1], [1., 0., -.1], [0., 1., -.1]]),
                faces=np.array([[0, 1, 2]], dtype=np.int32))
            (root / 'evaluation.json').write_text(json.dumps(dict(
                format='neural-scene-evaluation-v1', status='completed',
                grid=dict(origin=[-1, -1, -1], shape=[40, 40, 40], voxel_size=.05),
                sequences=[dict(scene='sample', mesh=dict(file='mesh.npz', point_frame='world', units='metres'),
                    frames=[dict(file='frame.npz', frame_id=0, model=dict(f1=1.))])])) )
            common = [sys.executable, '-m', 'neural_scene.evaluation', 'play', '--input', str(root),
                      '--layers', 'prediction', 'observation', 'target', '--mesh', '--no-hold',
                      '--gt-opacity', '.2', '--fps', '1000']
            subprocess.run(common, check=True, capture_output=True, timeout=30)
            subprocess.run(common + ['--viewer', 'rerun', '--headless', '--voxels',
                '--recording', str(root / 'cli.rrd')], check=True, capture_output=True, timeout=30)
            self.assertGreater((root / 'cli.rrd').stat().st_size, 0)


if __name__ == '__main__':
    unittest.main()
