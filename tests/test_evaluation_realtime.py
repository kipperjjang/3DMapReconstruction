"""Non-GUI tests for realtime input and ego integration."""
import unittest
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import numpy as np

from neural_scene.evaluation_realtime import keyboard_command, integrate_pose, Controls, advance_scene
from neural_scene.geometry import pose_matrix


class RealtimeTests(unittest.TestCase):
    def test_sensor_step_visits_all_command_deadlines(self):
        # Two command changes fall inside one scan interval; the third is exactly
        # on its boundary. SceneBatch rejects any step crossing these deadlines.
        deadlines = iter([23, 61, 100, 145])
        calls = []
        scene = SimpleNamespace(time_ns=0, deadline=next(deadlines))
        scene.next_command_ns = lambda: scene.deadline

        def step(timestamp):
            self.assertLessEqual(timestamp, scene.deadline)
            self.assertGreaterEqual(timestamp, scene.time_ns)
            calls.append(timestamp)
            scene.time_ns = timestamp
            if timestamp == scene.deadline:
                scene.deadline = next(deadlines)

        scene.step = step
        advance_scene(scene, 0)
        advance_scene(scene, 100)
        advance_scene(scene, 120)
        self.assertEqual(calls, [0, 23, 61, 100, 120])

    @unittest.skipUnless(os.environ.get('RUN_SIM_TESTS'), 'requires DynaRaySim and Warp')
    def test_fresh_raycast_after_ego_motion_without_gui(self):
        import dynaraysim
        import yaml
        from neural_scene.evaluation_realtime import LiveScene
        source = Path(dynaraysim.__file__).resolve().parents[1] / 'examples/demo.yaml'
        data = yaml.safe_load(source.read_text())
        data['run']['environments'] = 1
        for asset in data['assets'].values():
            asset['path'] = str((source.parent / asset['path']).resolve())
        args = SimpleNamespace(sim_device='cpu', device='cpu', sensor='lidar', speed=.5, angular_speed=30.)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'scene.yaml'
            path.write_text(yaml.safe_dump(data))
            live = LiveScene(path, args)
            first = live.acquire(np.zeros(6), args)
            second = live.acquire(np.array([1, 0, 0, 0, 0, 0]), args)
            self.assertEqual(first[3], 0)
            self.assertEqual(second[3], 1)
            self.assertGreater(second[2], first[2])
            self.assertGreater(len(first[0]), 0)
            self.assertTrue(np.isfinite(second[0]).all())
            self.assertGreater(np.linalg.norm(second[1][:3] - first[1][:3]), 0)

    def test_all_requested_key_directions_and_opposites(self):
        for axis, positive, negative in ((0, 'W', 'S'), (1, 'A', 'D'), (2, 'H', 'G'),
                                         (3, 'Y', 'T'), (4, 'I', 'U'), (5, 'Q', 'E')):
            self.assertEqual(keyboard_command([positive])[axis], 1)
            self.assertEqual(keyboard_command([negative])[axis], -1)
            np.testing.assert_array_equal(keyboard_command([positive, negative]), np.zeros(6))

    def test_body_translation_world_height_and_normalized_quaternion(self):
        pose = np.array([0, 0, 1., 0, 0, np.sin(np.pi/4), np.cos(np.pi/4)])
        moved = integrate_pose(pose, [1, 0, 1, 0, 0, 0], .1, 2, 1)
        np.testing.assert_allclose(moved[:3], [0, .2, 1.2], atol=1e-8)
        rotated = integrate_pose(pose, [0, 0, 0, 1, 1, 1], .1, 2, 1)
        self.assertAlmostEqual(np.linalg.norm(rotated[3:]), 1.)
        pose_matrix(rotated)
        np.testing.assert_array_equal(integrate_pose(pose, np.zeros(6), .1, 2, 1), pose)

    def test_joystick_deadzone_axis_signs_and_clipping(self):
        controls = Controls.__new__(Controls)
        controls.pressed = {'W'}
        controls.args = SimpleNamespace(joystick=0, joystick_axes=list(range(6)),
            joystick_signs=[1, -1, 1, 1, 1, 1], deadzone=.2)
        values = [.6, .6, .1, 0, 0, 0]
        controls.joystick = SimpleNamespace(get_init=lambda: True, get_axis=lambda i: values[i])
        controls.pygame = SimpleNamespace(event=SimpleNamespace(pump=lambda: None),
            joystick=SimpleNamespace(get_count=lambda: 1))
        np.testing.assert_allclose(controls.command(), [1, -.5, 0, 0, 0, 0])
        controls.pygame.joystick.get_count = lambda: 0
        with self.assertRaises(RuntimeError):
            controls.command()

    def test_key_release_and_scene_action(self):
        controls = Controls.__new__(Controls)
        controls.pressed, controls.action = set(), None
        event = SimpleNamespace(key=SimpleNamespace(name='W'))
        controls.press(event)
        self.assertEqual(keyboard_command(controls.pressed)[0], 1)
        controls.release(event)
        self.assertEqual(keyboard_command(controls.pressed)[0], 0)
        controls.press(SimpleNamespace(key=SimpleNamespace(name='N')))
        self.assertEqual(controls.action, 'next')


if __name__ == '__main__':
    unittest.main()
