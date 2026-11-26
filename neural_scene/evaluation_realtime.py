"""Drive a DynaRaySim ego and run checkpoint inference on freshly acquired scans.

Run with scripts/run_dyna.sh -m neural_scene.evaluation_realtime --help.
The simulation clock advances at the sensor frequency; inference overruns slow
playback rather than queue stale observations. GUI validation is separate.
"""
import argparse
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

from .geometry import pose_matrix
from .evaluation import comparison_snapshot, opacity, positive_float, positive_int
from .inference_timing import InferenceTiming


# Command order: body forward, body left, world height, body roll/pitch/yaw.
KEY_BINDINGS = {
    'W': (0, 1), 'S': (0, -1), 'A': (1, 1), 'D': (1, -1),
    'H': (2, 1), 'G': (2, -1), 'Y': (3, 1), 'T': (3, -1),
    'I': (4, 1), 'U': (4, -1), 'Q': (5, 1), 'E': (5, -1),
}


def keyboard_command(pressed):
    """Combine held keys into a bounded six-axis command."""
    command = np.zeros(6)
    for key in pressed:
        if key in KEY_BINDINGS:
            axis, sign = KEY_BINDINGS[key]
            command[axis] += sign
    return np.clip(command, -1, 1)


def integrate_pose(pose, command, dt, speed, angular_speed):
    """Integrate body XY/rotation and world Z; angular speed is radians/s."""
    pose = np.asarray(pose, dtype=np.float64).copy()
    command = np.asarray(command, dtype=np.float64)
    if pose.shape != (7,) or command.shape != (6,) or not np.isfinite(command).all():
        raise ValueError('Expected XYZ/XYZW pose and finite six-axis command')
    rotation = pose_matrix(pose)[:3, :3]
    # Normalize diagonal travel to the configured planar speed.
    planar = command[:2] / max(1., np.linalg.norm(command[:2]))
    pose[:3] += rotation @ np.r_[planar * speed * dt, 0.]
    pose[2] += command[2] * speed * dt
    rotvec = command[3:] * angular_speed * dt
    angle = np.linalg.norm(rotvec)
    dq = np.r_[rotvec * (np.sin(angle / 2) / angle if angle else .5), np.cos(angle / 2)]
    q = pose[3:] / np.linalg.norm(pose[3:])
    pose[3:] = np.r_[q[3] * dq[:3] + dq[3] * q[:3] + np.cross(q[:3], dq[:3]),
                      q[3] * dq[3] - np.dot(q[:3], dq[:3])]
    pose[3:] /= np.linalg.norm(pose[3:])
    return pose


class Controls:
    """VisPy held-key input and optional SDL joystick; no Isaac Gym dependency."""
    def __init__(self, canvas, args):
        self.pressed = set()
        self.action = None
        self.args = args
        self.pygame = self.joystick = None
        canvas.events.key_press.connect(lambda event: canvas.close()
                                        if event.key is not None and event.key.name.upper() == 'ESCAPE' else None)
        canvas.events.key_press.connect(self.press)
        canvas.events.key_release.connect(self.release)
        canvas.events.close.connect(lambda event: self.pressed.clear())
        if args.joystick is not None:
            os.environ.setdefault('SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS', '1')
            try:
                import pygame
            except ImportError as error:
                raise ImportError('Joystick input requires pygame in the dyna environment') from error
            self.pygame = pygame
            pygame.display.init()
            pygame.joystick.init()
            try:
                if args.joystick >= pygame.joystick.get_count():
                    raise ValueError('Joystick index unavailable; check USB connection and --joystick')
                self.joystick = pygame.joystick.Joystick(args.joystick)
                self.joystick.init()
                if max(args.joystick_axes) >= self.joystick.get_numaxes():
                    raise ValueError('Joystick lacks mapped axes; configure --joystick-axes')
            except BaseException:
                self.close()
                raise

    def press(self, event):
        """Latch movement keys; queue scene/reset actions on the first press."""
        key = event.key.name.upper() if event.key is not None else ''
        if key not in self.pressed and key in ('N', 'B', 'BACKSPACE'):
            self.action = {'N': 'next', 'B': 'previous', 'BACKSPACE': 'reset'}[key]
        self.pressed.add(key)

    def release(self, event):
        """Stop a command when its key is released."""
        if event.key is not None:
            self.pressed.discard(event.key.name.upper())

    def command(self):
        """Poll joystick and merge axes with keyboard input, clamped to [-1, 1]."""
        result = keyboard_command(self.pressed)
        if self.joystick is not None:
            self.pygame.event.pump()
            if not self.joystick.get_init() or self.pygame.joystick.get_count() <= self.args.joystick:
                raise RuntimeError('Joystick disconnected; stopping realtime evaluation')
            values = np.array([self.joystick.get_axis(i) for i in self.args.joystick_axes])
            values *= self.args.joystick_signs
            deadzone = self.args.deadzone
            values = np.sign(values) * np.maximum(0., np.abs(values) - deadzone) / (1 - deadzone)
            result += values
        return np.clip(result, -1, 1)

    def close(self):
        """Release SDL resources without affecting the VisPy application."""
        if self.joystick is not None:
            self.joystick.quit()
        if self.pygame is not None:
            self.pygame.joystick.quit()
            self.pygame.display.quit()


def advance_scene(scene, timestamp_ns):
    """Visit every command/keyframe deadline before the next sensor sample."""
    while scene.next_command_ns() < timestamp_ns:
        scene.step(scene.next_command_ns())
    scene.step(timestamp_ns)


class RealtimeInputWindow:
    """Use the VisPy view for input, or a separate control pad for Rerun."""
    def __init__(self, window, renderer):
        self.closed = False
        self.separate = renderer == 'rerun'
        if self.separate:
            from vispy import app, scene
            self.app = app
            self.canvas = scene.SceneCanvas(title='Neural Scene controls', size=(620, 160),
                                            show=True, bgcolor='#20242a')
            self.label = scene.visuals.Text(
                'Click here for keyboard control\nW/S A/D: move | Q/E: yaw | H/G: height\n'
                'N/B: scene | Backspace: reset | Esc: stop',
                parent=self.canvas.scene, pos=(310, 70), color='white', font_size=12)
        else:
            self.app, self.canvas = window.app, window.canvas
        self.canvas.events.close.connect(self._close)

    def _close(self, event):
        self.closed = True

    def wait_until(self, deadline, window):
        while not self.closed and not window.closed:
            self.app.process_events()
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                return True
            time.sleep(min(.005, remaining))
        return False

    def close(self):
        if self.separate:
            self.canvas.close()


class LiveScene:
    """One environment with direct safe ego commands and fresh sensor queries."""
    def __init__(self, path, args):
        from dynaraysim.main.config import resolve
        from dynaraysim.main.scene import SceneBatch
        from dynaraysim.sensors.warp_backend import WarpBackend
        config = resolve(path, args.sim_device or args.device)
        data = config.data
        if data['run']['environments'] != 1:
            raise ValueError('Realtime control requires one environment per scenario')
        if data['ego'].get('per_environment'):
            raise ValueError('Use a single ego configuration for realtime control')
        # Retain the resolved initial trajectory pose, then take over motion.
        data['ego']['mode'] = 'stationary'
        self.config = replace(config, json=json.dumps(data))
        self.scene = SceneBatch(self.config)
        sensor_index = next((i for i, s in enumerate(data['sensors']) if args.sensor is None or s['id'] == args.sensor), None)
        if sensor_index is None:
            raise ValueError('Requested sensor is absent from scenario')
        self.spec = data['sensors'][sensor_index]
        self.pattern = config.patterns[sensor_index]
        self.backend = WarpBackend(self.scene.environments, data['run']['device'], 1,
                                   use_cuda_graph=data['run']['cuda_graph'], pipeline_depth=1)
        self.dt = 1 / self.spec['frequency_hz']
        self.frame = 0
        self.blocked = False

    def acquire(self, command, args):
        """Advance accepted scene motion and return robot-frame valid XYZ and pose."""
        import torch
        from dynaraysim.utils.math3d import compose, transform
        scene = self.scene
        # Step moving scene objects through the simulator's own safety checks.
        advance_scene(scene, round(self.frame * self.dt * 1e9))
        if self.frame:
            proposed = integrate_pose(scene.egos[0].detach().cpu().numpy(), command,
                                      self.dt, args.speed, np.deg2rad(args.angular_speed))
            candidate = torch.as_tensor(proposed, dtype=scene.egos[0].dtype, device=scene.device)
            # _safe is the same terrain/swept-collision guard used by SceneBatch.step.
            self.blocked = not scene._safe(0, [obj.pose for obj in scene.environments[0]], candidate)
            if not self.blocked:
                scene.egos[0] = candidate
        self.backend.update_scene()
        extrinsic = torch.as_tensor(self.spec['extrinsic'], dtype=scene.egos[0].dtype, device=scene.device)
        sensor_pose = compose(scene.egos[0], extrinsic)
        cursor = self.frame * self.pattern.samples if self.pattern.kind == 'livox' else 0
        ranges, status, _ = self.backend.cast(self.pattern, [sensor_pose], [0], [cursor], self.spec['range_m'])
        indices = self.pattern.indices(cursor)
        ranges = ranges[0].to(self.pattern.origins.device)
        valid = (status[0] == 1).to(ranges.device)
        sensor_points = self.pattern.origins[indices] + ranges[:, None] * self.pattern.directions[indices]
        points = transform(extrinsic.to(dtype=sensor_points.dtype), sensor_points[valid])
        result = (points.detach().cpu().numpy(), scene.egos[0].detach().cpu().numpy(),
                  round(self.frame * self.dt * 1e9), self.frame)
        self.frame += 1
        return result


def run(args):
    """Run sequential fresh acquisition/inference with a responsive comparison window."""
    if args.viewer == 'rerun':
        # Keep the camera-capable SDK local to realtime inference; saved-result
        # playback can continue using the environment's installed SDK.
        deps = Path(__file__).resolve().parents[1] / '.deps-rerun'
        if deps.is_dir():
            sys.path[:0] = [str(deps / 'rerun_sdk'), str(deps)]
    import torch
    from .training import load_checkpoint
    from .mapping import NeuralSceneMapper, PointCloudFrame
    from .voxel import VoxelGrid
    from .evaluation_viewer import ComparisonViewer
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise ValueError('CUDA unavailable; select --device cpu explicitly')
    model = load_checkpoint(args.checkpoint, args.device).eval()
    grid = VoxelGrid(args.voxel_size, tuple(args.grid_shape), tuple(args.grid_origin))
    window = ComparisonViewer(args.viewer, ['prediction', 'observation'],
        dict(prediction=args.prediction_opacity, observation=args.observation_opacity, target=1., mesh=args.mesh_opacity),
        follow_ego=True,
        title='Neural Scene realtime', max_points=args.max_points, point_size=args.point_size, distance=args.distance)
    controls = input_window = None
    try:
        input_window = RealtimeInputWindow(window, args.viewer)
        controls = Controls(input_window.canvas, args)
        index, live, mapper = 0, None, None
        count = 0
        inference_timing = InferenceTiming(args.device)
        while not window.closed and not input_window.closed and (args.max_frames is None or count < args.max_frames):
            started = time.perf_counter()
            input_window.app.process_events()
            if window.closed or input_window.closed:
                break
            if live is None or controls.action:
                action = controls.action
                if action in ('next', 'previous'):
                    index = (index + (1 if action == 'next' else -1)) % len(args.scenarios)
                controls.action = None
                controls.pressed.clear()
                live = None
                mapper = None
                live = LiveScene(args.scenarios[index], args)
                mapper = NeuralSceneMapper(model, 1 / live.dt, grid)
                window.set_mesh(None)
                # Compose every object's mesh in world space, including moving objects below.
            points, pose, timestamp, frame = live.acquire(controls.command(), args)
            if args.no_feedback:
                mapper.reset()
            input_frame = PointCloudFrame(points, pose, timestamp, frame)
            with inference_timing.measure():
                step = mapper.update(input_frame)
            vertices, faces, offset = [], [], 0
            for obj in live.scene.environments[0]:
                local = obj.asset.vertices.detach().cpu().numpy()
                matrix = pose_matrix(obj.pose.detach().cpu().numpy())
                vertices.append(local @ matrix[:3, :3].T + matrix[:3, 3])
                triangles = obj.asset.faces.detach().cpu().numpy()
                faces.append(triangles + offset)
                offset += len(local)
            if vertices:
                window.set_mesh((np.concatenate(vertices).astype(np.float32), np.concatenate(faces).astype(np.uint32)))
            data = dict(prediction_points=step.points.detach().cpu().numpy(), current_points=points,
                        robot_pose=pose_matrix(pose), timestamp_ns=timestamp, frame_id=frame)
            snapshot = comparison_snapshot(data, 'prediction', 'observation', asdict(grid), args.voxels)
            window.update(snapshot)
            elapsed = time.perf_counter() - started
            input_window.canvas.title = ('{} | frame {} | infer {:.1f} ms | avg {:.1f} ms | total {:.2f} s | loop {:.1f} ms | {} | N/B scene, Backspace reset'.format(
                args.scenarios[index].stem, frame, inference_timing.last_s * 1000,
                inference_timing.summary()['mean_inference_ms'], inference_timing.total_s,
                elapsed * 1000, 'motion blocked' if live.blocked else 'ego control'))
            count += 1
            if not input_window.wait_until(started + live.dt, window):
                break
    finally:
        if controls is not None:
            controls.close()
        if input_window is not None:
            input_window.close()
        window.close()


def main():
    """Parse realtime control, scene and checkpoint settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--scenarios', type=Path, nargs='+', required=True, help='One or more DynaRaySim scenario YAML files')
    parser.add_argument('--device', default='cuda:0', help='Model device')
    parser.add_argument('--sim-device', help='Raycast device (defaults to model device)')
    parser.add_argument('--viewer', choices=('vispy', 'rerun'), default='vispy',
                        help='3D viewer; Rerun uses a separate keyboard control window')
    parser.add_argument('--sensor', help='Sensor ID; default first sensor')
    parser.add_argument('--speed', type=positive_float, default=.5, help='Translation speed m/s')
    parser.add_argument('--angular-speed', type=positive_float, default=30., help='Rotation speed degrees/s')
    parser.add_argument('--joystick', type=int, help='SDL joystick index; requires pygame')
    parser.add_argument('--joystick-axes', type=int, nargs=6, default=[1, 0, 3, 4, 5, 2], metavar='AXIS')
    parser.add_argument('--joystick-signs', type=float, nargs=6, default=[-1, -1, -1, 1, 1, -1], metavar='SIGN')
    parser.add_argument('--deadzone', type=opacity, default=.15)
    parser.add_argument('--voxel-size', type=positive_float, default=.05)
    parser.add_argument('--grid-shape', type=positive_int, nargs=3, default=[64, 64, 64])
    parser.add_argument('--grid-origin', type=float, nargs=3, default=[-1.6, -1.6, -1.6])
    parser.add_argument('--no-feedback', action='store_true')
    parser.add_argument('--voxels', action='store_true')
    parser.add_argument('--prediction-opacity', type=opacity, default=.9)
    parser.add_argument('--observation-opacity', type=opacity, default=.7)
    parser.add_argument('--mesh-opacity', type=opacity, default=.25)
    parser.add_argument('--max-points', type=positive_int, default=200000)
    parser.add_argument('--point-size', type=positive_float, default=3.)
    parser.add_argument('--distance', type=positive_float, default=6.)
    parser.add_argument('--max-frames', type=positive_int, help='Optional finite run limit')
    args = parser.parse_args()
    if args.deadzone >= 1 or min(args.joystick_axes) < 0 or (args.joystick is not None and args.joystick < 0):
        parser.error('deadzone must be <1 and joystick indices must be nonnegative')
    if not np.isfinite(args.grid_origin).all() or not np.isin(args.joystick_signs, [-1, 1]).all():
        parser.error('grid origin must be finite and joystick signs must be +1 or -1')
    run(args)


if __name__ == '__main__':
    main()
