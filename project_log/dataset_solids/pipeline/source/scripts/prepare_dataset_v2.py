"""Collect current GT and robot-frame DynaMapping sequences within a 5 GB cap.

Run with a Python environment containing the simulator dependencies. Source trees are
read-only. Each scene runs in a separate process to release GT caches.
"""

import argparse
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SIM = Path(os.environ.get('DYNARAYSIM_ROOT', ROOT.parent / 'DynaRaySim'))
MAPPING = Path(os.environ.get('DYNAMAPPING_ROOT', ROOT.parent / 'DynaMapping'))
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['DYNARAYSIM_WARP_CACHE'] = str(ROOT / '.build/warp-cache-v2')
sys.dont_write_bytecode = True
sys.path[:0] = [str(MAPPING), str(SIM)]

import numpy as np
import torch
from dynamapping import DynaMapping, PointCloudFrame, inverse, transform_points
from dynamapping.readers import DynaRaySimReader
from dynaraysim.data.dataset import Dataset
from dynaraysim.data.ground_truth import load_config as load_gt_config
from dynaraysim.main.config import resolve
from dynaraysim.main.runner import run
from prepare_training_data import crop, centroid_cloud, atomic_json, SCENES

RESERVE = 256 * 1024 ** 2


def directory_bytes(path):
    """Count logical and allocated bytes conservatively, including hidden files."""
    total = 0
    for item in path.rglob('*'):
        try:
            if not item.is_file():
                continue
            stat = item.stat()
            total += max(stat.st_size, stat.st_blocks * 512)
        except FileNotFoundError:
            # Archive writers atomically rename temporary shards.
            continue
    return total


def source_hashes():
    """Fingerprint the actual simulator and mapping Python implementations."""
    result = {}
    for root, folder in ((SIM, 'dynaraysim'), (MAPPING, 'dynamapping')):
        for path in sorted((root / folder).rglob('*.py')):
            result[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def robot_frame(frame):
    """Convert current DynaMapping sensor points into the recorded ego frame."""
    extrinsic = inverse(frame.robot_pose) @ frame.sensor_pose
    points = transform_points(extrinsic, frame.points).to(torch.float32)
    return PointCloudFrame(points, frame.robot_pose, frame.timestamp_ns,
                           frame_id=frame.frame_id, device='cpu')


def checked_write(path, payload, root, limit):
    """Write atomically only if the payload fits the data budget with reserve."""
    if directory_bytes(root) + len(payload) + RESERVE >= limit:
        raise RuntimeError('Dataset size budget exhausted before writing ' + str(path))
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_bytes(payload)
    temporary.replace(path)


def prepare_scene(args):
    """Collect a fresh full scenario and serialize aligned robot-frame training data."""
    scene = args.scene
    split = dict(SCENES)[scene]
    config = SIM / 'examples/scenarios' / (scene + '.yaml')
    gt_config = SIM / 'examples/gt' / (scene + '.yaml')
    archive = args.output / 'archives' / scene
    destination = args.output / 'sequences' / scene
    destination.mkdir(parents=True, exist_ok=True)
    completed = destination / 'sequence.json'
    if completed.exists():
        record = json.loads(completed.read_text())
        for frame in record['frames']:
            path = destination / frame['file']
            if hashlib.sha256(path.read_bytes()).hexdigest() != frame['sha256']:
                raise ValueError('Completed frame hash mismatch: ' + str(path))
        return
    settings = load_gt_config(gt_config, 'composed')
    audit_status = 'matching_auto_report'
    if not (archive / 'manifest.json').exists():
        if archive.exists():
            raise RuntimeError('Incomplete archive needs inspection: ' + str(archive))
        resolved = resolve(config, args.device)
        packets = [0]

        def monitor(packet):
            """Check storage and expose collection progress before appending packets."""
            packets[0] += 1
            if packets[0] % 30 == 0:
                size = directory_bytes(args.output)
                print('{} packets={} data={:.3f} GB'.format(scene, packets[0], size / 1e9), flush=True)
                if size + RESERVE >= args.max_bytes:
                    raise RuntimeError('Dataset size guard reached')

        print('Collecting full {} using current composed GT'.format(scene), flush=True)
        try:
            result = run(resolved, str(archive), ground_truth=settings, on_packet=monitor)
        except ValueError as error:
            if 'GT validation report is unverified or does not match this scenario' not in str(error) or archive.exists():
                raise
            # Tensor migration changed ray-pattern hashes. Do not pretend that
            # an old NumPy fingerprint validates the current geometry pipeline.
            audit_status = 'numeric_0.05m_old_report_fingerprint_mismatch'
            settings['boundary_voxel_size_m'] = 0.05
            settings['validation_report'] = None
            print('Old GT audit fingerprint differs; explicit 5 cm resolution, no current audit claim', flush=True)
            result = run(resolved, str(archive), ground_truth=settings, on_packet=monitor)
        if result['termination'] != 'completed':
            raise RuntimeError('Collection did not complete: ' + str(result['termination']))
        atomic_json(destination / 'collection.json', {'audit_status': audit_status})
    audit_status = json.loads((destination / 'collection.json').read_text())['audit_status']
    dataset = Dataset(archive, verify=True)
    reader = DynaRaySimReader(archive, device='cpu')
    mapper = DynaMapping(reader.frequency_hz, history_frames=1, device='cpu')
    origin = np.array([-1.6, -1.6, -1.6], dtype=np.float32)
    shape = np.array([64, 64, 64], dtype=np.int32)
    frames = []
    for index in range(len(reader)):
        raw = reader.load_frame(index)
        alignment = mapper.update(robot_frame(raw))
        gt = dataset.read_gt_window(0, 'lidar', index, 1, mode='composed', frame='world', radius_m=4.0)[0]
        if int(gt['timestamp_ns']) != raw.timestamp_ns or int(gt['frame_id']) != raw.frame_id:
            raise ValueError('GT and observation identity differs')
        current = crop(alignment.current.points.numpy(), origin, shape, 0.05)
        target = transform_points(inverse(raw.robot_pose), gt['points']).numpy()
        target = centroid_cloud(crop(target, origin, shape, 0.05), origin, 0.05)
        path = destination / 'frame_{:05d}.npz'.format(index)
        # A partial preprocessing restart replays actual alignment in order.
        payload = io.BytesIO()
        np.savez_compressed(payload, current_points=current, target_points=target,
            robot_pose=raw.robot_pose.numpy(), transformation=alignment.transformation.numpy(),
            sensor_pose=raw.sensor_pose.numpy(), timestamp_ns=np.int64(raw.timestamp_ns),
            frame_id=np.int64(raw.frame_id), origin=origin, voxel_size_m=np.float32(0.05),
            point_frame='current_robot', input_policy='aligned_merge_t0')
        checked_write(path, payload.getvalue(), args.output, args.max_bytes)
        frames.append({'file': path.name, 'current_points': len(current), 'target_voxels': len(target),
                       'sha256': hashlib.sha256(payload.getvalue()).hexdigest()})
        if index % 30 == 0:
            print('{} frame {}/{} observations={} targets={}'.format(scene, index, len(reader), len(current), len(target)), flush=True)
    if len(frames) < 12 or not any(f['target_voxels'] for f in frames):
        raise ValueError('Insufficient frames or empty GT')
    record = {'scene': scene, 'split': split, 'archive': str(archive), 'frames': frames,
        'frequency_hz': reader.frequency_hz, 'collection_device': args.device,
        'config_sha256': hashlib.sha256(config.read_bytes()).hexdigest(),
        'ground_truth': dataset.manifest['ground_truth'], 'gt_audit_status': audit_status,
        'preprocessing': {'frames': 'full_sequence', 'origin': origin.tolist(),
            'mapping_source': str(MAPPING), 'mapping_input': 'sensor transformed to robot before DynaMapping.update',
            'gt_mode': 'composed', 'roi_radius_m': 4.0}}
    atomic_json(completed, record)


def main():
    """Run isolated scene workers under a size monitor, then validate all frames."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/training_dataset_v2')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--max-bytes', type=int, default=5_000_000_000)
    parser.add_argument('--scene', choices=[item[0] for item in SCENES], help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not RESERVE < args.max_bytes <= 5_000_000_000:
        parser.error('--max-bytes must be >256 MiB and <=5,000,000,000')
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    if args.scene:
        # Permit independent collection workers while preventing duplicate
        # writes when the coordinator reaches an already-running scene.
        with (args.output / (args.scene + '.lock')).open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            prepare_scene(args)
        return
    provenance = {'source_sha256': source_hashes(), 'mapping_source': str(MAPPING),
                  'simulator_source': str(SIM), 'max_dataset_bytes': args.max_bytes,
                  'gt_mode': 'composed', 'device': args.device}
    provenance_path = args.output / 'provenance.json'
    if provenance_path.exists() and json.loads(provenance_path.read_text()) != provenance:
        raise ValueError('Sources or collection settings changed; use a new output directory')
    atomic_json(provenance_path, provenance)
    sequences = []
    for scene, split in SCENES:
        command = [sys.executable, str(Path(__file__).resolve()), '--output', str(args.output),
                   '--device', args.device, '--max-bytes', str(args.max_bytes), '--scene', scene]
        child = subprocess.Popen(command)
        try:
            while child.poll() is None:
                if directory_bytes(args.output) + RESERVE >= args.max_bytes:
                    raise RuntimeError('Stopping before the 5 GB data limit')
                time.sleep(0.5)
            if child.returncode:
                raise RuntimeError('{} worker failed with {}'.format(scene, child.returncode))
        finally:
            if child.poll() is None:
                child.terminate()
                child.wait()
        sequences.append(json.loads((args.output / 'sequences' / scene / 'sequence.json').read_text()))
        atomic_json(args.output / 'preparation_progress.json', {'completed_sequences': sequences})
    manifest = {'format': 'neural-scene-dynamapping-v1', 'input_policy': 'aligned_merge_t0',
        'grid': {'voxel_size': 0.05, 'shape': [64, 64, 64], 'origin': [-1.6, -1.6, -1.6]},
        'split_policy': 'whole scenario, no adjacent-frame leakage',
        'gt_mode': 'composed', 'max_dataset_bytes': args.max_bytes, 'sequences': sequences}
    atomic_json(args.output / 'dataset.json', manifest)
    from validate_training_data import validate
    report = validate(args.output)
    report['dataset_bytes'] = directory_bytes(args.output)
    report['max_dataset_bytes'] = args.max_bytes
    if report['dataset_bytes'] >= args.max_bytes:
        raise RuntimeError('Dataset exceeds storage limit')
    atomic_json(args.output / 'validation_report.json', report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
