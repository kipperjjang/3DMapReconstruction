"""Collect upright randomized scenes with DynaRaySim/torch under conda dyna.

Static scenes share one current-simulator boundary GT cloud. Every measurement
is freshly ray-cast by runner.run; actual DynaMapping computes feedback poses.
No archived v1/v2 frame, augmentation or replay counts toward the 200,000 target.
"""

import argparse
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import h5py
import numpy as np
import torch
import yaml
from scipy.spatial import cKDTree

from dynamapping import DynaMapping, PointCloudFrame, pose_matrix, inverse, transform_points
from dynaraysim.main.config import resolve
from dynaraysim.main.scene import SceneBatch
from dynaraysim.main.runner import run, state_record
from dynaraysim.data.ground_truth import GroundTruth, load_config
from neural_scene.packed import ORIGIN, SIZE, crop, keys, centroids, encode

ROOT = Path(__file__).resolve().parents[1]
SIM = Path('/home/intern/DynaRaySim/torch')
MAP = Path('/home/intern/DynaMapping/torch')
RESERVE = 512 * 1024**2


def atomic_json(path, data):
    """Commit metadata using atomic replacement."""
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)


def disk_bytes(root):
    """Count allocated/logical data bytes, tolerating concurrent atomic renames."""
    result = 0
    for path in root.rglob('*'):
        try:
            if path.is_file():
                stat = path.stat()
                result += max(stat.st_size, stat.st_blocks * 512)
        except FileNotFoundError:
            pass
    return result


def box_mesh(lower, upper):
    """Return outward triangles for a solid world-aligned rectangular box."""
    x, y, z = lower
    X, Y, Z = upper
    vertices = [[x,y,z],[X,y,z],[X,Y,z],[x,Y,z],[x,y,Z],[X,y,Z],[X,Y,Z],[x,Y,Z]]
    faces = [[0,2,1],[0,3,2],[4,5,6],[4,6,7],[0,1,5],[0,5,4],
             [1,2,6],[1,6,5],[2,3,7],[2,7,6],[3,0,4],[3,4,7]]
    return np.array(vertices), np.array(faces)


def make_scene(folder, scene_id, frames, channels, azimuth_step):
    """Create unique geometry and a continuous yaw-only collision-free corridor route."""
    rng = np.random.default_rng(20261028 + scene_id)
    meshes = [box_mesh([-5,-4,-2.5], [5,4,0])]
    corridor_half = float(rng.uniform(.9, 1.5))
    for side in (-1, 1):
        for x in np.linspace(-3.3, 3.3, 5):
            depth, width, height = rng.uniform(.4, 1.0), rng.uniform(.35, .9), rng.uniform(.2, 1.5)
            center = np.array([x + rng.uniform(-.12,.12), side*(corridor_half+width/2), height/2])
            half = np.array([depth, width, height])/2
            meshes.append(box_mesh(center-half, center+half))
        # Stair-like adjacent blocks at one end, outside the clear trajectory.
        for step in range(4):
            low = np.array([-1.0 + step*.28, side*2.5 - .45, 0.])
            meshes.append(box_mesh(low, low + [.28, .9, (step+1)*rng.uniform(.08,.2)]))
        wall_height = rng.uniform(.6, 1.8)
        meshes.append(box_mesh([-4,side*3-.08,0], [4,side*3+.08,wall_height]))
    vertices, faces, count = [], [], 0
    for v, f in meshes:
        vertices.extend(v.tolist()); faces.extend((f+count+1).tolist()); count += len(v)
    mesh = folder / 'scene.obj'
    with mesh.open('w') as stream:
        for row in vertices:
            stream.write('v {} {} {}\n'.format(*row))
        for row in faces:
            stream.write('f {} {} {}\n'.format(*row))
    duration = frames / 10
    height = float(rng.uniform(.55, .85))
    # Each leg has different endpoints and yaw; no periodic repeated poses.
    stamps = np.linspace(0, duration, max(2, int(np.ceil(duration/8))+1))
    trajectory = []
    yaw = float(rng.uniform(-180,180))
    for index, stamp in enumerate(stamps):
        yaw += float(rng.uniform(-80,80))
        trajectory.append({'time_s': float(stamp), 'pose': {'translation': [
            float((-1 if index%2 == 0 else 1)*rng.uniform(2.8,3.4)),
            float(rng.uniform(-.35,.35)), height], 'rpy_deg': [0,0,yaw]}})
    config = {'run': {'seed':20261028+scene_id, 'duration_s':duration, 'dt_s':.1,
                     'environments':1, 'device':'cpu'},
        'assets': {'terrain': {'path':str(mesh.resolve()), 'units':'m', 'up_axis':'Z'}},
        'objects': [{'id':0,'asset':'terrain','static':True}],
        'sensors': [{'id':'lidar','frequency_hz':10,'range_m':[.05,20.],
            'extrinsic':{'translation':[0,0,0],'rpy_deg':[0,0,0]},
            'pattern': {'kind':'lidar','channels':channels,'vertical_fov_range':[-85,45],
                        'horizontal_fov_range':[-180,180],'horizontal_res':azimuth_step}}],
        'ego': {'mode':'scripted','trajectory':trajectory,'terrain_alignment':False,
                'radius':.15,'clearance':.05,'sensor_height':height,'max_slope_deg':60},
        'domain_randomization': {'enabled':False},
        'output': {'directory':str(folder/'unused-raw-archive'), 'shard_bytes':67108864,
                   'chunk_frames':8,'queue_size':4,'compression':'lzf','store_xyz':True}}
    path = folder / 'scenario.yaml'
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return path, config


def shared_gt(config, authored, folder):
    """Generate current DynaRaySim boundary GT once for an unchanged static scene."""
    scene = SceneBatch(config)
    settings = load_config(mode='composed')
    gt = GroundTruth(scene, settings)
    pose = scene.egos[0].clone()
    packet = {'states':[state_record(scene,0)], 'scans':[{'env':0,'episode':0,
              'sensor':'lidar','timestamp_ns':0,'sensor_pose_world':pose}]}
    gt.process(packet)
    points = packet['scans'][0]['_gt_background']['points'].numpy()
    full_count = len(points)
    # Retain all samples that any training cube can reach; no visible-only GT.
    route = np.array([item['pose']['translation'] for item in authored['ego']['trajectory']])
    lower, upper = route.min(0)-2.8, route.max(0)+2.8
    points = points[np.all((points >= lower) & (points <= upper), axis=1)].copy()
    np.savez_compressed(folder/'surface_gt.npz', points=points)
    metadata = {'settings':gt.settings,'full_scene_surface_samples':full_count,
                'retained_surface_samples':len(points),'roi_union_lower':lower.tolist(),
                'roi_union_upper':upper.tolist(),'static_scene':True}
    del scene, gt, packet
    gc.collect()
    return points, metadata


def collect_scene(args, scene_id):
    """Freshly ray-cast one sequence, measure coverage and write its packed shard."""
    name = 'scene_{:04d}'.format(scene_id)
    folder = args.output/'sequences'/name
    folder.mkdir(parents=True, exist_ok=True)
    result_path = folder/'sequence.json'
    if result_path.exists():
        result = json.loads(result_path.read_text())
        if result['collection_settings'] != {'frames':args.frames_per_scene,'channels':args.channels,'azimuth_step':args.azimuth_step}:
            raise ValueError('Completed scene settings differ')
        return result
    if (folder/'frames.h5.partial').exists():
        raise RuntimeError('Partial shard retained for inspection: '+str(folder))
    path, authored = make_scene(folder, scene_id, args.frames_per_scene, args.channels, args.azimuth_step)
    resolved = resolve(path, args.device)
    begun = time.monotonic()
    print(name+' generating current surface boundary', flush=True)
    surface, gt_meta = shared_gt(resolved, authored, folder)
    tree = cKDTree(surface)
    atomic_json(folder/'gt_metadata.json',gt_meta)
    print('{} GT ready in {:.1f}s; {} retained samples'.format(name,time.monotonic()-begun,len(surface)),flush=True)
    mapper = DynaMapping(10, history_frames=1, device='cpu')
    partial = folder/'frames.h5.partial'
    records, pose_hashes = [], set()
    previous_pose = None
    maximum_transform_error = maximum_up_error = 0.
    with h5py.File(partial, 'w') as file:
        file.attrs.update(format='neural-scene-packed-v1', point_frame='current_robot',
                          voxel_size_m=SIZE, origin=ORIGIN, input_policy='aligned_merge_t0')
        for kind in ('current','target'):
            file.create_dataset(kind, shape=(6,0), maxshape=(6,None), dtype='u1',
                                chunks=(6,8192), compression='gzip', compression_opts=3, shuffle=True)
            offsets = file.create_dataset(kind+'_offsets', (args.frames_per_scene+1,), dtype='i8')
            offsets[0] = 0
        for field in ('robot_pose','sensor_pose','transformation'):
            file.create_dataset(field, (args.frames_per_scene,4,4), dtype='f8', compression='gzip')
        file.create_dataset('timestamp_ns', (args.frames_per_scene,), dtype='i8')

        def consume(packet):
            """Convert each acquired packet, validate alignment, and commit compact points."""
            nonlocal previous_pose, maximum_transform_error, maximum_up_error
            for scan in packet['scans']:
                row = len(records)
                state = next(s for s in packet['states'] if s['env']==scan['env'] and s['episode']==scan['episode'])
                robot = pose_matrix(state['ego'], device='cpu')
                sensor = pose_matrix(scan['sensor_pose_world'], device='cpu')
                up_error = float(torch.max(torch.abs(sensor[:3,2]-torch.tensor([0.,0.,1.]))))
                maximum_up_error = max(maximum_up_error, up_error)
                if up_error > 1e-9:
                    raise ValueError('Recorded lidar Z differs from world Z')
                signature = hashlib.sha256(robot.numpy().tobytes()).hexdigest()
                if signature in pose_hashes:
                    raise ValueError('Repeated acquired pose; refusing to count it as a unique frame')
                pose_hashes.add(signature)
                valid = scan['status']==1
                points = scan['points_sensor'][valid]
                current_robot = transform_points(inverse(robot)@sensor, points).float()
                stamp = int(scan['timestamp_ns'])
                if stamp != row*100000000:
                    raise ValueError('Missing or duplicate acquisition time')
                alignment = mapper.update(PointCloudFrame(current_robot, robot, stamp, frame_id=row, device='cpu'))
                pose = robot.numpy()
                expected = np.eye(4) if previous_pose is None else np.linalg.inv(pose)@previous_pose
                transform_error = float(np.abs(expected-alignment.transformation.numpy()).max())
                maximum_transform_error = max(maximum_transform_error,transform_error)
                if transform_error > 1e-8:
                    raise ValueError('DynaMapping feedback transform mismatch')
                previous_pose = pose
                current = crop(current_robot.numpy())
                nearby = surface[tree.query_ball_point(pose[:3,3], 2.78)]
                local_surface = crop((nearby.astype(np.float64)-pose[:3,3])@pose[:3,:3])
                target = centroids(local_surface)
                if not len(current) or not len(target):
                    raise ValueError('Empty local measurement or GT; inspect pilot geometry')
                measurement_keys = np.unique(keys(current))
                target_keys = keys(target)
                voxel_recall = len(np.intersect1d(measurement_keys,target_keys))/len(target_keys)
                # GT is the denominator, evaluated on dense unquantized surface samples.
                distances = cKDTree(current).query(local_surface, distance_upper_bound=.05)[0]
                coverage = float(np.mean(np.isfinite(distances)))
                for kind, cloud in (('current',current),('target',target)):
                    packed = encode(cloud)
                    begin = int(file[kind+'_offsets'][row]); end = begin+len(packed)
                    file[kind].resize((6,end)); file[kind][:,begin:end] = packed.T
                    file[kind+'_offsets'][row+1] = end
                for field,value in (('robot_pose',pose),('sensor_pose',sensor.numpy()),
                                     ('transformation',alignment.transformation.numpy()),('timestamp_ns',stamp)):
                    file[field][row] = value
                records.append({'index':row,'current_points':len(current),'target_voxels':len(target),
                    'surface_gt_points':len(local_surface),'gt_point_coverage_5cm':coverage,'gt_voxel_recall':voxel_recall})
                if row%100 == 0:
                    file.flush()
                    size = disk_bytes(args.output)
                    print('{} {}/{} coverage={:.3f} voxel_recall={:.3f} data={:.3f}GB'.format(
                        name,row,args.frames_per_scene,coverage,voxel_recall,size/1e9), flush=True)
                    if size+RESERVE > args.max_bytes:
                        raise RuntimeError('Data cap reserve reached; retain partial shard')
        metrics = run(resolved, record=False, on_packet=consume)
        if len(records)!=args.frames_per_scene or metrics['termination']!='completed':
            raise RuntimeError('Incomplete sequence')
        if any(metrics.get('joint_motion_rejections', [])):
            raise ValueError('Rejected motion; inspect acquired route')
    final = folder/'frames.h5'
    partial.replace(final)
    split = 'validation' if scene_id%10==0 else 'test' if scene_id%10==1 else 'train'
    result = {'scene':name,'split':split,'frames':records,'shard':str(final.relative_to(args.output)),
        'frequency_hz':10.,'unique_pose_count':len(pose_hashes),'max_sensor_up_error':maximum_up_error,
        'max_feedback_transform_error':maximum_transform_error,
        'collection_settings':{'frames':args.frames_per_scene,'channels':args.channels,'azimuth_step':args.azimuth_step},
        'config_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'geometry_sha256':hashlib.sha256((folder/'scene.obj').read_bytes()).hexdigest(),
        'shard_sha256':hashlib.sha256(final.read_bytes()).hexdigest(),
        'ground_truth':gt_meta,'mean_gt_point_coverage_5cm':float(np.mean([r['gt_point_coverage_5cm'] for r in records])),
        'mean_gt_voxel_recall':float(np.mean([r['gt_voxel_recall'] for r in records])),
        'elapsed_s':time.monotonic()-begun,'simulator_metrics':metrics}
    atomic_json(result_path,result)
    print('{} completed {} frames in {:.1f}s, coverage {:.3f}'.format(name,len(records),result['elapsed_s'],result['mean_gt_point_coverage_5cm']),flush=True)
    return result


def provenance(args):
    """Record active package source hashes and acquisition settings."""
    import dynaraysim, dynamapping, scipy
    if Path(dynaraysim.__file__).resolve().parent != SIM/'dynaraysim' or Path(dynamapping.__file__).resolve().parent != MAP/'dynamapping':
        raise ValueError('Expected DynaRaySim/torch and DynaMapping/torch')
    hashes = {}
    for folder in (SIM/'dynaraysim',MAP/'dynamapping'):
        for path in sorted(folder.rglob('*.py')):
            hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {'sources':hashes,'python':sys.version,'python_executable':sys.executable,
        'torch':torch.__version__,'numpy':np.__version__,'scipy':scipy.__version__,
        'settings':{k:v for k,v in vars(args).items() if k not in ('scene','output','jobs')},
        'storage':'uint8 cell XYZ + uint8 subvoxel XYZ, lossless gzip on quantized records',
        'quantization_max_axis_error_m':SIZE/512+1e-6,'quantization_max_3d_error_m':np.sqrt(3)*SIZE/512+2e-6,
        'sensor':'explicit synthetic spinning lidar, 360 degree azimuth, -85 to +45 degree elevation; zero roll/pitch mounting',
        'paper_reference':'local 2206.08077v1.pdf; 43% is a comparison reference, 5cm nearest-neighbour is our explicit proxy'}


def main():
    """Coordinate independent scene workers under the global 10 GB size cap."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/training_dataset_v3')
    parser.add_argument('--scenes',type=int,default=40)
    parser.add_argument('--frames-per-scene',type=int,default=5000)
    parser.add_argument('--start-scene',type=int,default=0)
    parser.add_argument('--channels',type=int,default=32)
    parser.add_argument('--azimuth-step',type=float,default=3.)
    parser.add_argument('--jobs',type=int,default=4)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--max-bytes',type=int,default=10_000_000_000)
    parser.add_argument('--scene',type=int,help=argparse.SUPPRESS)
    args = parser.parse_args()
    if min(args.scenes,args.frames_per_scene,args.jobs,args.channels)<1 or args.max_bytes>10_000_000_000:
        parser.error('Positive counts and a <=10 GB cap required')
    args.output = args.output.resolve(); args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    if args.scene is not None:
        with (args.output/'scene_{:04d}.lock'.format(args.scene)).open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            collect_scene(args,args.scene)
        return
    record = provenance(args)
    path = args.output/'provenance.json'
    if path.exists() and json.loads(path.read_text())!=record:
        raise ValueError('Acquisition provenance changed; use a fresh directory')
    atomic_json(path,record)
    pending = list(range(args.start_scene,args.start_scene+args.scenes))
    active = {}
    try:
        while pending or active:
            while pending and len(active)<args.jobs:
                scene = pending.pop(0)
                command = [sys.executable,str(Path(__file__).resolve()),'--output',str(args.output),
                    '--scene',str(scene),'--frames-per-scene',str(args.frames_per_scene),
                    '--channels',str(args.channels),'--azimuth-step',str(args.azimuth_step),
                    '--device',args.device,'--max-bytes',str(args.max_bytes)]
                active[scene] = subprocess.Popen(command)
            for scene,child in list(active.items()):
                if child.poll() is not None:
                    if child.returncode:
                        raise RuntimeError('Scene {} failed ({})'.format(scene,child.returncode))
                    del active[scene]
            if disk_bytes(args.output)+RESERVE>args.max_bytes:
                raise RuntimeError('Global storage cap reserve reached')
            time.sleep(1)
    finally:
        for child in active.values():
            if child.poll() is None:
                child.terminate()
            child.wait()
    sequences = [json.loads((args.output/'sequences'/'scene_{:04d}'.format(index)/'sequence.json').read_text())
                 for index in range(args.start_scene,args.start_scene+args.scenes)]
    manifest = {'format':'neural-scene-packed-v1','input_policy':'aligned_merge_t0',
        'grid':{'voxel_size':SIZE,'shape':[64,64,64],'origin':[-1.6,-1.6,-1.6]},
        'split_policy':'whole independently randomized scene, 80/10/10 by scene ID',
        'max_dataset_bytes':args.max_bytes,'sequences':sequences}
    atomic_json(args.output/'dataset.json',manifest)
    summary = {'status':'collected','total_frames':sum(len(s['frames']) for s in sequences),
        'unique_acquired_poses':sum(s['unique_pose_count'] for s in sequences),
        'split_frames':{split:sum(len(s['frames']) for s in sequences if s['split']==split) for split in ('train','validation','test')},
        'dataset_bytes':disk_bytes(args.output),
        'mean_gt_point_coverage_5cm':float(np.mean([s['mean_gt_point_coverage_5cm'] for s in sequences])),
        'mean_gt_voxel_recall':float(np.mean([s['mean_gt_voxel_recall'] for s in sequences])),
        'max_sensor_up_error':max(s['max_sensor_up_error'] for s in sequences)}
    atomic_json(args.output/'collection_summary.json',summary)
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    main()
