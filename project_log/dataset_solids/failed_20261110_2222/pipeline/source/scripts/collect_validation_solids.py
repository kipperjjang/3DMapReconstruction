"""Collect and verify a standalone validation scene containing every basic solid."""

import argparse
from collections import Counter
import fcntl
from functools import partial
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np
import torch
import trimesh
import yaml

import collect_dataset_v3 as base
import collect_dataset_v4 as v4
from validate_dataset_v3 import validate

ROOT = Path(__file__).resolve().parents[1]


def rounded_route(frames):
    """Return timed constant-speed samples of rounded aisle waypoints."""
    controls = []
    # Five slightly displaced circuits give unique poses and multiple views.
    for lap in range(5):
        shift = -lap * .035
        for row in range(6):
            y = -8.5 + row * 3 + shift
            ends = (-10., 10.) if row % 2 == 0 else (10., -10.)
            controls.extend([[ends[0], y], [ends[1], y]])
        controls.extend([[-12.5, 6.5 + shift], [-12.5, -11.], [-10., -11.]])
    controls = np.asarray(controls)
    padded = np.vstack([controls[0], controls, controls[-1]])
    dense = [controls[0]]
    # Quadratic corner fillets stay inside the convex hull of each corner.
    for index in range(1, len(padded)-1):
        prev, point, following = padded[index-1:index+2]
        incoming, outgoing = prev-point, following-point
        if min(np.linalg.norm(incoming), np.linalg.norm(outgoing)) < 1e-8:
            continue
        cut = min(.6, np.linalg.norm(incoming)/3, np.linalg.norm(outgoing)/3)
        entry = point + incoming/np.linalg.norm(incoming)*cut
        leave = point + outgoing/np.linalg.norm(outgoing)*cut
        dense.append(entry)
        for t in np.linspace(0, 1, 40)[1:]:
            dense.append((1-t)**2*entry + 2*(1-t)*t*point + t*t*leave)
    dense.append(controls[-1])
    dense = np.asarray(dense)
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(dense, axis=0), axis=1))]
    duration = frames/10.
    speed = .65
    if duration*speed > arc[-1]-.1:
        raise ValueError('Requested route exceeds five circuits')
    times = np.arange(int(np.ceil(duration*4))+1)/4
    xy = np.column_stack([np.interp(times*speed, arc, dense[:,i]) for i in range(2)])
    ahead = np.column_stack([np.interp(times*speed+.002, arc, dense[:,i]) for i in range(2)])
    yaw = np.rad2deg(np.unwrap(np.arctan2(*(ahead-xy)[:, ::-1].T)))
    return times, xy, yaw, controls, speed


def make_scene(folder, scene_id, frames, channels, azimuth_step, *, assets):
    """Merge native solid meshes and author a checked scripted scenario."""
    times, xy, yaw, controls, speed = rounded_route(frames)
    paths = sorted(assets.glob('*.obj'))
    if len(paths) != 27:
        raise ValueError('Expected all 27 basic solid assets')
    rng = np.random.default_rng(20261110)
    placements = [(path, copy) for copy in range(2) for path in paths]
    rng.shuffle(placements)
    vertices, faces = base.box_mesh([-15,-13,-2.5], [15,11,0])
    meshes = [(vertices, faces)]
    objects = []
    for index, (path, copy) in enumerate(placements):
        center = np.array([-8.+2.*(index%9), -7.5+3.*(index//9)])
        angle = float(rng.uniform(-np.pi, np.pi))
        rotation = np.array([[np.cos(angle),-np.sin(angle),0],
                             [np.sin(angle),np.cos(angle),0], [0,0,1]])
        mesh = trimesh.load(path, force='mesh', process=False)
        transformed = np.asarray(mesh.vertices) @ rotation.T
        transformed[:,2] -= transformed[:,2].min()
        transformed[:,:2] += center
        lower, upper = transformed.min(0), transformed.max(0)
        if v4.intersects_route(xy, lower, upper, padding=.21):
            raise ValueError('Solid intersects the swept ego route: '+path.name)
        # Check approach at recorded 10 Hz poses, including the actual acquisition end.
        stamps = np.arange(frames)/10.
        acquired_xy = np.column_stack([np.interp(stamps,times,xy[:,i]) for i in range(2)])
        distances = np.linalg.norm(acquired_xy-center, axis=1)
        if distances.min() > 1.25:
            raise ValueError('Acquisition does not approach '+path.name)
        objects.append({'asset':path.name, 'copy':copy, 'center_xy_m':center.tolist(),
            'yaw_deg':float(np.rad2deg(angle)), 'lower':lower.tolist(),'upper':upper.tolist(),
            'closest_acquired_ego_distance_m':float(distances.min()),
            'frames_with_center_within_1_25m':int(np.sum(distances<1.25)),
            'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
        meshes.append((transformed, np.asarray(mesh.faces)))
    mesh_path = folder/'scene.obj'
    offset = 0
    with mesh_path.open('w') as stream:
        for index,(vertices,faces) in enumerate(meshes):
            stream.write('o floor\n' if index==0 else 'o solid_{:03d}\n'.format(index-1))
            for vertex in vertices:
                stream.write('v {:.12f} {:.12f} {:.12f}\n'.format(*vertex))
            for face in faces+offset+1:
                stream.write('f {} {} {}\n'.format(*face))
            offset += len(vertices)
    # Use the existing v4 configuration exactly, replacing only geometry and route.
    height = .7
    config = {'run':{'seed':20261110,'duration_s':frames/10.,'dt_s':.1,'environments':1,'device':'cpu'},
        'assets':{'terrain':{'path':str(mesh_path.resolve()),'units':'m','up_axis':'Z'}},
        'objects':[{'id':0,'asset':'terrain','static':True}],
        'sensors':[{'id':'lidar','frequency_hz':10,'range_m':[.05,40.],
            'extrinsic':{'translation':[0,0,0],'rpy_deg':[0,0,0]},
            'pattern':{'kind':'lidar','channels':channels,'vertical_fov_range':[-85,45],
                       'horizontal_fov_range':[-180,180],'horizontal_res':azimuth_step}}],
        'ego':{'mode':'scripted','trajectory':[{'time_s':float(t), 'pose':{
            'translation':[float(p[0]),float(p[1]),height], 'rpy_deg':[0,0,float(a)]}}
            for t,p,a in zip(times,xy,yaw)],
            'radius':.15,'clearance':.05,'sensor_height':height,'max_slope_deg':60},
        'domain_randomization':{'enabled':False},
        'output':{'directory':str(folder/'unused-raw-archive'),'shard_bytes':67108864,
                  'chunk_frames':8,'queue_size':4,'compression':'lzf','store_xyz':True}}
    path = folder/'scenario.yaml'
    path.write_text(yaml.safe_dump(config,sort_keys=False))
    counts = dict(Counter(o['asset'] for o in objects))
    base.atomic_json(folder/'terrain_metadata.json',{
        'world_size_m':[30,24], 'flat_floor_z_m':0,'objects':objects,'asset_counts':counts,
        'all_assets_used_at_least_twice':min(counts.values())>=2,
        'object_count':len(objects),'all_segments_clear':True,'swept_aabb_padding_m':.21,
        'trajectory':'rounded aisle waypoints, constant arc-length samples, simulator scripted interpolation',
        'control_waypoints_xy_m':controls.tolist(),'speed_m_s':speed,
        'acquisition_duration_s':frames/10.,'dynamic_objects':False})
    # Native SVG preview is reviewable without a plotting dependency.
    shapes = ['<rect x="-15" y="-13" width="30" height="24" fill="#eee"/>']
    for obj in objects:
        x,y = obj['center_xy_m']
        shapes.append(f'<circle cx="{x}" cy="{y}" r="0.35" fill="#dd872b"><title>{obj["asset"]} copy {obj["copy"]}</title></circle>')
    points = ' '.join(f'{x:.3f},{y:.3f}' for x,y in xy)
    shapes.append(f'<polyline points="{points}" fill="none" stroke="#1764a0" stroke-width=".045"/>')
    (folder/'terrain_preview.svg').write_text('<svg xmlns="http://www.w3.org/2000/svg" viewBox="-15 -13 30 24">'+''.join(shapes)+'</svg>')
    return path, config


def main():
    """Collect, validate and independently audit the frozen validation scene."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/validation_basic_solids')
    parser.add_argument('--assets',type=Path,default=base.SIM/'resources/assets/basic_solid')
    parser.add_argument('--frames-per-scene',type=int,default=10000)
    parser.add_argument('--prepare-only',action='store_true')
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.assets = args.assets.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    args.channels, args.azimuth_step, args.device, args.max_bytes = 32,3.,'cpu',10_000_000_000
    if args.frames_per_scene < 2300:
        raise ValueError('At least 2300 frames are required to visit all objects')
    torch.set_num_threads(2)
    folder = args.output/'sequences/scene_0000'
    folder.mkdir(parents=True,exist_ok=True)
    if args.prepare_only:
        make_scene(folder,0,args.frames_per_scene,args.channels,args.azimuth_step,assets=args.assets)
        print('Scene and all asset/route checks prepared',flush=True)
        return
    with (args.output/'pipeline.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        begun = time.time()
        status = {'phase':'collecting','target_frames':args.frames_per_scene,'started_at_unix':begun}
        base.atomic_json(args.output/'status.json',status)
        try:
            # Pin asset bytes in the output; source assets are never modified.
            asset_copy = args.output/'source_assets'
            if not asset_copy.exists():
                shutil.copytree(args.assets,asset_copy)
            provenance_args = argparse.Namespace(**{k:str(v) if isinstance(v,Path) else v
                                                   for k,v in vars(args).items()})
            provenance = base.provenance(provenance_args)
            provenance['pipeline_sources'] = {p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                for p in Path(__file__).parent.glob('*.py')}
            base.atomic_json(args.output/'provenance.json',provenance)
            base.make_scene = partial(make_scene,assets=asset_copy)
            base.shared_gt = v4.shared_gt
            result = base.collect_scene(args,0)
            manifest = {'format':'neural-scene-packed-v1','input_policy':'aligned_temporal_t01',
                'grid':{'voxel_size':.05,'shape':[64,64,64],'origin':[-1.6,-1.6,-1.6]},
                'split_policy':'standalone validation only; never merge into training',
                'max_dataset_bytes':args.max_bytes,'sequences':[result]}
            base.atomic_json(args.output/'dataset.json',manifest)
            status['phase']='validating'
            base.atomic_json(args.output/'status.json',status)
            report = validate(args.output,args.frames_per_scene)
            base.atomic_json(args.output/'validation_report.json',report)
            status['phase']='auditing'
            base.atomic_json(args.output/'status.json',status)
            subprocess.run([sys.executable,str(Path(__file__).with_name('audit_dataset_v3.py')),
                '--dataset',str(args.output)],check=True)
            audit = json.loads((args.output/'sample_audit.json').read_text())
            if audit['status']!='passed' or audit['frames']!=3:
                raise RuntimeError('Independent audit did not pass')
            final = dict(report,status='ready_for_validation',audited_frames=audit['frames'],elapsed_s=time.time()-begun)
            base.atomic_json(args.output/'result.json',final)
            status.update(phase='ready_for_validation',elapsed_s=time.time()-begun)
            base.atomic_json(args.output/'status.json',status)
            print(json.dumps(final,indent=2),flush=True)
        except BaseException as error:
            status.update(phase='failed',error=str(error))
            base.atomic_json(args.output/'status.json',status)
            traceback.print_exc()
            raise


if __name__=='__main__':
    main()
