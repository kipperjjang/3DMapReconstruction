"""Collect wide flat box scenes using the established packed v3 frame pipeline."""

import argparse
import fcntl
from functools import partial
import gc
import json
from pathlib import Path

import numpy as np
import torch
import yaml

import collect_dataset_v3 as base


def smooth_route(rng, duration, world_size, speed):
    """Return constant-distance samples of a smooth roaming path and tangent yaw."""
    limit = world_size * .36
    controls = [np.array([-limit * .7, -limit * .6]), np.array([-limit * .1, -limit * .4])]
    while len(controls) < max(90, int(duration * speed / 6) + 30):
        incoming = controls[-1] - controls[-2]
        for _ in range(10000):
            point = rng.uniform(-limit, limit, 2)
            delta = point - controls[-1]
            distance = np.linalg.norm(delta)
            cosine = np.dot(delta, incoming) / max(distance * np.linalg.norm(incoming), 1e-12)
            if world_size * .14 < distance < world_size * .55 and cosine > -.75:
                controls.append(point)
                break
        else:
            raise RuntimeError('Could not extend forward roaming route')
    controls = np.array([2*controls[0]-controls[1], *controls, 2*controls[-1]-controls[-2]])
    dense = []
    for i in range(1, len(controls)-2):
        p0,p1,p2,p3 = controls[i-1:i+3]
        t = np.linspace(0, 1, 100, endpoint=False)[:,None]
        dense.append(.5*((2*p1)+(-p0+p2)*t+(2*p0-5*p1+4*p2-p3)*t*t+(-p0+3*p1-3*p2+p3)*t*t*t))
    dense = np.concatenate(dense)
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(dense,axis=0),axis=1))]
    if arc[-1] < duration * speed + 1 or np.abs(dense).max() >= world_size/2-2:
        raise RuntimeError('Route leaves terrain or is shorter than requested duration')
    times = np.arange(int(round(duration*4))+1) / 4
    distances = times * speed
    xy = np.stack([np.interp(distances,arc,dense[:,axis]) for axis in range(2)],axis=1)
    ahead = np.stack([np.interp(distances+.02,arc,dense[:,axis]) for axis in range(2)],axis=1)
    yaw = np.rad2deg(np.unwrap(np.arctan2(ahead[:,1]-xy[:,1],ahead[:,0]-xy[:,0])))
    if np.linalg.norm(np.diff(xy,axis=0),axis=1).min() < .01:
        raise RuntimeError('Stationary segment in roaming route')
    return times,xy,yaw


def intersects_route(xy, lower, upper, padding=.55):
    """Test every continuous path segment against an inflated box XY footprint."""
    starts,delta = xy[:-1],np.diff(xy,axis=0)
    enter,leave = np.zeros(len(delta)),np.ones(len(delta))
    possible = np.ones(len(delta),dtype=bool)
    for axis in range(2):
        low,high = lower[axis]-padding,upper[axis]+padding
        moving = np.abs(delta[:,axis]) > 1e-12
        possible &= moving | ((starts[:,axis] >= low) & (starts[:,axis] <= high))
        divisor = np.where(moving,delta[:,axis],1.)
        first,last = (low-starts[:,axis])/divisor,(high-starts[:,axis])/divisor
        enter = np.maximum(enter,np.where(moving,np.minimum(first,last),-np.inf))
        leave = np.minimum(leave,np.where(moving,np.maximum(first,last),np.inf))
    return bool(np.any(possible & (enter <= leave)))


def make_scene(folder, scene_id, frames, channels, azimuth_step, *, settings):
    """Author flat terrain and static boxes outside a moving, turning trajectory."""
    rng = np.random.default_rng(settings.seed + scene_id)
    # Pilot tests the same long-path construction, even when acquiring few scans.
    route_duration = max(1000.,frames/10.)
    speed = float(rng.uniform(.65,.85))
    times,xy,yaw = smooth_route(rng,route_duration,settings.world_size,speed)
    height = float(rng.uniform(.65,.85))
    half = settings.world_size/2
    # Keep the underside below the robot-centred training cube, as in v3.
    meshes = [base.box_mesh([-half,-half,-2.5],[half,half,0])]
    boxes = []
    attempts = 0
    while len(boxes) < settings.boxes and attempts < 100000:
        attempts += 1
        # Log-uniform footprints include small obstacles and large blocks.
        size = np.r_[np.exp(rng.uniform(np.log(.2),np.log(3.),2)),rng.uniform(.1,2.2)]
        center = rng.uniform(-half+2,half-2,2)
        lower = np.r_[center-size[:2]/2,0.]
        upper = lower + size
        if intersects_route(xy,lower,upper):
            continue
        if any(np.all(upper[:2]+.12 > box['lower'][:2]) and np.all(lower[:2]-.12 < box['upper'][:2]) for box in boxes):
            continue
        boxes.append({'lower':lower.tolist(),'upper':upper.tolist(),'size':size.tolist()})
        meshes.append(base.box_mesh(lower,upper))
    if len(boxes) != settings.boxes:
        raise RuntimeError('Insufficient box placement space around the protected route')
    mesh = folder/'scene.obj'
    count = 0
    with mesh.open('w') as stream:
        for vertices,faces in meshes:
            for row in vertices:
                stream.write('v {} {} {}\n'.format(*row))
            for row in faces+count+1:
                stream.write('f {} {} {}\n'.format(*row))
            count += len(vertices)
    trajectory = [{'time_s':float(t),'pose':{'translation':[float(p[0]),float(p[1]),height],
                   'rpy_deg':[0,0,float(angle)]}} for t,p,angle in zip(times,xy,yaw)]
    config = {'run':{'seed':settings.seed+scene_id,'duration_s':frames/10.,'dt_s':.1,'environments':1,'device':'cpu'},
        'assets':{'terrain':{'path':str(mesh.resolve()),'units':'m','up_axis':'Z'}},
        'objects':[{'id':0,'asset':'terrain','static':True}],
        'sensors':[{'id':'lidar','frequency_hz':10,'range_m':[.05,40.],
            'extrinsic':{'translation':[0,0,0],'rpy_deg':[0,0,0]},
            'pattern':{'kind':'lidar','channels':channels,'vertical_fov_range':[-85,45],
                       'horizontal_fov_range':[-180,180],'horizontal_res':azimuth_step}}],
        'ego':{'mode':'scripted','trajectory':trajectory,'terrain_alignment':False,
               'radius':.15,'clearance':.05,'sensor_height':height,'max_slope_deg':60},
        'domain_randomization':{'enabled':False},
        'output':{'directory':str(folder/'unused-raw-archive'),'shard_bytes':67108864,
                  'chunk_frames':8,'queue_size':4,'compression':'lzf','store_xyz':True}}
    path = folder/'scenario.yaml'
    path.write_text(yaml.safe_dump(config,sort_keys=False))
    sizes = np.array([box['size'] for box in boxes])
    metadata = {'world_size_m':[settings.world_size,settings.world_size], 'static_boxes':boxes,
        'box_count':len(boxes),'box_size_min_xyz_m':sizes.min(0).tolist(),'box_size_max_xyz_m':sizes.max(0).tolist(),
        'trajectory':'arc-length sampled Catmull-Rom roaming curve; yaw follows tangent',
        'route_duration_s':route_duration,'acquisition_duration_s':frames/10.,
        'speed_m_s':speed,'route_length_m':float(np.linalg.norm(np.diff(xy,axis=0),axis=1).sum()),
        'route_xy_min':xy.min(0).tolist(),'route_xy_max':xy.max(0).tolist(),
        'swept_route_box_clearance_m':.55,'all_segments_clear':True,
        'dynamic_objects':False,'sensor_up_world':[0,0,1]}
    base.atomic_json(folder/'terrain_metadata.json',metadata)
    return path,config


def shared_gt(config, authored, folder):
    """Generate the full static boundary with current DynaRaySim's GT algorithm."""
    scene = base.SceneBatch(config)
    settings = base.load_config(mode='composed')
    settings['memory_limit_mb'] = 4096
    gt = base.GroundTruth(scene,settings)
    packet = {'states':[base.state_record(scene,0)],'scans':[{'env':0,'episode':0,'sensor':'lidar',
              'timestamp_ns':0,'sensor_pose_world':scene.egos[0].clone()}]}
    gt.process(packet)
    points = packet['scans'][0]['_gt_background']['points'].numpy()
    full_count = len(points)
    route = np.array([item['pose']['translation'] for item in authored['ego']['trajectory']])
    lower,upper = route.min(0)-2.8,route.max(0)+2.8
    points = points[np.all((points >= lower) & (points <= upper),axis=1)].copy()
    np.savez_compressed(folder/'surface_gt.npz',points=points)
    metadata = {'settings':gt.settings,'full_scene_surface_samples':full_count,
        'retained_surface_samples':len(points),'roi_union_lower':lower.tolist(),'roi_union_upper':upper.tolist(),
        'static_scene':True,'visibility_filtering':False}
    del scene,gt,packet
    gc.collect()
    return points,metadata


def parser():
    """Return collection options shared by scene workers and the supervisor."""
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--output',type=Path,required=True)
    result.add_argument('--scene',type=int,required=True)
    result.add_argument('--frames-per-scene',type=int,default=10000)
    result.add_argument('--world-size',type=float,default=40.)
    result.add_argument('--boxes',type=int,default=160)
    result.add_argument('--seed',type=int,default=20261109)
    result.add_argument('--channels',type=int,default=32)
    result.add_argument('--azimuth-step',type=float,default=3.)
    result.add_argument('--device',default='cpu',choices=['cpu'])
    result.add_argument('--max-bytes',type=int,default=10000000000)
    return result


def main():
    """Acquire a single locked scene through the tested v3 frame encoder/mapper."""
    args = parser().parse_args()
    if args.frames_per_scene < 12 or args.boxes < 1 or args.world_size < 20 or not 0 < args.max_bytes <= 10000000000:
        raise ValueError('Require >=12 frames, positive boxes, >=20m world, <=10GB cap')
    args.output = args.output.resolve()
    args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    base.make_scene = partial(make_scene,settings=args)
    base.shared_gt = shared_gt
    with (args.output/'scene_{:04d}.lock'.format(args.scene)).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        result = base.collect_scene(args,args.scene)
    print(json.dumps({'status':'completed','scene':result['scene'],'frames':len(result['frames'])}),flush=True)


if __name__ == '__main__':
    main()
