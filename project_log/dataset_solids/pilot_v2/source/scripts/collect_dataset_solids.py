"""Native basic solids mixed with randomized boxes on the v4 roaming terrain."""

from collections import Counter
from functools import lru_cache
import hashlib
from types import SimpleNamespace

import numpy as np
from scipy.spatial import cKDTree
import trimesh

import collect_dataset_v3 as base
import collect_dataset_v4 as v4


@lru_cache(maxsize=64)
def load_asset(path):
    mesh = trimesh.load(path, force='mesh', process=False)
    if not mesh.is_watertight:
        raise ValueError('Asset is not watertight: '+str(path))
    return np.asarray(mesh.vertices), np.asarray(mesh.faces)


def make_scene(folder, scene_id, frames, channels, azimuth_step, *, settings):
    # Reuse the tested long roaming route, sensor setup, and upright ego policy.
    route_settings = SimpleNamespace(**vars(settings))
    route_settings.boxes = 1
    path, config = v4.make_scene(folder,scene_id,frames,channels,azimuth_step,settings=route_settings)
    trajectory = config['ego']['trajectory']
    xy = np.array([p['pose']['translation'][:2] for p in trajectory])
    route_tree = cKDTree(xy)
    paths = sorted(settings.assets.glob('*.obj'))
    if len(paths) != 27 or settings.boxes < 2*len(paths):
        raise ValueError('Require 27 basic solids and >=54 total objects')
    rng = np.random.default_rng(settings.seed+scene_id+1_000_000)
    # Two unscaled copies of every supplied solid, plus variable-size v4 boxes.
    names = sorted(paths*2,key=lambda p:-np.ptp(load_asset(p)[0],axis=0).max())
    names += [None]*(settings.boxes-2*len(paths))
    half = settings.world_size/2
    meshes = [base.box_mesh([-half,-half,-2.5],[half,half,0])]
    objects = []
    for asset in names:
        if asset is None:
            size = np.r_[np.exp(rng.uniform(np.log(.2),np.log(3.),2)),rng.uniform(.1,2.2)]
            vertices, faces = base.box_mesh(np.r_[-size[:2]/2,0.],np.r_[size[:2]/2,size[2]])
            angle = 0.
        else:
            vertices, faces = load_asset(asset)
            angle = float(rng.uniform(-np.pi,np.pi))
            c,s = np.cos(angle),np.sin(angle)
            vertices = vertices @ np.array([[c,-s,0],[s,c,0],[0,0,1]]).T
        vertices = vertices.copy()
        vertices[:,2] -= vertices[:,2].min()
        local_low,local_high = vertices.min(0),vertices.max(0)
        for _ in range(100000):
            center = rng.uniform(-half+2,half-2,2)
            # Every solid lies within the local input cube along the full route.
            if asset is not None and route_tree.query(center)[0] > 1.55:
                continue
            shift = np.r_[center,0.]
            lower,upper = local_low+shift,local_high+shift
            if v4.intersects_route(xy,lower,upper,padding=.55):
                continue
            if any(np.all(upper[:2]+.12 > obj['lower'][:2]) and
                   np.all(lower[:2]-.12 < obj['upper'][:2]) for obj in objects):
                continue
            break
        else:
            raise RuntimeError('Cannot place all assets around the protected route')
        meshes.append((vertices+shift,faces))
        objects.append({'asset':asset.name if asset else 'random_box',
                        'sha256':hashlib.sha256(asset.read_bytes()).hexdigest() if asset else None,
                        'yaw_deg':float(np.rad2deg(angle)), 'scale':1.,
                        'center_xy_m':center.tolist(),'lower':lower.tolist(),'upper':upper.tolist(),
                        'closest_route_center_distance_m':float(route_tree.query(center)[0])})
    offset = 0
    with (folder/'scene.obj').open('w') as stream:
        for index,(vertices,faces) in enumerate(meshes):
            stream.write('o object_{}\n'.format(index))
            for vertex in vertices:
                stream.write('v {:.12f} {:.12f} {:.12f}\n'.format(*vertex))
            for face in faces+offset+1:
                stream.write('f {} {} {}\n'.format(*face))
            offset += len(vertices)
    counts = dict(Counter(obj['asset'] for obj in objects))
    base.atomic_json(folder/'terrain_metadata.json',{
        'world_size_m':[settings.world_size]*2,'flat_floor_z_m':0.,
        'objects':objects,'asset_counts':counts,'object_count':len(objects),
        'native_asset_count':54,'random_box_count':settings.boxes-54,
        'all_native_assets_used_twice':all(counts[p.name]==2 for p in paths),
        'route_xy_min':xy.min(0).tolist(),'route_xy_max':xy.max(0).tolist(),
        'acquisition_duration_s':frames/10.,'route_duration_s':max(1000.,frames/10.),
        'swept_route_object_clearance_m':.55,'all_segments_clear':True,
        'dynamic_objects':False,'sensor_up_world':[0,0,1]})
    return path,config
