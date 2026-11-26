"""Independently re-ray-cast representative poses and reconstruct dense-surface GT."""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from scipy.spatial import cKDTree
from dynaraysim.main.config import resolve
from dynaraysim.main.scene import SceneBatch
from dynaraysim.sensors.warp_backend import WarpBackend
from neural_scene.packed import crop, centroids, read_frame
from validate_dataset_v3 import load_manifest


def main():
    """Check first/middle/last acquisitions against source mesh and shared GT."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--completed-only',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(2)
    manifest=load_manifest(args.dataset,args.completed_only)
    rows=[]
    for sequence in manifest['sequences']:
        folder=args.dataset/'sequences'/sequence['scene']
        config=resolve(folder/'scenario.yaml','cpu')
        scene=SceneBatch(config)
        pattern=config.patterns[0]
        backend=WarpBackend(scene.environments,'cpu',1,{pattern.hash:pattern.count},False,1)
        surface=np.load(folder/'surface_gt.npz',allow_pickle=False)['points'].astype(np.float64)
        for index in sorted({0,len(sequence['frames'])//2,len(sequence['frames'])-1}):
            frame=read_frame(args.dataset/sequence['shard'],index)
            pose=frame['robot_pose']; sensor=frame['sensor_pose']
            yaw=np.arctan2(sensor[1,0],sensor[0,0])
            sensor_xyzw=torch.tensor([*sensor[:3,3],0,0,np.sin(yaw/2),np.cos(yaw/2)],dtype=torch.float64)
            result=backend.cast(pattern,[sensor_xyzw],[0],[0],tuple(config.data['sensors'][0]['range_m']))
            ranges,status=result[0][0],result[1][0]
            sensor_points=(pattern.origins+pattern.directions*ranges[:,None])[status==1].numpy()
            world_points=sensor_points.astype(np.float64)@sensor[:3,:3].T+sensor[:3,3]
            observations=crop((world_points-pose[:3,3])@pose[:3,:3])
            # Full retained surface scan, independent of collection's KD-tree ROI.
            target=centroids(crop((surface-pose[:3,3])@pose[:3,:3]))
            errors={}
            for field,reference in (('current_points',observations),('target_points',target)):
                stored=frame[field]
                if len(stored)!=len(reference):
                    raise ValueError('{} frame {} {} count mismatch {} != {}'.format(sequence['scene'],index,field,len(stored),len(reference)))
                error=max(cKDTree(stored).query(reference)[0].max(),cKDTree(reference).query(stored)[0].max())
                if error>.00018:
                    raise ValueError('Reconstruction exceeds documented quantization bound: '+str(error))
                errors[field+'_max_distance_m']=float(error)
            rows.append(dict(scene=sequence['scene'],index=index,**errors))
            print(json.dumps(rows[-1]),flush=True)
    report={'status':'passed_subset' if args.completed_only else 'passed','frames':len(rows),'method':'independent reraycast + full retained dense GT scan; mutual distance <=0.18mm including float rounding', 'samples':rows}
    name='partial_sample_audit.json' if args.completed_only else 'sample_audit.json'
    (args.dataset/name).write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    main()
