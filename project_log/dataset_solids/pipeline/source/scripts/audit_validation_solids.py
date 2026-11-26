"""Measure GT and acquired-return representation for every placed solid."""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial import cKDTree

from neural_scene.packed import decode


def audit(directory):
    """Count above-floor GT samples and acquired returns inside each solid bound."""
    folder = directory/'sequences/scene_0000'
    metadata = json.loads((folder/'terrain_metadata.json').read_text())
    objects = metadata['objects']
    lower = np.asarray([obj['lower'] for obj in objects])
    upper = np.asarray([obj['upper'] for obj in objects])
    lower[:,2] = .001  # Exclude floor samples and their subvoxel quantization error.
    gt = np.load(folder/'surface_gt.npz')['points']
    tree = cKDTree(gt)
    gt_counts = []
    for lo,hi in zip(lower,upper):
        selected = gt[tree.query_ball_point((lo+hi)/2,np.linalg.norm(hi-lo)/2+.0002)]
        gt_counts.append(int(np.sum(np.all((selected>=lo-.0002)&(selected<=hi+.0002),axis=1))))
    returns = np.zeros(len(objects),dtype=np.int64)
    frames = returns.copy()
    first = np.full(len(objects),-1,dtype=np.int64)
    with h5py.File(folder/'frames.h5','r') as file:
        offsets = file['current_offsets'][:]
        poses = file['robot_pose'][:]
        for index,pose in enumerate(poses):
            points = decode(file['current'][:,offsets[index]:offsets[index+1]].T)
            world = points.astype(np.float64)@pose[:3,:3].T+pose[:3,3]
            counts = np.sum(np.all((world[:,None,:]>=lower[None]-.0002)&
                                  (world[:,None,:]<=upper[None]+.0002),axis=2),axis=0)
            seen = counts>0
            returns += counts
            frames += seen
            first[(first<0)&seen] = index
    rows = [dict(asset=obj['asset'],copy=obj['copy'],gt_samples_above_floor=g,
                 acquired_returns_in_bounds=int(n),frames_with_returns=int(f),first_observed_frame=int(i))
            for obj,g,n,f,i in zip(objects,gt_counts,returns,frames,first)]
    report = {'status':'passed' if all(gt_counts) and np.all(frames>0) else 'representation_gaps',
        'method':'World AABB membership above Z=1mm, tolerance 0.2mm; counts use stored local-crop measurements',
        'placed_instances':len(objects),'unique_assets':len(metadata['asset_counts']),
        'minimum_copies_per_asset':min(metadata['asset_counts'].values()),
        'instances_with_gt_samples':int(np.count_nonzero(gt_counts)),
        'instances_with_acquired_returns':int(np.count_nonzero(frames)), 'instances':rows}
    (directory/'asset_coverage_report.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True)
    args = parser.parse_args()
    result = audit(args.dataset)
    print(json.dumps({k:v for k,v in result.items() if k!='instances'},indent=2))
