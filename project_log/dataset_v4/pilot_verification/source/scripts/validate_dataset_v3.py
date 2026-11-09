"""Verify every compact training record, acquisition identity and upright sensor."""

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


def load_manifest(directory, completed_only=False):
    """Read final metadata or an explicitly labelled snapshot of closed shards."""
    if not completed_only:
        return json.loads((directory/'dataset.json').read_text())
    sequences=[json.loads(p.read_text()) for p in sorted(directory.glob('sequences/*/sequence.json'))]
    if not sequences:
        raise ValueError('No completed scenes to validate')
    return {'format':'neural-scene-packed-v1','max_dataset_bytes':10000000000,'sequences':sequences}


def validate(directory, expected_frames, completed_only=False):
    """Read every shard and assert integrity without caching the whole dataset."""
    directory = directory.resolve()
    manifest = load_manifest(directory,completed_only)
    assert manifest['format']=='neural-scene-packed-v1'
    seen_scenes, seen_geometry = set(), set()
    total, split_counts, coverages, recalls = 0, {}, [], []
    max_transform = max_up = 0.
    for sequence in manifest['sequences']:
        assert sequence['scene'] not in seen_scenes
        assert sequence['geometry_sha256'] not in seen_geometry
        seen_scenes.add(sequence['scene']); seen_geometry.add(sequence['geometry_sha256'])
        path = (directory/sequence['shard']).resolve()
        assert directory in path.parents
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda:stream.read(1024*1024),b''):
                digest.update(block)
        assert digest.hexdigest()==sequence['shard_sha256']
        count = len(sequence['frames'])
        with h5py.File(path,'r') as f:
            pose, sensor, transforms = f['robot_pose'][:], f['sensor_pose'][:], f['transformation'][:]
            assert pose.shape==(count,4,4) and np.isfinite(pose).all()
            assert np.isfinite(sensor).all() and np.isfinite(transforms).all()
            np.testing.assert_array_equal(f['timestamp_ns'][:],np.arange(count,dtype=np.int64)*100000000)
            assert len(set(row.tobytes() for row in pose))==count==sequence['unique_pose_count']
            up = float(np.abs(sensor[:,:3,2]-[0,0,1]).max())
            assert up<1e-9
            expected = np.concatenate((np.eye(4)[None],np.linalg.inv(pose[1:])@pose[:-1]))
            error = float(np.abs(expected-transforms).max())
            assert error<1e-8
            max_transform=max(max_transform,error); max_up=max(max_up,up)
            for kind,field in (('current','current_points'),('target','target_voxels')):
                offsets=f[kind+'_offsets'][:]
                np.testing.assert_array_equal(np.diff(offsets),[r[field] for r in sequence['frames']])
                assert len(offsets)==count+1 and offsets[0]==0 and offsets[-1]==f[kind].shape[1]
                assert np.all(np.diff(offsets)>0)
                # Spatial order allows linear-time GT uniqueness verification.
                cells=f[kind][:3,:].astype(np.int32)
                assert np.all(cells<64)
                if kind=='target':
                    key=(cells[0]*64+cells[1])*64+cells[2]
                    differences=np.diff(key)
                    differences[offsets[1:-1]-1]=1
                    assert np.all(differences>0), 'Duplicate/unsorted GT voxel'
        total+=count
        split_counts[sequence['split']]=split_counts.get(sequence['split'],0)+count
        coverages.extend(r['gt_point_coverage_5cm'] for r in sequence['frames'])
        recalls.extend(r['gt_voxel_recall'] for r in sequence['frames'])
    if not completed_only:
        assert total==expected_frames
    assert all(0<=x<=1 for x in coverages+recalls)
    size=sum(max(p.stat().st_size,p.stat().st_blocks*512) for p in directory.rglob('*') if p.is_file())
    assert size<manifest['max_dataset_bytes']<=10_000_000_000
    return {'status':'validated_subset' if completed_only else 'validated','total_frames':total,'unique_geometry_scenes':len(seen_geometry),
        'split_frames':split_counts,'dataset_bytes':size,'max_sensor_up_error':max_up,
        'max_feedback_transform_error':max_transform,'mean_gt_point_coverage_5cm':float(np.mean(coverages)),
        'gt_point_coverage_p05_p50_p95':np.quantile(coverages,[.05,.5,.95]).tolist(),
        'mean_gt_voxel_recall':float(np.mean(recalls))}


def main():
    """Validate the final data before any model training starts."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--expected-frames',type=int,default=200000)
    parser.add_argument('--completed-only',action='store_true',help='Audit only closed shards while collection continues')
    args=parser.parse_args()
    report=validate(args.dataset,args.expected_frames,args.completed_only)
    name='partial_validation_report.json' if args.completed_only else 'validation_report.json'
    (args.dataset/name).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
