"""Split v4 into two scene-disjoint 200,000-frame datasets without data copies."""

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

import yaml

ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    """Read dataset metadata."""
    return json.loads(path.read_text())


def write_json(path, value):
    """Write independently owned metadata in a private staging directory."""
    path.write_text(json.dumps(value,indent=2)+'\n')


def split_dataset(source, destinations):
    """Hard-link immutable scene data and copy metadata into two new dataset roots.

    Hard links survive removal of the parent path and satisfy the reader's
    containment checks. Mutable metadata/scenario files are independent copies.
    """
    source=source.resolve(); destinations=[p.resolve() for p in destinations]
    if len(set(destinations))!=2 or any(p.exists() or p==source or source in p.parents for p in destinations):
        raise ValueError('Require two distinct, new destination directories outside the source')
    manifest=read_json(source/'dataset.json')
    parent_hash=hashlib.sha256((source/'dataset.json').read_bytes()).hexdigest()
    validation=read_json(source/'validation_report.json')
    audit=read_json(source/'sample_audit.json')
    settings=read_json(source/'collection_settings.json')
    sequences=sorted(manifest['sequences'],key=lambda s:s['scene'])
    if manifest['format']!='neural-scene-packed-v1' or validation['status']!='validated' or validation['total_frames']!=400000:
        raise ValueError('Source must be a validated packed 400,000-frame dataset')
    if len(sequences)!=40 or any(len(s['frames'])!=10000 for s in sequences):
        raise ValueError('Expected 40 whole scenes, each with 10,000 frames')
    if len({s['scene'] for s in sequences})!=40 or len({s['geometry_sha256'] for s in sequences})!=40:
        raise ValueError('Source scenes and geometry must be unique')
    if audit['status']!='passed' or audit['frames']!=120:
        raise ValueError('Expected 120 passing parent audit samples')
    groups=[sequences[:20],sequences[20:]]
    for group in groups:
        counts=Counter()
        for s in group:
            counts[s['split']]+=len(s['frames'])
        if counts!=Counter(train=160000,validation=20000,test=20000):
            raise ValueError('Each partition must preserve the original 80/10/10 split')
    staged=[]; entries=[]
    try:
        for number,(destination,group) in enumerate(zip(destinations,groups),1):
            destination.parent.mkdir(parents=True,exist_ok=True)
            stage=Path(tempfile.mkdtemp(prefix='.'+destination.name+'.staging-',dir=destination.parent))
            staged.append(stage)
            partition={'part':number,'parts':2,'parent_dataset':str(source),'parent_manifest_sha256':parent_hash,
                'method':'20 whole scenes per part, sorted by scene ID; original splits preserved',
                'storage':'hard-linked immutable files, independent metadata and scenario files'}
            copied=[]; shared_bytes=0
            for sequence in group:
                name=sequence['scene']
                if Path(name).name!=name:
                    raise ValueError('Invalid scene name')
                original=source/'sequences'/name
                if (source/sequence['shard']).resolve()!=original.resolve()/'frames.h5':
                    raise ValueError('Unexpected source shard path')
                folder=stage/'sequences'/name; folder.mkdir(parents=True)
                for filename in ('frames.h5','surface_gt.npz','scene.obj','gt_metadata.json','terrain_metadata.json'):
                    item=original/filename
                    if item.is_symlink() or not item.is_file():
                        raise ValueError('Expected regular source file: '+str(item))
                    os.link(item,folder/filename)
                    shared_bytes+=item.stat().st_size
                scenario=yaml.safe_load((original/'scenario.yaml').read_text())
                scenario['assets']['terrain']['path']=str(destination/'sequences'/name/'scene.obj')
                scenario['output']['directory']=str(destination/'sequences'/name/'unused-raw-archive')
                (folder/'scenario.yaml').write_text(yaml.safe_dump(scenario,sort_keys=False))
                record={**sequence,'shard':str(Path('sequences')/name/'frames.h5'),
                    'config_sha256':hashlib.sha256((folder/'scenario.yaml').read_bytes()).hexdigest()}
                write_json(folder/'sequence.json',record); copied.append(record)
            write_json(stage/'dataset.json',{**manifest,'sequences':copied,'expected_frames':200000,'partition':partition})
            write_json(stage/'collection_settings.json',{**settings,'scenes':20,
                'start_scene':int(group[0]['scene'].split('_')[-1]),'partition':partition})
            names={s['scene'] for s in group}
            frames={'train':160000,'validation':20000,'test':20000}
            write_json(stage/'partition.json',{**partition,'scene_names':sorted(names),'split_frames':frames,'shared_file_bytes':shared_bytes})
            write_json(stage/'collection_summary.json',{'status':'partitioned','total_frames':200000,
                'unique_acquired_poses':sum(s['unique_pose_count'] for s in group),'split_frames':frames,'partition':partition})
            write_json(stage/'provenance.json',{'partition':partition,'parent_acquisition_provenance':read_json(source/'provenance.json')})
            samples=[row for row in audit['samples'] if row['scene'] in names]
            if Counter(row['scene'] for row in samples)!=Counter({name:3 for name in names}):
                raise ValueError('Parent audit is missing scene samples')
            write_json(stage/'sample_audit.json',{**audit,'frames':len(samples),'samples':samples,
                'inheritance':{'parent_dataset':str(source),'new_raycast_performed':False,
                    'parent_audit_sha256':hashlib.sha256((source/'sample_audit.json').read_bytes()).hexdigest(),
                    'reason':'identical mesh, point/GT/pose data; scenario paths only changed'}})
            entries.append({'dataset':str(destination),'part':number,'frames':200000,
                'scene_names':sorted(names),'split_frames':frames,'shared_file_bytes':shared_bytes})
        for stage,destination in zip(staged,destinations):
            if destination.exists():
                raise FileExistsError(str(destination))
            stage.rename(destination)
        return {'parent_dataset':str(source),'parent_manifest_sha256':parent_hash,
                'total_frames':400000,'scene_overlap':[],'partitions':entries}
    finally:
        for stage in staged:
            if stage.exists():
                shutil.rmtree(stage)


def finalize_partition(directory):
    """Publish readiness only after the whole partition's validation passes."""
    directory=directory.resolve()
    validation=read_json(directory/'validation_report.json')
    partition=read_json(directory/'partition.json')
    audit=read_json(directory/'sample_audit.json')
    if validation['status']!='validated' or validation['total_frames']!=200000 or validation['unique_geometry_scenes']!=20:
        raise ValueError('Partition validation is incomplete')
    if validation['split_frames']!={'train':160000,'validation':20000,'test':20000} or audit['status']!='passed' or audit['frames']!=60:
        raise ValueError('Unexpected partition splits or inherited audit')
    size=sum(max(p.stat().st_size,p.stat().st_blocks*512) for p in directory.rglob('*') if p.is_file())
    result={'status':'ready_for_training','total_frames':200000,'scenes':20,
        'split_frames':validation['split_frames'],'dataset_bytes':size,'partition':partition,
        'validation':validation,'audited_frames':60,'audit_inherited_from_parent':True}
    write_json(directory/'result.json',result)
    write_json(directory/'status.json',{'phase':'ready_for_training','target_frames':200000,
        'completed_scenes':20,'acquired_frames_lower_bound':200000,'heartbeat_unix':time.time(),
        'output':str(directory),'result':str(directory/'result.json'),'partition':partition['part']})
    return result


def main():
    """Create, validate and finalize both partitions without replacing existing data."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=ROOT/'outputs/training_dataset_v4')
    parser.add_argument('--part1',type=Path,default=ROOT/'outputs/training_dataset_v4_part1')
    parser.add_argument('--part2',type=Path,default=ROOT/'outputs/training_dataset_v4_part2')
    args=parser.parse_args()
    result=split_dataset(args.source,[args.part1,args.part2])
    for entry in result['partitions']:
        directory=Path(entry['dataset'])
        with (directory/'validation.log').open('w') as log:
            subprocess.run([sys.executable,str(Path(__file__).resolve().parent/'validate_dataset.py'),
                '--dataset',str(directory),'--expected-frames','200000'],check=True,stdout=log,stderr=subprocess.STDOUT)
        finalize_partition(directory)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
