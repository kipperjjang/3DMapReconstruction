"""Reverify a complete solids dataset blocked solely by DynaMapping viewer drift.

Preserve original provenance; record the narrowly scoped exception separately.
Never accept changes to acquisition, geometry, simulator or mapping core code.
"""

import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path,value):
    tmp=path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(value,indent=2)+'\n')
    tmp.replace(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--logdir',type=Path,required=True)
    args=parser.parse_args()
    dataset=args.dataset.resolve(); logs=args.logdir.resolve()
    with (dataset/'pipeline.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        provenance=json.loads((dataset/'provenance.json').read_text())
        manifest=json.loads((dataset/'dataset.json').read_text())
        status=json.loads((dataset/'status.json').read_text())
        previous=dict(status)
        snapshot=logs/'source'
        # Only the unimported standalone viewer may differ from the acquisition hash.
        mapping_viewer=dataset.parents[1].parent/'DynaMapping/dynamapping/viewer.py'
        changes={p:{'acquisition_sha256':h,'current_sha256':sha(Path(p))}
                 for p,h in provenance['sources'].items() if sha(Path(p))!=h}
        if set(changes)!={str(mapping_viewer)}:
            raise RuntimeError('Recovery requires exactly a DynaMapping/viewer.py change: '+str(list(changes)))
        for rel,digest in provenance['pipeline_sources'].items():
            if sha(snapshot/rel)!=digest:
                raise RuntimeError('Acquisition snapshot changed: '+rel)
        for name,digest in provenance['asset_sources'].items():
            if sha(dataset/'source_assets'/name)!=digest:
                raise RuntimeError('Pinned source asset changed: '+name)
        sequences=manifest['sequences']
        if len(sequences)!=20 or {s['scene'] for s in sequences}!={f'scene_{i:04d}' for i in range(20)}:
            raise RuntimeError('All 20 completed scenes are required')
        if any(len(s['frames'])!=10000 for s in sequences) or list(dataset.rglob('*.partial')):
            raise RuntimeError('Incomplete scene or partial shard remains')
        for seq in sequences:
            folder=dataset/'sequences'/seq['scene']
            if sha(folder/'scene.obj')!=seq['geometry_sha256'] or sha(folder/'scenario.yaml')!=seq['config_sha256']:
                raise RuntimeError('Geometry/config changed: '+seq['scene'])
            if json.loads((folder/'sequence.json').read_text())!=seq:
                raise RuntimeError('Manifest differs from scene record: '+seq['scene'])
        evidence={'previous_status':previous,'source_changes':changes,
                  'reason':'Standalone viewer is not imported by acquisition; all acquisition core hashes unchanged',
                  'original_provenance_sha256':sha(dataset/'provenance.json'),
                  'recovery_script_sha256':sha(Path(__file__)),
                  'started_at_unix':time.time()}
        atomic_json(logs/'recovery_evidence.json',evidence)
        status.pop('error',None)
        try:
            for name,script,extra in [('validating','validate_dataset_v3.py',['--expected-frames','200000']),
                                      ('auditing','audit_dataset_v3.py',[])]:
                status.update(phase=name,heartbeat_unix=time.time(),recovery=True)
                atomic_json(dataset/'status.json',status)
                subprocess.run([sys.executable,'-u',str(snapshot/'scripts'/script),
                                '--dataset',str(dataset)]+extra,check=True)
            validation=json.loads((dataset/'validation_report.json').read_text())
            audit=json.loads((dataset/'sample_audit.json').read_text())
            if (validation['status']!='validated' or validation['total_frames']!=200000
                    or validation['split_frames']!={'train':160000,'validation':20000,'test':20000}
                    or audit['status']!='passed' or audit['frames']!=60):
                raise RuntimeError('Unexpected final verification counts')
            # Reject additional external edits during the recovery checks.
            current={p:sha(Path(p)) for p in provenance['sources']}
            for p,h in provenance['sources'].items():
                if current[p] != changes.get(p,{}).get('current_sha256',h):
                    raise RuntimeError('Source changed again during recovery: '+p)
            summary=json.loads((dataset/'collection_summary.json').read_text())
            size=sum(max(p.stat().st_size,p.stat().st_blocks*512)
                     for p in dataset.rglob('*') if p.is_file())
            if size>=manifest['max_dataset_bytes']:
                raise RuntimeError('Dataset size cap exceeded')
            evidence.update(status='passed',finished_at_unix=time.time(),
                            validation=validation,audited_frames=60)
            atomic_json(logs/'recovery_evidence.json',evidence)
            summary.update(status='ready_for_training',dataset_bytes=size,validation=validation,
                           audited_frames=60,recovery_evidence=str(logs/'recovery_evidence.json'))
            atomic_json(dataset/'result.json',summary)
            status.update(phase='ready_for_training',heartbeat_unix=time.time(),result=str(dataset/'result.json'))
            atomic_json(dataset/'status.json',status)
            (logs/'RESULTS.md').write_text('# Solids dataset verified after viewer-only source drift\n\n```json\n'+
                                         json.dumps(summary,indent=2)+'\n```\n')
            print(json.dumps(summary,indent=2),flush=True)
        except BaseException as error:
            status.update(phase='failed',error=str(error),heartbeat_unix=time.time())
            atomic_json(dataset/'status.json',status)
            raise


if __name__=='__main__':
    main()
