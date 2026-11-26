"""Supervise wide-terrain collection, then validate and independently audit it."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
STOP = False


def atomic_json(path, value):
    """Atomically publish a JSON status or manifest."""
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n')
    temporary.replace(path)


def disk_bytes(folder):
    """Count logical or allocated storage, tolerating concurrent shard renames."""
    result = 0
    for path in folder.rglob('*'):
        try:
            if path.is_file():
                stat=path.stat(); result+=max(stat.st_size,stat.st_blocks*512)
        except FileNotFoundError:
            pass
    return result


def stop_requested(signum, frame):
    """Request orderly termination on a process signal."""
    global STOP
    STOP = True


def run_stage(command, log, status, publish, phase, output):
    """Run a verification subprocess while publishing liveness and failures."""
    with log.open('a') as stream:
        child = subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
        try:
            while child.poll() is None:
                if STOP or (output/'stop.requested').exists():
                    raise RuntimeError('Stop requested')
                status.update(phase=phase,stage_log=str(log))
                publish()
                time.sleep(5)
            if child.returncode:
                raise RuntimeError('{} failed ({}); see {}'.format(phase,child.returncode,log))
        finally:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill(); child.wait()


def supervise(args):
    """Own workers and persist all state independently of model sampling."""
    output=args.output.resolve(); logdir=args.logdir.resolve()
    output.mkdir(parents=True,exist_ok=True); logdir.mkdir(parents=True,exist_ok=True)
    lock=(output/'pipeline.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    signal.signal(signal.SIGTERM,stop_requested); signal.signal(signal.SIGINT,stop_requested)
    begun=time.time()
    status={'phase':'starting','target_frames':args.scenes*args.frames_per_scene,
            'completed_scenes':0,'acquired_frames_lower_bound':0,'started_at_unix':begun,
            'supervisor_pid_in_namespace':os.getpid(),'output':str(output),'logs':str(logdir)}
    counts={}; active={}; retries={}

    def publish():
        """Publish a heartbeat and conservative progress without opening live HDF5."""
        progress=counts.copy()
        for scene,record in active.items():
            path=record[2]
            if path.exists():
                with path.open('rb') as stream:
                    stream.seek(max(0,path.stat().st_size-16000))
                    matches=re.findall(rb'scene_\d+ (\d+)/(\d+) coverage=',stream.read())
                if matches:
                    progress[scene]=int(matches[-1][0])+1
        memory={}
        for scene,(child,_,_) in active.items():
            try:
                info=Path('/proc/{}/status'.format(child.pid)).read_text()
                memory[str(scene)]={key:int(re.search(r'^'+key+r':\s+(\d+)',info,re.M).group(1))
                                    for key in ('VmRSS','VmHWM') if re.search(r'^'+key+':',info,re.M)}
            except FileNotFoundError:
                pass
        status.update(heartbeat_unix=time.time(),elapsed_s=time.time()-begun,
            completed_scenes=len(counts),active_scenes=sorted(active),
            acquired_frames_lower_bound=sum(progress.values()),dataset_bytes=disk_bytes(output),
            worker_memory_kib=memory,scene_retries=retries)
        atomic_json(output/'status.json',status)

    try:
        if args.pilot_dataset is not None:
            pilot=args.pilot_dataset.resolve()
            deadline=time.monotonic()+7200
            status.update(phase='waiting_for_pilot',pilot_dataset=str(pilot))
            while True:
                if STOP or (output/'stop.requested').exists():
                    raise RuntimeError('Stop requested while waiting for pilot')
                path=pilot/'status.json'
                if path.exists():
                    pilot_status=json.loads(path.read_text())
                    status['pilot_phase']=pilot_status['phase']
                    if pilot_status['phase']=='failed':
                        raise RuntimeError('Pilot failed: '+pilot_status.get('error','see pilot logs'))
                    if pilot_status['phase']=='ready_for_training':
                        result=json.loads((pilot/'result.json').read_text())
                        if result['total_frames']!=120 or result['audited_frames']!=3:
                            raise RuntimeError('Unexpected pilot verification count')
                        pilot_settings=json.loads((pilot/'collection_settings.json').read_text())
                        for key in ('world_size','boxes','seed','channels','azimuth_step'):
                            if pilot_settings[key]!=getattr(args,key):
                                raise RuntimeError('Pilot setting differs from production: '+key)
                        if pilot_settings.get('assets') != (str(args.assets) if args.assets else None):
                            raise RuntimeError('Pilot assets differ from production')
                        status['pilot_verified']=True
                        break
                if time.monotonic()>deadline:
                    raise TimeoutError('Pilot did not finish within two hours')
                publish(); time.sleep(5)
        settings={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()
                  if k not in ('output','logdir','jobs','pilot_dataset') and not (k=='assets' and v is None)}
        setting_path=output/'collection_settings.json'
        if setting_path.exists() and json.loads(setting_path.read_text())!=settings:
            raise RuntimeError('Output settings differ; use a fresh dataset directory')
        atomic_json(setting_path,settings)
        # Workers and validators use immutable copies even if the project is edited.
        snapshot=logdir/'source'
        files=['collect_dataset_v4.py','collect_dataset_solids.py','collect_dataset_v3.py','validate_dataset_v3.py','audit_dataset_v3.py']
        if not snapshot.exists():
            (snapshot/'scripts').mkdir(parents=True)
            for name in files:
                shutil.copy2(ROOT/'scripts'/name,snapshot/'scripts'/name)
            shutil.copytree(ROOT/'neural_scene',snapshot/'neural_scene',ignore=shutil.ignore_patterns('__pycache__'))
        sys.path.insert(0,str(snapshot/'scripts'))
        import collect_dataset_v3 as base
        provenance_args=argparse.Namespace(**vars(args)); del provenance_args.logdir; del provenance_args.pilot_dataset
        if provenance_args.assets is not None:
            provenance_args.assets=str(provenance_args.assets)
        else:
            del provenance_args.assets
        provenance=base.provenance(provenance_args)
        if args.assets is not None:
            assets=output/'source_assets'
            if not assets.exists():
                shutil.copytree(args.assets,assets)
            provenance['asset_sources']={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                                         for p in sorted(assets.glob('*.obj'))}
            if provenance['asset_sources']!={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                                           for p in sorted(args.assets.glob('*.obj'))}:
                raise RuntimeError('Pinned assets differ from source assets')
        provenance['pipeline_sources']={str(p.relative_to(snapshot)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in snapshot.rglob('*.py')}
        provenance_path=output/'provenance.json'
        if provenance_path.exists() and json.loads(provenance_path.read_text())!=provenance:
            raise RuntimeError('Sources/settings changed since previous acquisition')
        atomic_json(provenance_path,provenance)
        os.environ['PYTHONPATH']=str(snapshot)+os.pathsep+os.environ.get('PYTHONPATH','')
        indices=list(range(args.start_scene,args.start_scene+args.scenes))
        pending=[]
        for scene in indices:
            path=output/'sequences'/('scene_{:04d}'.format(scene))/'sequence.json'
            if path.exists():
                record=json.loads(path.read_text())
                if len(record['frames'])!=args.frames_per_scene:
                    raise RuntimeError('Closed scene has unexpected frame count')
                counts[scene]=len(record['frames'])
            else:
                pending.append(scene)
        status['phase']='collecting'; publish()
        while pending or active:
            if STOP or (output/'stop.requested').exists():
                raise RuntimeError('Stop requested; partial shards retained')
            if disk_bytes(output)+512*1024**2 >= args.max_bytes:
                raise RuntimeError('Dataset storage cap reserve reached; partial shards retained')
            while pending and len(active)<args.jobs:
                scene=pending.pop(0)
                path=logdir/('scene_{:04d}.log'.format(scene))
                stream=path.open('a')
                command=[sys.executable,'-u',str(snapshot/'scripts/collect_dataset_v4.py'),
                    '--output',str(output),'--scene',str(scene),'--frames-per-scene',str(args.frames_per_scene),
                    '--world-size',str(args.world_size),'--boxes',str(args.boxes),'--seed',str(args.seed),
                    '--max-bytes',str(args.max_bytes),'--channels',str(args.channels),
                    '--azimuth-step',str(args.azimuth_step)]
                if args.assets is not None:
                    command.extend(['--assets',str(output/'source_assets')])
                child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
                active[scene]=(child,stream,path)
            for scene,(child,stream,path) in list(active.items()):
                if child.poll() is not None:
                    stream.close(); del active[scene]
                    if child.returncode:
                        if child.returncode == -signal.SIGKILL and retries.get(scene,0)<2:
                            retries[scene]=retries.get(scene,0)+1
                            folder=output/'sequences'/('scene_{:04d}'.format(scene))
                            # A killed shard is never appended to or counted as collected.
                            # Retain the entire attempt for inspection outside the dataset.
                            if folder.exists():
                                recovery=logdir/'interrupted_scenes'
                                recovery.mkdir(exist_ok=True)
                                folder.rename(recovery/('scene_{:04d}_attempt_{}_{}'.format(
                                    scene,retries[scene],time.time_ns())))
                            pending.insert(0,scene)
                            # Reduce concurrency after a kill, even for other v4 callers.
                            args.jobs=1
                            print('Scene {} was SIGKILLed; retry {} with one worker'.format(
                                scene,retries[scene]),flush=True)
                            continue
                        raise RuntimeError('Scene {} failed ({}); see {}'.format(scene,child.returncode,path))
                    record=json.loads((output/'sequences'/('scene_{:04d}'.format(scene))/'sequence.json').read_text())
                    counts[scene]=len(record['frames'])
                    print('Scene {} closed: {}/{} frames'.format(scene,sum(counts.values()),status['target_frames']),flush=True)
            publish(); time.sleep(5)
        sequences=[json.loads((output/'sequences'/('scene_{:04d}'.format(i))/'sequence.json').read_text()) for i in indices]
        manifest={'format':'neural-scene-packed-v1','input_policy':'aligned_temporal_t01',
            'grid':{'voxel_size':.05,'shape':[64,64,64],'origin':[-1.6,-1.6,-1.6]},
            'split_policy':'whole scene: ID modulo 10 = 0 validation, 1 test, otherwise train',
            'max_dataset_bytes':args.max_bytes,'sequences':sequences}
        atomic_json(output/'dataset.json',manifest)
        summary={'status':'collected','total_frames':sum(counts.values()),'unique_acquired_poses':sum(s['unique_pose_count'] for s in sequences),
            'split_frames':{name:sum(len(s['frames']) for s in sequences if s['split']==name) for name in ('train','validation','test')},
            'world_size_m':[args.world_size,args.world_size],'boxes_per_scene':args.boxes,
            'duration_s_per_scene':args.frames_per_scene/10.,'dynamic_objects':False}
        if args.assets is not None:
            summary.update(geometry='native basic solids and randomized boxes',
                           objects_per_scene=args.boxes,asset_types=len(provenance['asset_sources']))
            summary.pop('boxes_per_scene')
        atomic_json(output/'collection_summary.json',summary)
        del sequences,manifest
        run_stage([sys.executable,str(snapshot/'scripts/validate_dataset_v3.py'),'--dataset',str(output),
                   '--expected-frames',str(status['target_frames'])],logdir/'validation.log',status,publish,'validating',output)
        run_stage([sys.executable,'-u',str(snapshot/'scripts/audit_dataset_v3.py'),'--dataset',str(output)],
                  logdir/'audit.log',status,publish,'auditing',output)
        validation=json.loads((output/'validation_report.json').read_text())
        audit=json.loads((output/'sample_audit.json').read_text())
        if validation['status']!='validated' or validation['total_frames']!=status['target_frames'] or audit['status']!='passed' or audit['frames']!=3*args.scenes:
            raise RuntimeError('Unexpected final verification report')
        changed=[p for p,h in provenance['sources'].items() if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h]
        if changed:
            raise RuntimeError('External source files changed during acquisition: '+str(changed))
        if disk_bytes(output)>=args.max_bytes or list(output.rglob('*.partial')):
            raise RuntimeError('Final storage/partial-shard check failed')
        summary.update(status='ready_for_training',dataset_bytes=disk_bytes(output),
            validation=validation,audited_frames=audit['frames'],elapsed_s=time.time()-begun)
        atomic_json(output/'result.json',summary)
        status.update(phase='ready_for_training',result=str(output/'result.json')); publish()
        (logdir/'RESULTS.md').write_text('# Wide-terrain collection completed\n\n```json\n'+json.dumps(summary,indent=2)+'\n```\n')
        print(json.dumps(summary,indent=2),flush=True)
    except BaseException as error:
        status.update(phase='failed',error=str(error))
        publish(); traceback.print_exc()
        raise
    finally:
        for child,stream,path in active.values():
            if child.poll() is None:
                child.terminate()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill(); child.wait()
            stream.close()


def main():
    """Start the collection/verification supervisor with a fixed storage ceiling."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/training_dataset_v4')
    parser.add_argument('--logdir',type=Path,default=ROOT/'project_log/dataset_v4/production')
    parser.add_argument('--pilot-dataset',type=Path,help='Wait for a matching 120-frame pilot validation/audit before collection')
    parser.add_argument('--scenes',type=int,default=40)
    parser.add_argument('--frames-per-scene',type=int,default=10000)
    parser.add_argument('--start-scene',type=int,default=0)
    parser.add_argument('--jobs',type=int,default=4)
    parser.add_argument('--world-size',type=float,default=40.)
    parser.add_argument('--boxes',type=int,default=160)
    parser.add_argument('--assets',type=Path,help='Directory of native basic-solid OBJ assets')
    parser.add_argument('--seed',type=int,default=20261109)
    parser.add_argument('--channels',type=int,default=32)
    parser.add_argument('--azimuth-step',type=float,default=3.)
    parser.add_argument('--device',default='cpu',choices=['cpu'])
    parser.add_argument('--max-bytes',type=int,default=10000000000)
    args=parser.parse_args()
    if min(args.scenes,args.jobs,args.boxes)<1 or args.frames_per_scene<12 or args.world_size<20 or not 0<args.max_bytes<=10000000000:
        parser.error('Invalid counts, terrain dimensions or storage ceiling')
    supervise(args)


if __name__=='__main__':
    main()
