"""Detached basic-solid collection, verification, and concurrent CUDA training."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

STOP = False


def atomic_json(path, data):
    temp = path.with_suffix('.json.tmp')
    temp.write_text(json.dumps(data,indent=2)+'\n')
    temp.replace(path)


def request_stop(signum, frame):
    global STOP
    STOP = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--pilot',type=Path,required=True)
    parser.add_argument('--collection-jobs',type=int,default=1,
                        choices=[1],help='Serialize memory-heavy full-scene GT generation')
    args = parser.parse_args()
    root = args.root.resolve()
    logs = root/'project_log/dataset_solids/pipeline'
    dataset = root/'outputs/training_dataset_v4_solids'
    training = root/'outputs/training_run_v4_solids'
    logs.mkdir(parents=True,exist_ok=True)
    lock = (logs/'pipeline.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    signal.signal(signal.SIGTERM,request_stop)
    signal.signal(signal.SIGINT,request_stop)
    status = {'phase':'starting','pid':os.getpid(),'started_at_unix':time.time(),
              'dataset':str(dataset),'training_output':str(training),
              'target_frames':200000,'training_updates':20000,
              'concurrent_training':True}

    def publish(**changes):
        status.update(changes,heartbeat_unix=time.time())
        atomic_json(logs/'status.json',status)

    def check_stop():
        if STOP or (logs/'stop.requested').exists():
            raise RuntimeError('Stop requested')

    snapshot = logs/'source'
    if not snapshot.exists():
        for name in ('scripts','neural_scene'):
            shutil.copytree(root/name,snapshot/name,ignore=shutil.ignore_patterns('__pycache__'))
        hashes = {str(p.relative_to(snapshot)):hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in snapshot.rglob('*') if p.is_file()}
        atomic_json(logs/'source_hashes.json',hashes)
    env = dict(os.environ,NEURAL_SCENE_DYNA_DEPS=str(root/'.deps-dyna-cuda'),
               DYNARAYSIM_ROOT=str(root.parent/'DynaRaySim'),
               DYNAMAPPING_ROOT=str(root.parent/'DynaMapping'),
               NEURAL_SCENE_PYTHON=sys.executable,PYTHONUNBUFFERED='1')
    wrapper = ['bash',str(snapshot/'scripts/run_dyna.sh')]

    def stage(command, phase, logname):
        check_stop()
        logpath = logs/logname
        with logpath.open('a') as stream:
            stream.write('\nSTART '+json.dumps(command)+'\n'); stream.flush()
            child = subprocess.Popen(command,cwd=root,env=env,stdin=subprocess.DEVNULL,
                                     stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            try:
                while child.poll() is None:
                    check_stop()
                    publish(phase=phase,child_pid=child.pid,stage_log=str(logpath))
                    time.sleep(5)
                if child.returncode:
                    raise RuntimeError(f'{phase} failed ({child.returncode}); see {logpath}')
            finally:
                if child.poll() is None:
                    os.killpg(child.pid,signal.SIGTERM)
                    try:
                        child.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid,signal.SIGKILL); child.wait()

    try:
        stage(wrapper+['-m','neural_scene','smoke','--device','cuda:0'],
              'checking_training_gpu','gpu_smoke.log')
        publish(phase='waiting_for_pilot')
        deadline = time.monotonic()+7200
        while True:
            check_stop()
            pilot = json.loads((args.pilot/'status.json').read_text())
            if pilot['phase']=='failed':
                raise RuntimeError('Pilot failed: '+pilot.get('error','see pilot logs'))
            if pilot['phase']=='ready_for_training':
                result = json.loads((args.pilot/'result.json').read_text())
                if result['total_frames']!=120 or result['audited_frames']!=3:
                    raise RuntimeError('Unexpected pilot counts')
                break
            if time.monotonic()>deadline:
                raise RuntimeError('Pilot did not complete within two hours')
            publish(pilot_phase=pilot['phase']); time.sleep(5)
        if not (dataset/'result.json').exists():
            command = wrapper+['-u',str(snapshot/'scripts/run_dataset_v4.py'),
                '--output',str(dataset),'--logdir',str(root/'project_log/dataset_solids/production'),
                '--scenes','20','--frames-per-scene','10000','--jobs',str(args.collection_jobs),
                '--assets',str(root.parent/'DynaRaySim/resources/assets/basic_solid'),
                '--seed','20261110','--pilot-dataset',str(args.pilot.resolve())]
            stage(command,'collecting_and_verifying','collection.log')
        result = json.loads((dataset/'result.json').read_text())
        if (result['status']!='ready_for_training' or result['total_frames']!=200000
                or result['audited_frames']!=60
                or result['split_frames']!={'train':160000,'validation':20000,'test':20000}):
            raise RuntimeError('Production verification/splits did not pass')
        if not (training/'report.json').exists():
            command = ['bash',str(snapshot/'scripts/train_v4.sh'),
                '--dataset',str(dataset),'--output',str(training),
                '--updates','20000','--input-policy','aligned_temporal_t01']
            if (training/'latest.pt').exists():
                command.append('--resume')
            stage(command,'training','training.log')
        report = json.loads((training/'report.json').read_text())
        publish(phase='complete',finished_at_unix=time.time(),report=str(training/'report.json'))
        print(json.dumps(report,indent=2),flush=True)
    except BaseException as error:
        publish(phase='stopped' if STOP else 'failed',error=str(error))
        traceback.print_exc()
        raise


if __name__=='__main__':
    main()
