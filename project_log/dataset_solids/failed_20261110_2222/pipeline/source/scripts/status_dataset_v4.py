"""Print wide-terrain job progress and warn when its heartbeat is stale."""

import argparse
import json
from pathlib import Path
import time


def main():
    """Read published progress without touching a live HDF5 shard."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,default=Path(__file__).resolve().parents[1]/'outputs/training_dataset_v4')
    args=parser.parse_args()
    path=args.dataset/'status.json'
    if not path.exists():
        raise SystemExit('No supervisor status yet: '+str(path))
    status=json.loads(path.read_text())
    age=max(0,time.time()-status['heartbeat_unix'])
    status['heartbeat_age_s']=round(age,1)
    status['heartbeat_stale']=age>90 and status['phase'] not in ('ready_for_training','failed')
    print(json.dumps(status,indent=2))


if __name__=='__main__':
    main()
