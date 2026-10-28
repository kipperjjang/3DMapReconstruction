"""Report a conservative acquired-frame count from flushed collection progress."""

import json
from pathlib import Path
import re

root = Path(__file__).resolve().parents[1]
data = root/'outputs/training_dataset_v3'
log = root/'project_log/dataset_v3/collection.log'
counts = {}
for log in [log]+sorted(log.parent.glob('tail_scene_*.log')):
    if not log.exists():
        continue
    for name, index, total in re.findall(r'(scene_\d+) (\d+)/(\d+) coverage=', log.read_text()):
        counts[name] = int(index)+1
for path in data.glob('sequences/*/sequence.json'):
    record=json.loads(path.read_text())
    counts[record['scene']]=len(record['frames'])
size=0
for path in data.rglob('*'):
    try:
        if path.is_file():
            stat=path.stat(); size+=max(stat.st_size,stat.st_blocks*512)
    except FileNotFoundError:
        pass
print(json.dumps({'acquired_frames_lower_bound':sum(counts.values()),'target_frames':200000,
    'completed_scenes':sum(n==5000 for n in counts.values()),'started_scenes':len(counts),
    'dataset_bytes':size,'max_dataset_bytes':10000000000,'frames_by_scene':counts},indent=2))
