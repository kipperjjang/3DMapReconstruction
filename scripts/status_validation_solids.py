"""Show progress without opening an actively written HDF5 file."""

import json
from pathlib import Path
import re

root = Path(__file__).resolve().parents[1]
dataset = root/'outputs/validation_basic_solids'
status = json.loads((dataset/'status.json').read_text())
log = root/'project_log/validation_basic_solids/collection.log'
matches = re.findall(r'scene_0000 (\d+)/(\d+) coverage=',log.read_text()) if log.exists() else []
status['acquired_frames_lower_bound'] = int(matches[-1][0])+1 if matches else 0
if (dataset/'result.json').exists():
    status['result'] = json.loads((dataset/'result.json').read_text())
    status['acquired_frames_lower_bound'] = status['result']['total_frames']
print(json.dumps(status,indent=2))
