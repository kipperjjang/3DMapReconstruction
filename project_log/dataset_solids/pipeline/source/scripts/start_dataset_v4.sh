#!/usr/bin/env bash
set -euo pipefail
NEURAL_SCENE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "${NEURAL_SCENE_ROOT}"
mkdir -p project_log/dataset_v4/production

# The managed terminal keeps its own process alive after the assistant returns.
# A normal host terminal can use the default detached nohup mode instead.
if [[ "${1:-}" == "--foreground" ]]; then
  shift
  exec bash scripts/run_dyna.sh -u scripts/run_dataset_v4.py "$@" \
    >> project_log/dataset_v4/production/supervisor.log 2>&1
fi

nohup bash scripts/run_dyna.sh -u scripts/run_dataset_v4.py "$@" \
  >> project_log/dataset_v4/production/supervisor.log 2>&1 < /dev/null &
NEURAL_SCENE_JOB_PID=$!
echo "${NEURAL_SCENE_JOB_PID}" > project_log/dataset_v4/production/launcher.pid
echo "Started dataset supervisor PID ${NEURAL_SCENE_JOB_PID}"
echo "Progress: python scripts/status_dataset_v4.py"
echo "Log: project_log/dataset_v4/production/supervisor.log"
