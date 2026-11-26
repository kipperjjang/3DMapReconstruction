#!/usr/bin/env bash
set -euo pipefail
task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "${task_root}"
mkdir -p project_log/dataset_solids/pipeline
nohup setsid bash scripts/run_dyna.sh -u scripts/run_solids_pipeline.py \
  --pilot "${task_root}/outputs/dataset_solids_pilot_v3" "$@" \
  >> project_log/dataset_solids/pipeline/launcher.log 2>&1 < /dev/null &
task_pid=$!
echo "${task_pid}" > project_log/dataset_solids/pipeline/launcher.pid
echo "Started collection and training pipeline PID ${task_pid}"
