#!/usr/bin/env bash
set -euo pipefail
NEURAL_SCENE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "${NEURAL_SCENE_ROOT}"
export NEURAL_SCENE_DEPS="${NEURAL_SCENE_ROOT}/.deps-cuda"
if [[ " ${*} " == *" --help "* ]]; then
  exec bash scripts/run.sh -m neural_scene.experiment --help
fi
NEURAL_SCENE_RUN_OUTPUT="outputs/training_run_v1"
training_arguments=("$@")
for ((i=0; i<${#training_arguments[@]}; i++)); do
  if [[ "${training_arguments[i]}" == "--output" ]] && ((i+1<${#training_arguments[@]})); then
    NEURAL_SCENE_RUN_OUTPUT="${training_arguments[i+1]}"
  elif [[ "${training_arguments[i]}" == --output=* ]]; then
    NEURAL_SCENE_RUN_OUTPUT="${training_arguments[i]#--output=}"
  fi
done
mkdir -p "${NEURAL_SCENE_RUN_OUTPUT}"
exec > >(tee -a "${NEURAL_SCENE_RUN_OUTPUT}/console.log") 2>&1
bash scripts/run.sh scripts/check_gpu.py --require-cuda
if [[ ! -f outputs/training_dataset_v1/dataset.json ]]; then
  /home/intern/DynaRaySim/.venv/bin/python scripts/prepare_training_data.py --device cuda:0
fi
bash scripts/run.sh scripts/validate_training_data.py
bash scripts/run.sh -m neural_scene smoke --device cuda:0
exec bash scripts/run.sh -m neural_scene.experiment \
  --dataset outputs/training_dataset_v1 --output outputs/training_run_v1 \
  --device cuda:0 "$@"
