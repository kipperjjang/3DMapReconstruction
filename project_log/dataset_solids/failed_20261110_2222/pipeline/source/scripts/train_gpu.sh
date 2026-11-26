#!/usr/bin/env bash
set -euo pipefail
NEURAL_SCENE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "${NEURAL_SCENE_ROOT}"
export NEURAL_SCENE_DEPS="${NEURAL_SCENE_ROOT}/.deps-cuda"
if [[ " ${*} " == *" --help "* ]]; then
  exec bash scripts/run.sh -m neural_scene.experiment --help
fi
NEURAL_SCENE_RUN_OUTPUT="outputs/training_run_v1"
NEURAL_SCENE_DATASET="outputs/training_dataset_v1"
training_arguments=("$@")
for ((i=0; i<${#training_arguments[@]}; i++)); do
  if [[ "${training_arguments[i]}" == "--output" ]] && ((i+1<${#training_arguments[@]})); then
    NEURAL_SCENE_RUN_OUTPUT="${training_arguments[i+1]}"
  elif [[ "${training_arguments[i]}" == --output=* ]]; then
    NEURAL_SCENE_RUN_OUTPUT="${training_arguments[i]#--output=}"
  elif [[ "${training_arguments[i]}" == "--dataset" ]] && ((i+1<${#training_arguments[@]})); then
    NEURAL_SCENE_DATASET="${training_arguments[i+1]}"
  elif [[ "${training_arguments[i]}" == --dataset=* ]]; then
    NEURAL_SCENE_DATASET="${training_arguments[i]#--dataset=}"
  fi
done
mkdir -p "${NEURAL_SCENE_RUN_OUTPUT}"
exec > >(tee -a "${NEURAL_SCENE_RUN_OUTPUT}/console.log") 2>&1
bash scripts/run.sh scripts/check_gpu.py --require-cuda
if [[ ! -f "${NEURAL_SCENE_DATASET}/dataset.json" ]]; then
  if [[ "${NEURAL_SCENE_DATASET}" == "outputs/training_dataset_v1" ]]; then
    bash scripts/run_dyna.sh scripts/prepare_training_data.py --device cuda:0
  else
    echo "Dataset is not prepared: ${NEURAL_SCENE_DATASET}/dataset.json" >&2
    exit 1
  fi
fi
bash scripts/run.sh scripts/validate_training_data.py --dataset "${NEURAL_SCENE_DATASET}"
bash scripts/run.sh -m neural_scene smoke --device cuda:0
exec bash scripts/run.sh -m neural_scene.experiment \
  --dataset "${NEURAL_SCENE_DATASET}" --output "${NEURAL_SCENE_RUN_OUTPUT}" \
  --device cuda:0 "$@"
