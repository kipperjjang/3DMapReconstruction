#!/usr/bin/env bash
set -euo pipefail
NEURAL_SCENE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "${NEURAL_SCENE_ROOT}"
export NEURAL_SCENE_DYNA_DEPS="${NEURAL_SCENE_ROOT}/.deps-dyna-cuda"
export LD_LIBRARY_PATH="/home/intern/cuda-11.7/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
if [[ " ${*} " == *" --help "* ]]; then
  exec bash scripts/run_dyna.sh -m neural_scene.experiment --help
fi
NEURAL_SCENE_DATASET="outputs/training_dataset_v3"
NEURAL_SCENE_RUN_OUTPUT="outputs/training_run_v3"
training_arguments=("$@")
for ((i=0; i<${#training_arguments[@]}; i++)); do
  case "${training_arguments[i]}" in
    --dataset) NEURAL_SCENE_DATASET="${training_arguments[i+1]}" ;;
    --dataset=*) NEURAL_SCENE_DATASET="${training_arguments[i]#--dataset=}" ;;
    --output) NEURAL_SCENE_RUN_OUTPUT="${training_arguments[i+1]}" ;;
    --output=*) NEURAL_SCENE_RUN_OUTPUT="${training_arguments[i]#--output=}" ;;
  esac
done
mkdir -p "${NEURAL_SCENE_RUN_OUTPUT}"
exec > >(tee -a "${NEURAL_SCENE_RUN_OUTPUT}/console.log") 2>&1
if [[ ! -f "${NEURAL_SCENE_DATASET}/dataset.json" ]]; then
  echo "Collection is not complete: ${NEURAL_SCENE_DATASET}/dataset.json is missing." >&2
  echo "Check progress with: bash scripts/run_dyna.sh scripts/status_dataset_v3.py" >&2
  exit 2
fi
bash scripts/run_dyna.sh scripts/check_gpu.py --require-cuda
bash scripts/run_dyna.sh scripts/validate_dataset_v3.py --dataset "${NEURAL_SCENE_DATASET}"
bash scripts/run_dyna.sh -m neural_scene smoke --device cuda:0
exec bash scripts/run_dyna.sh -m neural_scene.experiment \
  --dataset "${NEURAL_SCENE_DATASET}" --output "${NEURAL_SCENE_RUN_OUTPUT}" \
  --device cuda:0 --updates 20000 --batch-size 2 --rollout 12 --eval-every 1000 \
  --position-noise 0.005 --initial-lr 0.01 --final-lr 0.0001 --seed 20261028 "$@"
