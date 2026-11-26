#!/usr/bin/env bash
set -euo pipefail
NEURAL_SCENE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "${NEURAL_SCENE_ROOT}"
export NEURAL_SCENE_DYNA_DEPS="${NEURAL_SCENE_DYNA_DEPS:-${NEURAL_SCENE_ROOT}/.deps-dyna-cuda}"
if [[ " ${*} " == *" --help "* ]]; then
  exec bash scripts/run_dyna.sh -m neural_scene.experiment --help
fi
NEURAL_SCENE_DATASET="outputs/training_dataset_v4_part1"
NEURAL_SCENE_RUN_OUTPUT="outputs/training_run_v4_part1"
training_arguments=("$@")
duration_arguments=(--updates 20000)
for ((i=0; i<${#training_arguments[@]}; i++)); do
  case "${training_arguments[i]}" in
    --epochs|--epochs=*|--updates|--updates=*) duration_arguments=() ;;
    --dataset) if (( i+1 >= ${#training_arguments[@]} )) || [[ "${training_arguments[i+1]}" == --* ]]; then
      echo "--dataset requires a path" >&2; exit 2
    fi; NEURAL_SCENE_DATASET="${training_arguments[i+1]}" ;;
    --dataset=*) NEURAL_SCENE_DATASET="${training_arguments[i]#--dataset=}" ;;
    --output) if (( i+1 >= ${#training_arguments[@]} )) || [[ "${training_arguments[i+1]}" == --* ]]; then
      echo "--output requires a path" >&2; exit 2
    fi; NEURAL_SCENE_RUN_OUTPUT="${training_arguments[i+1]}" ;;
    --output=*) NEURAL_SCENE_RUN_OUTPUT="${training_arguments[i]#--output=}" ;;
  esac
done
mkdir -p "${NEURAL_SCENE_RUN_OUTPUT}"
exec > >(tee -a "${NEURAL_SCENE_RUN_OUTPUT}/console.log") 2>&1
if [[ ! -f "${NEURAL_SCENE_DATASET}/dataset.json" ]]; then
  echo "Collection is not complete: ${NEURAL_SCENE_DATASET}/dataset.json is missing." >&2
  echo "See DATASET_V4_GUIDE.md for collection and partition preparation." >&2
  exit 2
fi
bash scripts/run_dyna.sh scripts/check_gpu.py --require-cuda
NEURAL_SCENE_EXPECTED_FRAMES="$(bash scripts/run_dyna.sh -c '
import json, sys
from pathlib import Path
settings = Path(sys.argv[1]) / "collection_settings.json"
config = json.loads(settings.read_text()) if settings.exists() else None
print(config["scenes"] * config["frames_per_scene"] if config is not None else 200000)
' "${NEURAL_SCENE_DATASET}")"
bash scripts/run_dyna.sh scripts/validate_dataset.py --dataset "${NEURAL_SCENE_DATASET}" \
  --expected-frames "${NEURAL_SCENE_EXPECTED_FRAMES}"
bash scripts/run_dyna.sh -m neural_scene smoke --device cuda:0
exec bash scripts/run_dyna.sh -m neural_scene.experiment \
  --dataset "${NEURAL_SCENE_DATASET}" --output "${NEURAL_SCENE_RUN_OUTPUT}" \
  --device cuda:0 "${duration_arguments[@]}" --batch-size 4 --rollout 12 --eval-every 1000 \
  --position-noise 0.005 --initial-lr 0.01 --final-lr 0.0001 --seed 20261109 "$@"
