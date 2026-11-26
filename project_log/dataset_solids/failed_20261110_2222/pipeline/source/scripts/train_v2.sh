#!/usr/bin/env bash
set -euo pipefail
NEURAL_SCENE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "${NEURAL_SCENE_ROOT}"
duration_arguments=(--updates 1500)
for argument in "$@"; do
  case "${argument}" in
    --epochs|--epochs=*|--updates|--updates=*) duration_arguments=() ;;
  esac
done
exec bash scripts/train_gpu.sh \
  --dataset outputs/training_dataset_v2 --output outputs/training_run_v2 \
  "${duration_arguments[@]}" --batch-size 2 --rollout 12 --eval-every 100 \
  --position-noise 0.005 --initial-lr 0.01 --final-lr 0.0001 \
  --seed 20261020 "$@"
