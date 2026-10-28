#!/usr/bin/env bash
set -euo pipefail
NEURAL_SCENE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source /home/intern/anaconda3/etc/profile.d/conda.sh
conda activate dyna
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${NEURAL_SCENE_ROOT}:${NEURAL_SCENE_DYNA_DEPS:-${NEURAL_SCENE_ROOT}/.deps-dyna}:${NEURAL_SCENE_ROOT}/.deps-dyna:${NEURAL_SCENE_ROOT}/.deps-tensorboard:/home/intern/DynaRaySim/torch:/home/intern/DynaMapping/torch${PYTHONPATH:+:${PYTHONPATH}}"
export DYNARAYSIM_WARP_CACHE="${NEURAL_SCENE_ROOT}/.build/warp-dyna"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
exec python "$@"
