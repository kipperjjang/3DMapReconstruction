#!/usr/bin/env bash
set -euo pipefail
NEURAL_SCENE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONDONTWRITEBYTECODE=1
sim_root="${DYNARAYSIM_ROOT:-${NEURAL_SCENE_ROOT}/../DynaRaySim}"
mapping_root="${DYNAMAPPING_ROOT:-${NEURAL_SCENE_ROOT}/../DynaMapping}"
[[ -d "${sim_root}/torch/dynaraysim" ]] && sim_root="${sim_root}/torch"
[[ -d "${mapping_root}/torch/dynamapping" ]] && mapping_root="${mapping_root}/torch"
default_deps="${NEURAL_SCENE_ROOT}/.deps-dyna"
if [[ ! -d "${default_deps}/MinkowskiEngine" && -d "${NEURAL_SCENE_ROOT}/.deps-dyna-cuda/MinkowskiEngine" ]]; then
  default_deps="${NEURAL_SCENE_ROOT}/.deps-dyna-cuda"
fi
export PYTHONPATH="${NEURAL_SCENE_ROOT}:${NEURAL_SCENE_DYNA_DEPS:-${default_deps}}:${NEURAL_SCENE_ROOT}/.deps-dyna:${NEURAL_SCENE_ROOT}/.deps-tensorboard:${sim_root}:${mapping_root}${PYTHONPATH:+:${PYTHONPATH}}"
source "${NEURAL_SCENE_ROOT}/scripts/cuda_env.sh"
export DYNARAYSIM_WARP_CACHE="${NEURAL_SCENE_ROOT}/.build/warp-dyna"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
exec bash "${NEURAL_SCENE_ROOT}/scripts/conda_python.sh" "${NEURAL_SCENE_CONDA_ENV:-dyna}" "$@"
