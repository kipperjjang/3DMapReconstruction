#!/usr/bin/env bash
set -euo pipefail
NEURAL_SCENE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${NEURAL_SCENE_ROOT}:${NEURAL_SCENE_DEPS:-${NEURAL_SCENE_ROOT}/.deps}:${NEURAL_SCENE_ROOT}/.deps-tensorboard${PYTHONPATH:+:${PYTHONPATH}}"
source "${NEURAL_SCENE_ROOT}/scripts/cuda_env.sh"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
exec bash "${NEURAL_SCENE_ROOT}/scripts/conda_python.sh" "${NEURAL_SCENE_CONDA_ENV:-sp_conv}" "$@"
