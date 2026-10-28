#!/usr/bin/env bash
set -euo pipefail
NEURAL_SCENE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${NEURAL_SCENE_ROOT}:${NEURAL_SCENE_DEPS:-${NEURAL_SCENE_ROOT}/.deps}:${NEURAL_SCENE_ROOT}/.deps-tensorboard${PYTHONPATH:+:${PYTHONPATH}}"
NEURAL_SCENE_CUDA_LIB="${NEURAL_SCENE_CUDA_HOME:-/home/intern/cuda-11.7}/lib64"
if [[ -d "${NEURAL_SCENE_CUDA_LIB}" ]]; then
  export LD_LIBRARY_PATH="${NEURAL_SCENE_CUDA_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
exec "${NEURAL_SCENE_PYTHON:-/home/intern/anaconda3/envs/sp_conv/bin/python}" "$@"
