#!/usr/bin/env bash
# Optional toolkit libraries; PyTorch may also provide its own CUDA runtime.
cuda_root="${NEURAL_SCENE_CUDA_HOME:-${CUDA_HOME:-${CUDA_PATH:-}}}"
if [[ -z "${cuda_root}" ]] && command -v nvcc >/dev/null 2>&1; then
  cuda_root="$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")"
fi
if [[ -z "${cuda_root}" && -d /usr/local/cuda ]]; then
  cuda_root=/usr/local/cuda
fi
if [[ -n "${cuda_root}" && -d "${cuda_root}/lib64" ]]; then
  export LD_LIBRARY_PATH="${cuda_root}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi
