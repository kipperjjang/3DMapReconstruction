#!/usr/bin/env bash
# Execute Python in a named conda environment without assuming its install path.
set -euo pipefail
env_name="$1"
shift
if [[ -n "${NEURAL_SCENE_PYTHON:-}" ]]; then
  exec "${NEURAL_SCENE_PYTHON}" "$@"
fi
if [[ "${CONDA_DEFAULT_ENV:-}" == "${env_name}" && -n "${CONDA_PREFIX:-}" ]]; then
  exec "${CONDA_PREFIX}/bin/python" "$@"
fi
if [[ -n "${CONDA_HOME:-}" && -x "${CONDA_HOME}/bin/conda" ]]; then
  conda_command="${CONDA_HOME}/bin/conda"
elif [[ -n "${CONDA_EXE:-}" && -x "${CONDA_EXE}" ]]; then
  conda_command="${CONDA_EXE}"
elif command -v conda >/dev/null 2>&1; then
  conda_command="$(command -v conda)"
else
  echo "Conda not found. Set CONDA_HOME, activate ${env_name}, or set NEURAL_SCENE_PYTHON." >&2
  exit 1
fi
exec "${conda_command}" run --no-capture-output -n "${env_name}" python "$@"
