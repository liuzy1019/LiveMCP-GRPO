#!/bin/bash
# Configure the verified ARL prefix and optionally run one project command.
#
# Usage:
#   bash scripts/run_in_runtime_environment.sh --check
#   bash scripts/run_in_runtime_environment.sh -- bash scripts/generate_data.sh --count 50 --val-count 10
#   source scripts/run_in_runtime_environment.sh

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARL_ENV="${ARL_ENV:-$(cd "${PROJECT_ROOT}/.." && pwd)/envs/arl}"

if [ ! -x "${ARL_ENV}/bin/python" ]; then
    echo "ERROR: ARL Python not found: ${ARL_ENV}/bin/python" >&2
    return 1 2>/dev/null || exit 1
fi

export ARL_ENV
export PYTHON_BIN="${ARL_ENV}/bin/python"
export PYTHONNOUSERSITE=1
export CUDA_HOME="${CUDA_HOME:-${ARL_ENV}}"
export CUDA_PATH="${CUDA_PATH:-${CUDA_HOME}}"
export FLASHINFER_NVCC="${FLASHINFER_NVCC:-${CUDA_HOME}/bin/nvcc}"
export PATH="${ARL_ENV}/bin:${PATH}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-${TMPDIR:-/tmp}/livemcp_vllm_cache}"
export VLLM_CONFIG_ROOT="${VLLM_CONFIG_ROOT:-${TMPDIR:-/tmp}/livemcp_vllm_config}"
export VLLM_NO_USAGE_STATS="${VLLM_NO_USAGE_STATS:-1}"
mkdir -p "${VLLM_CACHE_ROOT}" "${VLLM_CONFIG_ROOT}"

NVIDIA_SITE="${ARL_ENV}/lib/python3.11/site-packages/nvidia"
CUDA_INCLUDE_DIRS=("${CUDA_HOME}/include" "${CUDA_HOME}/targets/x86_64-linux/include")
if [ -d "${NVIDIA_SITE}" ]; then
    while IFS= read -r include_dir; do
        CUDA_INCLUDE_DIRS+=("${include_dir}")
    done < <(find "${NVIDIA_SITE}" -maxdepth 3 -type d -name include | sort)
fi
CUDA_INCLUDE_PATH="$(IFS=:; echo "${CUDA_INCLUDE_DIRS[*]}")"
export CPATH="${CUDA_INCLUDE_PATH}${CPATH:+:${CPATH}}"

CUDA_LIBRARY_DIRS=(
    "${ARL_ENV}/lib"
    "${ARL_ENV}/targets/x86_64-linux/lib"
    "${ARL_ENV}/targets/x86_64-linux/lib/stubs"
)
CUDA_LIBRARY_PATH="$(IFS=:; echo "${CUDA_LIBRARY_DIRS[*]}")"
export LIBRARY_PATH="${CUDA_LIBRARY_PATH}${LIBRARY_PATH:+:${LIBRARY_PATH}}"
export LD_LIBRARY_PATH="${CUDA_LIBRARY_PATH}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

_check_runtime() {
    "${PYTHON_BIN}" - <<'PY'
import importlib
import importlib.metadata as metadata
import sys

import torch

required = (("vllm", "vllm"), ("flashinfer-python", "flashinfer"), ("verl", "verl"))
print(f"python={sys.version.split()[0]} executable={sys.executable}")
print(f"torch={torch.__version__} cuda={torch.version.cuda} available={torch.cuda.is_available()}")
for distribution, module in required:
    version = metadata.version(distribution)
    importlib.import_module(module)
    print(f"{distribution}={version} import=ok")
try:
    print(f"flash-attn={metadata.version('flash-attn')} import=optional")
except metadata.PackageNotFoundError:
    print("flash-attn=not-installed optional=true")
PY
    if [ ! -x "${FLASHINFER_NVCC}" ]; then
        echo "ERROR: nvcc not found: ${FLASHINFER_NVCC}" >&2
        return 1
    fi
    "${FLASHINFER_NVCC}" --version | tail -1
}

if [ "${BASH_SOURCE[0]}" != "$0" ]; then
    _check_runtime
    echo "Runtime environment exported from ${ARL_ENV}"
    return 0
fi

if [ "${1:-}" = "--check" ] || [ "$#" -eq 0 ]; then
    _check_runtime
    exit 0
fi
if [ "${1:-}" = "--" ]; then
    shift
fi
if [ "$#" -eq 0 ]; then
    echo "ERROR: no command provided" >&2
    exit 2
fi
_check_runtime
cd "${PROJECT_ROOT}"
exec "$@"
