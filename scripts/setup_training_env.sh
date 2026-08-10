#!/usr/bin/env bash
# Create or validate the isolated Qwen3 rollout / GRPO environment.
#
# Usage:
#   bash scripts/setup_training_env.sh
#   LIVEMCP_ENV=/path/to/env bash scripts/setup_training_env.sh
#   bash scripts/setup_training_env.sh --check
#
# This environment intentionally does not install the Gemma Teacher stack.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_ENV_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)/envs/livemcp"
LIVEMCP_ENV="${LIVEMCP_ENV:-${DEFAULT_ENV_ROOT}}"
MODE="install"
export PYTHONNOUSERSITE=1
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/oval_triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/tmp/oval_torchinductor}"
mkdir -p "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}"

usage() {
    cat <<'EOF'
Usage: bash scripts/setup_training_env.sh [--check] [--help]

Options:
  --check  Validate an existing training environment without installing.
  --help   Show this help message.

Environment:
  LIVEMCP_ENV  Conda prefix to create or validate.
               Default: <project-parent>/envs/livemcp
  CONDA_EXE    Conda executable. Defaults to the conda found on PATH.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --check)
            MODE="check"
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "ERROR: unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

PYTHON_BIN="${LIVEMCP_ENV}/bin/python"

create_environment() {
    if [[ -x "${PYTHON_BIN}" ]]; then
        echo "[setup] Reusing existing environment: ${LIVEMCP_ENV}"
        return
    fi

    local conda_exe="${CONDA_EXE:-}"
    if [[ -z "${conda_exe}" ]]; then
        conda_exe="$(command -v conda || true)"
    fi
    if [[ -z "${conda_exe}" || ! -x "${conda_exe}" ]]; then
        echo "ERROR: conda was not found. Set CONDA_EXE or add conda to PATH." >&2
        exit 1
    fi

    echo "[setup] Creating Python 3.11 environment: ${LIVEMCP_ENV}"
    "${conda_exe}" create -p "${LIVEMCP_ENV}" python=3.11 pip -y
}

install_dependencies() {
    echo "[setup] Installing pinned rollout / GRPO dependencies"
    PYTHONNOUSERSITE=1 "${PYTHON_BIN}" -m pip install \
        -r "${PROJECT_ROOT}/requirements-train.txt"

    echo "[setup] Installing vendored verl in editable mode"
    PYTHONNOUSERSITE=1 "${PYTHON_BIN}" -m pip install \
        -e "${PROJECT_ROOT}/verl"

    echo "[setup] Installing LiveMCP-GRPO in editable mode"
    PYTHONNOUSERSITE=1 "${PYTHON_BIN}" -m pip install \
        -e "${PROJECT_ROOT}" --no-deps
}

validate_environment() {
    if [[ ! -x "${PYTHON_BIN}" ]]; then
        echo "ERROR: training environment does not exist: ${LIVEMCP_ENV}" >&2
        exit 1
    fi

    echo "[check] Running pip dependency validation"
    PYTHONNOUSERSITE=1 "${PYTHON_BIN}" -m pip check

    echo "[check] Validating imports and pinned versions"
    PYTHONNOUSERSITE=1 "${PYTHON_BIN}" - <<'PY'
import importlib
import importlib.metadata as metadata
import sys

import torch

expected = {
    "torch": "2.8.0",
    "vllm": "0.11.0",
    "transformers": "4.57.1",
    "tokenizers": "0.22.1",
    "ray": "2.54.1",
    "verl": "0.6.1",
    "flash-attn": "2.8.3",
}

modules = {
    "torch": "torch",
    "vllm": "vllm",
    "transformers": "transformers",
    "tokenizers": "tokenizers",
    "ray": "ray",
    "verl": "verl",
    "flash-attn": "flash_attn",
}

errors = []
print(f"python={sys.version.split()[0]} executable={sys.executable}")
for distribution, expected_version in expected.items():
    actual_version = metadata.version(distribution)
    importlib.import_module(modules[distribution])
    print(f"{distribution}={actual_version}")
    if actual_version != expected_version:
        errors.append(
            f"{distribution}: expected {expected_version}, got {actual_version}"
        )

print(f"torch_cuda={torch.version.cuda}")
print(f"cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"gpu={torch.cuda.get_device_name(0)}")
    print(f"compute_capability={torch.cuda.get_device_capability(0)}")

if sys.version_info[:2] != (3, 11):
    errors.append(
        f"python: expected 3.11, got {sys.version_info.major}.{sys.version_info.minor}"
    )
if torch.version.cuda != "12.8":
    errors.append(f"torch CUDA runtime: expected 12.8, got {torch.version.cuda}")
if not torch.cuda.is_available():
    errors.append("CUDA is not available to PyTorch")

if errors:
    raise SystemExit("environment validation failed:\n- " + "\n- ".join(errors))
PY

    echo "[check] Training environment is ready"
}

if [[ "${MODE}" == "install" ]]; then
    create_environment
    install_dependencies
fi

validate_environment

cat <<EOF

Activate with:
  conda activate "${LIVEMCP_ENV}"
  export PYTHON_BIN="${PYTHON_BIN}"
  export PYTHONNOUSERSITE=1
  export TRITON_CACHE_DIR="${TRITON_CACHE_DIR}"
  export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR}"

Start training with:
  bash scripts/train_grpo.sh
EOF
