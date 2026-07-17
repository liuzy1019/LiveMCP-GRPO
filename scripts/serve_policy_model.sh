#!/bin/bash
# Start a vLLM OpenAI-compatible API server for local model serving.
# Used for RL rollout/training（teacher generation 由 generate_data.sh 统一管理）。
#
# Usage:
#   bash scripts/serve_policy_model.sh <model_path> [port] [tp_size] [gpu_ids]
#
# Example:
#   bash scripts/serve_policy_model.sh models/Qwen/Qwen3-4B 8001 2 "0,1"

set -euo pipefail
export PYTHONNOUSERSITE=1
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-${TMPDIR:-/tmp}/livemcp_vllm_cache}"
export VLLM_CONFIG_ROOT="${VLLM_CONFIG_ROOT:-${TMPDIR:-/tmp}/livemcp_vllm_config}"
export VLLM_NO_USAGE_STATS="${VLLM_NO_USAGE_STATS:-1}"
mkdir -p "${VLLM_CACHE_ROOT}" "${VLLM_CONFIG_ROOT}"

if [ -z "${PYTHON_BIN:-}" ]; then
  if [ -n "${CONDA_PREFIX:-}" ] && [ -x "${CONDA_PREFIX}/bin/python" ]; then
    PYTHON_BIN="${CONDA_PREFIX}/bin/python"
  else
    PYTHON_BIN="$(command -v python3 || command -v python)"
  fi
fi
PYTHON_BIN_DIR="$(cd "$(dirname "${PYTHON_BIN}")" && pwd)"
VLLM_BIN="${PYTHON_BIN_DIR}/vllm"
if [ ! -x "${VLLM_BIN}" ]; then
  echo "ERROR: vLLM CLI not found beside ${PYTHON_BIN}" >&2
  echo "Set PYTHON_BIN to the Python executable in the project environment." >&2
  exit 1
fi

MODEL="${1:-models/Qwen/Qwen3-4B}"
PORT="${2:-8001}"
TP_SIZE="${3:-1}"
GPU_IDS="${4:-0}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-28672}"

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
cd "$(dirname "$0")/.."

"${VLLM_BIN}" serve "${MODEL}" \
  --tensor-parallel-size "${TP_SIZE}" \
  --gpu-memory-utilization 0.90 \
  --max-model-len "${MAX_MODEL_LEN}" \
  --port "${PORT}" \
  2>&1 | tee "logs/serve_policy_$(date +%m%d_%H%M).log"
