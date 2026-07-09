#!/bin/bash
# Start a vLLM OpenAI-compatible API server for local model serving.
# Used for RL rollout/training（teacher generation 由 generate_data.sh 统一管理）。
#
# Usage:
#   bash scripts/start_vllm.sh <model_path> [port] [tp_size] [gpu_ids]
#
# Example:
#   bash scripts/start_vllm.sh models/Qwen/Qwen3-4B 8001 2 "0,1"

MODEL="${1:-models/Qwen/Qwen3-4B}"
PORT="${2:-8001}"
TP_SIZE="${3:-1}"
GPU_IDS="${4:-0}"

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
cd "$(dirname "$0")/.."

vllm serve "${MODEL}" \
  --tensor-parallel-size "${TP_SIZE}" \
  --gpu-memory-utilization 0.90 \
  --max-model-len 8192 \
  --port "${PORT}" \
  2>&1 | tee "logs/vllm_$(date +%m%d_%H%M).log"
