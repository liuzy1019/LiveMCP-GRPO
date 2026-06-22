#!/usr/bin/env bash
# E2: SFT 基线
# Config: configs/exp2_sft.yaml（可通过 EXP_CONFIG 覆盖）
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$PROJECT_DIR"

N_GPUS="${N_GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l)}"
if [ "${N_GPUS}" -lt 1 ] 2>/dev/null || [ -z "${N_GPUS}" ]; then
    echo "❌ 未检测到 GPU (N_GPUS=${N_GPUS})，无法启动训练" >&2
    exit 1
fi

EXP_CONFIG="${EXP_CONFIG:-${PROJECT_DIR}/configs/exp2_sft.yaml}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/data1/zhanyiliu/liuzhanyi/anaconda3/envs/arl/bin/python}"
EXP_NAME=$("$PYTHON_BIN" -c "import yaml; print(yaml.safe_load(open('${EXP_CONFIG}'))['exp']['name'])")
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${PROJECT_DIR}/logs/${EXP_NAME}_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

echo "E2: SFT start | gpus=${N_GPUS} | config=${EXP_CONFIG}" | tee "${LOG_DIR}/train.log"

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export WANDB_PROJECT="${WANDB_PROJECT:-schemashift-grpo}"
export EXP_CONFIG

TORCHRUN_BIN="${TORCHRUN_BIN:-/mnt/data1/zhanyiliu/liuzhanyi/anaconda3/envs/arl/bin/torchrun}"

"$TORCHRUN_BIN" \
    --nproc_per_node="${N_GPUS}" \
    --master_port="${MASTER_PORT:-29501}" \
    "${PROJECT_DIR}/scripts/train/sft/run_exp2_sft.py" \
    2>&1 | tee -a "${LOG_DIR}/train.log"

echo "E2 完成"
