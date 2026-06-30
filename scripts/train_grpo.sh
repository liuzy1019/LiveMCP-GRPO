#!/bin/bash
# GRPO 训练入口 —— PyTorch Lightning 风格配置 + 实验管理。
#
# 支持模式：
#   - 单卡:    bash scripts/train_grpo.sh --model models/Qwen3-4B
#   - 多卡 FSDP: bash scripts/train_grpo.sh --gpus 0,1,2,3
#   - WandB:   bash scripts/train_grpo.sh --wandb
#   - 环境变量覆盖所有 TrainerConfig 字段（OVAL_* 前缀）
#
# Options:
#   --model PATH              模型路径（default: models/Qwen3-4B）
#   --gpus IDS                指定 GPU（如 0,1,2,3）
#   --devices N               限制 GPU 数量
#   --total-steps N           训练步数
#   --wandb                   启用 WandB 日志
#   --wandb-project PROJECT   WandB 项目名
#   --wandb-entity ENTITY     WandB entity
#   --wandb-tags TAGS         WandB 标签（逗号分隔）
#   --strategy {fsdp,deepspeed,ddp}  分布式策略（default: fsdp）
#   --lr LR                   学习率
#   --batch-size N            训练 batch size
#   --rollout-n N             Rollout 每组数量
#   --debug                   调试模式（更多日志）
#
# 实验命名：自动生成 {日期}_{strategy}_{GPU数}gpu_b{batch}_lr{学习率}，如：
#   20260629_fsdp_4gpu_b32_lr1e-6/
#
# Env var 覆盖（优先级最高）：
#   OVAL_MODEL_PATH, OVAL_TRAIN_FILE, OVAL_VAL_FILE
#   OVAL_TOTAL_STEPS, OVAL_ROLLOUT_N, OVAL_RESPONSE_LENGTH
#   OVAL_GPU_MEM_UTIL, OVAL_USE_WANDB, OVAL_WANDB_PROJECT
#   OVAL_SEED, OVAL_LR 等

set -euo pipefail

# ── vLLM orphan cleanup trap ────────────────────────────────────────
_cleanup_vllm_orphans() {
    local exit_code=$?
    VLLM_ORPHANS=$(ps -eo pid,comm --no-headers 2>/dev/null | awk '/VLLM::EngineCore/{print $1}' || true)
    if [ -n "$VLLM_ORPHANS" ]; then
        echo "[cleanup] Killing orphaned VLLM::EngineCore: $VLLM_ORPHANS" >&2
        for pid in $VLLM_ORPHANS; do
            kill -KILL "$pid" 2>/dev/null || true
        done
        sleep 1
    fi
    exit $exit_code
}
trap _cleanup_vllm_orphans EXIT INT TERM

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${PROJECT_ROOT}"

# ── Environment ─────────────────────────────────────────────────────
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export NVCC_APPEND_FLAGS=-allow-unsupported-compiler
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export HYDRA_FULL_ERROR=0
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
export RAY_DEDUP_LOGS=1
export LOGURU_LEVEL=INFO
unset PYTORCH_CUDA_ALLOC_CONF 2>/dev/null || true
export TMPDIR="${TMPDIR:-/tmp/ssgrpo_tmp}"
export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/ssgrpo_ray}"
export LIVEMCP_RAY_TMPDIR="${LIVEMCP_RAY_TMPDIR:-${RAY_TMPDIR}}"
mkdir -p "${TMPDIR}" "${RAY_TMPDIR}"

# ── Parse CLI args ──────────────────────────────────────────────────
GPU_ARG=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --gpus)       GPU_ARG="$2"; shift 2 ;;
        --gpus=*)     GPU_ARG="${1#*=}"; shift ;;
        --devices)    export OVAL_DEVICES="$2"; shift 2 ;;
        --devices=*)  export OVAL_DEVICES="${1#*=}"; shift ;;
        --model)      export OVAL_MODEL_PATH="$2"; shift 2 ;;
        --model=*)    export OVAL_MODEL_PATH="${1#*=}"; shift ;;
        --total-steps)   export OVAL_TOTAL_STEPS="$2"; shift 2 ;;
        --total-steps=*) export OVAL_TOTAL_STEPS="${1#*=}"; shift ;;
        --wandb)      export OVAL_USE_WANDB=1; shift ;;
        --no-wandb)   export OVAL_USE_WANDB=0; shift ;;
        --wandb-project) export OVAL_WANDB_PROJECT="$2"; shift 2 ;;
        --wandb-project=*) export OVAL_WANDB_PROJECT="${1#*=}"; shift ;;
        --wandb-entity) export OVAL_WANDB_ENTITY="$2"; shift 2 ;;
        --wandb-entity=*) export OVAL_WANDB_ENTITY="${1#*=}"; shift ;;
        --wandb-tags) export OVAL_WANDB_TAGS="$2"; shift 2 ;;
        --wandb-tags=*) export OVAL_WANDB_TAGS="${1#*=}"; shift ;;
        --strategy)   export OVAL_STRATEGY="$2"; shift 2 ;;
        --strategy=*) export OVAL_STRATEGY="${1#*=}"; shift ;;
        --lr)         export OVAL_LR="$2"; shift 2 ;;
        --lr=*)       export OVAL_LR="${1#*=}"; shift ;;
        --batch-size) export OVAL_TRAIN_BATCH_SIZE="$2"; shift 2 ;;
        --batch-size=*) export OVAL_TRAIN_BATCH_SIZE="${1#*=}"; shift ;;
        --rollout-n)  export OVAL_ROLLOUT_N="$2"; shift 2 ;;
        --rollout-n=*) export OVAL_ROLLOUT_N="${1#*=}"; shift ;;
        --debug)      export OVAL_DEBUG=1; shift ;;
        *)            echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

# ── Cleanup stale Ray ───────────────────────────────────────────────
if ray status &>/dev/null 2>&1; then
    echo "[cleanup] Stopping stale Ray cluster..."
    ray stop --force 2>/dev/null || true
    sleep 2
fi

# ── GPU detection ───────────────────────────────────────────────────
if [ -n "${GPU_ARG}" ]; then
    . scripts/gpu_config.sh "${GPU_ARG}"
else
    . scripts/gpu_config.sh
fi

# Allow env override of mem_util
GPU_MEM_UTIL="${OVAL_GPU_MEM_UTIL:-${GPU_MEM_UTIL}}"

# ── Per-tier defaults (overrideable via env) ────────────────────────
if [ "${GPU_TIER}" = "L20" ]; then
    : "${OVAL_PROMPT_LENGTH:=12384}"
    : "${OVAL_RESPONSE_LENGTH:=16384}"
    : "${OVAL_MAX_NUM_SEQS:=64}"
    : "${OVAL_MICRO_BATCH:=2}"
    : "${OVAL_TRAIN_BATCH_SIZE:=32}"
    : "${OVAL_MINI_BATCH_SIZE:=8}"
    : "${OVAL_ROLLOUT_N:=16}"
elif [ "${GPU_TIER}" = "A100" ] || [ "${GPU_TIER}" = "Hopper" ]; then
    : "${OVAL_PROMPT_LENGTH:=16384}"
    : "${OVAL_RESPONSE_LENGTH:=16384}"
    : "${OVAL_MAX_NUM_SEQS:=128}"
    : "${OVAL_MICRO_BATCH:=4}"
    : "${OVAL_TRAIN_BATCH_SIZE:=64}"
    : "${OVAL_MINI_BATCH_SIZE:=16}"
    : "${OVAL_ROLLOUT_N:=16}"
elif [ "${GPU_TIER}" = "A10" ]; then
    : "${OVAL_PROMPT_LENGTH:=10240}"
    : "${OVAL_RESPONSE_LENGTH:=4096}"
    : "${OVAL_MAX_NUM_SEQS:=8}"
    : "${OVAL_MICRO_BATCH:=1}"
    : "${OVAL_TRAIN_BATCH_SIZE:=8}"
    : "${OVAL_MINI_BATCH_SIZE:=8}"
    : "${OVAL_ROLLOUT_N:=8}"
else
    : "${OVAL_PROMPT_LENGTH:=10240}"
    : "${OVAL_RESPONSE_LENGTH:=2048}"
    : "${OVAL_MAX_NUM_SEQS:=8}"
    : "${OVAL_MICRO_BATCH:=1}"
    : "${OVAL_TRAIN_BATCH_SIZE:=8}"
    : "${OVAL_MINI_BATCH_SIZE:=8}"
    : "${OVAL_ROLLOUT_N:=4}"
fi

# ── Export tier-derived defaults ────────────────────────────────────
# 这些 : "${VAR:=default}" 已经在上面按 tier 设置好了，这里兜底
: "${OVAL_PROMPT_LENGTH:=10240}"
: "${OVAL_RESPONSE_LENGTH:=2048}"
: "${OVAL_MAX_NUM_SEQS:=8}"
: "${OVAL_MICRO_BATCH:=1}"
: "${OVAL_TRAIN_BATCH_SIZE:=8}"
: "${OVAL_MINI_BATCH_SIZE:=8}"
: "${OVAL_ROLLOUT_N:=8}"

export OVAL_PROMPT_LENGTH OVAL_RESPONSE_LENGTH OVAL_MAX_NUM_SEQS
export OVAL_MICRO_BATCH OVAL_TRAIN_BATCH_SIZE OVAL_MINI_BATCH_SIZE OVAL_ROLLOUT_N
export OVAL_GPU_MEM_UTIL="${GPU_MEM_UTIL}"

# ── Setup Python (config + experiment) ──────────────────────────────
# 将结果写入临时 JSON 文件，避免 shell 字符串转义 / marker 匹配问题
CONDA_PYTHON="${CONDA_PYTHON:-python3}"
CONFIG_JSON_FILE=$(mktemp /tmp/livemcp_config.XXXXXX.json)

# 把 shell 布尔值转成 Python 可识别的 True/False
_py_bool() {
    case "${1:-false}" in true|1|True|TRUE) echo "True" ;; *) echo "False" ;; esac
}

"${CONDA_PYTHON}" -c "
import json, os, sys
sys.path.insert(0, '${PROJECT_ROOT}')

from src.training.trainer_config import (
    TrainerConfig, ExperimentManager, resolve_gpu_info, print_config_summary,
)

config = TrainerConfig.from_env(
    devices=${GPU_COUNT},
    strategy='${OVAL_STRATEGY:-fsdp}',
    gpu_mem_util=${GPU_MEM_UTIL:-0.6},
    enforce_eager=$(_py_bool "${ENFORCE_EAGER:-false}"),
    free_cache_engine=$(_py_bool "${FREE_CACHE_ENGINE:-true}"),
    fsdp_param_offload=$(_py_bool "${PARAM_OFFLOAD:-false}"),
    rollout_tp=${OVAL_ROLLOUT_TP:-1},
    log_prob_micro_batch=${OVAL_LOG_PROB_MICRO_BATCH:-1},
)

num_gpu, gpu_ids, gpu_model = resolve_gpu_info(config.devices)

exp = ExperimentManager(config)
run_dir = exp.setup()

wandb_dir = run_dir / 'wandb'
wandb_dir.mkdir(parents=True, exist_ok=True)

overrides = config.to_hydra_overrides()
overrides.append(f'trainer.default_local_dir={run_dir}/checkpoints')
overrides.append(f'trainer.logger={config.to_logger_list()}')

print_config_summary(config, num_gpu, gpu_model)

result = {
    'overrides': overrides,
    'run_dir': str(run_dir),
    'wandb_dir': str(wandb_dir),
    'gpu_ids': gpu_ids,
    'num_gpu': num_gpu,
    'use_wandb': config.use_wandb,
    'wandb_project': config.wandb_project,
    'run_name': config.run_name,
}
with open('${CONFIG_JSON_FILE}', 'w') as f:
    json.dump(result, f, indent=2)
"

# 读取 JSON 结果
RUN_DIR=$("${CONDA_PYTHON}" -c "import json; print(json.load(open('${CONFIG_JSON_FILE}'))['run_dir'])")
OVERRIDES_STR=$("${CONDA_PYTHON}" -c "import json; print(' '.join(json.load(open('${CONFIG_JSON_FILE}'))['overrides']))")
USE_WANDB=$("${CONDA_PYTHON}" -c "import json; print(json.load(open('${CONFIG_JSON_FILE}'))['use_wandb'])")
WANDB_DIR=$("${CONDA_PYTHON}" -c "import json; print(json.load(open('${CONFIG_JSON_FILE}'))['wandb_dir'])")
rm -f "${CONFIG_JSON_FILE}"

# ── Validate data ───────────────────────────────────────────────────
echo ""
echo "=== Validating data ==="
"${CONDA_PYTHON}" -c "
import sys, pandas as pd
from pathlib import Path

train_file = '${OVAL_TRAIN_FILE:-data/train.parquet}'
val_file = '${OVAL_VAL_FILE:-data/val.parquet}'

for path in [train_file, val_file]:
    if not Path(path).exists():
        print(f'  WARNING: {path} does not exist!')
        continue
    df = pd.read_parquet(path)
    domains = set()
    from src.utils import normalize_extra_info
    for _, row in df.iterrows():
        ei = normalize_extra_info(row['extra_info'])
        domains.add(ei.get('domain', 'unknown'))
    print(f'  {path}: {len(df)} rows, domains={sorted(domains)}')
    if len(df) > 0:
        ei = normalize_extra_info(df.iloc[0]['extra_info'])
        print(f'    sample: domain={ei.get(\"domain\")}, scenario={ei.get(\"scenario_type\")}')
"
echo ""

# ── Setup WandB env ─────────────────────────────────────────────────
if [ "${USE_WANDB}" = "True" ]; then
    export WANDB_DIR="${WANDB_DIR}"
    export WANDB_PROJECT="${OVAL_WANDB_PROJECT:-oval-mcp-grpo}"
    [ -n "${OVAL_WANDB_ENTITY:-}" ] && export WANDB_ENTITY="${OVAL_WANDB_ENTITY}"
    echo "[wandb] Enabled: project=${WANDB_PROJECT}, dir=${WANDB_DIR}"
else
    echo "[wandb] Disabled (use --wandb to enable)"
fi

# ── Launch Training ─────────────────────────────────────────────────
echo ""
echo "=== Launching GRPO Training ==="
echo "  Experiment: ${RUN_DIR}"
echo "  Log:        ${RUN_DIR}/logs/train.log"
echo ""

exec "${CONDA_PYTHON}" "scripts/train_grpo.py" \
    ${OVERRIDES_STR} \
    2>&1 | tee "${RUN_DIR}/logs/train.log"

echo ""
echo "=== Training Complete ==="
echo "  Results: ${RUN_DIR}"
echo "  Log:     ${RUN_DIR}/logs/train.log"
echo "  Checkpoints: ${RUN_DIR}/checkpoints/"
