#!/bin/bash
# SchemaShift GRPO 正式训练
# 基于 smoke test 验证通过的配置，调整为完整训练参数
# 8×L20 44GB colocated 模式

set -euo pipefail

# ---- 路径基准 ----
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${PROJECT_ROOT}"

# ---- 环境 ----
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_NO_ADVISORY_WARNINGS="${TRANSFORMERS_NO_ADVISORY_WARNINGS:-1}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-0}"
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO="${RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO:-0}"
export RAY_DEDUP_LOGS="${RAY_DEDUP_LOGS:-1}"
export SCHEMASHIFT_CONSOLE_LOG_MODE="${SCHEMASHIFT_CONSOLE_LOG_MODE:-compact}"
export SCHEMASHIFT_VAL_NUM_EXAMINE="${SCHEMASHIFT_VAL_NUM_EXAMINE:-0}"
export SCHEMASHIFT_VERBOSE_VALIDATION="${SCHEMASHIFT_VERBOSE_VALIDATION:-0}"
unset PYTORCH_CUDA_ALLOC_CONF 2>/dev/null || true
export PYTHONWARNINGS="${PYTHONWARNINGS:+${PYTHONWARNINGS},}ignore:.*FSDP\\.state_dict_type\\(\\).*:FutureWarning"
export TMPDIR="${TMPDIR:-/tmp/ssgrpo_tmp}"
export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/ssgrpo_ray}"
export SCHEMASHIFT_RAY_TMPDIR="${SCHEMASHIFT_RAY_TMPDIR:-${RAY_TMPDIR}}"
mkdir -p "${TMPDIR}" "${RAY_TMPDIR}" outputs

# ---- wandb 配置 ----
export WANDB_PROJECT="${WANDB_PROJECT:-schemashift_grpo}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-schemashift_grpo_$(date +%m%d_%H%M)}"
# 如果需要指定 entity（团队/用户名），取消下面的注释：
# export WANDB_ENTITY="your_entity"
# 离线模式（无网络时）：export WANDB_MODE=offline

# ---- 抑制 wandb atexit BrokenPipeError ----
# wandb 在 Ray worker 退出时偶尔触发 atexit 清理 BrokenPipe，无害但噪音大
export WANDB_CONSOLE="${WANDB_CONSOLE:-off}"

# ---- YAML 配置（命令行参数会覆盖 YAML） ----
CONFIG_PATH="${GRPO_CONFIG:-}"
for ((i=1; i<=$#; i++)); do
    if [ "${!i}" = "--config" ]; then
        next=$((i + 1))
        CONFIG_PATH="${!next:-}"
        break
    fi
done

_yaml_grpo_defaults() {
    local config_path="$1"
    [ -z "${config_path}" ] && return 0
    python - "${config_path}" <<'PYEOF'
import shlex
import sys
from pathlib import Path

import yaml

path = Path(sys.argv[1])
if not path.is_absolute():
    path = Path.cwd() / path
with path.open("r", encoding="utf-8") as f:
    data = yaml.safe_load(f) or {}
if not isinstance(data, dict):
    raise SystemExit(f"GRPO YAML config must be a mapping: {path}")

mapping = {
    ("model", "path"): "MODEL_PATH",
    ("data", "train_file"): "TRAIN_FILE",
    ("data", "val_file"): "VAL_FILE",
    ("reward", "function_path"): "REWARD_FN_PATH",
    ("resources", "n_gpus"): "N_GPUS",
    ("resources", "cuda_visible_devices"): "CUDA_VISIBLE_DEVICES",
    ("trainer", "total_epochs"): "TOTAL_EPOCHS",
    ("trainer", "total_steps"): "TOTAL_STEPS",
    ("trainer", "save_freq"): "SAVE_FREQ",
    ("trainer", "test_freq"): "TEST_FREQ",
    ("optimization", "lr"): "LR",
    ("optimization", "lr_warmup_ratio"): "LR_WARMUP_RATIO",
    ("optimization", "kl_coef"): "KL_COEF",
    ("optimization", "ppo_epochs"): "PPO_EPOCHS",
    ("optimization", "grad_clip"): "GRAD_CLIP",
    ("batch", "train_batch_size"): "TRAIN_BATCH_SIZE",
    ("batch", "val_batch_size"): "VAL_BATCH_SIZE",
    ("batch", "mini_batch_size"): "MINI_BATCH_SIZE",
    ("batch", "micro_batch_size_per_gpu"): "MICRO_BATCH",
    ("batch", "log_prob_micro_batch_size_per_gpu"): "LOG_PROB_MICRO_BATCH",
    ("rollout", "n"): "ROLLOUT_N",
    ("rollout", "temperature"): "TEMPERATURE",
    ("rollout", "top_p"): "TOP_P",
    ("rollout", "tensor_model_parallel_size"): "ROLLOUT_TP",
    ("rollout", "gpu_memory_utilization"): "GPU_MEM_UTIL",
    ("rollout", "free_cache_engine"): "FREE_CACHE_ENGINE",
    ("rollout", "enforce_eager"): "ENFORCE_EAGER",
    ("rollout", "prompt_length"): "PROMPT_LENGTH",
    ("rollout", "response_length"): "RESPONSE_LENGTH",
    ("rollout", "max_num_seqs"): "MAX_NUM_SEQS",
    ("offload", "ref_param_offload"): "PARAM_OFFLOAD",
    ("offload", "actor_param_offload"): "ACTOR_PARAM_OFFLOAD",
    ("wandb", "project"): "WANDB_PROJECT",
    ("wandb", "run_name"): "WANDB_RUN_NAME",
    ("terminal", "log_mode"): "SCHEMASHIFT_CONSOLE_LOG_MODE",
    ("terminal", "val_examine"): "SCHEMASHIFT_VAL_NUM_EXAMINE",
    ("terminal", "verbose_validation"): "SCHEMASHIFT_VERBOSE_VALIDATION",
}
for (section, key), env_name in mapping.items():
    section_data = data.get(section, {})
    if isinstance(section_data, dict) and key in section_data:
        value = section_data[key]
        if isinstance(value, bool):
            value = "True" if value else "False"
        print(f"export {env_name}={shlex.quote(str(value))}")
PYEOF
}

if [ -n "${CONFIG_PATH}" ]; then
    eval "$(_yaml_grpo_defaults "${CONFIG_PATH}")"
fi

# ---- 清理残留 Ray 进程 ----
if ray status &>/dev/null 2>&1; then
    echo "[cleanup] Stopping stale Ray cluster..."
    ray stop --force 2>/dev/null || true
    sleep 2
fi
if pgrep -f "ray::WorkerDict" &>/dev/null; then
    echo "[cleanup] Killing orphan ray workers..."
    pkill -9 -f "ray::WorkerDict" 2>/dev/null || true
    sleep 1
fi

# ---- 可配置参数 ----
N_GPUS="${N_GPUS:-8}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export CUDA_VISIBLE_DEVICES
MODEL_PATH="${MODEL_PATH:-outputs/sft_cold_start_4b/final}"
TRAIN_FILE="${TRAIN_FILE:-data/grpo_train.parquet}"
VAL_FILE="${VAL_FILE:-data/grpo_val.parquet}"
REWARD_FN_PATH="${REWARD_FN_PATH:-src/reward/schemashift_reward_fn.py}"

# 训练超参
TOTAL_EPOCHS="${TOTAL_EPOCHS:--1}"
TOTAL_STEPS="${TOTAL_STEPS:-350}"

# 保证 epoch 循环非空：verl 用 total_epochs 控制外层 for 循环，
# total_training_steps 控制提前退出。当只设步数不设 epoch 时，给一个安全的大值
if [ "${TOTAL_EPOCHS}" = "-1" ] && [ "${TOTAL_STEPS}" -gt 0 ]; then
    TOTAL_EPOCHS=1000
fi
SAVE_FREQ="${SAVE_FREQ:-100}"
TEST_FREQ="${TEST_FREQ:-50}"
LR="${LR:-1e-6}"
LR_WARMUP_RATIO="${LR_WARMUP_RATIO:-0.05}"
KL_COEF="${KL_COEF:-0.01}"
PPO_EPOCHS="${PPO_EPOCHS:-1}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
ROLLOUT_N="${ROLLOUT_N:-16}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.95}"

# ---- 命令行注入 ----
HYDRA_OVERRIDES=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            cat <<'EOF'
Usage:
  bash scripts/run_grpo.sh [options] [hydra_overrides...]

Options:
  --config PATH             YAML 配置文件；命令行参数会覆盖 YAML
  --model PATH              模型路径
  --train-file PATH         训练数据
  --val-file PATH           验证数据
  --reward-fn PATH          reward function 路径
  --n-gpus N                GPU 数量
  --cuda-visible-devices IDS
  --total-epochs N          总 epoch 数（默认 3）
  --total-steps N           总步数（-1 表示由 epoch 决定）
  --lr FLOAT                学习率（默认 5e-7）
  --lr-warmup-ratio FLOAT   warmup 比例
  --kl-coef FLOAT           KL 系数（默认 0.01）
  --ppo-epochs N            PPO epochs
  --grad-clip FLOAT         actor grad clip
  --save-freq N             checkpoint 保存频率
  --test-freq N             验证频率
  --rollout-n N             每个 prompt 采样数（默认 4）
  --temperature FLOAT       rollout temperature
  --top-p FLOAT             rollout top_p
  --prompt-length N
  --response-length N
  --train-batch-size N
  --val-batch-size N
  --mini-batch-size N
  --micro-batch N
  --log-prob-micro-batch N
  --rollout-tp N
  --gpu-mem-util FLOAT
  --max-num-seqs N
  --free-cache-engine BOOL
  --enforce-eager BOOL
  --param-offload BOOL
  --actor-param-offload BOOL
  --wandb-name NAME         wandb run name
  --wandb-project NAME      wandb project name
  --log-mode MODE           terminal metrics: compact/full（默认 compact）
  --val-examine N           validation 展开打印样本数（默认 0）
  --verbose-validation      打印 validation gen_batch 调试信息

Unknown arguments are passed through as Hydra overrides.
EOF
            exit 0
            ;;
        --model) MODEL_PATH="$2"; shift 2 ;;
        --train-file) TRAIN_FILE="$2"; shift 2 ;;
        --val-file) VAL_FILE="$2"; shift 2 ;;
        --reward-fn) REWARD_FN_PATH="$2"; shift 2 ;;
        --n-gpus) N_GPUS="$2"; shift 2 ;;
        --total-epochs) TOTAL_EPOCHS="$2"; shift 2 ;;
        --total-steps) TOTAL_STEPS="$2"; shift 2 ;;
        --lr) LR="$2"; shift 2 ;;
        --lr-warmup-ratio) LR_WARMUP_RATIO="$2"; shift 2 ;;
        --kl-coef) KL_COEF="$2"; shift 2 ;;
        --ppo-epochs) PPO_EPOCHS="$2"; shift 2 ;;
        --grad-clip) GRAD_CLIP="$2"; shift 2 ;;
        --save-freq) SAVE_FREQ="$2"; shift 2 ;;
        --test-freq) TEST_FREQ="$2"; shift 2 ;;
        --rollout-n) ROLLOUT_N="$2"; shift 2 ;;
        --temperature) TEMPERATURE="$2"; shift 2 ;;
        --top-p) TOP_P="$2"; shift 2 ;;
        --prompt-length) PROMPT_LENGTH="$2"; shift 2 ;;
        --response-length) RESPONSE_LENGTH="$2"; shift 2 ;;
        --train-batch-size) TRAIN_BATCH_SIZE="$2"; shift 2 ;;
        --val-batch-size) VAL_BATCH_SIZE="$2"; shift 2 ;;
        --mini-batch-size) MINI_BATCH_SIZE="$2"; shift 2 ;;
        --micro-batch) MICRO_BATCH="$2"; shift 2 ;;
        --log-prob-micro-batch) LOG_PROB_MICRO_BATCH="$2"; shift 2 ;;
        --rollout-tp) ROLLOUT_TP="$2"; shift 2 ;;
        --gpu-mem-util) GPU_MEM_UTIL="$2"; shift 2 ;;
        --max-num-seqs) MAX_NUM_SEQS="$2"; shift 2 ;;
        --free-cache-engine) FREE_CACHE_ENGINE="$2"; shift 2 ;;
        --enforce-eager) ENFORCE_EAGER="$2"; shift 2 ;;
        --param-offload) PARAM_OFFLOAD="$2"; shift 2 ;;
        --actor-param-offload) ACTOR_PARAM_OFFLOAD="$2"; shift 2 ;;
        --wandb-name) export WANDB_RUN_NAME="$2"; shift 2 ;;
        --wandb-project) export WANDB_PROJECT="$2"; shift 2 ;;
        --log-mode) export SCHEMASHIFT_CONSOLE_LOG_MODE="$2"; shift 2 ;;
        --val-examine) export SCHEMASHIFT_VAL_NUM_EXAMINE="$2"; shift 2 ;;
        --verbose-validation) export SCHEMASHIFT_VERBOSE_VALIDATION=1; shift ;;
        --cuda-visible-devices) CUDA_VISIBLE_DEVICES="$2"; export CUDA_VISIBLE_DEVICES; shift 2 ;;
        --config) CONFIG_PATH="$2"; shift 2 ;;
        --) shift; HYDRA_OVERRIDES+=("$@"); break ;;
        *) HYDRA_OVERRIDES+=("$1"); shift ;;
    esac
done

# ---- 探测 GPU 显存 ----
FIRST_GPU=$(echo "${CUDA_VISIBLE_DEVICES}" | cut -d',' -f1)
GPU_MEM_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "${FIRST_GPU}" | tr -d ' ')
GPU_MEM_GIB=$((GPU_MEM_MIB / 1024))

echo "=== GPU Info ==="
echo "N_GPUS: ${N_GPUS}, GPU_MEM: ${GPU_MEM_GIB} GiB per card"

# ---- 根据显存档位自适应超参 ----
# Qwen3-4B BF16 ~8GB，比 7B (~15GB) 小一半，显存宽裕
if [ "${GPU_MEM_GIB}" -ge 40 ]; then
    # L20 44GB + 4B 模型：显存充裕，可以提高吞吐
    ROLLOUT_TP="${ROLLOUT_TP:-1}"
    GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.60}"
    FREE_CACHE_ENGINE="${FREE_CACHE_ENGINE:-True}"
    ENFORCE_EAGER="${ENFORCE_EAGER:-False}"
    MICRO_BATCH="${MICRO_BATCH:-4}"
    PROMPT_LENGTH="${PROMPT_LENGTH:-10240}"
    RESPONSE_LENGTH="${RESPONSE_LENGTH:-2048}"
    PARAM_OFFLOAD="${PARAM_OFFLOAD:-False}"
    ACTOR_PARAM_OFFLOAD="${ACTOR_PARAM_OFFLOAD:-False}"
    LOG_PROB_MICRO_BATCH="${LOG_PROB_MICRO_BATCH:-4}"
    MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
elif [ "${GPU_MEM_GIB}" -ge 20 ]; then
    # A10 23GB + 4B 模型：紧凑但可行
    ROLLOUT_TP="${ROLLOUT_TP:-1}"
    GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.50}"
    FREE_CACHE_ENGINE="${FREE_CACHE_ENGINE:-True}"
    ENFORCE_EAGER="${ENFORCE_EAGER:-True}"
    MICRO_BATCH="${MICRO_BATCH:-2}"
    PROMPT_LENGTH="${PROMPT_LENGTH:-10240}"
    RESPONSE_LENGTH="${RESPONSE_LENGTH:-1024}"
    PARAM_OFFLOAD="${PARAM_OFFLOAD:-True}"
    ACTOR_PARAM_OFFLOAD="${ACTOR_PARAM_OFFLOAD:-False}"
    LOG_PROB_MICRO_BATCH="${LOG_PROB_MICRO_BATCH:-2}"
    MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
else
    echo "ERROR: GPU 显存不足 20GB，无法运行"
    exit 1
fi

# batch size 随卡数线性扩展
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-$((N_GPUS * 2))}"
MINI_BATCH_SIZE="${MINI_BATCH_SIZE:-${N_GPUS}}"

echo "=== Training Config ==="
echo "MODEL: ${MODEL_PATH}"
echo "TRAIN: ${TRAIN_FILE} | VAL: ${VAL_FILE}"
echo "EPOCHS: ${TOTAL_EPOCHS}, STEPS: ${TOTAL_STEPS}, LR: ${LR}, WARMUP: ${LR_WARMUP_RATIO}"
echo "PPO_EPOCHS: ${PPO_EPOCHS}, GRAD_CLIP: ${GRAD_CLIP}, KL_COEF: ${KL_COEF}"
echo "ROLLOUT_TP: ${ROLLOUT_TP}, ROLLOUT_N: ${ROLLOUT_N}, TEMP: ${TEMPERATURE}, TOP_P: ${TOP_P}"
echo "GPU_MEM_UTIL: ${GPU_MEM_UTIL}, MAX_NUM_SEQS: ${MAX_NUM_SEQS}"
echo "TRAIN_BATCH: ${TRAIN_BATCH_SIZE}, MINI_BATCH: ${MINI_BATCH_SIZE}, MICRO_BATCH: ${MICRO_BATCH}"
echo "PROMPT_LEN: ${PROMPT_LENGTH}, RESP_LEN: ${RESPONSE_LENGTH}"
echo "SAVE_FREQ: ${SAVE_FREQ}, TEST_FREQ: ${TEST_FREQ}"
echo "WANDB: project=${WANDB_PROJECT}, run=${WANDB_RUN_NAME}"
echo "TERMINAL_LOG: mode=${SCHEMASHIFT_CONSOLE_LOG_MODE}, val_examine=${SCHEMASHIFT_VAL_NUM_EXAMINE}, verbose_validation=${SCHEMASHIFT_VERBOSE_VALIDATION}"
echo ""

# ---- PYTHONPATH ----
export PYTHONPATH=".:${PYTHONPATH:-}"

# ---- 数据校验 ----
python - "${TRAIN_FILE}" "${VAL_FILE}" <<'PYEOF'
import sys
import pandas as pd

required = {"perturbation_level", "scenario_type", "group_id", "uid"}
for path in sys.argv[1:]:
    df = pd.read_parquet(path)
    missing = set()
    for key in required:
        if key in df.columns:
            continue
        if "extra_info" in df.columns and all(
            isinstance(x, dict) and key in x for x in df["extra_info"].head(min(len(df), 32))
        ):
            continue
        missing.add(key)
    if missing:
        raise SystemExit(f"{path} missing fields: {sorted(missing)}")
print("Data validation: OK")
PYEOF

# ---- 启动训练 ----
CONDA_PYTHON="$(which python)"
LOG_FILE="outputs/grpo_train_$(date +%Y%m%d_%H%M%S).log"

echo "=== Starting GRPO Training ==="
echo "Log: ${LOG_FILE}"
echo ""

# 条件化 total_training_steps：-1 表示不设步数限制，走 epoch 控制
if [ "${TOTAL_STEPS}" != "-1" ]; then
    HYDRA_OVERRIDES+=("trainer.total_training_steps=${TOTAL_STEPS}")
fi

"${CONDA_PYTHON}" "scripts/train_grpo.py" \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    data.max_prompt_length="${PROMPT_LENGTH}" \
    data.max_response_length="${RESPONSE_LENGTH}" \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.val_batch_size="${VAL_BATCH_SIZE}" \
    data.shuffle=True \
    data.filter_overlong_prompts=True \
    data.truncation=left \
    data.reward_fn_key=data_source \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.actor.ppo_mini_batch_size="${MINI_BATCH_SIZE}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${MICRO_BATCH}" \
    actor_rollout_ref.actor.ppo_epochs="${PPO_EPOCHS}" \
    actor_rollout_ref.actor.grad_clip="${GRAD_CLIP}" \
    actor_rollout_ref.actor.optim.lr="${LR}" \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio="${LR_WARMUP_RATIO}" \
    actor_rollout_ref.actor.fsdp_config.param_offload="${ACTOR_PARAM_OFFLOAD}" \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP}" \
    actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEM_UTIL}" \
    actor_rollout_ref.rollout.free_cache_engine="${FREE_CACHE_ENGINE}" \
    actor_rollout_ref.rollout.enforce_eager="${ENFORCE_EAGER}" \
    actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
    actor_rollout_ref.rollout.temperature="${TEMPERATURE}" \
    actor_rollout_ref.rollout.top_p="${TOP_P}" \
    actor_rollout_ref.rollout.prompt_length="${PROMPT_LENGTH}" \
    actor_rollout_ref.rollout.response_length="${RESPONSE_LENGTH}" \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((PROMPT_LENGTH + RESPONSE_LENGTH)) \
    actor_rollout_ref.rollout.max_num_seqs="${MAX_NUM_SEQS}" \
    actor_rollout_ref.rollout.enable_prefix_caching=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO_BATCH}" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO_BATCH}" \
    actor_rollout_ref.ref.fsdp_config.param_offload="${PARAM_OFFLOAD}" \
    algorithm.adv_estimator=schemashift_grpo \
    algorithm.use_kl_in_reward=True \
    algorithm.kl_penalty=kl \
    algorithm.kl_ctrl.type=fixed \
    algorithm.kl_ctrl.kl_coef="${KL_COEF}" \
    custom_reward_function.path="${REWARD_FN_PATH}" \
    custom_reward_function.name=compute_score \
    trainer.project_name="${WANDB_PROJECT}" \
    trainer.experiment_name="${WANDB_RUN_NAME}" \
    trainer.logger='["console","wandb"]' \
    trainer.total_epochs="${TOTAL_EPOCHS}" \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node="${N_GPUS}" \
    trainer.save_freq="${SAVE_FREQ}" \
    trainer.val_before_train=True \
    trainer.test_freq="${TEST_FREQ}" \
    reward_model.enable=False \
    "${HYDRA_OVERRIDES[@]}" \
    2>&1 | tee "${LOG_FILE}"

echo ""
echo "=== Training Complete ==="
echo "Log saved to: ${LOG_FILE}"
echo "Checkpoints in: outputs/${WANDB_PROJECT}/${WANDB_RUN_NAME}/"
