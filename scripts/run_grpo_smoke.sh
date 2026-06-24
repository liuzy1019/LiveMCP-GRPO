#!/bin/bash
# SchemaShift GRPO smoke test
# 验收标准：step 1 完成 + loss finite + 无 traceback
# 自适应 GPU 数量和显存（A10 23GB / L20 48GB）

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
export LOGURU_LEVEL="${LOGURU_LEVEL:-INFO}"  # 默认 INFO，避免 DEBUG 刷屏；需要 DEBUG 的用 LOGURU_LEVEL=DEBUG 启动
# 注意：不要设置 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# vLLM 0.11 的 CuMemAllocator 与 expandable_segments 互斥（会 assert 失败）
# verl 自己通过 torch.cuda.memory._set_allocator_settings() 在训练阶段动态开启
unset PYTORCH_CUDA_ALLOC_CONF 2>/dev/null || true
export PYTHONWARNINGS="${PYTHONWARNINGS:+${PYTHONWARNINGS},}ignore:.*FSDP\\.state_dict_type\\(\\).*:FutureWarning"
export TMPDIR="${TMPDIR:-/tmp/ssgrpo_tmp}"
export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/ssgrpo_ray}"
export SCHEMASHIFT_RAY_TMPDIR="${SCHEMASHIFT_RAY_TMPDIR:-${RAY_TMPDIR}}"
mkdir -p "${TMPDIR}" "${RAY_TMPDIR}" outputs

# ---- YAML 配置（命令行参数会覆盖 YAML） ----
CONFIG_PATH="${GRPO_SMOKE_CONFIG:-${GRPO_CONFIG:-}}"
for ((i=1; i<=$#; i++)); do
    if [ "${!i}" = "--config" ]; then
        next=$((i + 1))
        CONFIG_PATH="${!next:-}"
        break
    fi
done

_yaml_grpo_smoke_defaults() {
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
    ("trainer", "total_steps"): "TOTAL_STEPS",
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
    eval "$(_yaml_grpo_smoke_defaults "${CONFIG_PATH}")"
fi

# ---- 清理残留 Ray 进程（防止上次 OOM 崩溃后 GPU 显存未释放） ----
if ray status &>/dev/null 2>&1; then
    echo "[cleanup] Stopping stale Ray cluster..."
    ray stop --force 2>/dev/null || true
    sleep 2
fi
# 兜底：杀掉可能残留的 vllm/ray worker 进程
if pgrep -f "ray::WorkerDict" &>/dev/null; then
    echo "[cleanup] Killing orphan ray workers..."
    pkill -9 -f "ray::WorkerDict" 2>/dev/null || true
    sleep 1
fi

# ---- 可配置参数（均可通过环境变量覆盖） ----
N_GPUS="${N_GPUS:-8}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export CUDA_VISIBLE_DEVICES
MODEL_PATH="${MODEL_PATH:-outputs/sft_cold_start_4b/final}"
TRAIN_FILE="${TRAIN_FILE:-data/oval_grpo_train.parquet}"
VAL_FILE="${VAL_FILE:-data/oval_grpo_val.parquet}"
REWARD_FN_PATH="${REWARD_FN_PATH:-src/reward/oval_reward_fn.py}"
TOTAL_STEPS="${TOTAL_STEPS:-2}"
LR="${LR:-1e-6}"
LR_WARMUP_RATIO="${LR_WARMUP_RATIO:-0.1}"
KL_COEF="${KL_COEF:-0.01}"
PPO_EPOCHS="${PPO_EPOCHS:-1}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
ROLLOUT_N="${ROLLOUT_N:-4}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.95}"

# ---- 命令行注入（未知参数会透传给 Hydra overrides） ----
HYDRA_OVERRIDES=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            cat <<'EOF'
Usage:
  bash scripts/run_grpo_smoke.sh [options] [hydra_overrides...]

Common options:
  --config PATH
  --model PATH
  --train-file PATH
  --val-file PATH
  --reward-fn PATH
  --n-gpus N
  --cuda-visible-devices IDS
  --total-steps N
  --lr FLOAT
  --lr-warmup-ratio FLOAT
  --kl-coef FLOAT
  --ppo-epochs N
  --grad-clip FLOAT
  --rollout-n N
  --temperature FLOAT
  --top-p FLOAT
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
  --log-mode MODE
  --val-examine N
  --verbose-validation

Unknown arguments are passed through as Hydra overrides, for example:
  bash scripts/run_grpo_smoke.sh --n-gpus 4 actor_rollout_ref.rollout.n=2
EOF
            exit 0
            ;;
        --model) MODEL_PATH="$2"; shift 2 ;;
        --config) CONFIG_PATH="$2"; shift 2 ;;
        --train-file) TRAIN_FILE="$2"; shift 2 ;;
        --val-file) VAL_FILE="$2"; shift 2 ;;
        --reward-fn) REWARD_FN_PATH="$2"; shift 2 ;;
        --n-gpus) N_GPUS="$2"; shift 2 ;;
        --cuda-visible-devices) CUDA_VISIBLE_DEVICES="$2"; export CUDA_VISIBLE_DEVICES; shift 2 ;;
        --total-steps) TOTAL_STEPS="$2"; shift 2 ;;
        --lr) LR="$2"; shift 2 ;;
        --lr-warmup-ratio) LR_WARMUP_RATIO="$2"; shift 2 ;;
        --kl-coef) KL_COEF="$2"; shift 2 ;;
        --ppo-epochs) PPO_EPOCHS="$2"; shift 2 ;;
        --grad-clip) GRAD_CLIP="$2"; shift 2 ;;
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
        --log-mode) export SCHEMASHIFT_CONSOLE_LOG_MODE="$2"; shift 2 ;;
        --val-examine) export SCHEMASHIFT_VAL_NUM_EXAMINE="$2"; shift 2 ;;
        --verbose-validation) export SCHEMASHIFT_VERBOSE_VALIDATION=1; shift ;;
        --free-cache-engine) FREE_CACHE_ENGINE="$2"; shift 2 ;;
        --enforce-eager) ENFORCE_EAGER="$2"; shift 2 ;;
        --param-offload) PARAM_OFFLOAD="$2"; shift 2 ;;
        --actor-param-offload) ACTOR_PARAM_OFFLOAD="$2"; shift 2 ;;
        --) shift; HYDRA_OVERRIDES+=("$@"); break ;;
        *) HYDRA_OVERRIDES+=("$1"); shift ;;
    esac
done

# ---- 探测 GPU 显存 ----
FIRST_GPU=$(echo "${CUDA_VISIBLE_DEVICES}" | cut -d',' -f1)
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: 未找到 nvidia-smi，无法运行 GRPO smoke" >&2
    exit 1
fi
if ! GPU_MEM_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "${FIRST_GPU}" 2>/dev/null | tr -d ' '); then
    echo "ERROR: nvidia-smi 无法读取 GPU ${FIRST_GPU}，无法运行 GRPO smoke" >&2
    echo "       请确认 NVIDIA driver 正常，并在 arl 环境运行: nvidia-smi -L" >&2
    exit 1
fi
if ! [[ "${GPU_MEM_MIB}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: nvidia-smi 返回的显存不是数字: ${GPU_MEM_MIB}" >&2
    exit 1
fi
GPU_MEM_GIB=$((GPU_MEM_MIB / 1024))

echo "=== GPU Info ==="
echo "N_GPUS: ${N_GPUS}, GPU_MEM: ${GPU_MEM_GIB} GiB per card"

# ---- 根据显存档位自适应超参 ----
# colocated 模式：FSDP actor + vLLM rollout 共享 GPU
# Qwen3-4B BF16 ≈ 8GB，TP=1 即可

if [ "${GPU_MEM_GIB}" -ge 40 ]; then
    # ---- L20 44GB + 4B 模型：显存充裕 ----
    ROLLOUT_TP="${ROLLOUT_TP:-1}"
    GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.60}"
    FREE_CACHE_ENGINE="${FREE_CACHE_ENGINE:-True}"
    ENFORCE_EAGER="${ENFORCE_EAGER:-False}"
    MICRO_BATCH="${MICRO_BATCH:-4}"
    PROMPT_LENGTH="${PROMPT_LENGTH:-10240}"
    RESPONSE_LENGTH="${RESPONSE_LENGTH:-1024}"
    PARAM_OFFLOAD="${PARAM_OFFLOAD:-False}"
    ACTOR_PARAM_OFFLOAD="${ACTOR_PARAM_OFFLOAD:-False}"
    LOG_PROB_MICRO_BATCH="${LOG_PROB_MICRO_BATCH:-4}"
    MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
elif [ "${GPU_MEM_GIB}" -ge 20 ]; then
    # ---- A10 23GB + 4B 模型：紧凑但可行 ----
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
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-$((N_GPUS * 4))}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-$((N_GPUS * 2))}"
MINI_BATCH_SIZE="${MINI_BATCH_SIZE:-${N_GPUS}}"

echo "=== Adaptive Config ==="
echo "ROLLOUT_TP: ${ROLLOUT_TP}"
echo "GPU_MEM_UTIL: ${GPU_MEM_UTIL}"
echo "FREE_CACHE_ENGINE: ${FREE_CACHE_ENGINE}, ENFORCE_EAGER: ${ENFORCE_EAGER}"
echo "LR: ${LR}, WARMUP: ${LR_WARMUP_RATIO}, PPO_EPOCHS: ${PPO_EPOCHS}, GRAD_CLIP: ${GRAD_CLIP}, KL_COEF: ${KL_COEF}"
echo "ROLLOUT_N: ${ROLLOUT_N}, TEMP: ${TEMPERATURE}, TOP_P: ${TOP_P}"
echo "TRAIN_BATCH: ${TRAIN_BATCH_SIZE}, MINI_BATCH: ${MINI_BATCH_SIZE}, MICRO_BATCH: ${MICRO_BATCH}"
echo "PROMPT_LEN: ${PROMPT_LENGTH}, RESP_LEN: ${RESPONSE_LENGTH}"
echo "MAX_NUM_SEQS: ${MAX_NUM_SEQS}"
echo "PARAM_OFFLOAD(ref): ${PARAM_OFFLOAD}, ACTOR_PARAM_OFFLOAD: ${ACTOR_PARAM_OFFLOAD}"
echo "TERMINAL_LOG: mode=${SCHEMASHIFT_CONSOLE_LOG_MODE}, val_examine=${SCHEMASHIFT_VAL_NUM_EXAMINE}, verbose_validation=${SCHEMASHIFT_VERBOSE_VALIDATION}"
echo ""

# ---- 注册 estimator ----
export PYTHONPATH=".:${PYTHONPATH:-}"

echo "=== SchemaShift GRPO Smoke Test ==="
echo "MODEL: ${MODEL_PATH}"
echo "TRAIN: ${TRAIN_FILE}"
echo ""

python - "${TRAIN_FILE}" "${VAL_FILE}" <<'PYEOF'
import sys

import pandas as pd

# 必须字段：可以在顶层列，也可以在 extra_info dict 内部
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
        raise SystemExit(
            f"{path} is missing required SchemaShift fields: {sorted(missing)}\n"
            "Fields must be either top-level columns or inside extra_info dict.\n"
"Regenerate data with:\n"
"  python scripts/generate_oval_data.py "
        )
    if "perturbation_level" in df.columns:
        levels = df["perturbation_level"].value_counts().to_dict()
    elif "extra_info" in df.columns:
        levels = pd.Series([x.get("perturbation_level", "none") for x in df["extra_info"]]).value_counts().to_dict()
    else:
        levels = {}
    if "scenario_type" in df.columns:
        scenarios = df["scenario_type"].value_counts().to_dict()
    elif "extra_info" in df.columns:
        scenarios = pd.Series([x.get("scenario_type", "unknown") for x in df["extra_info"]]).value_counts().to_dict()
    else:
        scenarios = {}
    print(f"{path}: perturbation_level={levels}, scenario_type={scenarios}")
print("GRPO parquet metadata columns: OK")
PYEOF

# ---- 启动 ----
CONDA_PYTHON="$(which python)"
"${CONDA_PYTHON}" "scripts/train_grpo.py" \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    data.max_prompt_length="${PROMPT_LENGTH}" \
    data.max_response_length="${RESPONSE_LENGTH}" \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.val_batch_size="${VAL_BATCH_SIZE}" \
    data.shuffle=False \
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
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.agent.default_agent_loop=schemashift_replay \
    actor_rollout_ref.rollout.agent.agent_loop_config_path=configs/agent_loop.yaml \
    actor_rollout_ref.rollout.agent.num_workers="${N_GPUS}" \
    data.return_raw_chat=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO_BATCH}" \
    actor_rollout_ref.ref.fsdp_config.param_offload="${PARAM_OFFLOAD}" \
    algorithm.adv_estimator=schemashift_grpo \
    algorithm.use_kl_in_reward=True \
    algorithm.kl_penalty=kl \
    algorithm.kl_ctrl.type=fixed \
    algorithm.kl_ctrl.kl_coef="${KL_COEF}" \
    custom_reward_function.path="${REWARD_FN_PATH}" \
    custom_reward_function.name=compute_score \
    trainer.project_name=schemashift_grpo \
    trainer.experiment_name=smoke_test \
    trainer.logger='["console"]' \
    trainer.total_epochs=1 \
    trainer.total_training_steps="${TOTAL_STEPS}" \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node="${N_GPUS}" \
    trainer.save_freq=-1 \
    trainer.val_before_train=False \
    trainer.test_freq=-1 \
    reward_model.enable=False \
    "${HYDRA_OVERRIDES[@]}" \
    2>&1 | tee "outputs/grpo_smoke.log"

echo ""
echo "=== Smoke Test Complete ==="
echo "Check outputs/grpo_smoke.log for results"
