#!/usr/bin/env bash
# E4: SchemaShift-GRPO — 混合扰动, schemashift_grpo estimator
#
# MODE=direct（默认）：直接 GRPO，使用原始 Qwen3-4B 权重
#   → configs/grpo_direct.yaml
# MODE=cold：SFT 冷启动 → GRPO，使用 SFT 产出权重
#   → configs/grpo_cold.yaml
#
# 自适应多卡：设置 N_GPUS 即可自动计算兼容的 batch size。
# E4 数据每 task 9 条记录（3 none + 3 mild + 3 strong），
# train_batch_size 必须为 9 的倍数，且与 GPU 数兼容。
set -euo pipefail

# ── 确保脚本退出时（正常/异常）清理 Ray 进程 ──
_cleanup_ray() {
    if ray status &>/dev/null 2>&1; then
        ray stop --force 2>/dev/null || true
    fi
}
trap _cleanup_ray EXIT

PROJECT_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$PROJECT_DIR"

# ── 命令行参数解析 ──
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            cat <<'EOF'
SchemaShift-GRPO E4 训练入口

用法:
  bash scripts/train/grpo/run_schemashift.sh [选项] [Hydra overrides...]

选项:
  --help, -h              显示此帮助信息
  --config PATH           指定 YAML 配置文件
                          默认: configs/grpo_direct.yaml (MODE=direct)
                                configs/grpo_cold.yaml (MODE=cold)

环境变量覆盖:
  MODE=direct|cold        选择训练路线 (默认: direct)
  EXP_NAME                实验名称
  MODEL_PATH              模型路径
  BETA                    Schemashift beta 参数
  N_GPUS                  GPU 数量
  TOTAL_STEPS             总训练步数
  SAVE_FREQ               保存频率
  TEST_FREQ               验证频率
  MICRO_BATCH_PER_GPU     micro batch size per GPU
  TRAINER_LOGGER          verl logger 列表，默认 ['console','wandb']
  EXTRA_HYDRA_OVERRIDES   额外 Hydra 覆盖参数

其余位置参数将作为 Hydra overrides 透传。
EOF
            exit 0
            ;;
        --config)
            if [ $# -lt 2 ]; then
                echo "❌ --config 需要指定配置文件路径" >&2
                exit 1
            fi
            EXP_CONFIG="$2"
            shift 2
            ;;
        *)
            EXTRA_HYDRA_OVERRIDES="${EXTRA_HYDRA_OVERRIDES:-} ${1}"
            shift
            ;;
    esac
done

# ── Ray 短临时目录（避免 plasma store AF_UNIX socket path 超 107 bytes）──
export TMPDIR="${TMPDIR:-/tmp/ssgrpo_tmp}"
export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/ssgrpo_ray}"
mkdir -p "${TMPDIR}" "${RAY_TMPDIR}"

MODE="${MODE:-direct}"

case "${MODE}" in
    direct)
        EXP_CONFIG="${EXP_CONFIG:-${PROJECT_DIR}/configs/grpo_direct.yaml}"
        ;;
    cold|cold_start)
        EXP_CONFIG="${EXP_CONFIG:-${PROJECT_DIR}/configs/grpo_cold.yaml}"
        ;;
    *)
        echo "❌ 未知 MODE=${MODE}，可选: direct, cold" >&2
        exit 1
        ;;
esac

_detect_n_gpus() {
    local gpu_list
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "0"
        return
    fi
    if ! gpu_list="$(nvidia-smi -L 2>/dev/null)"; then
        echo "0"
        return
    fi
    if [ -z "${gpu_list}" ]; then
        echo "0"
        return
    fi
    printf '%s\n' "${gpu_list}" | grep -c '^GPU '
}

N_GPUS="${N_GPUS:-$(_detect_n_gpus)}"
if ! [[ "${N_GPUS}" =~ ^[0-9]+$ ]] || [ "${N_GPUS}" -lt 1 ]; then
    echo "❌ 未检测到可用 GPU (N_GPUS=${N_GPUS})，无法启动训练" >&2
    echo "   请确认 NVIDIA driver 正常，并在 arl 环境运行: nvidia-smi -L" >&2
    exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-python}"

get() {
  "$PYTHON_BIN" - "$EXP_CONFIG" "$1" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
keys = sys.argv[2].split(".")
v = cfg
for k in keys:
    v = v[k]
print(v)
PY
}

EXP_NAME="${EXP_NAME:-$(get exp.name)}"
MODEL_PATH="${MODEL_PATH:-$(get model.path)}"
TRAIN_FILES="${PROJECT_DIR}/$(get data.train_file)"
VAL_FILES="${PROJECT_DIR}/$(get data.val_file)"
MAX_PROMPT_LEN=$(get data.max_prompt_length)
MAX_RESPONSE_LEN=$(get data.max_response_length)
GROUP_SIZE=$(get rollout.group_size)
MAX_TURNS=$(get rollout.max_turns)
AGENT_LOOP=$(get rollout.agent_loop)
AGENT_LOOP_CONFIG="${PROJECT_DIR}/$(get rollout.agent_loop_config)"
TP_SIZE=$(get rollout.tensor_parallel_size)
PPO_EPOCHS=$(get actor.ppo_epochs)
CLIP_RATIO=$(get actor.clip_ratio)
KL_COEF=$(get actor.kl_loss_coef)
ADV_EST=$(get algorithm.adv_estimator)
BETA="${BETA:-$(get algorithm.schemashift.beta)}"
TOTAL_STEPS="${TOTAL_STEPS:-$(get trainer.total_training_steps)}"
SAVE_FREQ="${SAVE_FREQ:-$(get trainer.save_freq)}"
TEST_FREQ="${TEST_FREQ:-$(get trainer.test_freq)}"
PROJECT_NAME=$(get trainer.project_name)
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-$(get trainer.val_before_train)}"
TRAINER_LOGGER="${TRAINER_LOGGER:-['console','wandb']}"

# ── batch size 自适应 ──
MICRO_BATCH_PER_GPU="${MICRO_BATCH_PER_GPU:-$(get actor.ppo_micro_batch_size_per_gpu)}"
MINI_BATCH_SIZE=$((N_GPUS * MICRO_BATCH_PER_GPU))
TRAIN_BATCH_SIZE=$("$PYTHON_BIN" -c "
import math
mbs = $MINI_BATCH_SIZE
g = 9
print(mbs * g // math.gcd(mbs, g))
")
VAL_BATCH_SIZE=$((N_GPUS * 1))
LOG_PROB_MICRO_BATCH="${LOG_PROB_MICRO_BATCH:-4}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${PROJECT_DIR}/logs/${EXP_NAME}_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

export SCHEMASHIFT_BETA="${BETA}"
export PYTHONPATH="${PROJECT_DIR}:${PROJECT_DIR}/verl:${PYTHONPATH:-}"

# vLLM 0.11 + flashinfer 0.6.4 + CUDA 11.8 不兼容，默认走 flash_attn 路径
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"

echo "E4: SchemaShift start | mode=${MODE} | model=${MODEL_PATH} | gpus=${N_GPUS} | micro=${MICRO_BATCH_PER_GPU} | train_batch=${TRAIN_BATCH_SIZE} | mini_batch=${MINI_BATCH_SIZE} | beta=${BETA} | config=${EXP_CONFIG}" \
  | tee "${LOG_DIR}/train.log"

"$PYTHON_BIN" src/training/run_grpo.py \
    data.train_files="${TRAIN_FILES}" \
    data.val_files="${VAL_FILES}" \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.val_batch_size="${VAL_BATCH_SIZE}" \
    data.max_prompt_length=${MAX_PROMPT_LEN} \
    data.max_response_length=${MAX_RESPONSE_LEN} \
    data.prompt_key="prompt" \
    data.return_raw_chat=True \
    data.shuffle=False \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.ppo_mini_batch_size="${MINI_BATCH_SIZE}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${MICRO_BATCH_PER_GPU}" \
    actor_rollout_ref.actor.ppo_epochs=${PPO_EPOCHS} \
    actor_rollout_ref.actor.clip_ratio=${CLIP_RATIO} \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=${KL_COEF} \
    actor_rollout_ref.actor.loss_agg_mode="token-mean" \
    \
    actor_rollout_ref.rollout.name="vllm" \
    actor_rollout_ref.rollout.mode="async" \
    actor_rollout_ref.rollout.agent.default_agent_loop="${AGENT_LOOP}" \
    actor_rollout_ref.rollout.agent.agent_loop_config_path="${AGENT_LOOP_CONFIG}" \
    actor_rollout_ref.rollout.agent.num_workers=${N_GPUS} \
    actor_rollout_ref.rollout.n="${GROUP_SIZE}" \
    +actor_rollout_ref.rollout.max_turns="${MAX_TURNS}" \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${TP_SIZE} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH} \
    \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH} \
    \
    algorithm.adv_estimator="${ADV_EST}" \
    algorithm.norm_adv_by_std_in_grpo=True \
    +algorithm.beta="${BETA}" \
    \
    reward_model.enable=False \
    custom_reward_function.path="${PROJECT_DIR}/src/reward/schemashift_reward_fn.py" \
    custom_reward_function.name=compute_score \
    \
    trainer.total_epochs=1 \
    trainer.n_gpus_per_node=${N_GPUS} \
    trainer.total_training_steps="${TOTAL_STEPS}" \
    trainer.default_local_dir="${PROJECT_DIR}/checkpoints/${EXP_NAME}" \
    trainer.save_freq="${SAVE_FREQ}" \
    trainer.test_freq="${TEST_FREQ}" \
    trainer.logger="${TRAINER_LOGGER}" \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXP_NAME}" \
    trainer.val_before_train=${VAL_BEFORE_TRAIN} \
    ${EXTRA_HYDRA_OVERRIDES:-} \
    2>&1 | tee -a "${LOG_DIR}/train.log"

echo "E4 完成"
