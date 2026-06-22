#!/usr/bin/env bash
# E5: Aug Only — 混合扰动 schema + 标准 GRPO
# Config: configs/exp5_aug_only.yaml（可通过 EXP_CONFIG 覆盖）
# 3 records/task (none/mild/strong), rollout.n=3 per record
# 每层 3 个 rollout → 组内比较有效 → 标准 GRPO 有信号
# 与 E4 对比: 同样的 schema 混合，不同的 advantage 计算方式
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$PROJECT_DIR"

N_GPUS="${N_GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l)}"
if [ "${N_GPUS}" -lt 1 ] 2>/dev/null || [ -z "${N_GPUS}" ]; then
    echo "❌ 未检测到 GPU (N_GPUS=${N_GPUS})，无法启动训练" >&2
    exit 1
fi

EXP_CONFIG="${EXP_CONFIG:-${PROJECT_DIR}/configs/exp5_aug_only.yaml}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/data1/zhanyiliu/liuzhanyi/anaconda3/envs/arl/bin/python}"

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
PPO_MICRO_BS=$(get actor.ppo_micro_batch_size_per_gpu)
PPO_EPOCHS=$(get actor.ppo_epochs)
CLIP_RATIO=$(get actor.clip_ratio)
KL_COEF=$(get actor.kl_loss_coef)
ADV_EST=$(get algorithm.adv_estimator)
TOTAL_STEPS="${TOTAL_STEPS:-$(get trainer.total_training_steps)}"
SAVE_FREQ="${SAVE_FREQ:-$(get trainer.save_freq)}"
TEST_FREQ="${TEST_FREQ:-$(get trainer.test_freq)}"
PROJECT_NAME=$(get trainer.project_name)
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-True}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${PROJECT_DIR}/logs/${EXP_NAME}_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

export PYTHONPATH="${PROJECT_DIR}:${PROJECT_DIR}/verl:${PYTHONPATH:-}"

# vLLM 0.11 + flashinfer 0.6.4 + CUDA 11.8 不兼容，默认走 flash_attn 路径
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"

echo "E5: Aug Only start | model=${MODEL_PATH} | gpus=${N_GPUS} | group=${GROUP_SIZE} | standard GRPO | config=${EXP_CONFIG}" \
  | tee "${LOG_DIR}/train.log"

if [ ! -f "${TRAIN_FILES}" ] || [ ! -f "${VAL_FILES}" ]; then
    echo "❌ E5 数据缺失：${TRAIN_FILES} 或 ${VAL_FILES} 不存在" >&2
    echo "   请先运行：${PYTHON_BIN} scripts/build_parquet.py" >&2
    exit 1
fi

"$PYTHON_BIN" src/training/run_exp3.py \
    data.train_files="${TRAIN_FILES}" \
    data.val_files="${VAL_FILES}" \
    data.train_batch_size=$((N_GPUS * 2)) \
    data.val_batch_size=$((N_GPUS * 1)) \
    data.max_prompt_length=${MAX_PROMPT_LEN} \
    data.max_response_length=${MAX_RESPONSE_LEN} \
    data.prompt_key="prompt" \
    data.return_raw_chat=True \
    \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$((N_GPUS * 2)) \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BS} \
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
    actor_rollout_ref.rollout.tensor_model_parallel_size=${TP_SIZE} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    \
    algorithm.adv_estimator="${ADV_EST}" \
    algorithm.norm_adv_by_std_in_grpo=True \
    \
    reward_model.enable=False \
    \
    trainer.total_epochs=1 \
    trainer.n_gpus_per_node=${N_GPUS} \
    trainer.total_training_steps="${TOTAL_STEPS}" \
    trainer.default_local_dir="${PROJECT_DIR}/checkpoints/${EXP_NAME}" \
    trainer.save_freq="${SAVE_FREQ}" \
    trainer.test_freq="${TEST_FREQ}" \
    trainer.logger="['console','wandb']" \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXP_NAME}" \
    trainer.val_before_train=${VAL_BEFORE_TRAIN} \
    ${EXTRA_HYDRA_OVERRIDES:-} \
    2>&1 | tee -a "${LOG_DIR}/train.log"

echo "E5 完成"
