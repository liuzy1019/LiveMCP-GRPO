#!/usr/bin/env bash
# E4: SchemaShift-GRPO — 混合扰动, schemashift_grpo estimator
# Config: configs/exp4_schemashift.yaml（可通过 EXP_CONFIG 覆盖）
#
# 自适应多卡：设置 N_GPUS 即可自动计算兼容的 batch size。
# E4 数据每 task 9 条记录（3 none + 3 mild + 3 strong），
# train_batch_size 必须为 9 的倍数，且与 GPU 数兼容。
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$PROJECT_DIR"

N_GPUS="${N_GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l)}"
if [ "${N_GPUS}" -lt 1 ] 2>/dev/null || [ -z "${N_GPUS}" ]; then
    echo "❌ 未检测到 GPU (N_GPUS=${N_GPUS})，无法启动训练" >&2
    exit 1
fi

EXP_CONFIG="${EXP_CONFIG:-${PROJECT_DIR}/configs/exp4_schemashift.yaml}"
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
PPO_EPOCHS=$(get actor.ppo_epochs)
CLIP_RATIO=$(get actor.clip_ratio)
KL_COEF=$(get actor.kl_loss_coef)
ADV_EST=$(get algorithm.adv_estimator)
BETA="${BETA:-$(get algorithm.schemashift.beta)}"
TOTAL_STEPS="${TOTAL_STEPS:-$(get trainer.total_training_steps)}"
SAVE_FREQ="${SAVE_FREQ:-$(get trainer.save_freq)}"
TEST_FREQ="${TEST_FREQ:-$(get trainer.test_freq)}"
PROJECT_NAME=$(get trainer.project_name)
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-True}"

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

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${PROJECT_DIR}/logs/${EXP_NAME}_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

export SCHEMASHIFT_BETA="${BETA}"
export PYTHONPATH="${PROJECT_DIR}:${PROJECT_DIR}/verl:${PYTHONPATH:-}"

# vLLM 0.11 + flashinfer 0.6.4 + CUDA 11.8 不兼容，默认走 flash_attn 路径
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"

echo "E4: SchemaShift start | model=${MODEL_PATH} | gpus=${N_GPUS} | micro=${MICRO_BATCH_PER_GPU} | train_batch=${TRAIN_BATCH_SIZE} | mini_batch=${MINI_BATCH_SIZE} | beta=${BETA} | config=${EXP_CONFIG}" \
  | tee "${LOG_DIR}/train.log"

"$PYTHON_BIN" src/training/run_exp4.py \
    data.train_files="${TRAIN_FILES}" \
    data.val_files="${VAL_FILES}" \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.val_batch_size="${VAL_BATCH_SIZE}" \
    data.max_prompt_length=${MAX_PROMPT_LEN} \
    data.max_response_length=${MAX_RESPONSE_LEN} \
    data.prompt_key="prompt" \
    data.return_raw_chat=True \
    data.shuffle=False \
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
    actor_rollout_ref.rollout.tensor_model_parallel_size=${TP_SIZE} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    \
    algorithm.adv_estimator="${ADV_EST}" \
    algorithm.norm_adv_by_std_in_grpo=True \
    +algorithm.beta="${BETA}" \
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

echo "E4 完成"
