#!/bin/bash
# OVAL-MCP GRPO smoke test — live MCP execution + audit + safety reward
# 自适应 8×A10 23GB 配置

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${PROJECT_ROOT}"

# ---- 环境 ----
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export HYDRA_FULL_ERROR=0
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
export RAY_DEDUP_LOGS=1
export SCHEMASHIFT_CONSOLE_LOG_MODE=compact
export SCHEMASHIFT_VAL_NUM_EXAMINE=0
export SCHEMASHIFT_VERBOSE_VALIDATION=0
export LOGURU_LEVEL=INFO
unset PYTORCH_CUDA_ALLOC_CONF 2>/dev/null || true
export PYTHONWARNINGS="ignore:.*FSDP\\.state_dict_type\\(\\).*:FutureWarning"
export TMPDIR="${TMPDIR:-/tmp/ssgrpo_tmp}"
export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/ssgrpo_ray}"
export SCHEMASHIFT_RAY_TMPDIR="${SCHEMASHIFT_RAY_TMPDIR:-${RAY_TMPDIR}}"
mkdir -p "${TMPDIR}" "${RAY_TMPDIR}" outputs

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

# ---- A10 23GB 固定配置 ----
N_GPUS=8
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
MODEL_PATH="outputs/sft_cold_start_4b/final"
TRAIN_FILE="data/oval_grpo_train.parquet"
VAL_FILE="data/oval_grpo_val.parquet"
REWARD_FN_PATH="src/reward/oval_reward_fn.py"
TOTAL_STEPS=2
LR=1e-6
LR_WARMUP_RATIO=0.1
KL_COEF=0.01
PPO_EPOCHS=1
GRAD_CLIP=1.0
ROLLOUT_N=4
TEMPERATURE=0.7
TOP_P=0.95

ROLLOUT_TP=1
GPU_MEM_UTIL=0.50
FREE_CACHE_ENGINE=True
ENFORCE_EAGER=True
MICRO_BATCH=1
PROMPT_LENGTH=10240
RESPONSE_LENGTH=1024
PARAM_OFFLOAD=True
ACTOR_PARAM_OFFLOAD=True
LOG_PROB_MICRO_BATCH=1
MAX_NUM_SEQS=8

TRAIN_BATCH_SIZE=$((N_GPUS * 4))
VAL_BATCH_SIZE=$((N_GPUS * 2))
MINI_BATCH_SIZE=${N_GPUS}

echo "============================================"
echo "OVAL-MCP GRPO Smoke Test (8×A10 23GB)"
echo "============================================"
echo "MODEL: ${MODEL_PATH}"
echo "TRAIN: ${TRAIN_FILE} (64 samples)"
echo "VAL:   ${VAL_FILE} (16 samples)"
echo "REWARD: ${REWARD_FN_PATH}"
echo "AGENT:  schemashift_oval (live MCP)"
echo "============================================"

# ---- 注册 estimator ----
export PYTHONPATH=".:${PYTHONPATH:-}"

# ---- 验证数据 ----
echo ""
echo "=== Validating data ==="
python - "${TRAIN_FILE}" "${VAL_FILE}" <<'PYEOF'
import sys, pandas as pd
for path in sys.argv[1:]:
    df = pd.read_parquet(path)
    print(f"  {path}: {len(df)} rows, cols={list(df.columns)}")
    ei = df.iloc[0]["extra_info"]
    print(f"    domain={ei.get('domain')}, tools={ei.get('required_tools')}")
PYEOF
echo ""

# ---- 启动 ----
CONDA_PYTHON="/mnt/data1/zhanyiliu/liuzhanyi/anaconda3/envs/arl/bin/python"
"${CONDA_PYTHON}" "scripts/train_grpo.py" \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    data.max_prompt_length="${PROMPT_LENGTH}" \
    data.max_response_length="${RESPONSE_LENGTH}" \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.val_batch_size="${VAL_BATCH_SIZE}" \
    data.shuffle=False \
    data.filter_overlong_prompts=False \
    data.truncation=left \
    data.reward_fn_key=data_source \
    data.return_raw_chat=True \
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
    actor_rollout_ref.rollout.agent.default_agent_loop=schemashift_oval \
    actor_rollout_ref.rollout.agent.agent_loop_config_path=configs/agent_loop.yaml \
    actor_rollout_ref.rollout.agent.num_workers="${N_GPUS}" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO_BATCH}" \
    actor_rollout_ref.ref.fsdp_config.param_offload="${PARAM_OFFLOAD}" \
    algorithm.adv_estimator=schemashift_grpo \
    algorithm.use_kl_in_reward=True \
    algorithm.kl_penalty=kl \
    algorithm.kl_ctrl.type=fixed \
    algorithm.kl_ctrl.kl_coef="${KL_COEF}" \
    custom_reward_function.path="${REWARD_FN_PATH}" \
    custom_reward_function.name=compute_score \
    trainer.project_name=oval_mcp_grpo \
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
    2>&1 | tee "outputs/oval_grpo_smoke.log"

echo ""
echo "=== OVAL Smoke Test Complete ==="
echo "Check outputs/oval_grpo_smoke.log for results"
