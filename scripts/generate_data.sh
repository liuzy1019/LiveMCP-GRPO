#!/bin/bash
# Unified data generation for LiveMCP-GRPO.
#
# Auto-detects model size from config.json, compares with GPU memory,
# and picks the optimal parallel strategy:
#   - Small model (fits 1 GPU) → local transformers, 1 process per GPU
#   - Large model (needs TP) → vLLM API server(s), 1 process per instance
#
# Usage:
#   bash scripts/generate_data.sh --count 500 --val-count 100
#   bash scripts/generate_data.sh --model gemini-2.5-flash --api-base https://your-gemini-proxy/v1 --count 500 --val-count 100
#   bash scripts/generate_data.sh --domain calendar --count 200
#   GPU_COUNT=4 bash scripts/generate_data.sh --model models/Qwen/Qwen3-8B --count 200
#
# Env override:
#   OUTPUT_DIR=data  GPU_COUNT=8  VLLM_PORT_START=8001
#   VLLM_CLIENTS_PER_INSTANCE=4  VLLM_MAX_NUM_SEQS=16
#   GENERATION_WORKERS_PER_PROCESS=2  DEPENDENCY_CACHE_PREWARM=1

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${PROJECT_ROOT}"

if [ -z "${PYTHON_BIN:-}" ]; then
    if [ -n "${CONDA_PREFIX:-}" ] && [ -x "${CONDA_PREFIX}/bin/python" ]; then
        PYTHON_BIN="${CONDA_PREFIX}/bin/python"
    else
        PYTHON_BIN="$(which python3 2>/dev/null || which python 2>/dev/null || echo python)"
    fi
fi
export PYTHON_BIN
export PYTHONNOUSERSITE=1
PYTHON_BIN_DIR="$(cd "$(dirname "${PYTHON_BIN}")" && pwd)"
PYTHON_PREFIX="$(cd "${PYTHON_BIN_DIR}/.." && pwd)"
export PATH="${PYTHON_BIN_DIR}:${PATH}"

# Keep CUDA JIT compilation aligned with the selected Python environment.
# vLLM/FlashInfer otherwise falls back to /usr/local/cuda, which can point to
# a stale system toolkit or an older CUDA version than the torch wheel.
if [ -x "${PYTHON_PREFIX}/bin/nvcc" ]; then
    export CUDA_HOME="${CUDA_HOME:-${PYTHON_PREFIX}}"
    export CUDA_PATH="${CUDA_PATH:-${CUDA_HOME}}"
    export FLASHINFER_NVCC="${FLASHINFER_NVCC:-${CUDA_HOME}/bin/nvcc}"
fi
NVIDIA_SITE="${PYTHON_PREFIX}/lib/python3.11/site-packages/nvidia"
CUDA_INCLUDE_DIRS=()
if [ -d "${NVIDIA_SITE}" ]; then
    while IFS= read -r inc_dir; do
        CUDA_INCLUDE_DIRS+=("${inc_dir}")
    done < <(find "${NVIDIA_SITE}" -maxdepth 3 -type d -name include | sort)
fi
if [ "${#CUDA_INCLUDE_DIRS[@]}" -gt 0 ]; then
    CUDA_INCLUDE_PATH="$(IFS=:; echo "${CUDA_INCLUDE_DIRS[*]}")"
    export CPATH="${CUDA_INCLUDE_PATH}${CPATH:+:${CPATH}}"
fi
CUDA_LIBRARY_DIRS=()
for lib_dir in \
    "${PYTHON_PREFIX}/lib" \
    "${PYTHON_PREFIX}/targets/x86_64-linux/lib" \
    "${PYTHON_PREFIX}/targets/x86_64-linux/lib/stubs"
do
    if [ -d "${lib_dir}" ]; then
        CUDA_LIBRARY_DIRS+=("${lib_dir}")
    fi
done
if [ "${#CUDA_LIBRARY_DIRS[@]}" -gt 0 ]; then
    CUDA_LIBRARY_PATH="$(IFS=:; echo "${CUDA_LIBRARY_DIRS[*]}")"
    export LIBRARY_PATH="${CUDA_LIBRARY_PATH}${LIBRARY_PATH:+:${LIBRARY_PATH}}"
    export LD_LIBRARY_PATH="${CUDA_LIBRARY_PATH}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

# ── Parse args ─────────────────────────────────────────────────────
MODEL="models/Google/Gemma-4-31B-it"
COUNT=5000
VAL_COUNT=500
DOMAIN="all"
SUITE="configs/live_mcp/suite_mvp.yaml"
SEED=42
OUTPUT_DIR="${OUTPUT_DIR:-data}"
GEN_OVERSAMPLE_PCT="${GEN_OVERSAMPLE_PCT:-10}"  # 10% oversample; set GEN_OVERSAMPLE_PCT env var to override
RUN_ID="${RUN_ID:-$(date +%m%d_%H%M)}"

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)          MODEL="$2";          shift 2 ;;
        --model=*)        MODEL="${1#*=}";     shift ;;
        --count)          COUNT="$2";          shift 2 ;;
        --count=*)        COUNT="${1#*=}";     shift ;;
        --val-count)      VAL_COUNT="$2";      shift 2 ;;
        --val-count=*)    VAL_COUNT="${1#*=}"; shift ;;
        --domain)         DOMAIN="$2";         shift 2 ;;
        --domain=*)       DOMAIN="${1#*=}";    shift ;;
        --suite)          SUITE="$2";          shift 2 ;;
        --suite=*)        SUITE="${1#*=}";     shift ;;
        --output-dir)     OUTPUT_DIR="$2";     shift 2 ;;
        --output-dir=*)   OUTPUT_DIR="${1#*=}"; shift ;;
        --seed)           SEED="$2";           shift 2 ;;
        --seed=*)         SEED="${1#*=}";      shift ;;
        --run-id)         RUN_ID="$2";          shift 2 ;;
        --run-id=*)       RUN_ID="${1#*=}";    shift ;;
        *) echo "ERROR: unknown arg: $1" >&2; exit 1 ;;
    esac
done

# ── Output dirs ────────────────────────────────────────────────────
RUN_DIR="${OUTPUT_DIR}/runs/${RUN_ID}"
mkdir -p "${RUN_DIR}"
mkdir -p logs

# 主日志：tee 到 logs/
MAIN_LOG="logs/${RUN_ID}_gen_${COUNT}.log"
exec > >(tee -a "${MAIN_LOG}") 2>&1

# ── GPU detection (via shared gpu_config.sh) ────────────────────────
source scripts/gpu_config.sh
GPU_MEM_GB=${GPU_MEM_GB:-0}

echo "============================================"
echo "LiveMCP-GRPO Data Generation"
echo "============================================"
echo "Model:    ${MODEL}"
echo "GPUs:     ${GPU_COUNT}x ${GPU_MODEL} (${GPU_MEM_GB}GB)"
echo "Target:   ${COUNT} train + ${VAL_COUNT} val"
echo "Oversample candidates: +${GEN_OVERSAMPLE_PCT}% before quality merge"
echo "Domain:   ${DOMAIN}"
echo "Run ID:   ${RUN_ID}"
echo "Output:   ${RUN_DIR}/"
echo "Log:      ${MAIN_LOG}"
echo "============================================"

# ── Detect model size & decide strategy ────────────────────────────
# Resolve model path: absolute path → as-is, relative → PROJECT_ROOT prefix
if [[ "$MODEL" = /* ]]; then
    MODEL_PATH="$MODEL"
else
    MODEL_PATH="${PROJECT_ROOT}/${MODEL}"
fi

MODEL_INFO=$("${PYTHON_BIN}" -c "
import json, sys
try:
    cfg_path = '${MODEL_PATH}/config.json'
    with open(cfg_path) as f:
        c = json.load(f)
    # Some models (e.g. Gemma-4) nest text params under 'text_config'
    tc = c.get('text_config', {})
    n  = tc.get('num_hidden_layers') or c.get('num_hidden_layers', 0)
    d  = tc.get('hidden_size')        or c.get('hidden_size', 0)
    di = tc.get('intermediate_size')  or c.get('intermediate_size', 0)
    v  = tc.get('vocab_size')         or c.get('vocab_size', 0)
    nh = tc.get('num_attention_heads') or c.get('num_attention_heads', 0)
    # Rough param count (attention + FFN + embedding)
    params = n * (4*d*d + 3*d*di) + v*d
    bf16_gb = params * 2 / 1e9
    print(f'{params/1e9:.1f} {bf16_gb:.1f} {nh}')
except Exception as e:
    print(f'ERROR {e}', file=sys.stderr)
    sys.exit(1)
" 2>/dev/null)

if [ -z "$MODEL_INFO" ] || [[ "$MODEL_INFO" == ERROR* ]]; then
    echo "ERROR: Cannot read model config: ${MODEL_PATH}/config.json" >&2
    exit 1
fi

MODEL_PARAMS_B=$(echo "$MODEL_INFO" | awk '{print $1}')
MODEL_BF16_GB=$(echo "$MODEL_INFO" | awk '{print $2}')
MODEL_NUM_HEADS=$(echo "$MODEL_INFO" | awk '{print $3}')
echo ""
echo "Model: ${MODEL_PARAMS_B}B params (~${MODEL_BF16_GB} GB BF16), ${MODEL_NUM_HEADS} heads"

# Heuristic: model fits if BF16 size < 70% of single GPU memory
FITS_SINGLE_GPU=$("${PYTHON_BIN}" -c "
fits = ${MODEL_BF16_GB} < ${GPU_MEM_GB} * 0.70
print('1' if fits else '0')
")

# ── Cleanup trap ────────────────────────────────────────────────────
VLLM_PIDS=()
VLLM_PORTS=()
VLLM_LOGS=()
GEN_SUCCESS=0
_pids_listening_on_port() {
    local port="$1"
    ss -ltnp "sport = :${port}" 2>/dev/null \
        | sed -n 's/.*pid=\([0-9]\+\).*/\1/p' \
        | sort -u
}

_cleanup() {
    local exit_code=$?
    echo "[cleanup] stopping vLLM processes and releasing ports..." >&2
    # SIGTERM first
    for pid in "${VLLM_PIDS[@]}"; do
        if [ -n "${pid:-}" ] && kill -0 "$pid" 2>/dev/null; then
            kill -TERM "$pid" 2>/dev/null || true
        fi
    done
    for port in "${VLLM_PORTS[@]}"; do
        for pid in $(_pids_listening_on_port "${port}"); do
            if [ -n "${pid:-}" ] && kill -0 "$pid" 2>/dev/null; then
                echo "[cleanup] SIGTERM port ${port} pid=${pid}" >&2
                kill -TERM "$pid" 2>/dev/null || true
            fi
        done
    done
    sleep 2
    # SIGKILL stragglers
    for pid in "${VLLM_PIDS[@]}"; do
        if [ -n "${pid:-}" ] && kill -0 "$pid" 2>/dev/null; then
            kill -KILL "$pid" 2>/dev/null || true
        fi
    done
    for port in "${VLLM_PORTS[@]}"; do
        for pid in $(_pids_listening_on_port "${port}"); do
            if [ -n "${pid:-}" ] && kill -0 "$pid" 2>/dev/null; then
                echo "[cleanup] SIGKILL port ${port} pid=${pid}" >&2
                kill -KILL "$pid" 2>/dev/null || true
            fi
        done
    done
    # 成功时删除 vLLM 日志；失败时保留用于排查
    if [ "${GEN_SUCCESS}" = "1" ]; then
        for log in "${VLLM_LOGS[@]}"; do
            [ -f "${log}" ] && rm -f "${log}" && echo "[cleanup] removed vLLM log: ${log}" >&2
        done
    else
        if [ ${#VLLM_LOGS[@]} -gt 0 ]; then
            echo "[cleanup] generation failed — vLLM logs preserved for debugging:" >&2
            for log in "${VLLM_LOGS[@]}"; do
                [ -f "${log}" ] && echo "  ${log}" >&2
            done
        fi
    fi
    exit $exit_code
}
trap _cleanup EXIT INT TERM

# ── Environment ────────────────────────────────────────────────────
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-$HOME}"
export NVCC_APPEND_FLAGS=-allow-unsupported-compiler
DEPENDENCY_CACHE_PREWARM="${DEPENDENCY_CACHE_PREWARM:-1}"
if [[ "${DEPENDENCY_CACHE_PREWARM}" != "0" && "${DEPENDENCY_CACHE_PREWARM}" != "1" ]]; then
    echo "ERROR: DEPENDENCY_CACHE_PREWARM must be 0 or 1, got ${DEPENDENCY_CACHE_PREWARM}" >&2
    exit 1
fi
GENERATION_CLIENT_SEED_STRIDE="${GENERATION_CLIENT_SEED_STRIDE:-1000000}"
if ! [[ "${GENERATION_CLIENT_SEED_STRIDE}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: GENERATION_CLIENT_SEED_STRIDE must be >= 1, got ${GENERATION_CLIENT_SEED_STRIDE}" >&2
    exit 1
fi
if [ "${GENERATION_CLIENT_SEED_STRIDE}" -le 200000 ]; then
    echo "ERROR: GENERATION_CLIENT_SEED_STRIDE must exceed the 200000 recovery range, got ${GENERATION_CLIENT_SEED_STRIDE}" >&2
    exit 1
fi

# ═══════════════════════════════════════════════════════════════════
# MODE 1: Local transformers — 1 process per GPU
# ═══════════════════════════════════════════════════════════════════
if [ "$FITS_SINGLE_GPU" = "1" ]; then
    GENERATION_WORKERS_PER_PROCESS="${GENERATION_WORKERS_PER_PROCESS:-1}"
    if ! [[ "${GENERATION_WORKERS_PER_PROCESS}" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: GENERATION_WORKERS_PER_PROCESS must be >= 1, got ${GENERATION_WORKERS_PER_PROCESS}" >&2
        exit 1
    fi
    export LIVEMCP_GENERATION_MAX_WORKERS="${GENERATION_WORKERS_PER_PROCESS}"
    echo ""
    echo "Strategy: LOCAL — ${GPU_COUNT} parallel processes, 1 per GPU"
    echo "Generation workers: ${GENERATION_WORKERS_PER_PROCESS} per process"

    GEN_COUNT=$(( (COUNT * (100 + GEN_OVERSAMPLE_PCT) + 99) / 100 ))
    GEN_VAL_COUNT=$(( (VAL_COUNT * (100 + GEN_OVERSAMPLE_PCT) + 99) / 100 ))
    PER_GPU_TRAIN=$(( (GEN_COUNT + GPU_COUNT - 1) / GPU_COUNT ))
    PER_GPU_VAL=$(( (GEN_VAL_COUNT + GPU_COUNT - 1) / GPU_COUNT ))
    TMPDIR_SHARD="${TMPDIR:-/tmp}/livemcp_gen_$$"
    mkdir -p "${TMPDIR_SHARD}"

    if [ "${DEPENDENCY_CACHE_PREWARM}" = "1" ]; then
        echo ""
        echo "Prewarming dependency graph cache for domain=${DOMAIN}..."
        CUDA_VISIBLE_DEVICES="${GPU_INDEX_ARRAY[0]}" \
            "${PYTHON_BIN}" scripts/dependency_graph.py live \
                --domain "${DOMAIN}" \
                --model "${MODEL}" \
                --suite "${SUITE}" \
                --device 0
    fi

    PIDS=()
    for ((i=0; i<GPU_COUNT; i++)); do
        GPU_ID="${GPU_INDEX_ARRAY[$i]}"
        SHARD_SEED=$((SEED + i * GENERATION_CLIENT_SEED_STRIDE))

        echo "  [shard $i] GPU=${GPU_ID}, train=${PER_GPU_TRAIN}, val=${PER_GPU_VAL}, seed=${SHARD_SEED}"

        CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" scripts/generate_data.py \
            --count "${PER_GPU_TRAIN}" \
            --val-count "${PER_GPU_VAL}" \
            --seed "${SHARD_SEED}" \
            --domain "${DOMAIN}" \
            --model "${MODEL}" \
            --suite "${SUITE}" \
            --output "${TMPDIR_SHARD}/shard_${i}_train.parquet" \
            --val-output "${TMPDIR_SHARD}/shard_${i}_val.parquet" \
            --log-file "${TMPDIR_SHARD}/shard_${i}.log" \
            --device 0 &
        PIDS+=($!)
    done

    echo ""
    echo "Waiting for ${GPU_COUNT} processes..."
    FAILED=0
    for i in "${!PIDS[@]}"; do
        wait "${PIDS[$i]}" || { echo "  [shard $i] FAILED" >&2; FAILED=$((FAILED + 1)); }
    done

    if [ "$FAILED" -gt 0 ]; then
        echo "ERROR: ${FAILED}/${GPU_COUNT} shards failed" >&2
        exit 1
    fi

    "${PYTHON_BIN}" scripts/merge_rollout_shards.py \
        --tmpdir "${TMPDIR_SHARD}" \
        --output-dir "${RUN_DIR}" \
        --count "${COUNT}" \
        --val-count "${VAL_COUNT}"
    rm -f "${TMPDIR_SHARD}"/shard_*_train.parquet "${TMPDIR_SHARD}"/shard_*_val.parquet

# ═════════════════════════════════════════════════════════════════
# MODE 2: vLLM API — TP across multiple GPUs
# ═════════════════════════════════════════════════════════════════
else
VLLM_VERSION=$("${PYTHON_BIN}" -c "import vllm; print(vllm.__version__)" 2>/dev/null || true)
    if [ -z "${VLLM_VERSION}" ]; then
        echo "ERROR: vLLM is not importable from ${PYTHON_BIN}" >&2
        exit 1
    fi
    # Auto-detect current vLLM version. Only warn if major version differs from last-known-good.
    # The environment dictates the version; we don't force a specific one.
    RECOMMENDED_VLLM_MAJOR_MINOR="0.19"
    VLLM_MAJOR_MINOR=$(echo "${VLLM_VERSION}" | cut -d. -f1,2)
    if [ -n "${EXPECTED_VLLM_VERSION:-}" ]; then
        # Explicit override: user knows what they're doing, bypass check.
        :
    elif [ "${VLLM_MAJOR_MINOR}" != "${RECOMMENDED_VLLM_MAJOR_MINOR}" ]; then
        echo "WARNING: vLLM ${VLLM_VERSION} differs from last-known-good ${RECOMMENDED_VLLM_MAJOR_MINOR}.x" >&2
        echo "         Data generation should still work. If you see errors, set EXPECTED_VLLM_VERSION to suppress this." >&2
    fi

    # Calculate optimal TP and number of vLLM instances.
    # vLLM requires TP to divide num_attention_heads evenly
    # (VLLM_GPU_MEMORY_UTILIZATION is set below by dynamic scaling; TP only needs a rough estimate)
    VLLM_TP_UTIL="${VLLM_GPU_MEMORY_UTILIZATION:-0.88}"
    TP_SIZE=$("${PYTHON_BIN}" -c "
import math
mem_need = ${MODEL_BF16_GB}
mem_gpu = ${GPU_MEM_GB}
num_heads = ${MODEL_NUM_HEADS}
util = ${VLLM_TP_UTIL}
tp = max(1, math.ceil(mem_need / (mem_gpu * util)))
# Ensure TP divides num_heads (vLLM requirement)
if num_heads > 0:
    while tp > 1 and num_heads % tp != 0:
        tp += 1
    if num_heads % tp != 0:
        # Fallback: find the largest divisor of num_heads >= tp
        for d in range(tp, num_heads + 1):
            if num_heads % d == 0:
                tp = d
                break
print(tp)
")

    NUM_INSTANCES=$(( GPU_COUNT / TP_SIZE ))
    if [ -n "${VLLM_NUM_INSTANCES:-}" ]; then
        if [ "${VLLM_NUM_INSTANCES}" -lt 1 ]; then
            echo "ERROR: VLLM_NUM_INSTANCES must be >= 1, got ${VLLM_NUM_INSTANCES}" >&2
            exit 1
        fi
        if [ "${VLLM_NUM_INSTANCES}" -gt "${NUM_INSTANCES}" ]; then
            echo "ERROR: VLLM_NUM_INSTANCES=${VLLM_NUM_INSTANCES} requires $(( VLLM_NUM_INSTANCES * TP_SIZE )) GPUs, have ${GPU_COUNT}" >&2
            exit 1
        fi
        NUM_INSTANCES="${VLLM_NUM_INSTANCES}"
    fi
    if [ "$NUM_INSTANCES" -lt 1 ]; then
        echo "ERROR: Need ${TP_SIZE} GPUs for TP=${TP_SIZE}, have ${GPU_COUNT}" >&2
        exit 1
    fi

    PORT_START="${VLLM_PORT_START:-8001}"

    # ── vLLM generation defaults ──
    # All parameters can be overridden via environment variables.
    VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-16384}"
    VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.88}"
    VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"
    VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-8}"
    VLLM_CLIENTS_PER_INSTANCE="${VLLM_CLIENTS_PER_INSTANCE:-8}"
    VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-0}"
    if ! [[ "${VLLM_MAX_NUM_SEQS}" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: VLLM_MAX_NUM_SEQS must be >= 1, got ${VLLM_MAX_NUM_SEQS}" >&2
        exit 1
    fi
    if ! [[ "${VLLM_CLIENTS_PER_INSTANCE}" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: VLLM_CLIENTS_PER_INSTANCE must be >= 1, got ${VLLM_CLIENTS_PER_INSTANCE}" >&2
        exit 1
    fi
    AUTO_GENERATION_WORKERS=$(( VLLM_MAX_NUM_SEQS / VLLM_CLIENTS_PER_INSTANCE ))
    if [ "${AUTO_GENERATION_WORKERS}" -lt 1 ]; then
        AUTO_GENERATION_WORKERS=1
    fi
    GENERATION_WORKERS_PER_PROCESS="${GENERATION_WORKERS_PER_PROCESS:-${AUTO_GENERATION_WORKERS}}"
    if ! [[ "${GENERATION_WORKERS_PER_PROCESS}" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: GENERATION_WORKERS_PER_PROCESS must be >= 1, got ${GENERATION_WORKERS_PER_PROCESS}" >&2
        exit 1
    fi
    export LIVEMCP_GENERATION_MAX_WORKERS="${GENERATION_WORKERS_PER_PROCESS}"
    TOTAL_REQUEST_WORKERS_PER_INSTANCE=$(( VLLM_CLIENTS_PER_INSTANCE * GENERATION_WORKERS_PER_PROCESS ))
    if [ "${TOTAL_REQUEST_WORKERS_PER_INSTANCE}" -gt "${VLLM_MAX_NUM_SEQS}" ]; then
        echo "WARNING: generation request workers per instance (${TOTAL_REQUEST_WORKERS_PER_INSTANCE}) exceed VLLM_MAX_NUM_SEQS=${VLLM_MAX_NUM_SEQS}; excess requests will queue" >&2
    fi

    echo ""
    echo "Strategy: vLLM API — TP=${TP_SIZE}, ${NUM_INSTANCES} instance(s)"
    echo "vLLM: version=${VLLM_VERSION}, gpu_memory_utilization=${VLLM_GPU_MEMORY_UTILIZATION}, max_model_len=${VLLM_MAX_MODEL_LEN}, max_num_seqs=${VLLM_MAX_NUM_SEQS}, max_num_batched_tokens=${VLLM_MAX_NUM_BATCHED_TOKENS}, enforce_eager=${VLLM_ENFORCE_EAGER}"
    echo "Clients: ${VLLM_CLIENTS_PER_INSTANCE} generation process(es) per vLLM instance"
    echo "Generation workers: ${GENERATION_WORKERS_PER_PROCESS} per process"
    echo "Dependency cache prewarm: ${DEPENDENCY_CACHE_PREWARM}"

    TOTAL_GEN_CLIENTS=$(( NUM_INSTANCES * VLLM_CLIENTS_PER_INSTANCE ))
    GEN_COUNT=$(( (COUNT * (100 + GEN_OVERSAMPLE_PCT) + 99) / 100 ))
    GEN_VAL_COUNT=$(( (VAL_COUNT * (100 + GEN_OVERSAMPLE_PCT) + 99) / 100 ))
    PER_CLIENT_TRAIN=$(( (GEN_COUNT + TOTAL_GEN_CLIENTS - 1) / TOTAL_GEN_CLIENTS ))
    PER_CLIENT_VAL=$(( (GEN_VAL_COUNT + TOTAL_GEN_CLIENTS - 1) / TOTAL_GEN_CLIENTS ))
    TMPDIR_SHARD="${TMPDIR:-/tmp}/livemcp_gen_$$"
    mkdir -p "${TMPDIR_SHARD}"

    # Start vLLM instances
    for ((inst=0; inst<NUM_INSTANCES; inst++)); do
        GPU_START=$(( inst * TP_SIZE ))
        GPU_END=$(( GPU_START + TP_SIZE - 1 ))
        GPU_SLICE=("${GPU_INDEX_ARRAY[@]:$GPU_START:$TP_SIZE}")
        GPU_LIST=$(IFS=','; echo "${GPU_SLICE[*]}")
        PORT=$(( PORT_START + inst ))
        VLLM_PORTS+=("${PORT}")
        LOG="logs/${RUN_ID}_vllm_instance${inst}.log"
        VLLM_LOGS+=("${LOG}")

        # Derive served model name from directory name (for local vLLM).
        # When using Gemini or other cloud APIs, the model name is passed directly.
        SERVED_MODEL="$(basename "${MODEL}")"
        if [[ "$SERVED_MODEL" == Qwen* && "$SERVED_MODEL" != *Instruct* ]]; then
            SERVED_MODEL="${SERVED_MODEL}-Instruct"
        fi

        echo "  Starting vLLM instance ${inst} on GPUs ${GPU_LIST}, port ${PORT}"

        VLLM_ARGS=(
            -m vllm.entrypoints.openai.api_server
            --model "${MODEL}" \
            --served-model-name "${SERVED_MODEL}" \
            --tensor-parallel-size "${TP_SIZE}" \
            --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
            --max-model-len "${VLLM_MAX_MODEL_LEN}" \
            --max-num-seqs "${VLLM_MAX_NUM_SEQS}" \
            --max-num-batched-tokens "${VLLM_MAX_NUM_BATCHED_TOKENS}" \
            --port "${PORT}" \
            --trust-remote-code
        )
        # Gemma-4 is a multimodal model; vLLM allocates extra encoder cache for vision.
        # We only use text generation — disable vision encoder to reclaim memory on A10.
        if [[ "${MODEL}" == *"Gemma-4"* ]]; then
            VLLM_ARGS+=(--limit-mm-per-prompt '{"image": 0}')
        fi
        if [ "${VLLM_ENFORCE_EAGER}" = "1" ]; then
            VLLM_ARGS+=(--enforce-eager)
        fi

        CUDA_VISIBLE_DEVICES="${GPU_LIST}" "${PYTHON_BIN}" "${VLLM_ARGS[@]}" > "${LOG}" 2>&1 &
        VLLM_PIDS+=($!)
    done

    # Wait for all instances
    echo ""
    echo "Waiting for vLLM instances to be ready..."
    MAX_WAIT=600

    for ((inst=0; inst<NUM_INSTANCES; inst++)); do
        PORT=$(( PORT_START + inst ))
        PID="${VLLM_PIDS[$inst]}"
        SERVED_MODEL="$(basename "${MODEL}")"
        if [[ "$SERVED_MODEL" == Qwen* && "$SERVED_MODEL" != *Instruct* ]]; then
            SERVED_MODEL="${SERVED_MODEL}-Instruct"
        fi
        WAITED=0
        while [ $WAITED -lt $MAX_WAIT ]; do
            if ! kill -0 "${PID}" 2>/dev/null; then
                echo "ERROR: vLLM instance ${inst} exited during startup; see ${LOG}" >&2
                exit 1
            fi
            MODELS_JSON=$(curl -sf "http://localhost:${PORT}/v1/models" 2>/dev/null || true)
            if [[ "${MODELS_JSON}" == *"\"id\":\"${SERVED_MODEL}\""* ]] || \
               [[ "${MODELS_JSON}" == *"\"id\": \"${SERVED_MODEL}\""* ]]; then
                echo "  Instance ${inst} (port ${PORT}) ready after ${WAITED}s"
                break
            fi
            sleep 10
            WAITED=$((WAITED + 10))
        done
        if [ $WAITED -ge $MAX_WAIT ]; then
            echo "ERROR: Instance ${inst} not ready after ${MAX_WAIT}s" >&2
            exit 1
        fi
    done

    if [ "${DEPENDENCY_CACHE_PREWARM}" = "1" ]; then
        echo ""
        echo "Prewarming dependency graph cache for domain=${DOMAIN}..."
        "${PYTHON_BIN}" scripts/dependency_graph.py live \
            --domain "${DOMAIN}" \
            --model "${SERVED_MODEL}" \
            --api-base "http://localhost:${PORT_START}/v1" \
            --suite "${SUITE}"
    fi

    # Generate
    echo ""
    echo "Generating data (${NUM_INSTANCES} instance(s) in parallel)..."

    GEN_PIDS=()
    for ((inst=0; inst<NUM_INSTANCES; inst++)); do
        PORT=$(( PORT_START + inst ))
        for ((client=0; client<VLLM_CLIENTS_PER_INSTANCE; client++)); do
            CLIENT_ID=$(( inst * VLLM_CLIENTS_PER_INSTANCE + client ))
            SHARD_SEED=$((SEED + CLIENT_ID * GENERATION_CLIENT_SEED_STRIDE))

            echo "  Instance ${inst}/client ${client}: train=${PER_CLIENT_TRAIN}, val=${PER_CLIENT_VAL}, seed=${SHARD_SEED}"

            "${PYTHON_BIN}" scripts/generate_data.py \
                --count "${PER_CLIENT_TRAIN}" \
                --val-count "${PER_CLIENT_VAL}" \
                --seed "${SHARD_SEED}" \
                --domain "${DOMAIN}" \
                --model "${SERVED_MODEL}" \
                --api-base "http://localhost:${PORT}/v1" \
                --suite "${SUITE}" \
                --output "${TMPDIR_SHARD}/shard_${inst}_${client}_train.parquet" \
                --val-output "${TMPDIR_SHARD}/shard_${inst}_${client}_val.parquet" \
                --log-file "${TMPDIR_SHARD}/shard_${inst}_${client}.log" \
                > "${TMPDIR_SHARD}/shard_${inst}_${client}.stdout" 2>&1 &
            GEN_PIDS+=($!)
        done
    done

    echo ""
    echo "Waiting for ${TOTAL_GEN_CLIENTS} generation processes..."
    FAILED=0
    for i in "${!GEN_PIDS[@]}"; do
        wait "${GEN_PIDS[$i]}" || { echo "  [Instance $i] FAILED" >&2; FAILED=$((FAILED + 1)); }
    done

    if [ "$FAILED" -gt 0 ]; then
        echo "ERROR: ${FAILED}/${TOTAL_GEN_CLIENTS} generation processes failed" >&2
        exit 1
    fi

    "${PYTHON_BIN}" scripts/merge_rollout_shards.py \
        --tmpdir "${TMPDIR_SHARD}" \
        --output-dir "${RUN_DIR}" \
        --count "${COUNT}" \
        --val-count "${VAL_COUNT}"
    rm -f "${TMPDIR_SHARD}"/shard_*_train.parquet "${TMPDIR_SHARD}"/shard_*_val.parquet
fi

# ── Update symlinks & print stats ──────────────────────────────────
GEN_SUCCESS=1
echo ""
echo "=== Generation Complete ==="
echo "Run dir:       ${RUN_DIR}/"
echo "Train parquet: ${RUN_DIR}/train.parquet"
echo "Val parquet:   ${RUN_DIR}/val.parquet"

# 更新 data/train.parquet 和 data/val.parquet 符号链接指向最新 run
ln -sfn "runs/${RUN_ID}/train.parquet" "${OUTPUT_DIR}/train.parquet"
ln -sfn "runs/${RUN_ID}/val.parquet"   "${OUTPUT_DIR}/val.parquet"
echo "Symlinks:      ${OUTPUT_DIR}/train.parquet → runs/${RUN_ID}/train.parquet"
echo "               ${OUTPUT_DIR}/val.parquet   → runs/${RUN_ID}/val.parquet"

# ── Parquet integrity validation ────────────────────────────────────
echo ""
echo "=== Parquet Integrity Check ==="
"${PYTHON_BIN}" -c "
import json, sys
import pandas as pd

issues = 0
for label, path in [('train', '${RUN_DIR}/train.parquet'), ('val', '${RUN_DIR}/val.parquet')]:
    try:
        df = pd.read_parquet(path)
    except Exception as e:
        print(f'  FAIL {label}: cannot read parquet — {e}')
        issues += 1
        continue

    print(f'  {label}: {len(df)} rows')

    for col in ('prompt', 'reward_model', 'extra_info', 'scenario_type'):
        if col not in df.columns:
            print(f'    FAIL: missing column {col}')
            issues += 1

    if issues > 0:
        continue

    # Abstain scenarios (clarification_required, missing_function,
    # no_tool_or_abstention, irrelevant) are expected to have empty
    # success_criteria — don't flag them.
    ABSTAIN_SCENARIOS = {'clarification_required', 'missing_function', 'no_tool_or_abstention', 'irrelevant'}

    empty_oc = 0
    empty_sc = 0
    bad_prompt = 0
    for i, row in df.iterrows():
        gt = row['reward_model'].get('ground_truth', {})
        oc = gt.get('oracle_calls', '')
        sc = gt.get('success_criteria', '')
        if not oc or (isinstance(oc, str) and oc in ('[]', '')):
            empty_oc += 1
        st = row.get('scenario_type', '')
        if st not in ABSTAIN_SCENARIOS:
            if not sc or (isinstance(sc, str) and sc in ('[]', '')):
                empty_sc += 1
        try:
            json.loads(row['prompt'])
        except (json.JSONDecodeError, TypeError):
            bad_prompt += 1

    if empty_oc:
        print(f'    WARN: {empty_oc} rows have empty oracle_calls')
    if empty_sc:
        print(f'    WARN: {empty_sc} rows have empty success_criteria (non-abstain scenarios)')
    if bad_prompt:
        print(f'    FAIL: {bad_prompt} rows have invalid prompt JSON')
        issues += 1

    # Spot-check: first row _build_task_dict
    try:
        from src.reward.oval_reward_fn import _build_task_dict
        first_extra = df.iloc[0]['extra_info']
        td = _build_task_dict(first_extra)
        if not isinstance(td, dict) or 'required_tool_calls' not in td:
            print(f'    FAIL: _build_task_dict spot-check returned invalid dict')
            issues += 1
    except Exception as e:
        print(f'    FAIL: _build_task_dict spot-check crashed — {e}')
        issues += 1

if issues:
    print(f'\n  Parquet validation FAILED ({issues} issue(s))')
    sys.exit(1)
else:
    print(f'  Parquet validation PASSED')
"

if [ $? -ne 0 ]; then
    echo "ERROR: Parquet integrity check failed. See above for details." >&2
    exit 1
fi

echo ""
echo "Done. [$(date '+%Y-%m-%d %H:%M:%S')]"
