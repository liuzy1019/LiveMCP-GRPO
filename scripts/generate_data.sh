#!/bin/bash
# Unified data generation for LiveMCP-GRPO.
#
# Defaults to the validated 4-GPU Teacher-generation profile, detects model
# size from config.json, compares with GPU memory,
# and picks the optimal parallel strategy:
#   - Small model (fits 1 GPU) → local transformers, 1 process per GPU
#   - Large model (needs TP) → vLLM API server(s), 1 process per instance
#
# Usage:
#   bash scripts/generate_data.sh --count 500 --val-count 100
#   bash scripts/generate_data.sh --domain calendar --count 200
#   GPU_COUNT=8 bash scripts/generate_data.sh --model models/Qwen/Qwen3-8B --count 200
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
# Keep vLLM's writable runtime state off the user-home filesystem.  The model
# and generated data remain project-relative; these directories only contain
# disposable compiler/config state.  Callers may override either path.
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-${TMPDIR:-/tmp}/livemcp_vllm_cache}"
export VLLM_CONFIG_ROOT="${VLLM_CONFIG_ROOT:-${TMPDIR:-/tmp}/livemcp_vllm_config}"
export VLLM_NO_USAGE_STATS="${VLLM_NO_USAGE_STATS:-1}"
mkdir -p "${VLLM_CACHE_ROOT}" "${VLLM_CONFIG_ROOT}"
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
SUITE="configs/live_mcp/ten_domain_suite.yaml"
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
export LIVEMCP_TEACHER_TRACE_PATH="${LIVEMCP_TEACHER_TRACE_PATH:-logs/${RUN_ID}_teacher_trace.jsonl}"
exec > >(tee -a "${MAIN_LOG}") 2>&1

# ── GPU detection (via shared gpu_config.sh) ────────────────────────
# The current formal Teacher-generation baseline is 4×A10 / one TP=4 vLLM
# instance.  Callers can still opt into another resource count explicitly.
GPU_COUNT="${GPU_COUNT:-4}"
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
    sleep 2
    # SIGKILL stragglers
    for pid in "${VLLM_PIDS[@]}"; do
        if [ -n "${pid:-}" ] && kill -0 "$pid" 2>/dev/null; then
            kill -KILL "$pid" 2>/dev/null || true
        fi
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
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-$HOME}"
export NVCC_APPEND_FLAGS=-allow-unsupported-compiler
DEPENDENCY_CACHE_PREWARM="${DEPENDENCY_CACHE_PREWARM:-1}"
if [[ "${DEPENDENCY_CACHE_PREWARM}" != "0" && "${DEPENDENCY_CACHE_PREWARM}" != "1" ]]; then
    echo "ERROR: DEPENDENCY_CACHE_PREWARM must be 0 or 1, got ${DEPENDENCY_CACHE_PREWARM}" >&2
    exit 1
fi
GENERATION_CLIENT_SEED_STRIDE="${GENERATION_CLIENT_SEED_STRIDE:-1000000}"
GENERATION_MAX_RECOVERY_ROUNDS="${GENERATION_MAX_RECOVERY_ROUNDS:-3}"
MERGE_TOPUP_ROUNDS="${MERGE_TOPUP_ROUNDS:-3}"
if ! [[ "${GENERATION_MAX_RECOVERY_ROUNDS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: GENERATION_MAX_RECOVERY_ROUNDS must be >= 1, got ${GENERATION_MAX_RECOVERY_ROUNDS}" >&2
    exit 1
fi
if ! [[ "${GENERATION_CLIENT_SEED_STRIDE}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: GENERATION_CLIENT_SEED_STRIDE must be >= 1, got ${GENERATION_CLIENT_SEED_STRIDE}" >&2
    exit 1
fi
if ! [[ "${MERGE_TOPUP_ROUNDS}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: MERGE_TOPUP_ROUNDS must be >= 0, got ${MERGE_TOPUP_ROUNDS}" >&2
    exit 1
fi
MAX_RECOVERY_SEED_OFFSET=$(( (GENERATION_MAX_RECOVERY_ROUNDS - 1) * 100000 + 90000 ))
if [ "${GENERATION_CLIENT_SEED_STRIDE}" -le "${MAX_RECOVERY_SEED_OFFSET}" ]; then
    echo "ERROR: GENERATION_CLIENT_SEED_STRIDE must exceed max recovery offset ${MAX_RECOVERY_SEED_OFFSET}, got ${GENERATION_CLIENT_SEED_STRIDE}" >&2
    exit 1
fi

merge_vllm_with_topups() {
    local deficits_path="${TMPDIR_SHARD}/merge_deficits.json"
    local topup_round=0
    while ! "${PYTHON_BIN}" scripts/merge_generation_shards.py \
        --tmpdir "${TMPDIR_SHARD}" \
        --output-dir "${RUN_DIR}" \
        --count "${COUNT}" \
        --val-count "${VAL_COUNT}" \
        --domain "${DOMAIN}" \
        --deficits-output "${deficits_path}"; do
        if [ "${topup_round}" -ge "${MERGE_TOPUP_ROUNDS}" ]; then
            echo "ERROR: global merge still has deficits after ${topup_round} top-up round(s)" >&2
            return 1
        fi
        topup_round=$((topup_round + 1))
        local -a topup_deficits=()
        mapfile -t topup_deficits < <("${PYTHON_BIN}" -c '
import json, sys
report = json.load(open(sys.argv[1], encoding="utf-8"))
for domain, missing in sorted(report.get("deficits", {}).items()):
    if domain != "__all__" and int(missing) > 0:
        suggested = int(
            report.get("suggested_topup_by_domain", {}).get(
                domain, int(missing) + max(2, (int(missing) + 1) // 2)
            )
        )
        print(f"{domain}\t{int(missing)}\t{suggested}")
' "${deficits_path}")
        if [ "${#topup_deficits[@]}" -eq 0 ]; then
            echo "ERROR: merge failed but reported no domain-specific deficit" >&2
            return 1
        fi

        echo "Top-up round ${topup_round}: ${#topup_deficits[@]} deficit domain(s)"
        local -a topup_pids=()
        local topup_index=0
        local total_topup_slots="${TOTAL_GEN_CLIENTS:-${NUM_INSTANCES}}"
        local slots_per_domain=$(( total_topup_slots / ${#topup_deficits[@]} ))
        if [ "${slots_per_domain}" -lt 1 ]; then slots_per_domain=1; fi
        local entry topup_domain missing topup_count topup_inst topup_port topup_seed topup_prefix
        local chunk_count chunk_index chunk_base chunk_remainder chunk_size
        for entry in "${topup_deficits[@]}"; do
            IFS=$'\t' read -r topup_domain missing topup_count <<< "${entry}"
            chunk_count="${slots_per_domain}"
            if [ "${topup_count}" -lt "${chunk_count}" ]; then chunk_count="${topup_count}"; fi
            chunk_base=$(( topup_count / chunk_count ))
            chunk_remainder=$(( topup_count % chunk_count ))
            echo "  [${topup_domain}] missing=${missing}, generating=${topup_count} across ${chunk_count} shard(s)"
            for ((chunk_index=0; chunk_index<chunk_count; chunk_index++)); do
                chunk_size="${chunk_base}"
                if [ "${chunk_index}" -lt "${chunk_remainder}" ]; then
                    chunk_size=$((chunk_size + 1))
                fi
                topup_inst=$((topup_index % NUM_INSTANCES))
                topup_port=$((PORT_START + topup_inst))
                topup_seed=$((SEED + (TOTAL_GEN_CLIENTS + 1 + topup_round * total_topup_slots + topup_index) * GENERATION_CLIENT_SEED_STRIDE))
                topup_prefix="topup_${topup_round}_${topup_domain}_${chunk_index}"
                echo "    shard=${chunk_index}, count=${chunk_size}, port=${topup_port}"
                "${PYTHON_BIN}" scripts/generate_data.py \
                    --count "${chunk_size}" \
                    --val-count 0 \
                    --seed "${topup_seed}" \
                    --domain "${topup_domain}" \
                    --model "${SERVED_MODEL}" \
                    --api-base "http://localhost:${topup_port}/v1" \
                    --suite "${SUITE}" \
                    --shard-mode \
                    --pool-oversample-pct 0 \
                    --irrelevance-ratio 0 \
                    --max-recovery-rounds "${GENERATION_MAX_RECOVERY_ROUNDS}" \
                    --checkpoint-path "${TMPDIR_SHARD}/shard_${topup_prefix}_checkpoint.json" \
                    --output "${TMPDIR_SHARD}/shard_${topup_prefix}_train.parquet" \
                    --val-output "${TMPDIR_SHARD}/shard_${topup_prefix}_val.parquet" \
                    --log-file "${TMPDIR_SHARD}/shard_${topup_prefix}.log" \
                    > "${TMPDIR_SHARD}/shard_${topup_prefix}.stdout" 2>&1 &
                topup_pids+=($!)
                topup_index=$((topup_index + 1))
            done
        done
        local topup_failed=0
        local i
        for i in "${!topup_pids[@]}"; do
            wait "${topup_pids[$i]}" || {
                echo "  [top-up $i] FAILED" >&2
                topup_failed=$((topup_failed + 1))
            }
        done
        if [ "${topup_failed}" -gt 0 ]; then
            echo "WARNING: ${topup_failed}/${#topup_pids[@]} top-up processes failed; preserving successful candidate shards and recomputing global deficits" >&2
        fi
    done
}

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
export OVAL_SUITE_PATH="${SUITE}"
    echo ""
    echo "Strategy: LOCAL — ${GPU_COUNT} parallel processes, 1 per GPU"
    echo "Generation workers: ${GENERATION_WORKERS_PER_PROCESS} per process"

    GEN_COUNT=$(( (COUNT * (100 + GEN_OVERSAMPLE_PCT) + 99) / 100 ))
    GEN_VAL_COUNT=$(( (VAL_COUNT * (100 + GEN_OVERSAMPLE_PCT) + 99) / 100 ))
    BASE_GPU_TRAIN=$(( GEN_COUNT / GPU_COUNT ))
    REM_GPU_TRAIN=$(( GEN_COUNT % GPU_COUNT ))
    BASE_GPU_VAL=$(( GEN_VAL_COUNT / GPU_COUNT ))
    REM_GPU_VAL=$(( GEN_VAL_COUNT % GPU_COUNT ))
    TMPDIR_SHARD="${TMPDIR:-/tmp}/livemcp_gen_$$"
    mkdir -p "${TMPDIR_SHARD}"

    if [ "${DEPENDENCY_CACHE_PREWARM}" = "1" ]; then
        echo ""
        echo "Prewarming dependency graph cache for domain=${DOMAIN}..."
        CUDA_VISIBLE_DEVICES="${GPU_INDEX_ARRAY[0]}" \
            "${PYTHON_BIN}" scripts/build_dependency_cache.py \
                --domain "${DOMAIN}" \
                --model "${MODEL}" \
                --suite "${SUITE}" \
                --device 0
    fi

    PIDS=()
    for ((i=0; i<GPU_COUNT; i++)); do
        GPU_ID="${GPU_INDEX_ARRAY[$i]}"
        SHARD_SEED=$((SEED + i * GENERATION_CLIENT_SEED_STRIDE))
        SHARD_TRAIN=$(( BASE_GPU_TRAIN + (i < REM_GPU_TRAIN ? 1 : 0) ))
        SHARD_VAL=$(( BASE_GPU_VAL + (i < REM_GPU_VAL ? 1 : 0) ))

        echo "  [shard $i] GPU=${GPU_ID}, train=${SHARD_TRAIN}, val=${SHARD_VAL}, seed=${SHARD_SEED}"

        if [ $((SHARD_TRAIN + SHARD_VAL)) -eq 0 ]; then
            echo "  [shard $i] skipped (zero quota)"
            continue
        fi

        CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" scripts/generate_data.py \
            --count "${SHARD_TRAIN}" \
            --val-count "${SHARD_VAL}" \
            --seed "${SHARD_SEED}" \
            --domain "${DOMAIN}" \
            --model "${MODEL}" \
            --suite "${SUITE}" \
            --shard-mode \
            --pool-oversample-pct 0 \
            --max-recovery-rounds "${GENERATION_MAX_RECOVERY_ROUNDS}" \
            --checkpoint-path "${TMPDIR_SHARD}/shard_${i}_checkpoint.json" \
            --output "${TMPDIR_SHARD}/shard_${i}_train.parquet" \
            --val-output "${TMPDIR_SHARD}/shard_${i}_val.parquet" \
            --log-file "${TMPDIR_SHARD}/shard_${i}.log" \
            --device 0 &
        PIDS+=($!)
    done

    echo ""
    ACTIVE_SHARDS=${#PIDS[@]}
    echo "Waiting for ${ACTIVE_SHARDS} active processes..."
    FAILED=0
    for i in "${!PIDS[@]}"; do
        wait "${PIDS[$i]}" || { echo "  [shard $i] FAILED" >&2; FAILED=$((FAILED + 1)); }
    done

    if [ "$FAILED" -gt 0 ]; then
        echo "ERROR: ${FAILED}/${ACTIVE_SHARDS} active shards failed" >&2
        exit 1
    fi

    "${PYTHON_BIN}" scripts/merge_generation_shards.py \
        --tmpdir "${TMPDIR_SHARD}" \
        --output-dir "${RUN_DIR}" \
        --count "${COUNT}" \
        --val-count "${VAL_COUNT}" \
        --domain "${DOMAIN}"
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
    BASE_CLIENT_TRAIN=$(( GEN_COUNT / TOTAL_GEN_CLIENTS ))
    REM_CLIENT_TRAIN=$(( GEN_COUNT % TOTAL_GEN_CLIENTS ))
    BASE_CLIENT_VAL=$(( GEN_VAL_COUNT / TOTAL_GEN_CLIENTS ))
    REM_CLIENT_VAL=$(( GEN_VAL_COUNT % TOTAL_GEN_CLIENTS ))
    TMPDIR_SHARD="${TMPDIR:-/tmp}/livemcp_gen_$$"
    mkdir -p "${TMPDIR_SHARD}"

    # Start vLLM instances
    for ((inst=0; inst<NUM_INSTANCES; inst++)); do
        GPU_START=$(( inst * TP_SIZE ))
        GPU_END=$(( GPU_START + TP_SIZE - 1 ))
        GPU_SLICE=("${GPU_INDEX_ARRAY[@]:$GPU_START:$TP_SIZE}")
        GPU_LIST=$(IFS=','; echo "${GPU_SLICE[*]}")
        PORT=$(( PORT_START + inst ))
        EXISTING_PORT_PIDS="$(_pids_listening_on_port "${PORT}")"
        if [ -n "${EXISTING_PORT_PIDS}" ]; then
            echo "ERROR: vLLM port ${PORT} is already in use by pid(s): ${EXISTING_PORT_PIDS}" >&2
            exit 1
        fi
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
        # Gemma-4 is multimodal, but the Teacher pipeline is text-only.  Disable every
        # multimodal input (including video), otherwise vLLM profiles an encoder cache
        # and leaves too little KV cache for concurrent 8192-token requests on A10.
        if [[ "${MODEL}" == *"Gemma-4"* ]]; then
            VLLM_ARGS+=(--language-model-only)
        fi
        if [ "${VLLM_ENFORCE_EAGER}" = "1" ]; then
            VLLM_ARGS+=(--enforce-eager)
        fi

        # These variables configure this launcher; vLLM receives the resolved
        # values through CLI flags. Do not leak launcher-only names into
        # vLLM's reserved VLLM_* environment namespace.
        env \
            -u VLLM_CLIENTS_PER_INSTANCE \
            -u VLLM_NUM_INSTANCES \
            -u VLLM_MAX_NUM_SEQS \
            -u VLLM_MAX_NUM_BATCHED_TOKENS \
            -u VLLM_MAX_MODEL_LEN \
            -u VLLM_GPU_MEMORY_UTILIZATION \
            -u VLLM_ENFORCE_EAGER \
            -u VLLM_PORT_START \
            CUDA_VISIBLE_DEVICES="${GPU_LIST}" \
            "${PYTHON_BIN}" "${VLLM_ARGS[@]}" > "${LOG}" 2>&1 &
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
        "${PYTHON_BIN}" scripts/build_dependency_cache.py \
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
            SHARD_TRAIN=$(( BASE_CLIENT_TRAIN + (CLIENT_ID < REM_CLIENT_TRAIN ? 1 : 0) ))
            SHARD_VAL=$(( BASE_CLIENT_VAL + (CLIENT_ID < REM_CLIENT_VAL ? 1 : 0) ))

            echo "  Instance ${inst}/client ${client}: train=${SHARD_TRAIN}, val=${SHARD_VAL}, seed=${SHARD_SEED}"

            if [ $((SHARD_TRAIN + SHARD_VAL)) -eq 0 ]; then
                echo "  Instance ${inst}/client ${client}: skipped (zero quota)"
                continue
            fi

            "${PYTHON_BIN}" scripts/generate_data.py \
                --count "${SHARD_TRAIN}" \
                --val-count "${SHARD_VAL}" \
                --seed "${SHARD_SEED}" \
                --domain "${DOMAIN}" \
                --model "${SERVED_MODEL}" \
                --api-base "http://localhost:${PORT}/v1" \
                --suite "${SUITE}" \
                --shard-mode \
                --pool-oversample-pct 0 \
                --max-recovery-rounds "${GENERATION_MAX_RECOVERY_ROUNDS}" \
                --checkpoint-path "${TMPDIR_SHARD}/shard_${inst}_${client}_checkpoint.json" \
                --output "${TMPDIR_SHARD}/shard_${inst}_${client}_train.parquet" \
                --val-output "${TMPDIR_SHARD}/shard_${inst}_${client}_val.parquet" \
                --log-file "${TMPDIR_SHARD}/shard_${inst}_${client}.log" \
                > "${TMPDIR_SHARD}/shard_${inst}_${client}.stdout" 2>&1 &
            GEN_PIDS+=($!)
        done
    done

    echo ""
    ACTIVE_GEN_CLIENTS=${#GEN_PIDS[@]}
    echo "Waiting for ${ACTIVE_GEN_CLIENTS} active generation processes..."
    FAILED=0
    for i in "${!GEN_PIDS[@]}"; do
        wait "${GEN_PIDS[$i]}" || { echo "  [Instance $i] FAILED" >&2; FAILED=$((FAILED + 1)); }
    done

    if [ "$FAILED" -gt 0 ]; then
        echo "ERROR: ${FAILED}/${ACTIVE_GEN_CLIENTS} active generation processes failed" >&2
        exit 1
    fi

    merge_vllm_with_topups
    rm -f "${TMPDIR_SHARD}"/shard_*_train.parquet "${TMPDIR_SHARD}"/shard_*_val.parquet
fi

# ── Update symlinks & print stats ──────────────────────────────────
echo ""
echo "=== Generation Complete ==="
echo "Run dir:       ${RUN_DIR}/"
echo "Train parquet: ${RUN_DIR}/train.parquet"
echo "Val parquet:   ${RUN_DIR}/val.parquet"

# ── Parquet integrity validation ────────────────────────────────────
echo ""
echo "=== Parquet Integrity Check ==="
if ! "${PYTHON_BIN}" scripts/audit_generated_data.py \
    "${RUN_DIR}/train.parquet" "${RUN_DIR}/val.parquet"; then
    echo "ERROR: Parquet integrity check failed. See above for details." >&2
    exit 1
fi

# Publish default training inputs only after both parquet files pass all gates.
ln -sfn "runs/${RUN_ID}/train.parquet" "${OUTPUT_DIR}/train.parquet"
ln -sfn "runs/${RUN_ID}/val.parquet"   "${OUTPUT_DIR}/val.parquet"
echo "Symlinks:      ${OUTPUT_DIR}/train.parquet → runs/${RUN_ID}/train.parquet"
echo "               ${OUTPUT_DIR}/val.parquet   → runs/${RUN_ID}/val.parquet"
GEN_SUCCESS=1

echo ""
echo "Done. [$(date '+%Y-%m-%d %H:%M:%S')]"
