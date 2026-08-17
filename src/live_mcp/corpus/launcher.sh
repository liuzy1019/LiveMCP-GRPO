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
#   Internal only. Use: python -m src.live_mcp.corpus.cli run ...
#
# Env override:
#   OUTPUT_DIR=data  GPU_COUNT=8  VLLM_PORT_START=8001
#   VLLM_CLIENTS_PER_INSTANCE=4  VLLM_MAX_NUM_SEQS=16
#   GENERATION_WORKERS_PER_PROCESS=2  DEPENDENCY_CACHE_PREWARM=1
#   GENERATION_RESUME_CANDIDATE_DIR=data/runs/<run>/candidates
# Incremental jobs may use a larger initial candidate budget while requiring a
# smaller number of rows that are globally new relative to immutable base data:
#   --count 1100 --candidate-budget 3000 --base-train ... --base-val ...

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
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
# Separate Python for vLLM server (may need different transformers version for model)
export VLLM_PYTHON_BIN="${VLLM_PYTHON_BIN:-$PYTHON_BIN}"
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

# Build CUDA JIT extensions with the compiler settings of this environment.
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
GEN_OVERSAMPLE_PCT="${GEN_OVERSAMPLE_PCT:-10}"  # one launcher-level pool margin; shard-level oversample stays 0
LIVEMCP_FIXED_ATTEMPT_BUDGET="${LIVEMCP_FIXED_ATTEMPT_BUDGET:-0}"
if [[ "${LIVEMCP_FIXED_ATTEMPT_BUDGET}" != "0" \
    && "${LIVEMCP_FIXED_ATTEMPT_BUDGET}" != "1" ]]; then
    echo "ERROR: LIVEMCP_FIXED_ATTEMPT_BUDGET must be 0 or 1, got ${LIVEMCP_FIXED_ATTEMPT_BUDGET}" >&2
    exit 1
fi
RUN_ID="${RUN_ID:-$(date +%m%d_%H%M)}"
GENERATION_RESUME_CANDIDATE_DIR="${GENERATION_RESUME_CANDIDATE_DIR:-}"
GENERATION_PRESERVE_CANDIDATES="${GENERATION_PRESERVE_CANDIDATES:-0}"
CANDIDATE_BUDGET=""
BASE_TRAIN=""
BASE_VAL=""
TOOL_REQUIRED_ONLY=0
IRRELEVANCE_RATIO="0.05"
MISSING_FUNCTION_RATE="0.1210165389"
DISTRACTOR_RATE="0.40"
DIFFICULTY=""
CHECKPOINT_INTERVAL="${GENERATION_CHECKPOINT_INTERVAL:-25}"
CHAIN_BIN_QUOTAS=""
PUBLISH_ACTIVE=0
LIVEMCP_SEMANTIC_GATE_PROFILE="${LIVEMCP_SEMANTIC_GATE_PROFILE:-diagnostic_only}"
case "${LIVEMCP_SEMANTIC_GATE_PROFILE}" in
    diagnostic_only|deterministic_v1) ;;
    *)
        echo "ERROR: LIVEMCP_SEMANTIC_GATE_PROFILE must be diagnostic_only or deterministic_v1, got ${LIVEMCP_SEMANTIC_GATE_PROFILE}" >&2
        exit 1
        ;;
esac
export LIVEMCP_SEMANTIC_GATE_PROFILE

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
        --candidate-budget) CANDIDATE_BUDGET="$2"; shift 2 ;;
        --candidate-budget=*) CANDIDATE_BUDGET="${1#*=}"; shift ;;
        --base-train)     BASE_TRAIN="$2";     shift 2 ;;
        --base-train=*)   BASE_TRAIN="${1#*=}"; shift ;;
        --base-val)       BASE_VAL="$2";       shift 2 ;;
        --base-val=*)     BASE_VAL="${1#*=}"; shift ;;
        --tool-required-only) TOOL_REQUIRED_ONLY=1; shift ;;
        --irrelevance-ratio) IRRELEVANCE_RATIO="$2"; shift 2 ;;
        --irrelevance-ratio=*) IRRELEVANCE_RATIO="${1#*=}"; shift ;;
        --missing-function-rate) MISSING_FUNCTION_RATE="$2"; shift 2 ;;
        --missing-function-rate=*) MISSING_FUNCTION_RATE="${1#*=}"; shift ;;
        --distractor-rate) DISTRACTOR_RATE="$2"; shift 2 ;;
        --distractor-rate=*) DISTRACTOR_RATE="${1#*=}"; shift ;;
        --difficulty) DIFFICULTY="$2"; shift 2 ;;
        --difficulty=*) DIFFICULTY="${1#*=}"; shift ;;
        --checkpoint-interval) CHECKPOINT_INTERVAL="$2"; shift 2 ;;
        --checkpoint-interval=*) CHECKPOINT_INTERVAL="${1#*=}"; shift ;;
        --chain-bin-quotas) CHAIN_BIN_QUOTAS="$2"; shift 2 ;;
        --chain-bin-quotas=*) CHAIN_BIN_QUOTAS="${1#*=}"; shift ;;
        --publish)         PUBLISH_ACTIVE=1; shift ;;
        *) echo "ERROR: unknown arg: $1" >&2; exit 1 ;;
    esac
done

if [ -n "${CANDIDATE_BUDGET}" ]; then
    if ! [[ "${CANDIDATE_BUDGET}" =~ ^[1-9][0-9]*$ ]] \
        || [ "${CANDIDATE_BUDGET}" -lt "${COUNT}" ]; then
        echo "ERROR: --candidate-budget must be an integer >= --count" >&2
        exit 1
    fi
fi
if [ -n "${BASE_TRAIN}" ] || [ -n "${BASE_VAL}" ]; then
    if [ -z "${BASE_TRAIN}" ] || [ -z "${BASE_VAL}" ]; then
        echo "ERROR: --base-train and --base-val must be provided together" >&2
        exit 1
    fi
    if [ ! -f "${BASE_TRAIN}" ] || [ ! -f "${BASE_VAL}" ]; then
        echo "ERROR: base parquet not found: ${BASE_TRAIN}, ${BASE_VAL}" >&2
        exit 1
    fi
fi
if ! [[ "${CHECKPOINT_INTERVAL}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: --checkpoint-interval must be an integer >= 1" >&2
    exit 1
fi
"${PYTHON_BIN}" -c '
import sys
for name, raw in (
    ("irrelevance-ratio", sys.argv[1]),
    ("missing-function-rate", sys.argv[2]),
    ("distractor-rate", sys.argv[3]),
):
    value = float(raw)
    if not 0.0 <= value <= 1.0:
        raise SystemExit(f"ERROR: --{name} must be in [0, 1], got {raw}")
' "${IRRELEVANCE_RATIO}" "${MISSING_FUNCTION_RATE}" "${DISTRACTOR_RATE}"
if [ -n "${DIFFICULTY}" ] \
    && [ "${DIFFICULTY}" != "complete" ] \
    && [ "${DIFFICULTY}" != "missing" ] \
    && [ "${DIFFICULTY}" != "minimal" ]; then
    echo "ERROR: --difficulty must be complete, missing, or minimal" >&2
    exit 1
fi
INITIAL_TRAIN_CANDIDATES="${CANDIDATE_BUDGET:-${COUNT}}"
MERGE_BASE_ARGS=()
if [ -n "${BASE_TRAIN}" ]; then
    MERGE_BASE_ARGS=(
        --base-train "${BASE_TRAIN}"
        --base-val "${BASE_VAL}"
    )
fi
MERGE_SELECTION_ARGS=()
if [ -n "${CHAIN_BIN_QUOTAS}" ]; then
    MERGE_SELECTION_ARGS=(
        --chain-bin-quotas "${CHAIN_BIN_QUOTAS}"
    )
fi
MERGE_DIAGNOSTIC_ARGS=()
if [ "${LIVEMCP_FIXED_ATTEMPT_BUDGET}" = "1" ]; then
    MERGE_DIAGNOSTIC_ARGS=(--diagnostic-fixed-attempt)
fi
GENERATION_FILTER_ARGS=()
if [ "${TOOL_REQUIRED_ONLY}" = "1" ]; then
    GENERATION_FILTER_ARGS=(--require-tool-calls)
fi
GENERATION_DIFFICULTY_ARGS=()
if [ -n "${DIFFICULTY}" ]; then
    GENERATION_DIFFICULTY_ARGS=(--difficulty "${DIFFICULTY}")
fi

# ── Output dirs ────────────────────────────────────────────────────
RUN_DIR="${OUTPUT_DIR}/runs/${RUN_ID}"
mkdir -p "${RUN_DIR}"
mkdir -p logs

# 主日志：tee 到 logs/
MAIN_LOG="logs/${RUN_ID}_gen_${COUNT}.log"
export LIVEMCP_TEACHER_TRACE_PATH="${LIVEMCP_TEACHER_TRACE_PATH:-${RUN_DIR}/teacher_trace.jsonl}"
exec > >(tee -a "${MAIN_LOG}") 2>&1

# ── GPU detection (via shared gpu_config.sh) ────────────────────────
# The current formal Teacher-generation baseline is 4×A10 / one TP=4 vLLM
# instance.  Callers can still opt into another resource count explicitly.
GPU_COUNT="${GPU_COUNT:-4}"
# A retained exact-model service owns its GPU lease. Probe it before the
# free-GPU filter; otherwise its healthy allocation is misclassified as a
# resource conflict and the launcher never reaches the reuse branch below.
VLLM_PORT_START="${VLLM_PORT_START:-8001}"
SERVED_MODEL_PROBE="$(basename "${MODEL}")"
if [[ "${SERVED_MODEL_PROBE}" == Qwen* && "${SERVED_MODEL_PROBE}" != *Instruct* ]]; then
    SERVED_MODEL_PROBE="${SERVED_MODEL_PROBE}-Instruct"
fi
REUSABLE_VLLM_PID=""
if MODELS_JSON="$(curl -sf "http://localhost:${VLLM_PORT_START}/v1/models" 2>/dev/null)" \
    && printf '%s' "${MODELS_JSON}" | "${PYTHON_BIN}" -c '
import json, sys
expected = sys.argv[1]
payload = json.load(sys.stdin)
raise SystemExit(0 if any(
    isinstance(item, dict) and item.get("id") == expected
    for item in payload.get("data", [])
) else 1)
' "${SERVED_MODEL_PROBE}"
then
    REUSABLE_VLLM_PID="$(
        ss -ltnp "sport = :${VLLM_PORT_START}" 2>/dev/null \
            | sed -n 's/.*pid=\([0-9]\+\).*/\1/p' | sort -u | head -n 1
    )"
    if [ -n "${REUSABLE_VLLM_PID}" ] \
        && [ -r "/proc/${REUSABLE_VLLM_PID}/environ" ]; then
        REUSABLE_GPU_IDS="$(
            tr '\0' '\n' < "/proc/${REUSABLE_VLLM_PID}/environ" \
                | sed -n 's/^CUDA_VISIBLE_DEVICES=//p' | head -n 1
        )"
        if [ -n "${REUSABLE_GPU_IDS}" ]; then
            export CUDA_VISIBLE_DEVICES="${REUSABLE_GPU_IDS}"
        fi
    fi
    GPU_FREE_ONLY=0
    echo "[gpu_config] exact-model service detected before allocation: model=${SERVED_MODEL_PROBE} port=${VLLM_PORT_START} pid=${REUSABLE_VLLM_PID:-unknown}" >&2
fi
# When no explicit GPU list is passed, auto-select the first GPU_COUNT
# free GPUs instead of defaulting to 0..N-1.  Set CUDA_VISIBLE_DEVICES
# or GPU_FREE_ONLY=0 to override.
if [ -z "${GPU_FREE_ONLY+x}" ]; then
    if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
        GPU_FREE_ONLY=0
    else
        GPU_FREE_ONLY=1
    fi
fi
source scripts/gpu_config.sh
GPU_MEM_GB=${GPU_MEM_GB:-0}

echo "============================================"

_rounded_irrelevance_count() {
    "${PYTHON_BIN}" -c '
import sys
total = int(sys.argv[1])
ratio = float(sys.argv[2])
print(int(total * ratio + 0.5))
' "$1" "${IRRELEVANCE_RATIO}"
}
FINAL_IRRELEVANCE_COUNT="$(_rounded_irrelevance_count "$((COUNT + VAL_COUNT))")"
MERGE_STRATUM_ARGS=(--irrelevance-count "${FINAL_IRRELEVANCE_COUNT}")
if [ -n "${DIFFICULTY}" ]; then
    MERGE_STRATUM_ARGS+=(--difficulty "${DIFFICULTY}")
fi
echo "LiveMCP-GRPO Data Generation"
echo "============================================"
echo "Model:    ${MODEL}"
echo "GPUs:     ${GPU_COUNT}x ${GPU_MODEL} (${GPU_MEM_GB}GB)"
echo "Target:   ${COUNT} train + ${VAL_COUNT} val"
echo "Initial candidate budget: ${INITIAL_TRAIN_CANDIDATES} train"
if [ -n "${BASE_TRAIN}" ]; then
    echo "Base:     ${BASE_TRAIN} + ${BASE_VAL}"
    echo "Target semantics: ${COUNT} net-new train rows relative to base"
fi
echo "Oversample candidates: +${GEN_OVERSAMPLE_PCT}% before quality merge"
echo "Fixed attempt budget: ${LIVEMCP_FIXED_ATTEMPT_BUDGET}"
echo "Domain:   ${DOMAIN}"
echo "Tool-required only: ${TOOL_REQUIRED_ONLY}"
echo "Irrelevance ratio: ${IRRELEVANCE_RATIO}"
echo "Missing-function rate: ${MISSING_FUNCTION_RATE}"
echo "Distractor rate: ${DISTRACTOR_RATE}"
echo "Difficulty: ${DIFFICULTY:-mixed}"
echo "Semantic gate profile: ${LIVEMCP_SEMANTIC_GATE_PROFILE}"
echo "Checkpoint interval: ${CHECKPOINT_INTERVAL} accepted tasks"
echo "Publish active corpus: ${PUBLISH_ACTIVE}"
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
VLLM_OWNED=()
VLLM_SHUTDOWN_ON_EXIT="${VLLM_SHUTDOWN_ON_EXIT:-0}"
if [ "${VLLM_SHUTDOWN_ON_EXIT}" != "0" ] \
    && [ "${VLLM_SHUTDOWN_ON_EXIT}" != "1" ]; then
    echo "ERROR: VLLM_SHUTDOWN_ON_EXIT must be 0 or 1, got ${VLLM_SHUTDOWN_ON_EXIT}" >&2
    exit 1
fi
GEN_SUCCESS=0
_pids_listening_on_port() {
    local port="$1"
    ss -ltnp "sport = :${port}" 2>/dev/null \
        | sed -n 's/.*pid=\([0-9]\+\).*/\1/p' \
        | sort -u
}

_port_is_listening() {
    local port="$1"
    ss -ltnH "sport = :${port}" 2>/dev/null | grep -q .
}

_port_serves_model() {
    local port="$1"
    local expected_model="$2"
    local models_json
    models_json="$(curl -sf "http://localhost:${port}/v1/models" 2>/dev/null)" \
        || return 1
    printf '%s' "${models_json}" | "${PYTHON_BIN}" -c '
import json
import sys

expected = sys.argv[1]
payload = json.load(sys.stdin)
models = payload.get("data", []) if isinstance(payload, dict) else []
raise SystemExit(0 if any(
    isinstance(item, dict) and item.get("id") == expected
    for item in models
) else 1)
' "${expected_model}"
}

_cleanup() {
    local exit_code=$?
    if [ "${VLLM_SHUTDOWN_ON_EXIT}" = "1" ]; then
        echo "[cleanup] stopping launcher-owned vLLM processes..." >&2
        for index in "${!VLLM_PIDS[@]}"; do
            pid="${VLLM_PIDS[$index]}"
            if [ "${VLLM_OWNED[$index]:-0}" = "1" ] \
                && [ -n "${pid:-}" ] \
                && { kill -0 -- "-${pid}" 2>/dev/null \
                    || kill -0 "${pid}" 2>/dev/null; }; then
                kill -TERM -- "-${pid}" 2>/dev/null \
                    || kill -TERM "${pid}" 2>/dev/null \
                    || true
            fi
        done
        sleep 2
        for index in "${!VLLM_PIDS[@]}"; do
            pid="${VLLM_PIDS[$index]}"
            if [ "${VLLM_OWNED[$index]:-0}" = "1" ] \
                && [ -n "${pid:-}" ] \
                && { kill -0 -- "-${pid}" 2>/dev/null \
                    || kill -0 "${pid}" 2>/dev/null; }; then
                kill -KILL -- "-${pid}" 2>/dev/null \
                    || kill -KILL "${pid}" 2>/dev/null \
                    || true
            fi
        done
    elif [ ${#VLLM_PIDS[@]} -gt 0 ]; then
        echo "[cleanup] generation stopped; vLLM service(s) retained:" >&2
        for index in "${!VLLM_PIDS[@]}"; do
            echo "  pid=${VLLM_PIDS[$index]:-unknown} port=${VLLM_PORTS[$index]}" >&2
        done
    fi
    if [ ${#VLLM_LOGS[@]} -gt 0 ]; then
        for log in "${VLLM_LOGS[@]}"; do
            [ -n "${log:-}" ] && [ -f "${log}" ] \
                && echo "[cleanup] vLLM log retained: ${log}" >&2
        done
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
if [ "${LIVEMCP_FIXED_ATTEMPT_BUDGET}" = "1" ]; then
    if [ "${GEN_OVERSAMPLE_PCT}" != "0" ] \
        || [ "${GENERATION_MAX_RECOVERY_ROUNDS}" != "1" ] \
        || [ "${MERGE_TOPUP_ROUNDS}" != "0" ]; then
        echo "ERROR: fixed attempt budget requires GEN_OVERSAMPLE_PCT=0, GENERATION_MAX_RECOVERY_ROUNDS=1, MERGE_TOPUP_ROUNDS=0" >&2
        exit 1
    fi
fi
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

max_persisted_topup_round() {
    local candidate_dir="$1"
    local max_round=0
    local artifact_name
    while IFS= read -r artifact_name; do
        if [[ "${artifact_name}" =~ ^shard_topup_([0-9]+)_ ]]; then
            local candidate_round="${BASH_REMATCH[1]}"
            if [ "${candidate_round}" -gt "${max_round}" ]; then
                max_round="${candidate_round}"
            fi
        fi
    done < <(
        find "${candidate_dir}" -maxdepth 1 -type f \
            -name 'shard_topup_*' -printf '%f\n' | sort
    )
    echo "${max_round}"
}

merge_vllm_with_topups() {
    local deficits_path="${TMPDIR_SHARD}/merge_deficits.json"
    local topup_round=0
    local persisted_topup_round=0
    if [ -n "${GENERATION_RESUME_CANDIDATE_DIR}" ]; then
        persisted_topup_round="$(max_persisted_topup_round "${TMPDIR_SHARD}")"
        echo "Resume top-up artifact numbering after round ${persisted_topup_round}"
    fi
    while ! "${PYTHON_BIN}" -m src.live_mcp.corpus.merge \
        --tmpdir "${TMPDIR_SHARD}" \
        --output-dir "${RUN_DIR}" \
        --count "${COUNT}" \
        --val-count "${VAL_COUNT}" \
        --domain "${DOMAIN}" \
        --deficits-output "${deficits_path}" \
        "${MERGE_DIAGNOSTIC_ARGS[@]}" \
        "${MERGE_SELECTION_ARGS[@]}" \
        "${MERGE_STRATUM_ARGS[@]}" \
        "${MERGE_BASE_ARGS[@]}"; do
        if [ "${topup_round}" -ge "${MERGE_TOPUP_ROUNDS}" ]; then
            echo "ERROR: global merge still has deficits after ${topup_round} top-up round(s)" >&2
            return 1
        fi
        local fatal_integrity
        fatal_integrity=$("${PYTHON_BIN}" -c '
import json, sys
report = json.load(open(sys.argv[1], encoding="utf-8"))
errors = report.get("fatal_integrity_errors", {})
if errors:
    print(json.dumps(errors, sort_keys=True))
' "${deficits_path}")
        if [ -n "${fatal_integrity}" ]; then
            echo "ERROR: global merge found non-recoverable integrity errors; refusing top-up: ${fatal_integrity}" >&2
            return 2
        fi
        topup_round=$((topup_round + 1))
        local topup_artifact_round=$((persisted_topup_round + topup_round))
        local -a topup_deficits=()
        mapfile -t topup_deficits < <("${PYTHON_BIN}" -c '
import json, sys
report = json.load(open(sys.argv[1], encoding="utf-8"))
requests = report.get("topup_candidate_requests", [])
if requests:
    for request in requests:
        domain = str(request["domain"])
        stratum = str(request["stratum"])
        missing = int(request["missing"])
        candidate_count = int(request["candidate_count"])
        token = "__irrelevance__" if stratum == "irrelevance" else stratum
        print(f"{domain}\t{token}\t{missing}\t{candidate_count}")
    raise SystemExit(0)
irrelevance = report.get("irrelevance_deficits_by_domain", {})
has_irrelevance_deficit = any(int(value) > 0 for value in irrelevance.values())
for domain in sorted(irrelevance):
    missing = int(irrelevance[domain])
    if missing > 0:
        suggested = missing + max(2, missing)
        print(f"{domain}\t__irrelevance__\t{missing}\t{suggested}")
strata = report.get("difficulty_deficits_by_domain", {})
if any(strata.values()):
    for domain in sorted(strata):
        for difficulty, missing in strata[domain].items():
            missing = int(missing)
            if missing > 0:
                suggested = missing + max(2, missing)
                print(f"{domain}\t{difficulty}\t{missing}\t{suggested}")
elif not has_irrelevance_deficit:
  for domain, missing in sorted(report.get("deficits", {}).items()):
    if domain != "__all__" and int(missing) > 0:
        suggested = int(
            report.get("suggested_topup_by_domain", {}).get(
                domain, int(missing) + max(2, (int(missing) + 1) // 2)
            )
        )
        print(f"{domain}\t__mixed__\t{int(missing)}\t{suggested}")
' "${deficits_path}")
        if [ "${#topup_deficits[@]}" -eq 0 ]; then
            echo "ERROR: merge failed but reported no domain-specific deficit" >&2
            return 1
        fi

        echo "Top-up attempt ${topup_round} (artifact round ${topup_artifact_round}): ${#topup_deficits[@]} deficit domain(s)"
        local -a topup_pids=()
        local topup_index=0
        local total_topup_slots="${TOTAL_GEN_CLIENTS:-${NUM_INSTANCES}}"
        local slots_per_domain=$(( total_topup_slots / ${#topup_deficits[@]} ))
        if [ "${slots_per_domain}" -lt 1 ]; then slots_per_domain=1; fi
        local entry topup_domain topup_difficulty missing topup_count topup_inst topup_port topup_seed topup_prefix
        local topup_prefix_difficulty
        local chunk_count chunk_index chunk_base chunk_remainder chunk_size
        for entry in "${topup_deficits[@]}"; do
            IFS=$'\t' read -r topup_domain topup_difficulty missing topup_count <<< "${entry}"
            if [ "${topup_difficulty}" = "__mixed__" ]; then
                topup_difficulty=""
            fi
            local topup_is_irrelevance=0
            if [ "${topup_difficulty}" = "__irrelevance__" ]; then
                topup_is_irrelevance=1
                topup_difficulty=""
            fi
            topup_prefix_difficulty="${topup_difficulty:-mixed}"
            if [ "${topup_is_irrelevance}" = "1" ]; then
                topup_prefix_difficulty="irrelevance"
            fi
            chunk_count="${slots_per_domain}"
            if [ "${topup_count}" -lt "${chunk_count}" ]; then chunk_count="${topup_count}"; fi
            chunk_base=$(( topup_count / chunk_count ))
            chunk_remainder=$(( topup_count % chunk_count ))
            echo "  [${topup_domain}/${topup_prefix_difficulty}] missing=${missing}, generating=${topup_count} across ${chunk_count} shard(s)"
            for ((chunk_index=0; chunk_index<chunk_count; chunk_index++)); do
                chunk_size="${chunk_base}"
                if [ "${chunk_index}" -lt "${chunk_remainder}" ]; then
                    chunk_size=$((chunk_size + 1))
                fi
                topup_inst=$((topup_index % NUM_INSTANCES))
                topup_port=$((PORT_START + topup_inst))
                topup_seed=$("${PYTHON_BIN}" -m src.live_mcp.corpus.candidate_identity \
                    --base-seed "${SEED}" \
                    --stride "${GENERATION_CLIENT_SEED_STRIDE}" \
                    --run-id "${RUN_ID}" \
                    --artifact-round "${topup_artifact_round}" \
                    --domain-scope "${topup_domain}" \
                    --stratum "${topup_prefix_difficulty}" \
                    --chunk-index "${chunk_index}")
                topup_prefix="topup_${topup_artifact_round}_${topup_domain}_${topup_prefix_difficulty}_${chunk_index}"
                echo "    shard=${chunk_index}, count=${chunk_size}, port=${topup_port}"
                local -a topup_difficulty_args=()
                local -a topup_irrelevance_args=(--irrelevance-count 0)
                if [ "${topup_is_irrelevance}" = "1" ]; then
                    topup_irrelevance_args=(--irrelevance-count "${chunk_size}")
                elif [ -n "${topup_difficulty}" ]; then
                    topup_difficulty_args=(--difficulty "${topup_difficulty}")
                else
                    topup_difficulty_args=("${GENERATION_DIFFICULTY_ARGS[@]}")
                fi
                "${PYTHON_BIN}" -m src.live_mcp.corpus.shard \
                    --count "${chunk_size}" \
                    --val-count 0 \
                    --seed "${topup_seed}" \
                    --domain "${topup_domain}" \
                    --model "${SERVED_MODEL}" \
                    --teacher-model-id "${MODEL}" \
                    --api-base "http://localhost:${topup_port}/v1" \
                    --suite "${SUITE}" \
                    --shard-mode \
                    --pool-oversample-ratio 0 \
                    --irrelevance-ratio 0 \
                    "${topup_irrelevance_args[@]}" \
                    --missing-function-rate "${MISSING_FUNCTION_RATE}" \
                    --distractor-rate "${DISTRACTOR_RATE}" \
                    --max-recovery-rounds "${GENERATION_MAX_RECOVERY_ROUNDS}" \
                    --checkpoint-path "${TMPDIR_SHARD}/shard_${topup_prefix}_checkpoint.json" \
                    --failure-records-path "${RUN_DIR}/failures/shard_${topup_prefix}.jsonl" \
                    --checkpoint-interval "${CHECKPOINT_INTERVAL}" \
                    --retained-sequences-file "${deficits_path}" \
                    "${topup_difficulty_args[@]}" \
                    "${GENERATION_FILTER_ARGS[@]}" \
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

    GEN_COUNT=$(( (INITIAL_TRAIN_CANDIDATES * (100 + GEN_OVERSAMPLE_PCT) + 99) / 100 ))
    GEN_VAL_COUNT=$(( (VAL_COUNT * (100 + GEN_OVERSAMPLE_PCT) + 99) / 100 ))
    BASE_GPU_TRAIN=$(( GEN_COUNT / GPU_COUNT ))
    REM_GPU_TRAIN=$(( GEN_COUNT % GPU_COUNT ))
    BASE_GPU_VAL=$(( GEN_VAL_COUNT / GPU_COUNT ))
    REM_GPU_VAL=$(( GEN_VAL_COUNT % GPU_COUNT ))
    if [ -n "${GENERATION_RESUME_CANDIDATE_DIR}" ]; then
        echo "ERROR: preserved-candidate resume currently requires the vLLM API generation path" >&2
        exit 1
    fi
    if [ "${LIVEMCP_FIXED_ATTEMPT_BUDGET}" = "1" ]; then
        TMPDIR_SHARD="${RUN_DIR}/candidates"
    else
        TMPDIR_SHARD="${TMPDIR:-/tmp}/livemcp_gen_$$"
    fi
    mkdir -p "${TMPDIR_SHARD}"

    if [ "${DEPENDENCY_CACHE_PREWARM}" = "1" ]; then
        echo ""
        echo "Prewarming dependency graph cache for domain=${DOMAIN}..."
        CUDA_VISIBLE_DEVICES="${GPU_INDEX_ARRAY[0]}" \
            "${PYTHON_BIN}" -m src.live_mcp.corpus.cli build-cache \
                --domain "${DOMAIN}" \
                --model "${MODEL}" \
                --teacher-model-id "${MODEL}" \
                --suite "${SUITE}" \
                --device 0
    fi

    PIDS=()
    IRRELEVANCE_CANDIDATES_ASSIGNED=0
    for ((i=0; i<GPU_COUNT; i++)); do
        GPU_ID="${GPU_INDEX_ARRAY[$i]}"
        SHARD_SEED=$("${PYTHON_BIN}" -m src.live_mcp.corpus.candidate_identity \
            --base-seed "${SEED}" \
            --stride "${GENERATION_CLIENT_SEED_STRIDE}" \
            --run-id "${RUN_ID}" \
            --artifact-round 0 \
            --domain-scope "${DOMAIN}" \
            --stratum initial \
            --chunk-index "${i}")
        SHARD_TRAIN=$(( BASE_GPU_TRAIN + (i < REM_GPU_TRAIN ? 1 : 0) ))
        SHARD_VAL=$(( BASE_GPU_VAL + (i < REM_GPU_VAL ? 1 : 0) ))

        SHARD_TOTAL=$((SHARD_TRAIN + SHARD_VAL))
        IRRELEVANCE_BEFORE="$(_rounded_irrelevance_count "${IRRELEVANCE_CANDIDATES_ASSIGNED}")"
        IRRELEVANCE_CANDIDATES_ASSIGNED=$((IRRELEVANCE_CANDIDATES_ASSIGNED + SHARD_TOTAL))
        IRRELEVANCE_AFTER="$(_rounded_irrelevance_count "${IRRELEVANCE_CANDIDATES_ASSIGNED}")"
        SHARD_IRRELEVANCE=$((IRRELEVANCE_AFTER - IRRELEVANCE_BEFORE))

        echo "  [shard $i] GPU=${GPU_ID}, train=${SHARD_TRAIN}, val=${SHARD_VAL}, irrelevance=${SHARD_IRRELEVANCE}, seed=${SHARD_SEED}"

        if [ $((SHARD_TRAIN + SHARD_VAL)) -eq 0 ]; then
            echo "  [shard $i] skipped (zero quota)"
            continue
        fi

        CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" -m src.live_mcp.corpus.shard \
            --count "${SHARD_TRAIN}" \
            --val-count "${SHARD_VAL}" \
            --seed "${SHARD_SEED}" \
            --domain "${DOMAIN}" \
            --model "${MODEL}" \
            --teacher-model-id "${MODEL}" \
            --suite "${SUITE}" \
            --shard-mode \
            --pool-oversample-ratio 0 \
            --irrelevance-ratio "${IRRELEVANCE_RATIO}" \
            --irrelevance-count "${SHARD_IRRELEVANCE}" \
            --missing-function-rate "${MISSING_FUNCTION_RATE}" \
            --distractor-rate "${DISTRACTOR_RATE}" \
            --max-recovery-rounds "${GENERATION_MAX_RECOVERY_ROUNDS}" \
            --checkpoint-path "${TMPDIR_SHARD}/shard_${i}_checkpoint.json" \
            --failure-records-path "${RUN_DIR}/failures/shard_${i}.jsonl" \
            --checkpoint-interval "${CHECKPOINT_INTERVAL}" \
            "${GENERATION_DIFFICULTY_ARGS[@]}" \
            "${GENERATION_FILTER_ARGS[@]}" \
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

    if [ "$FAILED" -eq "$ACTIVE_SHARDS" ]; then
        echo "ERROR: all ${ACTIVE_SHARDS} active shards failed" >&2
        exit 1
    elif [ "$FAILED" -gt 0 ]; then
        echo "WARNING: ${FAILED}/${ACTIVE_SHARDS} active shards failed; merging successful shards" >&2
    fi

    "${PYTHON_BIN}" -m src.live_mcp.corpus.merge \
        --tmpdir "${TMPDIR_SHARD}" \
        --output-dir "${RUN_DIR}" \
        --count "${COUNT}" \
        --val-count "${VAL_COUNT}" \
        --domain "${DOMAIN}" \
        "${MERGE_DIAGNOSTIC_ARGS[@]}" \
        "${MERGE_SELECTION_ARGS[@]}" \
        "${MERGE_STRATUM_ARGS[@]}" \
        "${MERGE_BASE_ARGS[@]}"
    if [ "${GENERATION_PRESERVE_CANDIDATES}" != "1" ]; then
        rm -f "${TMPDIR_SHARD}"/shard_*_train.parquet "${TMPDIR_SHARD}"/shard_*_val.parquet
    fi

# ═════════════════════════════════════════════════════════════════
# MODE 2: vLLM API — TP across multiple GPUs
# ═════════════════════════════════════════════════════════════════
else
VLLM_VERSION=$("${VLLM_PYTHON_BIN}" -c "import vllm; print(vllm.__version__)" 2>/dev/null || true)
    if [ -z "${VLLM_VERSION}" ]; then
        echo "ERROR: vLLM is not importable from ${VLLM_PYTHON_BIN}" >&2
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
    TP_SIZE=$("${VLLM_PYTHON_BIN}" -c "
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

    PORT_START="${VLLM_PORT_START}"

    # ── vLLM generation defaults ──
    # All parameters can be overridden via environment variables.
    VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-16384}"
    VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.88}"
    VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"
    VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-8}"
    # Keep one orchestrator per vLLM instance so the formal path shares the
    # per-environment sampling epoch, chain sampler, and session manager.
    # Request concurrency comes from worker threads in that process.
    VLLM_CLIENTS_PER_INSTANCE="${VLLM_CLIENTS_PER_INSTANCE:-1}"
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
    GEN_COUNT=$(( (INITIAL_TRAIN_CANDIDATES * (100 + GEN_OVERSAMPLE_PCT) + 99) / 100 ))
    GEN_VAL_COUNT=$(( (VAL_COUNT * (100 + GEN_OVERSAMPLE_PCT) + 99) / 100 ))
    BASE_CLIENT_TRAIN=$(( GEN_COUNT / TOTAL_GEN_CLIENTS ))
    REM_CLIENT_TRAIN=$(( GEN_COUNT % TOTAL_GEN_CLIENTS ))
    BASE_CLIENT_VAL=$(( GEN_VAL_COUNT / TOTAL_GEN_CLIENTS ))
    REM_CLIENT_VAL=$(( GEN_VAL_COUNT % TOTAL_GEN_CLIENTS ))
    if [ -n "${GENERATION_RESUME_CANDIDATE_DIR}" ]; then
        if [ ! -d "${GENERATION_RESUME_CANDIDATE_DIR}" ]; then
            echo "ERROR: resume candidate directory not found: ${GENERATION_RESUME_CANDIDATE_DIR}" >&2
            exit 1
        fi
        TMPDIR_SHARD="$(cd "${GENERATION_RESUME_CANDIDATE_DIR}" && pwd)"
    elif [ "${LIVEMCP_FIXED_ATTEMPT_BUDGET}" = "1" ]; then
        TMPDIR_SHARD="${RUN_DIR}/candidates"
    else
        TMPDIR_SHARD="${TMPDIR:-/tmp}/livemcp_gen_$$"
    fi
    mkdir -p "${TMPDIR_SHARD}"

    # Start vLLM instances
    for ((inst=0; inst<NUM_INSTANCES; inst++)); do
        GPU_START=$(( inst * TP_SIZE ))
        GPU_END=$(( GPU_START + TP_SIZE - 1 ))
        GPU_SLICE=("${GPU_INDEX_ARRAY[@]:$GPU_START:$TP_SIZE}")
        GPU_LIST=$(IFS=','; echo "${GPU_SLICE[*]}")
        PORT=$(( PORT_START + inst ))
        # Derive served model name from directory name (for local vLLM).
        # When using Gemini or other cloud APIs, the model name is passed directly.
        SERVED_MODEL="$(basename "${MODEL}")"
        if [[ "$SERVED_MODEL" == Qwen* && "$SERVED_MODEL" != *Instruct* ]]; then
            SERVED_MODEL="${SERVED_MODEL}-Instruct"
        fi

        EXISTING_PORT_PIDS="$(_pids_listening_on_port "${PORT}")"
        if _port_is_listening "${PORT}"; then
            if ! _port_serves_model "${PORT}" "${SERVED_MODEL}"; then
                echo "ERROR: port ${PORT} is occupied but does not serve exact model ${SERVED_MODEL}; pid(s): ${EXISTING_PORT_PIDS:-unknown}" >&2
                exit 1
            fi
            EXISTING_PORT_PID="$(printf '%s\n' "${EXISTING_PORT_PIDS}" | head -n 1)"
            VLLM_PORTS+=("${PORT}")
            VLLM_PIDS+=("${EXISTING_PORT_PID}")
            VLLM_LOGS+=("")
            VLLM_OWNED+=("0")
            echo "  Reusing vLLM model ${SERVED_MODEL} on port ${PORT} (pid ${EXISTING_PORT_PID})"
            continue
        fi
        VLLM_PORTS+=("${PORT}")
        LOG="logs/${RUN_ID}_vllm_instance${inst}.log"
        VLLM_LOGS+=("${LOG}")
        VLLM_OWNED+=("1")

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
        setsid env \
            -u VLLM_CLIENTS_PER_INSTANCE \
            -u VLLM_NUM_INSTANCES \
            -u VLLM_MAX_NUM_SEQS \
            -u VLLM_MAX_NUM_BATCHED_TOKENS \
            -u VLLM_MAX_MODEL_LEN \
            -u VLLM_GPU_MEMORY_UTILIZATION \
            -u VLLM_ENFORCE_EAGER \
            -u VLLM_PORT_START \
            CUDA_VISIBLE_DEVICES="${GPU_LIST}" \
            "${VLLM_PYTHON_BIN}" "${VLLM_ARGS[@]}" \
            > "${LOG}" 2>&1 < /dev/null &
        VLLM_PIDS+=("$!")
    done

    # Wait for all instances
    echo ""
    echo "Waiting for vLLM instances to be ready..."
    MAX_WAIT=600

    for ((inst=0; inst<NUM_INSTANCES; inst++)); do
        PORT=$(( PORT_START + inst ))
        PID="${VLLM_PIDS[$inst]}"
        LOG="${VLLM_LOGS[$inst]}"
        SERVED_MODEL="$(basename "${MODEL}")"
        if [[ "$SERVED_MODEL" == Qwen* && "$SERVED_MODEL" != *Instruct* ]]; then
            SERVED_MODEL="${SERVED_MODEL}-Instruct"
        fi
        WAITED=0
        while [ $WAITED -lt $MAX_WAIT ]; do
            if [ -n "${PID}" ] && ! kill -0 "${PID}" 2>/dev/null; then
                echo "ERROR: vLLM instance ${inst} exited during startup; see ${LOG}" >&2
                exit 1
            fi
            if _port_serves_model "${PORT}" "${SERVED_MODEL}"; then
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
        "${PYTHON_BIN}" -m src.live_mcp.corpus.cli build-cache \
            --domain "${DOMAIN}" \
            --model "${SERVED_MODEL}" \
            --teacher-model-id "${MODEL}" \
            --api-base "http://localhost:${PORT_START}/v1" \
            --suite "${SUITE}" \
            --prompt-profile "${LIVEMCP_PROMPT_PROFILE}"
    fi

    if [ -n "${GENERATION_RESUME_CANDIDATE_DIR}" ]; then
        echo ""
        echo "Resuming global merge/top-up from preserved candidates: ${TMPDIR_SHARD}"
    else
        # Generate the initial candidate shards.
        echo ""
        echo "Generating data (${NUM_INSTANCES} instance(s) in parallel)..."

        GEN_PIDS=()
        IRRELEVANCE_CANDIDATES_ASSIGNED=0
        for ((inst=0; inst<NUM_INSTANCES; inst++)); do
            PORT=$(( PORT_START + inst ))
            for ((client=0; client<VLLM_CLIENTS_PER_INSTANCE; client++)); do
                CLIENT_ID=$(( inst * VLLM_CLIENTS_PER_INSTANCE + client ))
                SHARD_SEED=$("${PYTHON_BIN}" -m src.live_mcp.corpus.candidate_identity \
                    --base-seed "${SEED}" \
                    --stride "${GENERATION_CLIENT_SEED_STRIDE}" \
                    --run-id "${RUN_ID}" \
                    --artifact-round 0 \
                    --domain-scope "${DOMAIN}" \
                    --stratum initial \
                    --chunk-index "${CLIENT_ID}")
                SHARD_TRAIN=$(( BASE_CLIENT_TRAIN + (CLIENT_ID < REM_CLIENT_TRAIN ? 1 : 0) ))
                SHARD_VAL=$(( BASE_CLIENT_VAL + (CLIENT_ID < REM_CLIENT_VAL ? 1 : 0) ))

                SHARD_TOTAL=$((SHARD_TRAIN + SHARD_VAL))
                IRRELEVANCE_BEFORE="$(_rounded_irrelevance_count "${IRRELEVANCE_CANDIDATES_ASSIGNED}")"
                IRRELEVANCE_CANDIDATES_ASSIGNED=$((IRRELEVANCE_CANDIDATES_ASSIGNED + SHARD_TOTAL))
                IRRELEVANCE_AFTER="$(_rounded_irrelevance_count "${IRRELEVANCE_CANDIDATES_ASSIGNED}")"
                SHARD_IRRELEVANCE=$((IRRELEVANCE_AFTER - IRRELEVANCE_BEFORE))

                echo "  Instance ${inst}/client ${client}: train=${SHARD_TRAIN}, val=${SHARD_VAL}, irrelevance=${SHARD_IRRELEVANCE}, seed=${SHARD_SEED}"

                if [ $((SHARD_TRAIN + SHARD_VAL)) -eq 0 ]; then
                    echo "  Instance ${inst}/client ${client}: skipped (zero quota)"
                    continue
                fi

                "${PYTHON_BIN}" -m src.live_mcp.corpus.shard \
                    --count "${SHARD_TRAIN}" \
                    --val-count "${SHARD_VAL}" \
                    --seed "${SHARD_SEED}" \
                    --domain "${DOMAIN}" \
                    --model "${SERVED_MODEL}" \
                    --teacher-model-id "${MODEL}" \
                    --api-base "http://localhost:${PORT}/v1" \
                    --suite "${SUITE}" \
                    --shard-mode \
                    --pool-oversample-ratio 0 \
                    --irrelevance-ratio "${IRRELEVANCE_RATIO}" \
                    --irrelevance-count "${SHARD_IRRELEVANCE}" \
                    --missing-function-rate "${MISSING_FUNCTION_RATE}" \
                    --distractor-rate "${DISTRACTOR_RATE}" \
                    --max-recovery-rounds "${GENERATION_MAX_RECOVERY_ROUNDS}" \
                    --checkpoint-path "${TMPDIR_SHARD}/shard_${inst}_${client}_checkpoint.json" \
                    --failure-records-path "${RUN_DIR}/failures/shard_${inst}_${client}.jsonl" \
                    --checkpoint-interval "${CHECKPOINT_INTERVAL}" \
                    "${GENERATION_DIFFICULTY_ARGS[@]}" \
                    "${GENERATION_FILTER_ARGS[@]}" \
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

        if [ "$FAILED" -eq "$ACTIVE_GEN_CLIENTS" ]; then
            echo "ERROR: all ${ACTIVE_GEN_CLIENTS} active generation processes failed" >&2
            exit 1
        elif [ "$FAILED" -gt 0 ]; then
            echo "WARNING: ${FAILED}/${ACTIVE_GEN_CLIENTS} active generation processes failed; preserving successful shards and recomputing global deficits" >&2
        fi
    fi

    merge_vllm_with_topups
    if [ -z "${GENERATION_RESUME_CANDIDATE_DIR}" ] \
        && [ "${GENERATION_PRESERVE_CANDIDATES}" != "1" ]; then
        rm -f "${TMPDIR_SHARD}"/shard_*_train.parquet "${TMPDIR_SHARD}"/shard_*_val.parquet
    fi
fi

# ── Print stats ────────────────────────────────────────────────────
echo ""
echo "=== Generation Complete ==="
echo "Run dir:       ${RUN_DIR}/"
if [ "${LIVEMCP_FIXED_ATTEMPT_BUDGET}" = "1" ]; then
    echo "Accepted audit: ${RUN_DIR}/accepted.parquet"
else
    echo "Train parquet: ${RUN_DIR}/train.parquet"
    echo "Val parquet:   ${RUN_DIR}/val.parquet"
fi
if [ "${GENERATION_PRESERVE_CANDIDATES}" = "1" ]; then
    echo "Raw candidates: ${TMPDIR_SHARD}/"
fi

# ── Parquet integrity validation ────────────────────────────────────
echo ""
echo "=== Parquet Integrity Check ==="
AUDIT_ARTIFACTS=("${RUN_DIR}/train.parquet" "${RUN_DIR}/val.parquet")
if [ "${LIVEMCP_FIXED_ATTEMPT_BUDGET}" = "1" ]; then
    AUDIT_ARTIFACTS=("${RUN_DIR}/accepted.parquet")
fi
if ! "${PYTHON_BIN}" -m src.live_mcp.corpus.audit \
    "${AUDIT_ARTIFACTS[@]}"; then
    echo "ERROR: Parquet integrity check failed. See above for details." >&2
    exit 1
fi

# A train-only run is an incremental candidate pool. Publishing it would
# replace the active validation file with an empty parquet. Canonical corpus
# publication is an explicit copy so active data never depends on symlinks.
if [ "${VAL_COUNT}" -gt 0 ] && [ "${PUBLISH_ACTIVE}" = "1" ]; then
    cp "${RUN_DIR}/train.parquet" "${OUTPUT_DIR}/.train.parquet.tmp"
    cp "${RUN_DIR}/val.parquet" "${OUTPUT_DIR}/.val.parquet.tmp"
    mv -f "${OUTPUT_DIR}/.train.parquet.tmp" "${OUTPUT_DIR}/train.parquet"
    mv -f "${OUTPUT_DIR}/.val.parquet.tmp" "${OUTPUT_DIR}/val.parquet"
    echo "Published copies: ${OUTPUT_DIR}/train.parquet"
    echo "                  ${OUTPUT_DIR}/val.parquet"
elif [ "${VAL_COUNT}" -eq 0 ]; then
    echo "Active corpus: unchanged (train-only incremental run)"
else
    echo "Active corpus: unchanged (use public --publish explicitly)"
fi
GEN_SUCCESS=1

echo ""
echo "Done. [$(date '+%Y-%m-%d %H:%M:%S')]"
