#!/bin/bash
# Unified data generation for LiveMCP-GRPO.
#
# Auto-detects model size from config.json, compares with GPU memory,
# and picks the optimal parallel strategy:
#   - Small model (fits 1 GPU) → local transformers, 1 process per GPU
#   - Large model (needs TP) → vLLM API server(s), 1 process per instance
#
# Usage:
#   bash scripts/generate_data.sh --model models/Qwen/Qwen3-8B --count 500
#   bash scripts/generate_data.sh --model models/Qwen/Qwen3-32B --count 500 --val-count 100
#   bash scripts/generate_data.sh --model models/Qwen/Qwen3-8B --domain calendar --count 200
#   GPU_COUNT=4 bash scripts/generate_data.sh --model models/Qwen/Qwen3-8B --count 200
#
# Env override:
#   OUTPUT_DIR=data  GPU_COUNT=8  VLLM_PORT_START=8001

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${PROJECT_ROOT}"

if [ -z "${PYTHON_BIN:-}" ]; then
    if [ -x "/mnt/data1/zhanyiliu/liuzhanyi/anaconda3/envs/arl/bin/python" ]; then
        PYTHON_BIN="/mnt/data1/zhanyiliu/liuzhanyi/anaconda3/envs/arl/bin/python"
    elif [ -n "${CONDA_PREFIX:-}" ] && [ -x "${CONDA_PREFIX}/bin/python" ]; then
        PYTHON_BIN="${CONDA_PREFIX}/bin/python"
    else
        PYTHON_BIN="python"
    fi
fi
export PYTHON_BIN
export PYTHONNOUSERSITE=1

# ── Parse args ─────────────────────────────────────────────────────
MODEL=""
COUNT=5000
VAL_COUNT=500
DOMAIN="all"
SUITE="configs/live_mcp/suite_mvp.yaml"
SEED=42
OUTPUT_DIR="${OUTPUT_DIR:-data}"

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
        *) echo "ERROR: unknown arg: $1" >&2; exit 1 ;;
    esac
done

if [ -z "$MODEL" ]; then
    echo "ERROR: --model is required" >&2
    exit 1
fi

# ── GPU detection (via shared gpu_config.sh) ────────────────────────
source scripts/gpu_config.sh
GPU_MEM_GB=${GPU_MEM_GB:-0}

echo "============================================"
echo "LiveMCP-GRPO Data Generation"
echo "============================================"
echo "Model:    ${MODEL}"
echo "GPUs:     ${GPU_COUNT}x ${GPU_MODEL} (${GPU_MEM_GB}GB)"
echo "Target:   ${COUNT} train + ${VAL_COUNT} val"
echo "Domain:   ${DOMAIN}"
echo "Output:   ${OUTPUT_DIR}/"
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
    n = c.get('num_hidden_layers', 0)
    d = c.get('hidden_size', 0)
    di = c.get('intermediate_size', 0)
    v = c.get('vocab_size', 0)
    nh = c.get('num_attention_heads', 0)
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
_cleanup() {
    local exit_code=$?
    echo "[cleanup] stopping..." >&2
    for pid in "${VLLM_PIDS[@]}"; do
        if [ -n "${pid:-}" ] && kill -0 "$pid" 2>/dev/null; then
            kill -TERM "$pid" 2>/dev/null || true
        fi
    done
    sleep 2
    for pid in "${VLLM_PIDS[@]}"; do
        if [ -n "${pid:-}" ] && kill -0 "$pid" 2>/dev/null; then
            kill -KILL "$pid" 2>/dev/null || true
        fi
    done
    exit $exit_code
}
trap _cleanup EXIT INT TERM

# ── Environment ────────────────────────────────────────────────────
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_USE_FLASHINFER_SAMPLER=0
export NVCC_APPEND_FLAGS=-allow-unsupported-compiler

mkdir -p "${OUTPUT_DIR}"

# ═══════════════════════════════════════════════════════════════════
# MODE 1: Local transformers — 1 process per GPU
# ═══════════════════════════════════════════════════════════════════
if [ "$FITS_SINGLE_GPU" = "1" ]; then
    echo ""
    echo "Strategy: LOCAL — ${GPU_COUNT} parallel processes, 1 per GPU"

    PER_GPU_TRAIN=$(( (COUNT + GPU_COUNT - 1) / GPU_COUNT ))
    PER_GPU_VAL=$(( (VAL_COUNT + GPU_COUNT - 1) / GPU_COUNT ))
    TMPDIR_SHARD="${TMPDIR:-/tmp}/livemcp_gen_$$"
    mkdir -p "${TMPDIR_SHARD}"

    PIDS=()
    for ((i=0; i<GPU_COUNT; i++)); do
        GPU_ID="${GPU_INDEX_ARRAY[$i]}"
        SHARD_SEED=$((SEED + i * 20000))

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

    # Merge with global semantic dedup and integrity audit.
    "${PYTHON_BIN}" -c "
import pandas as pd, json, sys, hashlib
from pathlib import Path

def _row_fingerprint(row):
    ei = row['extra_info']
    if isinstance(ei, str): ei = json.loads(ei)
    domain = ei.get('domain', '')
    query = ' '.join((ei.get('user_query', '') or '').lower().split())
    oc = ei.get('oracle_calls', [])
    if isinstance(oc, str): oc = json.loads(oc)
    sig = json.dumps({'d': domain, 'q': query, 'c': oc}, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(sig.encode()).hexdigest()

def merge(pattern, outpath, target):
    dfs = [pd.read_parquet(p) for p in sorted(Path('${TMPDIR_SHARD}').glob(pattern))]
    if not dfs: print(f'WARNING: no {pattern} data!'); return False
    merged = pd.concat(dfs, ignore_index=True)
    # P1-5: global semantic dedup across shards.
    before = len(merged)
    seen = set()
    keep = []
    for _, row in merged.iterrows():
        fp = _row_fingerprint(row)
        if fp not in seen:
            seen.add(fp)
            keep.append(row.to_dict())
    merged = pd.DataFrame(keep)
    dropped = before - len(merged)
    if dropped:
        print(f'  dedup: dropped {dropped} cross-shard duplicates, {len(merged)} remaining')
    if target is not None and target > 0 and len(merged) > target:
        merged = merged.head(target).reset_index(drop=True)
    merged.to_parquet(outpath, index=False)
    print(f'  {outpath}: {len(merged)} rows (target={target})')
    if target is not None and target > 0 and len(merged) < target:
        print(f'  FATAL: {outpath} has {len(merged)} rows, below target {target}')
        return False, merged
    return True, merged

ok1, train_df = merge('shard_*_train.parquet', '${OUTPUT_DIR}/train.parquet', ${COUNT})
ok2, val_df = merge('shard_*_val.parquet', '${OUTPUT_DIR}/val.parquet', ${VAL_COUNT})
if not (ok1 and ok2): sys.exit(1)
# P1-5: cross-dataset semantic-fingerprint integrity check.
train_fps = {_row_fingerprint(row) for _, row in train_df.iterrows()}
val_fps = {_row_fingerprint(row) for _, row in val_df.iterrows()}
fp_overlap = train_fps & val_fps
if fp_overlap:
    print(f'  FATAL: {len(fp_overlap)} semantic fingerprint overlaps between train and val!')
    sys.exit(1)
# Also check task_id overlap.
train_ids = {r['extra_info'].get('task_id','') if isinstance(r['extra_info'],dict) else json.loads(r['extra_info']).get('task_id','') for _, r in train_df.iterrows()}
val_ids = {r['extra_info'].get('task_id','') if isinstance(r['extra_info'],dict) else json.loads(r['extra_info']).get('task_id','') for _, r in val_df.iterrows()}
tid_overlap = train_ids & val_ids
if tid_overlap: print(f'WARNING: {len(tid_overlap)} train/val task_id overlaps!')
print(f'  merge ok: {len(train_df)} train + {len(val_df)} val, fp_overlap={len(fp_overlap)}, tid_overlap={len(tid_overlap)}')
"
    rm -f "${TMPDIR_SHARD}"/shard_*_train.parquet "${TMPDIR_SHARD}"/shard_*_val.parquet

# ═════════════════════════════════════════════════════════════════
# MODE 2: vLLM API — TP across multiple GPUs
# ═════════════════════════════════════════════════════════════════
else
    # Calculate optimal TP and number of vLLM instances.
    # vLLM requires TP to divide num_attention_heads evenly
    TP_SIZE=$("${PYTHON_BIN}" -c "
import math
mem_need = ${MODEL_BF16_GB}
mem_gpu = ${GPU_MEM_GB}
num_heads = ${MODEL_NUM_HEADS}
tp = max(1, math.ceil(mem_need / (mem_gpu * 0.82)))
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
    VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.82}"
    VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-12288}"
    VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-4}"

    echo ""
    echo "Strategy: vLLM API — TP=${TP_SIZE}, ${NUM_INSTANCES} instance(s)"
    echo "vLLM: gpu_memory_utilization=${VLLM_GPU_MEMORY_UTILIZATION}, max_model_len=${VLLM_MAX_MODEL_LEN}, max_num_seqs=${VLLM_MAX_NUM_SEQS}"

    PER_INSTANCE_TRAIN=$(( (COUNT + NUM_INSTANCES - 1) / NUM_INSTANCES ))
    PER_INSTANCE_VAL=$(( (VAL_COUNT + NUM_INSTANCES - 1) / NUM_INSTANCES ))
    TMPDIR_SHARD="${TMPDIR:-/tmp}/livemcp_gen_$$"
    mkdir -p "${TMPDIR_SHARD}"

    # Start vLLM instances
    for ((inst=0; inst<NUM_INSTANCES; inst++)); do
        GPU_START=$(( inst * TP_SIZE ))
        GPU_END=$(( GPU_START + TP_SIZE - 1 ))
        GPU_SLICE=("${GPU_INDEX_ARRAY[@]:$GPU_START:$TP_SIZE}")
        GPU_LIST=$(IFS=','; echo "${GPU_SLICE[*]}")
        PORT=$(( PORT_START + inst ))
        LOG="${OUTPUT_DIR}/vllm_instance${inst}_$(date +%H%M).log"

        # Derive served model name from directory name:
        #   Qwen3-32B    → Qwen3-32B-Instruct
        #   Gemma-4-31B-it → Gemma-4-31B-it (keep as-is)
        SERVED_MODEL="$(basename "${MODEL}")"
        if [[ "$SERVED_MODEL" == Qwen* && "$SERVED_MODEL" != *Instruct* ]]; then
            SERVED_MODEL="${SERVED_MODEL}-Instruct"
        fi

        echo "  Starting vLLM instance ${inst} on GPUs ${GPU_LIST}, port ${PORT}"

        CUDA_VISIBLE_DEVICES="${GPU_LIST}" "${PYTHON_BIN}" -m vllm.entrypoints.openai.api_server \
            --model "${MODEL}" \
            --served-model-name "${SERVED_MODEL}" \
            --tensor-parallel-size "${TP_SIZE}" \
            --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
            --max-model-len "${VLLM_MAX_MODEL_LEN}" \
            --max-num-seqs "${VLLM_MAX_NUM_SEQS}" \
            --port "${PORT}" \
            --trust-remote-code \
            > "${LOG}" 2>&1 &
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

    # Generate
    echo ""
    echo "Generating data (${NUM_INSTANCES} instance(s) in parallel)..."

    GEN_PIDS=()
    for ((inst=0; inst<NUM_INSTANCES; inst++)); do
        PORT=$(( PORT_START + inst ))
        SHARD_SEED=$((SEED + inst * 20000))

        echo "  Instance ${inst}: train=${PER_INSTANCE_TRAIN}, val=${PER_INSTANCE_VAL}, seed=${SHARD_SEED}"

        "${PYTHON_BIN}" scripts/generate_data.py \
            --count "${PER_INSTANCE_TRAIN}" \
            --val-count "${PER_INSTANCE_VAL}" \
            --seed "${SHARD_SEED}" \
            --domain "${DOMAIN}" \
            --model "$(basename ${MODEL})-Instruct" \
            --api-base "http://localhost:${PORT}/v1" \
            --suite "${SUITE}" \
            --output "${TMPDIR_SHARD}/shard_${inst}_train.parquet" \
            --val-output "${TMPDIR_SHARD}/shard_${inst}_val.parquet" \
            --log-file "${TMPDIR_SHARD}/shard_${inst}.log" \
            > "${TMPDIR_SHARD}/shard_${inst}.stdout" 2>&1 &
        GEN_PIDS+=($!)
    done

    echo ""
    echo "Waiting for ${NUM_INSTANCES} generation processes..."
    FAILED=0
    for i in "${!GEN_PIDS[@]}"; do
        wait "${GEN_PIDS[$i]}" || { echo "  [Instance $i] FAILED" >&2; FAILED=$((FAILED + 1)); }
    done

    if [ "$FAILED" -gt 0 ]; then
        echo "ERROR: ${FAILED}/${NUM_INSTANCES} generation processes failed" >&2
        exit 1
    fi

    # Merge with global semantic dedup and integrity audit.
"${PYTHON_BIN}" -c "
import pandas as pd, json, sys, hashlib
from pathlib import Path

def _row_fingerprint(row):
    ei = row['extra_info']
    if isinstance(ei, str): ei = json.loads(ei)
    domain = ei.get('domain', '')
    query = ' '.join((ei.get('user_query', '') or '').lower().split())
    oc = ei.get('oracle_calls', [])
    if isinstance(oc, str): oc = json.loads(oc)
    sig = json.dumps({'d': domain, 'q': query, 'c': oc}, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(sig.encode()).hexdigest()

def merge(pattern, outpath, target):
    dfs = [pd.read_parquet(p) for p in sorted(Path('${TMPDIR_SHARD}').glob(pattern))]
    if not dfs: print(f'WARNING: no {pattern} data!'); return False
    merged = pd.concat(dfs, ignore_index=True)
    before = len(merged)
    seen = set()
    keep = []
    for _, row in merged.iterrows():
        fp = _row_fingerprint(row)
        if fp not in seen:
            seen.add(fp)
            keep.append(row.to_dict())
    merged = pd.DataFrame(keep)
    dropped = before - len(merged)
    if dropped:
        print(f'  dedup: dropped {dropped} cross-shard duplicates, {len(merged)} remaining')
    if target is not None and target > 0 and len(merged) > target:
        merged = merged.head(target).reset_index(drop=True)
    merged.to_parquet(outpath, index=False)
    print(f'  {outpath}: {len(merged)} rows (target={target})')
    if target is not None and target > 0 and len(merged) < target:
        print(f'  FATAL: {outpath} has {len(merged)} rows, below target {target}')
        return False, merged
    return True, merged

ok1, train_df = merge('shard_*_train.parquet', '${OUTPUT_DIR}/train.parquet', ${COUNT})
ok2, val_df = merge('shard_*_val.parquet', '${OUTPUT_DIR}/val.parquet', ${VAL_COUNT})
if not (ok1 and ok2): sys.exit(1)
train_fps = {_row_fingerprint(row) for _, row in train_df.iterrows()}
val_fps = {_row_fingerprint(row) for _, row in val_df.iterrows()}
fp_overlap = train_fps & val_fps
if fp_overlap:
    print(f'  FATAL: {len(fp_overlap)} semantic fingerprint overlaps between train and val!')
    sys.exit(1)
train_ids = {r['extra_info'].get('task_id','') if isinstance(r['extra_info'],dict) else json.loads(r['extra_info']).get('task_id','') for _, r in train_df.iterrows()}
val_ids = {r['extra_info'].get('task_id','') if isinstance(r['extra_info'],dict) else json.loads(r['extra_info']).get('task_id','') for _, r in val_df.iterrows()}
tid_overlap = train_ids & val_ids
if tid_overlap: print(f'WARNING: {len(tid_overlap)} train/val task_id overlaps!')
print(f'  merge ok: {len(train_df)} train + {len(val_df)} val, fp_overlap={len(fp_overlap)}, tid_overlap={len(tid_overlap)}')
"
    rm -f "${TMPDIR_SHARD}"/shard_*_train.parquet "${TMPDIR_SHARD}"/shard_*_val.parquet
fi

# ── Print stats ────────────────────────────────────────────────────
echo ""
echo "=== Generation Complete ==="
echo "Train parquet: ${OUTPUT_DIR}/train.parquet"
echo "Val parquet:   ${OUTPUT_DIR}/val.parquet"

echo ""
echo "Done. [$(date '+%Y-%m-%d %H:%M:%S')]"
