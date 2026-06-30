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

MODEL_INFO=$(python3 -c "
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
FITS_SINGLE_GPU=$(python3 -c "
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

        CUDA_VISIBLE_DEVICES="${GPU_ID}" python3 scripts/generate_data.py \
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

    # Merge
    python3 -c "
import pandas as pd, sys
from pathlib import Path
tmpdir = Path('${TMPDIR_SHARD}')
def merge(pattern, outpath, target):
    dfs = [pd.read_parquet(p) for p in sorted(tmpdir.glob(pattern))]
    if not dfs: print(f'WARNING: no {pattern} data!'); return False
    merged = pd.concat(dfs, ignore_index=True)
    # P2-10: shards round up; trim to the contractual count.
    if target is not None and target > 0 and len(merged) > target:
        merged = merged.head(target).reset_index(drop=True)
    merged.to_parquet(outpath, index=False)
    print(f'  {outpath}: {len(merged)} rows (target={target})')
    return True
ok = merge('shard_*_train.parquet', '${OUTPUT_DIR}/train.parquet', ${COUNT})
merge('shard_*_val.parquet', '${OUTPUT_DIR}/val.parquet', ${VAL_COUNT})
if not ok: sys.exit(1)
"
    rm -f "${TMPDIR_SHARD}"/shard_*_train.parquet "${TMPDIR_SHARD}"/shard_*_val.parquet

# ═════════════════════════════════════════════════════════════════
# MODE 2: vLLM API — TP across multiple GPUs
# ═════════════════════════════════════════════════════════════════
else:    # Calculate optimal TP and number of vLLM instances
    # vLLM requires TP to divide num_attention_heads evenly
    TP_SIZE=$(python3 -c "
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
    if [ "$NUM_INSTANCES" -lt 1 ]; then
        echo "ERROR: Need ${TP_SIZE} GPUs for TP=${TP_SIZE}, have ${GPU_COUNT}" >&2
        exit 1
    fi

    PORT_START="${VLLM_PORT_START:-8001}"

    echo ""
    echo "Strategy: vLLM API — TP=${TP_SIZE}, ${NUM_INSTANCES} instance(s)"

    PER_INSTANCE_TRAIN=$(( (COUNT + NUM_INSTANCES - 1) / NUM_INSTANCES ))
    PER_INSTANCE_VAL=$(( (VAL_COUNT + NUM_INSTANCES - 1) / NUM_INSTANCES ))
    TMPDIR_SHARD="${TMPDIR:-/tmp}/livemcp_gen_$$"
    mkdir -p "${TMPDIR_SHARD}"

    # Start vLLM instances
    for ((inst=0; inst<NUM_INSTANCES; inst++)); do
        GPU_START=$(( inst * TP_SIZE ))
        GPU_END=$(( GPU_START + TP_SIZE - 1 ))
        GPU_LIST=$(IFS=','; echo "${GPU_INDEX_ARRAY[@]:$GPU_START:$TP_SIZE}")
        PORT=$(( PORT_START + inst ))
        LOG="${OUTPUT_DIR}/vllm_instance${inst}_$(date +%H%M).log"

        echo "  Starting vLLM instance ${inst} on GPUs ${GPU_LIST}, port ${PORT}"

        CUDA_VISIBLE_DEVICES="${GPU_LIST}" python3 -m vllm.entrypoints.openai.api_server \
            --model "${MODEL}" \
            --served-model-name "$(basename ${MODEL})-Instruct" \
            --tensor-parallel-size "${TP_SIZE}" \
            --gpu-memory-utilization 0.82 \
            --max-model-len 8192 \
            --max-num-seqs 4 \
            --port "${PORT}" \
            > "${LOG}" 2>&1 &
        VLLM_PIDS+=($!)
    done

    # Wait for all instances
    echo ""
    echo "Waiting for vLLM instances to be ready..."
    MAX_WAIT=600

    for ((inst=0; inst<NUM_INSTANCES; inst++)); do
        PORT=$(( PORT_START + inst ))
        WAITED=0
        while [ $WAITED -lt $MAX_WAIT ]; do
            if curl -s "http://localhost:${PORT}/health" > /dev/null 2>&1; then
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

        python3 scripts/generate_data.py \
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

    # Merge
    python3 -c "
import pandas as pd, sys
from pathlib import Path
tmpdir = Path('${TMPDIR_SHARD}')
def merge(pattern, outpath, target):
    dfs = [pd.read_parquet(p) for p in sorted(tmpdir.glob(pattern))]
    if not dfs: print(f'WARNING: no {pattern} data!'); return False
    merged = pd.concat(dfs, ignore_index=True)
    # P2-10: instances round up; trim to the contractual count.
    if target is not None and target > 0 and len(merged) > target:
        merged = merged.head(target).reset_index(drop=True)
    merged.to_parquet(outpath, index=False)
    print(f'  {outpath}: {len(merged)} rows (target={target})')
    return True
ok = merge('shard_*_train.parquet', '${OUTPUT_DIR}/train.parquet', ${COUNT})
merge('shard_*_val.parquet', '${OUTPUT_DIR}/val.parquet', ${VAL_COUNT})
if not ok: sys.exit(1)
"
    rm -f "${TMPDIR_SHARD}"/shard_*_train.parquet "${TMPDIR_SHARD}"/shard_*_val.parquet
fi

# ── Print stats ────────────────────────────────────────────────────
echo ""
echo "=== Generation Complete ==="
python3 -c "
import pandas as pd
for path in ['${OUTPUT_DIR}/train.parquet', '${OUTPUT_DIR}/val.parquet']:
    df = pd.read_parquet(path)
    domains = sorted(set(row['extra_info']['domain'] for _, row in df.iterrows()))
    print(f'{path}: {len(df)} rows, domains={domains}')
    if len(df) > 0:
        ei = df.iloc[0]['extra_info']
        print(f'  sample: domain={ei.get(\"domain\")}, scenario={ei.get(\"scenario_type\")}')
"

echo ""
echo "Done. [$(date '+%Y-%m-%d %H:%M:%S')]"
