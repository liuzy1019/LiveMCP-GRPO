#!/bin/bash
# Shared GPU autodetection — source this in any script that needs GPU info.
#
# Usage:
#   source scripts/gpu_config.sh          # auto-detect all GPUs
#   source scripts/gpu_config.sh 0,1,2,3  # use specific GPUs
#   GPU_COUNT=4 source scripts/gpu_config.sh  # limit to N GPUs (from detected)
#
# Exported variables:
#   GPU_COUNT       — number of GPUs to use
#   GPU_IDS         — comma-separated GPU indices, e.g. "0,1,2,3"
#   GPU_INDEX_ARRAY — bash array of GPU indices
#   GPU_TIER        — "L20" | "A10" | "A100" | "unknown"
#   GPU_MODEL       — full GPU model name from nvidia-smi
#   GPU_MEM_MB      — total memory per GPU in MiB
#   GPU_MEM_GB      — total memory per GPU in GiB (floor)

set -euo pipefail
GPU_DETECT_PYTHON="${GPU_DETECT_PYTHON:-${PYTHON_BIN:-python}}"

# ── Resolve requested GPUs ──────────────────────────────────────────
if [ $# -ge 1 ]; then
    # Explicit GPU list from command line
    IFS=',' read -ra GPU_INDEX_ARRAY <<< "$1"
else
    # Use CUDA_VISIBLE_DEVICES if set, otherwise all GPUs
    if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
        IFS=',' read -ra GPU_INDEX_ARRAY <<< "${CUDA_VISIBLE_DEVICES}"
    else
        # Detect all available GPUs
        if DETECTED_GPU_INDEXES=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null); then
            DETECTED_COUNT=$(printf '%s\n' "${DETECTED_GPU_INDEXES}" | awk 'NF {n++} END {print n+0}')
        elif TORCH_GPU_COUNT=$("${GPU_DETECT_PYTHON}" -c 'import torch; print(torch.cuda.device_count() if torch.cuda.is_available() else 0)' 2>/dev/null); then
            DETECTED_COUNT="${TORCH_GPU_COUNT:-0}"
        else
            DETECTED_COUNT=0
        fi
        if [ "${DETECTED_COUNT}" -eq 0 ]; then
            echo "ERROR: No GPUs detected via nvidia-smi" >&2
            exit 1
        fi
        GPU_INDEX_ARRAY=()
        for ((i=0; i<DETECTED_COUNT; i++)); do
            GPU_INDEX_ARRAY+=("$i")
        done
    fi
fi

# ── Limit GPU count ─────────────────────────────────────────────────
if [ -n "${GPU_COUNT:-}" ] && [ "${GPU_COUNT}" -lt "${#GPU_INDEX_ARRAY[@]}" ]; then
    GPU_INDEX_ARRAY=("${GPU_INDEX_ARRAY[@]:0:${GPU_COUNT}}")
fi

# ── Filter busy GPUs (if GPU_FREE_ONLY=1 or --free-only passed) ──────
if [ "${GPU_FREE_ONLY:-0}" = "1" ] || echo " $* " | grep -q " --free-only "; then
    FREE_GPUS=()
    for gpu_id in "${GPU_INDEX_ARRAY[@]}"; do
        MEM_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${gpu_id}" 2>/dev/null | awk 'NR==1{print $1}') || MEM_USED="999999"
        MEM_TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "${gpu_id}" 2>/dev/null | awk 'NR==1{print $1}') || MEM_TOTAL="1"
        MEM_USED="${MEM_USED:-999999}"
        MEM_TOTAL="${MEM_TOTAL:-1}"
        if [ "${MEM_TOTAL}" != "1" ] && [ -n "${MEM_USED}" ] && [ -n "${MEM_TOTAL}" ] && [ "${MEM_TOTAL}" != "0" ]; then
            USAGE_PCT=$(( MEM_USED * 100 / MEM_TOTAL ))
            if [ "${USAGE_PCT}" -lt 20 ]; then
                FREE_GPUS+=("${gpu_id}")
            else
                echo "[gpu_config] Skipping GPU ${gpu_id}: ${MEM_USED}/${MEM_TOTAL} MiB (${USAGE_PCT}%)" >&2
            fi
        else
            echo "[gpu_config] Skipping GPU ${gpu_id}: cannot query memory" >&2
        fi
    done
    if [ ${#FREE_GPUS[@]} -eq 0 ]; then
        echo "ERROR: No free GPUs found!" >&2
        exit 1
    fi
    GPU_INDEX_ARRAY=("${FREE_GPUS[@]}")
fi

# ── Export results ──────────────────────────────────────────────────
GPU_COUNT=${#GPU_INDEX_ARRAY[@]}
GPU_IDS=$(IFS=','; echo "${GPU_INDEX_ARRAY[*]}")
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"

# ── Detect GPU model & tier ─────────────────────────────────────────
if GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits 2>/dev/null | awk 'NR==1{print}' || true); then
    GPU_MODEL=$(printf '%s\n' "${GPU_INFO}" | awk -F',' 'NF {gsub(/^ +| +$/, "", $1); print $1; exit}')
    GPU_MEM_MB=$(printf '%s\n' "${GPU_INFO}" | awk -F',' 'NF > 1 {gsub(/^ +| +$/, "", $2); if ($2 ~ /^[0-9]+$/) print $2; exit}')
fi
GPU_MODEL="${GPU_MODEL:-}"
GPU_MEM_MB="${GPU_MEM_MB:-}"
if [ -z "${GPU_MODEL}" ] || [ -z "${GPU_MEM_MB}" ]; then
    if TORCH_GPU_INFO=$("${GPU_DETECT_PYTHON}" -c 'import torch; p=torch.cuda.get_device_properties(0); print(torch.cuda.get_device_name(0)); print(p.total_memory // 1024 // 1024)' 2>/dev/null); then
        GPU_MODEL=$(printf '%s\n' "${TORCH_GPU_INFO}" | sed -n '1p')
        GPU_MEM_MB=$(printf '%s\n' "${TORCH_GPU_INFO}" | sed -n '2p')
    fi
fi
GPU_MODEL="${GPU_MODEL:-unknown}"
GPU_MEM_MB="${GPU_MEM_MB:-0}"
GPU_MEM_GB=0
if [ "${GPU_MEM_MB}" != "0" ] && [ -n "${GPU_MEM_MB}" ]; then
    GPU_MEM_GB=$(( GPU_MEM_MB / 1024 ))
fi

if echo "${GPU_MODEL}" | grep -qi "L20"; then
    GPU_TIER="L20"
    DEFAULT_GPU_MEM_UTIL=0.60
    DEFAULT_PARAM_OFFLOAD=false
    DEFAULT_ENFORCE_EAGER=false
    DEFAULT_FREE_CACHE=true
elif echo "${GPU_MODEL}" | grep -qi "A100"; then
    GPU_TIER="A100"
    DEFAULT_GPU_MEM_UTIL=0.80
    DEFAULT_PARAM_OFFLOAD=false
    DEFAULT_ENFORCE_EAGER=false
    DEFAULT_FREE_CACHE=true
elif echo "${GPU_MODEL}" | grep -qi "A10"; then
    GPU_TIER="A10"
    DEFAULT_GPU_MEM_UTIL=0.50
    DEFAULT_PARAM_OFFLOAD=true
    DEFAULT_ENFORCE_EAGER=true
    DEFAULT_FREE_CACHE=false
elif echo "${GPU_MODEL}" | grep -qi "H100\|H800"; then
    GPU_TIER="Hopper"
    DEFAULT_GPU_MEM_UTIL=0.80
    DEFAULT_PARAM_OFFLOAD=false
    DEFAULT_ENFORCE_EAGER=false
    DEFAULT_FREE_CACHE=true
elif echo "${GPU_MODEL}" | grep -qi "T4"; then
    GPU_TIER="T4"
    DEFAULT_GPU_MEM_UTIL=0.40
    DEFAULT_PARAM_OFFLOAD=true
    DEFAULT_ENFORCE_EAGER=true
    DEFAULT_FREE_CACHE=false
else
    GPU_TIER="unknown"
    DEFAULT_GPU_MEM_UTIL=0.40
    DEFAULT_PARAM_OFFLOAD=true
    DEFAULT_ENFORCE_EAGER=true
    DEFAULT_FREE_CACHE=false
fi

# Default per-tier tunables (scripts can override)
GPU_MEM_UTIL="${GPU_MEM_UTIL:-${DEFAULT_GPU_MEM_UTIL}}"
export GPU_MEM_UTIL
PARAM_OFFLOAD="${PARAM_OFFLOAD:-${DEFAULT_PARAM_OFFLOAD}}"
export PARAM_OFFLOAD
ENFORCE_EAGER="${ENFORCE_EAGER:-${DEFAULT_ENFORCE_EAGER}}"
export ENFORCE_EAGER
FREE_CACHE_ENGINE="${FREE_CACHE_ENGINE:-${DEFAULT_FREE_CACHE}}"
export FREE_CACHE_ENGINE

# ── Print summary ───────────────────────────────────────────────────
echo "[gpu_config] ${GPU_COUNT}x ${GPU_MODEL} (${GPU_MEM_GB}GB) → tier=${GPU_TIER}, ids=${GPU_IDS}"
