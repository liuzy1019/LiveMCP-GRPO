#!/usr/bin/env bash
# Run repeated live Qwen rollout -> MCP execution -> reward smoke tests.

set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

GPUS=""
SEEDS="41,42,43"
STEPS=3
BATCH_SIZE=16
ROLLOUT_N=16
MODEL="models/Qwen/Qwen3-4B-Instruct-2507"
TRAIN_FILE=""
VAL_FILE=""
REWARD_PROFILE=""
EXPERIMENT_PROFILE=""
ARTIFACT_ID=""

usage() {
    cat <<'EOF'
Usage:
  bash scripts/smoke_rollout_reward.sh --gpus IDS [options]

Required:
  --gpus IDS             Exact physical GPU IDs, for example 0,1,2,3 or 0,2,5,7
  --reward-profile NAME  prove_baseline or oval_full
  --experiment-profile NAME  prove_local_v1, oval_local_v1,
                             prove_reward_gray_v1, oval_reward_gray_v1,
                             or custom for explicitly supplied diagnostic data

Options:
  --seeds IDS            Comma-separated seeds (default: 41,42,43)
  --steps N              GRPO steps per seed (default: 3)
  --batch-size N         Prompt groups per step (default: 16)
  --rollout-n N          Rollouts per prompt group (default: 16)
  --model PATH           Policy model (default: paper Qwen3-4B-Instruct-2507)
  --train-file PATH      Training parquet (default: immutable PROVE proxy)
  --val-file PATH        Validation parquet (default: immutable PROVE proxy)
  --artifact-id ID       Explicit owner ID for experiment and Ray temp paths
  --help                  Show this message

Environment overrides for the smoke profile:
  OVAL_PROMPT_LENGTH       default 12384
  OVAL_RESPONSE_LENGTH     default 16384
  OVAL_ROLLOUT_TP          default 2
  OVAL_MAX_NUM_SEQS        default 16
  OVAL_GPU_MEM_UTIL        default 0.55
  OVAL_MINI_BATCH_SIZE     default: selected GPU count
  OVAL_MICRO_BATCH         default 1
  OVAL_LOG_PROB_MICRO_BATCH default 1
  OVAL_GPU_RELEASE_TIMEOUT_S default 180
  OVAL_GPU_RELEASE_THRESHOLD_MIB default 256
  OVAL_PLAIN_FINAL_COMPAT default 0; set 1 only for local compatibility tests
  PARAM_OFFLOAD            default false
  FREE_CACHE_ENGINE        default true
  ENFORCE_EAGER            default false
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus) GPUS="$2"; shift 2 ;;
        --gpus=*) GPUS="${1#*=}"; shift ;;
        --seeds) SEEDS="$2"; shift 2 ;;
        --seeds=*) SEEDS="${1#*=}"; shift ;;
        --steps) STEPS="$2"; shift 2 ;;
        --steps=*) STEPS="${1#*=}"; shift ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --batch-size=*) BATCH_SIZE="${1#*=}"; shift ;;
        --rollout-n) ROLLOUT_N="$2"; shift 2 ;;
        --rollout-n=*) ROLLOUT_N="${1#*=}"; shift ;;
        --model) MODEL="$2"; shift 2 ;;
        --model=*) MODEL="${1#*=}"; shift ;;
        --train-file) TRAIN_FILE="$2"; shift 2 ;;
        --train-file=*) TRAIN_FILE="${1#*=}"; shift ;;
        --val-file) VAL_FILE="$2"; shift 2 ;;
        --val-file=*) VAL_FILE="${1#*=}"; shift ;;
        --reward-profile) REWARD_PROFILE="$2"; shift 2 ;;
        --reward-profile=*) REWARD_PROFILE="${1#*=}"; shift ;;
        --experiment-profile) EXPERIMENT_PROFILE="$2"; shift 2 ;;
        --experiment-profile=*) EXPERIMENT_PROFILE="${1#*=}"; shift ;;
        --artifact-id) ARTIFACT_ID="$2"; shift 2 ;;
        --artifact-id=*) ARTIFACT_ID="${1#*=}"; shift ;;
        --help|-h) usage; exit 0 ;;
        *) echo "ERROR: unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ -z "${GPUS}" ]]; then
    echo "ERROR: --gpus is required; smoke tests never claim GPUs implicitly." >&2
    exit 2
fi
if [[ -z "${REWARD_PROFILE}" ]]; then
    echo "ERROR: --reward-profile is required; smoke tests never infer the objective." >&2
    exit 2
fi
if [[ -z "${EXPERIMENT_PROFILE}" ]]; then
    echo "ERROR: --experiment-profile is required; smoke tests never infer the experiment contract." >&2
    exit 2
fi
if [[ ! "${GPUS}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    echo "ERROR: --gpus must be a comma-separated list of integer IDs." >&2
    exit 2
fi
if [[ ! "${SEEDS}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    echo "ERROR: --seeds must be a comma-separated list of integers." >&2
    exit 2
fi
if [[ -n "${ARTIFACT_ID}" && ! "${ARTIFACT_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$ ]]; then
    echo "ERROR: --artifact-id must be 1-64 safe path characters." >&2
    exit 2
fi
for value in "${STEPS}" "${ROLLOUT_N}"; do
    if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: steps and rollout-n must be positive integers." >&2
        exit 2
    fi
done
if [[ "${REWARD_PROFILE}" != "oval_full" && "${REWARD_PROFILE}" != "prove_baseline" ]]; then
    echo "ERROR: --reward-profile must be oval_full or prove_baseline." >&2
    exit 2
fi
if [[ "${EXPERIMENT_PROFILE}" != "prove_local_v1" \
    && "${EXPERIMENT_PROFILE}" != "oval_local_v1" \
    && "${EXPERIMENT_PROFILE}" != "prove_reward_gray_v1" \
    && "${EXPERIMENT_PROFILE}" != "oval_reward_gray_v1" \
    && "${EXPERIMENT_PROFILE}" != "custom" ]]; then
    echo "ERROR: unsupported --experiment-profile for reward smoke." >&2
    exit 2
fi
if [[ ("${EXPERIMENT_PROFILE}" = "prove_local_v1" \
    || "${EXPERIMENT_PROFILE}" = "prove_reward_gray_v1") \
    && "${REWARD_PROFILE}" != "prove_baseline" ]]; then
    echo "ERROR: PROVE experiment profiles require --reward-profile prove_baseline." >&2
    exit 2
fi
if [[ ("${EXPERIMENT_PROFILE}" = "oval_local_v1" \
    || "${EXPERIMENT_PROFILE}" = "oval_reward_gray_v1") \
    && "${REWARD_PROFILE}" != "oval_full" ]]; then
    echo "ERROR: OVAL experiment profiles require --reward-profile oval_full." >&2
    exit 2
fi
if [[ -z "${TRAIN_FILE}" ]]; then
    if [[ "${EXPERIMENT_PROFILE}" == *"_reward_gray_v1" ]]; then
        TRAIN_FILE="data/runs/20260728_reward_gray_r48dee_v1_train8/train.parquet"
    else
        TRAIN_FILE="data/runs/20260728_gt_v1_prove_composition_proxy_r48dee_train3221_val500/train.parquet"
    fi
fi
if [[ -z "${VAL_FILE}" ]]; then
    VAL_FILE="data/runs/20260728_gt_v1_prove_composition_proxy_r48dee_train3221_val500/val.parquet"
fi
if [[ ! -f "${TRAIN_FILE}" || ! -f "${VAL_FILE}" ]]; then
    echo "ERROR: train/val parquet not found: ${TRAIN_FILE}, ${VAL_FILE}" >&2
    exit 2
fi

IFS=',' read -r -a GPU_ARRAY <<< "${GPUS}"
declare -A SEEN_GPUS=()
for gpu_id in "${GPU_ARRAY[@]}"; do
    if [[ -n "${SEEN_GPUS[${gpu_id}]:-}" ]]; then
        echo "ERROR: duplicate GPU ID: ${gpu_id}" >&2
        exit 2
    fi
    SEEN_GPUS["${gpu_id}"]=1
    if ! nvidia-smi -i "${gpu_id}" --query-gpu=index,name,memory.total \
        --format=csv,noheader >/dev/null 2>&1; then
        echo "ERROR: GPU ${gpu_id} is not available to nvidia-smi." >&2
        exit 2
    fi
done
GPU_COUNT="${#GPU_ARRAY[@]}"

if [[ ! "${BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: --batch-size must be a positive integer." >&2
    exit 2
fi
if (( BATCH_SIZE * ROLLOUT_N < GPU_COUNT )); then
    echo "ERROR: batch-size * rollout-n must be at least the selected GPU count." >&2
    exit 2
fi

LIVEMCP_ENV="${LIVEMCP_ENV:-$(cd "${PROJECT_ROOT}/.." && pwd)/envs/livemcp}"
PYTHON_BIN="${PYTHON_BIN:-${LIVEMCP_ENV}/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "ERROR: training Python is not executable: ${PYTHON_BIN}" >&2
    exit 2
fi

export PYTHON_BIN PYTHONNOUSERSITE=1
export VLLM_NO_USAGE_STATS=1
export OVAL_EXPERIMENT_PROFILE="${EXPERIMENT_PROFILE}"
export OVAL_DIAGNOSTIC_OVERRIDES=1
export OVAL_TRAIN_FILE="${TRAIN_FILE}"
export OVAL_VAL_FILE="${VAL_FILE}"
: "${OVAL_FILTER_OVERLONG_PROMPTS:=false}"
: "${OVAL_PROMPT_LENGTH:=12384}"
: "${OVAL_RESPONSE_LENGTH:=16384}"
: "${OVAL_ROLLOUT_TP:=2}"
: "${OVAL_MAX_NUM_SEQS:=16}"
: "${OVAL_GPU_MEM_UTIL:=0.55}"
: "${OVAL_MINI_BATCH_SIZE:=${GPU_COUNT}}"
: "${OVAL_MICRO_BATCH:=1}"
: "${OVAL_LOG_PROB_MICRO_BATCH:=1}"
: "${OVAL_GPU_RELEASE_TIMEOUT_S:=180}"
: "${OVAL_GPU_RELEASE_THRESHOLD_MIB:=256}"
: "${OVAL_PLAIN_FINAL_COMPAT:=0}"
: "${PARAM_OFFLOAD:=false}"
: "${FREE_CACHE_ENGINE:=true}"
: "${ENFORCE_EAGER:=false}"
: "${OVAL_I_SHAPE:=0}"
: "${OVAL_I_PROCESS:=0}"
: "${OVAL_LAMBDA_SHAPE:=0.5}"
: "${OVAL_LAMBDA_PROCESS:=0.3}"
if [[ ! "${OVAL_ROLLOUT_TP}" =~ ^[1-9][0-9]*$ ]] \
    || (( GPU_COUNT % OVAL_ROLLOUT_TP != 0 )); then
    echo "ERROR: OVAL_ROLLOUT_TP must be a positive divisor of the selected GPU count." >&2
    exit 2
fi
if [[ ! "${OVAL_MINI_BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: OVAL_MINI_BATCH_SIZE must be a positive integer." >&2
    exit 2
fi
if (( OVAL_MINI_BATCH_SIZE > BATCH_SIZE * ROLLOUT_N \
    || (BATCH_SIZE * ROLLOUT_N) % OVAL_MINI_BATCH_SIZE != 0 \
    || OVAL_MINI_BATCH_SIZE % GPU_COUNT != 0 )); then
    echo "ERROR: OVAL_MINI_BATCH_SIZE must divide batch-size * rollout-n" >&2
    echo "       and be divisible by the selected GPU count." >&2
    exit 2
fi
export OVAL_PROMPT_LENGTH OVAL_RESPONSE_LENGTH OVAL_MAX_NUM_SEQS
export OVAL_ROLLOUT_TP
export OVAL_GPU_MEM_UTIL OVAL_MINI_BATCH_SIZE OVAL_MICRO_BATCH
export OVAL_LOG_PROB_MICRO_BATCH PARAM_OFFLOAD FREE_CACHE_ENGINE ENFORCE_EAGER
export OVAL_FILTER_OVERLONG_PROMPTS
export OVAL_PLAIN_FINAL_COMPAT
export OVAL_I_SHAPE OVAL_I_PROCESS OVAL_LAMBDA_SHAPE OVAL_LAMBDA_PROCESS
# A one-step reward smoke does not need optimizer/model checkpoints.  Formal
# profiles retain their normal checkpoint cadence outside this diagnostic.
export OVAL_SAVE_FREQ=-1

IFS=',' read -r -a SEED_ARRAY <<< "${SEEDS}"
STAMP="$(date +%m%d_%H%M%S)_${$}"

wait_for_selected_gpus() {
    "${PYTHON_BIN}" scripts/wait_for_gpu_quiescence.py \
        --gpus "${GPUS}" \
        --memory-threshold-mib "${OVAL_GPU_RELEASE_THRESHOLD_MIB}" \
        --timeout-s "${OVAL_GPU_RELEASE_TIMEOUT_S}" \
        --poll-interval-s 2
}

audit_run() {
    local run_dir="$1"
    local expected_steps="$2"
    local rollout_n="$3"
    "${PYTHON_BIN}" - "${run_dir}" "${expected_steps}" "${REWARD_PROFILE}" "${rollout_n}" <<'PY'
from collections import defaultdict
import json
import math
import os
import re
import statistics
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
expected_steps = int(sys.argv[2])
reward_profile = sys.argv[3]
rollout_n = int(sys.argv[4])
i_shape = int(os.environ["OVAL_I_SHAPE"])
i_process = int(os.environ["OVAL_I_PROCESS"])
lambda_shape = float(os.environ["OVAL_LAMBDA_SHAPE"])
lambda_process = float(os.environ["OVAL_LAMBDA_PROCESS"])
files = sorted((run_dir / "rollouts").glob("*.jsonl"))
if len(files) != expected_steps:
    raise SystemExit(
        f"rollout file count mismatch: expected={expected_steps}, actual={len(files)}"
    )

required = {
    "input", "output", "score", "step", "j", "r_task", "r_validity",
    "r_coverage", "r_efficiency", "c_safety", "f_gamma", "p_process",
    "lambda_safe",
    "r_round_ok", "n_events", "n_model_tool_calls", "n_exec_success",
    "post_kl_return", "trajectory_advantage",
}
rows = []
for path in files:
    for line_no, line in enumerate(path.read_text().splitlines(), start=1):
        row = json.loads(line)
        missing = sorted(required - set(row))
        if missing:
            raise SystemExit(f"{path}:{line_no}: missing fields {missing}")
        for key in (
            "score", "j", "r_task", "r_validity", "r_coverage",
            "r_efficiency", "c_safety", "f_gamma", "p_process",
            "lambda_safe", "r_round_ok",
            "n_events", "n_model_tool_calls", "n_exec_success",
            "post_kl_return", "trajectory_advantage",
        ):
            if not math.isfinite(float(row[key])):
                raise SystemExit(f"{path}:{line_no}: non-finite {key}={row[key]!r}")
        if not math.isclose(
            float(row["score"]), float(row["j"]), rel_tol=1e-7, abs_tol=1e-7
        ):
            raise SystemExit(
                f"{path}:{line_no}: score != j ({row['score']} != {row['j']})"
            )
        if float(row["c_safety"]) not in (0.0, 1.0):
            raise SystemExit(f"{path}:{line_no}: invalid c_safety={row['c_safety']}")
        if float(row["r_round_ok"]) not in (0.0, 1.0):
            raise SystemExit(f"{path}:{line_no}: invalid r_round_ok={row['r_round_ok']}")
        for key in ("r_validity", "r_coverage"):
            if not 0.0 <= float(row[key]) <= 1.0:
                raise SystemExit(f"{path}:{line_no}: {key} out of range: {row[key]}")
        if float(row["r_efficiency"]) > 0.0:
            raise SystemExit(
                f"{path}:{line_no}: positive r_efficiency={row['r_efficiency']}"
            )
        if reward_profile == "prove_baseline":
            expected_j = float(row["r_task"])
        else:
            contract_multiplier = float(row["r_round_ok"])
            expected_j = (
                float(row["r_task"])
                + contract_multiplier * i_shape * lambda_shape
                * float(row["f_gamma"])
                + contract_multiplier * i_process * lambda_process
                * float(row["p_process"])
                - float(row["lambda_safe"]) * float(row["c_safety"])
            )
        if not math.isclose(
            float(row["j"]), expected_j, rel_tol=1e-7, abs_tol=1e-7
        ):
            raise SystemExit(
                f"{path}:{line_no}: reward formula mismatch "
                f"(j={row['j']}, expected={expected_j})"
            )
        rows.append(row)

if not rows:
    raise SystemExit("no rollout rows were saved")

scores = [float(row["score"]) for row in rows]
unsafe = sum(float(row["c_safety"]) for row in rows)
round_ok = sum(float(row["r_round_ok"]) for row in rows)
tool_calls = sum(float(row["n_model_tool_calls"]) for row in rows)
exec_ok = sum(float(row["n_exec_success"]) for row in rows)
unique_scores = len({round(value, 8) for value in scores})
tool_pattern = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
terminal_pattern = re.compile(
    r"<(final_answer|report_error|ask_clarification)>",
    re.DOTALL,
)

def action_signature(output):
    actions = []
    for match in tool_pattern.finditer(str(output)):
        try:
            call = json.loads(match.group(1))
        except json.JSONDecodeError:
            actions.append(("invalid_tool_call", match.group(1).strip()))
            continue
        actions.append(
            (
                "tool_call",
                str(call.get("name", "")),
                json.dumps(call.get("arguments", {}), sort_keys=True, separators=(",", ":")),
            )
        )
    actions.extend(("terminal", match.group(1)) for match in terminal_pattern.finditer(str(output)))
    return tuple(actions)

groups = defaultdict(list)
for row in rows:
    groups[(int(row["step"]), str(row["input"]))].append(row)
bad_group_sizes = {
    key: len(group) for key, group in groups.items() if len(group) != rollout_n
}
if bad_group_sizes:
    raise SystemExit(
        "rollout group size mismatch: "
        + ", ".join(f"{key[0]}:{size}" for key, size in list(bad_group_sizes.items())[:5])
    )
saturated_groups = 0
single_action_groups = 0
spurious_advantage_groups = []
raw_saturated_nonzero_advantage_groups = 0
group_summaries = []
for (step, _), group in groups.items():
    group_scores = [float(row["j"]) for row in group]
    group_post_kl = [float(row["post_kl_return"]) for row in group]
    raw_std = statistics.pstdev(group_scores)
    post_kl_std = statistics.pstdev(group_post_kl)
    unique_actions = len({action_signature(row["output"]) for row in group})
    saturated_groups += raw_std < 1e-6
    single_action_groups += unique_actions == 1
    if raw_std < 1e-6 and max(
        abs(float(row["trajectory_advantage"])) for row in group
    ) > 1e-7:
        raw_saturated_nonzero_advantage_groups += 1
    if len(set(group_post_kl)) == 1:
        max_abs_advantage = max(abs(float(row["trajectory_advantage"])) for row in group)
        if max_abs_advantage > 1e-7:
            spurious_advantage_groups.append((step, max_abs_advantage))
    group_summaries.append(
        f"step={step}:raw_std={raw_std:.3e},post_kl_std={post_kl_std:.3e},"
        f"actions={unique_actions}/{len(group)}"
    )
if spurious_advantage_groups:
    raise SystemExit(
        "exactly saturated post-KL groups produced non-zero trajectory advantage: "
        + ", ".join(
            f"step={step}:max_abs_advantage={value:.3e}"
            for step, value in spurious_advantage_groups[:5]
        )
    )
print(
    "[audit] "
    f"rows={len(rows)} score=[{min(scores):.4f},{max(scores):.4f}] "
    f"unique_scores={unique_scores} unsafe_rate={unsafe / len(rows):.3f} "
    f"round_ok_rate={round_ok / len(rows):.3f} "
    f"exec_success={exec_ok:.0f}/{tool_calls:.0f}"
)
print(
    "[audit] "
    f"groups={len(groups)} raw_saturated={saturated_groups}/{len(groups)} "
    f"raw_saturated_nonzero_advantage="
    f"{raw_saturated_nonzero_advantage_groups}/{len(groups)} "
    f"single_action_sequence={single_action_groups}/{len(groups)}"
)
print("[audit] group_details=" + " | ".join(group_summaries))
if unique_scores == 1:
    print("[audit] WARNING: all saved rollouts have the same score")
PY
}

if [[ "${OVAL_PLAIN_FINAL_COMPAT}" != "0" && "${OVAL_PLAIN_FINAL_COMPAT}" != "1" ]]; then
    echo "ERROR: OVAL_PLAIN_FINAL_COMPAT must be 0 or 1." >&2
    exit 2
fi
echo "[smoke] GPUs=${GPUS} seeds=${SEEDS} steps=${STEPS} batch=${BATCH_SIZE} rollout_n=${ROLLOUT_N} plain_final_compat=${OVAL_PLAIN_FINAL_COMPAT}"
if ! wait_for_selected_gpus; then
    echo "[smoke] FAIL: selected GPUs are not quiescent before launch" >&2
    exit 1
fi

for seed in "${SEED_ARRAY[@]}"; do
    if [[ -n "${ARTIFACT_ID}" ]]; then
        run_name="smoke_rollout_reward_${ARTIFACT_ID}_seed${seed}"
        ray_dir="/tmp/oval_ray_smoke_${ARTIFACT_ID}_${seed}"
    else
        run_name="smoke_rollout_reward_${STAMP}_seed${seed}"
        ray_dir="/tmp/oval_ray_smoke_${USER:-user}_${$}_${seed}"
    fi
    run_dir="${PROJECT_ROOT}/experiments/oval-mcp-grpo/${run_name}"

    echo "[smoke] starting seed=${seed} run=${run_name}"
    if OVAL_SEED="${seed}" OVAL_RAY_TMPDIR="${ray_dir}" \
        bash scripts/train_grpo.sh \
            --gpus "${GPUS}" \
            --model "${MODEL}" \
            --total-steps "${STEPS}" \
            --batch-size "${BATCH_SIZE}" \
            --rollout-n "${ROLLOUT_N}" \
            --reward-profile "${REWARD_PROFILE}" \
            --experiment-profile "${EXPERIMENT_PROFILE}" \
            --diagnostic-overrides \
            --run-name "${run_name}" \
            --save-rollouts \
            --no-wandb \
            --debug; then
        if audit_run "${run_dir}" "${STEPS}" "${ROLLOUT_N}"; then
            echo "[smoke] PASS seed=${seed} artifacts=${run_dir}"
        else
            echo "[smoke] FAIL seed=${seed}: rollout audit failed" >&2
            exit 1
        fi
    else
        echo "[smoke] FAIL seed=${seed}: training command failed" >&2
        wait_for_selected_gpus || true
        exit 1
    fi
    if ! wait_for_selected_gpus; then
        echo "[smoke] FAIL seed=${seed}: selected GPUs were not released" >&2
        exit 1
    fi
done

echo "[smoke] all seeds passed"
