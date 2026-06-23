#!/bin/bash
# ============================================================
# SchemaShift-GRPO 训练路由器
# ============================================================
#
# MODE 控制 E4 路线的模型起点：
#
#   MODE=direct（默认）：直接 GRPO，模型起点 = models/Qwen3-4B
#   MODE=cold：SFT 冷启动 → GRPO，模型起点 = outputs/sft_cold_start_4b/final
#
# 示例：
#   bash scripts/run_grpo.sh                           # 直接 GRPO
#   MODE=cold bash scripts/run_grpo.sh                 # SFT 冷启动 → GRPO
#
# ============================================================
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODE="${MODE:-direct}"

echo "[run_grpo.sh] → E4 SchemaShift (mode=${MODE}): scripts/train/grpo/run_schemashift.sh"
exec env MODE="${MODE}" bash "${PROJECT_ROOT}/scripts/train/grpo/run_schemashift.sh" "$@"
