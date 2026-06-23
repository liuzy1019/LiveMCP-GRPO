#!/usr/bin/env bash
# SchemaShift-GRPO 环境配置脚本
# 用法: bash scripts/setup.sh
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

echo "=========================================="
echo " SchemaShift-GRPO 环境配置"
echo "=========================================="

# Python 版本检查
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "[1/4] Python 版本: $PYTHON_VERSION"

# 创建虚拟环境
if [ ! -d "venv" ]; then
    echo "[2/4] 创建虚拟环境..."
    python3 -m venv venv
fi
source venv/bin/activate

# 升级 pip
pip install --upgrade pip -q

# 安装核心依赖
echo "[3/4] 安装项目依赖..."
pip install -e ".[dev]" -q

# 安装本地 verl fork（包含 schemashift 所需的 ray_trainer.py 修改）
echo "       安装本地 verl fork..."
pip install -e verl/ -q

# 安装训练依赖（如果有 GPU）
if python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "       检测到 GPU，安装训练依赖..."
    pip install -e ".[train,bfcl]" -q
else
    echo "       未检测到 GPU，跳过 bfcl-eval 安装"
    echo "       训练服务器上请手动执行: pip install -e '.[train,bfcl]'"
fi

# 创建目录结构
echo "[4/4] 创建目录结构..."
mkdir -p checkpoints
mkdir -p logs
mkdir -p experiments

echo "=========================================="
echo " 环境配置完成"
echo ""
echo " 下一步:"
echo "   python scripts/prepare_grpo_data.py --episode_seeds data/toucan/episode_seeds.jsonl --output data/grpo_train_replay.parquet --val_output data/grpo_val_replay.parquet"
echo "   bash scripts/run_grpo.sh                 # 直接 GRPO"
echo "   MODE=cold bash scripts/run_grpo.sh       # SFT 冷启动 → GRPO"
echo "=========================================="
