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
echo "[1/5] Python 版本: $PYTHON_VERSION"

# 创建虚拟环境
if [ ! -d "venv" ]; then
    echo "[2/5] 创建虚拟环境..."
    python3 -m venv venv
fi
source venv/bin/activate

# 升级 pip
pip install --upgrade pip -q

# 安装核心依赖
echo "[3/5] 安装项目依赖..."
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
echo "[4/5] 创建目录结构..."
mkdir -p data/possible_answer
mkdir -p data/verl
mkdir -p checkpoints
mkdir -p logs
mkdir -p experiments

# 下载 BFCL 数据
echo "[5/5] 下载 BFCL V3 数据..."
if python3 scripts/download_data.py 2>/dev/null; then
    echo "       数据下载完成"
    echo "       生成训练 parquet: python scripts/build_parquet.py"
else
    echo "       数据下载失败或已存在，跳过"
fi

echo "=========================================="
echo " 环境配置完成"
echo ""
echo " 下一步:"
echo "   python scripts/build_parquet.py          # 生成训练数据"
echo "   bash scripts/train/sft/run_exp2_sft.sh   # E2: SFT 基线"
echo "   bash scripts/train/grpo/run_vanilla_grpo.sh  # E3: GRPO 基线"
echo "   bash scripts/train/grpo/run_schemashift.sh   # E4: SchemaShift-GRPO"
echo "   bash scripts/train/grpo/run_aug_only.sh      # E5: Aug Only"
echo "   python scripts/eval/eval_zero_shot.py        # E1: Zero-shot 评估"
echo "=========================================="
