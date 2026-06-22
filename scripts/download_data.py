#!/usr/bin/env python3
"""BFCL V3 数据下载脚本。

从 HuggingFace 下载 BFCL V3 多轮数据集、ground truth、单轮 SFT 数据。
"""

import json
import sys
from pathlib import Path
from loguru import logger

try:
    from huggingface_hub import hf_hub_download, list_repo_files
except ImportError:
    print("请先安装 huggingface_hub: pip install huggingface_hub")
    sys.exit(1)

REPO_ID = "gorilla-llm/Berkeley-Function-Calling-Leaderboard"
REPO_TYPE = "dataset"

# ── 需要下载的文件列表 ──
MULTI_TURN_FILES = [
    "BFCL_v3_multi_turn_base.json",
    "BFCL_v3_multi_turn_composite.json",
    "BFCL_v3_multi_turn_long_context.json",
    "BFCL_v3_multi_turn_miss_func.json",
    "BFCL_v3_multi_turn_miss_param.json",
]

SINGLE_TURN_FILES = [
    "BFCL_v3_simple.json",
    "BFCL_v3_multiple.json",
    "BFCL_v3_parallel.json",
    "BFCL_v3_parallel_multiple.json",
]

GROUND_TRUTH_FILES = [
    f"possible_answer/{f}" for f in [
        "BFCL_v3_multi_turn_base.json",
        "BFCL_v3_multi_turn_composite.json",
        "BFCL_v3_multi_turn_long_context.json",
        "BFCL_v3_multi_turn_miss_func.json",
        "BFCL_v3_multi_turn_miss_param.json",
        "BFCL_v3_simple.json",
        "BFCL_v3_multiple.json",
        "BFCL_v3_parallel.json",
        "BFCL_v3_parallel_multiple.json",
    ]
]

FUNC_DOC_FILES = [
    f"multi_turn_func_doc/{f}" for f in [
        "gorilla_file_system.json",
        "math_api.json",
        "message_api.json",
        "posting_api.json",
        "ticket_api.json",
        "trading_bot.json",
        "travel_booking.json",
        "vehicle_control.json",
    ]
]

ALL_FILES = MULTI_TURN_FILES + SINGLE_TURN_FILES + GROUND_TRUTH_FILES + FUNC_DOC_FILES


def download_file(repo_path: str, local_dir: Path) -> Path:
    """下载单个文件。"""
    local_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=repo_path,
        repo_type=REPO_TYPE,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
    )
    return Path(local_path)


def count_lines(file_path: Path) -> int:
    """计算 JSONL 文件的行数。"""
    count = 0
    with open(file_path) as f:
        for _ in f:
            count += 1
    return count


def main():
    # 创建数据目录
    base_dir = Path(__file__).resolve().parent.parent / "data"
    base_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"数据目录: {base_dir}")
    logger.info(f"将从 {REPO_ID} 下载 {len(ALL_FILES)} 个文件")

    downloaded = 0
    skipped = 0
    for repo_path in ALL_FILES:
        local_path = base_dir / repo_path
        if local_path.exists() and local_path.stat().st_size > 0:
            # 检查本地文件是否完整（和远程比较行数或大小）
            skipped += 1
            continue

        local_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            result = download_file(repo_path, base_dir)
            n_lines = count_lines(result)
            logger.info(f"  [{result.parent.name}] {result.name} ({n_lines} 条)")
            downloaded += 1
        except Exception as e:
            logger.error(f"  下载失败: {repo_path} - {e}")

    logger.info(
        f"下载完成: 新增 {downloaded} 个, 已存在 {skipped} 个"
    )

    # 打印统计数据
    logger.info("=" * 50)
    logger.info("数据统计:")
    total_lines = 0
    for repo_path in MULTI_TURN_FILES:
        local_path = base_dir / repo_path
        if local_path.exists():
            n = count_lines(local_path)
            total_lines += n
            logger.info(f"  {local_path.name}: {n} 条")
    logger.info(f"  多轮总任务数: {total_lines}")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
