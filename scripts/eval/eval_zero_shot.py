#!/usr/bin/env python3
"""E1: Zero-shot 评估 — 不做任何训练，直接用基座模型评估 BFCL pass@1。

用法:
    python scripts/eval/eval_zero_shot.py \
        --model Qwen/Qwen2.5-1.5B-Instruct \
        --data data/verl/exp4_schemashift/val.parquet \
        --output experiments/e1_zero_shot
"""
import argparse
import json
import sys
from pathlib import Path

_PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_DIR))

from loguru import logger

from src.eval.bfcl_eval import evaluate_checkpoint


def main():
    parser = argparse.ArgumentParser(description="E1 Zero-shot BFCL 评估")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--data", type=str, default="data/verl/exp4_schemashift/val.parquet")
    parser.add_argument("--output", type=str, default="experiments/e1_zero_shot")
    parser.add_argument("--num-samples", type=int, default=None)
    args = parser.parse_args()

    if not Path(args.data).exists():
        logger.error(f"数据文件不存在: {args.data}，请先运行 scripts/build_parquet.py")
        sys.exit(1)

    logger.info(f"E1 Zero-shot 评估: model={args.model}")
    result = evaluate_checkpoint(
        model_path=args.model,
        data_path=args.data,
        output_dir=args.output,
        num_samples=args.num_samples,
    )

    print(json.dumps(result, indent=2, ensure_ascii=False))
    logger.info(f"E1 评估完成: macro_overall_pass@1={result['macro_overall_pass@1']:.4f}, "
                f"raw_pass@1={result['raw_pass@1']:.4f}, "
                f"robustness_gap={result.get('robustness_gap', 0):.4f}")


if __name__ == "__main__":
    main()
