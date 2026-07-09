"""将外部数据集转换为 livemcp-grpo 训练 parquet 格式。

支持：
  - MadeAgents/xlam-irrelevance-7.5k  (无关查询，answers=[])
  - nvidia/When2Call (NAACL 2025)     (no-tool-call 场景)

用法：
  python scripts/convert_external_datasets.py \\
      --dataset xlam_irrelevance \\
      --n 316 \\
      --output data/external/xlam_irrelevance_316.parquet \\
      --seed 42

  python scripts/convert_external_datasets.py \\
      --dataset when2call \\
      --n 806 \\
      --output data/external/when2call_806.parquet \\
      --seed 42
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import pandas as pd


# ──────────────────────────────────────────────────────────────────────────────
# 公共工具
# ──────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT_NO_TOOL = (
    "You are a helpful AI assistant.\n\n"
    "## Response Format\n"
    "Output exactly ONE action per turn using XML tags:\n\n"
    "- <final_answer>your answer</final_answer>\n"
    "  When you can answer directly without any tool.\n\n"
    "- <report_error>brief reason</report_error>\n"
    "  When the query is irrelevant or cannot be answered.\n\n"
    "## Rules\n"
    "- Do NOT call any tool for this query.\n"
    "- If the query is unrelated to available tools, use <report_error>.\n"
    "- If you can answer directly, use <final_answer>."
)


def _make_uid(prefix: str, idx: int, query: str) -> str:
    h = hashlib.md5(query.encode()).hexdigest()[:8]
    return f"{prefix}_{idx:05d}_{h}"


def _no_tool_row(uid: str, query: str, source: str, *, allow_final_answer: bool = False) -> dict:
    """构造一条 no-tool 训练行，格式与 generate_data.py 的 row 完全一致。

    Args:
        allow_final_answer: True 时 allowed_terminal_actions 包含 final_answer。
            When2Call 的 no-toolcall 场景中部分查询可被直接回答，需要 final_answer；
            xLAM-Irrelevance 明确 answers=[] 不可回答，只能 report_error。
    """
    terminal_actions = ["report_error"]
    if allow_final_answer:
        terminal_actions.append("final_answer")
    prompt = [
        {"role": "system", "content": SYSTEM_PROMPT_NO_TOOL},
        {"role": "user", "content": query},
    ]
    success_criteria: list = []
    oracle_calls_serialized = [{"action": "report_error", "tool_name": "", "arguments": {}}]
    extra_info = {
        "task_id": uid,
        "domain": "external",
        "target_servers": [],
        "required_tools": [],
        "session_seed": 0,
        "initial_state_hash": "",
        "user_query": query,
        "budget": 1,
        "perturbation_level": 0,
        "scenario_type": "no_tool_or_abstention",
        "group_id": uid,
        "uid": uid,
        "has_distractors": False,
        "has_missing_function": False,
        "enum_stripped": False,
        "identity_policy": "domain_defined",
        "target_resource_ids": [],
        "protected_resources": [],
        "protected_fields": [],
        "protected_fields_by_resource": "{}",
        "allowed_terminal_actions": terminal_actions,
        "semantic_fingerprint": "",
        "generation_method": source,
        "oracle_calls": json.dumps(oracle_calls_serialized, ensure_ascii=False),
        "success_criteria": json.dumps(success_criteria, ensure_ascii=False),
        "hidden_tools": [],
        "visible_tool_names": [],
        "conversation_rounds": 1,
        "conversation_queries": json.dumps([query], ensure_ascii=False),
    }
    return {
        "prompt": json.dumps(prompt, ensure_ascii=False),
        "data_source": source,
        "reward_model": {
            "style": "rule",
            "ground_truth": {
                "oracle_calls": json.dumps(oracle_calls_serialized, ensure_ascii=False),
                "success_criteria": json.dumps(success_criteria, ensure_ascii=False),
                "required_tools": [],
            },
        },
        "extra_info": extra_info,
        "uid": uid,
        "group_id": uid,
        "perturbation_level": 0,
        "scenario_type": "no_tool_or_abstention",
    }


# ──────────────────────────────────────────────────────────────────────────────
# xLAM-Irrelevance
# ──────────────────────────────────────────────────────────────────────────────

def convert_xlam_irrelevance(n: int, seed: int) -> list[dict]:
    """从 MadeAgents/xlam-irrelevance-7.5k 随机采样 n 条，转换为训练行。"""
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: pip install datasets", file=sys.stderr)
        sys.exit(1)

    print(f"[xlam_irrelevance] 下载数据集 MadeAgents/xlam-irrelevance-7.5k ...")
    ds = load_dataset("MadeAgents/xlam-irrelevance-7.5k", split="train")
    print(f"[xlam_irrelevance] 总条数: {len(ds)}，采样 {n} 条 (seed={seed})")

    rng = random.Random(seed)
    indices = rng.sample(range(len(ds)), min(n, len(ds)))

    rows = []
    for i, idx in enumerate(indices):
        sample = ds[idx]
        query = sample["query"]
        uid = _make_uid("xlam_irr", i, query)
        rows.append(_no_tool_row(uid, query, "xlam_irrelevance", allow_final_answer=False))

    print(f"[xlam_irrelevance] 转换完成: {len(rows)} 行")
    return rows


# ──────────────────────────────────────────────────────────────────────────────
# When2Call（HuggingFace 输入）
# ──────────────────────────────────────────────────────────────────────────────

def convert_when2call(n: int, seed: int) -> list[dict]:
    """从 nvidia/When2Call train_sft 中随机采样 no-tool-call 条目，转换为训练行。

    train_sft 共 15000 条，全部为 no-toolcall 场景（assistant 回复不含 <TOOLCALL>）。
    论文使用 806 条。
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: pip install datasets", file=sys.stderr)
        sys.exit(1)

    print(f"[when2call] 下载数据集 nvidia/When2Call (train_sft) ...")
    dd = load_dataset("nvidia/When2Call", "train_sft")
    ds = dd["train"]
    print(f"[when2call] 总条数: {len(ds)}，采样 {n} 条 (seed={seed})")

    rng = random.Random(seed)
    indices = rng.sample(range(len(ds)), min(n, len(ds)))

    rows = []
    for i, idx in enumerate(indices):
        sample = ds[idx]
        msgs = sample.get("messages", [])
        query = msgs[0]["content"] if msgs else ""
        uid = _make_uid("w2c", i, query)
        rows.append(_no_tool_row(uid, query, "when2call", allow_final_answer=True))

    print(f"[when2call] 转换完成: {len(rows)} 行")
    return rows


# ──────────────────────────────────────────────────────────────────────────────
# 主入口
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="外部数据集 → livemcp-grpo parquet")
    parser.add_argument("--dataset", required=True,
                        choices=["xlam_irrelevance", "when2call"],
                        help="数据集类型")
    parser.add_argument("--n", type=int, default=316,
                        help="采样条数（xlam_irrelevance 默认 316，when2call 默认 806）")
    parser.add_argument("--output", required=True,
                        help="输出 parquet 路径")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.dataset == "xlam_irrelevance":
        rows = convert_xlam_irrelevance(args.n, args.seed)
    elif args.dataset == "when2call":
        rows = convert_when2call(args.n, args.seed)
    else:
        print(f"ERROR: 未知数据集 {args.dataset}", file=sys.stderr)
        sys.exit(1)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_parquet(out, index=False)
    print(f"[done] 写入 {out}  ({len(df)} 行, {out.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
