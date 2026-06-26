#!/usr/bin/env python3
"""Task generation for GRPO training via two-phase planner.

Phase 1: LLM plans user_query + tool sequence (names only, ~200 tokens).
Phase 2: Execution records oracle trace against live MCP session.

Supports two deployment modes:
  1. Local transformers:  --model models/Qwen3-8B
  2. vLLM server:         --model Qwen3-8B --api-base http://localhost:8000/v1

Data distribution:
  complete=60%    : user query contains all information
  missing=20%     : user query omits one critical parameter
  minimal=20%     : user query is very brief, model must infer

Robustness knobs (applied post-generation on a subset):
  - distractor tools (3-8 irrelevant tools injected)
  - missing function (one required tool hidden)
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from loguru import logger


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate GRPO GRPO training data with LLM teacher"
    )
    p.add_argument("--count", type=int, default=500,
                    help="Number of training tasks to generate")
    p.add_argument("--val-count", type=int, default=50,
                    help="Number of validation tasks to generate")
    p.add_argument("--domain", default="all",
                    help="Domain (all, banking, calendar, etc.)")
    p.add_argument("--model", required=True,
                    help="Model name (vLLM served name) or local path (models/Qwen/Qwen3-8B). "
                         "vLLM mode: must match --served-model-name from vLLM startup. "
                         "Local mode: absolute or relative path to model directory.")
    p.add_argument("--api-base", default=None,
                    help="OpenAI-compatible API base URL (local transformers if unset)")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--output", default="data/train.parquet",
                    help="Training data output path")
    p.add_argument("--val-output", default="data/val.parquet",
                    help="Validation data output path")
    p.add_argument("--suite", default="configs/live_mcp/suite_mvp.yaml",
                    help="Suite config path")
    p.add_argument("--irrelevance-ratio", type=float, default=0.05,
                    help="Fraction of tasks that require report_error (0 to disable)")
    p.add_argument("--distractor-rate", type=float, default=0.40,
                    help="Probability of injecting distractor tools (0 to disable)")
    p.add_argument("--missing-function-rate", type=float, default=0.20,
                    help="Probability of hiding a required tool (0 to disable)")
    p.add_argument("--experiment-tag", default=None,
                    help="Tag for experiment tracking. If set, writes config.json and "
                         "result.json to data/experiments/{YYYY-MM-DD}_{tag}/")
    return p


def generate_data(args: argparse.Namespace):
    """Generate GRPO training data with LLM teacher."""
    from src.live_mcp.api import LiveMCPBranch

    start_time = datetime.now(timezone.utc)

    logger.info(f"Generating GRPO data: {args.count} train + {args.val_count} val tasks")
    logger.info(f"  Domain: {args.domain}")
    logger.info(f"  Model: {args.model}")
    logger.info(f"  Difficulty mix: complete=60%, missing=20%, minimal=20%")

    branch = LiveMCPBranch.from_suite(args.suite)
    difficulty_mix = {"complete": 0.6, "missing": 0.2, "minimal": 0.2}

    try:
        branch.start()
        train_tasks = branch.generate_tasks_llm(
            server_name=args.domain, count=args.count, seed=args.seed,
            difficulty_mix=difficulty_mix, model_path=args.model,
            api_base=args.api_base,
            irrelevance_ratio=args.irrelevance_ratio,
            distractor_rate=args.distractor_rate,
            missing_function_rate=args.missing_function_rate,
        )
        val_tasks = branch.generate_tasks_llm(
            server_name=args.domain, count=args.val_count, seed=args.seed + 10000,
            difficulty_mix=difficulty_mix, model_path=args.model,
            api_base=args.api_base,
            irrelevance_ratio=args.irrelevance_ratio,
            distractor_rate=args.distractor_rate,
            missing_function_rate=args.missing_function_rate,
        )
        all_rows = _tasks_to_rows(train_tasks, args.seed)
        val_rows = _tasks_to_rows(val_tasks, args.seed + 10000)
    finally:
        branch.stop()

    df_train = pd.DataFrame(all_rows) if all_rows else pd.DataFrame()
    df_val = pd.DataFrame(val_rows) if val_rows else pd.DataFrame()
    df_train.to_parquet(Path(args.output), index=False)
    df_val.to_parquet(Path(args.val_output), index=False)

    if args.experiment_tag:
        _save_experiment_record(args, df_train, df_val, start_time)

    _print_stats(df_train, df_val, Path(args.output), Path(args.val_output), args)


def _tasks_to_rows(tasks: list, base_seed: int) -> list[dict]:
    """Convert LiveTask list to verl-compatible data rows."""
    rows = []
    for i, task in enumerate(tasks):
        # Determine visible tools — use task-provided tools, fall back to required
        visible_tools = task.visible_tools if task.visible_tools else []
        if not visible_tools:
            continue  # Skip tasks without tool schemas

        tools_desc_lines = []
        for t in visible_tools:
            name = t.get("name", "")
            desc = t.get("description", "")
            params = t.get("input_schema", {}).get("properties", {})
            required = t.get("input_schema", {}).get("required", [])
            tools_desc_lines.append(f"- {name}: {desc}")
            for pname, pinfo in params.items():
                req = " (required)" if pname in required else ""
                ptype = pinfo.get("type", "any")
                pdesc = pinfo.get("description", "")
                tools_desc_lines.append(
                    f"    - {pname} ({ptype}{req}): {pdesc}"
                )
        tools_block = "\n".join(tools_desc_lines)

        domain = task.target_servers[0] if task.target_servers else "unknown"

        system_prompt = (
            f"You are a helpful assistant with access to the following tools. "
            f"Use them when needed to answer the user's question.\n\n"
            f"Available tools:\n{tools_block}\n\n"
            f"Response format:\n"
            f'- To call a tool: <tool_call>{{"name": "tool_name", "arguments": {{...}}}}</tool_call>\n'
            f"- To give final answer: <final_answer>your answer</final_answer>\n"
            f"- To report error: <report_error>error description</report_error>\n"
            f"- To ask clarification: <ask_clarification>your question</ask_clarification>"
        )

        prompt = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task.user_prompt},
        ]

        task_type = task.task_type
        has_distractors = task.metadata.get("has_distractors", False)
        has_missing_func = task.metadata.get("has_missing_function", False)

        if has_missing_func:
            scenario_type = "missing_function"
            perturbation_level = "hard"
        elif has_distractors:
            scenario_type = "distractor"
            perturbation_level = "medium"
        else:
            scenario_type = task_type or "normal"
            perturbation_level = task.difficulty

        group_id = f"group_{domain}_{len(rows) // 4}"

        extra_info = {
            "task_id": task.task_id,
            "domain": domain,
            "target_servers": task.target_servers,
            "required_tools": task.required_tools,
            "session_seed": task.session_seed,
            "budget": task.max_turns,
            "perturbation_level": perturbation_level,
            "scenario_type": scenario_type,
            "group_id": group_id,
            "uid": task.task_id,
            "has_distractors": has_distractors,
            "has_missing_function": has_missing_func,
            "generation_method": task.metadata.get("generation_method", "task_planner"),
        }

        row = {
            "prompt": json.dumps(prompt, ensure_ascii=False),
            "data_source": "live_mcp_state_machine",
            "reward_model": {
                "style": "rule",
                "ground_truth": {"task_id": task.task_id},
            },
            "extra_info": extra_info,
            "uid": extra_info["uid"],
            "group_id": group_id,
            "perturbation_level": perturbation_level,
            "scenario_type": scenario_type,
        }
        rows.append(row)

    return rows


def _save_experiment_record(args, df_train, df_val, start_time: datetime):
    """Save experiment config and results to data/experiments/{date}_{tag}/."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    exp_dir = PROJECT_ROOT / "data" / "experiments" / f"{today}_{args.experiment_tag}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Git commit hash
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT, text=True,
        ).strip()
    except Exception:
        git_commit = "unknown"

    # GPU info
    try:
        gpu_info = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True,
        ).strip().split("\n")[0] if shutil.which("nvidia-smi") else "unknown"
    except Exception:
        gpu_info = "unknown"

    end_time = datetime.now(timezone.utc)
    duration = (end_time - start_time).total_seconds()

    # config.json
    config = {
        "run_id": f"{today}_{args.experiment_tag}",
        "command": " ".join(sys.argv),
        "model": args.model,
        "api_base": args.api_base or "local",
        "domain": args.domain,
        "count": args.count,
        "val_count": args.val_count,
        "seed": args.seed,
        "distractor_rate": args.distractor_rate,
        "missing_function_rate": args.missing_function_rate,
        "irrelevance_ratio": args.irrelevance_ratio,
        "difficulty_mix": {"complete": 0.6, "missing": 0.2, "minimal": 0.2},
        "git_commit": git_commit,
        "gpu_model": gpu_info,
        "timestamp": start_time.isoformat(),
    }
    (exp_dir / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n"
    )

    # result.json
    if len(df_train) > 0:
        domain_dist = (
            df_train["extra_info"].apply(lambda x: x.get("domain")).value_counts().to_dict()
        )
        scenario_dist = df_train["scenario_type"].value_counts().to_dict()
        difficulty_dist = df_train["perturbation_level"].value_counts().to_dict()
    else:
        domain_dist = {}
        scenario_dist = {}
        difficulty_dist = {}

    result = {
        "run_id": config["run_id"],
        "train_rows": int(len(df_train)),
        "val_rows": int(len(df_val)),
        "yield": round(len(df_train) / args.count, 3) if args.count > 0 else 0.0,
        "duration_seconds": round(duration, 1),
        "domain_distribution": domain_dist,
        "scenario_distribution": scenario_dist,
        "difficulty_distribution": difficulty_dist,
        "timestamp": end_time.isoformat(),
    }
    (exp_dir / "result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    )

    logger.info(f"Experiment record saved: {exp_dir}")


def _print_stats(df_train, df_val, train_path, val_path, args):
    """Print generation statistics."""
    print(f"\n{'='*60}")
    print(f"GRPO Data Generation Complete")
    print(f"{'='*60}")
    print(f"Train: {len(df_train)} rows → {train_path}")
    print(f"Val:   {len(df_val)} rows → {val_path}")

    if len(df_train) == 0:
        print("\nWARNING: No training data generated!")
        return

    # Per-domain stats
    domains = set()
    for _, row in df_train.iterrows():
        domains.add(row["extra_info"]["domain"])
    for domain in sorted(domains):
        domain_rows = df_train[
            df_train["extra_info"].apply(lambda x: x.get("domain") == domain)
        ]
        print(f"\n  {domain}: {len(domain_rows)} rows")
        for stype in ["normal", "distractor", "missing_function", "irrelevant", "task_planner"]:
            count = len(
                domain_rows[domain_rows["scenario_type"] == stype]
            )
            if count > 0:
                print(f"    {stype}: {count}")

    # Difficulty distribution
    difficulty_dist = df_train["perturbation_level"].value_counts().to_dict()
    print(f"\n  Difficulty distribution: {difficulty_dist}")

    # Sample queries (show 3)
    print("\n  Sample queries:")
    for i in range(min(3, len(df_train))):
        ei = df_train.iloc[i]["extra_info"]
        prompt_raw = df_train.iloc[i]["prompt"]
        prompt = json.loads(prompt_raw) if isinstance(prompt_raw, str) else prompt_raw
        user_msg = prompt[1]["content"] if isinstance(prompt, list) and len(prompt) > 1 else str(prompt_raw)[:100]
        print(
            f"    [{i}] domain={ei['domain']} scenario={ei['scenario_type']} "
            f"query=\"{str(user_msg)[:100]}...\""
        )


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    generate_data(args)
