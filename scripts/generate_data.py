#!/usr/bin/env python3
"""Task generation for GRPO training via PROVE-style state-machine teacher.

The teacher uses LLM-in-the-loop at every turn: the LLM sees the full domain
context (tools, live state, execution history) and decides the next action
(tool_call with real arguments, or terminal). The resulting oracle trace is
built by actual execution against live MCP servers.

Deployment modes:
  1. Local transformers:  --model models/Qwen3-8B
  2. vLLM server:         --model Qwen3-8B --api-base http://localhost:8000/v1

PROVE-aligned defaults:
  - Difficulty mix: complete=60%, missing=20%, minimal=20%
  - Irrelevance ratio: 5%
  - Distractor rate: 40% (injects 3-8 irrelevant tools)
  - Missing function rate: 20% (hides one required tool)
  - Enum stripping: 30% per domain
  - Jaccard dedup threshold: 0.70
  - Conversation rounds: 2-3 (turn-decay schedule, PROVE min_turns=2 max_turns=3)
  - Personas: 10 role templates, reference dates: 10 anchors
  - Recovery: explicit retry_same / retry_alt / give_up states
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
                    help="Domain (all, banking, calendar, etc.). "
                         "Comma-separated list supported, e.g. calendar,shopping,banking")
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
    p.add_argument("--device", type=int, default=None,
                    help="GPU device ID for local inference (default: auto). "
                         "Use with CUDA_VISIBLE_DEVICES for multi-GPU data-parallel via "
                         "scripts/generate_data_parallel.sh")
    p.add_argument("--experiment-tag", default=None,
                    help="Tag for experiment tracking. If set, writes config.json and "
                         "result.json to data/experiments/{YYYY-MM-DD}_{tag}/")
    p.add_argument("--log-file", default=None,
                    help="Write all logs to this file (auto-flushed, avoids pipe buffering)")
    return p


def generate_data(args: argparse.Namespace):
    """Generate GRPO training data with LLM teacher."""
    from src.live_mcp.api import LiveMCPBranch

    # ── Validate parameters ──
    if args.count < 1:
        raise ValueError(f"--count must be >= 1, got {args.count}")
    if args.val_count < 0:
        raise ValueError(f"--val-count must be >= 0, got {args.val_count}")
    for name, val in [("irrelevance_ratio", args.irrelevance_ratio),
                       ("distractor_rate", args.distractor_rate),
                       ("missing_function_rate", args.missing_function_rate)]:
        if not (0.0 <= val <= 1.0):
            raise ValueError(f"--{name} must be in [0.0, 1.0], got {val}")
    if Path(args.output).resolve() == Path(args.val_output).resolve():
        raise ValueError(
            f"--output and --val-output point to the same file: {args.output}"
        )

    # 如果指定了 --log-file，添加文件 sink 确保实时可见
    if args.log_file:
        log_path = Path(args.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            str(log_path),
            level="DEBUG",
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
            enqueue=False,
            catch=True,
        )

    start_time = datetime.now(timezone.utc)

    print(f"[generate_data] Target: {args.count} train + {args.val_count} val tasks, domain={args.domain}, model={args.model}")
    logger.info(f"Generating GRPO data: {args.count} train + {args.val_count} val tasks")
    logger.info(f"  Domain: {args.domain}")
    logger.info(f"  Model: {args.model}")
    logger.info(f"  Difficulty mix: complete=60%, missing=20%, minimal=20%")

    branch = LiveMCPBranch.from_suite(args.suite)
    difficulty_mix = {"complete": 0.6, "missing": 0.2, "minimal": 0.2}

    try:
        branch.start()
        total_count = args.count + args.val_count
        # Oversample 50% (was 10%) plus a floor — LLM generation, replay,
        # provenance checks, dedup, and training-contract filtering all
        # discard tasks.  Recovery loop below regenerates with different seed
        # offsets when the first pool still falls short.
        pool_target = total_count + max(10, total_count // 2)
        print(f"[generate_data] Generating pool of ~{pool_target} tasks...", flush=True)

        all_tasks = []
        MAX_RECOVERY_ROUNDS = 3
        for recovery_round in range(MAX_RECOVERY_ROUNDS):
            round_seed = args.seed + recovery_round * 100000
            round_target = pool_target if recovery_round == 0 else pool_target // 2
            print(
                f"[generate_data] Round {recovery_round + 1}/{MAX_RECOVERY_ROUNDS}: "
                f"generating up to {round_target} tasks (seed={round_seed})...",
                flush=True,
            )
            round_tasks = branch.generate_tasks_llm(
                server_name=args.domain, count=round_target, seed=round_seed,
                difficulty_mix=difficulty_mix, model_path=args.model,
                api_base=args.api_base,
                device=args.device,
                irrelevance_ratio=args.irrelevance_ratio,
                distractor_rate=args.distractor_rate,
                missing_function_rate=args.missing_function_rate,
            )
            all_tasks.extend(round_tasks)
            logger.info(
                f"Round {recovery_round + 1}: got {len(round_tasks)} tasks "
                f"(cumulative {len(all_tasks)})"
            )

            # Try to split early — if we already have enough, break out.
            eligible = _filter_training_eligible_tasks(all_tasks)
            if len(eligible) >= total_count:
                try:
                    train_tasks, val_tasks = _stratified_task_split(
                        eligible, train_count=args.count,
                        val_count=args.val_count, seed=args.seed,
                    )
                    print(
                        f"[generate_data] Early split success: "
                        f"{len(train_tasks)} train + {len(val_tasks)} val "
                        f"from {len(eligible)} eligible tasks "
                        f"(pool {len(all_tasks)})",
                        flush=True,
                    )
                    all_tasks = eligible  # use filtered tasks for downstream
                    break
                except RuntimeError:
                    logger.info(
                        f"Round {recovery_round + 1}: {len(eligible)} eligible "
                        f"tasks not enough for stratified split, generating more..."
                    )
        else:
            # Exhausted recovery rounds — fall through with whatever we have.
            eligible = _filter_training_eligible_tasks(all_tasks)
            train_tasks, val_tasks = _stratified_task_split(
                eligible, train_count=args.count,
                val_count=args.val_count, seed=args.seed,
            )
            print(
                f"[generate_data] Final split: {len(train_tasks)} train + "
                f"{len(val_tasks)} val from {len(eligible)} eligible "
                f"(pool {len(all_tasks)} after {MAX_RECOVERY_ROUNDS} rounds)",
                flush=True,
            )

        all_tasks = eligible  # ensure downstream uses filtered tasks
        all_rows = _tasks_to_rows(train_tasks, args.seed)
        val_rows = _tasks_to_rows(val_tasks, args.seed + 10000)
    finally:
        branch.stop()

    df_train = pd.DataFrame(all_rows) if all_rows else pd.DataFrame()
    df_val = pd.DataFrame(val_rows) if val_rows else pd.DataFrame()

    _assert_split_integrity(df_train, df_val, args)

    # Ensure output parent directories exist
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.val_output).parent.mkdir(parents=True, exist_ok=True)

    df_train.to_parquet(Path(args.output), index=False)
    df_val.to_parquet(Path(args.val_output), index=False)

    if args.experiment_tag:
        _save_experiment_record(args, df_train, df_val, start_time)

    _print_stats(df_train, df_val, Path(args.output), Path(args.val_output), args)


def _task_scenario(task) -> str:
    explicit = task.metadata.get("scenario_type") if task.metadata else None
    if explicit:
        return str(explicit)
    if task.task_type == "missing_function":
        return "no_tool_or_abstention"
    if task.task_type == "irrelevant":
        return "no_tool_or_abstention"
    return "normal_safe_success"


def _identity_policy(domain: str) -> str:
    return {
        "calendar": "preserve",
        "banking": "preserve",
        "filesystem": "domain_defined",
        "payments": "preserve",
        "crm": "preserve",
        "issue_tracker": "preserve",
        "email": "append_only",
        "team_chat": "append_only",
        "shopping": "create_new",
        "food_delivery": "create_new",
    }.get(domain, "domain_defined")


def _task_fingerprint(task) -> str:
    """Semantic identity used before splitting, never as a deletion rule."""
    import hashlib

    domain = task.target_servers[0] if task.target_servers else ""
    calls = []
    for call in task.oracle_program.calls if task.oracle_program else []:
        if getattr(call, "action", "tool_call") != "tool_call":
            continue
        calls.append({
            "tool_name": call.tool_name,
            "arguments": call.arguments or {},
        })
    payload = {
        "domain": domain,
        "scenario": _task_scenario(task),
        "query": " ".join((task.user_prompt or "").lower().split()),
        "calls": calls,
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


def _serialize_training_oracle(task) -> list[dict]:
    """Return tool calls plus exactly one terminal action for training."""
    raw_calls = []
    if task.oracle_program and task.oracle_program.calls:
        for oc in task.oracle_program.calls:
            raw_calls.append({
                "tool_name": oc.tool_name,
                "arguments": dict(oc.arguments) if oc.arguments else {},
                "action": getattr(oc, "action", "tool_call"),
            })

    terminals = [
        call for call in raw_calls
        if call.get("action") in ("final_answer", "ask_clarification", "report_error")
    ]
    if not terminals:
        raise ValueError(f"Task {task.task_id} has no explicit terminal oracle action")

    tool_calls = [
        call for call in raw_calls
        if call.get("action", "tool_call") == "tool_call"
    ]
    return tool_calls + [terminals[-1]]


def _has_stale_explicit_year(value) -> bool:
    import re

    raw = json.dumps(value, ensure_ascii=False, default=str)
    return bool(re.search(r"\b202[0-5][-/]", raw))


def _task_success_criteria(task) -> list:
    if task.oracle_program and task.oracle_program.success_criteria:
        return list(task.oracle_program.success_criteria)
    if hasattr(task, "success_criteria") and task.success_criteria:
        return list(task.success_criteria)
    return []


def _validate_task_training_contract(task) -> None:
    oracle_calls_serialized = _serialize_training_oracle(task)
    if _has_stale_explicit_year(oracle_calls_serialized):
        raise ValueError(
            f"Task {task.task_id} oracle contains an explicit pre-2026 date"
        )

    terminal_actions = [
        call["action"] for call in oracle_calls_serialized
        if call.get("action") in ("final_answer", "ask_clarification", "report_error")
    ]
    if len(terminal_actions) != 1:
        raise ValueError(
            f"Task {task.task_id} has {len(terminal_actions)} terminal oracle actions"
        )

    real_required_tools = [
        call["tool_name"] for call in oracle_calls_serialized
        if call.get("action", "tool_call") == "tool_call"
    ]
    scenario_type = _task_scenario(task)
    is_no_tool = scenario_type in (
        "no_tool_or_abstention", "clarification_required",
        "missing_function", "irrelevant",
    )
    if is_no_tool and real_required_tools:
        raise ValueError(
            f"No-tool task {task.task_id} unexpectedly has "
            f"{len(real_required_tools)} tool calls"
        )
    if not is_no_tool and not (2 <= len(real_required_tools) <= 5):
        raise ValueError(
            f"Tool task {task.task_id} has oracle length "
            f"{len(real_required_tools)}, expected 2-5"
        )


def _filter_training_eligible_tasks(tasks: list) -> list:
    eligible = []
    dropped = 0
    for task in tasks:
        try:
            _validate_task_training_contract(task)
        except ValueError as exc:
            dropped += 1
            logger.warning("Dropping generated task before split: {}", exc)
            continue
        eligible.append(task)
    if dropped:
        logger.warning(
            "Dropped {} generated task(s) that violate the training contract",
            dropped,
        )
    return eligible


def _stratified_task_split(
    tasks: list,
    train_count: int,
    val_count: int,
    seed: int,
) -> tuple[list, list]:
    """Split one deduplicated pool by domain/scenario/difficulty.

    Generating the splits independently and deleting validation rows by tool
    signature collapsed the old validation set.  This routine assigns semantic
    fingerprints exactly once, then reserves validation coverage across strata.
    """
    import random
    from collections import defaultdict

    unique = []
    seen = set()
    for task in tasks:
        fp = _task_fingerprint(task)
        if fp in seen:
            continue
        seen.add(fp)
        task.metadata["semantic_fingerprint"] = fp
        unique.append(task)

    required = train_count + val_count
    if len(unique) < required:
        raise RuntimeError(
            f"Generation produced {len(unique)} unique tasks, but {required} "
            "are required for the requested train/val split."
        )

    rng = random.Random(seed)

    if val_count == 0:
        rng.shuffle(unique)
        return unique[:train_count], []

    # A category represented once cannot appear in both disjoint splits.
    # Drop such unsplittable outliers when the oversampled pool still has
    # enough rows, repeating because removing one category can expose another.
    strict_unique = list(unique)
    strict_possible = True
    while True:
        domain_totals = defaultdict(int)
        scenario_totals = defaultdict(int)
        for task in strict_unique:
            domain = task.target_servers[0] if task.target_servers else "unknown"
            domain_totals[domain] += 1
            scenario_totals[_task_scenario(task)] += 1
        singleton_domains = {key for key, count in domain_totals.items() if count < 2}
        singleton_scenarios = {key for key, count in scenario_totals.items() if count < 2}
        if not singleton_domains and not singleton_scenarios:
            break
        filtered = [
            task for task in strict_unique
            if (task.target_servers[0] if task.target_servers else "unknown")
            not in singleton_domains
            and _task_scenario(task) not in singleton_scenarios
        ]
        removed = len(strict_unique) - len(filtered)
        if len(filtered) < required:
            strict_possible = False
            logger.warning(
                "Strict train/val coverage is infeasible after removing "
                "{} singleton-category task(s): have {}, need {}. "
                "Falling back to exact disjoint split. domains={}, scenarios={}",
                removed,
                len(filtered),
                required,
                sorted(singleton_domains),
                sorted(singleton_scenarios),
            )
            break
        logger.warning(
            "Dropping {} unsplittable singleton-category task(s): domains={}, "
            "scenarios={}",
            removed,
            sorted(singleton_domains),
            sorted(singleton_scenarios),
        )
        strict_unique = filtered

    if not strict_possible:
        return _fallback_task_split(unique, train_count, val_count, rng)

    def _domain(task) -> str:
        return task.target_servers[0] if task.target_servers else "unknown"

    domain_counts = defaultdict(int)
    scenario_counts = defaultdict(int)
    for task in strict_unique:
        domain_counts[_domain(task)] += 1
        scenario_counts[_task_scenario(task)] += 1

    scenarios = sorted(scenario_counts)
    domains = sorted(domain_counts)
    if val_count < len(scenarios) or val_count < len(domains):
        logger.warning(
            "Validation count {} cannot cover {} scenario(s) and {} domain(s); "
            "falling back to exact disjoint split.",
            val_count,
            len(scenarios),
            len(domains),
        )
        return _fallback_task_split(unique, train_count, val_count, rng)

    # Proportional scenario targets with a minimum of one row per scenario
    # when feasible.  This makes val representative instead of dominated by
    # whichever coverage buckets happen to be popped first.
    total_available = len(strict_unique)
    raw_targets = {
        scenario: scenario_counts[scenario] * val_count / total_available
        for scenario in scenarios
    }
    scenario_targets = {
        scenario: min(scenario_counts[scenario] - 1, max(1, int(raw_targets[scenario])))
        for scenario in scenarios
    }
    while sum(scenario_targets.values()) < val_count:
        candidates = [
            scenario for scenario in scenarios
            if scenario_targets[scenario] < scenario_counts[scenario] - 1
        ]
        if not candidates:
            break
        scenario = max(
            candidates,
            key=lambda s: (raw_targets[s] - scenario_targets[s], scenario_counts[s], s),
        )
        scenario_targets[scenario] += 1
    while sum(scenario_targets.values()) > val_count:
        candidates = [
            scenario for scenario in scenarios
            if scenario_targets[scenario] > 1
        ]
        if not candidates:
            break
        scenario = min(
            candidates,
            key=lambda s: (raw_targets[s] - scenario_targets[s], scenario_counts[s], s),
        )
        scenario_targets[scenario] -= 1

    remaining = list(strict_unique)
    rng.shuffle(remaining)
    val = []
    val_domain_counts = defaultdict(int)
    val_scenario_counts = defaultdict(int)

    def _can_take(task) -> bool:
        domain = _domain(task)
        scenario = _task_scenario(task)
        return domain_counts[domain] > 1 and scenario_counts[scenario] > 1

    def _take_best(predicate, score) -> bool:
        candidates = [task for task in remaining if predicate(task) and _can_take(task)]
        if not candidates:
            return False
        task = max(candidates, key=score)
        remaining.remove(task)
        val.append(task)
        domain = _domain(task)
        scenario = _task_scenario(task)
        domain_counts[domain] -= 1
        scenario_counts[scenario] -= 1
        val_domain_counts[domain] += 1
        val_scenario_counts[scenario] += 1
        return True

    # Cover every domain, preferring scenarios that are still below their
    # proportional validation quota.
    for domain in domains:
        if len(val) >= val_count:
            break
        _take_best(
            lambda task, d=domain: _domain(task) == d,
            lambda task: (
                scenario_targets[_task_scenario(task)] - val_scenario_counts[_task_scenario(task)],
                scenario_counts[_task_scenario(task)],
                rng.random(),
            ),
        )

    # Fill scenario quotas.
    for scenario in scenarios:
        while len(val) < val_count and val_scenario_counts[scenario] < scenario_targets[scenario]:
            if not _take_best(
                lambda task, s=scenario: _task_scenario(task) == s,
                lambda task: (
                    0 if val_domain_counts[_domain(task)] else 1,
                    domain_counts[_domain(task)],
                    rng.random(),
                ),
            ):
                break

    # Fill any remaining slots by largest scenario deficit, still reserving
    # one train row for each domain and scenario.
    while len(val) < val_count:
        if not _take_best(
            lambda task: True,
            lambda task: (
                scenario_targets[_task_scenario(task)] - val_scenario_counts[_task_scenario(task)],
                0 if val_domain_counts[_domain(task)] else 1,
                domain_counts[_domain(task)],
                rng.random(),
            ),
        ):
            break
    if len(val) != val_count:
        logger.warning(
            "Strict validation allocation produced {}/{} rows; falling back "
            "to exact disjoint split.",
            len(val),
            val_count,
        )
        return _fallback_task_split(unique, train_count, val_count, rng)

    train = []

    def _pop_train(predicate) -> bool:
        for task in list(remaining):
            if predicate(task):
                train.append(task)
                remaining.remove(task)
                return True
        return False

    val_domains = {_domain(task) for task in val}
    val_scenarios = {_task_scenario(task) for task in val}
    for domain in sorted(val_domains):
        if len(train) >= train_count:
            break
        _pop_train(lambda task, d=domain: _domain(task) == d)
    for scenario in sorted(val_scenarios):
        if len(train) >= train_count:
            break
        if any(_task_scenario(task) == scenario for task in train):
            continue
        _pop_train(lambda task, s=scenario: _task_scenario(task) == s)

    rng.shuffle(remaining)
    train.extend(remaining[:max(0, train_count - len(train))])
    if len(train) != train_count:
        logger.warning(
            "Strict train allocation produced {}/{} rows; falling back to "
            "exact disjoint split.",
            len(train),
            train_count,
        )
        return _fallback_task_split(unique, train_count, val_count, rng)
    return train, val


def _fallback_task_split(
    tasks: list,
    train_count: int,
    val_count: int,
    rng,
) -> tuple[list, list]:
    """Exact disjoint split for small or highly imbalanced generated pools."""
    from collections import Counter

    pool = list(tasks)
    rng.shuffle(pool)
    required = train_count + val_count
    if len(pool) < required:
        raise RuntimeError(
            f"Cannot allocate fallback split from {len(pool)} tasks; need {required}"
        )

    def _domain(task) -> str:
        return task.target_servers[0] if task.target_servers else "unknown"

    domain_counts = Counter(_domain(task) for task in pool)
    scenario_counts = Counter(_task_scenario(task) for task in pool)

    val = []
    remaining = list(pool)
    for task in list(remaining):
        if len(val) >= val_count:
            break
        domain = _domain(task)
        scenario = _task_scenario(task)
        if domain_counts[domain] <= 1 or scenario_counts[scenario] <= 1:
            continue
        val.append(task)
        remaining.remove(task)
        domain_counts[domain] -= 1
        scenario_counts[scenario] -= 1

    for task in list(remaining):
        if len(val) >= val_count:
            break
        val.append(task)
        remaining.remove(task)

    train = []
    val_domains = {_domain(task) for task in val}
    val_scenarios = {_task_scenario(task) for task in val}

    for domain in sorted(val_domains):
        if len(train) >= train_count:
            break
        for task in list(remaining):
            if _domain(task) == domain:
                train.append(task)
                remaining.remove(task)
                break

    for scenario in sorted(val_scenarios):
        if len(train) >= train_count:
            break
        if any(_task_scenario(task) == scenario for task in train):
            continue
        for task in list(remaining):
            if _task_scenario(task) == scenario:
                train.append(task)
                remaining.remove(task)
                break

    train.extend(remaining[:max(0, train_count - len(train))])
    if len(train) != train_count or len(val) != val_count:
        raise RuntimeError(
            f"Fallback split size mismatch: train={len(train)}/{train_count}, "
            f"val={len(val)}/{val_count}"
        )
    return train, val


def _row_fingerprint(row) -> str:
    return str(row["extra_info"].get("semantic_fingerprint", ""))


def _assert_split_integrity(df_train, df_val, args) -> None:
    if len(df_train) != args.count or len(df_val) != args.val_count:
        raise RuntimeError(
            f"Split size mismatch: train={len(df_train)}/{args.count}, "
            f"val={len(df_val)}/{args.val_count}"
        )
    train_fp = {_row_fingerprint(row) for _, row in df_train.iterrows()}
    val_fp = {_row_fingerprint(row) for _, row in df_val.iterrows()}
    train_fp.discard("")
    val_fp.discard("")
    overlap = train_fp & val_fp
    if overlap:
        raise RuntimeError(f"Train/val semantic leakage: {len(overlap)} fingerprints")
    if args.val_count:
        train_domains = {row["extra_info"].get("domain") for _, row in df_train.iterrows()}
        val_domains = {row["extra_info"].get("domain") for _, row in df_val.iterrows()}
        if not val_domains.issubset(train_domains):
            raise RuntimeError(
                f"Validation domain not represented in train: train={sorted(train_domains)} "
                f"val={sorted(val_domains)}"
            )
        if train_domains != val_domains:
            logger.warning(
                "Validation domain coverage is a subset of train: train={} val={}",
                sorted(train_domains),
                sorted(val_domains),
            )
        train_scenarios = set(df_train["scenario_type"])
        val_scenarios = set(df_val["scenario_type"])
        if not val_scenarios.issubset(train_scenarios):
            raise RuntimeError(
                f"Validation scenario not represented in train: train={sorted(train_scenarios)} "
                f"val={sorted(val_scenarios)}"
            )
        if train_scenarios != val_scenarios:
            logger.warning(
                "Validation scenario coverage is a subset of train: train={} val={}",
                sorted(train_scenarios),
                sorted(val_scenarios),
            )


def _tasks_to_rows(tasks: list, base_seed: int) -> list[dict]:
    """Convert LiveTask list to verl-compatible data rows."""
    rows = []
    skipped_no_tools = 0
    for i, task in enumerate(tasks):
        # Determine visible tools — use task-provided tools, fall back to required
        visible_tools = task.visible_tools if task.visible_tools else []
        if not visible_tools:
            skipped_no_tools += 1
            logger.warning(
                f"Skipping task {task.task_id}: no visible_tools "
                f"(required_tools={task.required_tools}, "
                f"oracle_calls={len(task.oracle_program.calls) if task.oracle_program else 0})"
            )
            continue  # Skip tasks without tool schemas

        tools_desc_lines = []
        strip_enums = bool(task.metadata.get("strip_enums", False))
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
                enum_values = pinfo.get("enum")
                enum_desc = (
                    f" Allowed values: {json.dumps(enum_values, ensure_ascii=False)}."
                    if enum_values is not None and not strip_enums else ""
                )
                tools_desc_lines.append(
                    f"    - {pname} ({ptype}{req}): {pdesc}{enum_desc}"
                )
        tools_block = "\n".join(tools_desc_lines)
        visible_tool_names = [t.get("name", "") for t in visible_tools if t.get("name")]

        domain = task.target_servers[0] if task.target_servers else "unknown"

        system_prompt = (
            f"You are a helpful assistant with access to the following tools. "
            f"Use them when needed to answer the user's question.\n\n"
            f"Available tools:\n{tools_block}\n\n"
            f"Response format:\n"
            f'- To call a tool: <tool_call>{{"name": "tool_name", "arguments": {{...}}}}</tool_call>\n'
            f"  Emit exactly ONE tool_call per assistant turn, then wait for its tool result.\n"
            f"- To give final answer: <final_answer>your answer</final_answer>\n"
            f"- To report error: <report_error>error description</report_error>\n"
            f"- To ask clarification: <ask_clarification>your question</ask_clarification>\n"
            f"\n"
            f"Examples:\n"
            f"---\n"
            f"Example 1 (single-step query):\n"
            f"User: what events do I have next Tuesday?\n"
            f"Assistant: <tool_call>{{\"name\": \"search_something\", \"arguments\": {{\"keyword\": \"next Tuesday\"}}}}</tool_call>\n"
            f"Tool result: [{{\"id\": \"evt_01\", \"title\": \"Team Standup\", \"time\": \"10:00\"}}]\n"
            f"Assistant: <final_answer>You have one event on Tuesday: Team Standup at 10:00.</final_answer>\n"
            f"\n"
            f"Example 2 (multi-step with dependency):\n"
            f"User: move $200 from my savings to checking\n"
            f"Assistant: <tool_call>{{\"name\": \"check_balance\", \"arguments\": {{\"account\": \"savings\"}}}}</tool_call>\n"
            f"Tool result: {{\"balance\": 500}}\n"
            f"Assistant: <tool_call>{{\"name\": \"transfer_money\", \"arguments\": {{\"from\": \"savings\", \"to\": \"checking\", \"amount\": 200}}}}</tool_call>\n"
            f"Tool result: {{\"status\": \"success\", \"new_balance\": 300}}\n"
            f"Assistant: <final_answer>Transferred $200 from savings to checking. Remaining balance: $300.</final_answer>"
        )

        # One row always starts from reset(session_seed).  Tool observations
        # make the live rollout multi-turn; teacher calls must never be exposed
        # in the initial prompt unless they are also replayed into the session.
        prompt = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task.user_prompt},
        ]
        n_conversation_rounds = len(task.conversation_queries) or 1

        # Guard: multi-round tasks have oracle traces that span follow-up
        # queries, but the prompt only contains the first user message.
        # The model would see an impossible oracle contract.
        # Reject until multi-round prompt replay is implemented.
        if n_conversation_rounds > 1:
            raise RuntimeError(
                f"Task {task.task_id} has {n_conversation_rounds} conversation "
                f"rounds but multi-round prompt rendering is not implemented. "
                f"Rounds > 1 will produce oracle/prompt mismatch."
            )

        has_distractors = task.metadata.get("has_distractors", False)
        has_missing_func = task.metadata.get("has_missing_function", False)

        # BUG-E fix: perturbation_level encodes the PROVE information level
        # (complete/missing/minimal at 60/20/20), NOT the perturbation knob
        # (distractor/missing-function). Previously we overwrote this field
        # to "hard"/"medium" whenever a knob fired, which destroyed the
        # information-level stratification needed for GRPO advantage
        # computation. Keep difficulty intact; expose knob status via the
        # separate scenario_type/has_* fields.
        perturbation_level = task.difficulty
        scenario_type = _task_scenario(task)

        # 每个 task 独立一组：verl repeat(N) 后同一 prompt 的 N 个 rollout
        # 自然形成一个 group，回归标准 GRPO per-prompt 对比语义
        group_id = task.task_id

        # The prompt contains no teacher trajectory, so the complete oracle
        # (2-5 tool calls plus one explicit terminal action) is the unresolved
        # ground truth from reset(session_seed).  Multi-round teacher internals
        # can include per-round terminal actions; training rows keep only the
        # final terminal so the reward contract remains single-terminal.
        oracle_calls_serialized = _serialize_training_oracle(task)
        _validate_task_training_contract(task)

        success_criteria = _task_success_criteria(task)

        # P0-1 fix: success_criteria is a list[dict] whose 'value' field holds
        # mixed types (str status, numeric balance, etc.). Storing the raw list
        # in a Parquet dict column makes pyarrow infer a single value type and
        # crash with ArrowInvalid ("Could not convert 'paid' with type str:
        # tried to convert to double"). Serialize to JSON string for safe
        # round-trip; reward side parses it back via json.loads.
        success_criteria_json = json.dumps(
            success_criteria, ensure_ascii=False, default=str
        )

        terminal_actions = [
            c["action"] for c in oracle_calls_serialized
            if c.get("action") in ("final_answer", "ask_clarification", "report_error")
        ]
        if len(terminal_actions) != 1:
            raise ValueError(
                f"Task {task.task_id} has {len(terminal_actions)} terminal oracle actions"
            )
        allowed_terminal_actions = [terminal_actions[-1]]
        real_required_tools = [
            c["tool_name"] for c in oracle_calls_serialized
            if c.get("action", "tool_call") == "tool_call"
        ]
        is_no_tool = scenario_type in (
            "no_tool_or_abstention", "clarification_required", "missing_function", "irrelevant"
        )
        if is_no_tool and real_required_tools:
            raise ValueError(
                f"No-tool task {task.task_id} unexpectedly has {len(real_required_tools)} tool calls"
            )
        if not is_no_tool and not (2 <= len(real_required_tools) <= 5):
            raise ValueError(
                f"Tool task {task.task_id} has oracle length {len(real_required_tools)}, expected 2-5"
            )

        extra_info = {
            "task_id": task.task_id,
            "domain": domain,
            "target_servers": task.target_servers,
            "required_tools": real_required_tools,
            "session_seed": task.session_seed,
            "initial_state_hash": task.metadata.get("initial_state_hash", ""),
            "user_query": task.user_prompt,
            "budget": task.max_turns,
            "perturbation_level": perturbation_level,
            "scenario_type": scenario_type,
            "group_id": group_id,
            "uid": task.task_id,
            "has_distractors": has_distractors,
            "has_missing_function": has_missing_func,
            "enum_stripped": strip_enums,
            "identity_policy": task.metadata.get("identity_policy", _identity_policy(domain)),
            "target_resource_ids": task.metadata.get("target_resource_ids", []),
            "protected_resources": task.metadata.get("protected_resources", []),
            "protected_fields": task.metadata.get("protected_fields", []),
            # JSON string avoids Arrow's unsupported empty struct type when
            # a split happens to contain no protected-field mappings.
            "protected_fields_by_resource": json.dumps(
                task.metadata.get("protected_fields_by_resource", {}),
                ensure_ascii=False,
                default=str,
            ),
            "allowed_terminal_actions": allowed_terminal_actions,
            "semantic_fingerprint": task.metadata.get("semantic_fingerprint", ""),
            "generation_method": task.metadata.get("generation_method", "task_planner"),
            # P1-11: serialize oracle_calls to JSON string to prevent
            # pyarrow struct unification. When different oracle_calls have
            # heterogeneous arguments (e.g. {"event_id": "x"} vs
            # {"title": "x", "start_time": "x"}), pyarrow collapses all
            # keys into a unified struct with null fill. Serializing
            # oracle_calls as JSON preserves sparse argument dicts.
            "oracle_calls": json.dumps(oracle_calls_serialized, ensure_ascii=False, default=str),
            "success_criteria": success_criteria_json,
            "hidden_tools": list(task.hidden_tools) if task.hidden_tools else [],
            "visible_tool_names": visible_tool_names,
            "conversation_rounds": n_conversation_rounds,
        }

        row = {
            "prompt": json.dumps(prompt, ensure_ascii=False),
            "data_source": "live_mcp_state_machine",
            "reward_model": {
                "style": "rule",
                "ground_truth": {
                    "task_id": task.task_id,
                    # Same JSON serialization as extra_info.oracle_calls
                    "oracle_calls": json.dumps(oracle_calls_serialized, ensure_ascii=False, default=str),
                    "success_criteria": success_criteria_json,
                    "required_tools": real_required_tools,
                    "allowed_terminal_actions": allowed_terminal_actions,
                },
            },
            "extra_info": extra_info,
            "uid": extra_info["uid"],
            "group_id": group_id,
            "perturbation_level": perturbation_level,
            "scenario_type": scenario_type,
        }
        rows.append(row)

    if skipped_no_tools > 0:
        logger.warning(
            f"_tasks_to_rows 跳过了 {skipped_no_tools}/{len(tasks)} 个任务 "
            f"（visible_tools 为空）。请检查 task_planner 是否正确产出了 "
            f"visible_tools 字段。"
        )

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
        domain_dist = _domain_distribution(df_train)
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
        "empty_success_criteria": _empty_success_criteria_counts(df_train),
        "val_scenario_distribution": (
            df_val["scenario_type"].value_counts().to_dict()
            if len(df_val) > 0 else {}
        ),
        "timestamp": end_time.isoformat(),
    }
    (exp_dir / "result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    )

    logger.info(f"Experiment record saved: {exp_dir}")


def _domain_distribution(df: pd.DataFrame) -> dict:
    if len(df) == 0:
        return {}
    return df["extra_info"].apply(lambda x: x.get("domain")).value_counts().to_dict()


def _empty_success_criteria_counts(df: pd.DataFrame) -> dict:
    if len(df) == 0:
        return {}

    counts: dict[str, int] = {}
    for _, row in df.iterrows():
        extra_info = row["extra_info"]
        raw = extra_info.get("success_criteria", [])
        if isinstance(raw, str):
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                parsed = []
        else:
            parsed = raw or []
        if parsed:
            continue
        domain = extra_info.get("domain", "unknown")
        scenario = extra_info.get("scenario_type", row.get("scenario_type", "unknown"))
        key = f"{domain}/{scenario}"
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


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

    # Per-domain and per-scenario stats.
    domains = sorted(_domain_distribution(df_train))
    for domain in domains:
        domain_rows = df_train[
            df_train["extra_info"].apply(lambda x: x.get("domain") == domain)
        ]
        print(f"\n  {domain}: {len(domain_rows)} rows")
        scenario_counts = domain_rows["scenario_type"].value_counts().to_dict()
        for scenario, count in sorted(scenario_counts.items()):
            print(f"    {scenario}: {count}")

    # Difficulty distribution
    difficulty_dist = df_train["perturbation_level"].value_counts().to_dict()
    print(f"\n  Difficulty distribution: {difficulty_dist}")
    print(f"  Scenario distribution: {df_train['scenario_type'].value_counts().to_dict()}")
    if len(df_val) > 0:
        print(f"  Val scenario distribution: {df_val['scenario_type'].value_counts().to_dict()}")

    empty_criteria = _empty_success_criteria_counts(df_train)
    if empty_criteria:
        print("  Empty success_criteria diagnostics:")
        for key, count in empty_criteria.items():
            print(f"    {key}: {count}")

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
