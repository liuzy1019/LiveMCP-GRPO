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
import math
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from loguru import logger

from src.live_mcp.task_planner import (
    _is_mutating_tool, _SELF_CONTAINED_WRITE_TOOLS,
    DOMAIN_DESCRIPTIONS, _format_tools,
)
from src.live_mcp.dedup import dedup_tasks


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate PROVE-style GRPO training data with LLM teacher"
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
    p.add_argument("--missing-function-rate", type=float, default=1500 / (10895 + 1500),
                    help="Probability of hiding a required tool (0 to disable)")
    p.add_argument("--device", type=int, default=None,
                    help="GPU device ID for local inference (default: auto). "
                         "Use with CUDA_VISIBLE_DEVICES for multi-GPU data-parallel.")
    p.add_argument("--experiment-tag", default=None,
                    help="Tag for experiment tracking. If set, writes config.json and "
                         "result.json to data/experiments/{YYYY-MM-DD}_{tag}/")
    p.add_argument("--log-file", default=None,
                    help="Write all logs to this file (auto-flushed, avoids pipe buffering)")
    p.add_argument("--shard-mode", action="store_true",
                    help="Use shard-local integrity checks; global coverage is checked after merge")
    p.add_argument("--pool-oversample-pct", type=float, default=0.50,
                    help="Candidate oversample ratio applied once by this process")
    p.add_argument("--max-recovery-rounds", type=int, default=6,
                    help="Maximum generation/recovery rounds (must be >= 1)")
    p.add_argument("--checkpoint-path", default=None,
                    help="Optional JSON checkpoint; resumes automatically when it exists")
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
    if args.max_recovery_rounds < 1:
        raise ValueError(
            f"--max-recovery-rounds must be >= 1, got {args.max_recovery_rounds}"
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
        if args.pool_oversample_pct < 0:
            raise ValueError("--pool-oversample-pct must be >= 0")
        # Oversampling is applied once. Multi-client launchers pass an already
        # allocated candidate budget and set this to zero for shard workers.
        pool_target = max(
            total_count,
            math.ceil(total_count * (1.0 + args.pool_oversample_pct)),
        )
        print(f"[generate_data] Generating pool of ~{pool_target} tasks...", flush=True)

        all_tasks = []
        max_recovery_rounds = args.max_recovery_rounds
        if args.domain == "all":
            requested_domains = list(branch.manager.server_names)
        else:
            requested_domains = [
                item.strip() for item in args.domain.split(",") if item.strip()
            ]
        train_domain_quotas = _domain_quotas(args.count, requested_domains)
        val_domain_quotas = _domain_quotas(args.val_count, requested_domains)
        desired_domain_counts = {
            domain: train_domain_quotas[domain] + val_domain_quotas[domain]
            for domain in requested_domains
        }
        round_requests: list[tuple[str, int]] = [(args.domain, pool_target)]
        start_round = 0
        if args.checkpoint_path and Path(args.checkpoint_path).exists():
            all_tasks, start_round, round_requests = _load_generation_checkpoint(
                Path(args.checkpoint_path), args
            )
            logger.info(
                f"Resumed generation checkpoint: {len(all_tasks)} tasks, "
                f"completed_rounds={start_round}, next requests={round_requests}"
            )
        for recovery_round in range(start_round, max_recovery_rounds):
            round_seed = args.seed + recovery_round * 100000
            print(
                f"[generate_data] Round {recovery_round + 1}/{max_recovery_rounds}: "
                f"requests={round_requests} (seed={round_seed})...",
                flush=True,
            )
            round_tasks = []
            for request_index, (request_domain, request_count) in enumerate(round_requests):
                if request_count <= 0:
                    continue
                try:
                    generated = branch.generate_tasks_llm(
                        server_name=request_domain,
                        count=request_count,
                        seed=round_seed + request_index * 10000,
                        difficulty_mix=difficulty_mix,
                        model_path=args.model,
                        api_base=args.api_base,
                        device=args.device,
                        irrelevance_ratio=args.irrelevance_ratio,
                        distractor_rate=args.distractor_rate,
                        missing_function_rate=args.missing_function_rate,
                    )
                except RuntimeError as exc:
                    # The initial request failing completely usually means the
                    # teacher/MCP path is unavailable and should remain fatal.
                    # During recovery, however, each request covers one
                    # deficient domain. A zero-yield domain must not prevent
                    # the remaining deficient domains from being attempted.
                    if not _zero_yield_is_recoverable(
                        exc,
                        recovery_round=recovery_round,
                        shard_mode=args.shard_mode,
                    ):
                        raise
                    logger.exception(
                        f"Round {recovery_round + 1}: recovery request for "
                        f"{request_domain} produced no tasks; continuing with "
                        "the other deficient domains"
                    )
                    continue
                round_tasks.extend(generated)
            all_tasks.extend(round_tasks)
            logger.info(
                f"Round {recovery_round + 1}: got {len(round_tasks)} tasks "
                f"(cumulative {len(all_tasks)})"
            )

            # Try to split early — if we already have enough, break out.
            eligible = _filter_training_eligible_tasks(all_tasks)
            unique_eligible = dedup_tasks(eligible, threshold=0.70)
            if args.shard_mode:
                # Shards are candidate pools. Per-domain quotas only become
                # meaningful after all shards share one global dedup pool.
                # Requiring every shard to satisfy them can fail even when the
                # combined pool has enough rows for every domain.
                remaining = max(0, total_count - len(unique_eligible))
                pending_domain_requests = (
                    [(args.domain, remaining)] if remaining else []
                )
            else:
                _, pending_domain_requests = _domain_recovery_requests(
                    unique_eligible,
                    requested_domains,
                    desired_domain_counts,
                )
            if len(eligible) >= total_count and not pending_domain_requests:
                try:
                    train_tasks, val_tasks = _stratified_task_split(
                        eligible, train_count=args.count,
                        val_count=args.val_count, seed=args.seed,
                        domain_quotas=(
                            None if args.shard_mode else desired_domain_counts
                        ),
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
                    unique = dedup_tasks(eligible, threshold=0.70)
                    unique_count = len(unique)
                    if args.shard_mode:
                        domain_counts = {}
                        round_requests = [(args.domain, max(1, total_count - unique_count))]
                    else:
                        domain_counts, round_requests = _domain_recovery_requests(
                            unique, requested_domains, desired_domain_counts,
                        )
                    logger.info(
                        f"Round {recovery_round + 1}: {len(eligible)} eligible "
                        f"tasks / {unique_count} Jaccard-unique; "
                        f"domain_counts={domain_counts}, next requests={round_requests}"
                    )
            else:
                unique = dedup_tasks(eligible, threshold=0.70)
                unique_count = len(unique)
                if args.shard_mode:
                    # A shard is only a candidate pool. Recover its global row
                    # shortfall without imposing per-domain quotas that are
                    # meaningful only after all shards are merged.
                    domain_counts = {}
                    round_requests = list(pending_domain_requests)
                else:
                    domain_counts, round_requests = _domain_recovery_requests(
                        unique, requested_domains, desired_domain_counts,
                    )
                logger.info(
                    f"Round {recovery_round + 1}: {len(eligible)} eligible / "
                    f"{unique_count} Jaccard-unique; domain_counts={domain_counts}, "
                    f"next requests={round_requests}"
                )
            if args.checkpoint_path:
                _write_generation_checkpoint(
                    Path(args.checkpoint_path), args, all_tasks,
                    completed_rounds=recovery_round + 1,
                    round_requests=round_requests,
                )
        else:
            # Exhausted recovery rounds — fall through with whatever we have.
            eligible = _filter_training_eligible_tasks(all_tasks)
            train_tasks, val_tasks = _stratified_task_split(
                eligible, train_count=args.count,
                val_count=args.val_count, seed=args.seed,
                domain_quotas=(None if args.shard_mode else desired_domain_counts),
            )
            print(
                f"[generate_data] Final split: {len(train_tasks)} train + "
                f"{len(val_tasks)} val from {len(eligible)} eligible "
                f"(pool {len(all_tasks)} after {max_recovery_rounds} rounds)",
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
        _save_experiment_record(args, df_train, df_val, start_time, difficulty_mix)

    _print_stats(df_train, df_val, Path(args.output), Path(args.val_output), args)


def _domain_recovery_requests(
    unique_tasks: list,
    requested_domains: list[str],
    desired_domain_counts: dict[str, int],
) -> tuple[dict[str, int], list[tuple[str, int]]]:
    """Target recovery at domains that are actually below final quota."""
    domain_counts = {
        domain: sum(
            1 for task in unique_tasks
            if task.target_servers and task.target_servers[0] == domain
        )
        for domain in requested_domains
    }
    requests = [
        (domain, max(1, math.ceil((target - domain_counts[domain]) * 1.25)))
        for domain, target in desired_domain_counts.items()
        if domain_counts[domain] < target
    ]
    return domain_counts, requests


def _domain_quotas(target: int, domains: list[str]) -> dict[str, int]:
    base, remainder = divmod(target, len(domains))
    return {
        domain: base + (1 if index < remainder else 0)
        for index, domain in enumerate(domains)
    }


def _checkpoint_config(args: argparse.Namespace) -> dict:
    """Fields that determine candidate semantics and deterministic seed ranges."""
    return {
        "count": args.count,
        "val_count": args.val_count,
        "domain": args.domain,
        "model": args.model,
        "api_base": args.api_base,
        "seed": args.seed,
        "suite": args.suite,
        "pool_oversample_pct": args.pool_oversample_pct,
        "irrelevance_ratio": args.irrelevance_ratio,
        "distractor_rate": args.distractor_rate,
        "missing_function_rate": args.missing_function_rate,
    }


def _write_generation_checkpoint(
    path: Path,
    args: argparse.Namespace,
    tasks: list,
    *,
    completed_rounds: int,
    round_requests: list[tuple[str, int]],
) -> None:
    """Atomically persist the internal candidate pool after a complete round."""
    from src.live_mcp.types import to_plain

    payload = {
        "version": 1,
        "config": _checkpoint_config(args),
        "completed_rounds": completed_rounds,
        "round_requests": [list(item) for item in round_requests],
        "tasks": [to_plain(task) for task in tasks],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str),
        encoding="utf-8",
    )
    tmp_path.replace(path)
    logger.info(
        f"Generation checkpoint saved: {path} "
        f"({len(tasks)} tasks, completed_rounds={completed_rounds})"
    )


def _load_generation_checkpoint(
    path: Path, args: argparse.Namespace
) -> tuple[list, int, list[tuple[str, int]]]:
    """Load a compatible candidate pool without weakening validation/dedup."""
    from src.live_mcp.types import live_task_from_dict

    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("version") != 1:
        raise ValueError(
            f"Unsupported generation checkpoint version: {payload.get('version')}"
        )
    expected = _checkpoint_config(args)
    actual = payload.get("config")
    if actual != expected:
        raise ValueError(
            "Generation checkpoint config mismatch; refusing to mix candidate pools. "
            f"checkpoint={actual}, current={expected}"
        )
    completed_rounds = int(payload.get("completed_rounds", 0))
    if completed_rounds < 0:
        raise ValueError("Generation checkpoint completed_rounds must be >= 0")
    tasks = [live_task_from_dict(item) for item in payload.get("tasks", [])]
    round_requests = [
        (str(item[0]), int(item[1])) for item in payload.get("round_requests", [])
    ]
    if not round_requests:
        raise ValueError("Generation checkpoint has no pending round_requests")
    return tasks, completed_rounds, round_requests


def _is_zero_yield_error(exc: RuntimeError) -> bool:
    """Recognize the orchestrator's explicit zero-yield contract failure."""
    return str(exc).startswith("generate_many produced 0 tasks")


def _zero_yield_is_recoverable(
    exc: RuntimeError,
    *,
    recovery_round: int,
    shard_mode: bool,
) -> bool:
    """Keep small shard quality rejections inside the existing recovery loop.

    A monolithic initial zero-yield remains fatal because it usually indicates
    that the Teacher or MCP path is unavailable.  A shard may request only one
    candidate, so its first rejection is not evidence of a systemic outage.
    """
    return _is_zero_yield_error(exc) and (shard_mode or recovery_round > 0)


def _task_scenario(task) -> str:
    explicit = task.metadata.get("scenario_type") if task.metadata else None
    if explicit:
        return str(explicit)
    # PROVE Step-3 missing-function variant produces clarification trajectories
    # (paper §3.2 Training Data: "1,500 clarification trajectories generated
    # by the missing-function variant of Step 3"). Abstention (report_error)
    # is reserved for the internal `irrelevant` scenario and the externally
    # sourced When2Call + xLAM-Irrelevance slice (G = ∅).
    if task.task_type == "missing_function":
        return "clarification_required"
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


def _required_round_oracle_calls(task, round_idx: int, round_calls: list) -> list:
    """Project Teacher actions onto required workflow steps for RL labels.

    The full live trace remains untouched on ``task.oracle_program`` and in the
    audit log.  Only calls explicitly tagged by the state machine as an exact
    no-progress repeat are omitted from the required workflow view.
    """
    histories = getattr(task, "execution_history_per_round", None) or []
    history = histories[round_idx] if round_idx < len(histories) else []
    history_cursor = 0
    required = []
    for call in round_calls:
        if getattr(call, "action", "tool_call") != "tool_call":
            required.append(call)
            continue
        matched = None
        for index in range(history_cursor, len(history)):
            event = history[index]
            if not isinstance(event, dict) or not bool(event.get("success")):
                continue
            if (
                str(event.get("tool_name") or "") == str(call.tool_name)
                and dict(event.get("arguments") or {}) == dict(call.arguments or {})
            ):
                matched = event
                history_cursor = index + 1
                break
        if matched is not None and matched.get("no_progress_warning"):
            continue
        required.append(call)
    return required


def _serialize_training_oracle(task) -> list[dict]:
    """Return tool calls plus exactly one terminal action for training.

    All task types that reach this function must have a non-empty oracle
    program.  The orchestrator always pre-fills oracle calls for
    missing_function and irrelevant tasks. External abstention rows
    (When2Call, xLAM-Irrelevance) are written directly to Parquet and do
    NOT flow through this path.
    """
    scenario_type = task.metadata.get("scenario_type") if task.metadata else None

    # ── Assert invariants ──────────────────────────────────────────
    # missing_function / clarification_required tasks MUST have a
    # pre-filled oracle.  If this fires, the orchestrator was changed
    # to skip oracle population — fix the caller, not this function.
    if task.task_type == "missing_function" or scenario_type == "clarification_required":
        if not (task.oracle_program and task.oracle_program.calls):
            raise ValueError(
                f"Task {task.task_id}: missing_function/clarification_required "
                f"task has no oracle calls — orchestrator should have pre-filled "
                f"ask_clarification terminal"
            )

    # irrelevance / abstention tasks MUST also have a pre-filled oracle.
    if scenario_type in ("no_tool_or_abstention", "irrelevant"):
        if not (task.oracle_program and task.oracle_program.calls):
            raise ValueError(
                f"Task {task.task_id}: {scenario_type} task has no oracle "
                f"calls — orchestrator should have pre-filled report_error"
            )

    raw_calls = []
    if task.oracle_program and task.oracle_program.calls:
        source_calls = list(task.oracle_program.calls)
        if getattr(task, "oracle_calls_per_round", None):
            source_calls = []
            for round_idx, round_calls in enumerate(task.oracle_calls_per_round):
                source_calls.extend(
                    _required_round_oracle_calls(task, round_idx, list(round_calls))
                )
        for oc in source_calls:
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


def _build_round_contracts(task) -> list[dict]:
    """P0-2: Build per-round contracts from oracle_calls_per_round.

    Each contract defines the expected tools and allowed terminal action
    for one conversation round.  The rollout loop enforces these contracts
    to prevent illegal terminal-advancing (e.g. report_error → follow-up).

    Returns:
        list[dict] with keys: round_idx, required_tools, allowed_terminal_actions
    """
    if not task.oracle_calls_per_round:
        # Single-round legacy: derive from flattened oracle_program
        oracle_calls = getattr(task.oracle_program, "calls", None) or []
        tools: list[str] = []
        terminal = "final_answer"
        for oc in oracle_calls:
            action = getattr(oc, "action", "tool_call")
            if action == "tool_call":
                tools.append(getattr(oc, "tool_name", ""))
            elif action in ("final_answer", "ask_clarification", "report_error"):
                terminal = action
        return [{
            "round_idx": 0,
            "required_tools": [t for t in tools if t],
            "allowed_terminal_actions": [terminal],
        }]

    contracts = []
    for round_idx, round_calls in enumerate(task.oracle_calls_per_round):
        if not round_calls:
            raise ValueError(
                f"Task {task.task_id}: oracle round {round_idx} is empty; "
                "a conversation round must contain a Teacher-emitted action"
            )
        required_round_calls = _required_round_oracle_calls(
            task, round_idx, list(round_calls),
        )
        tools: list[str] = []
        terminal = "final_answer"
        for oc in required_round_calls:
            action = getattr(oc, "action", "tool_call")
            if action == "tool_call":
                tools.append(getattr(oc, "tool_name", ""))
            elif action in ("final_answer", "ask_clarification", "report_error"):
                terminal = action
        contracts.append({
            "round_idx": round_idx,
            "required_tools": [t for t in tools if t],
            "allowed_terminal_actions": [terminal],
        })
    return contracts


def _minimum_action_budget(
    oracle_calls_serialized: list[dict],
    round_contracts: list[dict],
) -> int:
    """Minimum model actions needed to reproduce a multi-round reference.

    PROVE's 2--3 continuation turns are conversation rounds.  The rollout
    action loop also spends one iteration on every tool call and on the
    terminal that closes each round, so its engineering cap must cover both.
    """
    n_tool_calls = sum(
        1 for call in oracle_calls_serialized
        if call.get("action", "tool_call") == "tool_call"
    )
    return n_tool_calls + max(1, len(round_contracts))


def _task_success_criteria(task) -> list:
    if task.oracle_program and task.oracle_program.success_criteria:
        return list(task.oracle_program.success_criteria)
    if hasattr(task, "success_criteria") and task.success_criteria:
        return list(task.success_criteria)
    return []


def _validate_task_training_contract(task) -> None:
    oracle_calls_serialized = _serialize_training_oracle(task)
    terminal_actions = [
        call["action"] for call in oracle_calls_serialized
        if call.get("action") in ("final_answer", "ask_clarification", "report_error")
    ]
    if len(terminal_actions) != 1:
        raise ValueError(
            f"Task {task.task_id} has {len(terminal_actions)} terminal oracle actions"
        )
    terminal_action = terminal_actions[0]

    real_required_tools = [
        call["tool_name"] for call in oracle_calls_serialized
        if call.get("action", "tool_call") == "tool_call"
    ]
    scenario_type = _task_scenario(task)
    # PROVE alignment:
    #   missing_function variant (Step 3)   → ask_clarification (1,500 traj.)
    #   irrelevance queries + external      → report_error (1,122 abstention)
    #   normal / recovery / dependency      → final_answer (main slice)
    #
    is_no_tool = scenario_type in ("no_tool_or_abstention", "irrelevant")
    is_optional_tool = scenario_type in (
        "clarification_required", "missing_function",
    )
    if is_no_tool and real_required_tools:
        raise ValueError(
            f"No-tool task {task.task_id} unexpectedly has "
            f"{len(real_required_tools)} tool calls"
        )
    if not is_no_tool and not is_optional_tool and not real_required_tools:
        raise ValueError(
            f"Tool task {task.task_id} has oracle length "
            f"{len(real_required_tools)}, expected at least one call"
        )

    # ── P1-2(now P0): tool tasks must be chain-seeded ──
    # PROVE baseline: every normal MCP conversation is a dependency-graph
    # chain-seed query (§3.2 Step 2).  Unseeded fallback data pollutes the
    # training distribution — reject before Parquet.
    if not is_no_tool and not is_optional_tool:
        generation_mode = (
            task.metadata.get("generation_mode", "")
            if task.metadata
            else ""
        )
        if generation_mode != "chain_seeded":
            raise ValueError(
                f"Task {task.task_id}: generation_mode='{generation_mode}', "
                f"expected 'chain_seeded' for tool-task baseline. "
                f"Unseeded fallback is NOT allowed in baseline training data."
            )

    # ── P3c: Detect final_answer tasks whose oracle did not produce state
    # criteria despite the user query requesting a write/mutate action.
    # These tasks teach models to call a few tools then final_answer without
    # actually completing the user's request.  We only WARN (not reject)
    # because some legitimate operations (e.g. send_email) are not tracked
    # in the state machine and naturally have empty criteria.
    if terminal_action == "final_answer" and real_required_tools:
        criteria = _task_success_criteria(task)
        if not criteria:
            state_changing = [t for t in real_required_tools
                             if _is_mutating_tool(t)
                             and t not in _SELF_CONTAINED_WRITE_TOOLS]
            if state_changing:
                # PROVE does NOT reject tasks with empty criteria: R_coverage
                # operates on tool-call sequences, not state diffs (§3.3).
                # Rejecting here conflicts with the oracle length [1,8] gate
                # above (which already accepted the task) and causes ~50% yield
                # loss.  The task still has a valid oracle trace; empty criteria
                # just means R_state will not reward this specific dimension.
                logger.warning(
                    f"Task {task.task_id}: final_answer with {state_changing} "
                    f"but empty success_criteria.  Accepting — R_coverage "
                    f"will use pure tool-call matching."
                )

    # ── P3d: tool_error_recovery with empty criteria is semantically broken ──
    # tool_error_recovery indicates the oracle encountered execution failures
    # and performed recovery steps.  If the oracle trace uses only readonly
    # tools (e.g. get_order + list_orders in food_delivery), P3c won't catch
    # it because none of the tools are mutating.  But a recovery scenario
    # without any state change means the "recovery" was just reading data —
    # the task teaches nothing useful about error handling.
    if scenario_type == "tool_error_recovery":
        criteria = _task_success_criteria(task)
        if not criteria:
            # PROVE does NOT reject tasks with empty criteria (§3.3).
            # tool_error_recovery classification is based on execution
            # history heuristics, not ground truth.  An empty-criteria
            # recovery task still has a valid oracle trace; R_coverage
            # will use pure tool-call matching.
            logger.warning(
                f"Task {task.task_id} scenario=tool_error_recovery with "
                f"empty success_criteria — oracle tools {real_required_tools} "
                f"are all readonly.  Accepting — R_coverage will use pure "
                f"tool-call matching."
            )

    # ── P0-2: validate round contract integrity before Parquet export ──
    contracts = _build_round_contracts(task)
    queries = task.conversation_queries or [task.user_prompt]
    n_contracts = len(contracts)
    n_queries = len(queries)
    if n_contracts != n_queries:
        raise ValueError(
            f"Task {task.task_id}: {n_contracts} round_contracts vs "
            f"{n_queries} conversation queries — counts must match."
        )
    for i, c in enumerate(contracts):
        if c.get("round_idx", -1) != i:
            raise ValueError(
                f"Task {task.task_id}: round_contracts[{i}] "
                f"round_idx={c.get('round_idx')}, expected {i}"
            )
        required = c.get("required_tools", [])
        if not isinstance(required, list) or not all(isinstance(t, str) for t in required):
            raise ValueError(
                f"Task {task.task_id}: round_contracts[{i}].required_tools "
                f"must be list[str], got {type(required)}"
            )
        allowed = c.get("allowed_terminal_actions", [])
        if not isinstance(allowed, list) or not allowed:
            raise ValueError(
                f"Task {task.task_id}: round_contracts[{i}]."
                f"allowed_terminal_actions must be non-empty list[str], "
                f"got {allowed}"
            )
        for a in allowed:
            if a not in ("final_answer", "ask_clarification", "report_error"):
                raise ValueError(
                    f"Task {task.task_id}: round_contracts[{i}]."
                    f"allowed_terminal_actions contains unknown action '{a}'"
                )
    # ── P0-3: dependency edge integrity ──
    # Chain-seeded tasks MUST produce exactly len(chain_seed)-1 valid edges.
    # Incomplete or invalid edges are data integrity errors — reject before split.
    chain_seed = (
        task.metadata.get("chain_seed", [])
        if task.metadata
        else []
    )
    if chain_seed:
        dependency_edges = _compute_dependency_edges(
            oracle_calls_serialized,
            chain_seed,
        )
        expected_edge_count = len(chain_seed) - 1
        if len(dependency_edges) != expected_edge_count:
            raise ValueError(
                f"Task {task.task_id}: incomplete dependency graph: "
                f"got {len(dependency_edges)} edges, "
                f"expected {expected_edge_count}; "
                f"chain_seed={chain_seed}"
            )
        # Validate every edge: src < dst, indices in range of real_required_tools
        for edge in dependency_edges:
            if (
                not isinstance(edge, list)
                or len(edge) != 2
                or not all(isinstance(i, int) for i in edge)
                or edge[0] < 0
                or edge[1] >= len(real_required_tools)
                or edge[0] >= edge[1]
            ):
                raise ValueError(
                    f"Task {task.task_id}: invalid dependency edge "
                    f"{edge}; chain_seed={chain_seed}; "
                    f"oracle_tool_count={len(real_required_tools)}"
                )

    # ── P0: missing-function contract integrity ──
    # Enforce that missing-function samples are internally consistent before
    # they reach Parquet / rollout / reward.
    has_missing_func = bool(
        (task.metadata or {}).get("has_missing_function")
    )
    if has_missing_func:
        hidden_tools_list = list(task.hidden_tools) if task.hidden_tools else []
        hidden_tool = (task.metadata or {}).get("hidden_tool", "")
        visible_names = {t.get("name", "") for t in (task.visible_tools or [])}

        # 1. hidden_tools must be non-empty and consistent with metadata
        if not hidden_tools_list:
            raise ValueError(
                f"Task {task.task_id}: has_missing_function=True but "
                f"hidden_tools is empty — missing-function contract broken."
            )
        if hidden_tool and hidden_tool not in hidden_tools_list:
            raise ValueError(
                f"Task {task.task_id}: metadata.hidden_tool='{hidden_tool}' "
                f"not in hidden_tools={hidden_tools_list}."
            )

        # 2. hidden tool must NOT appear in visible_tool_names (schema leak)
        leaked = set(hidden_tools_list) & visible_names
        if leaked:
            raise ValueError(
                f"Task {task.task_id}: hidden tool(s) {leaked} still present "
                f"in visible_tools schema — schema leak."
            )

        # 3. hidden tool must NOT appear in oracle tool calls
        oracle_tool_names = {
            call["tool_name"] for call in oracle_calls_serialized
            if call.get("action", "tool_call") == "tool_call"
        }
        oracle_blocked = set(hidden_tools_list) & oracle_tool_names
        if oracle_blocked:
            raise ValueError(
                f"Task {task.task_id}: hidden tool(s) {oracle_blocked} "
                f"appear in oracle tool calls — execution block failed."
            )

        # 4. terminal must be ask_clarification or report_error
        if terminal_action not in ("ask_clarification", "report_error"):
            raise ValueError(
                f"Task {task.task_id}: missing_function terminal is "
                f"'{terminal_action}', expected ask_clarification or report_error."
            )


def _filter_training_eligible_tasks(tasks: list) -> list:
    eligible = []
    dropped = 0
    for task in tasks:
        if task.metadata.get("paper_replay_valid") is False:
            dropped += 1
            logger.warning(
                "Dropping generated task before split: {} failed PROVE replay validation",
                task.task_id,
            )
            continue
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
    domain_quotas: dict[str, int] | None = None,
) -> tuple[list, list]:
    """Split one deduplicated pool by domain/scenario/difficulty.

    Generating the splits independently and deleting validation rows by tool
    signature collapsed the old validation set.  This routine assigns semantic
    fingerprints exactly once, then reserves validation coverage across strata.
    """
    import random
    from collections import defaultdict

    # Jaccard 0.70 dedup on tool-call sequences (PROVE corpus dedup §3.3).
    # Position-aware: {(index, tool_name)} — order and repeat count matter,
    # arguments are ignored (dedup.py/jaccard_similarity).
    # All surviving conversations share one dedup pool; PROVE does not publish
    # a domain exemption.
    unique = dedup_tasks(tasks, threshold=0.70)
    # Assign fingerprints after dedup for downstream cross-shard exact dedup.
    for task in unique:
        fp = _task_fingerprint(task)
        task.metadata["semantic_fingerprint"] = fp

    required = train_count + val_count
    if len(unique) < required:
        raise RuntimeError(
            f"Generation produced {len(unique)} unique tasks, but {required} "
            "are required for the requested train/val split."
        )

    rng = random.Random(seed)
    if domain_quotas is not None:
        by_domain = defaultdict(list)
        for task in unique:
            domain = task.target_servers[0] if task.target_servers else "unknown"
            by_domain[domain].append(task)
        domains = list(domain_quotas)
        train_quotas = _domain_quotas(train_count, domains)
        val_quotas = _domain_quotas(val_count, domains)
        train: list = []
        val: list = []
        for domain in domains:
            quota = train_quotas[domain] + val_quotas[domain]
            candidates = by_domain[domain]
            if len(candidates) < quota:
                raise RuntimeError(
                    f"Domain {domain} produced {len(candidates)} unique tasks, "
                    f"but {quota} are required."
                )
            rng.shuffle(candidates)
            train.extend(candidates[:train_quotas[domain]])
            val.extend(candidates[train_quotas[domain]:quota])
        return train, val

    # Domain quotas are enforced during generation. PROVE does not require
    # every domain/scenario label to appear in both small disjoint splits, and
    # deleting singleton strata wastes valid replay-checked candidates.
    return _fallback_task_split(unique, train_count, val_count, rng)


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
        logger.warning(
            "Train/val fingerprint collision within shard — deduplicating val (%d rows). "
            "This is expected at shard level; final merge handles cross-shard dedup.",
            len(overlap)
        )
        drop_indices = []
        for idx, row in df_val.iterrows():
            fp = _row_fingerprint(row)
            if fp and fp in overlap:
                drop_indices.append(idx)
        df_val.drop(drop_indices, inplace=True)
        df_val.reset_index(drop=True, inplace=True)
        val_fp = {_row_fingerprint(row) for _, row in df_val.iterrows()}
        val_fp.discard("")
        logger.warning(
            "After dedup: train=%d val=%d (removed %d colliding val rows)",
            len(df_train), len(df_val), len(drop_indices)
        )
    if args.val_count and not args.shard_mode:
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
            logger.warning(
                "Validation contains scenario labels absent from train: train={} val={}",
                sorted(train_scenarios),
                sorted(val_scenarios),
            )


def _compute_dependency_edges(
    oracle_calls: list[dict],
    chain_seed: list[str],
) -> list[list[int]]:
    """Compute dependency edges E by aligning chain_seed to oracle_calls.

    PROVE eq.(1): o(g) = Π_{(j,g)∈E} ⊮[μ(j) < μ(g)] requires explicit edges.

    Algorithm:
      1. Map every tool_call in oracle_calls to its index, grouped by tool_name.
      2. Walk chain_seed left-to-right, consuming the next occurrence after the
         previous cursor.  A chain step can be both a dst (of the previous edge)
         and a src (of the next edge) — intermediate nodes are NOT consumed.
      3. If any chain step cannot be aligned (no occurrence after cursor), return
         an empty list.  The caller (_validate_task_training_contract) rejects
         chain-seeded tasks with incomplete edges.

    Returns:
        list[list[int]] — [[src_idx, dst_idx], ...] where src_idx < dst_idx,
        or [] if the chain cannot be aligned to the oracle sequence.
    """
    if not chain_seed:
        return []

    # Step 1: collect all tool_call positions grouped by tool_name
    tool_positions: dict[str, list[int]] = {}
    for idx, call in enumerate(oracle_calls):
        if call.get("action", "tool_call") != "tool_call":
            continue
        tool_name = call.get("tool_name")
        if tool_name:
            tool_positions.setdefault(tool_name, []).append(idx)

    # Step 2: align chain_seed → strictly increasing oracle index sequence
    chain_indices: list[int] = []
    cursor = -1

    for tool_name in chain_seed:
        positions = tool_positions.get(tool_name, [])
        next_idx = next(
            (pos for pos in positions if pos > cursor),
            None,
        )
        if next_idx is None:
            logger.warning(
                "_compute_dependency_edges: cannot align chain step "
                "'{}' after oracle index {}. chain_seed={}",
                tool_name, cursor, chain_seed,
            )
            return []
        chain_indices.append(next_idx)
        cursor = next_idx

    # Step 3: build edges from consecutive aligned indices
    return [
        [chain_indices[i], chain_indices[i + 1]]
        for i in range(len(chain_indices) - 1)
    ]


def _tasks_to_rows(tasks: list, _base_seed: int) -> list[dict]:
    """Convert LiveTask list to verl-compatible data rows."""
    rows = []
    skipped_no_tools = 0
    for task in tasks:
        _validate_task_training_contract(task)

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

        visible_tool_names = [t.get("name", "") for t in visible_tools if t.get("name")]

        domain = task.target_servers[0] if task.target_servers else "unknown"

        domain_desc = DOMAIN_DESCRIPTIONS.get(domain, "")
        reference_date = task.metadata.get("reference_date", "")
        # Robustness knobs (enum stripping, distractors, missing_function) are
        # applied inside generate_one BEFORE Teacher processing and Replay.
        # task.visible_tools already contains the Teacher-visible candidate set.
        tools_text = _format_tools(visible_tools)
        date_line = f"\nToday's date: {reference_date}." if reference_date else ""
        initial_action_context = (
            task.sampling_context.get("initial_action_context", {})
            if isinstance(task.sampling_context, dict) else {}
        )
        initial_entity_summaries = (
            initial_action_context.get("entity_summaries", [])
            if isinstance(initial_action_context, dict) else []
        )
        initial_entity_summaries = [
            str(summary) for summary in initial_entity_summaries[:15]
            if str(summary).strip()
        ]
        observable_context = ""
        if initial_entity_summaries:
            observable_context = (
                "\n\n## Current Grounded Entities (Observable Context)\n"
                + "\n".join(initial_entity_summaries)
            )

        system_prompt = (
            f"You are an AI assistant for the following domain:\n{domain_desc}\n\n"
            f"## Available Tools\n{tools_text}{observable_context}\n\n"
            f"## Response Format\n"
            f"Output exactly ONE action per turn using XML tags:\n\n"
            f"- <tool_call>{{\"name\": \"<tool_name>\", \"arguments\": {{...}}}}</tool_call>\n"
            f"  Call a tool with its required parameters.\n\n"
            f"- <final_answer>your answer</final_answer>\n"
            f"  When the task is fully completed.\n\n"
            f"- <report_error>brief reason</report_error>\n"
            f"  When the task cannot be completed with available tools.\n\n"
            f"- <ask_clarification>what you need to know</ask_clarification>\n"
            f"  When genuinely missing critical information and no tool can resolve it.\n\n"
            f"## Rules\n"
            f"- Call ONE tool at a time. Wait for the result before the next action.\n"
            f"- Do not output hidden reasoning, chain-of-thought, or <think> tags.\n"
            f"- Use entity IDs only when they appear in the user request or tool results. "
            f"Never invent or guess IDs.{date_line}"
        )

        # One row always starts from reset(session_seed).  Teacher tool calls
        # are never exposed in the initial prompt.  For PROVE continuation
        # data, the rollout loop injects conversation_queries[1:] after
        # intermediate terminal actions in the same live MCP session.
        prompt = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task.user_prompt},
        ]
        n_conversation_rounds = len(task.conversation_queries) or 1
        conversation_queries = list(task.conversation_queries) if task.conversation_queries else [task.user_prompt]

        has_distractors = task.metadata.get("has_distractors", False)
        has_missing_func = task.metadata.get("has_missing_function", False)

        # perturbation_level encodes the PROVE information level, not the
        # robustness knob. Keep difficulty intact; expose knob status via the
        # separate scenario_type/has_* fields.
        perturbation_level = task.difficulty
        scenario_type = _task_scenario(task)

        # 每个 task 独立一组：verl repeat(N) 后同一 prompt 的 N 个 rollout
        # 自然形成一个 group，回归标准 GRPO per-prompt 对比语义
        group_id = task.task_id

        # The prompt contains no teacher trajectory, so the complete oracle
        # (tool calls plus one explicit terminal action) is the unresolved
        # ground truth from reset(session_seed).  Multi-round teacher internals
        # can include per-round terminal actions; training rows keep only the
        # final terminal so the reward contract remains single-terminal.
        oracle_calls_serialized = _serialize_training_oracle(task)
        round_contracts = _build_round_contracts(task)
        minimum_action_budget = _minimum_action_budget(
            oracle_calls_serialized, round_contracts,
        )
        action_budget = max(int(task.max_turns), minimum_action_budget)

        success_criteria = _task_success_criteria(task)

        # success_criteria is a list[dict] whose 'value' field can hold mixed
        # types. Serialize to JSON for a stable Parquet round-trip; reward side
        # parses it back via json.loads.
        success_criteria_json = json.dumps(
            success_criteria, ensure_ascii=False, default=str
        )

        # terminal_actions and real_required_tools were checked above by
        # _validate_task_training_contract; keep the local assert as an internal
        # consistency guard for this serialization block.
        terminal_actions = [
            c["action"] for c in oracle_calls_serialized
            if c.get("action") in ("final_answer", "ask_clarification", "report_error")
        ]
        assert terminal_actions, f"Bug: {task.task_id} serialized oracle has no terminal"
        allowed_terminal_actions = [terminal_actions[-1]]
        real_required_tools = [
            c["tool_name"] for c in oracle_calls_serialized
            if c.get("action", "tool_call") == "tool_call"
        ]

        # ── Dependency edges (PROVE eq.(1) o(g) requires E) ──
        chain_seed = task.metadata.get("chain_seed", []) if task.metadata else []
        dependency_edges = _compute_dependency_edges(oracle_calls_serialized, chain_seed)
        dependency_edges_json = json.dumps(dependency_edges, ensure_ascii=False)
        # P0 quality flag: chain_seeded tasks MUST produce exactly
        # len(chain_seed)-1 edges, enforced by _validate_task_training_contract
        # before reaching this point.  This field is diagnostic only.
        expected_edges = len(chain_seed) - 1 if chain_seed else 0
        dependency_graph_complete = (
            len(dependency_edges) == expected_edges and expected_edges > 0
        ) or not chain_seed
        generation_mode = task.metadata.get("generation_mode", "chain_seeded") if task.metadata else "chain_seeded"

        extra_info = {
            "task_id": task.task_id,
            "domain": domain,
            "target_servers": task.target_servers,
            "required_tools": real_required_tools,
            "session_seed": task.session_seed,
            "initial_state_hash": task.metadata.get("initial_state_hash", ""),
            "user_query": task.user_prompt,
            "budget": action_budget,
            "minimum_action_budget": minimum_action_budget,
            "perturbation_level": perturbation_level,
            "scenario_type": scenario_type,
            "group_id": group_id,
            "uid": task.task_id,
            "has_distractors": has_distractors,
            "has_missing_function": has_missing_func,
            "enum_stripped": task.metadata.get("strip_enums", False),
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
            "chain_seed": list(chain_seed),
            "source_chain_seed": list(
                task.metadata.get("source_chain_seed", []) if task.metadata else []
            ),
            "query_generation_attempts": int(
                task.metadata.get("query_generation_attempts", 1)
            ),
            "query_target_capability": str(
                task.metadata.get("query_target_capability", "")
            ),
            "query_chain_supported": bool(
                task.metadata.get("query_chain_supported", False)
            ),
            # Preserve the completed Teacher conversation sequence for PROVE
            # Jaccard dedup even when the required RL oracle omits an execution-
            # tagged no-progress repeat.
            "teacher_trace_tool_sequence": [
                str(call.tool_name)
                for call in (task.oracle_program.calls if task.oracle_program else [])
                if getattr(call, "action", "tool_call") == "tool_call"
            ],
            # Serialize oracle_calls as JSON so sparse heterogeneous argument
            # dicts round-trip through Parquet without struct unification.
            "oracle_calls": json.dumps(oracle_calls_serialized, ensure_ascii=False, default=str),
            "success_criteria": success_criteria_json,
            "hidden_tools": list(task.hidden_tools) if task.hidden_tools else [],
            "visible_tool_names": visible_tool_names,
            "tool_owner_domains": json.dumps({
                str(t.get("name")): str(t.get("_server_name") or domain)
                for t in visible_tools if t.get("name")
            }, ensure_ascii=False),
            "conversation_rounds": n_conversation_rounds,
            # Unperturbed schemas retained for robustness auditing.
            "clean_visible_tools": json.dumps(
                task.metadata.get("clean_visible_tools", visible_tools),
                ensure_ascii=False,
                default=str,
            ),
            "domain_desc": domain_desc,
            "reference_date": task.metadata.get("reference_date", ""),
            # JSON string avoids pyarrow nested-list surprises and lets the
            # live rollout inject follow-up user turns deterministically.
            "conversation_queries": json.dumps(
                conversation_queries, ensure_ascii=False, default=str
            ),
            # P0-2: per-round contracts for rollout enforcement.
            # Each contract specifies required_tools and allowed_terminal_actions
            # for one conversation round.  The rollout loop MUST validate the
            # model's terminal against the contract before injecting follow-up.
            "round_contracts": json.dumps(
                round_contracts, ensure_ascii=False, default=str
            ),
            "dependency_edges": dependency_edges_json,
            "dependency_graph_complete": dependency_graph_complete,
            "generation_mode": generation_mode,
            # P0-3: data quality signals from replay validation.
            "paper_replay_valid": task.metadata.get("paper_replay_valid", True),
            "project_outcome_valid": task.metadata.get("project_outcome_valid", True),
            "replay_error_rate": task.metadata.get("replay_error_rate", 0.0),
            "replay_num_calls": int(task.metadata.get("replay_num_calls", 0)),
            "replay_num_errors": int(task.metadata.get("replay_num_errors", 0)),
            "teacher_attempt_count": int(
                task.metadata.get("teacher_attempt_count", 0)
            ),
            "teacher_failed_attempt_count": int(
                task.metadata.get("teacher_failed_attempt_count", 0)
            ),
            "criteria_failed_count": task.metadata.get("criteria_failed", 0),
            "fsm_final_state": task.metadata.get("fsm_final_state", ""),
            "fsm_transitions": json.dumps(
                task.metadata.get("fsm_transitions", []),
                ensure_ascii=False,
                default=str,
            ),
        }

        row = {
            "prompt": json.dumps(prompt, ensure_ascii=False),
            "data_source": "live_mcp_state_machine",
            "reward_model": {
                "style": "rule",
                "ground_truth": {
                    "oracle_calls": json.dumps(oracle_calls_serialized, ensure_ascii=False, default=str),
                    "success_criteria": success_criteria_json,
                    "required_tools": real_required_tools,
                    "dependency_edges": dependency_edges_json,
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


def _save_experiment_record(args, df_train, df_val, start_time: datetime, difficulty_mix: dict):
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
        "difficulty_mix": difficulty_mix,
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
