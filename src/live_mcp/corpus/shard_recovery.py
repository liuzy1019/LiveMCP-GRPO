"""Shard Recovery."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from loguru import logger

from src.live_mcp.task_planner import DOMAIN_DESCRIPTIONS
from src.live_mcp.planner_format import format_tools
from src.live_mcp.prompt_profiles import PROMPT_PROFILES, resolve_prompt_profile
from src.live_mcp.registry.tool_semantics import (
    SELF_CONTAINED_WRITE_TOOLS,
    is_mutating_tool,
    resolve_tool_execution_semantics,
)
from src.live_mcp.dedup import dedup_tasks
from src.live_mcp.dependency_trace import (
    align_sampled_chain, auxiliary_tool_call_indices,
    dependency_edges_from_alignment,
)

from src.live_mcp.corpus.shard_oracle import (
    _filter_required_tool_tasks,
    _filter_training_eligible_tasks,
)

def _accepted_generation_deficits(
    tasks: list,
    *,
    pool_target: int,
    configured_irrelevance_count: int | None,
    configured_irrelevance_ratio: float,
    difficulty_mix: dict[str, float],
) -> tuple[int, dict[str, float]]:
    """Return accepted-row deficits for the next generation request.

    The generation contract applies the irrelevance and difficulty mix to
    accepted candidates, not merely to the first batch of Teacher attempts.
    Recovery therefore targets the remaining accepted strata.  Once every
    configured stratum is full (for example, later Jaccard-capacity top-up),
    the original mix remains the least biased request distribution.
    """
    from src.live_mcp.generation.batch import largest_remainder_mix_quotas

    if configured_irrelevance_count is None:
        irrelevance_target = round(pool_target * configured_irrelevance_ratio)
    else:
        irrelevance_target = configured_irrelevance_count
    irrelevance_target = min(pool_target, max(0, irrelevance_target))
    accepted_irrelevance = sum(
        1 for task in tasks if str(getattr(task, "task_type", "")) == "irrelevant"
    )
    irrelevance_deficit = max(0, irrelevance_target - accepted_irrelevance)

    normal_target = pool_target - irrelevance_target
    difficulty_targets = largest_remainder_mix_quotas(
        normal_target, difficulty_mix,
    )
    accepted_difficulties = {
        difficulty: sum(
            1
            for task in tasks
            if str(getattr(task, "task_type", "")) != "irrelevant"
            and str(getattr(task, "difficulty", "")) == difficulty
        )
        for difficulty in difficulty_targets
    }
    difficulty_deficits = {
        difficulty: max(0, target - accepted_difficulties[difficulty])
        for difficulty, target in difficulty_targets.items()
    }
    if not any(difficulty_deficits.values()):
        return irrelevance_deficit, dict(difficulty_mix)
    return irrelevance_deficit, {
        difficulty: float(deficit)
        for difficulty, deficit in difficulty_deficits.items()
    }

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

def _generation_recovery_requests(
    tasks: list,
    *,
    args: argparse.Namespace,
    requested_domains: list[str],
    desired_domain_counts: dict[str, int],
    total_count: int,
) -> tuple[list, list[tuple[str, int]]]:
    """Return eligible candidates and the measured requests needed on resume."""
    eligible = _filter_training_eligible_tasks(tasks)
    if args.require_tool_calls:
        eligible = _filter_required_tool_tasks(eligible)
    if args.shard_mode:
        # Shards are candidate pools. Global Jaccard and domain allocation are
        # merge-owned, so a resumed shard only regenerates its eligible deficit.
        remaining = max(0, total_count - len(eligible))
        requests = [(args.domain, remaining)] if remaining else []
        return eligible, requests
    unique_eligible = dedup_tasks(eligible, threshold=0.70)
    _, requests = _domain_recovery_requests(
        unique_eligible,
        requested_domains,
        desired_domain_counts,
    )
    return eligible, requests

def _maybe_checkpoint_generation_progress(
    *,
    checkpoint_prefix: list,
    partial_tasks: list,
    last_checkpoint_size: int,
    args: argparse.Namespace,
    requested_domains: list[str],
    desired_domain_counts: dict[str, int],
    total_count: int,
    recovery_round: int,
) -> int:
    """Persist an accepted-task snapshot once the configured interval elapses."""
    current_size = len(checkpoint_prefix) + len(partial_tasks)
    if current_size - last_checkpoint_size < args.checkpoint_interval:
        return last_checkpoint_size
    cumulative_tasks = [*checkpoint_prefix, *partial_tasks]
    _, checkpoint_requests = _generation_recovery_requests(
        cumulative_tasks,
        args=args,
        requested_domains=requested_domains,
        desired_domain_counts=desired_domain_counts,
        total_count=total_count,
    )
    # Keep the checkpoint loadable even when this partial pool already
    # satisfies the target. The zero-count request is skipped after resume and
    # the saved pool proceeds directly to split.
    if not checkpoint_requests:
        checkpoint_requests = [(args.domain, 0)]
    _write_generation_checkpoint(
        Path(args.checkpoint_path),
        args,
        cumulative_tasks,
        # A partial checkpoint consumes this seed round. Resume continues from
        # the next recovery seed and measured deficit instead of regenerating
        # completed task seeds.
        completed_rounds=recovery_round + 1,
        round_requests=checkpoint_requests,
    )
    return current_size

def _checkpoint_config(args: argparse.Namespace) -> dict:
    """Fields that determine candidate semantics and deterministic seed ranges."""
    return {
        "count": args.count,
        "val_count": args.val_count,
        "domain": args.domain,
        "model": args.model,
        "teacher_model_id": getattr(args, "teacher_model_id", None) or args.model,
        "api_base": args.api_base,
        "seed": args.seed,
        "suite": args.suite,
        "pool_oversample_ratio": args.pool_oversample_ratio,
        "irrelevance_ratio": args.irrelevance_ratio,
        "irrelevance_count": getattr(args, "irrelevance_count", None),
        "distractor_rate": args.distractor_rate,
        "missing_function_rate": args.missing_function_rate,
        "prompt_profile": getattr(
            args, "prompt_profile", "paper_generation_baseline_v1"
        ),
        "semantic_gate_profile": getattr(
            args, "semantic_gate_profile", "diagnostic_only"
        ),
        "require_tool_calls": bool(getattr(args, "require_tool_calls", False)),
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
