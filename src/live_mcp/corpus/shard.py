#!/usr/bin/env python3
"""Internal shard generation for GRPO training with a state-machine teacher.

The teacher uses LLM-in-the-loop at every turn: the LLM sees the full domain
context (tools, live state, execution history) and decides the next action
(tool_call with real arguments, or terminal). The resulting oracle trace is
built by actual execution against live MCP servers.

Deployment modes:
  1. Local transformers:  --model models/Google/Gemma-4-31B-it
  2. vLLM server:         --model Gemma-4-31B-it --api-base http://localhost:8000/v1

Generation defaults:
- Difficulty mix: complete=60%, missing=20%, minimal=20%
  - Irrelevance ratio: 5%
  - Distractor rate: 40% (injects 3-8 irrelevant tools)
  - Missing function rate: 1500/(10895+1500) (derived local target; not a paper knob)
  - Enum stripping: 30% per domain
  - Jaccard dedup threshold: 0.70
  - Conversation rounds: 2-3 (turn-decay schedule)
  - Personas: 10 role templates, reference dates: 10 anchors
  - Recovery: explicit retry_same / retry_alt / give_up states
"""

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

from src.live_mcp.prompt_profiles import PROMPT_PROFILES
from src.live_mcp.dedup import dedup_tasks
from src.live_mcp.generation.mix_policy import default_difficulty_mix

from src.live_mcp.corpus.shard_recovery import (
    _accepted_generation_deficits,
    _domain_recovery_requests,
    _generation_recovery_requests,
    _maybe_checkpoint_generation_progress,
    _write_generation_checkpoint,
    _load_generation_checkpoint,
    _zero_yield_is_recoverable,
)
from src.live_mcp.corpus.shard_oracle import (
    _filter_training_eligible_tasks,
    _filter_required_tool_tasks,
)
from src.live_mcp.corpus.shard_row_projection import _tasks_to_rows
from src.live_mcp.corpus.shard_io import (
    _validate_canonical_rows_replay,
    _candidate_shard_split,
    _domain_quotas,
    _stratified_task_split,
    _assert_split_integrity,
    _print_stats,
)
from src.live_mcp.artifact.readback import validate_parquet_readback
from src.live_mcp.corpus.local_quality import (
    evaluate_persisted_candidate_quality,
)


def _filter_semantic_eligible_tasks(
    tasks: list,
    *,
    failure_writer,
    recovery_round: int,
) -> list:
    """Remove deterministic hard findings before shard accounting/writeout.

    Row projection is the first point at which the complete persisted semantic
    surface exists.  Consumer readback re-evaluates the same contract later,
    but must not be the first place a known-bad row is discovered.
    """
    accepted = []
    for task in tasks:
        try:
            rows = _tasks_to_rows([task], task.session_seed)
        except ValueError as exc:
            domain = str(
                task.target_servers[0]
                if getattr(task, "target_servers", None)
                else ""
            )
            failure_writer.append({
                "candidate_kind": "normal",
                "stage": "training_contract",
                "reason_code": "training_contract_invalid",
                "domain": domain,
                "generation_seed": int(
                    task.metadata.get("generation_seed", task.session_seed)
                ),
                "state_seed": int(task.session_seed),
                "difficulty": str(task.difficulty),
                "task_id": str(task.task_id),
                "recovery_round": recovery_round,
                "exception_type": type(exc).__name__,
                "message": str(exc),
            })
            logger.warning(
                "Prewrite training contract rejected task {}: {}",
                task.task_id,
                exc,
            )
            continue
        if len(rows) != 1:
            raise RuntimeError(
                "semantic prewrite projection must produce exactly one row: "
                f"task={task.task_id!r}, rows={len(rows)}"
            )
        extra = rows[0]["extra_info"]
        finding = evaluate_persisted_candidate_quality(extra)
        if finding is None:
            accepted.append(task)
            continue
        failure_writer.append({
            "candidate_kind": "normal",
            "stage": finding.stage,
            "reason_code": finding.reason_code,
            "domain": str(extra.get("domain") or ""),
            "generation_seed": int(
                task.metadata.get("generation_seed", task.session_seed)
            ),
            "state_seed": int(task.session_seed),
            "difficulty": str(task.difficulty),
            "task_id": str(task.task_id),
            "recovery_round": recovery_round,
            "message": finding.quality_issue,
            "details": {
                "semantic_gate_profile": str(
                    extra.get("semantic_gate_profile") or ""
                ),
                "quality_issue": finding.quality_issue,
            },
        })
        logger.warning(
            "Prewrite local quality rejected task {}: {}",
            task.task_id,
            finding.quality_issue,
        )
    return accepted




def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate GRPO training data with an LLM teacher"
    )
    p.add_argument("--count", type=int, default=500,
                    help="Number of training tasks to generate")
    p.add_argument("--val-count", type=int, default=50,
                    help="Number of validation tasks to generate")
    p.add_argument("--domain", default="all",
                    help="Domain (all, banking, calendar, etc.). "
                         "Comma-separated list supported, e.g. calendar,shopping,banking")
    p.add_argument("--model", required=True,
                    help="Model name (vLLM served name) or local path "
                         "(models/Google/Gemma-4-31B-it). "
                         "vLLM mode: must match --served-model-name from vLLM startup. "
                         "Local mode: absolute or relative path to model directory.")
    p.add_argument(
        "--teacher-model-id",
        default=None,
        help=(
            "Stable Teacher provenance ID for dependency-cache contracts; "
            "defaults to --model."
        ),
    )
    p.add_argument("--api-base", default=None,
                    help="OpenAI-compatible API base URL (local transformers if unset)")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument(
        "--prompt-profile",
        default=os.environ.get(
            "LIVEMCP_PROMPT_PROFILE", "paper_generation_baseline_v1"
        ),
        choices=tuple(PROMPT_PROFILES),
        help="Prompt contract used for causal generation gray tests",
    )
    p.add_argument(
        "--semantic-gate-profile",
        default=os.environ.get(
            "LIVEMCP_SEMANTIC_GATE_PROFILE", "diagnostic_only"
        ),
        choices=("diagnostic_only", "deterministic_v1"),
        help="Orthogonal completed-trace semantic audit disposition",
    )
    p.add_argument(
        "--fixed-attempt-budget",
        action="store_true",
        default=os.environ.get("LIVEMCP_FIXED_ATTEMPT_BUDGET", "0") == "1",
        help=(
            "Freeze diagnostic fixed-attempt semantics in checkpoints and "
            "output artifacts (normally supplied by corpus CLI/launcher)"
        ),
    )
    p.add_argument(
        "--retained-sequences-file",
        default=None,
        help=(
            "Merge deficit report whose retained plain tool sequences seed "
            "top-up scheduling novelty"
        ),
    )
    p.add_argument(
        "--difficulty",
        choices=("complete", "missing", "minimal"),
        default=None,
        help=(
            "Diagnostic fixed difficulty. Omit to retain the formal "
            "60/20/20 generation mix."
        ),
    )
    p.add_argument("--output", default="data/runs/manual/train.parquet",
                    help="Training data output path")
    p.add_argument("--val-output", default="data/runs/manual/val.parquet",
                    help="Validation data output path")
    p.add_argument("--suite", default="configs/live_mcp/ten_domain_suite.yaml",
                    help="Suite config path")
    p.add_argument("--irrelevance-ratio", type=float, default=0.05,
                    help="Fraction of tasks that require report_error (0 to disable)")
    p.add_argument(
        "--irrelevance-count",
        type=int,
        default=None,
        help=(
            "Exact irrelevance count for this initial shard. Production "
            "launchers allocate it globally; recovery rounds always use 0."
        ),
    )
    p.add_argument("--distractor-rate", type=float, default=0.40,
                    help="Probability of injecting distractor tools (0 to disable)")
    p.add_argument("--missing-function-rate", type=float, default=1500 / (10895 + 1500),
                    help="Probability of hiding a required tool (0 to disable)")
    p.add_argument("--device", type=int, default=None,
                    help="GPU device ID for local inference (default: auto). "
                         "Use with CUDA_VISIBLE_DEVICES for multi-GPU data-parallel.")
    p.add_argument("--log-file", default=None,
                    help="Write all logs to this file (auto-flushed, avoids pipe buffering)")
    p.add_argument("--shard-mode", action="store_true",
                    help="Use shard-local integrity checks; global coverage is checked after merge")
    p.add_argument("--pool-oversample-ratio", type=float, default=0.50,
                    help="Candidate pool oversampling ratio (e.g. 0.50 = 50%% extra). Applied once by this process.")
    p.add_argument("--max-recovery-rounds", type=int, default=3,
                    help="Maximum generation/recovery rounds (must be >= 1)")
    p.add_argument("--checkpoint-path", default=None,
                    help="Optional JSON checkpoint; resumes automatically when it exists")
    p.add_argument(
        "--failure-records-path",
        default=None,
        help=(
            "Append-only generation failure JSONL. Defaults beside the "
            "checkpoint, or beside --output when checkpointing is disabled."
        ),
    )
    p.add_argument(
        "--checkpoint-interval",
        type=int,
        default=25,
        help=(
            "Atomically checkpoint after this many newly accepted tasks "
            "(default: 25)"
        ),
    )
    p.add_argument(
        "--require-tool-calls",
        action="store_true",
        help="Retain only candidates whose required oracle contains a tool call",
    )
    return p


def generate_data(args: argparse.Namespace):
    """Generate GRPO training data with LLM teacher."""
    from src.live_mcp.generation_runtime import TeacherGenerationRuntime
    from src.live_mcp.corpus.failure_records import GenerationFailureWriter

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
    if args.irrelevance_count is not None and not (
        0 <= args.irrelevance_count <= args.count + args.val_count
    ):
        raise ValueError(
            "--irrelevance-count must be between 0 and count+val-count"
        )
    if Path(args.output).resolve() == Path(args.val_output).resolve():
        raise ValueError(
            f"--output and --val-output point to the same file: {args.output}"
        )
    if args.max_recovery_rounds < 1:
        raise ValueError(
            f"--max-recovery-rounds must be >= 1, got {args.max_recovery_rounds}"
        )
    if args.checkpoint_interval < 1:
        raise ValueError(
            f"--checkpoint-interval must be >= 1, got {args.checkpoint_interval}"
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

    print(f"[generate_data] Target: {args.count} train + {args.val_count} val tasks, domain={args.domain}, model={args.model}")
    logger.info(f"Generating GRPO data: {args.count} train + {args.val_count} val tasks")
    logger.info(f"  Domain: {args.domain}")
    logger.info(f"  Model: {args.model}")
    logger.info(
        f"  Difficulty: {args.difficulty or 'complete=60%, missing=20%, minimal=20%'}"
    )
    logger.info(f"  Prompt profile: {args.prompt_profile}")
    logger.info(f"  Semantic gate profile: {args.semantic_gate_profile}")

    branch = TeacherGenerationRuntime.from_suite(args.suite)
    configured_failure_path = getattr(args, "failure_records_path", None)
    if configured_failure_path:
        failure_records_path = Path(configured_failure_path)
    elif args.checkpoint_path:
        checkpoint_path = Path(args.checkpoint_path)
        failure_records_path = checkpoint_path.with_name(
            f"{checkpoint_path.stem}_failures.jsonl"
        )
    else:
        failure_records_path = Path(f"{args.output}.failures.jsonl")
    failure_writer = GenerationFailureWriter(failure_records_path)
    logger.info(f"Generation failure evidence: {failure_records_path}")
    if args.retained_sequences_file:
        retained_report = json.loads(
            Path(args.retained_sequences_file).read_text(encoding="utf-8")
        )
        retained_sequences = retained_report.get(
            "retained_tool_sequences_by_domain"
        )
        if not isinstance(retained_sequences, dict):
            raise ValueError(
                "retained sequence report is missing "
                "retained_tool_sequences_by_domain"
            )
        branch.preload_retained_sequences(retained_sequences)
    difficulty_mix = (
        {args.difficulty: 1.0}
        if args.difficulty
        else default_difficulty_mix()
    )

    try:
        branch.start()
        total_count = args.count + args.val_count
        if args.pool_oversample_ratio < 0:
            raise ValueError("--pool-oversample-ratio must be >= 0")
        # Oversampling is applied once. Multi-client launchers pass an already
        # allocated candidate budget and set this to zero for shard workers.
        pool_target = max(
            total_count,
            math.ceil(total_count * (1.0 + args.pool_oversample_ratio)),
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
        if len(requested_domains) != len(set(requested_domains)):
            raise ValueError(
                f"duplicate --domain entries are not allowed: {requested_domains}"
            )
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
            all_tasks = _filter_semantic_eligible_tasks(
                all_tasks,
                failure_writer=failure_writer,
                recovery_round=max(0, start_round - 1),
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
                checkpoint_prefix = [*all_tasks, *round_tasks]
                irrelevance_deficit, recovery_difficulty_mix = (
                    _accepted_generation_deficits(
                        checkpoint_prefix,
                        pool_target=pool_target,
                        configured_irrelevance_count=args.irrelevance_count,
                        configured_irrelevance_ratio=args.irrelevance_ratio,
                        difficulty_mix=difficulty_mix,
                    )
                )
                request_irrelevance_count = min(
                    request_count, irrelevance_deficit,
                )
                last_checkpoint_size = len(checkpoint_prefix)
                semantic_dispositions: dict[str, bool] = {}

                def filter_request_tasks(tasks: list) -> list:
                    unseen = [
                        task for task in tasks
                        if str(task.task_id) not in semantic_dispositions
                    ]
                    for task in unseen:
                        task.metadata["semantic_gate_profile"] = (
                            args.semantic_gate_profile
                        )
                        if task.metadata.get("fixed_attempt_budget") is not (
                            args.fixed_attempt_budget
                        ):
                            raise RuntimeError(
                                "generated task/run fixed-attempt contract "
                                f"mismatch: task={task.task_id!r}, task_value="
                                f"{task.metadata.get('fixed_attempt_budget')!r}, "
                                f"run_value={args.fixed_attempt_budget!r}"
                            )
                    accepted_unseen = _filter_semantic_eligible_tasks(
                        unseen,
                        failure_writer=failure_writer,
                        recovery_round=recovery_round,
                    )
                    accepted_ids = {
                        str(task.task_id) for task in accepted_unseen
                    }
                    for task in unseen:
                        semantic_dispositions[str(task.task_id)] = (
                            str(task.task_id) in accepted_ids
                        )
                    return [
                        task for task in tasks
                        if semantic_dispositions[str(task.task_id)]
                    ]

                def checkpoint_progress(partial_tasks: list) -> None:
                    nonlocal last_checkpoint_size
                    last_checkpoint_size = _maybe_checkpoint_generation_progress(
                        checkpoint_prefix=checkpoint_prefix,
                        partial_tasks=filter_request_tasks(partial_tasks),
                        last_checkpoint_size=last_checkpoint_size,
                        args=args,
                        requested_domains=requested_domains,
                        desired_domain_counts=desired_domain_counts,
                        total_count=total_count,
                        recovery_round=recovery_round,
                    )

                def record_failure(record: dict) -> None:
                    failure_writer.append({
                        "recovery_round": recovery_round,
                        "request_index": request_index,
                        "request_domain": request_domain,
                        "request_count": request_count,
                        "request_seed": round_seed + request_index * 10000,
                        **record,
                    })

                try:
                    generated = branch.generate_tasks(
                        server_name=request_domain,
                        count=request_count,
                        seed=round_seed + request_index * 10000,
                        difficulty_mix=recovery_difficulty_mix,
                        model_path=args.model,
                        teacher_model_id=args.teacher_model_id,
                        api_base=args.api_base,
                        device=args.device,
                        # Exact accepted deficit ownership avoids both initial
                        # under-yield and recovery-round ratio inflation.
                        irrelevance_ratio=0.0,
                        irrelevance_count=request_irrelevance_count,
                        distractor_rate=args.distractor_rate,
                        missing_function_rate=args.missing_function_rate,
                        prompt_profile=args.prompt_profile,
                        fixed_attempt_budget=args.fixed_attempt_budget,
                        progress_callback=(
                            checkpoint_progress if args.checkpoint_path else None
                        ),
                        failure_callback=record_failure,
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
                round_tasks.extend(filter_request_tasks(generated))
            all_tasks.extend(round_tasks)
            logger.info(
                f"Round {recovery_round + 1}: got {len(round_tasks)} tasks "
                f"(cumulative {len(all_tasks)})"
            )

            # Try to split early — if we already have enough, break out.
            eligible, pending_domain_requests = _generation_recovery_requests(
                all_tasks,
                args=args,
                requested_domains=requested_domains,
                desired_domain_counts=desired_domain_counts,
                total_count=total_count,
            )
            if len(eligible) >= total_count and not pending_domain_requests:
                try:
                    if args.shard_mode:
                        train_tasks, val_tasks = _candidate_shard_split(
                            eligible,
                            train_count=args.count,
                            val_count=args.val_count,
                            seed=args.seed,
                        )
                    else:
                        train_tasks, val_tasks = _stratified_task_split(
                            eligible, train_count=args.count,
                            val_count=args.val_count, seed=args.seed,
                            domain_quotas=desired_domain_counts,
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
                        round_requests = [
                            (args.domain, max(1, total_count - len(eligible)))
                        ]
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
            if args.require_tool_calls:
                eligible = _filter_required_tool_tasks(eligible)
            if args.shard_mode:
                if not eligible:
                    if not args.fixed_attempt_budget:
                        raise RuntimeError(
                            "Shard exhausted recovery without any eligible candidates"
                        )
                    train_tasks, val_tasks = [], []
                else:
                    shard_train_count = min(args.count, len(eligible))
                    shard_val_count = min(
                        args.val_count, len(eligible) - shard_train_count,
                    )
                    train_tasks, val_tasks = _candidate_shard_split(
                        eligible,
                        train_count=shard_train_count,
                        val_count=shard_val_count,
                        seed=args.seed,
                    )
            else:
                train_tasks, val_tasks = _stratified_task_split(
                    eligible, train_count=args.count,
                    val_count=args.val_count, seed=args.seed,
                    domain_quotas=desired_domain_counts,
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
        assert branch.executor is not None
        _validate_canonical_rows_replay(
            [*all_rows, *val_rows],
            manager=branch.manager,
            executor=branch.executor,
        )
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
    validate_parquet_readback(Path(args.output))
    validate_parquet_readback(Path(args.val_output))

    _print_stats(df_train, df_val, Path(args.output), Path(args.val_output), args)




if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    generate_data(args)
