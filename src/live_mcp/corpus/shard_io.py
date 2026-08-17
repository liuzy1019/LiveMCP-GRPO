"""Shard Io."""

from __future__ import annotations

import json

import pandas as pd
from loguru import logger

from src.live_mcp.dedup import dedup_tasks
from src.live_mcp.corpus.shard_oracle import _task_scenario
from src.live_mcp.prompt_profiles import requires_outcome_replay

def _validate_canonical_rows_replay(
    rows: list[dict],
    *,
    manager,
    executor,
) -> None:
    """Fresh-replay the exact required workflow that will be written.

    The Teacher attempt trace is replayed inside the orchestrator.  Export can
    subsequently project exact repeats and successful mutating no-ops out of
    the RL label, so the projected sequence needs its own isolated replay
    before Parquet publication.
    """
    from src.live_mcp.replay.gates import replay_validate
    from src.live_mcp.types import OracleCall
    from src.live_mcp.artifact.reward_task import (
        validate_ground_truth_consistency,
    )
    from src.utils import normalize_json_field

    for row_index, row in enumerate(rows):
        extra_info = row.get("extra_info") or {}
        reward_model = row.get("reward_model") or {}
        ground_truth = reward_model.get("ground_truth") or {}
        validate_ground_truth_consistency(extra_info, ground_truth)
        raw_calls = normalize_json_field(
            ground_truth.get("oracle_calls", "[]"), default=[],
        )
        raw_criteria = normalize_json_field(
            ground_truth.get("success_criteria", "[]"), default=[],
        )
        if not isinstance(raw_calls, list) or not isinstance(raw_criteria, list):
            raise RuntimeError(
                f"canonical replay row {row_index} has invalid oracle payload"
            )
        calls = [
            OracleCall(
                tool_name=str(call.get("tool_name") or ""),
                arguments=dict(call.get("arguments") or {}),
                action=str(call.get("action") or "tool_call"),
            )
            for call in raw_calls
            if isinstance(call, dict)
        ]
        hidden_raw = extra_info.get("hidden_tools", [])
        hidden = normalize_json_field(hidden_raw, default=[])
        if not isinstance(hidden, list):
            raise RuntimeError(
                f"canonical replay row {row_index} has invalid hidden_tools"
            )
        try:
            replay = replay_validate(
                oracle_calls=calls,
                manager=manager,
                executor=executor,
                seed=int(extra_info["session_seed"]),
                domain=str(extra_info["domain"]),
                success_criteria=raw_criteria,
                blocked_tools={str(name) for name in hidden},
                state_profiles=normalize_json_field(
                    extra_info.get("state_profiles"), default={},
                ),
            )
        except Exception as exc:
            raise RuntimeError(
                f"canonical replay failed for row {row_index} "
                f"task={extra_info.get('task_id', row.get('uid', 'unknown'))}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        valid, error_rate, num_errors, num_calls, criteria_ok, criteria_failed = replay
        outcome_required = requires_outcome_replay(
            str(extra_info.get("prompt_profile") or "")
        )
        if not valid or (outcome_required and not criteria_ok):
            raise RuntimeError(
                f"canonical replay rejected task="
                f"{extra_info.get('task_id', row.get('uid', 'unknown'))}: "
                f"valid={valid}, errors={num_errors}/{num_calls}, "
                f"criteria_failed={criteria_failed}"
            )
        extra_info["canonical_replay_valid"] = True
        extra_info["canonical_replay_error_rate"] = float(error_rate)
        extra_info["canonical_replay_num_errors"] = int(num_errors)
        extra_info["canonical_replay_num_calls"] = int(num_calls)
        extra_info["canonical_replay_criteria_ok"] = bool(criteria_ok)
        extra_info["canonical_replay_criteria_failed"] = int(criteria_failed)

def _candidate_shard_split(
    tasks: list,
    train_count: int,
    val_count: int,
    seed: int,
) -> tuple[list, list]:
    """Allocate an eligible candidate shard without local corpus gates.

    Global merge owns exact/fingerprint/Jaccard deduplication and domain
    quotas.  Keeping all eligible shard candidates is necessary for later
    top-up rounds to compose normally even when one domain has only a few
    distinct tool sequences.
    """
    required = train_count + val_count
    if len(tasks) < required:
        raise RuntimeError(
            f"Cannot allocate candidate shard from {len(tasks)} tasks; "
            f"need {required}"
        )
    candidates = list(tasks)
    for task in candidates:
        task.metadata["semantic_fingerprint"] = _task_fingerprint(task)
    import random
    random.Random(seed).shuffle(candidates)
    train = candidates[:train_count]
    val = candidates[train_count:required]
    return train, val

def _domain_quotas(target: int, domains: list[str]) -> dict[str, int]:
    base, remainder = divmod(target, len(domains))
    return {
        domain: base + (1 if index < remainder else 0)
        for index, domain in enumerate(domains)
    }

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

    # PROVE Jaccard 0.70 deduplication on plain tool-name sets.
    # All surviving conversations share one dedup pool.
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

    # Domain quotas are enforced during generation.
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
    if args.shard_mode:
        if (
            len(df_train) + len(df_val) == 0
            and not args.fixed_attempt_budget
        ):
            raise RuntimeError("Candidate shard contains no rows")
        if len(df_train) > args.count or len(df_val) > args.val_count:
            raise RuntimeError(
                f"Candidate shard exceeds allocation: "
                f"train={len(df_train)}/{args.count}, "
                f"val={len(df_val)}/{args.val_count}"
            )
    elif len(df_train) != args.count or len(df_val) != args.val_count:
        raise RuntimeError(
            f"Split size mismatch: train={len(df_train)}/{args.count}, "
            f"val={len(df_val)}/{args.val_count}"
        )
    train_fp = {_row_fingerprint(row) for _, row in df_train.iterrows()}
    val_fp = {_row_fingerprint(row) for _, row in df_val.iterrows()}
    train_fp.discard("")
    val_fp.discard("")
    overlap = train_fp & val_fp
    if overlap and not args.shard_mode:
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
