"""Merge Allocation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from collections import defaultdict
from collections.abc import Hashable
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.live_mcp.protocol.observation import (
    TRAJECTORY_SCHEMA_VERSION, OBSERVATION_SCHEMA_VERSION,
    OBSERVATION_PROJECTION_VERSION, compute_server_schema_hash,
)
from src.live_mcp.domain_allocation import (
    capacity_weighted_domain_quotas, position_aware_jaccard,
)
from src.live_mcp.registry.tool_semantics import (
    is_mutating_tool, unresolved_failed_tool_names,
)
from src.live_mcp.corpus.semantic_quarantine import evaluate_semantic_quarantine
from src.live_mcp.domain_contracts.semantic_policies import evaluate_domain_label_issue

from src.live_mcp.corpus.merge_validation import (
    _as_extra,
    _as_json_list,
    _row_tool_sequence,
)

def _proportional_stratum_order(df: pd.DataFrame) -> pd.DataFrame:
    """Order rows by their observed joint strata without changing supply.

    Global Jaccard is a greedy hard gate.  Feeding it shard arrival order lets
    an early, high-yield stratum claim tool sequences before later strata are
    considered.  Weighted round-robin makes that gate capacity-aware while
    preserving its threshold and the candidate pool's observed proportions.
    """
    if df.empty:
        return df.reset_index(drop=True)
    buckets: dict[tuple[str, str, str], list[Hashable]] = {}
    for idx, row in df.iterrows():
        extra = _as_extra(row["extra_info"])
        domain = str(extra.get("domain", ""))
        difficulty = str(row.get("difficulty") or extra.get("difficulty") or "")
        scenario = str(row.get("scenario_type") or extra.get("scenario_type") or "")
        key = (domain, difficulty, scenario)
        lst = buckets.get(key)
        if lst is None:
            lst = []
            buckets[key] = lst
        lst.append(idx)

    selected_by_key = {key: 0 for key in buckets}
    selected: list[Hashable] = []
    total = len(df)
    while len(selected) < total:
        eligible = [
            key for key in sorted(buckets)
            if selected_by_key[key] < len(buckets[key])
        ]
        key = min(
            eligible,
            key=lambda item: (
                selected_by_key[item] / len(buckets[item]),
                item,
            ),
        )
        selected.append(buckets[key][selected_by_key[key]])
        selected_by_key[key] += 1
    return df.loc[selected].reset_index(drop=True)

def _stratified_head(df: pd.DataFrame, target: int) -> pd.DataFrame:
    if target <= 0:
        return df.iloc[0:0].copy().reset_index(drop=True)
    ordered = _proportional_stratum_order(df)
    if len(ordered) <= target:
        return ordered
    buckets: dict[tuple[str, str, str], list[Hashable]] = {}
    for idx, row in ordered.iterrows():
        extra = _as_extra(row["extra_info"])
        key = (
            str(extra.get("domain", "")),
            str(row.get("difficulty") or extra.get("difficulty") or ""),
            str(row.get("scenario_type") or extra.get("scenario_type") or ""),
        )
        buckets.setdefault(key, []).append(idx)

    total = sum(len(indices) for indices in buckets.values())
    exact = {
        key: target * len(indices) / total
        for key, indices in buckets.items()
    }
    quotas = {
        key: min(len(buckets[key]), math.floor(value))
        for key, value in exact.items()
    }
    remaining = target - sum(quotas.values())
    remainder_order = sorted(
        buckets,
        key=lambda key: (-(exact[key] - math.floor(exact[key])), key),
    )
    for key in remainder_order:
        if remaining == 0:
            break
        if quotas[key] < len(buckets[key]):
            quotas[key] += 1
            remaining -= 1
    if remaining:
        raise RuntimeError(
            f"proportional stratum selection could not allocate {remaining} rows"
        )

    # Weighted round-robin keeps train/validation prefixes representative.
    selected_by_key = {key: 0 for key in buckets}
    selected: list[Hashable] = []
    while len(selected) < target:
        eligible = [
            key for key in sorted(buckets)
            if selected_by_key[key] < quotas[key]
        ]
        if not eligible:
            raise RuntimeError("proportional stratum selection exhausted early")
        key = min(
            eligible,
            key=lambda item: (
                selected_by_key[item] / quotas[item],
                item,
            ),
        )
        selected.append(buckets[key][selected_by_key[key]])
        selected_by_key[key] += 1
    return ordered.loc[selected].reset_index(drop=True)

def _domain_quotas(target: int, domains: list[str]) -> dict[str, int]:
    base, remainder = divmod(target, len(domains))
    return {
        domain: base + (1 if index < remainder else 0)
        for index, domain in enumerate(domains)
    }

def _capacity_weighted_domain_quotas(
    target: int,
    domains: list[str],
    capacities: dict[str, int],
    *,
    minimum_per_domain: int | None = None,
) -> dict[str, int]:
    """Compatibility wrapper for the shared generation/merge allocator."""
    return capacity_weighted_domain_quotas(
        target,
        domains,
        capacities,
        minimum_per_domain=minimum_per_domain,
    )

def _minimum_domain_coverage(
    target: int,
    domain_count: int,
    configured: int | None,
) -> int:
    if configured is not None:
        return max(0, int(configured))
    return max(1, target // (2 * domain_count)) if target >= domain_count else 0

def _add_weighted_with_caps(
    quotas: dict[str, int],
    maximums: dict[str, int],
    weights: dict[str, int],
    amount: int,
) -> bool:
    """Add units proportionally to frozen weights without exceeding caps."""
    added = {domain: 0 for domain in quotas}
    for _ in range(amount):
        eligible = [
            domain for domain in quotas
            if quotas[domain] < maximums.get(domain, 0)
        ]
        if not eligible:
            return False
        domain = min(
            eligible,
            key=lambda item: (
                (added[item] + 1) / max(1, int(weights.get(item, 0))),
                item,
            ),
        )
        quotas[domain] += 1
        added[domain] += 1
    return True

def _availability_constrained_split_quotas(
    count: int,
    val_count: int,
    domains: list[str],
    capacities: dict[str, int],
    available: dict[str, int],
    train_quotas: dict[str, int],
    val_quotas: dict[str, int],
    *,
    min_domain_train: int | None = None,
    min_domain_val: int | None = None,
) -> tuple[dict[str, int], dict[str, int], bool]:
    """Fit frozen-weight quotas to the eligible pool while preserving floors.

    Capacity weights remain frozen across top-up rounds, but their exact
    apportionment is not a corpus hard gate.  If total eligible supply is
    sufficient, a domain-local shortage is reassigned to domains with spare
    Jaccard-unique rows.  Separate train/validation floors remain mandatory.
    """
    if not domains:
        return train_quotas, val_quotas, False
    train_floor = _minimum_domain_coverage(
        count, len(domains), min_domain_train,
    )
    val_floor = _minimum_domain_coverage(
        val_count, len(domains), min_domain_val,
    )
    combined_floor = train_floor + val_floor
    if any(available.get(domain, 0) < combined_floor for domain in domains):
        return train_quotas, val_quotas, False
    if sum(available.get(domain, 0) for domain in domains) < count + val_count:
        return train_quotas, val_quotas, False

    desired_total = {
        domain: train_quotas[domain] + val_quotas[domain]
        for domain in domains
    }
    total_quotas = {
        domain: min(desired_total[domain], available[domain])
        for domain in domains
    }
    missing_total = count + val_count - sum(total_quotas.values())
    if missing_total and not _add_weighted_with_caps(
        total_quotas, available, capacities, missing_total,
    ):
        return train_quotas, val_quotas, False

    adjusted_val = {
        domain: min(
            max(val_floor, val_quotas[domain]),
            total_quotas[domain] - train_floor,
        )
        for domain in domains
    }
    missing_val = val_count - sum(adjusted_val.values())
    if missing_val < 0:
        return train_quotas, val_quotas, False
    val_maximums = {
        domain: total_quotas[domain] - train_floor
        for domain in domains
    }
    if missing_val and not _add_weighted_with_caps(
        adjusted_val, val_maximums, capacities, missing_val,
    ):
        return train_quotas, val_quotas, False
    adjusted_train = {
        domain: total_quotas[domain] - adjusted_val[domain]
        for domain in domains
    }
    changed = adjusted_train != train_quotas or adjusted_val != val_quotas
    return adjusted_train, adjusted_val, changed

def _sequence_jaccard(a: list[str], b: list[str]) -> float:
    return position_aware_jaccard(a, b)

def _domain_unique_chain_capacity(
    pool: pd.DataFrame,
    domains: list[str],
    threshold: float = 0.70,
) -> dict[str, int]:
    """Count rows retainable by the canonical PROVE dedup signature."""
    capacities: dict[str, int] = {}
    for domain in domains:
        kept: list[list[str]] = []
        domain_rows = pool.loc[
            pool["extra_info"].map(
                lambda value: str(_as_extra(value).get("domain", "")) == domain
            )
        ]
        for _, row in domain_rows.iterrows():
            sequence = [
                str(name) for name in _row_tool_sequence(row, mode="prove")
            ]
            if not sequence:
                continue
            if any(
                _sequence_jaccard(sequence, prior) >= threshold
                for prior in kept
            ):
                continue
            kept.append(sequence)
        capacities[domain] = len(kept)
    return capacities

def _domain_deficit_report(
    pool: pd.DataFrame,
    count: int,
    val_count: int,
    domains: list[str],
    candidate_by_domain: dict[str, int] | None = None,
    train_quotas: dict[str, int] | None = None,
    val_quotas: dict[str, int] | None = None,
    unique_chain_capacity: dict[str, int] | None = None,
) -> dict[str, Any]:
    train_quotas = train_quotas or _domain_quotas(count, domains)
    val_quotas = val_quotas or _domain_quotas(val_count, domains)
    available = {
        domain: int(
            pool["extra_info"].map(
                lambda value: str(_as_extra(value).get("domain", "")) == domain
            ).sum()
        )
        for domain in domains
    }
    required = {
        domain: train_quotas[domain] + val_quotas[domain]
        for domain in domains
    }
    candidates = candidate_by_domain or dict(available)
    retention = {
        domain: (
            available[domain] / candidates[domain]
            if candidates.get(domain, 0) > 0
            else 0.0
        )
        for domain in domains
    }
    deficits = {
        domain: required[domain] - available[domain]
        for domain in domains
        if available[domain] < required[domain]
    }
    return {
        "pool_size": len(pool),
        "required_total": count + val_count,
        "available_by_domain": available,
        "candidate_by_domain": candidates,
        "jaccard_retention_by_domain": retention,
        "unique_chain_capacity_by_domain": unique_chain_capacity or {},
        "train_quota_by_domain": train_quotas,
        "val_quota_by_domain": val_quotas,
        "required_by_domain": required,
        "deficits": deficits,
        "suggested_topup_by_domain": {
            domain: _suggest_topup_count(
                missing=missing,
                available=available[domain],
                candidates=candidates.get(domain, 0),
            )
            for domain, missing in deficits.items()
        },
    }

def _suggest_topup_count(
    *,
    missing: int,
    available: int,
    candidates: int,
) -> int:
    """Estimate candidate top-up from the observed global retention.

    The previous fixed ``missing + 50%`` estimate assumes at least 2/3 of new
    rows survive global Jaccard. Real domains can retain much less, causing
    repeated undersized rounds even though every generated shard is valid. This
    estimate changes only how many candidates are requested; all candidates
    still pass the unchanged replay/provenance/Jaccard gates.
    """
    missing = max(0, int(missing))
    if missing == 0:
        return 0
    observed = available / candidates if candidates > 0 else 0.0
    # Avoid division by zero for a domain with no survivor, while keeping the
    # estimate bounded.  The 20% margin absorbs ordinary sampling variance.
    effective = min(1.0, max(0.10, observed))
    base = math.ceil(missing / effective)
    margin = max(2, math.ceil(base * 0.20))
    return base + margin

def _write_deficit_report(path: Path | None, report: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)

def _frozen_allocation_capacity(
    path: Path | None,
    domains: list[str],
) -> dict[str, int] | None:
    """Reuse the first merge's capacity weights across top-up rounds."""
    if path is None or not path.exists():
        return None
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return None
    raw = report.get("allocation_capacity_by_domain")
    if not isinstance(raw, dict) or set(raw) != set(domains):
        return None
    try:
        return {domain: max(0, int(raw[domain])) for domain in domains}
    except (TypeError, ValueError):
        return None

def _initial_query_key(row: pd.Series) -> str:
    """Return the normalized policy-visible first user query for split isolation."""
    extra = _as_extra(row["extra_info"])
    return " ".join(str(extra.get("user_query", "")).casefold().split())

def _is_local_irrelevance_row(row: pd.Series) -> bool:
    extra = _as_extra(row["extra_info"])
    # Keep external abstention provenance separate.  Only rows created by this
    # repository's local irrelevance Teacher use this diversity constraint.
    return str(extra.get("generation_method", "")) == "irrelevant_teacher_fsm"

def _dedup_local_irrelevance_queries(
    df: pd.DataFrame,
    fixed_mask: pd.Series | None = None,
) -> tuple[pd.DataFrame, int]:
    """Drop exact normalized local-irrelevance query repeats.

    PROVE Jaccard remains defined only on tool-call sequences.  This separate
    local trainability pass prevents identical zero-tool policy inputs from
    surviving across independently generated shards.  Historical fixed rows
    are immutable; they seed the seen set and only incremental duplicates are
    removed.
    """
    if fixed_mask is not None and len(fixed_mask) != len(df):
        raise ValueError("fixed_mask length must match dataframe")
    fixed_flags = (
        [bool(value) for value in fixed_mask.tolist()]
        if fixed_mask is not None
        else [False] * len(df)
    )
    kept: list[Hashable] = []
    seen: set[str] = set()
    removed = 0

    def visit(idx: Hashable, row: pd.Series, *, fixed: bool) -> None:
        nonlocal removed
        if not _is_local_irrelevance_row(row):
            kept.append(idx)
            return
        key = _initial_query_key(row)
        if fixed:
            kept.append(idx)
            if key:
                seen.add(key)
            return
        if key and key in seen:
            removed += 1
            return
        kept.append(idx)
        if key:
            seen.add(key)

    if fixed_mask is not None:
        for position, (idx, row) in enumerate(df.iterrows()):
            if fixed_flags[position]:
                visit(idx, row, fixed=True)
        for position, (idx, row) in enumerate(df.iterrows()):
            if not fixed_flags[position]:
                visit(idx, row, fixed=False)
    else:
        for idx, row in df.iterrows():
            visit(idx, row, fixed=False)
    return df.loc[kept].reset_index(drop=True), removed

def _isolate_initial_queries(
    train: pd.DataFrame,
    val: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """Keep identical initial policy inputs wholly within one split.

    This is an evaluation isolation contract, not a corpus filter: rows are
    swapped between equally sized train/validation partitions and none are
    removed.  Moving the overlapping validation group into train is preferred;
    an equally sized set of complete train-side query groups moves to val.
    """
    train = train.reset_index(drop=True)
    val = val.reset_index(drop=True)
    while True:
        train_keys = train.apply(_initial_query_key, axis=1)
        val_keys = val.apply(_initial_query_key, axis=1)
        overlap = (set(train_keys) & set(val_keys)) - {""}
        if not overlap:
            return train, val

        key = sorted(overlap)[0]
        val_overlap_idx = list(val.index[val_keys == key])
        val_key_set = set(val_keys)
        domains = sorted({
            str(_as_extra(value).get("domain", "")) or "__all__"
            for value in pd.concat([train["extra_info"], val["extra_info"]])
        })

        def domain_counts(frame: pd.DataFrame, indices: list[Hashable]) -> tuple[int, ...]:
            counts = {domain: 0 for domain in domains}
            for idx in indices:
                domain = str(
                    _as_extra(frame.loc[idx, "extra_info"]).get("domain", "")
                ) or "__all__"
                counts[domain] += 1
            return tuple(counts[domain] for domain in domains)

        target_counts = domain_counts(val, val_overlap_idx)
        candidate_groups: list[tuple[str, list[Hashable], tuple[int, ...]]] = []
        for candidate_key in sorted(set(train_keys) - val_key_set - {""}):
            indices = list(train.index[train_keys == candidate_key])
            candidate_groups.append(
                (candidate_key, indices, domain_counts(train, indices))
            )

        # Multi-dimensional subset-sum over complete query groups.  Matching
        # the val-overlap domain vector keeps every per-domain quota unchanged
        # while moving the duplicate query wholly into train.
        zero = tuple(0 for _ in domains)
        choices: dict[tuple[int, ...], list[Hashable]] = {zero: []}
        for _, indices, group_counts in candidate_groups:
            for current, selected in list(choices.items())[::-1]:
                new_counts = tuple(
                    current[i] + group_counts[i] for i in range(len(domains))
                )
                if any(
                    new_counts[i] > target_counts[i]
                    for i in range(len(domains))
                ):
                    continue
                choices.setdefault(new_counts, selected + indices)
        train_to_val_idx = choices.get(target_counts)
        if train_to_val_idx is None:
            return None

        incoming_train = val.loc[val_overlap_idx].copy()
        incoming_val = train.loc[train_to_val_idx].copy()
        train = pd.concat(
            [train.drop(index=train_to_val_idx), incoming_train],
            ignore_index=True,
        )
        val = pd.concat(
            [val.drop(index=val_overlap_idx), incoming_val],
            ignore_index=True,
        )

def _balanced_domain_split(
    pool: pd.DataFrame,
    count: int,
    val_count: int,
    domains: list[str],
    train_quotas: dict[str, int] | None = None,
    val_quotas: dict[str, int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    train_quotas = train_quotas or _domain_quotas(count, domains)
    val_quotas = val_quotas or _domain_quotas(val_count, domains)
    train_parts: list[pd.DataFrame] = []
    val_parts: list[pd.DataFrame] = []
    for domain in domains:
        domain_rows = pool.loc[
            pool["extra_info"].map(
                lambda value: str(_as_extra(value).get("domain", "")) == domain
            )
        ].reset_index(drop=True)
        needed = train_quotas[domain] + val_quotas[domain]
        if len(domain_rows) < needed:
            print(
                f"  FATAL: domain {domain} has {len(domain_rows)} global unique "
                f"candidates, need {needed} "
                f"(train={train_quotas[domain]}, val={val_quotas[domain]})"
            )
            return None
        selected = _stratified_head(domain_rows, needed)
        domain_train = selected.iloc[:train_quotas[domain]].reset_index(drop=True)
        domain_val = selected.iloc[train_quotas[domain]:needed].reset_index(drop=True)
        train_parts.append(domain_train)
        val_parts.append(domain_val)
    train_df = pd.concat(train_parts, ignore_index=True) if train_parts else pool.iloc[:0].copy()
    val_df = pd.concat(val_parts, ignore_index=True) if val_parts else pool.iloc[:0].copy()
    isolated = _isolate_initial_queries(train_df, val_df)
    if isolated is None:
        print(
            "  FATAL: cannot isolate identical initial queries globally "
            "without changing per-domain quotas"
        )
        return None
    return isolated
