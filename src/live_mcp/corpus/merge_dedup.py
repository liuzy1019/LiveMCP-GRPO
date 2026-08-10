"""Merge Dedup."""

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

from src.live_mcp.corpus.merge_allocation import (
    _stratified_head,
)

from src.live_mcp.corpus.merge_validation import (
    _as_extra,
    _oracle_calls,
    _row_tool_sequence,
)

CHAIN_BINS = ("1-2", "3-5", "6+")

def _write_semantic_quarantine_report(
    path: Path,
    records: list[dict[str, Any]],
) -> None:
    """Persist semantic findings without conflating diagnostics and gates."""
    counts: dict[str, int] = {}
    for record in records:
        reason = str(record.get("reason_code") or "unknown")
        counts[reason] = counts.get(reason, 0) + 1
    quarantined_rows = sum(
        1 for record in records
        if str(record.get("disposition") or "quarantined") == "quarantined"
    )
    diagnostic_rows = sum(
        1 for record in records
        if str(record.get("disposition") or "") == "diagnostic_only"
    )
    payload = {
        "schema_version": 2,
        "total_findings": len(records),
        "rejected_rows": quarantined_rows,
        "diagnostic_rows": diagnostic_rows,
        "reason_counts": dict(sorted(counts.items())),
        "samples": records,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )

def _dedup_task_ids(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    seen: set[str] = set()
    keep: list[Hashable] = []
    removed = 0
    for idx, row in df.iterrows():
        task_id = str(_as_extra(row["extra_info"]).get("task_id", ""))
        if task_id and task_id in seen:
            removed += 1
            continue
        if task_id:
            seen.add(task_id)
        keep.append(idx)
    return df.loc[keep].reset_index(drop=True), removed

def _load_quarantined_task_ids(path: Path | None) -> set[str]:
    if path is None:
        return set()
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_samples: Any
    if isinstance(payload, list):
        raw_samples = payload
    elif isinstance(payload, dict):
        raw_samples = payload.get("samples", [])
    else:
        raise ValueError(f"invalid quarantine manifest root: {type(payload).__name__}")
    if not isinstance(raw_samples, list):
        raise ValueError("quarantine manifest samples must be a list")
    task_ids: set[str] = set()
    for index, sample in enumerate(raw_samples):
        if isinstance(sample, str):
            task_id = sample.strip()
        elif isinstance(sample, dict):
            disposition = str(
                sample.get("disposition") or "quarantined"
            ).strip()
            if disposition == "diagnostic_only":
                continue
            if disposition != "quarantined":
                raise ValueError(
                    f"quarantine manifest samples[{index}] has invalid "
                    f"disposition: {disposition!r}"
                )
            task_id = str(sample.get("task_id") or "").strip()
        else:
            raise ValueError(
                f"quarantine manifest samples[{index}] must be string or object"
            )
        if not task_id:
            raise ValueError(f"quarantine manifest samples[{index}] has empty task_id")
        if task_id in task_ids:
            raise ValueError(f"duplicate quarantine task_id: {task_id}")
        task_ids.add(task_id)
    return task_ids

def _drop_quarantined_tasks(
    df: pd.DataFrame,
    task_ids: set[str],
) -> tuple[pd.DataFrame, int]:
    if not task_ids or df.empty:
        return df.reset_index(drop=True), 0
    quarantined = df["extra_info"].map(
        lambda value: str(_as_extra(value).get("task_id") or "") in task_ids
    )
    return df.loc[~quarantined].reset_index(drop=True), int(quarantined.sum())

def _row_required_call_count(row: pd.Series) -> int:
    return sum(
        1
        for call in _oracle_calls(_as_extra(row["extra_info"]))
        if str(call.get("action") or "tool_call") == "tool_call"
    )

def _row_chain_bin(row: pd.Series) -> str:
    required_count = _row_required_call_count(row)
    if required_count <= 2:
        return "1-2"
    if required_count <= 5:
        return "3-5"
    return "6+"

def _normalize_chain_bin_quotas(
    raw: dict[str, int] | None,
    *,
    required: int,
) -> dict[str, int] | None:
    if raw is None:
        return None
    if set(raw) != set(CHAIN_BINS):
        raise ValueError(
            f"chain-bin quotas must contain exactly {CHAIN_BINS}"
        )
    quotas = {key: int(raw[key]) for key in CHAIN_BINS}
    if any(value < 0 for value in quotas.values()):
        raise ValueError("chain-bin quotas must be non-negative")
    if sum(quotas.values()) != required:
        raise ValueError(
            f"chain-bin quotas sum to {sum(quotas.values())}, "
            f"expected {required}"
        )
    return quotas

def _select_chain_bin_quotas(
    pool: pd.DataFrame,
    quotas: dict[str, int],
) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    bins = pool.apply(_row_chain_bin, axis=1)
    available = {
        key: int((bins == key).sum())
        for key in CHAIN_BINS
    }
    deficits = {
        key: quotas[key] - available[key]
        for key in CHAIN_BINS
        if available[key] < quotas[key]
    }
    report: dict[str, Any] = {
        "targets": quotas,
        "available": available,
        "deficits": deficits,
    }
    if deficits:
        return None, report
    selected = [
        _stratified_head(
            pool.loc[bins == key].reset_index(drop=True),
            quotas[key],
        )
        for key in CHAIN_BINS
        if quotas[key] > 0
    ]
    result = (
        pd.concat(selected, ignore_index=True)
        if selected
        else pool.iloc[0:0].copy()
    )
    report["selected"] = dict(quotas)
    return result, report

def _row_jaccard(a: pd.Series, b: pd.Series, mode: str = "prove") -> float:
    seq_a, seq_b = _row_tool_sequence(a, mode=mode), _row_tool_sequence(b, mode=mode)
    if not seq_a and not seq_b:
        return 0.0
    if not seq_a or not seq_b:
        return 0.0
    pos_a = set(enumerate(seq_a))
    pos_b = set(enumerate(seq_b))
    return len(pos_a & pos_b) / len(pos_a | pos_b)

def _dedup_jaccard(
    df: pd.DataFrame,
    threshold: float = 0.70,
    mode: str = "prove",
) -> tuple[pd.DataFrame, int]:
    """Jaccard dedup over the selected sequence signature.

    The production/default mode is PROVE's plain tool-call sequence.  The
    enriched local mode exists only for separately labelled diagnostics.
    """
    kept: list[Hashable] = []
    kept_sets: list[frozenset[tuple[int, str]]] = []
    inverted: dict[tuple[int, str], list[int]] = defaultdict(list)
    removed = 0
    for idx, row in df.iterrows():
        sequence_set = frozenset(enumerate(_row_tool_sequence(row, mode=mode)))
        if not sequence_set:
            kept.append(idx)
            kept_sets.append(sequence_set)
            continue
        possible_matches: set[int] = set()
        for token in sequence_set:
            possible_matches.update(inverted.get(token, ()))
        if any(
            len(sequence_set & kept_sets[position])
            / len(sequence_set | kept_sets[position]) >= threshold
            for position in possible_matches
        ):
            removed += 1
            continue
        position = len(kept)
        kept.append(idx)
        kept_sets.append(sequence_set)
        for token in sequence_set:
            inverted[token].append(position)
    return df.loc[kept].reset_index(drop=True), removed

def _dedup_jaccard_with_fixed_rows(
    df: pd.DataFrame,
    fixed_mask: pd.Series,
    threshold: float = 0.70,
    mode: str = "prove",
) -> tuple[pd.DataFrame, int]:
    """Preserve an immutable base while filtering incremental candidates.

    ``mode`` selects the sequence signature (see ``_row_tool_sequence``).
    """
    if len(fixed_mask) != len(df):
        raise ValueError("fixed_mask length must match dataframe")
    fixed_flags = [bool(value) for value in fixed_mask.tolist()]
    kept: list[Hashable] = []
    kept_sets: list[frozenset[tuple[int, str]]] = []
    inverted: dict[tuple[int, str], list[int]] = defaultdict(list)

    def retain(idx: Hashable, sequence_set: frozenset[tuple[int, str]]) -> None:
        position = len(kept)
        kept.append(idx)
        kept_sets.append(sequence_set)
        for token in sequence_set:
            inverted[token].append(position)

    # Existing corpus rows are immutable even if a newer canonical projection
    # reveals duplicates inside that historical artifact.
    for position, (idx, row) in enumerate(df.iterrows()):
        if not fixed_flags[position]:
            continue
        retain(idx, frozenset(enumerate(_row_tool_sequence(row, mode=mode))))

    removed = 0
    for position, (idx, row) in enumerate(df.iterrows()):
        if fixed_flags[position]:
            continue
        sequence_set = frozenset(enumerate(_row_tool_sequence(row, mode=mode)))
        if not sequence_set:
            retain(idx, sequence_set)
            continue
        possible_matches: set[int] = set()
        for token in sequence_set:
            possible_matches.update(inverted.get(token, ()))
        if any(
            len(sequence_set & kept_sets[prior])
            / len(sequence_set | kept_sets[prior]) >= threshold
            for prior in possible_matches
        ):
            removed += 1
            continue
        retain(idx, sequence_set)
    return df.loc[kept].reset_index(drop=True), removed
