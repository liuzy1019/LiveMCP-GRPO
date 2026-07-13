#!/usr/bin/env python3
"""Merge generated rollout shards with quality gates."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Hashable
from pathlib import Path
from typing import Any, cast

import pandas as pd

DOMAINS_ALL = [
    "banking", "calendar", "crm", "email", "filesystem",
    "food_delivery", "issue_tracker", "payments", "shopping", "team_chat",
]

_LEAK_MARKERS = (
    "oracle_calls",
    "success_criteria",
    "ground_truth",
    "allowed_terminal_actions",
    "hidden_tools",
)
def _as_extra(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        return json.loads(value)
    return {}


def _as_json_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if hasattr(value, "tolist"):
        return list(value.tolist())
    if isinstance(value, str):
        return json.loads(value)
    return list(value) if isinstance(value, tuple) else []


def _oracle_calls(extra: dict[str, Any]) -> list[dict[str, Any]]:
    calls = _as_json_list(extra.get("oracle_calls", []))
    return [call for call in calls if isinstance(call, dict)]


def _row_fingerprint(row: pd.Series) -> str:
    extra = _as_extra(row["extra_info"])
    domain = extra.get("domain", "")
    query = " ".join((extra.get("user_query", "") or "").lower().split())
    calls = _oracle_calls(extra)
    sig = json.dumps(
        {"d": domain, "q": query, "c": calls},
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    )
    return hashlib.sha256(sig.encode()).hexdigest()


def _quality_issue(row: pd.Series) -> str:
    extra = _as_extra(row["extra_info"])
    paper_replay_valid = extra.get("paper_replay_valid")
    if paper_replay_valid is not None and not bool(paper_replay_valid):
        return "paper_replay_invalid"
    scenario = str(row.get("scenario_type") or extra.get("scenario_type") or "")
    calls = _oracle_calls(extra)
    # PROVE filters completed traces by replay error rate, provenance, and
    # sequence similarity. Scenario labels are metadata, not a rejection gate;
    # recovery may legitimately end in final_answer or graceful report_error.

    hidden = set(_as_json_list(extra.get("hidden_tools", [])))
    visible = set(_as_json_list(extra.get("visible_tool_names", [])))
    overlap = hidden & visible
    if overlap:
        return f"hidden_tool_visible:{sorted(overlap)}"

    try:
        prompt = json.loads(row["prompt"])
    except Exception:
        return "prompt_json_invalid"
    prompt_text = "\n".join(str(message.get("content", "")) for message in prompt)
    if any(marker in prompt_text for marker in _LEAK_MARKERS):
        return "prompt_leaks_training_target"

    return ""


def _stratified_head(df: pd.DataFrame, target: int) -> pd.DataFrame:
    if target <= 0 or len(df) <= target:
        return df.reset_index(drop=True)
    buckets: dict[tuple[str, str], list[Hashable]] = {}
    for idx, row in df.iterrows():
        extra = _as_extra(row["extra_info"])
        domain = str(extra.get("domain", ""))
        scenario = str(row.get("scenario_type") or extra.get("scenario_type") or "")
        key = (domain, scenario)
        lst = buckets.get(key)
        if lst is None:
            lst = []
            buckets[key] = lst
        lst.append(idx)

    ordered_keys = sorted(buckets)
    selected: list[Hashable] = []
    while len(selected) < target and ordered_keys:
        next_keys: list[tuple[str, str]] = []
        for key in ordered_keys:
            if buckets[key] and len(selected) < target:
                selected.append(buckets[key].pop(0))
            if buckets[key]:
                next_keys.append(key)
        ordered_keys = next_keys
    return df.loc[selected].reset_index(drop=True)


def _domain_quotas(target: int, domains: list[str]) -> dict[str, int]:
    base, remainder = divmod(target, len(domains))
    return {
        domain: base + (1 if index < remainder else 0)
        for index, domain in enumerate(domains)
    }


def _balanced_domain_split(
    pool: pd.DataFrame,
    count: int,
    val_count: int,
    domains: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    train_quotas = _domain_quotas(count, domains)
    val_quotas = _domain_quotas(val_count, domains)
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
        train_parts.append(selected.iloc[:train_quotas[domain]])
        val_parts.append(selected.iloc[train_quotas[domain]:needed])
    train_df = pd.concat(train_parts, ignore_index=True) if train_parts else pool.iloc[:0].copy()
    val_df = pd.concat(val_parts, ignore_index=True) if val_parts else pool.iloc[:0].copy()
    return train_df, val_df


def merge_split(
    tmpdir: Path,
    pattern: str,
    outpath: Path,
    target: int,
    *,
    write_output: bool = True,
) -> tuple[bool, pd.DataFrame]:
    dfs = [pd.read_parquet(path) for path in sorted(tmpdir.glob(pattern))]
    if not dfs:
        print(f"WARNING: no {pattern} data")
        return False, pd.DataFrame()

    merged = pd.concat(dfs, ignore_index=True)
    if len(merged) == 0 or len(merged.columns) == 0:
        print(f"  {outpath}: 0 rows (empty parquet files)")
        return True, pd.DataFrame()
    merged["_quality_issue"] = merged.apply(_quality_issue, axis=1)
    bad_mask = merged["_quality_issue"].astype(bool)
    dropped_quality = int(bad_mask.sum())
    if dropped_quality:
        quality_counts = merged.loc[bad_mask, "_quality_issue"].value_counts().to_dict()
        print(f"  quality: dropped {dropped_quality} rows: {quality_counts}")
    merged = merged.loc[~bad_mask].drop(columns=["_quality_issue"]).reset_index(drop=True)

    before_dedup = len(merged)
    seen: set[str] = set()
    keep_rows: list[dict[str, Any]] = []
    for _, row in merged.iterrows():
        fingerprint = _row_fingerprint(row)
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        keep_rows.append(cast(dict[str, Any], row.to_dict()))
    merged = pd.DataFrame(keep_rows)
    dropped_dedup = before_dedup - len(merged)
    if dropped_dedup:
        print(f"  dedup: dropped {dropped_dedup} cross-shard duplicates, {len(merged)} remaining")

    if target > 0 and len(merged) > target:
        merged = _stratified_head(merged, target)
    if target > 0 and len(merged) < target:
        print(f"  FATAL: {outpath} has {len(merged)} rows, below target {target}")
        return False, merged
    if write_output:
        outpath.parent.mkdir(parents=True, exist_ok=True)
        merged.to_parquet(outpath, index=False)
        print(f"  {outpath}: {len(merged)} rows (target={target})")
    return True, merged


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


def _row_tool_sequence(row: pd.Series) -> list[str]:
    return [
        str(call.get("tool_name", ""))
        for call in _oracle_calls(_as_extra(row["extra_info"]))
        if call.get("action", "tool_call") == "tool_call"
    ]


def _row_jaccard(a: pd.Series, b: pd.Series) -> float:
    seq_a, seq_b = _row_tool_sequence(a), _row_tool_sequence(b)
    if not seq_a and not seq_b:
        return 0.0
    if not seq_a or not seq_b:
        return 0.0
    pos_a = set(enumerate(seq_a))
    pos_b = set(enumerate(seq_b))
    return len(pos_a & pos_b) / len(pos_a | pos_b)


def _dedup_jaccard(df: pd.DataFrame, threshold: float = 0.70) -> tuple[pd.DataFrame, int]:
    kept: list[Hashable] = []
    removed = 0
    for idx, row in df.iterrows():
        if any(_row_jaccard(row, df.loc[prior]) >= threshold for prior in kept):
            removed += 1
            continue
        kept.append(idx)
    return df.loc[kept].reset_index(drop=True), removed


def merge_shards(
    tmpdir: Path,
    output_dir: Path,
    count: int,
    val_count: int,
    domains: list[str] | None = None,
) -> int:
    """Globally deduplicate candidates before final train/val truncation."""
    train_path = output_dir / "train.parquet"
    val_path = output_dir / "val.parquet"
    ok_train, train_candidates = merge_split(
        tmpdir, "shard_*_train.parquet", train_path, 0, write_output=False,
    )
    ok_val, val_candidates = merge_split(
        tmpdir, "shard_*_val.parquet", val_path, 0, write_output=False,
    )
    if not ok_train or (val_count > 0 and not ok_val):
        return 1

    pool = pd.concat([train_candidates, val_candidates], ignore_index=True)
    pool, tid_removed = _dedup_task_ids(pool)
    before_exact = len(pool)
    pool["_semantic_fp"] = pool.apply(_row_fingerprint, axis=1)
    pool = pool.drop_duplicates(subset=["_semantic_fp"], keep="first").drop(
        columns=["_semantic_fp"],
    ).reset_index(drop=True)
    exact_removed = before_exact - len(pool)
    pool, jaccard_removed = _dedup_jaccard(pool, threshold=0.70)

    required = count + val_count
    if len(pool) < required:
        print(
            f"  FATAL: global unique candidates={len(pool)}, need {required} "
            f"(task_id_removed={tid_removed}, exact_removed={exact_removed}, "
            f"jaccard_removed={jaccard_removed})"
        )
        return 1

    if domains:
        split = _balanced_domain_split(pool, count, val_count, domains)
        if split is None:
            return 1
        train_df, val_df = split
    else:
        selected = _stratified_head(pool, required)
        train_df = selected.iloc[:count].reset_index(drop=True)
        val_df = selected.iloc[count:count + val_count].reset_index(drop=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)
    print(f"  {train_path}: {len(train_df)} rows (target={count})")
    print(f"  {val_path}: {len(val_df)} rows (target={val_count})")
    print(
        f"  merge ok: {len(train_df)} train + {len(val_df)} val, "
        f"task_id_removed={tid_removed}, exact_removed={exact_removed}, "
        f"jaccard_removed={jaccard_removed}"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tmpdir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--val-count", type=int, required=True)
    parser.add_argument("--domain", default="all")
    args = parser.parse_args()

    domains = (
        list(DOMAINS_ALL)
        if args.domain == "all"
        else [item.strip() for item in args.domain.split(",") if item.strip()]
    )
    unknown = sorted(set(domains) - set(DOMAINS_ALL))
    if not domains or unknown:
        parser.error(f"invalid --domain value: {args.domain!r}; unknown={unknown}")
    return merge_shards(
        Path(args.tmpdir), Path(args.output_dir), args.count, args.val_count,
        domains=domains,
    )


if __name__ == "__main__":
    raise SystemExit(main())
