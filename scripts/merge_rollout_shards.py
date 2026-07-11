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


_TERMINALS = {"final_answer", "ask_clarification", "report_error"}
_LEAK_MARKERS = (
    "oracle_calls",
    "success_criteria",
    "ground_truth",
    "allowed_terminal_actions",
    "hidden_tools",
)
_EXPECTED_TERMINALS = {
    "normal_safe_success": {"final_answer", "ask_clarification"},
    "missing_function": {"ask_clarification", "report_error"},
    "no_tool_or_abstention": {"report_error"},
    "irrelevant": {"report_error"},
    "clarification_required": {"ask_clarification"},
    "tool_error_recovery": {"final_answer", "report_error"},
    "missing_dependency": {"final_answer", "ask_clarification", "report_error"},
    "unsafe_temptation": {"final_answer"},
}


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


def _terminal_action(calls: list[dict[str, Any]]) -> str:
    terminals = [call.get("action") for call in calls if call.get("action") in _TERMINALS]
    return str(terminals[-1]) if terminals else ""


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
    project_outcome_valid = extra.get("project_outcome_valid")
    if project_outcome_valid is not None and not bool(project_outcome_valid):
        return "project_outcome_invalid"
    scenario = str(row.get("scenario_type") or extra.get("scenario_type") or "")
    calls = _oracle_calls(extra)
    terminal = _terminal_action(calls)
    expected = _EXPECTED_TERMINALS.get(scenario)
    if expected and terminal not in expected:
        return f"terminal_mismatch:{scenario}->{terminal or 'NONE'}"

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

    seen_calls: set[tuple[str, str]] = set()
    for call in calls:
        if call.get("action", "tool_call") != "tool_call":
            continue
        args = json.dumps(call.get("arguments", {}), sort_keys=True, ensure_ascii=False, default=str)
        key = (str(call.get("tool_name", "")), args)
        if key in seen_calls:
            return "duplicate_tool_call"
        seen_calls.add(key)

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


def merge_shards(tmpdir: Path, output_dir: Path, count: int, val_count: int) -> int:
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

    train_candidates, train_tid_removed = _dedup_task_ids(train_candidates)
    val_candidates, val_tid_internal_removed = _dedup_task_ids(val_candidates)
    if train_tid_removed or val_tid_internal_removed:
        print(
            "  task_id dedup: removed "
            f"train={train_tid_removed}, val={val_tid_internal_removed}"
        )

    if len(train_candidates) < count:
        print(f"  FATAL: train candidates={len(train_candidates)}, need {count}")
        return 1
    train_df = _stratified_head(train_candidates, count)
    train_fps = {_row_fingerprint(row) for _, row in train_df.iterrows()}
    train_ids = {
        str(_as_extra(row["extra_info"]).get("task_id", ""))
        for _, row in train_df.iterrows()
    }
    train_ids.discard("")

    fp_overlap_mask = val_candidates.apply(
        lambda row: _row_fingerprint(row) in train_fps, axis=1,
    ) if len(val_candidates) else pd.Series(dtype=bool)
    fp_removed = int(fp_overlap_mask.sum()) if len(val_candidates) else 0
    if fp_removed:
        val_candidates = val_candidates.loc[~fp_overlap_mask].reset_index(drop=True)

    tid_overlap_mask = val_candidates.apply(
        lambda row: str(_as_extra(row["extra_info"]).get("task_id", "")) in train_ids,
        axis=1,
    ) if len(val_candidates) else pd.Series(dtype=bool)
    tid_removed = int(tid_overlap_mask.sum()) if len(val_candidates) else 0
    if tid_removed:
        val_candidates = val_candidates.loc[~tid_overlap_mask].reset_index(drop=True)

    if len(val_candidates) < val_count:
        print(
            f"  FATAL: val candidates={len(val_candidates)} after cross-split dedup, "
            f"need {val_count} (fp_removed={fp_removed}, tid_removed={tid_removed})"
        )
        return 1
    val_df = (
        val_candidates.iloc[0:0].copy()
        if val_count <= 0
        else _stratified_head(val_candidates, val_count)
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)
    print(f"  {train_path}: {len(train_df)} rows (target={count})")
    print(f"  {val_path}: {len(val_df)} rows (target={val_count})")
    print(
        f"  merge ok: {len(train_df)} train + {len(val_df)} val, "
        f"fp_removed={fp_removed}, tid_removed={tid_removed}"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tmpdir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--val-count", type=int, required=True)
    args = parser.parse_args()

    return merge_shards(
        Path(args.tmpdir), Path(args.output_dir), args.count, args.val_count,
    )


if __name__ == "__main__":
    raise SystemExit(main())
