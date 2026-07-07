#!/usr/bin/env python3
"""Merge generated rollout shards with quality gates."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

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
    "normal_safe_success": {"final_answer"},
    "missing_function": {"report_error"},
    "no_tool_or_abstention": {"report_error"},
    "irrelevant": {"report_error"},
    "clarification_required": {"ask_clarification"},
    "tool_error_recovery": {"final_answer", "report_error"},
    "missing_dependency": {"final_answer", "ask_clarification", "report_error"},
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
    buckets: dict[tuple[str, str], list[int]] = defaultdict(list)
    for idx, row in df.iterrows():
        extra = _as_extra(row["extra_info"])
        domain = str(extra.get("domain", ""))
        scenario = str(row.get("scenario_type") or extra.get("scenario_type") or "")
        buckets[(domain, scenario)].append(idx)

    ordered_keys = sorted(buckets)
    selected: list[int] = []
    while len(selected) < target and ordered_keys:
        next_keys: list[tuple[str, str]] = []
        for key in ordered_keys:
            if buckets[key] and len(selected) < target:
                selected.append(buckets[key].pop(0))
            if buckets[key]:
                next_keys.append(key)
        ordered_keys = next_keys
    return df.loc[selected].reset_index(drop=True)


def merge_split(tmpdir: Path, pattern: str, outpath: Path, target: int) -> tuple[bool, pd.DataFrame]:
    dfs = [pd.read_parquet(path) for path in sorted(tmpdir.glob(pattern))]
    if not dfs:
        print(f"WARNING: no {pattern} data")
        return False, pd.DataFrame()

    merged = pd.concat(dfs, ignore_index=True)
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
        keep_rows.append(row.to_dict())
    merged = pd.DataFrame(keep_rows)
    dropped_dedup = before_dedup - len(merged)
    if dropped_dedup:
        print(f"  dedup: dropped {dropped_dedup} cross-shard duplicates, {len(merged)} remaining")

    if target > 0 and len(merged) > target:
        merged = _stratified_head(merged, target)
    if target > 0 and len(merged) < target:
        print(f"  FATAL: {outpath} has {len(merged)} rows, below target {target}")
        return False, merged
    outpath.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(outpath, index=False)
    print(f"  {outpath}: {len(merged)} rows (target={target})")
    return True, merged


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tmpdir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--val-count", type=int, required=True)
    args = parser.parse_args()

    tmpdir = Path(args.tmpdir)
    output_dir = Path(args.output_dir)
    ok_train, train_df = merge_split(tmpdir, "shard_*_train.parquet", output_dir / "train.parquet", args.count)
    ok_val, val_df = merge_split(tmpdir, "shard_*_val.parquet", output_dir / "val.parquet", args.val_count)
    if not (ok_train and ok_val):
        return 1

    train_fps = {_row_fingerprint(row) for _, row in train_df.iterrows()}
    val_fps = {_row_fingerprint(row) for _, row in val_df.iterrows()}
    fp_overlap = train_fps & val_fps
    if fp_overlap:
        print(f"  FATAL: {len(fp_overlap)} semantic fingerprint overlaps between train and val")
        return 1

    train_ids = {_as_extra(row["extra_info"]).get("task_id", "") for _, row in train_df.iterrows()}
    val_ids = {_as_extra(row["extra_info"]).get("task_id", "") for _, row in val_df.iterrows()}
    tid_overlap = train_ids & val_ids
    if tid_overlap:
        print(f"  FATAL: {len(tid_overlap)} train/val task_id overlaps")
        return 1

    print(
        f"  merge ok: {len(train_df)} train + {len(val_df)} val, "
        f"fp_overlap=0, tid_overlap=0"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
