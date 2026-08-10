#!/usr/bin/env python3
"""Safely merge incremental generated Parquet files into the active corpus."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.live_mcp.corpus.audit import audit_file
from src.live_mcp.corpus.merge import (
    _as_extra,
    _dedup_jaccard,
    _dedup_jaccard_with_fixed_rows,
    _dedup_task_ids,
    _initial_query_key,
    _quality_issue,
    _row_fingerprint,
)
from src.utils import normalize_extra_info, sha256_file


def _domain_counts(frame: pd.DataFrame) -> dict[str, int]:
    return dict(sorted(Counter(
        str(normalize_extra_info(value).get("domain") or "unknown")
        for value in frame["extra_info"]
    ).items()))


def _reward_fingerprints(frame: pd.DataFrame) -> list[str]:
    return sorted({
        str(normalize_extra_info(value).get("reward_fingerprint") or "")
        for value in frame["extra_info"]
    })


def _read_incremental(paths: list[Path]) -> tuple[pd.DataFrame, dict[str, int]]:
    parts: list[pd.DataFrame] = []
    rows_by_path: dict[str, int] = {}
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = pd.read_parquet(path)
        rows_by_path[str(path)] = len(frame)
        if frame.empty:
            continue
        frame = frame.copy()
        frame["_merge_source"] = str(path)
        parts.append(frame)
    if not parts:
        raise RuntimeError("incremental inputs contain no rows")
    return pd.concat(parts, ignore_index=True), rows_by_path


def _assert_quality(frame: pd.DataFrame, label: str) -> None:
    issues = frame.apply(_quality_issue, axis=1)
    bad = issues.astype(bool)
    if bad.any():
        counts = issues.loc[bad].value_counts().to_dict()
        raise RuntimeError(
            f"{label}: {int(bad.sum())}/{len(frame)} rows failed quality gates: "
            f"{counts}"
        )


def merge_incremental(
    *,
    base_train_path: Path,
    base_val_path: Path,
    incremental_paths: list[Path],
    output_dir: Path,
    publish: bool,
    quarantine_invalid_incremental: bool = False,
) -> dict[str, Any]:
    """Merge with base priority, audit the result, and optionally publish it."""
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    building_dir = output_dir.with_name(f".{output_dir.name}.building")
    if building_dir.exists():
        raise FileExistsError(f"building directory already exists: {building_dir}")

    base_train = pd.read_parquet(base_train_path).copy()
    base_val = pd.read_parquet(base_val_path).copy()
    incremental, input_rows = _read_incremental(incremental_paths)
    _assert_quality(base_train, "base train")
    _assert_quality(base_val, "base val")
    incremental_issues = incremental.drop(
        columns=["_merge_source"],
    ).apply(_quality_issue, axis=1)
    invalid_incremental = incremental_issues.astype(bool)
    quality_quarantine_counts: dict[str, int] = {}
    if invalid_incremental.any():
        quality_quarantine_counts = {
            str(reason): int(count)
            for reason, count in incremental_issues.loc[
                invalid_incremental
            ].value_counts().items()
        }
        if not quarantine_invalid_incremental:
            raise RuntimeError(
                f"incremental: {int(invalid_incremental.sum())}/"
                f"{len(incremental)} rows failed quality gates: "
                f"{quality_quarantine_counts}"
            )
        incremental = incremental.loc[~invalid_incremental].reset_index(
            drop=True,
        )
        if incremental.empty:
            raise RuntimeError(
                "all incremental rows were removed by quality quarantine"
            )

    base_train["_merge_source"] = "__base_train__"
    base_val["_merge_source"] = "__base_val__"
    pool = pd.concat(
        [base_train, base_val, incremental],
        ignore_index=True,
    )
    raw_total = len(pool)
    pool, task_id_removed = _dedup_task_ids(pool)
    pool["_semantic_fp"] = pool.apply(_row_fingerprint, axis=1)
    before_exact = len(pool)
    pool = pool.drop_duplicates("_semantic_fp", keep="first").drop(
        columns=["_semantic_fp"],
    ).reset_index(drop=True)
    exact_removed = before_exact - len(pool)
    fixed_base_mask = pool["_merge_source"].isin(
        ("__base_train__", "__base_val__")
    )
    pool, jaccard_removed = _dedup_jaccard_with_fixed_rows(
        pool,
        fixed_base_mask,
        threshold=0.70,
    )

    retained_base_train = pool.loc[
        pool["_merge_source"] == "__base_train__"
    ].copy()
    retained_base_val = pool.loc[
        pool["_merge_source"] == "__base_val__"
    ].copy()
    if len(retained_base_train) != len(base_train):
        raise RuntimeError("priority-preserving dedup changed the base train split")
    if len(retained_base_val) != len(base_val):
        raise RuntimeError("priority-preserving dedup changed the base val split")

    incoming = pool.loc[
        ~pool["_merge_source"].isin(("__base_train__", "__base_val__"))
    ].copy()
    val_query_keys = set(
        retained_base_val.apply(_initial_query_key, axis=1)
    ) - {""}
    incoming_query_keys = incoming.apply(_initial_query_key, axis=1)
    query_conflict_mask = incoming_query_keys.isin(val_query_keys)
    query_conflicts_removed = int(query_conflict_mask.sum())
    incoming = incoming.loc[~query_conflict_mask].copy()

    train = pd.concat(
        [retained_base_train, incoming],
        ignore_index=True,
    ).drop(columns=["_merge_source"]).reset_index(drop=True)
    val = retained_base_val.drop(
        columns=["_merge_source"],
    ).reset_index(drop=True)

    if train["uid"].nunique() != len(train):
        raise RuntimeError("final train UID uniqueness failed")
    if val["uid"].nunique() != len(val):
        raise RuntimeError("final val UID uniqueness failed")
    if set(train["uid"]) & set(val["uid"]):
        raise RuntimeError("final train/val UID isolation failed")
    train_query_keys = set(train.apply(_initial_query_key, axis=1)) - {""}
    final_val_query_keys = set(val.apply(_initial_query_key, axis=1)) - {""}
    if train_query_keys & final_val_query_keys:
        raise RuntimeError("final train/val initial-query isolation failed")
    final_pool = pd.concat([train, val], ignore_index=True)
    final_fixed_mask = pd.Series(
        [True] * len(retained_base_train)
        + [False] * len(incoming)
        + [True] * len(retained_base_val),
        dtype=bool,
    )
    _, final_jaccard_removed = _dedup_jaccard_with_fixed_rows(
        final_pool,
        final_fixed_mask,
        threshold=0.70,
    )
    if final_jaccard_removed:
        raise RuntimeError(
            "final incremental Jaccard uniqueness failed: "
            f"{final_jaccard_removed}"
        )
    immutable_base = pd.concat(
        [retained_base_train, retained_base_val],
        ignore_index=True,
    ).drop(columns=["_merge_source"])
    _, inherited_base_jaccard_duplicates = _dedup_jaccard(
        immutable_base,
        threshold=0.70,
    )

    fingerprints = sorted(
        set(_reward_fingerprints(train)) | set(_reward_fingerprints(val))
    )
    if len(fingerprints) != 1 or not fingerprints[0]:
        raise RuntimeError(
            f"final reward fingerprint mismatch: {fingerprints}"
        )

    building_dir.mkdir(parents=True)
    train_path = building_dir / "train.parquet"
    val_path = building_dir / "val.parquet"
    train.to_parquet(train_path, index=False)
    val.to_parquet(val_path, index=False)
    train_audit = audit_file(train_path)
    val_audit = audit_file(val_path)

    retained_by_source = dict(sorted(
        Counter(str(value) for value in incoming["_merge_source"]).items()
    ))
    report: dict[str, Any] = {
        "inputs": {
            "base_train": {"path": str(base_train_path), "rows": len(base_train)},
            "base_val": {"path": str(base_val_path), "rows": len(base_val)},
            "incremental_rows_by_path": input_rows,
        },
        "dedup": {
            "raw_total": raw_total,
            "quality_quarantine_removed": int(invalid_incremental.sum()),
            "quality_quarantine_reasons": quality_quarantine_counts,
            "task_id_removed": task_id_removed,
            "exact_removed": exact_removed,
            "jaccard_removed": jaccard_removed,
            "inherited_base_jaccard_duplicates": (
                inherited_base_jaccard_duplicates
            ),
            "val_query_conflicts_removed": query_conflicts_removed,
        },
        "outputs": {
            "train_rows": len(train),
            "val_rows": len(val),
            "total_rows": len(train) + len(val),
            "net_new_train_rows": len(incoming),
            "retained_incremental_rows_by_path": retained_by_source,
            "train_domains": _domain_counts(train),
            "val_domains": _domain_counts(val),
            "reward_fingerprint": fingerprints[0],
            "train_sha256": sha256_file(train_path),
            "val_sha256": sha256_file(val_path),
        },
        "audit": {"train": train_audit, "val": val_audit},
    }
    merge_report_path = building_dir / "merge_report.json"
    merge_report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    checksum_paths = (train_path, val_path, merge_report_path)
    (building_dir / "SHA256SUMS").write_text(
        "".join(
            f"{sha256_file(path)}  {path.name}\n"
            for path in checksum_paths
        ),
        encoding="utf-8",
    )
    building_dir.rename(output_dir)

    if publish:
        active_dir = ROOT / "data"
        for link, target in (
            (active_dir / "train.parquet", output_dir / "train.parquet"),
            (active_dir / "val.parquet", output_dir / "val.parquet"),
        ):
            relative_target = os.path.relpath(target, start=link.parent)
            temporary_link = link.with_name(f".{link.name}.next")
            if temporary_link.exists() or temporary_link.is_symlink():
                raise FileExistsError(
                    f"temporary publish link already exists: {temporary_link}"
                )
            temporary_link.symlink_to(relative_target)
            temporary_link.replace(link)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-train", type=Path, required=True)
    parser.add_argument("--base-val", type=Path, required=True)
    parser.add_argument(
        "--incremental", type=Path, action="append", required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--publish", action="store_true")
    parser.add_argument(
        "--quarantine-invalid-incremental",
        action="store_true",
        help=(
            "Drop incremental rows that fail deterministic quality gates and "
            "record the reasons; base rows remain strict."
        ),
    )
    args = parser.parse_args()
    report = merge_incremental(
        base_train_path=args.base_train,
        base_val_path=args.base_val,
        incremental_paths=args.incremental,
        output_dir=args.output_dir,
        publish=args.publish,
        quarantine_invalid_incremental=args.quarantine_invalid_incremental,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
