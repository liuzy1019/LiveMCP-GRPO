#!/usr/bin/env python3
"""Merge Teacher-generation shards with corpus and environment quality gates."""

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
from pathlib import Path
from typing import Any, cast

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.live_mcp.protocol.observation import (
    TRAJECTORY_SCHEMA_VERSION,
    OBSERVATION_SCHEMA_VERSION,
    OBSERVATION_PROJECTION_VERSION,
    compute_server_schema_hash,
)
from src.live_mcp.domain_allocation import (
    capacity_weighted_domain_quotas,
    position_aware_jaccard,
)
from src.live_mcp.registry.tool_semantics import (
    is_mutating_tool,
    unresolved_failed_tool_names,
)
from src.live_mcp.corpus.semantic_quarantine import (
    evaluate_semantic_quarantine,
)
from src.live_mcp.corpus.semantic_core import resolve_semantic_gate_profile
from src.live_mcp.domain_contracts.semantic_policies import (
    evaluate_domain_label_issue,
)
from src.live_mcp.corpus.merge_validation import (
    FatalShardIntegrityError,
    _as_extra,
    _as_json_list,
    _oracle_calls,
    _unresolved_failure_issue,
    _nested_strings,
    _action_dates,
    _round_terminal,
    _deterministic_label_issue,
    _current_tools,
    _current_schema_hashes,
    _runtime_observation_budget,
    _row_fingerprint,
    _quality_issue,
    _row_tool_sequence,
)
from src.live_mcp.corpus.merge_allocation import (
    _proportional_stratum_order,
    _stratified_head,
    _domain_quotas,
    _capacity_weighted_domain_quotas,
    _minimum_domain_coverage,
    _add_weighted_with_caps,
    _availability_constrained_split_quotas,
    _sequence_jaccard,
    _domain_unique_chain_capacity,
    _domain_deficit_report,
    _suggest_topup_count,
    _write_deficit_report,
    _frozen_allocation_capacity,
    _initial_query_key,
    _is_local_irrelevance_row,
    _dedup_local_irrelevance_queries,
    _isolate_initial_queries,
    _balanced_domain_split,
)
from src.live_mcp.corpus.merge_dedup import (
    _write_semantic_quarantine_report,
    _dedup_task_ids,
    _load_quarantined_task_ids,
    _drop_quarantined_tasks,
    _row_required_call_count,
    _row_chain_bin,
    _normalize_chain_bin_quotas,
    _select_chain_bin_quotas,
    _row_jaccard,
    _dedup_jaccard,
    _dedup_jaccard_with_fixed_rows,
)
from src.live_mcp.corpus.ledger import CorpusLedger

DOMAINS_ALL = [
    "banking", "calendar", "crm", "email", "filesystem",
    "food_delivery", "issue_tracker", "payments", "shopping", "team_chat",
]
CHAIN_BINS = ("1-2", "3-5", "6+")



_LEAK_MARKERS = (
    "oracle_calls",
    "success_criteria",
    "ground_truth",
    "allowed_terminal_actions",
    "hidden_tools",
)








_WEEKDAY_INDEX = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}
_TARGET_WEEKDAY_RE = re.compile(
    r"\b(?:next|this|on|for|every|by|due)\s+"
    r"(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b"
    r"|\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)"
    r"\s*,",
    re.IGNORECASE,
)
_ANY_WEEKDAY_RE = re.compile(
    r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    re.IGNORECASE,
)
_ISO_DATE_RE = re.compile(r"\b(20\d{2}-\d{2}-\d{2})(?=T|\b)")
_ACTION_DATE_FIELDS = frozenset({
    "date", "due_date", "execute_date", "scheduled_date", "start_time",
})
_QUESTION_VALUE_RE = re.compile(
    r"^(?:what|which|who|where|when|how|can you|could you|would you|"
    r"please (?:provide|tell|confirm|specify))\b|\?$",
    re.IGNORECASE,
)


















































def merge_split(
    tmpdir: Path,
    pattern: str,
    outpath: Path,
    target: int,
    *,
    write_output: bool = True,
    semantic_quarantine_records: list[dict[str, Any]] | None = None,
) -> tuple[bool, pd.DataFrame]:
    dfs = [pd.read_parquet(path) for path in sorted(tmpdir.glob(pattern))]
    if not dfs:
        print(f"WARNING: no {pattern} data")
        return False, pd.DataFrame()

    merged = pd.concat(dfs, ignore_index=True)
    if len(merged) == 0 or len(merged.columns) == 0:
        print(f"  {outpath}: 0 rows (empty parquet files)")
        return True, pd.DataFrame()
    if semantic_quarantine_records is not None:
        for _, row in merged.iterrows():
            extra = _as_extra(row["extra_info"])
            issue = evaluate_semantic_quarantine(extra)
            if issue is None:
                continue
            record = issue.to_dict(
                task_id=str(
                    extra.get("task_id")
                    or row.get("uid")
                    or row.get("group_id")
                    or ""
                ),
                domain=str(extra.get("domain") or ""),
            )
            record["split_pattern"] = pattern
            try:
                semantic_gate_profile = resolve_semantic_gate_profile(extra)
            except ValueError as exc:
                record["semantic_gate_profile"] = ""
                record["disposition"] = "invalid_profile"
                record["profile_error"] = str(exc)
                semantic_quarantine_records.append(record)
                continue
            record["semantic_gate_profile"] = semantic_gate_profile
            record["disposition"] = (
                "quarantined"
                if (
                    semantic_gate_profile == "deterministic_v1"
                    and issue.hard_gate
                )
                else "diagnostic_only"
            )
            semantic_quarantine_records.append(record)

    merged["_quality_issue"] = merged.apply(_quality_issue, axis=1)
    bad_mask = merged["_quality_issue"].astype(bool)
    dropped_quality = int(bad_mask.sum())
    if dropped_quality:
        quality_counts = merged.loc[bad_mask, "_quality_issue"].value_counts().to_dict()
        print(f"  quality: dropped {dropped_quality} rows: {quality_counts}")
        if dropped_quality == len(merged) and len(quality_counts) == 1:
            only_issue = str(next(iter(quality_counts)))
            if only_issue.startswith("environment_metadata_invalid:"):
                raise FatalShardIntegrityError(pattern, len(merged), only_issue)
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


























def merge_shards(
    tmpdir: Path,
    output_dir: Path,
    count: int,
    val_count: int,
    domains: list[str] | None = None,
    deficits_output: Path | None = None,
    min_domain_train: int | None = None,
    min_domain_val: int | None = None,
    quarantine_task_ids: set[str] | None = None,
    base_train_path: Path | None = None,
    base_val_path: Path | None = None,
    chain_bin_quotas: dict[str, int] | None = None,
) -> int:
    """Globally deduplicate candidates before final train/val truncation.

    Optional base splits participate in deduplication with strict priority but
    are not copied into this run's output. This lets an incremental generation
    job measure and fill its net-new deficit before spending more Teacher
    requests.
    """
    if (base_train_path is None) != (base_val_path is None):
        raise ValueError("base_train_path and base_val_path must be provided together")
    chain_bin_quotas = _normalize_chain_bin_quotas(
        chain_bin_quotas,
        required=count + val_count,
    )
    if chain_bin_quotas is not None and (
        val_count != 0 or domains is None or len(domains) != 1
    ):
        raise ValueError(
            "chain-bin quotas require a single-domain train-only run"
        )
    train_path = output_dir / "train.parquet"
    val_path = output_dir / "val.parquet"
    semantic_quarantine_records: list[dict[str, Any]] = []
    try:
        ok_train, train_candidates = merge_split(
            tmpdir,
            "shard_*_train.parquet",
            train_path,
            0,
            write_output=False,
            semantic_quarantine_records=semantic_quarantine_records,
        )
        ok_val, val_candidates = merge_split(
            tmpdir,
            "shard_*_val.parquet",
            val_path,
            0,
            write_output=False,
            semantic_quarantine_records=semantic_quarantine_records,
        )
    except FatalShardIntegrityError as exc:
        _write_semantic_quarantine_report(
            output_dir / "semantic_quarantine_report.json",
            semantic_quarantine_records,
        )
        report = {
            "pool_size": 0,
            "required_total": count + val_count,
            "available_by_domain": {},
            "required_by_domain": {},
            "deficits": {},
            "suggested_topup_by_domain": {},
            "fatal_integrity_errors": {
                exc.issue: exc.row_count,
            },
        }
        _write_deficit_report(deficits_output, report)
        print(f"  FATAL: {exc}")
        return 2
    _write_semantic_quarantine_report(
        output_dir / "semantic_quarantine_report.json",
        semantic_quarantine_records,
    )
    semantic_gate_removed = sum(
        1
        for record in semantic_quarantine_records
        if record.get("disposition") == "quarantined"
    )
    if not ok_train or (val_count > 0 and not ok_val):
        return 1

    pool = pd.concat([train_candidates, val_candidates], ignore_index=True)
    pool, quarantine_removed = _drop_quarantined_tasks(
        pool, quarantine_task_ids or set(),
    )
    candidate_rows_before_base_dedup = len(pool)
    base_rows = 0
    marker = "_incremental_candidate"
    if base_train_path is not None and base_val_path is not None:
        if not base_train_path.is_file() or not base_val_path.is_file():
            raise FileNotFoundError(
                f"base parquet not found: {base_train_path}, {base_val_path}"
            )
        base_train = pd.read_parquet(base_train_path)
        base_val = pd.read_parquet(base_val_path)
        base = pd.concat([base_train, base_val], ignore_index=True)
        if base.empty:
            raise RuntimeError("base corpus is empty")
        base_issues = base.apply(_quality_issue, axis=1)
        if base_issues.astype(bool).any():
            counts = base_issues.loc[base_issues.astype(bool)].value_counts().to_dict()
            raise RuntimeError(f"base corpus failed quality gates: {counts}")
        base_rows = len(base)
        base = base.copy()
        base[marker] = False
        pool = pool.copy()
        pool[marker] = True
        pool = pd.concat([base, pool], ignore_index=True)

    pool, tid_removed = _dedup_task_ids(pool)
    if pool.empty:
        if domains:
            deficit_report = _domain_deficit_report(
                pd.DataFrame({"extra_info": []}),
                count,
                val_count,
                domains,
                candidate_by_domain={domain: 0 for domain in domains},
            )
        else:
            deficit_report = {
                "pool_size": 0,
                "required_total": count + val_count,
                "available_by_domain": {},
                "required_by_domain": {},
                "deficits": (
                    {"__all__": count + val_count}
                    if count + val_count > 0 else {}
                ),
            }
        deficit_report.update({
            "semantic_gate_removed": semantic_gate_removed,
            "quarantine_removed": quarantine_removed,
            "task_id_removed": tid_removed,
            "irrelevance_query_removed": 0,
            "exact_removed": 0,
            "jaccard_removed": 0,
        })
        _write_deficit_report(deficits_output, deficit_report)
        print(f"  FATAL: global unique candidates=0, need {count + val_count}")
        return 1
    if base_rows:
        pool, irrelevance_query_removed = _dedup_local_irrelevance_queries(
            pool,
            ~pool[marker].astype(bool),
        )
    else:
        pool, irrelevance_query_removed = _dedup_local_irrelevance_queries(pool)
    if base_rows:
        retained_base = int((~pool[marker].astype(bool)).sum())
        if retained_base != base_rows:
            raise RuntimeError(
                "base priority failed after local irrelevance query dedup: "
                f"{retained_base}/{base_rows}"
            )
    before_exact = len(pool)
    pool["_semantic_fp"] = pool.apply(_row_fingerprint, axis=1)
    pool = pool.drop_duplicates(subset=["_semantic_fp"], keep="first").drop(
        columns=["_semantic_fp"],
    ).reset_index(drop=True)
    exact_removed = before_exact - len(pool)
    if base_rows:
        retained_base = int((~pool[marker].astype(bool)).sum())
        if retained_base != base_rows:
            raise RuntimeError(
                f"base priority failed after exact dedup: {retained_base}/{base_rows}"
            )
    capacity_pool = (
        pool.loc[pool[marker].astype(bool)].reset_index(drop=True)
        if base_rows else pool
    )
    candidate_by_domain = (
        {
            domain: int(
                capacity_pool["extra_info"].map(
                    lambda value: str(_as_extra(value).get("domain", "")) == domain
                ).sum()
            )
            for domain in domains
        }
        if domains else {}
    )
    observed_chain_capacity = {}
    allocation_capacity = {}
    if domains:
        observed_chain_capacity = _domain_unique_chain_capacity(
            capacity_pool, domains,
        )
        allocation_capacity = _frozen_allocation_capacity(
            deficits_output, domains,
        ) or observed_chain_capacity
    # PROVE hard gate: deduplicate the canonical plain tool-call sequence.
    # The enriched local signature is diagnostic only; hidden source-chain
    # metadata must not affect retention.
    if base_rows:
        fixed_pool = pool.loc[~pool[marker].astype(bool)].copy()
        candidate_pool = _proportional_stratum_order(
            pool.loc[pool[marker].astype(bool)].copy(),
        )
        pool = pd.concat([fixed_pool, candidate_pool], ignore_index=True)
        pool_before_jaccard = pool
        pool, jaccard_removed_prove = _dedup_jaccard_with_fixed_rows(
            pool,
            ~pool[marker].astype(bool),
            threshold=0.70,
            mode="prove",
        )
        _, jaccard_removed_local = _dedup_jaccard_with_fixed_rows(
            pool_before_jaccard,
            ~pool_before_jaccard[marker].astype(bool),
            threshold=0.70,
            mode="local",
        )
    else:
        pool = _proportional_stratum_order(pool)
        pool_before_jaccard = pool
        pool, jaccard_removed_prove = _dedup_jaccard(
            pool, threshold=0.70, mode="prove",
        )
        _, jaccard_removed_local = _dedup_jaccard(
            pool_before_jaccard, threshold=0.70, mode="local",
        )
    jaccard_removed = jaccard_removed_prove
    if base_rows:
        retained_base = int((~pool[marker].astype(bool)).sum())
        if retained_base != base_rows:
            raise RuntimeError(
                f"base priority failed after Jaccard dedup: {retained_base}/{base_rows}"
            )
        pool = pool.loc[pool[marker].astype(bool)].drop(
            columns=[marker],
        ).reset_index(drop=True)

    chain_bin_report = None
    pool_before_chain_selection = len(pool)
    if chain_bin_quotas is not None:
        selected_pool, chain_bin_report = _select_chain_bin_quotas(
            pool,
            chain_bin_quotas,
        )
        if selected_pool is None:
            missing = sum(chain_bin_report["deficits"].values())
            domain = domains[0]
            report = {
                "pool_size": len(pool),
                "required_total": count + val_count,
                "available_by_domain": {domain: len(pool)},
                "required_by_domain": {domain: count + val_count},
                "deficits": {domain: missing},
                "suggested_topup_by_domain": {
                    domain: max(2, math.ceil(missing * 2.0)),
                },
                "chain_bin_selection": chain_bin_report,
                "semantic_gate_removed": semantic_gate_removed,
                "quarantine_removed": quarantine_removed,
                "task_id_removed": tid_removed,
                "irrelevance_query_removed": irrelevance_query_removed,
                "exact_removed": exact_removed,
                "jaccard_removed": jaccard_removed,
                "jaccard_removed_prove": jaccard_removed_prove,
                "jaccard_removed_local_diagnostic": jaccard_removed_local,
                "base_rows": base_rows,
                "candidate_rows_before_base_dedup": (
                    candidate_rows_before_base_dedup
                ),
                "net_new_candidates": len(pool),
            }
            _write_deficit_report(deficits_output, report)
            print(
                "  FATAL: chain-bin quota deficit after global Jaccard: "
                f"{chain_bin_report['deficits']}"
            )
            return 1
        pool = selected_pool

    if domains:
        # A base-aware run supplements an already covered corpus. Reapplying
        # the default per-domain floor to only the incremental rows forces
        # exhausted domains to manufacture duplicates. Explicit CLI floors
        # still override this behaviour.
        effective_min_domain_train = (
            0 if base_rows and min_domain_train is None else min_domain_train
        )
        effective_min_domain_val = (
            0 if base_rows and min_domain_val is None else min_domain_val
        )
        train_quotas = _capacity_weighted_domain_quotas(
            count, domains, allocation_capacity,
            minimum_per_domain=effective_min_domain_train,
        )
        val_quotas = _capacity_weighted_domain_quotas(
            val_count, domains, allocation_capacity,
            minimum_per_domain=effective_min_domain_val,
        )
        frozen_train_quotas = dict(train_quotas)
        frozen_val_quotas = dict(val_quotas)
        available_by_domain = {
            domain: int(
                pool["extra_info"].map(
                    lambda value: str(_as_extra(value).get("domain", "")) == domain
                ).sum()
            )
            for domain in domains
        }
        train_quotas, val_quotas, quota_rebalanced = (
            _availability_constrained_split_quotas(
                count,
                val_count,
                domains,
                allocation_capacity,
                available_by_domain,
                train_quotas,
                val_quotas,
                min_domain_train=effective_min_domain_train,
                min_domain_val=effective_min_domain_val,
            )
        )
        required_by_domain = {
            domain: train_quotas[domain] + val_quotas[domain]
            for domain in domains
        }
        corpus_ledger = CorpusLedger.from_pool(pool, required_by_domain)
        if corpus_ledger.complete:
            pool = corpus_ledger.select(pool)
        deficit_report = _domain_deficit_report(
            pool, count, val_count, domains,
            candidate_by_domain=candidate_by_domain,
            train_quotas=train_quotas,
            val_quotas=val_quotas,
            unique_chain_capacity=observed_chain_capacity,
        )
        deficit_report["allocation_capacity_by_domain"] = allocation_capacity
        deficit_report["frozen_train_quota_by_domain"] = frozen_train_quotas
        deficit_report["frozen_val_quota_by_domain"] = frozen_val_quotas
        deficit_report["quota_rebalanced"] = quota_rebalanced
        deficit_report.update(corpus_ledger.report())
        for domain, difficulty_deficits in corpus_ledger.deficits.items():
            difficulty_missing = sum(difficulty_deficits.values())
            if not difficulty_missing:
                continue
            deficit_report["deficits"][domain] = max(
                int(deficit_report["deficits"].get(domain, 0)),
                difficulty_missing,
            )
            deficit_report["suggested_topup_by_domain"][domain] = max(
                int(
                    deficit_report["suggested_topup_by_domain"].get(
                        domain, 0,
                    )
                ),
                difficulty_missing + max(2, difficulty_missing),
            )
    else:
        required = count + val_count
        deficit_report = {
            "pool_size": len(pool),
            "required_total": required,
            "available_by_domain": {},
            "required_by_domain": {},
            "deficits": {"__all__": required - len(pool)} if len(pool) < required else {},
        }
    deficit_report.update({
        "semantic_gate_removed": semantic_gate_removed,
        "semantic_gate_removed_is_PROVE": False,  # LOCAL contract (OVAL-MCP.md §5)
        "jaccard_removed": jaccard_removed_prove,
        "jaccard_removed_is_PROVE": True,          # PROVE hard gate (Jaccard 0.70)
        "jaccard_removed_prove": jaccard_removed_prove,
        "jaccard_removed_local_diagnostic": jaccard_removed_local,
        "quarantine_removed": quarantine_removed,
        "task_id_removed": tid_removed,
        "irrelevance_query_removed": irrelevance_query_removed,
        "exact_removed": exact_removed,
        "base_rows": base_rows,
        "candidate_rows_before_base_dedup": candidate_rows_before_base_dedup,
        "net_new_candidates": len(pool),
        "pool_before_chain_selection": pool_before_chain_selection,
        "chain_bin_selection": chain_bin_report,
    })
    deficit_report.setdefault("retained_tool_sequences_by_domain", {})
    _write_deficit_report(deficits_output, deficit_report)

    required = count + val_count
    if len(pool) < required or deficit_report.get("deficits"):
        print(
            f"  FATAL: global unique candidates={len(pool)}, need {required} "
            f"(LOCAL semantic_gate_removed={semantic_gate_removed}, "
            f"PROVE jaccard_removed={jaccard_removed}, "
            f"quarantine_removed={quarantine_removed}, "
            f"task_id_removed={tid_removed}, "
            f"irrelevance_query_removed={irrelevance_query_removed}, "
            f"exact_removed={exact_removed})"
        )
        return 1

    if domains:
        split = _balanced_domain_split(
            pool, count, val_count, domains,
            train_quotas=train_quotas, val_quotas=val_quotas,
        )
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
        f"LOCAL_semantic_gate_removed={semantic_gate_removed}, "
        f"PROVE_jaccard_removed={jaccard_removed}, "
        f"total candidates in pool={len(pool)}"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tmpdir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--val-count", type=int, required=True)
    parser.add_argument("--domain", default="all")
    parser.add_argument("--deficits-output")
    parser.add_argument("--chain-bin-quotas")
    parser.add_argument("--min-domain-train", type=int)
    parser.add_argument("--min-domain-val", type=int)
    parser.add_argument("--quarantine-task-ids", type=Path)
    parser.add_argument("--base-train", type=Path)
    parser.add_argument("--base-val", type=Path)
    args = parser.parse_args()

    domains = (
        list(DOMAINS_ALL)
        if args.domain == "all"
        else [item.strip() for item in args.domain.split(",") if item.strip()]
    )
    unknown = sorted(set(domains) - set(DOMAINS_ALL))
    if not domains or unknown:
        parser.error(f"invalid --domain value: {args.domain!r}; unknown={unknown}")
    if len(domains) != len(set(domains)):
        parser.error(f"duplicate --domain entries are not allowed: {domains}")
    chain_bin_quotas = None
    if args.chain_bin_quotas:
        try:
            raw_chain_bin_quotas = json.loads(args.chain_bin_quotas)
        except json.JSONDecodeError as exc:
            parser.error(f"invalid --chain-bin-quotas JSON: {exc}")
        if not isinstance(raw_chain_bin_quotas, dict):
            parser.error("--chain-bin-quotas must be a JSON object")
        chain_bin_quotas = raw_chain_bin_quotas
    return merge_shards(
        Path(args.tmpdir), Path(args.output_dir), args.count, args.val_count,
        domains=domains,
        deficits_output=Path(args.deficits_output) if args.deficits_output else None,
        min_domain_train=args.min_domain_train,
        min_domain_val=args.min_domain_val,
        quarantine_task_ids=_load_quarantined_task_ids(args.quarantine_task_ids),
        base_train_path=args.base_train,
        base_val_path=args.base_val,
        chain_bin_quotas=chain_bin_quotas,
    )


if __name__ == "__main__":
    raise SystemExit(main())
