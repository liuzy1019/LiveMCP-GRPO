#!/usr/bin/env python3
"""Audit generated Parquet files with the production reward/data contracts."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.live_mcp.artifact.readback import validate_parquet_readback
from src.live_mcp.corpus.merge_validation import _as_extra, _quality_issue
from src.live_mcp.generation.teacher_contracts import (
    _final_answer_requests_user_input,
)
from src.live_mcp.artifact.validation import validate_artifact_contract
from src.utils import normalize_extra_info


def audit_file(path: Path) -> dict[str, object]:
    """Validate every row and return a compact corpus summary."""
    validate_parquet_readback(path)
    frame = pd.read_parquet(path)
    if frame.empty:
        summary = {
            "path": str(path),
            "rows": 0,
            "domains": {},
            "scenarios": {},
            "terminals": {},
            "required_workflow_projection": {},
            "diagnostics": {},
        }
        del frame
        gc.collect()
        return summary
    failures: list[str] = []
    domains: Counter[str] = Counter()
    scenarios: Counter[str] = Counter()
    terminals: Counter[str] = Counter()
    projection: Counter[str] = Counter()
    diagnostics: Counter[str] = Counter()

    for index, row in frame.iterrows():
        issue = _quality_issue(row)
        if issue:
            failures.append(f"row {index}: {issue}")
            continue
        extra = normalize_extra_info(_as_extra(row.get("extra_info")))
        validate_artifact_contract(extra, require_training=False)
        domains[str(extra.get("domain") or "unknown")] += 1
        scenarios[str(extra.get("scenario_type") or "unknown")] += 1
        oracle_calls = json.loads(extra["oracle_calls"])
        terminal = next(
            (
                str(call.get("action"))
                for call in reversed(oracle_calls)
                if str(call.get("action")) != "tool_call"
            ),
            "missing",
        )
        terminals[terminal] += 1
        projection["exact_repeat_dropped"] += int(
            extra.get("projection_exact_repeat_dropped", 0) or 0
        )
        projection["state_transition_noop_dropped"] += int(
            extra.get("projection_state_transition_noop_dropped", 0) or 0
        )
        projection["action_no_net_change_retained"] += int(
            extra.get("projection_action_no_net_change_retained", 0) or 0
        )
        terminal_call = next(
            (
                call for call in reversed(oracle_calls)
                if str(call.get("action")) != "tool_call"
            ),
            {},
        )
        terminal_args = terminal_call.get("arguments") or {}
        terminal_text = str(
            terminal_args.get("text")
            or terminal_args.get("question")
            or ""
        ) if isinstance(terminal_args, dict) else ""
        if terminal == "final_answer" and _final_answer_requests_user_input(
            terminal_text
        ):
            diagnostics["final_answer_requests_user_input"] += 1
        queries = extra.get("conversation_queries", [])
        if isinstance(queries, str):
            queries = json.loads(queries)
        normalized_queries = [
            " ".join(str(query).lower().split()) for query in queries
        ] if isinstance(queries, list) else []
        if len(normalized_queries) != len(set(normalized_queries)):
            diagnostics["repeated_continuation_query_rows"] += 1

    if failures:
        preview = "\n".join(failures[:20])
        raise RuntimeError(
            f"{path}: {len(failures)}/{len(frame)} rows failed quality gates\n"
            f"{preview}"
        )
    return {
        "path": str(path),
        "rows": len(frame),
        "domains": dict(sorted(domains.items())),
        "scenarios": dict(sorted(scenarios.items())),
        "terminals": dict(sorted(terminals.items())),
        "required_workflow_projection": dict(sorted(projection.items())),
        "diagnostics": dict(sorted(diagnostics.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    for path in args.paths:
        print(audit_file(path))


if __name__ == "__main__":
    main()
