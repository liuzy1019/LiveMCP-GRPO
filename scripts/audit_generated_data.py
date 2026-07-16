#!/usr/bin/env python3
"""Audit generated Parquet files with the production reward/data contracts."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.generate_data import _validate_parquet_readback
from scripts.merge_generation_shards import _as_extra, _quality_issue
from src.reward.oval_reward_fn import _build_task_dict
from src.utils import normalize_extra_info


def audit_file(path: Path) -> dict[str, object]:
    """Validate every row and return a compact corpus summary."""
    _validate_parquet_readback(path)
    frame = pd.read_parquet(path)
    failures: list[str] = []
    domains: Counter[str] = Counter()
    scenarios: Counter[str] = Counter()
    terminals: Counter[str] = Counter()

    for index, row in frame.iterrows():
        issue = _quality_issue(row)
        if issue:
            failures.append(f"row {index}: {issue}")
            continue
        extra = normalize_extra_info(_as_extra(row.get("extra_info")))
        _build_task_dict(extra)
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
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    for path in args.paths:
        print(audit_file(path))


if __name__ == "__main__":
    main()
