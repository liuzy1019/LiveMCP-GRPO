import json
from pathlib import Path

import pandas as pd

from src.live_mcp.corpus import merge
from src.live_mcp.corpus.merge_dedup import _load_quarantined_task_ids
from src.live_mcp.corpus.merge_allocation import _domain_unique_chain_capacity


def test_merge_split_accepts_mutable_quarantine_recorder(
    tmp_path: Path, monkeypatch,
) -> None:
    pd.DataFrame([{"extra_info": {"task_id": "task-1"}}]).to_parquet(
        tmp_path / "shard_0_train.parquet", index=False,
    )
    monkeypatch.setattr(merge, "evaluate_semantic_quarantine", lambda extra: None)
    monkeypatch.setattr(merge, "_quality_issue", lambda row: "")
    monkeypatch.setattr(merge, "_row_fingerprint", lambda row: "task-1")

    quarantine_records: list[dict] = []
    ok, rows = merge.merge_split(
        tmp_path,
        "shard_*_train.parquet",
        tmp_path / "unused.parquet",
        0,
        write_output=False,
        semantic_quarantine_records=quarantine_records,
    )

    assert ok is True
    assert len(rows) == 1
    assert quarantine_records == []


def test_quarantine_manifest_ignores_diagnostic_findings(tmp_path: Path) -> None:
    path = tmp_path / "semantic_quarantine.json"
    path.write_text(json.dumps({
        "samples": [
            {"task_id": "reject-me", "disposition": "quarantined"},
            {"task_id": "keep-me", "disposition": "diagnostic_only"},
        ],
    }))

    assert _load_quarantined_task_ids(path) == {"reject-me"}


def test_capacity_uses_same_executed_sequence_as_prove_dedup() -> None:
    pool = pd.DataFrame([
        {"extra_info": {
            "domain": "calendar",
            "source_chain_seed": ["list_events", "get_event"],
            "oracle_calls": [
                {"action": "tool_call", "tool_name": "list_events"},
            ],
        }},
        {"extra_info": {
            "domain": "calendar",
            "source_chain_seed": ["search_events", "update_event"],
            "oracle_calls": [
                {"action": "tool_call", "tool_name": "list_events"},
            ],
        }},
    ])

    assert _domain_unique_chain_capacity(pool, ["calendar"]) == {
        "calendar": 1,
    }
