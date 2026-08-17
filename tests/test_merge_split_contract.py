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


def test_fixed_attempt_merge_writes_partial_nontraining_artifact(
    tmp_path: Path, monkeypatch,
) -> None:
    candidate_dir = tmp_path / "candidates"
    output_dir = tmp_path / "run"
    candidate_dir.mkdir()
    rows = [
        {
            "uid": f"task-{index}",
            "extra_info": {
                "task_id": f"task-{index}",
                "domain": "banking",
                "difficulty": "complete",
                "generation_method": "task_planner",
                "fixed_attempt_budget": True,
                "artifact_purpose": "experiment",
                "oracle_calls": [],
            },
        }
        for index in range(2)
    ]
    pd.DataFrame(rows[:1]).to_parquet(
        candidate_dir / "shard_0_train.parquet", index=False,
    )
    pd.DataFrame(rows[1:]).to_parquet(
        candidate_dir / "shard_0_val.parquet", index=False,
    )
    monkeypatch.setattr(merge, "evaluate_semantic_quarantine", lambda extra: None)
    monkeypatch.setattr(merge, "_quality_issue", lambda row: "")
    monkeypatch.setattr(merge, "_row_fingerprint", lambda row: row["uid"])
    monkeypatch.setattr(
        merge,
        "_dedup_local_irrelevance_queries",
        lambda frame, *args: (frame, 0),
    )
    monkeypatch.setattr(
        merge, "_dedup_jaccard", lambda frame, **kwargs: (frame, 0),
    )
    report_path = tmp_path / "deficits.json"

    result = merge.merge_shards(
        candidate_dir,
        output_dir,
        count=32,
        val_count=16,
        domains=None,
        deficits_output=report_path,
        diagnostic_fixed_attempt=True,
    )

    assert result == 0
    accepted = pd.read_parquet(output_dir / "accepted.parquet")
    assert list(accepted["uid"]) == ["task-0", "task-1"]
    assert not (output_dir / "train.parquet").exists()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["completion_kind"] == "fixed_attempt_diagnostic"
    assert report["requested_candidate_attempts"] == 48
    assert report["candidate_rows_input"] == 2
    assert report["accepted_rows_after_all_filters"] == 2
    assert report["training_publishable"] is False


def test_production_merge_still_rejects_the_same_partial_pool(
    tmp_path: Path, monkeypatch,
) -> None:
    candidate_dir = tmp_path / "candidates"
    output_dir = tmp_path / "run"
    candidate_dir.mkdir()
    row = {
        "uid": "task-0",
        "extra_info": {
            "task_id": "task-0",
            "domain": "banking",
            "difficulty": "complete",
            "generation_method": "task_planner",
            "fixed_attempt_budget": False,
            "artifact_purpose": "training_candidate",
            "oracle_calls": [],
        },
    }
    pd.DataFrame([row]).to_parquet(
        candidate_dir / "shard_0_train.parquet", index=False,
    )
    pd.DataFrame([row]).iloc[:0].to_parquet(
        candidate_dir / "shard_0_val.parquet", index=False,
    )
    monkeypatch.setattr(merge, "evaluate_semantic_quarantine", lambda extra: None)
    monkeypatch.setattr(merge, "_quality_issue", lambda current: "")
    monkeypatch.setattr(merge, "_row_fingerprint", lambda current: current["uid"])
    monkeypatch.setattr(
        merge,
        "_dedup_local_irrelevance_queries",
        lambda frame, *args: (frame, 0),
    )
    monkeypatch.setattr(
        merge, "_dedup_jaccard", lambda frame, **kwargs: (frame, 0),
    )

    result = merge.merge_shards(
        candidate_dir,
        output_dir,
        count=32,
        val_count=16,
        domains=None,
        diagnostic_fixed_attempt=False,
    )

    assert result == 1
    assert not (output_dir / "accepted.parquet").exists()
    assert not (output_dir / "train.parquet").exists()


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
