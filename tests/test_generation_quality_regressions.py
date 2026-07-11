from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from scripts.generate_data import _filter_training_eligible_tasks
from scripts.merge_rollout_shards import _quality_issue, _row_fingerprint, merge_shards
from src.live_mcp.orchestrator import (
    TaskOrchestrator,
    _detect_missing_dependency,
    _query_requires_hidden_capability,
)
from src.live_mcp.task_planner import TaskPlanner
from src.live_mcp.types import OracleCall


def test_filesystem_natural_language_is_not_rejected_by_tool_name_shape() -> None:
    cases = [
        ("ls", "what's inside /home/user?"),
        ("mkdir", "can you make a new folder inside /home/user?"),
        ("find", "find all .sh files under /home/user"),
        ("touch", "make a new empty file called todo.txt"),
    ]
    for tool_name, query in cases:
        assert _query_requires_hidden_capability(query, tool_name, "filesystem")


def test_round_cannot_advance_after_goal_tool_failed_and_only_read_succeeded() -> None:
    read_only_recovery = [OracleCall("get_event", {"event_id": "evt_1"})]
    completed = [OracleCall("update_event", {"event_id": "evt_1"})]
    abstained = [OracleCall("report_error", {"text": "cannot complete"}, action="report_error")]

    assert not TaskOrchestrator._round_goal_satisfied(
        read_only_recovery, "update_event",
    )
    assert TaskOrchestrator._round_goal_satisfied(completed, "update_event")
    assert TaskOrchestrator._round_goal_satisfied(abstained, "update_event")


def test_query_prompt_requires_complete_chain_final_outcome() -> None:
    class CapturingClient:
        messages = None

        def generate_chat(self, messages, **kwargs):
            self.messages = messages
            return '{"user_query": "copy the file into the new complaints folder"}'

    client = CapturingClient()
    planner = TaskPlanner(client, "filesystem", seed=1)
    query = planner.generate_query(
        tool_schemas=[
            {"name": "mkdir", "description": "create a directory", "inputSchema": {}},
            {"name": "cp", "description": "copy a file", "inputSchema": {}},
        ],
        grounded_state={},
        difficulty="complete",
        rng=__import__("random").Random(1),
        chain_seed=["mkdir", "cp"],
    )
    assert query.startswith("copy the file")
    prompt = client.messages[1]["content"]
    assert "['mkdir', 'cp']" in prompt
    assert "final outcome" in prompt
    assert "copy a file" in prompt


def test_verify_account_is_a_valid_read_predecessor_for_apply_loan() -> None:
    calls = [
        OracleCall("verify_account", {"account_id": "acc_1", "owner_name": "me"}),
        OracleCall(
            "apply_loan",
            {"account_id": "acc_1", "amount": 5000, "term_months": 24},
        ),
    ]
    assert not _detect_missing_dependency(calls, "banking")


def test_find_is_a_valid_read_predecessor_for_remove_file() -> None:
    calls = [
        OracleCall("find", {"path": "/home/user", "pattern": "pipeline.sh"}),
        OracleCall("rm", {"path": "/home/user/pipeline.sh", "recursive": False}),
    ]
    assert not _detect_missing_dependency(calls, "filesystem")


def test_outcome_invalid_task_is_removed_before_split() -> None:
    task = SimpleNamespace(
        task_id="shopping_bad_outcome",
        metadata={"project_outcome_valid": False},
    )
    assert _filter_training_eligible_tasks([task]) == []


def test_merge_quality_gate_handles_non_builtin_false_values() -> None:
    row = pd.Series(_row("q-bad", "t-bad"))
    row["extra_info"]["project_outcome_valid"] = pd.array([False], dtype="boolean")[0]
    assert _quality_issue(row) == "project_outcome_invalid"


def _row(query: str, task_id: str) -> dict:
    oracle = [{
        "tool_name": "get_event",
        "arguments": {"event_id": query},
        "action": "tool_call",
    }, {
        "tool_name": "final_answer",
        "arguments": {"text": "done"},
        "action": "final_answer",
    }]
    return {
        "prompt": json.dumps([
            {"role": "system", "content": "calendar tools"},
            {"role": "user", "content": query},
        ]),
        "data_source": "live_mcp_state_machine",
        "reward_model": {
            "style": "rule",
            "ground_truth": {"oracle_calls": json.dumps(oracle)},
        },
        "extra_info": {
            "domain": "calendar",
            "user_query": query,
            "oracle_calls": json.dumps(oracle),
            "task_id": task_id,
            "hidden_tools": [],
            "visible_tool_names": ["get_event"],
        },
        "uid": task_id,
        "group_id": task_id,
        "perturbation_level": "complete",
        "scenario_type": "normal_safe_success",
    }


def test_merge_backfills_after_fingerprint_and_task_id_overlap(tmp_path: Path) -> None:
    shard_dir = tmp_path / "shards"
    output_dir = tmp_path / "out"
    shard_dir.mkdir()
    pd.DataFrame([
        _row("q1", "t1"),
        _row("q2", "t2"),
        _row("q3", "t3"),
    ]).to_parquet(shard_dir / "shard_0_train.parquet", index=False)
    pd.DataFrame([
        _row("q1", "v-overlap-fingerprint"),
        _row("q4", "t2"),
        _row("q5", "t5"),
        _row("q6", "t6"),
    ]).to_parquet(shard_dir / "shard_0_val.parquet", index=False)

    assert merge_shards(shard_dir, output_dir, count=2, val_count=2) == 0
    train = pd.read_parquet(output_dir / "train.parquet")
    val = pd.read_parquet(output_dir / "val.parquet")
    assert len(train) == 2
    assert len(val) == 2
    assert not (
        {_row_fingerprint(row) for _, row in train.iterrows()}
        & {_row_fingerprint(row) for _, row in val.iterrows()}
    )
    train_ids = {row["extra_info"]["task_id"] for _, row in train.iterrows()}
    val_ids = {row["extra_info"]["task_id"] for _, row in val.iterrows()}
    assert not (train_ids & val_ids)


def test_client_and_recovery_seed_ranges_do_not_overlap() -> None:
    base_seed = 42
    client_stride = 1_000_000
    seeds = {
        base_seed + client * client_stride + recovery_round * 100_000
        for client in range(8)
        for recovery_round in range(3)
    }
    assert len(seeds) == 8 * 3
