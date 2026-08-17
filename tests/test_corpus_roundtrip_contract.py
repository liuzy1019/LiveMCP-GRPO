from __future__ import annotations

import json

import pandas as pd

import pytest

from src.live_mcp.artifact.reward_task import (
    ArtifactIntegrityError,
    build_reward_task,
    validate_ground_truth_consistency,
)
from src.live_mcp.corpus.failure_records import GenerationFailureWriter
from src.live_mcp.corpus.shard import (
    _filter_semantic_eligible_tasks,
    build_arg_parser,
)
from src.live_mcp.corpus.merge_validation import _quality_issue
from src.live_mcp.corpus.shard_recovery import _checkpoint_config
from src.live_mcp.corpus.shard_row_projection import _tasks_to_rows
from src.live_mcp.artifact.readback import validate_parquet_readback
from src.live_mcp.types import LiveTask, OracleCall, OracleProgram
from src.utils import normalize_extra_info


def _no_tool_task(*, fixed_attempt_budget: bool) -> LiveTask:
    terminal = OracleCall(
        tool_name="",
        arguments={},
        action="report_error",
    )
    return LiveTask(
        task_id="banking_contract_roundtrip",
        source="contract_test",
        suite_name="ten_domain",
        user_prompt="Write a poem about the ocean.",
        session_id="",
        session_seed=7,
        target_servers=["banking"],
        visible_tools=[{
            "name": "list_accounts",
            "description": "List accounts",
            "inputSchema": {"type": "object", "properties": {}},
            "_server_name": "banking",
        }],
        required_tools=[],
        expected_outcome={"abstain": True},
        success_criteria=[],
        oracle_program=OracleProgram(
            task_id="banking_contract_roundtrip",
            calls=[terminal],
            success_criteria=[],
        ),
        sampling_context={},
        max_turns=4,
        difficulty="minimal",
        task_type="irrelevant",
        conversation_queries=["Write a poem about the ocean."],
        oracle_calls_per_round=[[terminal]],
        execution_history_per_round=[[]],
        metadata={
            "teacher_model_id": "teacher-contract-test",
            "prompt_profile": "local_trainable_v1",
            "semantic_gate_profile": "deterministic_v1",
            "fixed_attempt_budget": fixed_attempt_budget,
            "scenario_type": "no_tool_or_abstention",
            "generation_method": "contract_test",
            "paper_replay_valid": True,
            "provenance_valid": True,
        },
    )


def test_row_projection_uses_frozen_task_contract_not_current_environment(
    monkeypatch,
) -> None:
    monkeypatch.setenv("LIVEMCP_FIXED_ATTEMPT_BUDGET", "1")
    training_row = _tasks_to_rows(
        [_no_tool_task(fixed_attempt_budget=False)], 7,
    )[0]
    assert training_row["extra_info"]["fixed_attempt_budget"] is False
    assert training_row["extra_info"]["artifact_purpose"] == "training_candidate"

    monkeypatch.setenv("LIVEMCP_FIXED_ATTEMPT_BUDGET", "0")
    experiment_row = _tasks_to_rows(
        [_no_tool_task(fixed_attempt_budget=True)], 7,
    )[0]
    assert experiment_row["extra_info"]["fixed_attempt_budget"] is True
    assert experiment_row["extra_info"]["artifact_purpose"] == "experiment"


def test_row_projection_preserves_continuation_lineage_diagnostics() -> None:
    task = _no_tool_task(fixed_attempt_budget=True)
    task.metadata["continuation_goal_specs"] = [{
        "round_idx": 1,
        "verification": "diagnostic_unproven",
    }]

    row = _tasks_to_rows([task], 7)[0]

    assert json.loads(
        row["extra_info"]["continuation_goal_specs"]
    )[0]["round_idx"] == 1


def test_shard_quarantines_hard_semantic_issue_before_write(tmp_path) -> None:
    task = _no_tool_task(fixed_attempt_budget=True)
    task.oracle_program.calls[-1].arguments = {
        "text": "Use the list_accounts tool to inspect your accounts."
    }
    task.oracle_calls_per_round[-1][-1].arguments = dict(
        task.oracle_program.calls[-1].arguments
    )
    task.metadata["teacher_round_trace"] = [{
        "round_idx": 0,
        "user_query": task.user_prompt,
        "execution_history": [],
        "oracle_calls": [{
            "action": "report_error",
            "tool_name": "report_error",
            "arguments": dict(task.oracle_program.calls[-1].arguments),
        }],
    }]
    writer = GenerationFailureWriter(tmp_path / "failures.jsonl")

    accepted = _filter_semantic_eligible_tasks(
        [task], failure_writer=writer, recovery_round=0,
    )

    assert accepted == []
    record = json.loads(writer.path.read_text(encoding="utf-8"))
    assert record["stage"] == "semantic_quarantine"
    assert record["reason_code"] == "terminal_exposes_private_tool_name"


def test_shard_retains_diagnostic_only_semantic_finding(tmp_path) -> None:
    task = _no_tool_task(fixed_attempt_budget=True)
    task.metadata["semantic_gate_profile"] = "diagnostic_only"
    task.oracle_program.calls[-1].arguments = {
        "text": "Use the list_accounts tool to inspect your accounts."
    }
    task.oracle_calls_per_round[-1][-1].arguments = dict(
        task.oracle_program.calls[-1].arguments
    )
    task.metadata["teacher_round_trace"] = [{
        "round_idx": 0,
        "user_query": task.user_prompt,
        "execution_history": [],
        "oracle_calls": [{
            "action": "report_error",
            "tool_name": "report_error",
            "arguments": dict(task.oracle_program.calls[-1].arguments),
        }],
    }]
    writer = GenerationFailureWriter(tmp_path / "failures.jsonl")

    accepted = _filter_semantic_eligible_tasks(
        [task], failure_writer=writer, recovery_round=0,
    )

    assert accepted == [task]
    assert not writer.path.exists()


def test_shard_quarantines_candidate_projection_error(
    tmp_path, monkeypatch,
) -> None:
    task = _no_tool_task(fixed_attempt_budget=False)
    writer = GenerationFailureWriter(tmp_path / "failures.jsonl")

    def invalid_projection(_tasks, _seed):
        raise ValueError("sampled dependency chain does not align")

    monkeypatch.setattr(
        "src.live_mcp.corpus.shard._tasks_to_rows", invalid_projection,
    )

    accepted = _filter_semantic_eligible_tasks(
        [task], failure_writer=writer, recovery_round=2,
    )

    assert accepted == []
    record = json.loads(writer.path.read_text(encoding="utf-8"))
    assert record["stage"] == "training_contract"
    assert record["reason_code"] == "training_contract_invalid"
    assert record["recovery_round"] == 2


def test_canonical_task_survives_parquet_reward_readback(tmp_path) -> None:
    row = _tasks_to_rows(
        [_no_tool_task(fixed_attempt_budget=False)], 7,
    )[0]
    path = tmp_path / "contract.parquet"
    pd.DataFrame([row]).to_parquet(path, index=False)

    reloaded = pd.read_parquet(path).iloc[0]
    extra = normalize_extra_info(reloaded["extra_info"])
    validate_ground_truth_consistency(
        extra, reloaded["reward_model"]["ground_truth"],
    )
    reward_task = build_reward_task(extra)

    assert extra["artifact_purpose"] == "training_candidate"
    assert "initial_goal_spec" not in extra
    assert "task_goal_contract" not in extra
    assert "task_fact_contract" not in extra
    assert json.loads(extra["oracle_calls"])[-1]["action"] == "report_error"
    assert reward_task["required_tool_calls"] == []
    assert reward_task["allowed_terminal_actions"] == ["report_error"]


def test_ground_truth_mirror_cannot_diverge_from_reward_contract() -> None:
    row = _tasks_to_rows(
        [_no_tool_task(fixed_attempt_budget=False)], 7,
    )[0]
    row["reward_model"]["ground_truth"]["required_tools"] = [
        "list_accounts"
    ]

    with pytest.raises(
        ArtifactIntegrityError, match="ground_truth mismatch for required_tools",
    ):
        validate_ground_truth_consistency(
            row["extra_info"], row["reward_model"]["ground_truth"],
        )


def test_reward_readback_rejects_initial_mutation_absent_from_query_source() -> None:
    calls = [
        {"action": "tool_call", "tool_name": "search_emails", "arguments": {}},
        {"action": "tool_call", "tool_name": "archive_email", "arguments": {}},
        {"action": "final_answer", "tool_name": "", "arguments": {"text": "done"}},
    ]
    extra = {
        "task_id": "unauthorized-initial-mutation",
        "prompt_profile": "local_trainable_v1",
        "source_chain_seed": ["search_emails", "get_email"],
        "oracle_calls": json.dumps(calls),
        "success_criteria": "[]",
        "round_contracts": [{
            "round_idx": 0,
            "required_tools": ["search_emails", "archive_email"],
            "allowed_terminal_actions": ["final_answer"],
        }],
        "allowed_terminal_actions": ["final_answer"],
        "clean_visible_tools": json.dumps([
            {"name": "search_emails", "annotations": {"readonly": True}},
            {"name": "archive_email", "annotations": {"mutating": True}},
        ]),
    }

    with pytest.raises(
        ArtifactIntegrityError, match="mutation absent from source_chain_seed",
    ):
        build_reward_task(extra)


def test_reward_readback_rejects_mutation_without_outcome_evidence() -> None:
    calls = [
        {"action": "tool_call", "tool_name": "search_emails", "arguments": {}},
        {"action": "tool_call", "tool_name": "archive_email", "arguments": {}},
        {"action": "final_answer", "tool_name": "", "arguments": {"text": "done"}},
    ]
    extra = {
        "task_id": "authorized-initial-mutation",
        "prompt_profile": "local_trainable_v1",
        "source_chain_seed": ["search_emails", "archive_email"],
        "oracle_calls": json.dumps(calls),
        "success_criteria": "[]",
        "round_contracts": [{
            "round_idx": 0,
            "required_tools": ["search_emails", "archive_email"],
            "allowed_terminal_actions": ["final_answer"],
        }],
        "allowed_terminal_actions": ["final_answer"],
        "clean_visible_tools": json.dumps([
            {"name": "search_emails", "annotations": {"readonly": True}},
            {"name": "archive_email", "annotations": {"mutating": True}},
        ]),
    }

    with pytest.raises(
        ArtifactIntegrityError, match="mutation_success_criteria_missing",
    ):
        build_reward_task(extra)


def test_reward_readback_accepts_source_authorized_mutation_with_outcome() -> None:
    calls = [
        {"action": "tool_call", "tool_name": "search_emails", "arguments": {}},
        {"action": "tool_call", "tool_name": "archive_email", "arguments": {}},
        {"action": "final_answer", "tool_name": "", "arguments": {"text": "done"}},
    ]
    extra = {
        "task_id": "authorized-initial-mutation",
        "prompt_profile": "local_trainable_v1",
        "source_chain_seed": ["search_emails", "archive_email"],
        "oracle_calls": json.dumps(calls),
        "success_criteria": json.dumps([{
            "path": "emails.email_1.archived", "value": True,
        }]),
        "success_criteria_provenance": json.dumps([{
            "criterion_index": 0,
            "source_calls": [{"tool_name": "archive_email"}],
        }]),
        "round_contracts": [{
            "round_idx": 0,
            "required_tools": ["search_emails", "archive_email"],
            "allowed_terminal_actions": ["final_answer"],
        }],
        "allowed_terminal_actions": ["final_answer"],
        "clean_visible_tools": json.dumps([
            {"name": "search_emails", "annotations": {"readonly": True}},
            {"name": "archive_email", "annotations": {"mutating": True}},
        ]),
    }

    task = build_reward_task(extra)

    assert [call["tool_name"] for call in task["required_tool_calls"]] == [
        "search_emails", "archive_email",
    ]


def test_readback_and_merge_reject_divergent_ground_truth(
    tmp_path,
) -> None:
    row = _tasks_to_rows(
        [_no_tool_task(fixed_attempt_budget=False)], 7,
    )[0]
    row["reward_model"]["ground_truth"]["required_tools"] = [
        "list_accounts"
    ]
    assert _quality_issue(pd.Series(row)).startswith(
        "ground_truth_contract_invalid:ground_truth mismatch"
    )

    path = tmp_path / "divergent.parquet"
    pd.DataFrame([row]).to_parquet(path, index=False)
    with pytest.raises(RuntimeError, match="ground_truth mismatch"):
        validate_parquet_readback(path)


def test_all_oracle_shapes_preserve_ground_truth_mirror_through_parquet(
    tmp_path,
) -> None:
    payloads = [
        {
            "oracle_calls": [
                {"action": "tool_call", "tool_name": "list_accounts", "arguments": {}},
                {"action": "final_answer", "tool_name": "", "arguments": {}},
            ],
            "success_criteria": [],
            "required_tools": ["list_accounts"],
            "dependency_edges": [],
        },
        {
            "oracle_calls": [
                {"action": "tool_call", "tool_name": "list_accounts", "arguments": {}},
                {"action": "tool_call", "tool_name": "freeze_account", "arguments": {"account_id": "acc_1"}},
                {"action": "final_answer", "tool_name": "", "arguments": {}},
            ],
            "success_criteria": [{"path": "accounts.acc_1.status", "value": "frozen"}],
            "required_tools": ["list_accounts", "freeze_account"],
            "dependency_edges": [[0, 1]],
        },
        {
            "oracle_calls": [
                {"action": "tool_call", "tool_name": "list_accounts", "arguments": {}},
                {"action": "report_error", "tool_name": "", "arguments": {}},
            ],
            "success_criteria": [],
            "required_tools": ["list_accounts"],
            "dependency_edges": [],
        },
        {
            "oracle_calls": [
                {"action": "report_error", "tool_name": "", "arguments": {}},
            ],
            "success_criteria": [],
            "required_tools": [],
            "dependency_edges": [],
        },
    ]
    rows = []
    for payload in payloads:
        serialized = {
            key: (
                value if key == "required_tools"
                else json.dumps(value, ensure_ascii=False)
            )
            for key, value in payload.items()
        }
        rows.append({
            "extra_info": dict(serialized),
            "reward_model": {
                "style": "rule",
                "ground_truth": dict(serialized),
            },
        })

    path = tmp_path / "oracle_shapes.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    reloaded = pd.read_parquet(path)
    for _, row in reloaded.iterrows():
        validate_ground_truth_consistency(
            normalize_extra_info(row["extra_info"]),
            row["reward_model"]["ground_truth"],
        )


def test_checkpoint_contract_distinguishes_fixed_attempt_runs(monkeypatch) -> None:
    monkeypatch.setenv("LIVEMCP_FIXED_ATTEMPT_BUDGET", "0")
    regular_args = build_arg_parser().parse_args(["--model", "teacher"])
    monkeypatch.setenv("LIVEMCP_FIXED_ATTEMPT_BUDGET", "1")
    fixed_args = build_arg_parser().parse_args(["--model", "teacher"])

    regular = _checkpoint_config(regular_args)
    fixed = _checkpoint_config(fixed_args)
    assert regular["fixed_attempt_budget"] is False
    assert fixed["fixed_attempt_budget"] is True
    assert regular != fixed
