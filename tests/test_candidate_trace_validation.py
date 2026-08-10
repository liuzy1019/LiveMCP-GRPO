from __future__ import annotations

from src.live_mcp.fsm import RobustnessPlan
from src.live_mcp.generation.candidate_trace_validation import (
    validate_early_candidate_trace,
)
from src.live_mcp.types import OracleCall


def _validate(
    calls: list[OracleCall],
    *,
    plan: RobustnessPlan | None = None,
    source_chain: list[str] | None = None,
):
    return validate_early_candidate_trace(
        domain="calendar",
        difficulty="complete",
        plan=plan or RobustnessPlan(),
        oracle_calls=calls,
        execution_history=[{
            "success": True,
            "tool_name": call.tool_name,
            "server_name": "calendar",
            "state_changed": False,
        } for call in calls if call.action == "tool_call"],
        conversation_queries=["show my events"],
        oracle_calls_per_round=[calls],
        source_chain_seed=source_chain,
        server_tools=[],
        mutation_evidence=[],
        paper_baseline=True,
    )


def test_accepts_realized_readonly_source_chain() -> None:
    calls = [
        OracleCall("list_events", {}),
        OracleCall("", {"text": "done"}, action="final_answer"),
    ]
    result = _validate(calls, source_chain=["list_events"])

    assert result.accepted is True
    assert result.scenario_type == "normal_safe_success"
    assert result.terminal_action == "final_answer"


def test_rejects_successful_distractor_from_ground_truth() -> None:
    calls = [
        OracleCall("list_events", {}),
        OracleCall("", {"text": "done"}, action="final_answer"),
    ]
    result = _validate(
        calls,
        plan=RobustnessPlan(
            inject_distractors=True,
            distractor_tools=[{"name": "list_events"}],
        ),
        source_chain=["list_events"],
    )

    assert result.accepted is False
    assert result.reason.startswith("successful_distractor_calls:")


def test_rejects_complete_task_with_no_tool_and_final_answer() -> None:
    result = _validate([
        OracleCall("", {"text": "done"}, action="final_answer"),
    ])

    assert result.accepted is False
    assert result.reason == "invalid_zero_tool_terminal"


def test_paper_profile_rejects_unauthorized_missing_function_mutation() -> None:
    calls = [
        OracleCall("chown", {"path": "/tmp/a", "owner": "everyone"}),
        OracleCall("", {"text": "chmod is unavailable"}, action="report_error"),
    ]
    result = validate_early_candidate_trace(
        domain="filesystem",
        difficulty="complete",
        plan=RobustnessPlan(missing_function=True, hidden_tool="chmod"),
        oracle_calls=calls,
        execution_history=[{
            "success": True,
            "tool_name": "chown",
            "server_name": "filesystem",
            "state_changed": True,
        }],
        conversation_queries=["make the file executable"],
        oracle_calls_per_round=[calls],
        source_chain_seed=["symlink", "chmod"],
        server_tools=[{
            "name": "chown",
            "annotations": {"readonly": False, "mutating": True},
        }],
        mutation_evidence=[],
        paper_baseline=True,
    )

    assert result.accepted is False
    assert result.reason == "missing_function_nonprefix_mutation"


def test_missing_function_allows_authorized_source_chain_prefix_mutation() -> None:
    calls = [
        OracleCall("symlink", {
            "target": "/tmp/source", "link_path": "/tmp/link",
        }),
        OracleCall("", {"text": "permission change unavailable"}, action="report_error"),
    ]
    result = validate_early_candidate_trace(
        domain="filesystem",
        difficulty="complete",
        plan=RobustnessPlan(missing_function=True, hidden_tool="chmod"),
        oracle_calls=calls,
        execution_history=[{
            "success": True,
            "tool_name": "symlink",
            "server_name": "filesystem",
            "state_changed": True,
        }],
        conversation_queries=["create a link and make it read-only"],
        oracle_calls_per_round=[calls],
        source_chain_seed=["symlink", "chmod"],
        server_tools=[{
            "name": "symlink",
            "annotations": {"readonly": False, "mutating": True},
        }],
        mutation_evidence=[],
        paper_baseline=True,
    )

    assert result.accepted is True
