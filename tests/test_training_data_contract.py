import json

import pandas as pd
import pytest

from scripts.generate_data import _stratified_task_split, _tasks_to_rows
from src.live_mcp.task_planner import derive_success_criteria
from src.live_mcp.types import LiveTask, OracleCall, OracleProgram
from src.reward.oval_reward_fn import _build_task_dict


def _tool(name: str, *, mutating: bool = False) -> dict:
    return {
        "name": name,
        "description": name,
        "input_schema": {"type": "object", "properties": {}, "required": []},
        "annotations": {"mutating": mutating, "readonly": not mutating},
    }


def _task(idx: int, domain: str = "calendar", scenario: str = "normal_safe_success") -> LiveTask:
    calls = [
        OracleCall("list_events", {"date_range": "2026-06-24"}),
        OracleCall("get_event", {"event_id": f"evt_{idx:03d}"}),
        OracleCall("final_answer", {"text": "Done"}, action="final_answer"),
    ]
    return LiveTask(
        task_id=f"{domain}_{idx}",
        source="test",
        suite_name="test",
        user_prompt=f"check event evt_{idx:03d}",
        session_id="",
        session_seed=idx,
        target_servers=[domain],
        visible_tools=[_tool("list_events"), _tool("get_event")],
        required_tools=["list_events", "get_event"],
        expected_outcome={},
        success_criteria=[],
        oracle_program=OracleProgram(f"{domain}_{idx}", calls, []),
        sampling_context={},
        max_turns=8,
        difficulty="complete",
        task_type="task_planner",
        metadata={
            "scenario_type": scenario,
            "initial_state_hash": "abc",
            "identity_policy": "preserve",
            "target_resource_ids": [f"evt_{idx:03d}"],
        },
    )


def test_rows_hide_teacher_trace_and_keep_complete_oracle(tmp_path):
    row = _tasks_to_rows([_task(1)], 1)[0]
    prompt = json.loads(row["prompt"])
    assert [message["role"] for message in prompt] == ["system", "user"]
    assert "<tool_call>" not in prompt[1]["content"]

    calls = json.loads(row["extra_info"]["oracle_calls"])
    assert [call["action"] for call in calls] == [
        "tool_call", "tool_call", "final_answer"
    ]

    path = tmp_path / "contract.parquet"
    pd.DataFrame([row]).to_parquet(path, index=False)
    restored = pd.read_parquet(path).iloc[0]
    task = _build_task_dict(dict(restored.extra_info))
    assert len(task["required_tool_calls"]) == 2
    assert task["allowed_terminal_actions"] == ["final_answer"]


def test_stratified_split_is_exact_and_disjoint():
    tasks = []
    domains = ["calendar", "shopping"]
    scenarios = ["normal_safe_success", "unsafe_temptation"]
    for idx in range(24):
        tasks.append(_task(idx, domains[idx % 2], scenarios[(idx // 2) % 2]))

    train, val = _stratified_task_split(tasks, train_count=16, val_count=8, seed=7)
    assert len(train) == 16
    assert len(val) == 8
    assert {t.task_id for t in train}.isdisjoint({t.task_id for t in val})
    assert {t.target_servers[0] for t in val} == set(domains)
    assert {t.metadata["scenario_type"] for t in val} == set(scenarios)


def test_stratified_split_keeps_val_scenario_proportions():
    tasks = []
    for idx in range(80):
        tasks.append(_task(idx, "calendar", "normal_safe_success"))
    for idx in range(80, 100):
        tasks.append(_task(idx, "calendar", "clarification_required"))

    train, val = _stratified_task_split(tasks, train_count=80, val_count=20, seed=5)

    val_scenarios = [task.metadata["scenario_type"] for task in val]
    assert len(train) == 80
    assert len(val) == 20
    assert val_scenarios.count("normal_safe_success") == 16
    assert val_scenarios.count("clarification_required") == 4


def test_stratified_split_drops_unsplittable_singleton_scenario():
    tasks = [_task(idx) for idx in range(24)]
    tasks.append(_task(999, scenario="singleton_adversarial_case"))

    train, val = _stratified_task_split(tasks, train_count=16, val_count=8, seed=9)

    assert len(train) == 16
    assert len(val) == 8
    assert all(
        task.metadata["scenario_type"] != "singleton_adversarial_case"
        for task in train + val
    )


def test_stratified_split_val_zero_is_exact_for_small_smoke():
    tasks = [
        _task(0, scenario="task_planner"),
        _task(1, scenario="tool_error_recovery"),
        _task(2, scenario="clarification_required"),
        _task(3, scenario="no_tool_or_abstention"),
    ]

    train, val = _stratified_task_split(tasks, train_count=2, val_count=0, seed=11)

    assert len(train) == 2
    assert val == []
    assert len({task.task_id for task in train}) == 2


def test_stratified_split_falls_back_when_strict_coverage_impossible():
    tasks = [
        _task(0, scenario="task_planner"),
        _task(1, scenario="task_planner"),
        _task(2, scenario="clarification_required"),
        _task(3, scenario="no_tool_or_abstention"),
        _task(4, scenario="tool_error_recovery"),
    ]

    train, val = _stratified_task_split(tasks, train_count=2, val_count=1, seed=7)

    assert len(train) == 2
    assert len(val) == 1
    assert {task.task_id for task in train}.isdisjoint({task.task_id for task in val})


def test_no_tool_task_has_terminal_but_no_fake_tool_call():
    task = _task(99)
    task.oracle_program.calls = [
        OracleCall(
            "report_error",
            {"text": "Required tool is unavailable"},
            action="report_error",
        )
    ]
    task.required_tools = []
    task.task_type = "missing_function"
    task.metadata["scenario_type"] = "no_tool_or_abstention"
    task.metadata["has_missing_function"] = True
    row = _tasks_to_rows([task], 99)[0]
    calls = json.loads(row["extra_info"]["oracle_calls"])
    assert calls == [{
        "tool_name": "report_error",
        "arguments": {"text": "Required tool is unavailable"},
        "action": "report_error",
    }]


def test_missing_function_does_not_turn_read_request_into_abstention():
    from src.live_mcp.orchestrator import TaskOrchestrator

    task = _task(100)
    task.user_prompt = "what’s my schedule for tomorrow"
    task.conversation_queries = [task.user_prompt]
    task.visible_tools = [
        _tool("list_events"),
        _tool("create_event", mutating=True),
    ]
    task.required_tools = ["list_events", "create_event"]
    task.oracle_program.calls = [
        OracleCall("list_events", {"date_range": "tomorrow"}),
        OracleCall(
            "create_event",
            {
                "title": "Follow up",
                "start_time": "2026-07-02T10:00:00",
                "end_time": "2026-07-02T10:30:00",
            },
        ),
        OracleCall("final_answer", {"text": "Done"}, action="final_answer"),
    ]

    orchestrator = TaskOrchestrator.__new__(TaskOrchestrator)
    orchestrator._apply_missing_function(task)

    assert task.task_type == "task_planner"
    assert task.hidden_tools == []
    assert task.required_tools == ["list_events", "create_event"]
    assert [call.tool_name for call in task.oracle_program.calls[:2]] == [
        "list_events", "create_event"
    ]


def test_missing_function_allows_matching_write_request():
    from src.live_mcp.orchestrator import TaskOrchestrator

    task = _task(101)
    task.user_prompt = "remove jdoe from evt_002"
    task.conversation_queries = [task.user_prompt]
    task.visible_tools = [
        _tool("get_event"),
        _tool("remove_attendee", mutating=True),
    ]
    task.required_tools = ["get_event", "remove_attendee"]
    task.oracle_program.calls = [
        OracleCall("get_event", {"event_id": "evt_002"}),
        OracleCall("remove_attendee", {"event_id": "evt_002", "email": "jdoe"}),
        OracleCall("final_answer", {"text": "Done"}, action="final_answer"),
    ]

    orchestrator = TaskOrchestrator.__new__(TaskOrchestrator)
    orchestrator._apply_missing_function(task)

    assert task.task_type == "missing_function"
    assert task.hidden_tools == ["remove_attendee"]
    assert task.required_tools == []
    assert task.oracle_program.calls == [
        OracleCall(
            "report_error",
            {"text": "Required tool 'remove_attendee' is unavailable."},
            action="report_error",
        )
    ]


def test_classify_scenario_ignores_clarification_when_tool_calls_present():
    """P0-1 regression: a task that ran real tool_calls and finished with
    ask_clarification must NOT be tagged clarification_required, otherwise
    _tasks_to_rows will reject the row (NO_TOOL scenario + real calls) and
    generate_many will silently drop the whole task.
    """
    from src.live_mcp.orchestrator import _classify_scenario

    tool_calls = [
        OracleCall("list_events", {"date_range": "2026-06-24"}),
        OracleCall("get_event", {"event_id": "evt_1"}),
        OracleCall(
            "ask_clarification",
            {"question": "which timezone?"},
            action="ask_clarification",
        ),
    ]
    scenario = _classify_scenario(
        server_name="calendar",
        oracle_calls=tool_calls,
        execution_history=[],
        terminal_action="ask_clarification",
        seed=1,
    )
    assert scenario != "clarification_required"
    assert scenario in {
        "normal_safe_success", "unsafe_temptation", "missing_dependency"
    }


def test_classify_scenario_keeps_clarification_when_no_tool_calls():
    """clarification_required is still the correct label when the oracle
    contains zero real tool_calls (missing-difficulty path).
    """
    from src.live_mcp.orchestrator import _classify_scenario

    calls = [
        OracleCall(
            "ask_clarification",
            {"question": "which account?"},
            action="ask_clarification",
        ),
    ]
    scenario = _classify_scenario(
        server_name="banking",
        oracle_calls=calls,
        execution_history=[],
        terminal_action="ask_clarification",
        seed=2,
    )
    assert scenario == "clarification_required"


def test_tasks_to_rows_accepts_tool_task_ending_with_clarification():
    """End-to-end coverage of P0-1: rows with 2-5 tool_calls plus a trailing
    ask_clarification serialize cleanly and land in a TOOL_SCENARIOS bucket.
    """
    task = _task(42)
    task.oracle_program.calls = [
        OracleCall("list_events", {"date_range": "2026-06-24"}),
        OracleCall("get_event", {"event_id": "evt_042"}),
        OracleCall(
            "ask_clarification",
            {"question": "confirm timezone?"},
            action="ask_clarification",
        ),
    ]
    task.metadata["scenario_type"] = "normal_safe_success"
    row = _tasks_to_rows([task], 42)[0]
    ei = row["extra_info"]
    assert ei["scenario_type"] == "normal_safe_success"
    assert ei["allowed_terminal_actions"] == ["ask_clarification"]
    assert len(ei["required_tools"]) == 2


def test_tasks_to_rows_drops_intermediate_terminals():
    task = _task(43)
    task.oracle_program.calls = [
        OracleCall("list_events", {"date_range": "2026-06-24"}),
        OracleCall("final_answer", {"text": "round one done"}, action="final_answer"),
        OracleCall("get_event", {"event_id": "evt_043"}),
        OracleCall("final_answer", {"text": "Done"}, action="final_answer"),
    ]

    row = _tasks_to_rows([task], 43)[0]
    calls = json.loads(row["extra_info"]["oracle_calls"])

    assert [call["action"] for call in calls] == [
        "tool_call", "tool_call", "final_answer"
    ]
    assert row["extra_info"]["allowed_terminal_actions"] == ["final_answer"]


def test_tasks_to_rows_exports_prove_continuation_queries():
    task = _task(431)
    task.user_prompt = "check my calendar first"
    task.conversation_queries = [
        "check my calendar first",
        "great, now open the event details",
    ]
    task.oracle_program.calls = [
        OracleCall("list_events", {"date_range": "2026-06-24"}),
        OracleCall("final_answer", {"text": "I found one event."}, action="final_answer"),
        OracleCall("get_event", {"event_id": "evt_431"}),
        OracleCall("final_answer", {"text": "Done"}, action="final_answer"),
    ]

    row = _tasks_to_rows([task], 431)[0]
    prompt = json.loads(row["prompt"])
    calls = json.loads(row["extra_info"]["oracle_calls"])
    queries = json.loads(row["extra_info"]["conversation_queries"])

    assert [message["role"] for message in prompt] == ["system", "user"]
    assert prompt[1]["content"] == "check my calendar first"
    assert queries == [
        "check my calendar first",
        "great, now open the event details",
    ]
    assert row["extra_info"]["conversation_rounds"] == 2
    assert [call["action"] for call in calls] == [
        "tool_call", "tool_call", "final_answer"
    ]


def test_tasks_to_rows_rejects_stale_explicit_years():
    task = _task(44)
    task.oracle_program.calls[0] = OracleCall(
        "list_events", {"date_range": "2023-10-10"}
    )

    with pytest.raises(ValueError, match="pre-2026"):
        _tasks_to_rows([task], 44)


def test_derive_success_criteria_emits_state_absent_for_deletes():
    criteria = derive_success_criteria(
        initial_state={
            "events": {
                "evt_001": {"event_id": "evt_001", "title": "Delete me"},
                "evt_002": {"event_id": "evt_002", "title": "Keep me"},
            }
        },
        final_state={
            "events": {
                "evt_002": {"event_id": "evt_002", "title": "Keep me"},
            }
        },
        oracle_calls=[OracleCall("delete_event", {"event_id": "evt_001"})],
        domain="calendar",
    )

    assert {
        "type": "state_absent",
        "server": "calendar",
        "path": "events.evt_001",
        "path_parts": ["events", "evt_001"],
    } in criteria


def test_tasks_to_rows_allows_replay_valid_mutating_task_without_state_delta():
    task = _task(45, domain="banking")
    task.visible_tools = [
        _tool("list_accounts"),
        _tool("get_account_info"),
        _tool("freeze_account"),
    ]
    task.oracle_program.calls = [
        OracleCall("list_accounts", {}),
        OracleCall("get_account_info", {"account_id": "acc_frozen_demo"}),
        OracleCall("freeze_account", {"account_id": "acc_frozen_demo"}),
        OracleCall("final_answer", {"text": "Done"}, action="final_answer"),
    ]
    task.oracle_program.success_criteria = []
    task.success_criteria = []

    row = _tasks_to_rows([task], 45)[0]
    assert row["extra_info"]["success_criteria"] == "[]"
    assert json.loads(row["extra_info"]["oracle_calls"])[-1]["action"] == "final_answer"
