import json

import pandas as pd

from scripts.generate_data import _stratified_task_split, _tasks_to_rows
from src.live_mcp.types import LiveTask, OracleCall, OracleProgram
from src.reward.oval_reward_fn import _build_task_dict


def _tool(name: str) -> dict:
    return {
        "name": name,
        "description": name,
        "input_schema": {"type": "object", "properties": {}, "required": []},
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
