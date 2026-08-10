import copy

import pytest

from src.live_mcp.fsm import ConversationFSM, FSMStateGroup, RobustnessPlan
from src.live_mcp.generation.robustness import (
    build_teacher_visible_tools,
    profile_scenario_is_valid,
)


def test_conversation_fsm_records_legal_five_group_path() -> None:
    fsm = ConversationFSM()
    path = [
        (FSMStateGroup.TURN, "query_ready"),
        (FSMStateGroup.TOOL_EXECUTION, "tool_call"),
        (FSMStateGroup.RESPONSE, "observation"),
        (FSMStateGroup.CONTINUATION, "round_done"),
        (FSMStateGroup.QUERY, "followup"),
    ]

    for state, event in path:
        fsm.transition(state, event)

    assert fsm.state is FSMStateGroup.QUERY
    assert [item["to"] for item in fsm.transitions] == [
        state.value for state, _ in path
    ]


def test_conversation_fsm_rejects_illegal_control_edge() -> None:
    fsm = ConversationFSM()
    fsm.transition(FSMStateGroup.TURN, "query_ready")

    with pytest.raises(RuntimeError, match="turn -> query"):
        fsm.transition(FSMStateGroup.QUERY, "illegal_rewind")


def test_robustness_plan_is_deterministic_and_uses_three_to_eight_unique_distractors() -> None:
    domain_tools = [{"name": "primary"}]
    pool = [
        {"name": "primary", "_server_name": "other"},
        *({"name": f"d{i}", "_server_name": "other"} for i in range(12)),
        {"name": "d0", "_server_name": "duplicate"},
    ]

    first = RobustnessPlan.sample(7, pool, domain_tools, 1.0, 0.3, 0.1)
    second = RobustnessPlan.sample(7, pool, domain_tools, 1.0, 0.3, 0.1)

    assert first == second
    names = [tool["name"] for tool in first.distractor_tools]
    assert 3 <= len(names) <= 8
    assert len(names) == len(set(names))
    assert "primary" not in names


def test_teacher_visibility_applies_enum_strip_and_hidden_tool_without_mutating_source() -> None:
    tools = [
        {"name": "keep", "input_schema": {"type": "string", "enum": ["a"]}},
        {"name": "hide", "input_schema": {"type": "object"}},
    ]
    original = copy.deepcopy(tools)
    plan = RobustnessPlan(
        strip_enums=True, missing_function=True, hidden_tool="hide",
    )

    visible = build_teacher_visible_tools(tools, plan)

    assert [tool["name"] for tool in visible] == ["keep"]
    assert "enum" not in visible[0]["input_schema"]
    assert tools == original


def test_complete_clarification_gate_is_explicitly_local_not_a_prove_hard_gate() -> None:
    assert not profile_scenario_is_valid(
        profile=object(), difficulty="complete",
        scenario_type="clarification_required",
        missing_function=False, irrelevance=False,
    )
    assert profile_scenario_is_valid(
        profile=object(), difficulty="complete",
        scenario_type="clarification_required",
        missing_function=True, irrelevance=False,
    )
