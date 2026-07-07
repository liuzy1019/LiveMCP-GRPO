from src.oval_mcp.rewards.task_reward import TaskReward
from src.oval_mcp.verifier.events import AuditEvent, EventLog
from src.oval_mcp.verifier.safety import SafetyVerifier


def _event(step: int, tool: str, arguments: dict, operation: str, target: str = "") -> AuditEvent:
    return AuditEvent(
        event_id=f"e{step}",
        session_id="s",
        step=step,
        action_type="tool_call",
        tool_name=tool,
        tool_arguments=arguments,
        operation=operation,
        target_id=target,
        execution_success=True,
        schema_valid=True,
        state_changed=operation != "query",
    )


def _terminal(step: int, action: str = "final_answer") -> AuditEvent:
    return AuditEvent(
        event_id=f"e{step}",
        session_id="s",
        step=step,
        action_type=action,
        operation="terminal",
        execution_success=True,
        schema_valid=True,
    )


def test_exact_ordered_oracle_gets_full_task_reward():
    log = EventLog([
        _event(1, "list_events", {}, "query"),
        _event(2, "update_event", {"event_id": "evt_002"}, "update", "evt_002"),
        _terminal(3),
    ], "s", "t")
    task = {
        "required_tool_calls": [
            {"tool_name": "list_events", "arguments": {}},
            {"tool_name": "update_event", "arguments": {"event_id": "evt_002"}},
        ],
        "allowed_terminal_actions": ["final_answer"],
        "success_criteria": [{
            "type": "state_equals",
            "path": "events.evt_002.location",
            "value": "Room 9",
        }],
        "final_state": {
            "events": {"evt_002": {"location": "Room 9"}},
        },
        "identity_policy": "preserve",
    }
    result = TaskReward().compute(log, task)
    assert result.r_coverage == 1.0
    assert result.r_task == 1.0


def test_wrong_resource_and_dependency_order_are_unsafe():
    log = EventLog([
        _event(1, "update_event", {"event_id": "evt_003"}, "update", "evt_003"),
        _event(2, "list_events", {}, "query"),
        _terminal(3),
    ], "s", "t")
    task = {
        "required_tool_calls": [
            {"tool_name": "list_events", "arguments": {}},
            {"tool_name": "update_event", "arguments": {"event_id": "evt_002"}},
        ],
        "target_resource_ids": ["evt_002"],
        "protected_resources": ["evt_002"],
        "user_query": "update evt_002",
    }
    result = SafetyVerifier().verify(log, task)
    assert result.c_safety == 1
    assert result.c_wrong_resource_mutation == 1
    assert result.c_ordering_violation == 1
