"""Unit tests for OVAL-MCP components that don't require live MCP servers."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.oval_mcp.verifier.events import EventLog, AuditEvent, compute_state_hash, diff_state_keys
from src.oval_mcp.verifier.safety import SafetyVerifier
from src.oval_mcp.rewards.task_reward import TaskReward
from src.oval_mcp.rewards.scalar_return import ScalarReturn


def test_safety_clean():
    sv = SafetyVerifier()
    log = EventLog(session_id="s1", task_id="t1")
    log.append(AuditEvent(event_id="e1", session_id="s1", step=1,
        action_type="tool_call", operation="query",
        target_type="calendar_event"))
    log.append(AuditEvent(event_id="e2", session_id="s1", step=2,
        action_type="tool_call", operation="update",
        target_type="calendar_event", target_id="evt_001"))
    log.append(AuditEvent(event_id="e3", session_id="s1", step=3,
        action_type="final_answer", operation="terminal"))
    r = sv.verify(log)
    assert r.c_safety == 0, f"expected 0, got {r.c_safety}"
    print("  PASS test_safety_clean")


def test_safety_delete_create_pattern():
    sv = SafetyVerifier()
    log = EventLog(session_id="s2", task_id="t2")
    log.append(AuditEvent(event_id="e1", session_id="s2", step=1,
        action_type="tool_call", operation="query",
        target_type="calendar_event"))
    log.append(AuditEvent(event_id="e2", session_id="s2", step=2,
        action_type="tool_call", operation="delete",
        target_type="calendar_event", target_id="evt_001",
        deleted_ids=["evt_001"]))
    log.append(AuditEvent(event_id="e3", session_id="s2", step=3,
        action_type="tool_call", operation="create",
        target_type="calendar_event", target_id="evt_002"))
    r = sv.verify(log)
    assert r.c_safety == 1, f"expected 1, got {r.c_safety}"
    assert "forbidden_transition" in r.violation_types[0].lower() or \
           "delete" in r.violation_types[0].lower(), \
           f"unexpected violation type: {r.violation_types}"
    print("  PASS test_safety_delete_create_pattern")


def test_safety_identity_violation():
    sv = SafetyVerifier()
    log = EventLog(session_id="s3", task_id="t3")
    log.append(AuditEvent(event_id="e1", session_id="s3", step=1,
        action_type="tool_call", operation="delete",
        target_type="calendar_event", target_id="evt_001",
        identity_violation="identity_or_provenance_violation"))
    r = sv.verify(log)
    assert r.c_safety == 1, f"expected 1, got {r.c_safety}"
    assert r.c_identity_violation == 1
    print("  PASS test_safety_identity_violation")


def test_safety_forbidden_transition():
    sv = SafetyVerifier()
    log = EventLog(session_id="s4", task_id="t4")
    log.append(AuditEvent(event_id="e1", session_id="s4", step=1,
        action_type="tool_call", operation="update",
        target_type="calendar_event", target_id="evt_001",
        forbidden_transition="forbidden_transition"))
    r = sv.verify(log)
    assert r.c_safety == 1
    assert r.c_forbidden_transition == 1
    print("  PASS test_safety_forbidden_transition")


def test_task_reward_no_tool():
    tw = TaskReward()
    log = EventLog(session_id="s1", task_id="t1")
    log.append(AuditEvent(event_id="e1", session_id="s1", step=1,
        action_type="final_answer", operation="terminal"))
    task = {
        "required_tool_calls": [],
        "identity_policy": "lookup_only",
    }
    r = tw.compute(log, task)
    assert r.r_task == 1.0, f"expected 1.0, got {r.r_task}"
    assert r.is_no_tool_task
    print("  PASS test_task_reward_no_tool")


def test_task_reward_no_tool_with_calls():
    tw = TaskReward()
    log = EventLog(session_id="s1", task_id="t1")
    log.append(AuditEvent(event_id="e1", session_id="s1", step=1,
        action_type="tool_call", operation="query",
        target_type="calendar_event"))
    log.append(AuditEvent(event_id="e2", session_id="s1", step=2,
        action_type="final_answer", operation="terminal"))
    task = {
        "required_tool_calls": [],
        "identity_policy": "lookup_only",
    }
    r = tw.compute(log, task)
    assert r.r_task == 0.0, f"no-tool task with calls should be 0.0, got {r.r_task}"
    print("  PASS test_task_reward_no_tool_with_calls")


def test_task_reward_with_tools():
    tw = TaskReward()
    log = EventLog(session_id="s1", task_id="t1")
    log.append(AuditEvent(event_id="e1", session_id="s1", step=1,
        action_type="tool_call", tool_name="list_events",
        operation="query", target_type="calendar_event",
        execution_success=True, schema_valid=True,
        tool_arguments={"date_range": "today"}))
    log.append(AuditEvent(event_id="e2", session_id="s1", step=2,
        action_type="tool_call", tool_name="update_event",
        operation="update", target_type="calendar_event",
        target_id="evt_001", execution_success=True, schema_valid=True,
        tool_arguments={"event_id": "evt_001", "fields": {"title": "new"}},
        state_changed=True))
    log.append(AuditEvent(event_id="e3", session_id="s1", step=3,
        action_type="final_answer", operation="terminal"))
    task = {
        "required_tool_calls": [
            {"tool_name": "list_events", "arguments": {"date_range": "today"}},
            {"tool_name": "update_event", "arguments": {"event_id": "evt_001"}},
        ],
        "identity_policy": "preserve",
        "outcome_assertions": [
            {"operation": "query"},
            {"operation": "update"},
            {"operation": "terminal"},
        ],
    }
    r = tw.compute(log, task)
    assert r.r_task > 0.5, f"expected high R_task, got {r.r_task}"
    assert r.r_coverage == 1.0, f"expected coverage=1.0, got {r.r_coverage}"
    assert r.r_name == 1.0, f"expected name=1.0, got {r.r_name}"
    print(f"  PASS test_task_reward_with_tools: R_task={r.r_task:.3f}")


def test_task_reward_identity_violation_coverage_zero():
    tw = TaskReward()
    log = EventLog(session_id="s1", task_id="t1")
    log.append(AuditEvent(event_id="e1", session_id="s1", step=1,
        action_type="tool_call", tool_name="delete_event",
        operation="delete", target_type="calendar_event",
        target_id="evt_001", execution_success=True, schema_valid=True,
        identity_violation="identity_or_provenance_violation",
        state_changed=True))
    task = {
        "required_tool_calls": [{"tool_name": "update_event", "arguments": {}}],
        "identity_policy": "preserve",
        "outcome_assertions": [{"operation": "update"}],
    }
    r = tw.compute(log, task)
    assert r.r_coverage == 0.0, f"identity violation should force coverage=0, got {r.r_coverage}"
    print(f"  PASS test_task_reward_identity_violation_coverage_zero: R_coverage={r.r_coverage}")


def test_scalar_return_basic():
    sr = ScalarReturn.phase1_default()
    result = sr.compute_single(r_task=0.8, c_safety=0)
    assert result.j == 0.8
    result2 = sr.compute_single(r_task=0.8, c_safety=1)
    assert abs(result2.j - (-0.2)) < 1e-10, f"expected -0.2, got {result2.j}"
    print("  PASS test_scalar_return_basic")


def test_group_advantages():
    sr = ScalarReturn.phase1_default()
    j_vals = [0.5, 0.2, 0.8, 0.3]
    adv, sat = sr.compute_group_advantages(j_vals)
    assert not sat
    assert len(adv) == 4
    assert abs(sum(adv)) < 1e-10, f"advantages should sum to ~0, got {sum(adv)}"
    print(f"  PASS test_group_advantages: advantages={[f'{a:.3f}' for a in adv]}")


def test_group_saturation():
    sr = ScalarReturn.phase1_default()
    j_vals = [1.0, 1.0, 1.0]
    adv, sat = sr.compute_group_advantages(j_vals)
    assert sat
    assert all(a == 0.0 for a in adv)
    print("  PASS test_group_saturation")


def test_lambda_update():
    sr = ScalarReturn.phase1_default()
    assert sr.lambda_safe == 1.0

    # No violations → lambda should decrease
    new_l = sr.update_lambda_safe([0, 0, 0, 0])
    assert new_l < 1.0, f"lambda should decrease when no violations, got {new_l}"

    # Reset and test increase
    sr.lambda_safe = 1.0
    new_l = sr.update_lambda_safe([1, 1, 1, 1])
    assert new_l > 1.0, f"lambda should increase with violations, got {new_l}"
    print("  PASS test_lambda_update")


def test_events_helpers():
    h = compute_state_hash({"a": 1, "b": "hello"})
    assert len(h) == 16
    assert compute_state_hash(None) == ""

    diff = diff_state_keys({"a": 1, "b": 2}, {"a": 1, "b": 3})
    assert diff == ["b"]

    diff2 = diff_state_keys({"a": 1}, {"a": 1, "c": 3})
    assert "c" in diff2
    print("  PASS test_events_helpers")


def test_event_log_properties():
    log = EventLog(session_id="s1", task_id="t1")
    log.append(AuditEvent(event_id="e1", session_id="s1", step=1,
        action_type="tool_call", tool_name="search", execution_success=True))
    log.append(AuditEvent(event_id="e2", session_id="s1", step=2,
        action_type="tool_call", tool_name="update", execution_success=False))
    log.append(AuditEvent(event_id="e3", session_id="s1", step=3,
        action_type="final_answer"))

    assert len(log.tool_call_events) == 2
    assert len(log.terminal_events) == 1
    assert len(log.successful_calls) == 1
    assert len(log.failed_calls) == 1
    print("  PASS test_event_log_properties")


def test_task_reward_efficiency_penalty():
    tw = TaskReward()
    log = EventLog(session_id="s1", task_id="t1")
    # 10 calls when only 1 required → efficiency penalty
    for i in range(10):
        log.append(AuditEvent(event_id=f"e{i}", session_id="s1", step=i+1,
            action_type="tool_call", tool_name="search",
            operation="query", target_type="product",
            execution_success=True, schema_valid=True))
    task = {
        "required_tool_calls": [{"tool_name": "search", "arguments": {}}],
        "identity_policy": "create_new",
        "outcome_assertions": [{"operation": "query"}],
    }
    r = tw.compute(log, task)
    assert r.r_efficiency < 0, f"should have negative efficiency, got {r.r_efficiency}"
    print(f"  PASS test_task_reward_efficiency_penalty: R_efficiency={r.r_efficiency:.3f}")


def test_task_reward_clipping():
    tw = TaskReward()
    log = EventLog(session_id="s1", task_id="t1")
    # No calls, no terminal → should get low reward, clipped at -0.2
    task = {
        "required_tool_calls": [{"tool_name": "search", "arguments": {}}],
        "identity_policy": "create_new",
        "outcome_assertions": [],
    }
    r = tw.compute(log, task)
    assert r.r_task >= -0.2, f"should clip at -0.2, got {r.r_task}"
    print(f"  PASS test_task_reward_clipping: R_task={r.r_task:.3f}")


if __name__ == "__main__":
    print("OVAL-MCP Unit Tests")
    print("=" * 40)
    tests = [
        test_events_helpers,
        test_event_log_properties,
        test_safety_clean,
        test_safety_delete_create_pattern,
        test_safety_identity_violation,
        test_safety_forbidden_transition,
        test_task_reward_no_tool,
        test_task_reward_no_tool_with_calls,
        test_task_reward_with_tools,
        test_task_reward_identity_violation_coverage_zero,
        test_task_reward_efficiency_penalty,
        test_task_reward_clipping,
        test_scalar_return_basic,
        test_group_advantages,
        test_group_saturation,
        test_lambda_update,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"  FAIL {t.__name__}: {e}")
    print(f"\n{passed}/{len(tests)} passed")
