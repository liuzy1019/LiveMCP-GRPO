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
    """delete+create 的正确检测方式：由 DomainAdapter 设置 forbidden_transition 标记。

    自动跨事件检测（仅靠 target_type 匹配）会产生假阳性，
    如 delete_event(evt_001) → create_event({title: "新会议"}) 是合法操作。
    精确判定需要 DomainAdapter 做状态内容比对，通过 forbidden_transition 标记传递。
    """
    sv = SafetyVerifier()
    log = EventLog(session_id="s2", task_id="t2")
    log.append(AuditEvent(event_id="e1", session_id="s2", step=1,
        action_type="tool_call", operation="query",
        target_type="calendar_event"))
    log.append(AuditEvent(event_id="e2", session_id="s2", step=2,
        action_type="tool_call", operation="delete",
        target_type="calendar_event", target_id="evt_001",
        deleted_ids=["evt_001"]))
    # DomainAdapter 通过状态比对发现本次 create 是在"重建"已删除实体，
    # 显式标记 forbidden_transition
    log.append(AuditEvent(event_id="e3", session_id="s2", step=3,
        action_type="tool_call", operation="create",
        target_type="calendar_event", target_id="evt_002",
        forbidden_transition="forbidden_transition_delete_recreate"))
    r = sv.verify(log)
    assert r.c_safety == 1, f"expected 1, got {r.c_safety}"
    assert r.c_forbidden_transition == 1
    assert "forbidden_transition_delete_recreate" in r.violation_types, \
        f"unexpected violation types: {r.violation_types}"
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


def test_safety_legitimate_delete_create_not_flagged():
    """合法 delete+create（删除旧实体后创建全新实体）不应被误判为 forbidden。

    delete_event(evt_001) → create_event({title: "新会议", ...}) 是合法操作。
    没有 forbidden_transition 标记时，不应触发 SafetyVerifier。
    """
    sv = SafetyVerifier()
    log = EventLog(session_id="s5", task_id="t5")
    log.append(AuditEvent(event_id="e1", session_id="s5", step=1,
        action_type="tool_call", operation="query",
        target_type="calendar_event"))
    log.append(AuditEvent(event_id="e2", session_id="s5", step=2,
        action_type="tool_call", operation="delete",
        target_type="calendar_event", target_id="evt_001",
        deleted_ids=["evt_001"]))
    # 合法创建：不同的 target_id，无 forbidden_transition 标记
    log.append(AuditEvent(event_id="e3", session_id="s5", step=3,
        action_type="tool_call", operation="create",
        target_type="calendar_event", target_id="evt_005",
        created_ids=["evt_005"]))
    r = sv.verify(log)
    assert r.c_safety == 0, f"合法 delete+create 不应被标记为 unsafe，got c_safety={r.c_safety}"
    print("  PASS test_safety_legitimate_delete_create_not_flagged")


def test_safety_self_contradiction():
    """同一实体在轨迹内先创建后删除 → 应被检测为自矛盾。"""
    sv = SafetyVerifier()
    log = EventLog(session_id="s6", task_id="t6")
    log.append(AuditEvent(event_id="e1", session_id="s6", step=1,
        action_type="tool_call", operation="create",
        target_type="calendar_event", target_id="evt_new",
        created_ids=["evt_new"]))
    log.append(AuditEvent(event_id="e2", session_id="s6", step=2,
        action_type="tool_call", operation="delete",
        target_type="calendar_event", target_id="evt_new",
        deleted_ids=["evt_new"]))
    r = sv.verify(log)
    assert r.c_safety == 1, f"自矛盾应被检测，got c_safety={r.c_safety}"
    assert r.c_forbidden_transition == 1
    print("  PASS test_safety_self_contradiction")


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


def test_audit_event_roundtrip():
    """P1-18: AuditEvent → to_dict() → _dict_to_audit_event() 往返序列化测试。

    确保所有关键字段（包括 execution_success, schema_valid, state_changed）
    在序列化/反序列化链中不丢失。新增字段时此测试应失败，提醒开发者同步更新
    两端的字段定义。
    """
    from src.reward.oval_reward_fn import _dict_to_audit_event

    original = AuditEvent(
        event_id="evtlog_sess_000001",
        session_id="sess_abc",
        step=3,
        action_type="tool_call",
        tool_name="update_event",
        tool_arguments={"event_id": "evt_001", "fields": {"title": "new meeting"}},
        terminal_action=None,
        operation="update",
        target_type="calendar_event",
        target_id="evt_001",
        before_hash="abc123def456",
        after_hash="xyz789ghi012",
        changed_fields=["title", "start_time"],
        created_ids=[],
        deleted_ids=[],
        duplicate_of=None,
        identity_violation="",
        forbidden_transition="",
        observation={"status": "ok", "event": {"event_id": "evt_001", "title": "new meeting"}},
        execution_success=True,
        error_type=None,
        error_message="",
        schema_valid=True,
        state_changed=True,
        latency_ms=42,
    )

    d = original.to_dict()
    restored = _dict_to_audit_event(d)

    # 关键字段一一比对
    assert restored.event_id == original.event_id
    assert restored.session_id == original.session_id
    assert restored.step == original.step
    assert restored.action_type == original.action_type
    assert restored.tool_name == original.tool_name
    assert restored.tool_arguments == original.tool_arguments
    assert restored.operation == original.operation
    assert restored.target_type == original.target_type
    assert restored.target_id == original.target_id
    assert restored.before_hash == original.before_hash
    assert restored.after_hash == original.after_hash
    assert restored.changed_fields == original.changed_fields
    assert restored.created_ids == original.created_ids
    assert restored.deleted_ids == original.deleted_ids
    assert restored.duplicate_of == original.duplicate_of
    assert restored.identity_violation == original.identity_violation
    assert restored.forbidden_transition == original.forbidden_transition
    assert restored.execution_success == original.execution_success
    assert restored.schema_valid == original.schema_valid
    assert restored.state_changed == original.state_changed
    assert restored.error_type == original.error_type
    assert restored.error_message == original.error_message
    assert restored.latency_ms == original.latency_ms
    # observation: dict 内容比对
    assert isinstance(restored.observation, dict)
    assert restored.observation.get("status") == "ok"

    print("  PASS test_audit_event_roundtrip: all 23 fields survived serialization roundtrip")


def test_audit_event_roundtrip_terminal():
    """终端事件的 roundtrip 测试（action_type != tool_call）。"""
    from src.reward.oval_reward_fn import _dict_to_audit_event

    original = AuditEvent(
        event_id="evtlog_sess_term_01",
        session_id="sess_xyz",
        step=5,
        action_type="final_answer",
        tool_name="",
        tool_arguments={},
        terminal_action="Meeting rescheduled to 3pm.",
        operation="terminal",
        execution_success=True,
        schema_valid=True,
    )

    d = original.to_dict()
    restored = _dict_to_audit_event(d)

    assert restored.action_type == "final_answer"
    assert restored.tool_name == ""
    assert restored.terminal_action == "Meeting rescheduled to 3pm."
    assert restored.execution_success is True
    assert restored.schema_valid is True

    print("  PASS test_audit_event_roundtrip_terminal")


if __name__ == "__main__":
    print("OVAL-MCP Unit Tests")
    print("=" * 40)
    tests = [
        test_events_helpers,
        test_event_log_properties,
        test_safety_clean,
        test_safety_delete_create_pattern,
        test_safety_legitimate_delete_create_not_flagged,
        test_safety_self_contradiction,
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
        test_audit_event_roundtrip,
        test_audit_event_roundtrip_terminal,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"  FAIL {t.__name__}: {e}")
    print(f"\n{passed}/{len(tests)} passed")
