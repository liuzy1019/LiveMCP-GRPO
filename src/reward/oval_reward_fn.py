"""OVAL reward function — verl custom_reward_function 接口。

通过 verl config 的 custom_reward_function.path 指定本文件，
custom_reward_function.name 指定 "compute_score"。

接口签名：
    compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs) -> dict

核心流程：
  1. 从 extra_info 中读取 audit_events（由 SchemaShiftOvalLoop 产生）
  2. 重构 AuditEvent 对象，包装为 EventLog
  3. 通过 SafetyVerifier 计算 C_safety
  4. 通过 TaskReward 计算 R_task
  5. 计算 J = R_task - lambda_safe * C_safety
  6. 返回 score dict（含 score、r_task、c_safety、j 等诊断指标）

Phase 1 配置：
  - I_process = 0
  - I_shape = 0
  - J = R_task - lambda_safe * C_safety
  - lambda_safe 使用固定值（动态更新由 ScalarReturn 在训练循环中完成）
"""

from typing import Any

from src.oval_mcp.verifier.safety import SafetyVerifier
from src.oval_mcp.verifier.events import EventLog, AuditEvent
from src.oval_mcp.rewards.task_reward import TaskReward


# 模块级单例（避免每次调用重新创建）
_safety_verifier = SafetyVerifier()
_task_reward = TaskReward()

# Phase 1 固定 lambda_safe（不在此处动态更新）
_LAMBDA_SAFE_DEFAULT = 1.0


def _dict_to_audit_event(d: dict) -> AuditEvent:
    """从序列化 dict 重构 AuditEvent。

    字段名对齐 AuditEvent dataclass（使用 step 而非 step_index）。
    """
    return AuditEvent(
        event_id=d.get("event_id", ""),
        session_id=d.get("session_id", ""),
        step=d.get("step", d.get("step_index", 0)),
        action_type=d.get("action_type", ""),
        tool_name=d.get("tool_name", ""),
        tool_arguments=d.get("tool_arguments", {}),
        terminal_action=d.get("terminal_action"),
        operation=d.get("operation", ""),
        target_type=d.get("target_type", ""),
        target_id=d.get("target_id", ""),
        before_hash=d.get("before_hash", ""),
        after_hash=d.get("after_hash", ""),
        changed_fields=d.get("changed_fields", []),
        created_ids=d.get("created_ids", []),
        deleted_ids=d.get("deleted_ids", []),
        duplicate_of=d.get("duplicate_of"),
        identity_violation=d.get("identity_violation", ""),
        forbidden_transition=d.get("forbidden_transition", ""),
        observation=d.get("observation"),
        execution_success=d.get("execution_success", False),
        error_type=d.get("error_type"),
        error_message=d.get("error_message", ""),
        schema_valid=d.get("schema_valid", False),
        state_changed=d.get("state_changed", False),
        latency_ms=d.get("latency_ms", 0),
    )


def _parse_audit_events(raw: Any) -> list[AuditEvent]:
    """从 extra_info 中解析 audit_events。

    支持：JSON string / list[dict] / list[AuditEvent]。
    """
    import json

    if raw is None:
        return []

    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return []

    if not isinstance(raw, list):
        return []

    events: list[AuditEvent] = []
    for item in raw:
        if isinstance(item, AuditEvent):
            events.append(item)
        elif isinstance(item, dict):
            try:
                events.append(_dict_to_audit_event(item))
            except Exception:
                pass
    return events


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict | None = None,
    **kwargs,
) -> dict:
    """OVAL reward function — R_task + C_safety + J。

    Args:
        data_source: 数据源标识（"schemashift_oval"）
        solution_str: 模型生成的完整 response 文本
        ground_truth: oracle 信息（dict 或 JSON string）
        extra_info: 由 SchemaShiftOvalLoop 产生的额外字段
            {
                "audit_events": [dict],   # AuditEvent.to_dict() 序列化结果
                "task_id": str,
                "domain": str,
                "required_tools": list[str],
                "n_model_tool_calls": int,
                "n_exec_success": int,
            }

    Returns:
        dict with "score" key (float) + scalar diagnostic keys.
    """
    extra_info = extra_info or {}

    # ── 解析 audit_events ──
    audit_raw = extra_info.get("audit_events", [])
    audit_events = _parse_audit_events(audit_raw)

    if not audit_events:
        return {
            "score": 0.0,
            "r_task": 0.0,
            "c_safety": 0.0,
            "j": 0.0,
            "n_events": 0,
            "n_model_tool_calls": float(extra_info.get("n_model_tool_calls", 0)),
            "n_exec_success": float(extra_info.get("n_exec_success", 0)),
            "error": "no audit events",
        }

    # ── 构建 EventLog ──
    session_id = extra_info.get("session_id", "")
    task_id = extra_info.get("task_id", "unknown")
    event_log = EventLog(events=audit_events, session_id=session_id, task_id=task_id)

    # ── 构建 task_dict ──
    domain = extra_info.get("domain", "unknown")
    required_tools = extra_info.get("required_tools", [])
    if isinstance(required_tools, str):
        required_tools = [t.strip() for t in required_tools.split(",") if t.strip()]

    ot_map = {
        "list_events": "query",
        "create_event": "create",
        "update_event": "update",
        "delete_event": "delete",
        "search_products": "query",
        "add_to_cart": "update",
        "remove_from_cart": "update",
        "checkout": "create",
        "get_order": "query",
    }
    assertions: list[dict] = []
    for tn in required_tools:
        op = ot_map.get(tn, "query")
        assertions.append({"operation": op, "tool_name": tn})
    assertions.append({"operation": "terminal", "tool_name": ""})

    task_dict = {
        "task_id": task_id,
        "required_tool_calls": [
            {"tool_name": tn, "arguments": {}}
            for tn in required_tools
        ],
        "identity_policy": "preserve" if domain == "calendar" else "create_new",
        "budget": extra_info.get("budget", 8),
        "outcome_assertions": assertions,
        "allowed_terminal_actions": ["final_answer", "report_error"],
    }

    # ── 计算 R_task ──
    try:
        r_result = _task_reward.compute(event_log, task_dict)
        r_task = r_result.r_task
        r_validity = r_result.r_validity
        r_coverage = r_result.r_coverage
    except Exception:
        r_task = 0.0
        r_validity = 0.0
        r_coverage = 0.0

    # ── 计算 C_safety ──
    try:
        safety_result = _safety_verifier.verify(event_log)
        c_safety = safety_result.c_safety
        violations = safety_result.violation_types
    except Exception:
        c_safety = 0.0
        violations = []

    # ── 计算 J ──
    lambda_safe = float(extra_info.get("lambda_safe", _LAMBDA_SAFE_DEFAULT))
    j = r_task - lambda_safe * c_safety

    n_model_calls = float(extra_info.get("n_model_tool_calls", 0))
    n_exec_ok = float(extra_info.get("n_exec_success", 0))
    n_events = len(audit_events)

    return {
        "score": float(j),
        "r_task": float(r_task),
        "r_validity": float(r_validity),
        "r_coverage": float(r_coverage),
        "c_safety": float(c_safety),
        "c_violations": ",".join(violations) if violations else "",
        "j": float(j),
        "lambda_safe": float(lambda_safe),
        "n_events": float(n_events),
        "n_model_tool_calls": n_model_calls,
        "n_exec_success": n_exec_ok,
    }
