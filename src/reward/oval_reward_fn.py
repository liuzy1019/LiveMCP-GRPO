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
  5. 通过 ProgressTracker 计算 F_gamma（仅当 I_shape != 0）
  6. 通过 ProcessScorer 计算 P_process（仅当 I_process != 0）
  7. 从 LambdaState 读取当前 lambda_safe
  8. 计算 J = R_task + I_shape*lambda_shape*F + I_process*lambda_process*P - lambda_safe*C
  9. 返回 score dict（含 score、r_task、c_safety、f_gamma、p_process、j 等）

Phase 1 默认配置：
  - I_process = 0
  - I_shape = 0
  - J = R_task - lambda_safe * C_safety

Phase 2 可通过环境变量启用 shaping/process 进行消融。
"""

import os
from typing import Any

from src.oval_mcp.verifier.safety import SafetyVerifier
from src.oval_mcp.verifier.events import EventLog, AuditEvent
from src.oval_mcp.rewards.task_reward import TaskReward
from src.oval_mcp.rewards.f_gamma import ProgressTracker
from src.oval_mcp.rewards.p_process import ProcessScorer

try:
    from src.oval_mcp.training.lambda_state import LambdaState, DEFAULT_STATE_PATH
except ImportError:
    LambdaState = None  # type: ignore
    DEFAULT_STATE_PATH = "/tmp/ssgrpo_lambda_state.json"


# ── 模块级单例 ──
_safety_verifier = SafetyVerifier()
_task_reward = TaskReward()
_progress_tracker = ProgressTracker()
_process_scorer = ProcessScorer(p_max=0.3)

# ── 消融开关（环境变量控制） ──
_I_SHAPE = int(os.environ.get("OVAL_I_SHAPE", "0"))
_I_PROCESS = int(os.environ.get("OVAL_I_PROCESS", "0"))
_LAMBDA_SHAPE = float(os.environ.get("OVAL_LAMBDA_SHAPE", "0.5"))
_LAMBDA_PROCESS = float(os.environ.get("OVAL_LAMBDA_PROCESS", "0.3"))
_GAMMA = float(os.environ.get("OVAL_GAMMA", "1.0"))

# Phase 1 默认 lambda_safe（当 LambdaState 不可用时）
_LAMBDA_SAFE_DEFAULT = 1.0


def _dict_to_audit_event(d: dict) -> AuditEvent:
    """从序列化 dict 重构 AuditEvent。"""
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
    """从 extra_info 中解析 audit_events。"""
    import json as _json

    if raw is None:
        return []

    if isinstance(raw, str):
        try:
            raw = _json.loads(raw)
        except _json.JSONDecodeError:
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


def _build_task_dict(extra_info: dict) -> dict:
    """从 extra_info 构建 task_dict。"""
    domain = extra_info.get("domain", "unknown")
    task_id = extra_info.get("task_id", "unknown")
    required_tools = extra_info.get("required_tools", [])
    if isinstance(required_tools, str):
        required_tools = [t.strip() for t in required_tools.split(",") if t.strip()]

    ot_map = {
        "list_events": "query", "create_event": "create",
        "update_event": "update", "delete_event": "delete",
        "search_products": "query", "add_to_cart": "update",
        "remove_from_cart": "update", "checkout": "create",
        "get_order": "query",
    }
    assertions: list[dict] = []
    for tn in required_tools:
        op = ot_map.get(tn, "query")
        assertions.append({"operation": op, "tool_name": tn})
    assertions.append({"operation": "terminal", "tool_name": ""})

    return {
        "task_id": task_id,
        "required_tool_calls": [
            {"tool_name": tn, "arguments": {}} for tn in required_tools
        ],
        "identity_policy": "preserve" if domain == "calendar" else "create_new",
        "budget": extra_info.get("budget", 8),
        "outcome_assertions": assertions,
        "allowed_terminal_actions": ["final_answer", "report_error"],
    }


def _compute_f_gamma(event_log: EventLog, task_dict: dict) -> dict:
    """计算 F_gamma 及其分解值。"""
    try:
        fg_result = _progress_tracker.compute(event_log, task_dict, gamma=_GAMMA)
        return {
            "f_gamma": fg_result.f_gamma,
            "phi_initial": fg_result.phi_initial,
            "phi_final": fg_result.phi_final,
            "completed_required": float(fg_result.completed_required_states),
            "total_required": float(fg_result.total_required_states),
        }
    except Exception:
        return {"f_gamma": 0.0, "phi_initial": 0.0, "phi_final": 0.0,
                "completed_required": 0.0, "total_required": 0.0}


def _compute_p_process(event_log: EventLog, task_dict: dict) -> dict:
    """计算 P_process 及其分解值。"""
    try:
        pp_result = _process_scorer.compute(event_log, task_dict)
        return {
            "p_process": pp_result.p_process,
            "p_total_bonus": pp_result.total_bonus,
            "p_total_penalty": pp_result.total_penalty,
            "n_forbidden_steps": float(pp_result.n_forbidden_steps),
        }
    except Exception:
        return {"p_process": 0.0, "p_total_bonus": 0.0, "p_total_penalty": 0.0,
                "n_forbidden_steps": 0.0}


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict | None = None,
    **kwargs,
) -> dict:
    """OVAL reward function — R_task + I_shape*F + I_process*P - lambda_safe*C。

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
            "r_task": 0.0, "c_safety": 0.0, "j": 0.0,
            "f_gamma": 0.0, "p_process": 0.0,
            "n_events": 0.0,
            "n_model_tool_calls": float(extra_info.get("n_model_tool_calls", 0)),
            "n_exec_success": float(extra_info.get("n_exec_success", 0)),
            "error": "no audit events",
        }

    # ── 构建 EventLog ──
    session_id = extra_info.get("session_id", "")
    task_id = extra_info.get("task_id", "unknown")
    event_log = EventLog(events=audit_events, session_id=session_id, task_id=task_id)

    # ── 构建 task_dict ──
    task_dict = _build_task_dict(extra_info)

    # ── R_task ──
    try:
        r_result = _task_reward.compute(event_log, task_dict)
        r_task = r_result.r_task
        r_validity = r_result.r_validity
        r_coverage = r_result.r_coverage
        r_efficiency = r_result.r_efficiency
    except Exception:
        r_task = 0.0; r_validity = 0.0; r_coverage = 0.0; r_efficiency = 0.0

    # ── C_safety ──
    try:
        safety_result = _safety_verifier.verify(event_log)
        c_safety = safety_result.c_safety
        violations = safety_result.violation_types
    except Exception:
        c_safety = 0; violations = []

    # ── F_gamma (conditional on I_shape) ──
    fg_info = {"f_gamma": 0.0, "phi_final": 0.0}
    if _I_SHAPE:
        fg_info = _compute_f_gamma(event_log, task_dict)

    # ── P_process (conditional on I_process) ──
    pp_info = {"p_process": 0.0}
    if _I_PROCESS:
        pp_info = _compute_p_process(event_log, task_dict)

    # ── lambda_safe ──
    lambda_safe = float(extra_info.get("lambda_safe", _LAMBDA_SAFE_DEFAULT))
    # also try LambdaState file for dynamic updates
    if LambdaState is not None:
        try:
            state = LambdaState.load_or_default()
            lambda_safe = state.lambda_safe
        except Exception:
            pass

    # ── J = R_task + I_shape*lambda_shape*F + I_process*lambda_process*P - lambda_safe*C ──
    shape_term = _I_SHAPE * _LAMBDA_SHAPE * fg_info["f_gamma"]
    process_term = _I_PROCESS * _LAMBDA_PROCESS * pp_info["p_process"]
    j = r_task + shape_term + process_term - lambda_safe * c_safety

    n_model_calls = float(extra_info.get("n_model_tool_calls", 0))
    n_exec_ok = float(extra_info.get("n_exec_success", 0))
    n_events = len(audit_events)

    result = {
        "score": float(j),
        "r_task": float(r_task),
        "r_validity": float(r_validity),
        "r_coverage": float(r_coverage),
        "r_efficiency": float(r_efficiency),
        "c_safety": float(c_safety),
        "c_violations": ",".join(violations) if violations else "",
        "f_gamma": float(fg_info["f_gamma"]),
        "phi_final": float(fg_info.get("phi_final", 0.0)),
        "p_process": float(pp_info["p_process"]),
        "j": float(j),
        "lambda_safe": float(lambda_safe),
        "n_events": float(n_events),
        "n_model_tool_calls": n_model_calls,
        "n_exec_success": n_exec_ok,
    }

    # merge shape/process diag into result
    for k, v in fg_info.items():
        if k not in result:
            result[k] = float(v)
    for k, v in pp_info.items():
        if k not in result:
            result[k] = float(v)

    return result
