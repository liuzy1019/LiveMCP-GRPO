"""OVAL reward function — verl custom_reward_function interface.

Entry point: compute_score(data_source, solution_str, ground_truth, extra_info=None)

Pipeline:
  1. Parse audit_events from extra_info (produced by LiveMCPOvalLoop)
  2. Build EventLog, get DomainAdapter
  3. TaskReward → R_task
  4. SafetyVerifier → C_safety
  5. ProgressTracker → F_gamma (via DomainAdapter.evaluate_event)
  6. ProcessScorer → P_process (via DomainAdapter.evaluate_event)
  7. LambdaState → lambda_safe
  8. J = R_task + lambda_shape * F + lambda_process * P - lambda_safe * C
"""

import os
from typing import Any

from src.oval_mcp.envs.domain_adapter import get_adapter
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

try:
    from src.training.livemcp_hyperparams import get_config
except ImportError:
    get_config = None  # type: ignore

# ── 模块级单例（延迟初始化，由 _get_cfg() 统一管理） ──
_safety_verifier = SafetyVerifier()
_task_reward = TaskReward()
_progress_tracker = ProgressTracker()


def _get_cfg():
    """获取配置：优先从 LiveMCPHyperparams，fallback 到环境变量。"""
    if get_config is not None:
        return get_config()
    # Fallback: 从环境变量手动构建（兼容未安装 livemcp_hyperparams 的场景）
    from dataclasses import dataclass
    @dataclass
    class _FallbackCfg:
        i_shape: int = int(os.environ.get("OVAL_I_SHAPE", "0"))
        i_process: int = int(os.environ.get("OVAL_I_PROCESS", "1"))
        lambda_shape: float = float(os.environ.get("OVAL_LAMBDA_SHAPE", "0.5"))
        lambda_process: float = float(os.environ.get("OVAL_LAMBDA_PROCESS", "0.3"))
        gamma: float = float(os.environ.get("OVAL_GAMMA", "1.0"))
        lambda_safe_default: float = 1.0
        p_max: float = float(os.environ.get("OVAL_P_MAX", "0.3"))
        w_val: float = 0.5
        w_cov: float = 0.5
        w_eff: float = 0.15
        w_name: float = 0.2
        w_arg: float = 0.1
        w_struct: float = 0.6
        w_exec: float = 0.4
        alpha_eff: float = 0.3
        beta_budget: float = 0.3
    return _FallbackCfg()

# 模块加载时解析一次配置
_cfg = _get_cfg()

# ── 消融开关（从统一配置读取，环境变量由 export_env() 保证一致性） ──
_I_SHAPE = _cfg.i_shape
_I_PROCESS = _cfg.i_process
_LAMBDA_SHAPE = _cfg.lambda_shape
_LAMBDA_PROCESS = _cfg.lambda_process
_GAMMA = _cfg.gamma

_LAMBDA_SAFE_DEFAULT = _cfg.lambda_safe_default

# ── P_process scorer（可用 OVAL_P_MAX 环境变量覆盖） ──
_process_scorer = ProcessScorer(p_max=_cfg.p_max)

# ── 使用配置中的权重初始化 TaskReward（而非硬编码 DEFAULT_WEIGHTS） ──
_task_reward = TaskReward(weights={
    "w_val": _cfg.w_val,
    "w_cov": _cfg.w_cov,
    "w_eff": _cfg.w_eff,
    "w_name": _cfg.w_name,
    "w_arg": _cfg.w_arg,
    "w_struct": _cfg.w_struct,
    "w_exec": _cfg.w_exec,
    "alpha_eff": _cfg.alpha_eff,
    "beta_budget": _cfg.beta_budget,
})


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
    """从 extra_info 构建 task_dict，优先使用 ground_truth 中的 oracle 信息。"""
    import json as _json

    domain = extra_info.get("domain", "unknown")
    task_id = extra_info.get("task_id", "unknown")
    required_tools = extra_info.get("required_tools", [])
    if isinstance(required_tools, str):
        required_tools = [t.strip() for t in required_tools.split(",") if t.strip()]

    # Use saved oracle calls for accurate arg matching
    oracle_calls_raw = extra_info.get("oracle_calls", [])
    # P1-11: oracle_calls may be a JSON string (to prevent pyarrow struct
    # unification) or a legacy list[dict]. Parse accordingly.
    if isinstance(oracle_calls_raw, str):
        try:
            oracle_calls = _json.loads(oracle_calls_raw)
        except (_json.JSONDecodeError, TypeError):
            oracle_calls = []
    elif isinstance(oracle_calls_raw, list):
        oracle_calls = oracle_calls_raw
    elif hasattr(oracle_calls_raw, "tolist"):
        oracle_calls = oracle_calls_raw.tolist()
    else:
        oracle_calls = []
    if not isinstance(oracle_calls, list):
        oracle_calls = []

    # P0-3: Detect clarification-type oracle calls. Clarification is a
    # terminal action, not a real tool_call — it should NOT enter
    # required_tool_calls (otherwise the model is penalised for not
    # calling a tool literally named "" or "ask_clarification").
    has_clarification_oracle = any(
        isinstance(oc, dict) and oc.get("action") == "clarification"
        for oc in oracle_calls
    )
    real_oracle_calls = [
        oc for oc in oracle_calls
        if not (isinstance(oc, dict) and oc.get("action") == "clarification")
    ]

    if real_oracle_calls:
        required_tool_calls = [
            {"tool_name": oc["tool_name"], "arguments": oc.get("arguments", {})}
            for oc in real_oracle_calls
        ]
    else:
        # Fallback: empty arguments
        required_tool_calls = [
            {"tool_name": tn, "arguments": {}} for tn in required_tools
        ]

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

    # P0-1 fix: success_criteria may be a JSON string (post-Parquet
    # roundtrip) or a list. Normalise to list[dict].
    success_criteria_raw = extra_info.get("success_criteria", [])
    if isinstance(success_criteria_raw, str):
        try:
            success_criteria = _json.loads(success_criteria_raw)
        except _json.JSONDecodeError:
            success_criteria = []
    elif isinstance(success_criteria_raw, list):
        success_criteria = success_criteria_raw
    else:
        success_criteria = []

    # P0-3 / P1-5: scenario-aware terminal action whitelist.
    scenario_type = extra_info.get("scenario_type", "")
    has_missing_func = bool(extra_info.get("has_missing_function"))
    if has_missing_func or scenario_type == "missing_function":
        # Required tool was hidden → only correct response is report_error
        allowed_terminal = ["report_error"]
    elif scenario_type == "irrelevant":
        # Query unrelated to any available tool → only correct response is report_error
        allowed_terminal = ["report_error"]
    elif has_clarification_oracle:
        # Oracle solved this with ask_clarification → it is the only
        # correct terminal action.
        allowed_terminal = ["ask_clarification"]
    else:
        # Normal task: model should give a final_answer
        allowed_terminal = ["final_answer"]

    return {
        "task_id": task_id,
        "required_tool_calls": required_tool_calls,
        "identity_policy": "preserve" if domain == "calendar" else "create_new",
        "budget": extra_info.get("budget", 8),
        "outcome_assertions": assertions,
        "allowed_terminal_actions": allowed_terminal,
        "success_criteria": success_criteria,
    }


def _compute_f_gamma(event_log: EventLog, task_dict: dict, domain_adapter=None) -> dict:
    """计算 F_gamma 及其分解值。"""
    try:
        fg_result = _progress_tracker.compute(event_log, task_dict, gamma=_GAMMA, domain_adapter=domain_adapter)
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


def _compute_p_process(event_log: EventLog, task_dict: dict, domain_adapter=None) -> dict:
    """计算 P_process 及其分解值。"""
    try:
        pp_result = _process_scorer.compute(event_log, task_dict, domain_adapter=domain_adapter)
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

    # Merge ground_truth data (e.g., oracle_calls, success_criteria) into extra_info
    if isinstance(ground_truth, dict):
        for key in ("oracle_calls", "success_criteria", "required_tools"):
            if key not in extra_info and key in ground_truth:
                extra_info[key] = ground_truth[key]

    # ── 解析 audit_events ──
    audit_raw = extra_info.get("audit_events", [])
    audit_events = _parse_audit_events(audit_raw)

    if not audit_events:
        return {
            "score": 0.0,
            "r_task": 0.0, "r_validity": 0.0, "r_coverage": 0.0, "r_efficiency": 0.0,
            "c_safety": 0.0, "c_violations": "",
            "f_gamma": 0.0, "phi_final": 0.0,
            "p_process": 0.0,
            "j": 0.0, "lambda_safe": 1.0,
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

    # ── Domain adapter ──
    domain = extra_info.get("domain", "calendar")
    try:
        domain_adapter = get_adapter(domain)
    except Exception:
        domain_adapter = None

    # ── R_task ──
    try:
        r_result = _task_reward.compute(event_log, task_dict, domain_adapter=domain_adapter)
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
        fg_info = _compute_f_gamma(event_log, task_dict, domain_adapter=domain_adapter)

    # ── P_process (conditional on I_process) ──
    pp_info = {"p_process": 0.0}
    if _I_PROCESS:
        pp_info = _compute_p_process(event_log, task_dict, domain_adapter=domain_adapter)

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
        "error": "",
    }

    # merge shape/process diag into result
    for k, v in fg_info.items():
        if k not in result:
            result[k] = float(v)
    for k, v in pp_info.items():
        if k not in result:
            result[k] = float(v)

    return result
