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

from typing import Any

from src.oval_mcp.envs.domain_adapter import get_adapter
from src.oval_mcp.verifier.safety import SafetyVerifier
from src.oval_mcp.verifier.events import EventLog, AuditEvent
from src.oval_mcp.rewards.task_reward import TaskReward
from src.oval_mcp.rewards.f_gamma import ProgressTracker
from src.oval_mcp.rewards.p_process import ProcessScorer

from src.oval_mcp.training.lambda_state import LambdaState, DEFAULT_STATE_PATH
from src.training.livemcp_hyperparams import get_config

# ── 模块级单例（延迟初始化，由 _get_cfg() 统一管理） ──
_safety_verifier = SafetyVerifier()
_progress_tracker = ProgressTracker()


def _get_cfg():
    """Read the single project-owned training configuration."""
    return get_config()

# 模块加载时解析一次配置
_cfg = _get_cfg()

# ── 消融开关（从统一配置读取，环境变量由 export_env() 保证一致性） ──
_I_SHAPE = _cfg.i_shape
_I_PROCESS = _cfg.i_process
_LAMBDA_SHAPE = _cfg.lambda_shape
_LAMBDA_PROCESS = _cfg.lambda_process
_GAMMA = _cfg.gamma
_REWARD_PROFILE = _cfg.reward_profile

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
    "alpha_eff": _cfg.alpha_eff,
    "beta_budget": _cfg.beta_budget,
})


class RewardIntegrityError(RuntimeError):
    """Infrastructure/data failure that must not become a model reward."""


def _dict_to_audit_event(d: dict) -> AuditEvent:
    """从序列化 dict 重构 AuditEvent。"""
    return AuditEvent(
        event_id=d.get("event_id", ""),
        session_id=d.get("session_id", ""),
        step=d.get("step", d.get("step_index", 0)),
        round_idx=d.get("round_idx", 0),
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
        tool_name_known=d.get("tool_name_known", False),
        state_changed=d.get("state_changed", False),
        latency_ms=d.get("latency_ms", 0),
        pre_state_status=d.get("pre_state_status", "available"),
        post_state_status=d.get("post_state_status", "available"),
        state_evidence_errors=d.get("state_evidence_errors", []),
    )


def _parse_audit_events(raw: Any) -> list[AuditEvent]:
    """从 extra_info 中解析 audit_events。"""
    import json as _json

    if raw is None:
        return []

    if isinstance(raw, str):
        try:
            raw = _json.loads(raw)
        except _json.JSONDecodeError as exc:
            raise RewardIntegrityError("audit_events contains invalid JSON") from exc

    if not isinstance(raw, list):
        raise RewardIntegrityError("audit_events must contain a list")

    events: list[AuditEvent] = []
    for item in raw:
        if isinstance(item, AuditEvent):
            events.append(item)
        elif isinstance(item, dict):
            try:
                events.append(_dict_to_audit_event(item))
            except Exception as exc:
                raise RewardIntegrityError("invalid audit event") from exc
        else:
            raise RewardIntegrityError(
                f"invalid audit event type: {type(item).__name__}"
            )
    return events


def _build_task_dict(extra_info: dict) -> dict:
    """从 extra_info 构建 task_dict，优先使用 ground_truth 中的 oracle 信息。"""
    import json as _json

    def _as_list(value) -> list:
        if value is None:
            return []
        if isinstance(value, list):
            return value
        if hasattr(value, "tolist"):
            converted = value.tolist()
            return converted if isinstance(converted, list) else [converted]
        if isinstance(value, tuple):
            return list(value)
        return [value]

    domain = extra_info.get("domain", "unknown")
    task_id = extra_info.get("task_id", "unknown")

    # Use saved oracle calls for accurate arg matching
    oracle_calls_raw = extra_info.get("oracle_calls", [])
    if not isinstance(oracle_calls_raw, str):
        raise RewardIntegrityError("oracle_calls must be canonical JSON text")
    try:
        oracle_calls = _json.loads(oracle_calls_raw)
    except (_json.JSONDecodeError, TypeError) as exc:
        raise RewardIntegrityError("oracle_calls contains invalid JSON") from exc
    if not isinstance(oracle_calls, list):
        raise RewardIntegrityError("oracle_calls JSON must contain a list")

    terminal_actions = [
        oc.get("action") for oc in oracle_calls
        if isinstance(oc, dict)
        and oc.get("action") in ("final_answer", "ask_clarification", "report_error")
    ]
    terminal_action = terminal_actions[-1] if terminal_actions else ""
    real_oracle_calls = [
        oc for oc in oracle_calls
        if isinstance(oc, dict) and oc.get("action", "tool_call") == "tool_call"
    ]
    round_contracts = _parse_round_contracts(extra_info)
    contract_tool_names: list[str] = []
    oracle_call_rounds: list[int] = []
    for expected_round_idx, contract in enumerate(round_contracts):
        if contract.get("round_idx") != expected_round_idx:
            raise RewardIntegrityError(
                f"task {task_id} has non-canonical round_idx at "
                f"round_contracts[{expected_round_idx}]"
            )
        names = contract.get("required_tools", [])
        if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
            raise RewardIntegrityError(
                f"task {task_id} has invalid required_tools in round "
                f"{expected_round_idx}"
            )
        contract_tool_names.extend(names)
        oracle_call_rounds.extend([expected_round_idx] * len(names))
    oracle_tool_names = [str(call.get("tool_name", "")) for call in real_oracle_calls]
    if contract_tool_names != oracle_tool_names:
        raise RewardIntegrityError(
            f"task {task_id} round contracts do not align with canonical oracle calls"
        )

    # Irrelevance/no-tool tasks require zero calls. Missing-function and
    # clarification trajectories may contain successful visible-prefix calls
    # before the hidden capability is discovered; preserve those oracle calls.
    scenario_type = extra_info.get("scenario_type", "")
    is_abstain_task = scenario_type in ("irrelevant", "no_tool_or_abstention")
    is_missing_function_terminal = (
        scenario_type in ("missing_function", "clarification_required")
        and terminal_action in ("ask_clarification", "report_error")
    )

    if is_abstain_task:
        required_tool_calls = []
    elif is_missing_function_terminal and not real_oracle_calls:
        # PROVE graceful give-up permits a missing-function trajectory to end
        # with ask_clarification or report_error without first manufacturing a
        # failed tool call. Visible-prefix calls, when present, are preserved
        # by the real_oracle_calls branch below.
        required_tool_calls = []
    elif real_oracle_calls:
        # Derive required calls only from the replayed Teacher oracle.
        required_tool_calls = [
            {"tool_name": oc["tool_name"], "arguments": oc.get("arguments", {})}
            for oc in real_oracle_calls
        ]
    else:
        raise RewardIntegrityError(
            f"task {task_id} has no canonical oracle tool calls"
        )
    required_call_rounds = (
        oracle_call_rounds if required_tool_calls else []
    )

    success_criteria_raw = extra_info.get("success_criteria", [])
    if not isinstance(success_criteria_raw, str):
        raise RewardIntegrityError("success_criteria must be canonical JSON text")
    try:
        success_criteria = _json.loads(success_criteria_raw)
    except _json.JSONDecodeError as exc:
        raise RewardIntegrityError("success_criteria contains invalid JSON") from exc
    if not isinstance(success_criteria, list):
        raise RewardIntegrityError("success_criteria JSON must contain a list")

    # P0-3 / P1-5 / P4b: scenario-aware terminal action whitelist.
    explicit_allowed = extra_info.get("allowed_terminal_actions")
    if isinstance(explicit_allowed, str):
        try:
            explicit_allowed = _json.loads(explicit_allowed)
        except _json.JSONDecodeError as exc:
            raise RewardIntegrityError(
                "allowed_terminal_actions contains invalid JSON"
            ) from exc
    elif hasattr(explicit_allowed, "tolist"):
        explicit_allowed = explicit_allowed.tolist()
    if not isinstance(explicit_allowed, list) or not explicit_allowed:
        raise RewardIntegrityError(
            f"task {task_id} is missing allowed_terminal_actions"
        )
    allowed_terminal = [str(action) for action in explicit_allowed]
    invalid_terminal = sorted(
        set(allowed_terminal)
        - {"final_answer", "ask_clarification", "report_error"}
    )
    if invalid_terminal:
        raise RewardIntegrityError(
            f"task {task_id} has invalid allowed terminal actions: {invalid_terminal}"
        )

    protected_by_resource_raw = extra_info.get("protected_fields_by_resource", {})
    if isinstance(protected_by_resource_raw, str):
        try:
            protected_by_resource = _json.loads(protected_by_resource_raw)
        except (_json.JSONDecodeError, TypeError) as exc:
            raise RewardIntegrityError(
                "protected_fields_by_resource contains invalid JSON"
            ) from exc
    elif isinstance(protected_by_resource_raw, dict):
        protected_by_resource = protected_by_resource_raw
    else:
        protected_by_resource = {}

    return {
        "task_id": task_id,
        "required_tool_calls": required_tool_calls,
        "required_call_rounds": required_call_rounds,
        "identity_policy": extra_info.get("identity_policy", "domain_defined"),
        "budget": extra_info.get("budget", 8),
        "allowed_terminal_actions": allowed_terminal,
        "success_criteria": success_criteria,
        "target_resource_ids": _as_list(extra_info.get("target_resource_ids", [])),
        "protected_resources": _as_list(extra_info.get("protected_resources", [])),
        "protected_fields": _as_list(extra_info.get("protected_fields", [])),
        "protected_fields_by_resource": protected_by_resource,
        "user_query": str(extra_info.get("user_query", "")),
        "scenario_type": scenario_type,
        "final_state": extra_info.get("final_state", {}),
        # P0-2: round contracts for per-round terminal validation
        "round_contracts": round_contracts,
        # P0-3: dependency edges for partial-order coverage.
        # Deserialised from JSON string (Parquet-safe) or parsed from
        # ground_truth.dependency_edges.  Used by TaskReward._match_required_calls
        # to accept non-dependent tool reorderings.
        "dependency_edges": _parse_dependency_edges(extra_info),
    }


def _parse_dependency_edges(extra_info: dict) -> list[tuple[int, int]]:
    """P0-3: parse dependency_edges into a list of (src_idx, dst_idx) tuples.

    Data-side _compute_dependency_edges produces [[src_idx, dst_idx], ...]
    where indices refer to positions in the flattened oracle_calls (tool_call
    actions only).  We keep the index-based format so that _build_task_dict
    and TaskReward can directly construct preds_by_idx without tool-name
    lookup (which fails on repeated tool names and non-name keys).

    Accepts JSON string (Parquet round-trip), list of tuple/list, or empty.
    Malformed present metadata is an integrity failure.
    """
    import json as _json
    raw = extra_info.get("dependency_edges")
    if raw is None:
        raw = extra_info.get("ground_truth", {}).get("dependency_edges")
    if raw is None:
        return []
    if isinstance(raw, str):
        try:
            raw = _json.loads(raw)
        except (_json.JSONDecodeError, TypeError) as exc:
            raise RewardIntegrityError("dependency_edges contains invalid JSON") from exc
    elif hasattr(raw, "tolist"):
        raw = raw.tolist()
    if not isinstance(raw, list):
        raise RewardIntegrityError("dependency_edges must contain a list")
    edges: list[tuple[int, int]] = []
    for edge in raw:
        if not isinstance(edge, (list, tuple)) or len(edge) != 2:
            raise RewardIntegrityError(f"invalid dependency edge: {edge!r}")
        try:
            src, dst = int(edge[0]), int(edge[1])
        except (ValueError, TypeError) as exc:
            raise RewardIntegrityError(f"invalid dependency edge: {edge!r}") from exc
        if src < 0 or dst < 0 or src == dst:
            raise RewardIntegrityError(f"invalid dependency edge: {edge!r}")
        edges.append((src, dst))
    return edges


def _parse_round_contracts(extra_info: dict) -> list[dict]:
    """Parse the required canonical per-round contract."""
    import json as _json
    raw = extra_info.get("round_contracts")
    if isinstance(raw, str):
        try:
            parsed = _json.loads(raw)
        except (_json.JSONDecodeError, TypeError) as exc:
            raise RewardIntegrityError("round_contracts contains invalid JSON") from exc
        raw = parsed
    elif hasattr(raw, "tolist"):
        raw = raw.tolist()
    if not isinstance(raw, list) or not raw:
        raise RewardIntegrityError("round_contracts must contain a non-empty list")
    if not all(isinstance(item, dict) for item in raw):
        raise RewardIntegrityError("round_contracts contains a non-object entry")
    return raw


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
    except Exception as exc:
        raise RewardIntegrityError(f"progress reward failed: {exc}") from exc


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
    except Exception as exc:
        raise RewardIntegrityError(f"process reward failed: {exc}") from exc


def _apply_round_contract_penalty(
    components: tuple[float, float, float, float],
    *,
    round_contract_ok: bool,
    reward_profile: str,
) -> tuple[float, float, float, float]:
    """Reject incomplete local trajectories before profile-specific scoring.

    This is a structural eligibility gate, not a sixth PROVE reward component:
    valid trajectories keep the five-component score unchanged in every
    profile; invalid terminal/round protocols receive no positive task signal.
    """
    if round_contract_ok:
        return components
    return (0.0, 0.0, 0.0, 0.0)


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

    if extra_info.get("trajectory_integrity_ok") is False:
        raise RewardIntegrityError(
            f"trajectory integrity failed: {extra_info.get('trajectory_errors', [])}"
        )

    # Merge ground_truth data (e.g., oracle_calls, success_criteria) into extra_info
    if isinstance(ground_truth, dict):
        for key in ("oracle_calls", "success_criteria"):
            if key not in extra_info and key in ground_truth:
                extra_info[key] = ground_truth[key]

    # ── 解析 audit_events ──
    audit_raw = extra_info.get("audit_events", [])
    audit_events = _parse_audit_events(audit_raw)

    if not audit_events:
        raise RewardIntegrityError("no audit events")

    # ── 构建 EventLog ──
    session_id = extra_info.get("session_id", "")
    task_id = extra_info.get("task_id", "unknown")
    event_log = EventLog(events=audit_events, session_id=session_id, task_id=task_id)

    # ── 构建 task_dict ──
    task_dict = _build_task_dict(extra_info)
    task_dict["apply_terminal_validity_penalty"] = (
        _REWARD_PROFILE == "oval_full"
    )
    task_dict["apply_identity_coverage_penalty"] = (
        _REWARD_PROFILE == "oval_full"
    )

    # ── Domain adapter ──
    domain = extra_info.get("domain", "calendar")
    try:
        domain_adapter = get_adapter(domain)
    except Exception as exc:
        raise RewardIntegrityError(
            f"domain adapter unavailable for {domain!r}: {exc}"
        ) from exc

    # ── R_task ──
    try:
        r_result = _task_reward.compute(event_log, task_dict, domain_adapter=domain_adapter)
        r_task = r_result.r_task
        r_validity = r_result.r_validity
        r_coverage = r_result.r_coverage
        r_efficiency = r_result.r_efficiency
    except Exception as exc:
        raise RewardIntegrityError(f"task reward failed: {exc}") from exc

    # P0-2: validate per-round terminals against round_contracts.
    r_round_ok, r_round_details = _validate_round_contracts(audit_events, task_dict)
    # An incomplete local conversation is not a valid trajectory for either
    # profile.  Valid trajectories retain the unchanged PROVE formula.
    r_task, r_validity, r_coverage, r_efficiency = (
        _apply_round_contract_penalty(
            (r_task, r_validity, r_coverage, r_efficiency),
            round_contract_ok=r_round_ok,
            reward_profile=_REWARD_PROFILE,
        )
    )

    # ── C_safety ──
    if _REWARD_PROFILE == "prove_baseline":
        c_safety = 0.0
        violations = []
    else:
        try:
            safety_result = _safety_verifier.verify(event_log, task_dict)
            c_safety = safety_result.c_safety
            violations = safety_result.violation_types
        except Exception as exc:
            raise RewardIntegrityError(f"safety verification failed: {exc}") from exc

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
    if _REWARD_PROFILE != "prove_baseline" and LambdaState is not None:
        try:
            state = LambdaState.load_or_default()
            lambda_safe = state.lambda_safe
        except Exception as exc:
            raise RewardIntegrityError(f"lambda state unavailable: {exc}") from exc
    if _REWARD_PROFILE == "prove_baseline":
        lambda_safe = 0.0

    # ── J = R_task + I_shape*lambda_shape*F + I_process*lambda_process*P - lambda_safe*C ──
    contract_multiplier = 1.0 if r_round_ok else 0.0
    shape_term = (
        contract_multiplier * _I_SHAPE * _LAMBDA_SHAPE * fg_info["f_gamma"]
    )
    process_term = (
        contract_multiplier * _I_PROCESS * _LAMBDA_PROCESS * pp_info["p_process"]
    )
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
        "reward_profile": _REWARD_PROFILE,
        "n_events": float(n_events),
        "n_model_tool_calls": n_model_calls,
        "n_exec_success": n_exec_ok,
        "r_round_ok": 1.0 if r_round_ok else 0.0,
        "r_round_details": r_round_details,
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


def _validate_round_contracts(audit_events: list, task_dict: dict) -> tuple[bool, str]:
    """P0-2: validate per-round terminals against round_contracts.

    Extracts terminal events from audit_events and checks that each round's
    terminal matches the contract's allowed_terminal_actions.

    Returns (ok, details_str).
    """
    contracts = task_dict.get("round_contracts", [])
    if not contracts:
        return False, "missing_round_contracts"

    terminal_events: list[tuple[int, str]] = []
    for ev in audit_events:
        if hasattr(ev, "action_type"):
            action = ev.action_type
            round_idx = int(getattr(ev, "round_idx", -1))
        elif isinstance(ev, dict):
            action = ev.get("action_type", "")
            round_idx = int(ev.get("round_idx", -1))
        else:
            continue
        if action == "tool_call":
            if round_idx < 0 or round_idx >= len(contracts):
                return False, f"tool event has invalid round_idx={round_idx}"
            continue
        if action in ("final_answer", "ask_clarification", "report_error"):
            terminal_events.append((round_idx, action))
        elif action == "contract_violation":
            return False, f"contract_violation at round {round_idx}"
        elif action == "round_tool_violation":
            return False, f"round_tool_violation at round {round_idx}"
        elif action == "round_tool_diagnostic":
            continue
        else:
            return False, f"invalid terminal action '{action}' at round {round_idx}"

    n_expected = len(contracts)
    n_actual = len(terminal_events)
    if n_actual != n_expected:
        direction = "fewer" if n_actual < n_expected else "more"
        return False, (
            f"terminal count mismatch: got {n_actual} terminals, "
            f"expected {n_expected} ({direction})"
        )

    terminals_by_round: dict[int, list[str]] = {}
    for round_idx, term in terminal_events:
        terminals_by_round.setdefault(round_idx, []).append(term)
    for i, contract in enumerate(contracts):
        round_terminals = terminals_by_round.get(i, [])
        if len(round_terminals) != 1:
            return False, (
                f"round {i}: expected exactly one terminal, "
                f"got {len(round_terminals)}"
            )
        term = round_terminals[0]
        allowed = contract.get("allowed_terminal_actions", [])
        if allowed and term not in allowed:
            return False, (
                f"round {i}: terminal '{term}' not in "
                f"allowed {allowed}"
            )
    return True, (
        "single_round" if n_expected == 1 else f"rounds_ok={n_actual}/{n_expected}"
    )
