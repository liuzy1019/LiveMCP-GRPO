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

from dataclasses import dataclass
from typing import Any

from src.oval_mcp.envs.domain_adapter import get_adapter
from src.oval_mcp.verifier.safety import SafetyVerifier
from src.oval_mcp.verifier.events import EventLog, AuditEvent
from src.oval_mcp.rewards.task_reward import TaskReward
from src.oval_mcp.rewards.f_gamma import ProgressTracker
from src.oval_mcp.rewards.p_process import ProcessScorer

from src.oval_mcp.training.lambda_state import LambdaState
from src.training.hyperparams import get_config
from src.live_mcp.artifact.reward_task import (
    ArtifactIntegrityError,
)
from src.live_mcp.artifact.validation import validate_artifact_contract

_safety_verifier = SafetyVerifier()
_progress_tracker = ProgressTracker()


@dataclass(frozen=True)
class RewardRuntime:
    """One explicit reward invocation context.

    Configuration is resolved when scoring starts, never while importing the
    module.  This keeps read-only imports pure and prevents test/run profile
    changes from being frozen in a process-global singleton.
    """

    cfg: Any
    task_reward: TaskReward
    process_scorer: ProcessScorer

    @classmethod
    def from_environment(cls) -> "RewardRuntime":
        cfg = get_config()
        return cls(
            cfg=cfg,
            task_reward=TaskReward(
                weights={
                    "w_val": cfg.w_val,
                    "w_cov": cfg.w_cov,
                    "w_eff": cfg.w_eff,
                    "w_name": cfg.w_name,
                    "w_arg": cfg.w_arg,
                    "alpha_eff": cfg.alpha_eff,
                    "beta_budget": cfg.beta_budget,
                },
                reward_profile=cfg.reward_profile,
            ),
            process_scorer=ProcessScorer(p_max=cfg.p_max),
        )


RewardIntegrityError = ArtifactIntegrityError


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




def _compute_f_gamma(
    event_log: EventLog, task_dict: dict, *, gamma: float,
    domain_adapter=None,
) -> dict:
    """计算 F_gamma 及其分解值。"""
    try:
        fg_result = _progress_tracker.compute(
            event_log, task_dict, gamma=gamma, domain_adapter=domain_adapter,
        )
        return {
            "f_gamma": fg_result.f_gamma,
            "phi_initial": fg_result.phi_initial,
            "phi_final": fg_result.phi_final,
            "completed_required": float(fg_result.completed_required_states),
            "total_required": float(fg_result.total_required_states),
        }
    except Exception as exc:
        raise RewardIntegrityError(f"progress reward failed: {exc}") from exc


def _compute_p_process(
    event_log: EventLog, task_dict: dict, *, process_scorer: ProcessScorer,
    domain_adapter=None,
) -> dict:
    """计算 P_process 及其分解值。"""
    try:
        pp_result = process_scorer.compute(
            event_log, task_dict, domain_adapter=domain_adapter,
        )
        return {
            "p_process": pp_result.p_process,
            "p_total_bonus": pp_result.total_bonus,
            "p_total_penalty": pp_result.total_penalty,
            "n_forbidden_steps": float(pp_result.n_forbidden_steps),
        }
    except Exception as exc:
        raise RewardIntegrityError(f"process reward failed: {exc}") from exc


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
    runtime = RewardRuntime.from_environment()
    cfg = runtime.cfg
    reward_profile = cfg.reward_profile

    runtime_reward_profile = str(extra_info.get("reward_profile") or "")
    if not runtime_reward_profile:
        raise RewardIntegrityError("rollout evidence is missing reward_profile")
    if runtime_reward_profile != reward_profile:
        raise RewardIntegrityError(
            "rollout/reward profile mismatch: "
            f"rollout={runtime_reward_profile!r}, reward={reward_profile!r}"
        )

    if extra_info.get("trajectory_integrity_ok") is False:
        raise RewardIntegrityError(
            f"trajectory integrity failed: {extra_info.get('trajectory_errors', [])}"
        )

    # The verl ground_truth payload is a transport mirror, not a fallback
    # oracle. Reward must score the exact task contract validated by rollout.
    try:
        task_dict = validate_artifact_contract(
            extra_info,
            require_training=True,
            ground_truth=ground_truth,
        )
    except RuntimeError as exc:
        raise RewardIntegrityError(str(exc)) from exc

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
    task_dict["apply_terminal_validity_penalty"] = (
        reward_profile == "oval_full"
    )
    task_dict["apply_identity_coverage_penalty"] = (
        reward_profile == "oval_full"
    )

    # ── Domain adapter ──
    domain = str(extra_info.get("domain") or "")
    if not domain:
        raise RewardIntegrityError("rollout evidence is missing domain")
    domain_adapter = None
    if reward_profile == "oval_full":
        try:
            domain_adapter = get_adapter(domain)
        except Exception as exc:
            raise RewardIntegrityError(
                f"domain adapter unavailable for {domain!r}: {exc}"
            ) from exc

    # ── R_task ──
    try:
        r_result = runtime.task_reward.compute(
            event_log, task_dict, domain_adapter=domain_adapter,
        )
        r_task = r_result.r_task
        r_validity = r_result.r_validity
        r_coverage = r_result.r_coverage
        r_efficiency = r_result.r_efficiency
    except Exception as exc:
        raise RewardIntegrityError(f"task reward failed: {exc}") from exc

    # P0-2: validate per-round terminals against round_contracts.
    r_round_ok, r_round_details = _validate_round_contracts(audit_events, task_dict)
    # The five task-reward components preserve partial credit for malformed or
    # failed calls. A terminal/round mismatch remains diagnostic and suppresses
    # optional positive shaping below, but does not erase scored tool evidence.
    # No-tool tasks retain their binary terminal predicate inside TaskReward.

    # ── C_safety ──
    if reward_profile == "prove_baseline":
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
    if cfg.i_shape:
        fg_info = _compute_f_gamma(
            event_log, task_dict, gamma=cfg.gamma,
            domain_adapter=domain_adapter,
        )

    # ── P_process (conditional on I_process) ──
    pp_info = {"p_process": 0.0}
    if cfg.i_process:
        pp_info = _compute_p_process(
            event_log, task_dict, process_scorer=runtime.process_scorer,
            domain_adapter=domain_adapter,
        )

    # ── lambda_safe ──
    lambda_safe = float(extra_info.get("lambda_safe", cfg.lambda_safe_default))
    # also try LambdaState file for dynamic updates
    if reward_profile != "prove_baseline" and LambdaState is not None:
        try:
            state = LambdaState.load_or_default(path=cfg.lambda_state_path)
            lambda_safe = state.lambda_safe
        except Exception as exc:
            raise RewardIntegrityError(f"lambda state unavailable: {exc}") from exc
    if reward_profile == "prove_baseline":
        lambda_safe = 0.0

    # ── J = R_task + I_shape*lambda_shape*F + I_process*lambda_process*P - lambda_safe*C ──
    contract_multiplier = 1.0 if r_round_ok else 0.0
    shape_term = (
        contract_multiplier * cfg.i_shape * cfg.lambda_shape * fg_info["f_gamma"]
    )
    process_term = (
        contract_multiplier * cfg.i_process * cfg.lambda_process * pp_info["p_process"]
    )
    j = r_task + shape_term + process_term - lambda_safe * c_safety

    n_model_calls = float(extra_info.get("n_model_tool_calls", 0))
    n_exec_ok = float(extra_info.get("n_exec_success", 0))
    n_events = len(audit_events)

    result = {
        "score": float(j),
        "r_task": float(r_task),
        "r_validity": float(r_validity),
        "r_name_exists": float(r_result.r_name_exists),
        "r_args_present": float(r_result.r_args_present),
        "r_execution": float(r_result.r_execution),
        "r_coverage": float(r_coverage),
        "r_efficiency": float(r_efficiency),
        "r_name": float(r_result.r_name),
        "r_arg": float(r_result.r_arg),
        "reward_n_required_calls": float(r_result.n_required_calls),
        "reward_completed_predicates": float(r_result.completed_predicates),
        "reward_total_predicates": float(r_result.total_predicates),
        "reward_aligned_calls": float(r_result.aligned_calls),
        "reward_is_no_tool_task": 1.0 if r_result.is_no_tool_task else 0.0,
        "c_safety": float(c_safety),
        "c_violations": ",".join(violations) if violations else "",
        "f_gamma": float(fg_info["f_gamma"]),
        "phi_final": float(fg_info.get("phi_final", 0.0)),
        "p_process": float(pp_info["p_process"]),
        "j": float(j),
        "lambda_safe": float(lambda_safe),
        "reward_profile": reward_profile,
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
