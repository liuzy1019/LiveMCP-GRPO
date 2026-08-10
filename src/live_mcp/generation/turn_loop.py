"""One-round Action Teacher execution and bounded recovery FSM."""

from __future__ import annotations

import json
from typing import Any

from loguru import logger

from src.live_mcp.fsm import ConversationFSM, FSMStateGroup
from src.live_mcp.protocol.observation import tool_result_envelope
from src.live_mcp.registry.tool_semantics import (
    tool_call_invalidated_by_state_changes,
    unresolved_failed_tool_names,
)
from src.live_mcp.types import OracleCall
from src.live_mcp.generation.tool_resolution import fuzzy_match_tool as _fuzzy_match_tool

def run_turn_loop(
    self,
    teacher,
    current_query: str,
    server_tools: list[dict],
    server_name: str,
    session_id: str,
    difficulty: str,
    round_idx: int,
    turn_budget: int = 8,
    reference_date: str = "",
    max_calls_this_round: int = 0,
    chain_context: dict[str, Any] | None = None,
    blocked_tools: set[str] | None = None,
    missing_function_contract: bool = False,
    allowed_missing_mutations: set[str] | None = None,
    prior_execution_history: list[dict[str, Any]] | None = None,
    conversation_context: list[dict[str, Any]] | None = None,
    allow_direct_answer: bool = False,
    dependency_plan: list[dict[str, str]] | None = None,
    fsm: ConversationFSM | None = None,
) -> tuple[list, list[dict], list[Any], set[str], list[OracleCall], list[Any]]:
    """Run one conversation round of teacher-driven tool execution.

    max_calls_this_round: optional diagnostic/ablation cap on real tool
    calls. Baseline passes 0 (no per-round cap).

    chain_context: live-probed entity values for hallucination prevention
    in decide_action. Extracted from _extract_chain_context in generate_one.

    Returns (oracle_calls, execution_history, oracle_observations, required_tools).
    oracle_observations is 1:1 aligned with oracle_calls — each entry is the
    raw tool observation dict for the corresponding oracle call, or {} for
    terminal actions (ask_clarification, final_answer, report_error).
    """
    from src.live_mcp.types import ToolCall

    oracle_calls: list = []
    oracle_observations: list[Any] = []
    execution_history: list[dict[str, Any]] = []
    prior_history = list(prior_execution_history or [])
    required_tools: set[str] = set()
    attempt_calls: list[OracleCall] = []
    attempt_observations: list[Any] = []

    def _record_attempt(
        name: str,
        arguments: dict[str, Any],
        observation: Any,
        success: bool,
        owner_domain: str = "",
    ) -> None:
        attempt_calls.append(OracleCall(
            tool_name=name,
            arguments=dict(arguments),
            server_name=owner_domain or server_name,
            expected_success=bool(success),
        ))
        attempt_observations.append(observation if observation is not None else {})

    def _owner_domain(name: str) -> str:
        for schema in server_tools:
            if schema.get("name") == name:
                return str(schema.get("_server_name") or server_name)
        return server_name

    def _trace(stage: str, **payload: Any) -> None:
        recorder = getattr(teacher, "record_environment_event", None)
        if callable(recorder):
            recorder(stage, **payload)

    def _execution_event(
        name: str,
        arguments: dict[str, Any],
        result: Any,
        owner_domain: str = "",
    ) -> dict[str, Any]:
        """Normalize the real MCP result consumed by the next Agent step."""
        event = {
            "round_idx": round_idx,
            "tool_name": name,
            "server_name": owner_domain or _owner_domain(name),
            "arguments": dict(arguments),
            **tool_result_envelope(result),
        }
        delta_paths = getattr(result, "metadata", {}).get(
            "state_delta_paths", [],
        )
        if delta_paths:
            # Private audit metadata: _format_history intentionally does
            # not expose these internal state paths to the Teacher.
            event["state_delta_paths"] = list(delta_paths)
        return event

    def _add_oracle(call: OracleCall) -> bool:
        """Append the state-machine oracle without intra-trace pruning."""
        oracle_calls.append(call)
        return True

    max_tool_calls = max(1, int(turn_budget))

    attempt = 0          # raw LLM call count (for temperature scaling)
    _turn: int = 0       # real turn count (tool exec + terminal)
    tool_calls_dispatched = 0
    no_progress_rejections = 0

    # The extra iteration is reserved for a terminal decision after the
    # last permitted tool dispatch.  A tool call in that slot fails the
    # candidate closed instead of silently executing beyond the ceiling.
    while _turn <= max_tool_calls:
        if fsm is not None:
            fsm.transition(
                FSMStateGroup.TURN,
                "teacher_decision_requested",
                round_idx=round_idx,
                action_idx=_turn,
            )
        try:
            action = teacher.decide_action(
                tool_schemas=server_tools,
                user_query=current_query,
                execution_history=prior_history + execution_history,
                attempt=attempt,
                difficulty=difficulty,
                reference_date=reference_date,
                chain_context=chain_context,
                conversation_context=conversation_context,
                blocked_tools=blocked_tools,
                missing_function=missing_function_contract,
                allowed_missing_mutations=allowed_missing_mutations,
                allow_direct_answer=allow_direct_answer,
                dependency_plan=dependency_plan,
                round_idx=round_idx,
            )
        except RuntimeError:
            logger.debug(
                "_run_turn_loop: decide_action exhausted retries; "
                "breaking turn loop."
            )
            break

        if action.action == "ask_clarification":
            _trace(
                "terminal_action", round_idx=round_idx,
                action="ask_clarification", text=action.text,
            )
            if fsm is not None:
                fsm.transition(
                    FSMStateGroup.RESPONSE,
                    "teacher_terminal",
                    action="ask_clarification",
                )
            _add_oracle(OracleCall(
                tool_name="ask_clarification",
                arguments={"question": action.text},
                action="ask_clarification",
            ))
            oracle_observations.append({})
            break

        if action.action in ("final_answer", "report_error"):
            if action.action == "final_answer":
                unresolved = unresolved_failed_tool_names(execution_history)
                if unresolved:
                    raise RuntimeError(
                        "Teacher emitted final_answer with unresolved failed "
                        f"actions in round {round_idx}: {sorted(unresolved)}"
                    )
            _trace(
                "terminal_action", round_idx=round_idx,
                action=action.action, text=action.text,
            )
            if fsm is not None:
                fsm.transition(
                    FSMStateGroup.RESPONSE,
                    "teacher_terminal",
                    action=action.action,
                )
            _add_oracle(OracleCall(
                tool_name=action.action,
                arguments={"text": action.text},
                action=action.action,
            ))
            oracle_observations.append({})
            break

        if action.action != "tool_call" or not action.tool_name:
            continue

        tool_name = action.tool_name
        tool_name = _fuzzy_match_tool(tool_name, {t["name"] for t in server_tools}) or tool_name

        if tool_calls_dispatched >= max_tool_calls:
            raise RuntimeError(
                "Teacher reached the tool-call safety budget and did not "
                "emit a terminal response"
            )

        candidate_signature = json.dumps(
            {
                "tool_name": tool_name,
                "server_name": _owner_domain(tool_name),
                "arguments": dict(action.arguments),
            },
            sort_keys=True,
            ensure_ascii=False,
            default=str,
        )
        repeated_no_progress_call = False
        complete_history = prior_history + execution_history
        for prior_idx, prior in enumerate(complete_history):
            prior_signature = json.dumps(
                {
                    "tool_name": prior.get("tool_name"),
                    "server_name": str(
                        prior.get("server_name") or server_name
                    ),
                    "arguments": prior.get("arguments", {}),
                },
                sort_keys=True,
                ensure_ascii=False,
                default=str,
            )
            if not (
                prior.get("success") is True
                and not bool(prior.get("state_changed"))
                and prior_signature == candidate_signature
            ):
                continue
            if tool_call_invalidated_by_state_changes(
                tool_name,
                action.arguments,
                _owner_domain(tool_name),
                complete_history[prior_idx + 1:],
            ):
                continue
            repeated_no_progress_call = True
            break
        if repeated_no_progress_call:
            no_progress_rejections += 1
            _trace(
                "no_progress_action_rejected",
                round_idx=round_idx,
                tool_name=tool_name,
                arguments=dict(action.arguments),
                rejection_count=no_progress_rejections,
            )
            attempt += 1
            if no_progress_rejections >= 3:
                raise RuntimeError(
                    "Teacher repeated a successful no-progress "
                    "tool call after three pre-dispatch rejections"
                )
            continue

        execution_domain = _owner_domain(tool_name)
        if fsm is not None:
            fsm.transition(
                FSMStateGroup.TOOL_EXECUTION,
                "tool_call_dispatched",
                tool_name=tool_name,
                owner_domain=execution_domain,
            )
        result = self.executor.execute(
            session_id,
            ToolCall(tool_name, dict(action.arguments), call_id=f"sm_{_turn}"),
            blocked_tools=blocked_tools,
            domain=execution_domain,
        )
        tool_calls_dispatched += 1
        _record_attempt(
            tool_name, action.arguments, result.observation, result.success, execution_domain,
        )
        _trace(
            "mcp_feedback",
            **_execution_event(
                tool_name, dict(action.arguments), result, execution_domain,
            ),
        )

        if fsm is not None:
            fsm.transition(
                FSMStateGroup.RESPONSE,
                "tool_outcome",
                tool_name=tool_name,
                outcome=result.execution_status,
            )

        if result.execution_status == "FAILURE":
            logger.debug(
                f"_run_turn_loop: tool '{tool_name}' execution failed for "
                f"{server_name} (error_type={result.error_type}, "
                f"msg={result.error_message[:80]})"
            )
            execution_history.append(
                _execution_event(
                    tool_name, dict(action.arguments), result, execution_domain,
                )
            )
            failed_tool = tool_name
            failed_arguments = dict(action.arguments)
            failed_result = result
            failed_domain = execution_domain
            recovery_terminal = False
            recovery_deferred = False
            recovery_resolved = False

            # Every failed recovery execution re-enters Recovery.  The old
            # implementation handled only the first failure and silently
            # returned a failed retry/alternative to the ordinary Action
            # Teacher path, which violated the state-machine contract.
            for recovery_attempt in range(3):
                recovery = teacher.decide_recovery(
                    last_tool_name=failed_tool,
                    last_arguments=failed_arguments,
                    error_observation={
                        "error_type": str(
                            failed_result.error_type or "EXECUTION_ERROR"
                        ),
                        "error_message": str(
                            failed_result.error_message or ""
                        ),
                        "observation": failed_result.observation,
                    },
                    tool_schemas=server_tools,
                    execution_history=prior_history + execution_history,
                )
                rec_action = recovery.get("action", "give_up")
                _trace(
                    "parsed_recovery",
                    round_idx=round_idx,
                    failed_tool=failed_tool,
                    recovery_attempt=recovery_attempt,
                    recovery=recovery,
                )

                if rec_action == "give_up":
                    completed_mutations = {
                        str(event.get("tool_name") or "")
                        for event in execution_history
                        if (
                            isinstance(event, dict)
                            and event.get("success") is True
                        )
                    }
                    pending_independent_mutations = (
                        set(allowed_missing_mutations or set())
                        - completed_mutations
                        - {failed_tool}
                    )
                    if pending_independent_mutations:
                        _trace(
                            "recovery_give_up_deferred",
                            round_idx=round_idx,
                            failed_tool=failed_tool,
                            pending_independent_mutations=sorted(
                                pending_independent_mutations
                            ),
                        )
                        recovery_deferred = True
                        break
                    reason = str(
                        recovery.get("reason")
                        or "The request cannot be completed with the available tools and state."
                    )
                    if fsm is not None:
                        fsm.transition(
                            FSMStateGroup.RESPONSE,
                            "recovery_give_up",
                            action="report_error",
                        )
                    _add_oracle(OracleCall(
                        tool_name="report_error",
                        arguments={"text": reason},
                        action="report_error",
                    ))
                    oracle_observations.append({})
                    _trace(
                        "terminal_action", round_idx=round_idx,
                        action="report_error", text=reason,
                    )
                    recovery_terminal = True
                    break

                retry_tool = failed_tool
                retry_arguments = failed_arguments
                retry_domain = failed_domain
                recovery_kind = "retry"
                dispatch_event = "recovery_retry_dispatched"
                if rec_action in ("retry", "retry_same"):
                    retry_arguments = dict(
                        recovery.get("corrected_args", failed_arguments)
                    )
                elif rec_action == "retry_alt":
                    retry_tool = str(recovery.get("tool_name") or "")
                    if retry_tool not in {
                        str(tool.get("name") or "")
                        for tool in server_tools
                    }:
                        _trace(
                            "recovery_alternative_rejected",
                            round_idx=round_idx,
                            failed_tool=failed_tool,
                            alternative_tool=retry_tool,
                            reason="tool_not_visible",
                        )
                        continue
                    retry_arguments = dict(recovery.get("arguments", {}))
                    retry_domain = _owner_domain(retry_tool)
                    recovery_kind = "alternative"
                    dispatch_event = "recovery_alternative_dispatched"
                else:
                    _trace(
                        "recovery_action_rejected",
                        round_idx=round_idx,
                        failed_tool=failed_tool,
                        recovery_action=rec_action,
                    )
                    continue

                if fsm is not None:
                    fsm.transition(
                        FSMStateGroup.TOOL_EXECUTION,
                        dispatch_event,
                        tool_name=retry_tool,
                        owner_domain=retry_domain,
                    )
                if tool_calls_dispatched >= max_tool_calls:
                    reason = (
                        "The request could not be completed within the "
                        "bounded recovery budget."
                    )
                    if fsm is not None:
                        fsm.transition(
                            FSMStateGroup.RESPONSE,
                            "recovery_budget_exhausted",
                            action="report_error",
                        )
                    _add_oracle(OracleCall(
                        tool_name="report_error",
                        arguments={"text": reason},
                        action="report_error",
                    ))
                    oracle_observations.append({})
                    _trace(
                        "terminal_action", round_idx=round_idx,
                        action="report_error", text=reason,
                    )
                    recovery_terminal = True
                    break
                retry_result = self.executor.execute(
                    session_id,
                    ToolCall(
                        retry_tool,
                        retry_arguments,
                        call_id=(
                            f"sm_recover_{_turn}_{recovery_attempt}"
                        ),
                    ),
                    blocked_tools=blocked_tools,
                    domain=retry_domain,
                )
                tool_calls_dispatched += 1
                _record_attempt(
                    retry_tool, retry_arguments,
                    retry_result.observation, retry_result.success,
                    retry_domain,
                )
                execution_history.append(
                    _execution_event(
                        retry_tool, retry_arguments, retry_result,
                        retry_domain,
                    )
                )
                _trace(
                    "mcp_feedback", recovery=recovery_kind,
                    **execution_history[-1],
                )
                if fsm is not None:
                    fsm.transition(
                        FSMStateGroup.RESPONSE,
                        "tool_outcome",
                        tool_name=retry_tool,
                        outcome=retry_result.execution_status,
                        recovery=recovery_kind,
                    )
                if retry_result.execution_status != "FAILURE":
                    required_tools.add(retry_tool)
                    if _add_oracle(OracleCall(
                        tool_name=retry_tool,
                        arguments=retry_arguments,
                        server_name=retry_domain,
                    )):
                        oracle_observations.append(
                            retry_result.observation
                            if retry_result.observation is not None else {}
                        )
                    recovery_resolved = True
                    break

                failed_tool = retry_tool
                failed_arguments = retry_arguments
                failed_result = retry_result
                failed_domain = retry_domain
            else:
                reason = (
                    "The request could not be completed after repeated "
                    "recovery attempts."
                )
                if fsm is not None:
                    fsm.transition(
                        FSMStateGroup.RESPONSE,
                        "recovery_exhausted",
                        action="report_error",
                    )
                _add_oracle(OracleCall(
                    tool_name="report_error",
                    arguments={"text": reason},
                    action="report_error",
                ))
                oracle_observations.append({})
                _trace(
                    "terminal_action", round_idx=round_idx,
                    action="report_error", text=reason,
                )
                recovery_terminal = True

            _turn += 1
            attempt += 1
            if recovery_terminal:
                break
            # After recovery adds an ask_clarification terminal,
            # oracle call, check if we hit the per-round tool-call limit.
            if max_calls_this_round > 0:
                real_this_round = sum(
                    1 for oc in oracle_calls if oc.action == "tool_call"
                )
                if real_this_round >= max_calls_this_round:
                    break
            # A deferred give-up returns to the Action Teacher only to
            # finish a separately authorized mutation. A resolved recovery
            # likewise resumes ordinary processing with its success in
            # history. Both paths are intentional and explicitly distinct.
            assert recovery_deferred or recovery_resolved
            continue

        # Preserve PARTIAL_SUCCESS as a distinct execution outcome.
        # It is not a recovery failure, but the next Teacher decision sees
        # the exact outcome in execution_history instead of a folded SUCCESS.
        execution_history.append(
            _execution_event(
                tool_name, dict(action.arguments), result, execution_domain,
            )
        )
        current_event = execution_history[-1]
        if not bool(current_event.get("state_changed")):
            current_signature = json.dumps(
                {
                    "tool_name": tool_name,
                    "server_name": execution_domain,
                    "arguments": action.arguments,
                    "observation": result.observation,
                    "execution_status": result.execution_status,
                },
                sort_keys=True,
                ensure_ascii=False,
                default=str,
            )
            repeated = any(
                json.dumps(
                    {
                        "tool_name": prior.get("tool_name"),
                        "server_name": str(
                            prior.get("server_name") or server_name
                        ),
                        "arguments": prior.get("arguments", {}),
                        "observation": prior.get("observation"),
                        "execution_status": prior.get("execution_status"),
                    },
                    sort_keys=True,
                    ensure_ascii=False,
                    default=str,
                ) == current_signature
                for prior in execution_history[:-1]
            )
            if repeated:
                current_event["no_progress_warning"] = (
                    "This exact tool call already returned the same outcome "
                    "without changing state. Do not repeat it again; choose a "
                    "different action or emit a legal terminal response."
                )
                _trace(
                    "no_progress_detected",
                    round_idx=round_idx,
                    tool_name=tool_name,
                    arguments=dict(action.arguments),
                )

        if _add_oracle(OracleCall(
            tool_name=tool_name,
            arguments=dict(action.arguments),
            server_name=execution_domain,
        )):
            required_tools.add(tool_name)
            oracle_observations.append(
                result.observation if result.observation is not None else {}
            )

        _turn += 1
        attempt += 1

        # Optional diagnostic bound.  Baseline passes zero because the
        # dependency chain is an atomic task seed, not a per-turn schedule.
        if max_calls_this_round > 0:
            real_this_round = sum(
                1 for oc in oracle_calls if oc.action == "tool_call"
            )
            if real_this_round >= max_calls_this_round:
                break

    # A completed round must contain a Teacher-emitted terminal action.
    # Never fabricate success when the action budget is exhausted.
    if (any(oc.action == "tool_call" for oc in oracle_calls)
            and not any(oc.action in ("final_answer", "report_error", "ask_clarification")
                        for oc in oracle_calls)):
        raise RuntimeError(
            "Teacher exhausted the per-round action budget without a "
            "terminal response"
        )
    if not oracle_calls:
        raise RuntimeError(
            "Teacher produced no action for the current conversation round"
        )

    return (
        oracle_calls,
        execution_history,
        oracle_observations,
        required_tools,
        attempt_calls,
        attempt_observations,
    )


