"""Early semantic validation of one completed Teacher conversation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.live_mcp.fsm import RobustnessPlan
from src.live_mcp.generation.robustness import (
    hallucinated_report_error,
    has_successful_state_transition_noop,
    missing_function_has_nonprefix_mutation,
    single_round_clarification_with_successful_mutation,
    successful_conversation_realizes_source_chain,
    successful_distractor_names,
    zero_tool_terminal_is_valid,
)
from src.live_mcp.generation.scenario import classify_scenario
from src.live_mcp.types import OracleCall


@dataclass(frozen=True)
class EarlyTraceValidation:
    accepted: bool
    reason: str
    scenario_type: str
    terminal_action: str
    real_calls: list[OracleCall]
    initial_round_calls: list[OracleCall]


def validate_early_candidate_trace(
    *,
    domain: str,
    difficulty: str,
    plan: RobustnessPlan,
    oracle_calls: list[OracleCall],
    execution_history: list[Any],
    conversation_queries: list[str],
    oracle_calls_per_round: list[list[OracleCall]],
    source_chain_seed: list[str] | None,
    server_tools: list[dict[str, Any]],
    mutation_evidence: list[dict[str, Any]],
    paper_baseline: bool,
) -> EarlyTraceValidation:
    """Classify a trace and reject contradictions before dependency replay."""
    real_calls = [
        call for call in oracle_calls
        if getattr(call, "action", "tool_call") == "tool_call"
    ]
    abstentions = [
        call for call in oracle_calls
        if getattr(call, "action", "tool_call")
        in ("ask_clarification", "report_error")
    ]
    distractors = successful_distractor_names(
        real_calls, plan.distractor_tools,
    )
    if distractors:
        return _rejected(
            f"successful_distractor_calls:{sorted(distractors)}", real_calls,
        )
    if plan.missing_function:
        if not abstentions:
            return _rejected("missing_function_without_abstention", real_calls)
        authorized = {
            str(item.get("capability") or "")
            for item in mutation_evidence
            if str(item.get("capability") or "") != plan.hidden_tool
        }
        if missing_function_has_nonprefix_mutation(
            True,
            oracle_calls,
            source_chain_seed,
            plan.hidden_tool,
            server_tools,
            authorized,
        ):
            return _rejected(
                "missing_function_nonprefix_mutation", real_calls,
            )
    elif not real_calls and not zero_tool_terminal_is_valid(
        difficulty, oracle_calls,
    ):
        return _rejected("invalid_zero_tool_terminal", real_calls)

    terminal = next(
        (call.action for call in reversed(oracle_calls) if call.action != "tool_call"),
        "final_answer",
    )
    scenario = classify_scenario(
        server_name=domain,
        oracle_calls=real_calls,
        execution_history=execution_history,
        terminal_action=terminal,
    )
    if plan.missing_function:
        scenario = (
            "clarification_required"
            if terminal == "ask_clarification"
            else "missing_function"
        )
    initial_round = oracle_calls_per_round[0] if oracle_calls_per_round else []
    if not successful_conversation_realizes_source_chain(
        scenario_type=scenario,
        source_chain_seed=source_chain_seed,
        oracle_calls=initial_round,
        primary_domain=domain,
    ):
        return _rejected(
            "source_chain_not_realized", real_calls, scenario, terminal,
            initial_round,
        )
    if (
        not plan.missing_function
        and not plan.irrelevance
        and has_successful_state_transition_noop(execution_history, domain)
    ):
        return _rejected(
            "successful_state_transition_noop", real_calls, scenario, terminal,
            initial_round,
        )
    if single_round_clarification_with_successful_mutation(
        n_queries=len(conversation_queries),
        scenario_type=scenario,
        execution_history=execution_history,
        domain=domain,
    ):
        return _rejected(
            "single_round_clarification_with_mutation", real_calls, scenario,
            terminal, initial_round,
        )
    if hallucinated_report_error(
        terminal_action=terminal,
        scenario_type=scenario,
        execution_history=execution_history,
        domain=domain,
    ):
        return _rejected(
            "hallucinated_report_error", real_calls, scenario, terminal,
            initial_round,
        )
    return EarlyTraceValidation(
        accepted=True,
        reason="",
        scenario_type=scenario,
        terminal_action=terminal,
        real_calls=real_calls,
        initial_round_calls=initial_round,
    )


def _rejected(
    reason: str,
    real_calls: list[OracleCall],
    scenario_type: str = "",
    terminal_action: str = "",
    initial_round_calls: list[OracleCall] | None = None,
) -> EarlyTraceValidation:
    return EarlyTraceValidation(
        accepted=False,
        reason=reason,
        scenario_type=scenario_type,
        terminal_action=terminal_action,
        real_calls=real_calls,
        initial_round_calls=list(initial_round_calls or []),
    )


__all__ = ["EarlyTraceValidation", "validate_early_candidate_trace"]
