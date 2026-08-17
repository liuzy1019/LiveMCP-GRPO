"""Early semantic validation of one completed Teacher conversation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.live_mcp.dependency_trace import unauthorized_mutating_tool_names
from src.live_mcp.fsm import RobustnessPlan
from src.live_mcp.generation.robustness import (
    hallucinated_report_error,
    missing_function_has_mutation,
    missing_function_unresolved_failures,
    single_round_clarification_with_successful_mutation,
    successful_distractor_names,
    zero_tool_terminal_is_valid,
)
from src.live_mcp.generation.scenario import classify_scenario
from src.live_mcp.generation.teacher_contracts import (
    reference_entity_types,
    round_entity_occurrences,
    typed_entity_reference_visibility_from_rounds,
    user_visible_private_id_exposure,
    user_visible_terminal_tool_name_exposure,
)
from src.live_mcp.registry.tool_semantics import is_mutating_tool
from src.live_mcp.types import OracleCall


@dataclass(frozen=True)
class EarlyTraceValidation:
    accepted: bool
    reason: str
    scenario_type: str
    terminal_action: str
    real_calls: list[OracleCall]
    initial_round_calls: list[OracleCall]
    continuation_link_evidence: list[dict[str, Any]]


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
    paper_baseline: bool,
    teacher_visible_tools: list[dict[str, Any]] | None = None,
    oracle_observations: list[Any] | None = None,
    oracle_observations_per_round: list[list[Any]] | None = None,
    dependency_contracts: list[dict[str, str]] | None = None,
    live_state_public_entity_ids: set[str] | None = None,
) -> EarlyTraceValidation:
    """Classify a trace and reject contradictions before dependency replay."""
    real_calls = [
        call for call in oracle_calls
        if getattr(call, "action", "tool_call") == "tool_call"
    ]
    if not paper_baseline and source_chain_seed:
        unauthorized_mutations = unauthorized_mutating_tool_names(
            oracle_calls_per_round[0] if oracle_calls_per_round else oracle_calls,
            source_chain_seed,
            is_mutating=lambda name: is_mutating_tool(name, domain),
        )
        if unauthorized_mutations:
            return _rejected(
                "initial_round_unauthorized_mutation:"
                f"tools={unauthorized_mutations}",
                real_calls,
            )
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
    if not paper_baseline:
        entity_types = reference_entity_types(domain, server_tools)
        private_entity_ids, public_entity_ids = (
            typed_entity_reference_visibility_from_rounds(
            domain=domain,
            calls_per_round=oracle_calls_per_round,
            observations_per_round=oracle_observations_per_round or [],
            server_tools=server_tools,
            entity_types=entity_types,
            )
        )
        public_entity_ids.update(live_state_public_entity_ids or set())
        private_entity_ids.difference_update(public_entity_ids)
        private_id_exposure = user_visible_private_id_exposure(
            conversation_queries,
            oracle_calls_per_round,
            private_entity_ids=private_entity_ids,
            public_entity_ids=public_entity_ids,
        )
        if private_id_exposure is not None:
            return _rejected(
                "user_visible_private_entity_id:"
                f"round={private_id_exposure.round_idx}:"
                f"surface={private_id_exposure.surface}:"
                f"ids={list(private_id_exposure.leaked_ids)}",
                real_calls,
            )
    if not paper_baseline:
        visible_tool_names = {
            str(tool.get("name") or "")
            for tool in (teacher_visible_tools or server_tools)
            if str(tool.get("name") or "")
        }
        hidden_tool_names = {
            str(plan.hidden_tool)
        } if plan.hidden_tool else set()
        tool_name_exposure = user_visible_terminal_tool_name_exposure(
            oracle_calls_per_round,
            tool_names=visible_tool_names | hidden_tool_names,
            hidden_tool_names=hidden_tool_names,
        )
        if tool_name_exposure is not None:
            return _rejected(
                "user_visible_private_tool_name:"
                f"round={tool_name_exposure.round_idx}:"
                f"tools={list(tool_name_exposure.exposed_tool_names)}",
                real_calls,
            )
    continuation_link_evidence: list[dict[str, Any]] = []
    if not paper_baseline and len(oracle_calls_per_round) > 1:
        continuation_issue, continuation_link_evidence = (
            _validate_continuation_entity_links(
                domain=domain,
                oracle_calls_per_round=oracle_calls_per_round,
                oracle_observations_per_round=(
                    oracle_observations_per_round or []
                ),
                server_tools=server_tools,
            )
        )
        if continuation_issue:
            continuation_link_evidence.append({
                "verification": "diagnostic_unproven",
                "reason": continuation_issue,
            })
    if plan.missing_function:
        if not abstentions:
            return _rejected("missing_function_without_abstention", real_calls)
        if (
            not source_chain_seed
            or not plan.hidden_tool
            or plan.hidden_tool != source_chain_seed[-1]
        ):
            return _rejected(
                "missing_function_hidden_target_mismatch", real_calls,
            )
        if not paper_baseline and not plan.missing_function_evidence:
            return _rejected(
                "missing_function_capability_evidence_missing", real_calls,
            )
        unresolved_failures = missing_function_unresolved_failures(
            True, execution_history,
        )
        if unresolved_failures:
            return _rejected(
                "missing_function_unresolved_execution_failure:"
                f"tools={sorted(unresolved_failures)}",
                real_calls,
            )
        if missing_function_has_mutation(
            True,
            oracle_calls,
            server_tools,
        ):
            return _rejected(
                "missing_function_mutation", real_calls,
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
        continuation_link_evidence=continuation_link_evidence,
    )


def _validate_continuation_entity_links(
    *,
    domain: str,
    oracle_calls_per_round: list[list[OracleCall]],
    oracle_observations_per_round: list[list[Any]],
    server_tools: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    """Record how each local continuation relates to prior entity evidence.

    PROVE does not publish this as a corpus gate.  It is a local trainability
    contract for auditing follow-ups. Exact entity reuse is strong evidence,
    but its absence is not a contradiction: a follow-up may operate on a
    transformed entity, a result set, or another entity explicitly named by
    the user. Private-ID and domain-semantic gates remain fail-closed.
    """
    evidence: list[dict[str, Any]] = []
    lineage_facts: dict[tuple[str, str], dict[str, Any]] = {}
    for round_idx in range(1, len(oracle_calls_per_round)):
        immediate_previous_facts, _ = round_entity_occurrences(
            domain=domain,
            calls=oracle_calls_per_round[round_idx - 1],
            observations=(
                oracle_observations_per_round[round_idx - 1]
                if round_idx - 1 < len(oracle_observations_per_round)
                else []
            ),
            server_tools=server_tools,
        )
        if immediate_previous_facts:
            lineage_facts = immediate_previous_facts
        current_facts, current_inputs = round_entity_occurrences(
            domain=domain,
            calls=oracle_calls_per_round[round_idx],
            observations=(
                oracle_observations_per_round[round_idx]
                if round_idx < len(oracle_observations_per_round)
                else []
            ),
            server_tools=server_tools,
        )
        current_real_calls = [
            call for call in oracle_calls_per_round[round_idx]
            if call.action == "tool_call"
        ]
        if not current_real_calls:
            continue
        if not current_inputs:
            evidence.append({
                "previous_round_idx": round_idx - 1,
                "current_round_idx": round_idx,
                "verification": "not_applicable_no_typed_entity_input",
            })
            if current_facts:
                lineage_facts = current_facts
            continue
        if not lineage_facts:
            return (
                "continuation_entity_lineage_unproven:"
                f"round={round_idx}:reason=no_prior_typed_entity",
                evidence,
            )
        shared_keys = sorted(set(lineage_facts) & set(current_inputs))
        if not shared_keys:
            return (
                "continuation_entity_lineage_unproven:"
                f"round={round_idx}:reason=no_exact_typed_entity_reuse",
                evidence,
            )
        entity_type, serialized_value = shared_keys[0]
        previous = lineage_facts[(entity_type, serialized_value)]
        current = current_inputs[(entity_type, serialized_value)]
        evidence.append({
            "previous_round_idx": round_idx - 1,
            "current_round_idx": round_idx,
            "entity_type": entity_type,
            "value": previous["value"],
            "previous_capability": previous["capability"],
            "previous_field": previous["field"],
            "previous_surface": previous["surface"],
            "current_capability": current["capability"],
            "current_field": current["field"],
        })
        lineage_facts = current_facts
    return "", evidence


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
        continuation_link_evidence=[],
    )


__all__ = ["EarlyTraceValidation", "validate_early_candidate_trace"]
