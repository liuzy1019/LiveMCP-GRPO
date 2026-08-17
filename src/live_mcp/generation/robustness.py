"""Robustness visibility and scenario acceptance contracts."""

from __future__ import annotations

from typing import Any

from src.live_mcp.fsm import RobustnessPlan
from src.live_mcp.contracts.state_relations import render_predicate
from src.live_mcp.registry.tool_semantics import resolve_tool_execution_semantics
from src.live_mcp.registry.tool_semantics import unresolved_failed_tool_names
from src.live_mcp.contracts.factory import build_contract_registry
from src.live_mcp.dependency_chain_policy import missing_function_chain_issue


def profile_scenario_is_valid(
    *, profile: Any, difficulty: str, scenario_type: str,
    missing_function: bool, irrelevance: bool,
) -> bool:
    if missing_function or irrelevance:
        return True
    return not (
        difficulty == "complete"
        and scenario_type == "clarification_required"
    )


def normalized_policy_query(text: Any) -> str:
    return " ".join(str(text or "").casefold().split())


def single_round_clarification_with_successful_mutation(
    *,
    n_queries: int,
    scenario_type: str,
    execution_history: list[dict[str, Any]],
    domain: str,
) -> bool:
    """Reject single-round tasks where the Teacher performed a successful
    state mutation but terminated with ``ask_clarification``.

    In PROVE's single-round contract the terminal action is the entire
    reward-bearing surface: a rollout that emits ``ask_clarification`` on
    turn 0 is expected to satisfy the task without ever executing tools,
    so any prior successful mutation in the Teacher trajectory becomes
    reward-unattributable (there is no chain edge, no ``final_answer``
    grounded criterion, and no follow-up turn where the mutation could
    be verified).  Multi-round tasks legitimately combine an early
    ``ask_clarification`` with later mutations in subsequent rounds and
    are excluded from this gate.
    """
    if n_queries != 1 or scenario_type != "clarification_required":
        return False
    return any(
        isinstance(event, dict)
        and event.get("success") is True
        and resolve_tool_execution_semantics(
            str(event.get("tool_name") or ""),
            str(event.get("server_name") or domain),
        ) == "state_transition"
        for event in execution_history
    )


# Scenarios in which ``report_error`` is a legitimate terminal without a
# preceding failed tool call: the Teacher correctly refused to attempt an
# unavailable/irrelevant action.
_REPORT_ERROR_ABSTENTION_SCENARIOS: frozenset[str] = frozenset({
    "missing_function",
    "irrelevant",
    "no_tool_or_abstention",
})


def hallucinated_report_error(
    *,
    terminal_action: str,
    scenario_type: str,
    execution_history: list[dict[str, Any]],
    domain: str,
) -> bool:
    """Reject trajectories where the Teacher terminates with ``report_error``
    but the execution history does not support that terminal.

    Two logically-symmetric hallucination patterns are captured:

    (a) A successful state-transition mutation appears in the history
        (the Teacher already changed the world) yet the Teacher then
        reports failure.  This creates an unattributable reward signal:
        the environment state advanced but the terminal action denies
        it.

    (b) Neither a failed call nor a successful read with an explicitly empty
        or partial result appears in history.  Empty discovery results are
        factual inability evidence even though the MCP transport succeeded.

    Abstention-flavoured scenarios (``missing_function``, ``irrelevant``,
    ``no_tool_or_abstention``) are exempt because they are defined as
    "correctly refuse to act", so the absence of a failed call is expected.
    """
    if terminal_action != "report_error":
        return False
    if scenario_type in _REPORT_ERROR_ABSTENTION_SCENARIOS:
        return False
    successful_mutation = any(
        isinstance(event, dict)
        and event.get("success") is True
        and resolve_tool_execution_semantics(
            str(event.get("tool_name") or ""),
            str(event.get("server_name") or domain),
        ) == "state_transition"
        for event in execution_history
    )
    if successful_mutation:
        return True
    has_failed_tool_call = any(
        isinstance(event, dict)
        and event.get("success") is False
        and event.get("tool_name")
        and event.get("tool_name") != "__reject__"
        for event in execution_history
    )
    has_empty_result_evidence = any(
        isinstance(event, dict)
        and event.get("success") is True
        and event.get("execution_status") == "PARTIAL_SUCCESS"
        and resolve_tool_execution_semantics(
            str(event.get("tool_name") or ""),
            str(event.get("server_name") or domain),
        ) != "state_transition"
        for event in execution_history
    )
    return not (has_failed_tool_call or has_empty_result_evidence)


def successful_distractor_names(
    oracle_calls: list[Any], distractor_tools: list[dict[str, Any]],
) -> set[str]:
    distractor_names = {
        str(tool.get("name") or "")
        for tool in distractor_tools if str(tool.get("name") or "")
    }
    return {
        str(getattr(call, "tool_name", "") or "")
        for call in oracle_calls
        if getattr(call, "action", "tool_call") == "tool_call"
        and str(getattr(call, "tool_name", "") or "") in distractor_names
    }


def missing_function_original_round_should_abort(
    missing_function: bool, round_idx: int, oracle_calls: list[Any],
) -> bool:
    if not missing_function or round_idx != 0:
        return False
    return not any(
        getattr(call, "action", "tool_call")
        in ("ask_clarification", "report_error")
        for call in oracle_calls
    )


def zero_tool_terminal_is_valid(
    difficulty: str, oracle_calls: list[Any],
) -> bool:
    actions = {getattr(call, "action", "tool_call") for call in oracle_calls}
    if "report_error" in actions:
        return True
    return difficulty in ("missing", "minimal") and "ask_clarification" in actions


def missing_function_has_mutation(
    missing_function: bool,
    oracle_calls: list[Any],
    tool_schemas: list[dict[str, Any]],
) -> bool:
    """Return whether a missing-capability trace changed server state."""
    if not missing_function:
        return False
    schemas = {str(item.get("name") or ""): item for item in tool_schemas}

    def call_field(call: Any, field: str, default: Any = "") -> Any:
        if isinstance(call, dict):
            return call.get(field, default)
        return getattr(call, field, default)

    return any(
        call_field(call, "action", "tool_call") == "tool_call"
        and (schemas.get(str(call_field(call, "tool_name") or ""), {}).get(
            "annotations"
        ) or {}).get("mutating") is True
        for call in oracle_calls
    )


def bind_missing_function_contract(
    *, domain: str, source_chain_seed: list[str] | None,
    tool_schemas: list[dict[str, Any]], plan: RobustnessPlan,
    require_capability_evidence: bool = True,
) -> tuple[RobustnessPlan | None, str]:
    """Bind a sampled missing-function knob to audited state-effect evidence.

    Tool-name absence alone does not prove capability absence.  Until readonly
    information facets are declared, only a mutating target with at least one
    postcondition not produced by any remaining visible tool is eligible.
    """
    if not plan.missing_function:
        return plan, ""
    chain = list(source_chain_seed or [])
    if not chain:
        return None, "missing_function_without_dependency_chain"
    hidden_tool = chain[-1]
    schema_by_name = {
        str(schema.get("name") or ""): schema for schema in tool_schemas
    }
    hidden_schema = schema_by_name.get(hidden_tool)
    if not hidden_schema:
        return None, "missing_function_hidden_target_unknown"
    if not require_capability_evidence:
        plan.hidden_tool = hidden_tool
        return plan, ""
    registry = build_contract_registry({domain: tool_schemas})
    contract_issue = missing_function_chain_issue(registry, domain, chain)
    if contract_issue:
        return None, contract_issue
    target_postconditions = registry.get(domain, hidden_tool).postconditions
    visible_effects = {
        render_predicate(predicate)
        for contract in registry.domain(domain)
        if contract.name != hidden_tool
        for predicate in contract.postconditions
    }
    unique_effects = tuple(sorted(
        render_predicate(predicate)
        for predicate in target_postconditions
        if render_predicate(predicate) not in visible_effects
    ))
    if not unique_effects:
        return None, "missing_function_unique_state_effect_unproven"
    plan.hidden_tool = hidden_tool
    plan.missing_function_evidence = unique_effects
    return plan, ""


def missing_function_unresolved_failures(
    missing_function: bool,
    execution_history: list[dict[str, Any]],
) -> set[str]:
    """Return real execution failures that confound a missing-function row.

    Missing-function is a controlled robustness variant whose unavailable
    final capability is the intended blocker.  A visible call that failed and
    was never repaired introduces a second, observed cause of incompletion.
    """
    if not missing_function:
        return set()
    return {
        name for name in unresolved_failed_tool_names(execution_history)
        if name != "__reject__"
    }


def strip_enums_from_schemas(tools: list[dict]) -> list[dict]:
    def strip(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: strip(child) for key, child in value.items() if key != "enum"}
        if isinstance(value, list):
            return [strip(child) for child in value]
        return value

    result = []
    for tool in tools:
        copied = dict(tool)
        copied["input_schema"] = strip(tool.get("input_schema", {}))
        result.append(copied)
    return result


def build_teacher_visible_tools(
    domain_tools: list[dict], plan: RobustnessPlan,
) -> list[dict]:
    tools = [dict(tool) for tool in domain_tools]
    existing = {str(tool.get("name") or "") for tool in tools}
    for distractor in plan.distractor_tools:
        name = str(distractor.get("name") or "")
        if name and name not in existing:
            tools.append(dict(distractor))
            existing.add(name)
    if plan.strip_enums:
        tools = strip_enums_from_schemas(tools)
    if plan.missing_function and plan.hidden_tool:
        tools = [tool for tool in tools if tool["name"] != plan.hidden_tool]
    return tools
