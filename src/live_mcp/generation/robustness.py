"""Robustness visibility and scenario acceptance contracts."""

from __future__ import annotations

from typing import Any

from src.live_mcp.dependency_trace import align_sampled_chain
from src.live_mcp.fsm import RobustnessPlan
from src.live_mcp.registry.tool_semantics import resolve_tool_execution_semantics


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


def has_successful_state_transition_noop(
    execution_history: list[dict[str, Any]], domain: str,
) -> bool:
    return any(
        isinstance(event, dict)
        and event.get("success") is True
        and event.get("state_changed") is False
        and resolve_tool_execution_semantics(
            str(event.get("tool_name") or ""),
            str(event.get("server_name") or domain),
        ) == "state_transition"
        for event in execution_history
    )


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

    (b) No tool call in the history has ``success == False``.  Since
        ``report_error`` is meant to surface a real execution failure,
        emitting it without any failed call means the Teacher fabricated
        the error narrative.

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
    return not has_failed_tool_call


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


def successful_conversation_realizes_source_chain(
    *, scenario_type: str, source_chain_seed: list[str] | None,
    oracle_calls: list[Any], primary_domain: str,
) -> bool:
    if not scenario_type:
        raise ValueError("scenario must be classified before source-chain validation")
    if scenario_type not in {"normal_safe_success", "tool_error_recovery"}:
        return True
    chain = list(source_chain_seed or [])
    if not chain:
        return False
    primary_calls = [
        call for call in oracle_calls
        if getattr(call, "action", "tool_call") != "tool_call"
        or str(getattr(call, "server_name", "") or primary_domain) == primary_domain
    ]
    return align_sampled_chain(primary_calls, chain) is not None


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


def missing_function_has_nonprefix_mutation(
    missing_function: bool,
    oracle_calls: list[Any],
    source_chain_seed: list[str] | None,
    hidden_tool: str,
    tool_schemas: list[dict[str, Any]],
    authorized_mutations: set[str] | None = None,
) -> bool:
    if not missing_function or not hidden_tool or not source_chain_seed:
        return False
    if hidden_tool not in source_chain_seed:
        return False
    hidden_position = source_chain_seed.index(hidden_tool)
    authorized = set(source_chain_seed[:hidden_position])
    authorized.update(authorized_mutations or set())
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
        and str(call_field(call, "tool_name") or "") not in authorized
        for call in oracle_calls
    )


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
