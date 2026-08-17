"""Shared outcome-evidence checks for generation and artifact consumers."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from src.live_mcp.registry.tool_semantics import (
    SELF_CONTAINED_WRITE_TOOLS,
    resolve_tool_execution_semantics,
)


def _call_value(call: Any, field: str, default: Any = None) -> Any:
    if isinstance(call, dict):
        return call.get(field, default)
    return getattr(call, field, default)


def successful_state_transition_noop_indices(
    *,
    oracle_calls: list[Any],
    execution_history: list[Any],
    domain: str,
) -> set[int]:
    """Return canonical call indices that succeeded without changing state.

    Calls and execution events are matched in order by tool name and arguments,
    mirroring required-oracle projection. Read-only calls and action-execution
    tools are not state-transition no-ops.
    """
    history_cursor = 0
    noop_indices: set[int] = set()
    for call_index, call in enumerate(oracle_calls):
        if str(_call_value(call, "action", "tool_call")) != "tool_call":
            continue
        tool_name = str(_call_value(call, "tool_name", "") or "")
        arguments = dict(_call_value(call, "arguments", {}) or {})
        matched = None
        for event_index in range(history_cursor, len(execution_history)):
            event = execution_history[event_index]
            if not isinstance(event, dict) or event.get("success") is not True:
                continue
            if (
                str(event.get("tool_name") or "") == tool_name
                and dict(event.get("arguments") or {}) == arguments
            ):
                matched = event
                history_cursor = event_index + 1
                break
        if (
            matched is not None
            and matched.get("state_changed") is False
            and resolve_tool_execution_semantics(tool_name, domain)
            == "state_transition"
        ):
            noop_indices.add(call_index)
    return noop_indices


def mutation_outcome_issue(
    *,
    tool_names: list[str],
    success_criteria: list[dict[str, Any]],
    criterion_provenance: list[dict[str, Any]],
    is_mutating: Callable[[str], bool],
) -> str | None:
    """Require replayable, attributed outcomes for state-transition writes."""
    mutations = sorted({
        name for name in tool_names
        if name not in SELF_CONTAINED_WRITE_TOOLS and is_mutating(name)
    })
    if not mutations:
        return None
    if not success_criteria:
        return f"mutation_success_criteria_missing:tools={mutations}"

    provenance_by_index = {
        int(item.get("criterion_index", -1)): item
        for item in criterion_provenance
        if isinstance(item, dict)
    }
    missing = [
        index for index in range(len(success_criteria))
        if not isinstance(provenance_by_index.get(index), dict)
        or not provenance_by_index[index].get("source_calls")
    ]
    if missing:
        return f"mutation_success_criteria_provenance_missing:indices={missing}"
    return None


__all__ = [
    "mutation_outcome_issue",
    "successful_state_transition_noop_indices",
]
