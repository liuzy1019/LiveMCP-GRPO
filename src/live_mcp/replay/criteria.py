"""Generic replay criteria and verifier-progress derivation."""

from __future__ import annotations

import json
from typing import Any, Callable

from src.live_mcp.registry.tool_semantics import is_mutating_tool
from src.live_mcp.types import OracleCall


def derive_success_criteria(
    initial_state: dict[str, Any],
    final_state: dict[str, Any],
    oracle_calls: list[OracleCall],
    domain: str,
) -> list[dict[str, Any]]:
    """Derive replayable leaf postconditions from the factual state delta."""
    del oracle_calls  # The state delta, not tool names, defines the outcome.
    criteria: list[dict[str, Any]] = []

    def append_nested_delta(
        initial_value: Any,
        final_value: Any,
        path_parts: list[str],
    ) -> None:
        path = ".".join(path_parts)
        if isinstance(initial_value, dict) and isinstance(final_value, dict):
            for child_key in final_value.keys() - initial_value.keys():
                child_path = [*path_parts, str(child_key)]
                if isinstance(final_value[child_key], dict):
                    criteria.append({
                        "type": "state_exists",
                        "server": domain,
                        "path": ".".join(child_path),
                        "path_parts": child_path,
                    })
                append_nested_delta(None, final_value[child_key], child_path)
            for child_key in initial_value.keys() - final_value.keys():
                child_path = [*path_parts, str(child_key)]
                criteria.append({
                    "type": "state_absent",
                    "server": domain,
                    "path": ".".join(child_path),
                    "path_parts": child_path,
                })
            for child_key in initial_value.keys() & final_value.keys():
                if initial_value[child_key] != final_value[child_key]:
                    append_nested_delta(
                        initial_value[child_key],
                        final_value[child_key],
                        [*path_parts, str(child_key)],
                    )
            return
        if final_value is None:
            criteria.append({
                "type": "state_equals",
                "server": domain,
                "path": path,
                "path_parts": path_parts,
                "value": None,
            })
            return
        if isinstance(final_value, dict):
            for child_key, child_value in final_value.items():
                append_nested_delta(
                    None, child_value, [*path_parts, str(child_key)],
                )
            return
        if isinstance(final_value, list) or isinstance(
            final_value, (str, int, float, bool),
        ):
            criteria.append({
                "type": "state_equals",
                "server": domain,
                "path": path,
                "path_parts": path_parts,
                "value": final_value,
            })

    append_nested_delta(initial_state, final_state, [])
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for criterion in criteria:
        key = (
            str(criterion.get("type", "")),
            str(criterion.get("server", "")),
            str(criterion.get("path", "")),
            json.dumps(
                criterion.get("value", None),
                sort_keys=True,
                ensure_ascii=False,
                default=str,
            ),
        )
        if key not in seen:
            seen.add(key)
            deduped.append(criterion)
    return deduped


def derive_progress_predicates(
    oracle_calls: list[OracleCall],
    domain: str,
    *,
    entity_resolver: Callable[[str, str], str],
    requirements_resolver: Callable[[str, str], tuple[str, ...]],
) -> list[dict[str, Any]]:
    """Derive verifier-automaton progress events from the oracle trace."""
    predicates: list[dict[str, Any]] = []
    real_calls = [call for call in oracle_calls if call.action == "tool_call"]
    if not real_calls:
        return predicates

    read_prefixes = (
        "list_", "search_", "get_", "find_", "check_", "lookup_",
        "view_", "browse_", "ls", "cat", "stat", "head", "tail",
    )
    resolved_entities: set[str] = set()
    for index, call in enumerate(real_calls):
        entity = entity_resolver(call.tool_name, domain)
        if (
            any(call.tool_name.lower().startswith(prefix) for prefix in read_prefixes)
            and entity not in resolved_entities
        ):
            resolved_entities.add(entity)
            predicates.append({
                "step": index,
                "type": "resolved_required_entity",
                "tool": call.tool_name,
                "entity": entity,
            })

    for index, call in enumerate(real_calls):
        if not is_mutating_tool(call.tool_name, domain):
            continue
        target = next((
            value
            for key, value in (call.arguments or {}).items()
            if isinstance(value, str)
            and ("_id" in key.lower() or key.lower() in {"path", "event_id"})
        ), "")
        predicates.append({
            "step": index,
            "type": "completed_required_transition",
            "tool": call.tool_name,
            "entity": entity_resolver(call.tool_name, domain),
            "target_id": target,
        })

    creator_prefixes = (
        "create_", "add_", "send_", "schedule_", "mkdir", "touch",
    )
    for index in range(1, len(real_calls)):
        previous = real_calls[index - 1]
        current = real_calls[index]
        previous_name = previous.tool_name.lower()
        if not (
            (
                any(previous_name.startswith(prefix) for prefix in read_prefixes)
                or any(previous_name.startswith(prefix) for prefix in creator_prefixes)
            )
            and is_mutating_tool(current.tool_name, domain)
        ):
            continue
        previous_entity = entity_resolver(previous.tool_name, domain)
        acceptable_entities = set(
            requirements_resolver(current.tool_name, domain)
        ) | {entity_resolver(current.tool_name, domain)}
        if previous_entity in acceptable_entities:
            predicates.append({
                "step": index,
                "type": "satisfied_dependency_edge",
                "tool": current.tool_name,
                "from_step": index - 1,
                "entity": previous_entity,
            })

    terminal_actions = [
        call for call in oracle_calls if call.action != "tool_call"
    ]
    if terminal_actions:
        terminal = terminal_actions[-1]
        predicates.extend([
            {
                "step": len(real_calls),
                "type": "verified_postcondition",
                "action": terminal.action,
            },
            {
                "step": len(real_calls),
                "type": "produced_required_response",
                "action": terminal.action,
            },
        ])
    return predicates
