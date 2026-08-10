"""Shared rendering and dependency relations over typed state contracts."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from src.live_mcp.contracts.models import ArgumentValue, StatePredicate, ToolContract


def predicate_payload(predicate: StatePredicate) -> dict[str, Any]:
    value = predicate.value
    if isinstance(value, ArgumentValue):
        value = {"argument": value.name}
    return {
        "slot": predicate.slot,
        "subject": asdict(predicate.subject),
        "value": value,
        "observed_entity_required": predicate.observed_entity_required,
    }


def render_predicate(predicate: StatePredicate) -> str:
    value = predicate.value
    rendered_value = (
        f"argument({value.name})" if isinstance(value, ArgumentValue) else repr(value)
    )
    return (
        f"{predicate.slot}[{predicate.subject.entity_type}:"
        f"{predicate.subject.source}:{predicate.subject.name}]={rendered_value}"
    )


def transition_matches(
    postcondition: StatePredicate,
    precondition: StatePredicate,
) -> bool:
    return bool(
        postcondition.slot == precondition.slot
        and postcondition.subject.entity_type == precondition.subject.entity_type
        and (
            postcondition.value == precondition.value
            or isinstance(postcondition.value, ArgumentValue)
        )
    )


def implicit_directions(
    pair: tuple[str, str],
    contracts_by_name: dict[str, ToolContract],
) -> list[tuple[str, str]]:
    directions: list[tuple[str, str]] = []
    if len(pair) != 2:
        return directions
    for source_name, target_name in (pair, tuple(reversed(pair))):
        source = contracts_by_name.get(source_name)
        target = contracts_by_name.get(target_name)
        if source is None or target is None:
            continue
        target_predicates = (
            *target.preconditions,
            *(predicate for group in target.precondition_groups for predicate in group),
        )
        if any(
            transition_matches(postcondition, precondition)
            for postcondition in source.postconditions
            for precondition in target_predicates
        ):
            directions.append((source_name, target_name))
    return directions
