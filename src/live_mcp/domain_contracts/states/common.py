"""Small constructors for readable typed state fact declarations."""

from __future__ import annotations

from typing import Any

from src.live_mcp.contracts.models import (
    ArgumentValue,
    EntityBinding,
    StatePredicate,
    ToolStateFacts,
)


def arg(
    entity_type: str,
    name: str,
    slot: str,
    value: Any = True,
    *,
    observed: bool = True,
) -> StatePredicate:
    return StatePredicate(
        slot,
        EntityBinding(entity_type, name, "argument"),
        value,
        observed_entity_required=observed,
    )


def out(entity_type: str, name: str, slot: str, value: Any = True) -> StatePredicate:
    return StatePredicate(
        slot,
        EntityBinding(entity_type, name, "output"),
        value,
        observed_entity_required=False,
    )


def global_fact(name: str, slot: str, value: Any = True) -> StatePredicate:
    return StatePredicate(
        slot,
        EntityBinding("global", name, "global"),
        value,
        observed_entity_required=False,
    )


def argument_value(name: str) -> ArgumentValue:
    return ArgumentValue(name)


def facts(
    *,
    pre: tuple[StatePredicate, ...] = (),
    any_of: tuple[tuple[StatePredicate, ...], ...] = (),
    post: tuple[StatePredicate, ...] = (),
) -> ToolStateFacts:
    return ToolStateFacts(
        preconditions=pre,
        precondition_groups=any_of,
        postconditions=post,
    )
