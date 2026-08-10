"""Evaluate one live entity record across a typed dependency chain."""

from __future__ import annotations

from src.live_mcp.contracts.models import StatePredicate, ToolContract
from src.live_mcp.contracts.record_evaluator import evaluate_record_predicate
from src.live_mcp.contracts.registry import ContractRegistry
from src.live_mcp.contracts.state_relations import transition_matches
from src.live_mcp.contracts.value_flow import value_bindings


def _latest_effect(
    effects: list[StatePredicate],
    precondition: StatePredicate,
) -> StatePredicate | None:
    return next((
        effect
        for effect in reversed(effects)
        if effect.slot == precondition.slot
        and effect.subject.entity_type == precondition.subject.entity_type
    ), None)


def record_satisfies_chain(
    registry: ContractRegistry,
    domain: str,
    chain: list[str],
    entity_type: str,
    record: dict,
) -> bool:
    """Check initial-record predicates until the chain switches to a new entity."""
    contracts = [registry.get(domain, name) for name in chain]
    effects: list[StatePredicate] = []
    uses_initial_entity = True
    for index, contract in enumerate(contracts):
        relevant = [
            predicate
            for predicate in contract.preconditions
            if predicate.subject.entity_type == entity_type
            and predicate.observed_entity_required
        ]
        if uses_initial_entity:
            for predicate in relevant:
                effect = _latest_effect(effects, predicate)
                if effect is not None:
                    if not transition_matches(effect, predicate):
                        return False
                    continue
                if evaluate_record_predicate(domain, predicate, record) is not True:
                    return False
            for group in contract.precondition_groups:
                alternatives = [
                    predicate for predicate in group
                    if predicate.subject.entity_type == entity_type
                    and predicate.observed_entity_required
                ]
                if alternatives and not any(
                    (
                        transition_matches(effect, predicate)
                        if (effect := _latest_effect(effects, predicate)) is not None
                        else evaluate_record_predicate(domain, predicate, record) is True
                    )
                    for predicate in alternatives
                ):
                    return False
        effects.extend(
            predicate
            for predicate in contract.postconditions
            if predicate.subject.entity_type == entity_type
            and predicate.subject.source == "argument"
        )
        created_fields = {
            predicate.subject.name
            for predicate in contract.postconditions
            if predicate.subject.entity_type == entity_type
            and predicate.subject.source == "output"
        }
        if created_fields and any(
            source_field in created_fields
            for later in contracts[index + 1:]
            for source_field, _ in value_bindings(domain, contract, later)
        ):
            uses_initial_entity = False
    return True
