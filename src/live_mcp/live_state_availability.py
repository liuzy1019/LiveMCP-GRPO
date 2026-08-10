"""Shared PROVE Step-2 accounting for readonly-observed live state."""

from __future__ import annotations

from typing import Any

from src.live_mcp.contracts.record_evaluator import (
    evaluate_record_predicate,
    record_satisfies_tool_contract,
    record_state_is_known,
)
from src.live_mcp.contracts.registry import ContractRegistry
from src.live_mcp.live_state_globals import (
    build_live_global_state,
    global_contract_is_known_and_usable,
)


def build_live_state_availability(
    *,
    server_name: str,
    tool_schemas: list[dict[str, Any]],
    entity_ids: list[dict[str, Any]],
    entity_records: list[dict[str, Any]],
    contract_registry: ContractRegistry,
    probe_results: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Count discovered, state-known, state-unknown, and usable entities.

    Counts are target-tool-specific.  A globally "qualified" entity count
    cannot establish that a particular handler precondition is satisfied.
    """
    records_by_key = {
        (str(item.get("type") or ""), str(item.get("id") or "")): (
            item.get("data") if isinstance(item.get("data"), dict) else {}
        )
        for item in entity_records
        if isinstance(item, dict)
    }
    availability: dict[str, dict[str, Any]] = {}
    global_state = build_live_global_state(
        server_name, entity_ids, probe_results or [],
    )
    for schema in tool_schemas:
        tool_name = str(schema.get("name") or "")
        if not tool_name:
            continue
        contract = contract_registry.get(server_name, tool_name)
        requirements = sorted(contract.required_entity_types)
        minimum_counts = dict(contract.minimum_entity_counts)
        discovered_by_type: dict[str, int] = {}
        known_by_type: dict[str, int] = {}
        usable_by_type: dict[str, int] = {}
        for entity_type in requirements:
            matching = [
                item
                for item in entity_ids
                if str(item.get("type") or "") == entity_type
            ]
            discovered_by_type[entity_type] = len(matching)
            known_by_type[entity_type] = sum(
                1
                for item in matching
                if record_state_is_known(
                    server_name,
                    contract,
                    entity_type,
                    records_by_key.get(
                        (entity_type, str(item.get("id") or "")), {}
                    ),
                )
            )
            usable_by_type[entity_type] = sum(
                1
                for item in matching
                if record_satisfies_tool_contract(
                    server_name,
                    contract,
                    entity_type,
                    records_by_key.get(
                        (entity_type, str(item.get("id") or "")), {}
                    ),
                )
            )
        alternative_groups: list[dict[str, Any]] = []
        for group in contract.precondition_groups:
            alternatives = [
                predicate for predicate in group
                if predicate.subject.source == "argument"
                and predicate.observed_entity_required
            ]
            known = False
            usable = False
            for predicate in alternatives:
                entity_type = predicate.subject.entity_type
                for item in entity_ids:
                    if str(item.get("type") or "") != entity_type:
                        continue
                    result = evaluate_record_predicate(
                        server_name,
                        predicate,
                        records_by_key.get((
                            entity_type, str(item.get("id") or ""),
                        ), {}),
                    )
                    known = known or result is not None
                    usable = usable or result is True
            alternative_groups.append({
                "entity_types": sorted({
                    predicate.subject.entity_type
                    for predicate in alternatives
                }),
                "state_known": known,
                "has_usable_entity": usable,
            })
        global_known, global_usable = global_contract_is_known_and_usable(
            contract, global_state,
        )
        availability[tool_name] = {
            "required_entity_types": requirements,
            "minimum_entity_counts": minimum_counts,
            "discovered_by_type": discovered_by_type,
            "state_known_by_type": known_by_type,
            "state_unknown_by_type": {
                entity_type: (
                    discovered_by_type[entity_type] - known_by_type[entity_type]
                )
                for entity_type in requirements
            },
            "usable_by_type": usable_by_type,
            "alternative_entity_groups": alternative_groups,
            "global_state_known": global_known,
            "global_state_usable": global_usable,
            "has_usable_entities": all(
                usable_by_type.get(entity_type, 0)
                >= minimum_counts.get(entity_type, 1)
                for entity_type in requirements
            ) and all(
                group["has_usable_entity"] for group in alternative_groups
            ) and global_usable,
        }
    return {
        "observed_entity_count": len(entity_ids),
        "record_observed_entity_count": sum(
            1 for data in records_by_key.values() if bool(data)
        ),
        "target_tool_availability": availability,
    }
