"""Concrete Step-2 feasibility over canonical contracts and readonly state."""

from __future__ import annotations

from typing import Any

from src.live_mcp.contracts.chain_records import record_satisfies_chain
from src.live_mcp.contracts.registry import ContractRegistry
from src.live_mcp.contracts.value_flow import value_bindings
from src.live_mcp.dependency_value_flow import (
    _aggregate_probe_results_by_tool,
    _dependency_value_key,
)
from src.live_mcp.live_state_globals import (
    build_live_global_state,
    global_chain_is_feasible,
)


def _live_entities(
    registry: ContractRegistry,
    domain: str,
    chain: list[str],
    live_context: dict[str, Any],
) -> dict[str, set[str]]:
    entity_source = live_context.get("entity_ids", []) or []
    record_source = live_context.get("entity_records", []) or []
    records = {
        (str(item.get("type") or ""), str(item.get("id") or "")): (
            item.get("data") if isinstance(item.get("data"), dict) else {}
        )
        for item in record_source
        if isinstance(item, dict)
    }
    live: dict[str, set[str]] = {}
    for item in entity_source:
        if not isinstance(item, dict) or not item.get("type") or not item.get("id"):
            continue
        entity_type = str(item["type"])
        entity_id = str(item["id"])
        if record_satisfies_chain(
            registry,
            domain,
            chain,
            entity_type,
            records.get((entity_type, entity_id), {}),
        ):
            live.setdefault(entity_type, set()).add(entity_id)
    return live


def _binding_cardinality(
    target,
    bindings: tuple[tuple[str, str], ...],
    source_field: str,
) -> int:
    target_types = {
        binding.name: binding.entity_type for binding in target.input_entities
    }
    counts = dict(target.minimum_entity_counts)
    return max(
        (
            counts.get(target_types.get(argument, ""), 1)
            for output, argument in bindings
            if output == source_field
        ),
        default=1,
    )


def chain_is_feasible(
    chain: list[str],
    domain: str,
    live_context: dict[str, Any],
    registry: ContractRegistry,
) -> tuple[bool, str]:
    global_ok, global_reason = global_chain_is_feasible(
        registry,
        domain,
        chain,
        build_live_global_state(
            domain,
            list(live_context.get("entity_ids") or []),
            list(live_context.get("probe_results") or []),
        ),
    )
    if not global_ok:
        return False, global_reason

    live = _live_entities(registry, domain, chain, live_context)
    created: dict[str, int] = {}
    probes = _aggregate_probe_results_by_tool(
        list(live_context.get("probe_results") or [])
    )
    for index, tool_name in enumerate(chain):
        contract = registry.get(domain, tool_name.lower())
        missing = []
        for entity_type, minimum in contract.minimum_entity_counts:
            available = len(live.get(entity_type, set())) + created.get(
                entity_type, 0,
            )
            if available < minimum:
                missing.append(f"{entity_type}({available}/{minimum})")
        if missing:
            return False, (
                f"{contract.name} requires missing entity types {missing}"
            )
        for group in contract.precondition_groups:
            alternatives = {
                predicate.subject.entity_type
                for predicate in group
                if predicate.subject.source == "argument"
                and predicate.observed_entity_required
            }
            if alternatives and not any(
                live.get(entity_type) or created.get(entity_type, 0)
                for entity_type in alternatives
            ):
                return False, (
                    f"{contract.name} requires one of entity types "
                    f"{sorted(alternatives)}"
                )

        if index + 1 < len(chain):
            target = registry.get(domain, chain[index + 1].lower())
            bindings = value_bindings(domain, contract, target)
            source_fields = {source for source, _ in bindings}
            source_probe = probes.get(contract.name)
            if source_fields and source_probe is not None:
                output_counts = source_probe.get("output_field_counts") or {}
                output_values = source_probe.get("output_field_values") or {}
                sufficient = {
                    field for field in source_fields
                    if int(output_counts.get(field, 0))
                    >= _binding_cardinality(target, bindings, field)
                }
                if (
                    not source_probe.get("success")
                    or source_probe.get("state_changed")
                    or not sufficient
                ):
                    return False, (
                        f"{contract.name} produced insufficient live values "
                        f"for {sorted(source_fields)} required by {target.name}"
                    )
                target_types = {
                    binding.name: binding.entity_type
                    for binding in target.input_entities
                }
                joined = False
                for field in sufficient:
                    target_type = next((
                        target_types.get(argument)
                        for output, argument in bindings
                        if output == field and target_types.get(argument)
                    ), None)
                    source_keys = {
                        _dependency_value_key(value)
                        for value in (output_values.get(field) or [])
                    }
                    target_keys = {
                        _dependency_value_key(value)
                        for value in live.get(str(target_type), set())
                    }
                    if len(source_keys & target_keys) >= _binding_cardinality(
                        target, bindings, field,
                    ):
                        joined = True
                        break
                if not joined:
                    return False, (
                        f"{contract.name} output values for "
                        f"{sorted(source_fields)} do not identify an entity "
                        f"currently usable by {target.name}"
                    )
        for binding in contract.output_entities:
            if binding.name in contract.created_output_fields:
                created[binding.entity_type] = created.get(
                    binding.entity_type, 0,
                ) + 1
    return True, "ok"
