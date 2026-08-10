"""Build canonical tool contracts from live schemas and audited fact ledgers."""

from __future__ import annotations

from collections.abc import Iterable

from src.live_mcp.contracts.models import EntityBinding, ToolContract, ToolStateFacts
from src.live_mcp.contracts.registry import ContractRegistry
from src.live_mcp.domain_contracts.probes import (
    _DOMAIN_ENTITY_ID_FIELD_TYPES,
    _ENTITY_ID_FIELD_TYPES,
)
from src.live_mcp.domain_contracts.outputs import DOMAIN_VALUE_OUTPUT_FIELDS
from src.live_mcp.domain_contracts import dependency as dependency_facts
from src.live_mcp.domain_contracts.states import DOMAIN_STATE_FACTS
from src.live_mcp.domain_contracts.requirements import _DOMAIN_TOOL_REQUIREMENTS


def _entity_bindings(
    fields: Iterable[str],
    *,
    domain: str,
    source: str,
) -> tuple[EntityBinding, ...]:
    bindings = []
    for field in set(fields):
        field_types = {
            **_ENTITY_ID_FIELD_TYPES,
            **_DOMAIN_ENTITY_ID_FIELD_TYPES.get(domain, {}),
        }
        entity_type = field_types.get(field)
        if entity_type is None and field.endswith("s"):
            entity_type = field_types.get(field[:-1])
        if entity_type is not None:
            bindings.append(EntityBinding(entity_type, field, source))
    return tuple(sorted(
        bindings,
        key=lambda binding: (binding.entity_type, binding.name),
    ))


def _minimum_entity_counts(
    required_entity_types: frozenset[str],
    input_entities: tuple[EntityBinding, ...],
    input_schema: dict,
    preconditions: tuple,
) -> tuple[tuple[str, int], ...]:
    counts = {entity_type: 1 for entity_type in required_entity_types}
    fields_by_type: dict[str, list[EntityBinding]] = {}
    observed_bindings = {
        predicate.subject
        for predicate in preconditions
        if predicate.subject.source == "argument"
        and predicate.observed_entity_required
    }
    for binding in input_entities:
        if binding not in observed_bindings:
            continue
        fields_by_type.setdefault(binding.entity_type, []).append(binding)
    properties = input_schema.get("properties") or {}
    for entity_type, bindings in fields_by_type.items():
        if entity_type not in required_entity_types:
            continue
        distinct_required = len(bindings)
        array_minimum = max(
            (
                int((properties.get(binding.name) or {}).get("minItems") or 1)
                for binding in bindings
            ),
            default=1,
        )
        counts[entity_type] = max(
            counts.get(entity_type, 1), distinct_required, array_minimum,
        )
    return tuple(sorted(counts.items()))


def build_tool_contract(domain: str, schema: dict) -> ToolContract:
    name = str(schema.get("name") or "")
    input_schema = schema.get("input_schema") or schema.get("inputSchema") or {}
    required = frozenset(str(value) for value in input_schema.get("required") or ())
    arguments = frozenset(
        str(value) for value in (input_schema.get("properties") or {})
    )
    annotations = schema.get("annotations") or {}
    state_facts = DOMAIN_STATE_FACTS.get(domain, {}).get(name)
    if state_facts is None:
        if domain in DOMAIN_STATE_FACTS:
            raise ValueError(f"Missing typed state facts for {domain}.{name}")
        state_facts = ToolStateFacts()
    output_fields = frozenset(
        (
            DOMAIN_VALUE_OUTPUT_FIELDS.get(domain, {})
            if domain in DOMAIN_STATE_FACTS
            else dependency_facts._DEPENDENCY_TOOL_OUTPUT_FIELDS.get(domain, {})
        ).get(name, ())
    )
    output_fields = output_fields | frozenset(
        predicate.subject.name
        for predicate in state_facts.postconditions
        if predicate.subject.source == "output"
    )
    created_output_fields = frozenset(
        predicate.subject.name
        for predicate in state_facts.postconditions
        if predicate.subject.source == "output"
    )
    explicit_output_bindings = {
        predicate.subject
        for predicate in state_facts.postconditions
        if predicate.subject.source == "output"
    }
    explicit_output_fields = {
        binding.name for binding in explicit_output_bindings
    }
    inferred_output_bindings = {
        binding
        for binding in _entity_bindings(
            output_fields, domain=domain, source="output",
        )
        if binding.name not in explicit_output_fields
    }
    inferred_output_bindings.update(explicit_output_bindings)
    output_bindings = tuple(sorted(
        inferred_output_bindings,
        key=lambda binding: (binding.entity_type, binding.name),
    ))
    input_entities = _entity_bindings(
        arguments, domain=domain, source="argument",
    )
    if domain in DOMAIN_STATE_FACTS:
        required_entity_types = frozenset(
            predicate.subject.entity_type
            for predicate in state_facts.preconditions
            if predicate.subject.source == "argument"
            and predicate.observed_entity_required
        )
    else:
        required_entity_types = frozenset(
            _DOMAIN_TOOL_REQUIREMENTS.get(domain, {}).get(name, set())
        )
    return ToolContract(
        domain=domain,
        name=name,
        readonly=bool(annotations.get("readonly")),
        mutating=bool(annotations.get("mutating")),
        arguments=arguments,
        required_arguments=required,
        required_entity_types=required_entity_types,
        minimum_entity_counts=_minimum_entity_counts(
            required_entity_types, input_entities, input_schema,
            state_facts.preconditions,
        ),
        input_entities=input_entities,
        output_entities=output_bindings,
        output_fields=output_fields,
        created_output_fields=created_output_fields,
        preconditions=state_facts.preconditions,
        precondition_groups=state_facts.precondition_groups,
        postconditions=state_facts.postconditions,
    )


def build_contract_registry(
    domain_tools: dict[str, Iterable[dict]],
) -> ContractRegistry:
    return ContractRegistry(
        build_tool_contract(domain, schema)
        for domain, schemas in domain_tools.items()
        for schema in schemas
    )
