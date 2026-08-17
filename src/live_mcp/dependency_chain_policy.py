"""Generic evaluation of declarative dependency-chain constraints."""

from __future__ import annotations

from src.live_mcp.contracts.models import ArgumentValue, ToolContract
from src.live_mcp.contracts.registry import ContractRegistry
from src.live_mcp.contracts.state_relations import render_predicate, transition_matches
from src.live_mcp.contracts.value_flow import chain_novel_output_fields
from src.live_mcp.domain_contracts.chains import _DOMAIN_CHAIN_CONSTRAINTS
from src.live_mcp.domain_contracts.value_bindings import OUTPUT_ARGUMENT_ALIASES
from src.live_mcp.registry.tool_semantics import resolve_tool_operation


def chain_contract_issue(server_name: str, chain: list[str]) -> str | None:
    positions = {name: index for index, name in enumerate(chain)}
    present = set(chain)
    for rule in _DOMAIN_CHAIN_CONSTRAINTS.get(server_name, ()):
        if rule.kind == "forbidden_cooccurrence":
            if present & rule.sources and present & rule.targets:
                return rule.code
            continue
        for target in rule.targets & present:
            target_index = positions[target]
            preceding = chain[:target_index]
            if rule.kind == "requires_predecessor":
                if target_index == 0 or chain[target_index - 1] not in rule.sources:
                    return rule.code
            elif rule.kind == "forbidden_predecessor":
                if target_index > 0 and chain[target_index - 1] in rule.sources:
                    return rule.code
            elif rule.kind == "forbidden_prior":
                if any(source in preceding for source in rule.sources):
                    return rule.code
            elif rule.kind == "forbidden_order":
                if any(source in preceding for source in rule.sources):
                    return rule.code
            elif rule.kind == "require_between":
                for source in rule.sources & present:
                    source_index = positions[source]
                    if source_index >= target_index:
                        continue
                    if not any(
                        bridge in chain[source_index + 1:target_index]
                        for bridge in rule.bridges
                    ):
                        return rule.code
            else:
                raise ValueError(f"Unknown chain constraint kind: {rule.kind}")
    return None


def _all_preconditions(contract: ToolContract):
    yield from contract.preconditions
    for group in contract.precondition_groups:
        yield from group


def _mutation_enables(
    source: ToolContract,
    target: ToolContract,
    source_fields: frozenset[str],
    intermediate: tuple[ToolContract, ...] = (),
) -> bool:
    aliases = OUTPUT_ARGUMENT_ALIASES.get(source.domain, {})
    output_types = {
        binding.name: binding.entity_type
        for binding in source.output_entities
        if binding.name in source_fields
    }
    target_inputs = {
        binding.name: binding.entity_type
        for binding in target.input_entities
        if binding.name in target.required_arguments
    }
    shadowed_types = {
        binding.entity_type
        for contract in intermediate
        for binding in contract.output_entities
        if binding.name in contract.created_output_fields
    }
    if any(
        output_type == target_type
        and output_type not in shadowed_types
        and (
            output_field == target_argument
            or target_argument in aliases.get(output_field, ())
        )
        for output_field, output_type in output_types.items()
        for target_argument, target_type in target_inputs.items()
    ):
        return True
    return any(
        transition_matches(source_predicate, target_predicate)
        for source_predicate in source.postconditions
        for target_predicate in _all_preconditions(target)
    )


def _mutation_is_immediately_reversed(
    source: ToolContract,
    target: ToolContract,
) -> bool:
    """Detect create/delete and add/remove pairs from state facts."""
    return any(
        source_predicate.slot == target_predicate.slot
        and (
            source_predicate.slot.endswith(".exists")
            or source_predicate.slot.endswith(".member")
        )
        and source_predicate.subject.entity_type
        == target_predicate.subject.entity_type
        and _same_identity_field(
            source.domain,
            source_predicate.subject.name,
            target_predicate.subject.name,
        )
        and source_predicate.value is True
        and target_predicate.value is False
        for source_predicate in source.postconditions
        for target_predicate in target.postconditions
    )


def _same_identity_field(
    domain: str, source_name: str, target_name: str,
) -> bool:
    aliases = OUTPUT_ARGUMENT_ALIASES.get(domain, {})
    return bool(
        source_name == target_name
        or target_name in aliases.get(source_name, ())
    )


def _creator_target_effect_already_satisfied(
    prefix: tuple[ToolContract, ...],
    target: ToolContract,
) -> str | None:
    """Return a fixed target effect already established on a new entity."""
    for source in prefix:
        created_fields = set(source.created_output_fields)
        if not created_fields:
            continue
        for source_predicate in source.postconditions:
            if (
                source_predicate.subject.source != "output"
                or source_predicate.subject.name not in created_fields
            ):
                continue
            for target_predicate in target.postconditions:
                if (
                    target_predicate.subject.source != "argument"
                    or isinstance(target_predicate.value, ArgumentValue)
                ):
                    continue
                if (
                    transition_matches(source_predicate, target_predicate)
                    and _same_identity_field(
                        source.domain,
                        source_predicate.subject.name,
                        target_predicate.subject.name,
                    )
                ):
                    return target_predicate.slot
    return None


def _creator_update_is_redundant(
    source: ToolContract,
    target: ToolContract,
) -> bool:
    """Return whether target only re-sets fields available at creation time."""
    if not source.mutating or not target.mutating:
        return False
    if (
        resolve_tool_operation(source.name, source.domain) != "create"
        or resolve_tool_operation(target.name, target.domain) != "update"
    ):
        return False
    created_types = {
        binding.entity_type
        for binding in source.output_entities
        if binding.name in source.created_output_fields
    }
    identity_arguments = {
        binding.name
        for binding in target.input_entities
        if binding.entity_type in created_types
    }
    if not identity_arguments:
        return False
    effect_arguments = target.arguments - identity_arguments
    return bool(effect_arguments) and effect_arguments <= source.arguments


def missing_function_chain_issue(
    registry: ContractRegistry,
    server_name: str,
    chain: list[str],
) -> str | None:
    """Validate that hiding the final capability yields an atomic no-write task."""
    if not chain:
        return "missing_function_without_dependency_chain"
    contracts = [registry.get(server_name, name) for name in chain]
    for contract in contracts[:-1]:
        if contract.mutating:
            return f"missing_function_mutating_prefix:{contract.name}"
    target = contracts[-1]
    if not target.mutating:
        return "missing_function_readonly_capability_unproven"
    if not target.postconditions:
        return "missing_function_state_effect_unproven"
    visible_effects = {
        render_predicate(predicate)
        for contract in registry.domain(server_name)
        if contract.name != target.name
        for predicate in contract.postconditions
    }
    if not any(
        render_predicate(predicate) not in visible_effects
        for predicate in target.postconditions
    ):
        return "missing_function_unique_state_effect_unproven"
    return None


def scenario_chain_issue(
    registry: ContractRegistry,
    server_name: str,
    chain: list[str],
    *,
    difficulty: str,
    missing_function: bool,
) -> str | None:
    """Return a deterministic incompatibility between a chain and scenario."""
    contracts = [registry.get(server_name, name) for name in chain]
    if missing_function:
        return missing_function_chain_issue(registry, server_name, chain)
    if difficulty in {"missing", "minimal"}:
        mutations = [contract.name for contract in contracts if contract.mutating]
        if len(mutations) > 1:
            return f"incomplete_multi_mutation:{','.join(mutations)}"
    return None


def _creator_target_has_unestablished_internal_precondition(
    prefix: tuple[ToolContract, ...],
    target: ToolContract,
) -> str | None:
    """Reject mutation targets requiring hidden state on a newly created entity."""
    creator_outputs = {
        binding.entity_type
        for contract in prefix
        if contract.mutating
        for binding in contract.output_entities
        if binding.name in contract.created_output_fields
    }
    if not creator_outputs:
        return None
    target_inputs = {
        binding.entity_type for binding in target.input_entities
    }
    if creator_outputs.isdisjoint(target_inputs):
        return None
    established = [
        predicate
        for contract in prefix
        for predicate in contract.postconditions
    ]
    for predicate in _all_preconditions(target):
        if predicate.observed_entity_required:
            continue
        if not any(
            transition_matches(postcondition, predicate)
            for postcondition in established
        ):
            return predicate.slot
    return None


def goal_coherence_issue(
    registry: ContractRegistry,
    server_name: str,
    chain: list[str],
) -> str | None:
    """Reject structurally provable non-goals before Query Teacher.

    This local trainability gate uses only canonical tool/state contracts.  It
    deliberately leaves ambiguous workflows to Query Teacher instead of
    encoding domain/tool-name exceptions here.
    """
    contracts = [registry.get(server_name, name) for name in chain]
    novel_fields = [
        chain_novel_output_fields(contracts, index)
        for index in range(len(contracts))
    ]
    mutations = [
        (index, contract)
        for index, contract in enumerate(contracts)
        if contract.mutating
    ]
    if not mutations:
        return None
    for (_, source), (_, target) in zip(mutations, mutations[1:]):
        if _mutation_is_immediately_reversed(source, target):
            return (
                "mutation_reversal:"
                f"{source.name}->{target.name}"
            )
        if _creator_update_is_redundant(source, target):
            return (
                "redundant_create_update:"
                f"{source.name}->{target.name}"
            )
    for target_index, target in mutations:
        already_satisfied = _creator_target_effect_already_satisfied(
            tuple(contracts[:target_index]), target,
        )
        if already_satisfied:
            return (
                "created_entity_effect_already_satisfied:"
                f"{target.name}:{already_satisfied}"
            )
        hidden_precondition = (
            _creator_target_has_unestablished_internal_precondition(
                tuple(contracts[:target_index]), target,
            )
        )
        if hidden_precondition:
            return (
                "created_entity_hidden_precondition:"
                f"{target.name}:{hidden_precondition}"
            )
    for source_index, source in mutations:
        if source_index == len(contracts) - 1:
            continue
        if not any(
            _mutation_enables(
                source,
                candidate,
                novel_fields[source_index],
                tuple(contracts[source_index + 1:target_index]),
            )
            for target_index, candidate in enumerate(
                contracts[source_index + 1:], start=source_index + 1,
            )
        ):
            return (
                "independent_prior_mutation:"
                f"{source.name}"
            )
    return None
