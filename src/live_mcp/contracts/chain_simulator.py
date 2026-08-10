"""Symbolic contract simulation for dependency-chain candidates."""

from __future__ import annotations

from src.live_mcp.contracts.abstract_state import AbstractState, SimulationIssue
from src.live_mcp.contracts.models import ToolContract
from src.live_mcp.contracts.registry import ContractRegistry
from src.live_mcp.contracts.value_flow import novel_output_fields
from src.live_mcp.domain_contracts.value_bindings import OUTPUT_ARGUMENT_ALIASES


_STATE_PRESERVING_OUTPUTS: dict[
    tuple[str, str], tuple[str, str]
] = {
    ("filesystem", "cp"): ("source", "target"),
    ("filesystem", "mv"): ("source", "target"),
}


def _target_argument_for_output(
    domain: str,
    output_field: str,
    target_arguments: frozenset[str],
) -> str | None:
    aliases = OUTPUT_ARGUMENT_ALIASES.get(domain, {}).get(
        output_field, (),
    )
    alias = next((name for name in aliases if name in target_arguments), None)
    if alias is not None:
        return alias
    if output_field in target_arguments:
        return output_field
    return None


def symbolic_step_bindings(
    domain: str,
    contracts: list[ToolContract],
) -> list[dict[str, str]]:
    """Bind downstream arguments to the nearest compatible upstream output."""
    bindings: list[dict[str, str]] = []
    available_outputs: list[tuple[str, str]] = []
    for index, contract in enumerate(contracts):
        step = {
            f"argument:{argument}": f"live:{domain}:{argument}"
            for argument in contract.arguments
        }
        for output_field, symbol in reversed(available_outputs):
            target = _target_argument_for_output(
                domain, output_field, contract.arguments,
            )
            if (
                target is not None
                and step.get(f"argument:{target}", "").startswith("live:")
            ):
                step[f"argument:{target}"] = symbol
        state_outputs = (
            novel_output_fields(contract) | contract.created_output_fields
        )
        for output_field in contract.output_fields:
            symbol = f"step:{index}:{output_field}"
            step[f"output:{output_field}"] = symbol
            if output_field in state_outputs:
                available_outputs.append((output_field, symbol))
        bindings.append(step)
    return bindings


def simulate_symbolic_chain(
    registry: ContractRegistry,
    domain: str,
    chain: list[str],
) -> tuple[AbstractState, tuple[SimulationIssue, ...]]:
    """Reject only contradictions; unknown facts wait for Step-2 live state."""
    contracts = [registry.get(domain, tool_name) for tool_name in chain]
    bindings = symbolic_step_bindings(domain, contracts)
    state = AbstractState()
    issues: list[SimulationIssue] = []
    for index, (contract, step) in enumerate(zip(contracts, bindings)):
        for predicate in contract.preconditions:
            result = state.evaluate(predicate, step)
            if result is False:
                issues.append(SimulationIssue(
                    index, contract.name, predicate, "contradicted",
                ))
        for group in contract.precondition_groups:
            results = tuple(state.evaluate(predicate, step) for predicate in group)
            if results and all(result is False for result in results):
                issues.append(SimulationIssue(
                    index, contract.name, group[0], "contradicted",
                ))
        if issues:
            break
        preservation = _STATE_PRESERVING_OUTPUTS.get(
            (contract.domain, contract.name)
        )
        if preservation is not None:
            source_name, output_name = preservation
            source_subject = step.get(f"argument:{source_name}")
            output_subject = step.get(f"output:{output_name}")
            if source_subject is not None and output_subject is not None:
                for (slot, subject), value in list(state.facts.items()):
                    if subject == source_subject:
                        state.facts[(slot, output_subject)] = value
        for predicate in contract.postconditions:
            state.observe(predicate, step)
    return state, tuple(issues)
