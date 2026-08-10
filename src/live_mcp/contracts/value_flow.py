"""Domain-neutral explicit value-flow operations over canonical contracts."""

from __future__ import annotations

from src.live_mcp.contracts.models import ToolContract
from src.live_mcp.domain_contracts.value_bindings import OUTPUT_ARGUMENT_ALIASES


def novel_output_fields(contract: ToolContract) -> frozenset[str]:
    aliases = OUTPUT_ARGUMENT_ALIASES.get(contract.domain, {})
    echoed = {
        output_field
        for output_field in contract.output_fields
        if output_field in contract.arguments
        or set(aliases.get(output_field, ())) & set(contract.arguments)
    }
    # A creator may echo a caller-supplied identity (for example
    # ``touch(path) -> path`` or ``tar_create(archive) -> archive``).  The
    # state transition is novel, but the scalar value is not.  Keep those two
    # notions separate: created_output_fields drives symbolic entity/state
    # propagation, while this function is only for observation-derived value
    # flow.
    # Requiredness cannot establish novelty: an optional argument may still
    # be supplied by the caller.  Treat every declared input echo as
    # non-novel; concrete replay remains responsible for proving an actually
    # observation-derived binding.
    return frozenset(set(contract.output_fields) - echoed)


def value_bindings(
    domain: str,
    source: ToolContract,
    target: ToolContract,
) -> tuple[tuple[str, str], ...]:
    bindings: set[tuple[str, str]] = set()
    aliases = OUTPUT_ARGUMENT_ALIASES.get(domain, {})
    for output_field in novel_output_fields(source):
        if output_field in target.required_arguments:
            bindings.add((output_field, output_field))
        for argument in aliases.get(output_field, ()):
            if argument in target.required_arguments:
                bindings.add((output_field, argument))
    return tuple(sorted(bindings))
