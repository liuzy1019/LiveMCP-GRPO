"""Derive and evaluate domain-global state from readonly MCP observations."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from src.live_mcp.contracts.abstract_state import AbstractState
from src.live_mcp.contracts.chain_simulator import symbolic_step_bindings
from src.live_mcp.contracts.registry import ContractRegistry


GlobalObserver = Callable[
    [list[dict[str, Any]], list[dict[str, Any]]],
    dict[tuple[str, str], Any],
]


def _shopping_cart_state(
    entity_ids: list[dict[str, Any]],
    probe_results: list[dict[str, Any]],
) -> dict[tuple[str, str], Any]:
    observed = any(
        item.get("tool") == "get_cart"
        and item.get("success") is True
        and not bool(item.get("state_changed"))
        for item in probe_results
    )
    if not observed:
        return {}
    nonempty = any(
        str(item.get("type") or "") == "cart_item"
        for item in entity_ids
        if isinstance(item, dict)
    )
    return {("cart.contents", "cart"): "nonempty" if nonempty else "empty"}


GLOBAL_OBSERVERS: dict[str, tuple[GlobalObserver, ...]] = {
    "shopping": (_shopping_cart_state,),
}


def build_live_global_state(
    domain: str,
    entity_ids: list[dict[str, Any]],
    probe_results: list[dict[str, Any]],
) -> AbstractState:
    facts: dict[tuple[str, str], Any] = {}
    for observer in GLOBAL_OBSERVERS.get(domain, ()):
        facts.update(observer(entity_ids, probe_results))
    return AbstractState(facts=facts)


def global_contract_is_known_and_usable(
    contract,
    state: AbstractState,
) -> tuple[bool, bool]:
    predicates = [
        predicate for predicate in contract.preconditions
        if predicate.subject.source == "global"
    ]
    results = [state.evaluate(predicate, {}) for predicate in predicates]
    return (
        all(result is not None for result in results),
        all(result is True for result in results),
    )


def global_chain_is_feasible(
    registry: ContractRegistry,
    domain: str,
    chain: list[str],
    initial_state: AbstractState,
) -> tuple[bool, str]:
    contracts = [registry.get(domain, tool_name) for tool_name in chain]
    bindings = symbolic_step_bindings(domain, contracts)
    state = AbstractState(facts=dict(initial_state.facts))
    for contract, step in zip(contracts, bindings):
        for predicate in contract.preconditions:
            if predicate.subject.source != "global":
                continue
            result = state.evaluate(predicate, step)
            if result is not True:
                reason = "unknown" if result is None else "contradicted"
                return False, f"{contract.name} global state {predicate.slot} is {reason}"
        for predicate in contract.postconditions:
            if predicate.subject.source == "global":
                state.observe(predicate, step)
    return True, "ok"
