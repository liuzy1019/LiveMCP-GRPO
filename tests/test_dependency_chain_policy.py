from types import SimpleNamespace

import src.live_mcp.generation.context_provider as context_provider
from src.live_mcp.dependency_chain_policy import chain_contract_issue
from src.live_mcp.generation.context_provider import GenerationContextMixin
from src.live_mcp.servers.shopping.server import TOOLS as SHOPPING_TOOLS


def test_shopping_mixed_lifecycle_chain_is_rejected_by_contract() -> None:
    chain = [
        "get_return_status",
        "add_to_cart",
        "checkout",
        "get_order",
        "remove_from_wishlist",
    ]

    assert chain_contract_issue("shopping", chain) == (
        "chain_mixes_return_and_purchase_lifecycles"
    )


def test_adjacent_wishlist_chain_remains_eligible() -> None:
    assert chain_contract_issue(
        "shopping", ["get_wishlist", "remove_from_wishlist"],
    ) is None


def test_canonical_feasibility_filter_executes_domain_chain_contract(
    monkeypatch,
) -> None:
    subject = GenerationContextMixin()
    subject.manager = SimpleNamespace(
        registry=SimpleNamespace(server_tools=lambda _domain: SHOPPING_TOOLS),
    )
    monkeypatch.setattr(
        context_provider,
        "simulate_symbolic_chain",
        lambda *_args, **_kwargs: ({}, []),
    )
    monkeypatch.setattr(
        context_provider,
        "chain_is_feasible",
        lambda *_args, **_kwargs: (True, ""),
    )
    mixed = [
        "get_return_status",
        "add_to_cart",
        "checkout",
        "get_order",
        "remove_from_wishlist",
    ]
    valid = ["get_wishlist", "remove_from_wishlist"]

    retained = subject._filter_feasible_chains(
        [mixed, valid],
        "shopping",
        {"entity_ids": [], "observed_entity_count": 0},
    )

    assert retained == [valid]
