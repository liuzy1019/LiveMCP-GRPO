from src.live_mcp.contracts.factory import build_contract_registry
from src.live_mcp.live_state_feasibility import chain_is_feasible
from src.live_mcp.servers.banking.server import TOOLS as BANKING_TOOLS
from src.live_mcp.servers.crm.server import TOOLS as CRM_TOOLS
from src.live_mcp.servers.shopping.server import TOOLS as SHOPPING_TOOLS


def test_minimum_entity_cardinality_is_enforced() -> None:
    registry = build_contract_registry({"banking": BANKING_TOOLS})
    context = {
        "entity_ids": [{"type": "account", "id": "account-1"}],
        "entity_records": [{
            "type": "account",
            "id": "account-1",
            "data": {"balance": 100, "frozen": False},
        }],
        "probe_results": [],
    }

    ok, reason = chain_is_feasible(
        ["transfer"], "banking", context, registry,
    )

    assert ok is False
    assert "account(1/2)" in reason


def test_any_of_entity_group_requires_one_live_alternative() -> None:
    registry = build_contract_registry({"crm": CRM_TOOLS})
    missing = {"entity_ids": [], "entity_records": [], "probe_results": []}
    lead = {
        "entity_ids": [{"type": "lead", "id": "lead-1"}],
        "entity_records": [{
            "type": "lead", "id": "lead-1", "data": {"status": "new"},
        }],
        "probe_results": [],
    }

    assert chain_is_feasible(
        ["create_deal"], "crm", missing, registry,
    )[0] is False
    assert chain_is_feasible(
        ["create_deal"], "crm", lead, registry,
    )[0] is True


def test_prior_transition_can_satisfy_global_cart_precondition() -> None:
    registry = build_contract_registry({"shopping": SHOPPING_TOOLS})
    context = {
        "entity_ids": [{"type": "product", "id": "product-1"}],
        "entity_records": [{
            "type": "product",
            "id": "product-1",
            "data": {"stock": 3},
        }],
        "probe_results": [{
            "tool": "get_cart",
            "success": True,
            "state_changed": False,
            "output_field_counts": {},
            "output_field_values": {},
        }],
    }

    assert chain_is_feasible(
        ["checkout"], "shopping", context, registry,
    )[0] is False
    assert chain_is_feasible(
        ["add_to_cart", "checkout"], "shopping", context, registry,
    )[0] is True
