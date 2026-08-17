from __future__ import annotations

import pytest

from src.live_mcp.registry.schemas import SchemaRegistry
from src.live_mcp.servers.banking.server import TOOLS as BANKING_TOOLS
from src.live_mcp.servers.crm.server import TOOLS as CRM_TOOLS
from src.live_mcp.servers.email.server import TOOLS as EMAIL_TOOLS
from src.live_mcp.servers.food_delivery.server import TOOLS as FOOD_DELIVERY_TOOLS
from src.live_mcp.servers.payments.server import TOOLS as PAYMENTS_TOOLS
from src.live_mcp.servers.shopping.server import TOOLS as SHOPPING_TOOLS
from src.live_mcp.servers.team_chat.server import TOOLS as TEAM_CHAT_TOOLS


def _registry() -> SchemaRegistry:
    registry = SchemaRegistry()
    for domain, tools in (
        ("banking", BANKING_TOOLS),
        ("crm", CRM_TOOLS),
        ("email", EMAIL_TOOLS),
        ("food_delivery", FOOD_DELIVERY_TOOLS),
        ("payments", PAYMENTS_TOOLS),
        ("shopping", SHOPPING_TOOLS),
        ("team_chat", TEAM_CHAT_TOOLS),
    ):
        registry.register_tools(domain, tools)
    return registry


@pytest.mark.parametrize(
    ("domain", "tool_name", "invalid_args", "valid_args", "error_fragment"),
    [
        (
            "banking",
            "transfer",
            {"from_account": "a", "to_account": "b", "amount": -1},
            {"from_account": "a", "to_account": "b", "amount": 1},
            "amount: must be > 0",
        ),
        (
            "payments",
            "create_invoice",
            {"customer": "customer", "amount": -1},
            {"customer": "customer", "amount": 1},
            "amount: must be > 0",
        ),
        (
            "shopping",
            "add_to_cart",
            {"product_id": "product", "quantity": -1},
            {"product_id": "product", "quantity": 1},
            "quantity: must be >= 1",
        ),
        (
            "crm",
            "create_deal",
            {"name": "deal", "amount": -1, "contact_id": "contact"},
            {"name": "deal", "amount": 1, "contact_id": "contact"},
            "amount: must be > 0",
        ),
    ],
)
def test_cross_domain_numeric_mutations_reject_negative_values(
    domain: str,
    tool_name: str,
    invalid_args: dict,
    valid_args: dict,
    error_fragment: str,
) -> None:
    registry = _registry()

    invalid = registry.validate_arguments(tool_name, invalid_args, domain=domain)
    assert invalid.valid is False
    assert error_fragment in invalid.type_errors

    valid = registry.validate_arguments(tool_name, valid_args, domain=domain)
    assert valid.valid is True


@pytest.mark.parametrize("tool_name", ["get_order", "get_thread"])
def test_same_name_tools_require_an_owner_domain(tool_name: str) -> None:
    registry = _registry()
    arguments = {
        "get_order": {"order_id": "order"},
        "get_thread": {"thread_id": "thread"},
    }[tool_name]

    assert registry.get_schema(tool_name) is None
    assert registry.server_for_tool(tool_name, arguments) is None

    owners = {
        "get_order": ("shopping", "food_delivery"),
        "get_thread": ("email", "team_chat"),
    }[tool_name]
    for owner in owners:
        assert registry.get_schema(tool_name, domain=owner) is not None
        assert registry.server_for_tool(
            tool_name, arguments, domain=owner,
        ) == owner
