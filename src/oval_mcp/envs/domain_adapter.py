"""Reward-domain adapter registry.

Domain facts live in the domain_adapters package; this module is the stable
consumer entry point and contains only construction/registration logic.
"""

from __future__ import annotations

from src.oval_mcp.envs.domain_adapters.base import DomainAdapter
from src.oval_mcp.envs.domain_adapters.commerce import (
    FoodDeliveryAdapter,
    PaymentsAdapter,
    ShoppingAdapter,
)
from src.oval_mcp.envs.domain_adapters.operations import (
    BankingAdapter,
    CRMAdapter,
    FilesystemAdapter,
    IssueTrackerAdapter,
)
from src.oval_mcp.envs.domain_adapters.productivity import (
    CalendarAdapter,
    EmailAdapter,
    TeamChatAdapter,
)


_ADAPTER_TYPES: dict[str, type[DomainAdapter]] = {
    adapter.domain_name: adapter
    for adapter in (
        CalendarAdapter,
        ShoppingAdapter,
        BankingAdapter,
        EmailAdapter,
        FilesystemAdapter,
        PaymentsAdapter,
        CRMAdapter,
        IssueTrackerAdapter,
        TeamChatAdapter,
        FoodDeliveryAdapter,
    )
}
_ADAPTERS: dict[str, DomainAdapter] = {}


def get_adapter(domain_name: str) -> DomainAdapter:
    """Get or create one schema-bound domain adapter."""
    if domain_name not in _ADAPTERS:
        try:
            adapter_type = _ADAPTER_TYPES[domain_name]
        except KeyError as exc:
            raise ValueError(f"unknown domain: {domain_name}") from exc
        adapter = adapter_type()
        import importlib

        tools = list(importlib.import_module(
            f"src.live_mcp.servers.{domain_name}.server"
        ).TOOLS)
        adapter.register_tool_schemas(tools)
        _ADAPTERS[domain_name] = adapter
    return _ADAPTERS[domain_name]


__all__ = [
    "DomainAdapter",
    "CalendarAdapter",
    "ShoppingAdapter",
    "BankingAdapter",
    "EmailAdapter",
    "FilesystemAdapter",
    "PaymentsAdapter",
    "CRMAdapter",
    "IssueTrackerAdapter",
    "TeamChatAdapter",
    "FoodDeliveryAdapter",
    "get_adapter",
]
