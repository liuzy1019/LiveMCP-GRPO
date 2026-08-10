"""Cached canonical contracts for code paths that only receive a domain name."""

from __future__ import annotations

import importlib
from functools import lru_cache

from src.live_mcp.contracts.factory import build_contract_registry
from src.live_mcp.contracts.registry import ContractRegistry


@lru_cache(maxsize=None)
def domain_contract_registry(domain: str) -> ContractRegistry:
    module = importlib.import_module(f"src.live_mcp.servers.{domain}.server")
    tools = getattr(module, "TOOLS")
    return build_contract_registry({domain: tools})
