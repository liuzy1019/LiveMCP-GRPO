"""Validated registry for canonical PROVE domain contracts."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

from src.live_mcp.contracts.models import ToolContract


class ContractRegistry:
    """Own the one authoritative contract for each ``domain.tool`` pair."""

    def __init__(self, contracts: Iterable[ToolContract] = ()) -> None:
        self._contracts: dict[tuple[str, str], ToolContract] = {}
        for contract in contracts:
            self.register(contract)

    def register(self, contract: ToolContract) -> None:
        issues = contract.validate()
        if issues:
            rendered = "; ".join(issues)
            raise ValueError(f"Invalid contract {contract.domain}.{contract.name}: {rendered}")
        key = (contract.domain, contract.name)
        if key in self._contracts:
            raise ValueError(f"Duplicate tool contract: {contract.domain}.{contract.name}")
        self._contracts[key] = contract

    def get(self, domain: str, tool_name: str) -> ToolContract:
        try:
            return self._contracts[(domain, tool_name)]
        except KeyError as exc:
            raise KeyError(f"Missing tool contract: {domain}.{tool_name}") from exc

    def domain(self, domain: str) -> tuple[ToolContract, ...]:
        return tuple(
            contract
            for (registered_domain, _), contract in sorted(self._contracts.items())
            if registered_domain == domain
        )

    def audit_schema(
        self,
        domain: str,
        server_tools: Iterable[dict],
    ) -> tuple[str, ...]:
        schemas = {
            str(schema.get("name") or ""): schema for schema in server_tools
        }
        contracts = {contract.name: contract for contract in self.domain(domain)}
        issues: list[str] = []
        for missing in sorted(set(schemas) - set(contracts)):
            issues.append(f"missing contract for {domain}.{missing}")
        for stale in sorted(set(contracts) - set(schemas)):
            issues.append(f"contract has no live schema: {domain}.{stale}")
        for name in sorted(set(schemas) & set(contracts)):
            schema = schemas[name]
            contract = contracts[name]
            input_schema = schema.get("input_schema") or schema.get("inputSchema") or {}
            required = frozenset(str(value) for value in input_schema.get("required") or ())
            annotations = schema.get("annotations") or {}
            if contract.required_arguments != required:
                issues.append(f"required arguments drift: {domain}.{name}")
            if contract.readonly != bool(annotations.get("readonly")):
                issues.append(f"readonly annotation drift: {domain}.{name}")
            if contract.mutating != bool(annotations.get("mutating")):
                issues.append(f"mutating annotation drift: {domain}.{name}")
        return tuple(issues)

    def coverage(self) -> dict[str, int]:
        counts: dict[str, int] = defaultdict(int)
        for domain, _ in self._contracts:
            counts[domain] += 1
        return dict(sorted(counts.items()))
