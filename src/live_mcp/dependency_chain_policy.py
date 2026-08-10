"""Generic evaluation of declarative dependency-chain constraints."""

from __future__ import annotations

from src.live_mcp.domain_contracts.chains import _DOMAIN_CHAIN_CONSTRAINTS


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


def chain_respects_contracts(server_name: str, chain: list[str]) -> bool:
    return chain_contract_issue(server_name, chain) is None
