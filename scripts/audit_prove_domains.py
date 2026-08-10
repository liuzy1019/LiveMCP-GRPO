#!/usr/bin/env python3
"""Read-only ten-domain certification for the PROVE generation prerequisites."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.live_mcp.config import load_suite_config
from src.live_mcp.contracts.factory import build_contract_registry
from src.live_mcp.executor import LiveMCPExecutor
from src.live_mcp.orchestrator import TaskOrchestrator
from src.live_mcp.protocol.manager import LiveMCPManager


def _server_handler_audit(domain: str, tool_names: set[str]) -> dict[str, Any]:
    module = importlib.import_module(f"src.live_mcp.servers.{domain}.server")
    server_class = next(
        value
        for name, value in vars(module).items()
        if isinstance(value, type)
        and hasattr(value, "handle_request")
        and name != "StatefulToolServer"
    )
    registered = set(server_class().handlers)
    return {
        "missing_handlers": sorted(tool_names - registered),
        "extra_handlers": sorted(registered - tool_names),
    }


def _edge_count(graph: dict[str, Any]) -> dict[str, int]:
    return {
        relation: sum(
            len(node.get(relation, []))
            for node in graph.values()
            if isinstance(node, dict)
        )
        for relation in ("explicit", "implicit")
    }


_AUDITED_STATE_FIELDS = frozenset({
    "status", "state", "stage", "type", "frozen", "read", "archived", "open",
    "recurring", "cart_member", "wishlist_member", "review_eligible",
    "readonly", "is_dir", "active", "deletable", "payment_status",
    "dispute_open", "protected", "ownership_change_allowed",
})
_AUDITED_COLLECTION_FIELDS = frozenset({
    "recurrence", "reminders", "attendees", "menu",
})


def _entity_state_distributions(
    entity_records: list[dict[str, Any]],
) -> dict[str, dict[str, dict[str, int]]]:
    """Summarize persisted scalar state without domain-specific branches."""
    counts: dict[str, dict[str, Counter[str]]] = {}
    for item in entity_records:
        entity_type = str(item.get("type") or "")
        data = item.get("data") or {}
        if not entity_type or not isinstance(data, dict):
            continue
        fields = counts.setdefault(entity_type, {})
        for key, value in data.items():
            if key not in _AUDITED_STATE_FIELDS:
                if key not in _AUDITED_COLLECTION_FIELDS:
                    continue
                value = "present" if value else "empty"
            fields.setdefault(key, Counter())[str(value)] += 1
        if "stock" in data:
            availability = "available" if float(data["stock"]) > 0 else "empty"
            fields.setdefault("stock_availability", Counter())[availability] += 1
    return {
        entity_type: {
            field: dict(sorted(values.items()))
            for field, values in sorted(fields.items())
        }
        for entity_type, fields in sorted(counts.items())
        if fields
    }


def _classify_target_availability(
    availability: dict[str, dict[str, Any]],
    feasible_chains: list[list[str]],
) -> tuple[dict[str, dict[str, Any]], list[str], dict[str, dict[str, Any]]]:
    """Separate baseline availability from state-machine reachability."""
    baseline_unusable = {
        name: values
        for name, values in availability.items()
        if not values["has_usable_entities"]
    }
    state_machine_reachable = sorted({
        str(chain[-1])
        for chain in feasible_chains
        if len(chain) > 1 and str(chain[-1]) in baseline_unusable
    })
    genuinely_unreachable = {
        name: values
        for name, values in baseline_unusable.items()
        if name not in state_machine_reachable
    }
    return baseline_unusable, state_machine_reachable, genuinely_unreachable


def audit_domains(domains: list[str], seed: int) -> dict[str, Any]:
    suite = load_suite_config("configs/live_mcp/ten_domain_suite.yaml")
    manager = LiveMCPManager(suite)
    manager.start_suite()
    try:
        executor = LiveMCPExecutor(manager, manager.registry)
        client = SimpleNamespace(
            contract_model_id="models/Google/Gemma-4-31B-it",
            model_path="models/Google/Gemma-4-31B-it",
        )
        orchestrator = TaskOrchestrator(
            suite, manager, executor, client,
            prompt_profile="paper_generation_baseline_v1",
        )
        report: dict[str, Any] = {}
        for domain in domains:
            tools = manager.registry.server_tools(domain)
            tool_names = {str(tool.get("name") or "") for tool in tools}
            contract_registry = build_contract_registry({domain: tools})
            contract_issues = contract_registry.audit_schema(domain, tools)
            annotations_valid = all(
                bool((tool.get("annotations") or {}).get("readonly"))
                != bool((tool.get("annotations") or {}).get("mutating"))
                for tool in tools
            )
            handler_audit = _server_handler_audit(domain, tool_names)
            schema_hash = TaskOrchestrator._tool_schema_hash(tools, domain)
            cache = orchestrator._load_dependency_cache(
                domain, schema_hash, tools,
            )
            dependency: dict[str, Any] = {
                "cache_current": cache is not None,
                "schema_hash": schema_hash,
            }
            if cache is not None:
                cache_path = orchestrator._graph_cache_path(
                    domain,
                    schema_hash,
                    orchestrator._classifier_contract_hash(domain),
                )
                payload = json.loads(cache_path.read_text(encoding="utf-8"))
                dependency.update({
                    "cache_path": str(cache_path),
                    "expected_pair_count": int(payload["expected_pair_count"]),
                    "classified_pair_count": int(payload["classified_pair_count"]),
                    "raw_edge_count": _edge_count(payload["raw_graph"]),
                    "relation_audit_counts": dict(
                        payload.get("relation_audit_counts") or {}
                    ),
                    "graph_source": str(payload.get("graph_source") or ""),
                    "raw_graph_source": str(
                        payload.get("raw_graph_source") or ""
                    ),
                })
            session = manager.create_session(seed=seed, server_names=[domain])
            try:
                context = orchestrator._probe_live_sampling_context(
                    session.session_id, domain, tools,
                )
                availability = context["target_tool_availability"]
                entity_counts: dict[str, int] = {}
                for item in context["entity_ids"]:
                    entity_type = str(item.get("type") or "")
                    if entity_type:
                        entity_counts[entity_type] = (
                            entity_counts.get(entity_type, 0) + 1
                        )
                if cache is not None:
                    graph = cache
                    cache_key = orchestrator._dependency_cache_key(domain)
                    orchestrator._domain_graphs[cache_key] = graph
                    chains = orchestrator._extract_dependency_chains(domain)
                    feasible = orchestrator._filter_feasible_chains(
                        chains, domain, context,
                    )
                    dependency.update({
                        "edge_count": _edge_count(graph),
                        "sampled_chain_count": len(chains),
                        "live_feasible_chain_count": len(feasible),
                    })
                else:
                    feasible = []
                (
                    baseline_unusable,
                    state_machine_reachable,
                    genuinely_unreachable,
                ) = _classify_target_availability(availability, feasible)
                report[domain] = {
                    "tools": {
                        "schema_count": len(tools),
                        "annotations_valid": annotations_valid,
                        "contract_count": len(contract_registry.domain(domain)),
                        "contract_issues": list(contract_issues),
                        **handler_audit,
                    },
                    "entities": {
                        "observed": context["observed_entity_count"],
                        "with_observed_record": context[
                            "record_observed_entity_count"
                        ],
                        "by_type": dict(sorted(entity_counts.items())),
                        "state_distributions": _entity_state_distributions(
                            context["entity_records"],
                        ),
                    },
                    # A target can be unavailable in baseline state yet valid
                    # under PROVE when an accepted predecessor establishes its
                    # required entity/state.  Keep that distinct from a tool
                    # which is neither directly usable nor chain-reachable.
                    "baseline_unusable_target_tools": baseline_unusable,
                    "state_machine_reachable_target_tools": (
                        state_machine_reachable
                    ),
                    "genuinely_unreachable_target_tools": genuinely_unreachable,
                    "dependency": dependency,
                }
            finally:
                manager.close_session(session.session_id)
        return report
    finally:
        manager.stop_suite()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", default="all")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    suite = load_suite_config("configs/live_mcp/ten_domain_suite.yaml")
    enabled = [server.name for server in suite.servers if server.enabled]
    domains = enabled if args.domain == "all" else [
        item.strip() for item in args.domain.split(",") if item.strip()
    ]
    unknown = sorted(set(domains) - set(enabled))
    if unknown:
        parser.error(f"unknown or disabled domains: {unknown}")
    print(json.dumps(audit_domains(domains, args.seed), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
