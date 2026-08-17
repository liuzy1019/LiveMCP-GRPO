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
from src.live_mcp.dependency_value_flow import (
    _filter_relation_verifiable_chains,
)
from src.live_mcp.live_state_query_view import (
    SELECTOR_FIELDS,
    generation_query_prompt_state,
)
from src.live_mcp.domain_contracts.reference_visibility import (
    DOMAIN_OPAQUE_ENTITY_TYPES,
    record_exposes_entity_reference,
)
from src.live_mcp.orchestrator import TaskOrchestrator
from src.live_mcp.planner_format import format_state_compact
from src.live_mcp.prompt_profiles import PROMPT_PROFILES
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


def _query_state_view_audit(
    context: dict[str, Any],
    domain: str,
    *,
    natural_selector: bool,
) -> dict[str, Any]:
    """Audit the exact profile-selected state text sent to Query states."""
    state = generation_query_prompt_state(
        context, domain, natural_selector=natural_selector,
    )
    rendered = format_state_compact(state, max_entities=50)
    opaque_types = set(
        DOMAIN_OPAQUE_ENTITY_TYPES.get(domain, frozenset())
    )
    records_by_key = {
        (str(item.get("type") or ""), str(item.get("id") or "")): (
            item.get("data") if isinstance(item.get("data"), dict) else {}
        )
        for item in context.get("entity_records", [])
        if isinstance(item, dict)
    }
    candidate_references = {
        (str(item.get("type") or ""), str(item.get("id") or ""))
        for item in context.get("entity_ids", [])
        if isinstance(item, dict)
        and str(item.get("type") or "") in opaque_types
        and str(item.get("id") or "")
    }
    public_reference_ids = sorted({
        entity_id
        for entity_type, entity_id in candidate_references
        if record_exposes_entity_reference(
            domain,
            entity_type,
            entity_id,
            records_by_key.get((entity_type, entity_id), {}),
        )
    })
    opaque_ids = sorted({
        entity_id
        for entity_type, entity_id in candidate_references
        if not record_exposes_entity_reference(
            domain,
            entity_type,
            entity_id,
            records_by_key.get((entity_type, entity_id), {}),
        )
    })
    exposed_opaque_ids = [
        entity_id for entity_id in opaque_ids if entity_id in rendered
    ]
    public_summaries = state.get("public_entity_summaries", [])
    empty_selector_types = Counter(
        str(item).strip().split()[1]
        for item in public_summaries
        if str(item).rstrip().endswith("{}")
        and len(str(item).strip().split()) >= 2
    ) if isinstance(public_summaries, list) else Counter()
    empty_record_field_sets: dict[str, Counter[tuple[str, ...]]] = {}
    for item in context.get("entity_records", []):
        if not isinstance(item, dict):
            continue
        entity_type = str(item.get("type") or "")
        data = item.get("data") or {}
        if entity_type not in opaque_types or not isinstance(data, dict):
            continue
        if any(field in data for field in SELECTOR_FIELDS):
            continue
        empty_record_field_sets.setdefault(entity_type, Counter())[
            tuple(sorted(str(key) for key in data))
        ] += 1
    return {
        "view": "natural_selector" if natural_selector else "prove_real_id",
        "opaque_source_id_count": len(opaque_ids),
        "public_business_reference_count": len(public_reference_ids),
        "exposed_opaque_id_count": len(exposed_opaque_ids),
        "exposed_opaque_ids": exposed_opaque_ids[:5],
        "empty_selector_count": sum(empty_selector_types.values()),
        "empty_selector_types": dict(sorted(empty_selector_types.items())),
        "empty_selector_record_fields": {
            entity_type: [
                {"fields": list(fields), "count": count}
                for fields, count in field_sets.most_common(3)
            ]
            for entity_type, field_sets in sorted(
                empty_record_field_sets.items()
            )
        },
        "omitted_opaque_candidate_count": (
            sum(
                sum(field_sets.values())
                for field_sets in empty_record_field_sets.values()
            )
            if natural_selector else 0
        ),
        "omitted_opaque_candidate_types": (
            {
                entity_type: sum(field_sets.values())
                for entity_type, field_sets in sorted(
                    empty_record_field_sets.items()
                )
            }
            if natural_selector else {}
        ),
        "profile_contract_satisfied": (
            not exposed_opaque_ids and not empty_selector_types
            if natural_selector
            else len(exposed_opaque_ids) == len(opaque_ids)
        ),
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


def audit_domains(
    domains: list[str], seed: int, prompt_profile: str = "local_trainable_v1",
) -> dict[str, Any]:
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
            prompt_profile=prompt_profile,
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
                "prompt_profile": orchestrator.prompt_profile.name,
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
                    relation_verifiable, precheck_issues = (
                        _filter_relation_verifiable_chains(
                            feasible, graph, domain, tools,
                        )
                    )
                    production_eligible = (
                        relation_verifiable
                        if orchestrator.prompt_profile.dependency_necessary
                        else feasible
                    )
                    dependency.update({
                        "edge_count": _edge_count(graph),
                        "sampled_chain_count": len(chains),
                        "live_feasible_chain_count": len(feasible),
                        "relation_verifiable_chain_count": len(
                            relation_verifiable
                        ),
                        "production_eligible_chain_count": len(
                            production_eligible
                        ),
                        "relation_precheck_issue_counts": precheck_issues,
                    })
                else:
                    feasible = []
                    production_eligible = []
                (
                    baseline_unusable,
                    state_machine_reachable,
                    genuinely_unreachable,
                ) = _classify_target_availability(
                    availability, production_eligible,
                )
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
                    "query_state_view": _query_state_view_audit(
                        context,
                        domain,
                        natural_selector=(
                            orchestrator.prompt_profile.natural_selector
                        ),
                    ),
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
    parser.add_argument(
        "--prompt-profile",
        default="local_trainable_v1",
        choices=tuple(sorted(PROMPT_PROFILES)),
        help=(
            "chain eligibility profile; use local_trainable_v1 for training "
            "candidate audits and paper_generation_baseline_v1 only for "
            "paper-mechanism audits"
        ),
    )
    args = parser.parse_args()

    suite = load_suite_config("configs/live_mcp/ten_domain_suite.yaml")
    enabled = [server.name for server in suite.servers if server.enabled]
    domains = enabled if args.domain == "all" else [
        item.strip() for item in args.domain.split(",") if item.strip()
    ]
    unknown = sorted(set(domains) - set(enabled))
    if unknown:
        parser.error(f"unknown or disabled domains: {unknown}")
    print(json.dumps(
        audit_domains(domains, args.seed, args.prompt_profile),
        indent=2,
        sort_keys=True,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
