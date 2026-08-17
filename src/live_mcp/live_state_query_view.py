"""Build the chain-aligned, anti-leakage view used by Query Teacher."""

from __future__ import annotations

from collections.abc import Callable
import json
from typing import Any

from src.live_mcp.contracts.chain_records import record_satisfies_chain
from src.live_mcp.contracts.factory import build_contract_registry
from src.live_mcp.contracts.models import ToolContract
from src.live_mcp.contracts.value_flow import value_bindings
from src.live_mcp.dependency_value_flow import _aggregate_probe_results_by_tool
from src.live_mcp.domain_contracts.reference_visibility import (
    DOMAIN_OPAQUE_ENTITY_TYPES,
    is_sampler_private_handle,
    record_exposes_entity_reference,
)


SELECTOR_FIELDS = (
    "name", "title", "subject", "customer", "owner", "status", "type",
    "frozen", "amount", "currency", "due_date", "category", "balance",
    "price", "quantity", "stage", "priority", "wishlist_member",
    "cart_member", "description", "stock", "available", "in_stock",
    "created_at", "date", "total", "item_count", "item_names",
    "review_eligible", "author", "channel_name", "content", "timestamp",
    "url", "events", "active", "sender", "recipient",
    "account_last4",
)

PUBLIC_ENTITY_SUMMARIES_KEY = "public_entity_summaries"


def _contains_private_id(value: Any, private_ids: set[str]) -> bool:
    if isinstance(value, str):
        return any(entity_id in value for entity_id in private_ids)
    if isinstance(value, dict):
        return any(
            _contains_private_id(key, private_ids)
            or _contains_private_id(item, private_ids)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple, set)):
        return any(_contains_private_id(item, private_ids) for item in value)
    return False


def _redact_private_ids(text: str, private_ids: set[str]) -> str:
    rendered = text
    for entity_id in sorted(private_ids, key=len, reverse=True):
        rendered = rendered.replace(entity_id, "<private reference omitted>")
    return rendered


def _public_selectors(
    record: dict[str, Any], private_ids: set[str],
) -> dict[str, Any]:
    return {
        field: record[field]
        for field in SELECTOR_FIELDS
        if field in record
        and not _contains_private_id(record[field], private_ids)
    }


def public_query_prompt_state(
    live_context: dict[str, Any],
    domain: str,
) -> dict[str, list[str]]:
    """Build the local natural-selector view for user-message generation.

    This is a local-trainable extension, not the PROVE paper baseline. Opaque
    backend IDs remain available to feasibility checks and action execution.
    """
    opaque_types = set(
        DOMAIN_OPAQUE_ENTITY_TYPES.get(domain, frozenset())
    )
    source_ids = [
        item for item in live_context.get("entity_ids", [])
        if isinstance(item, dict) and item.get("id")
    ]
    records_by_key = {
        (str(item.get("type") or ""), str(item.get("id") or "")): (
            item.get("data") if isinstance(item.get("data"), dict) else {}
        )
        for item in live_context.get("entity_records", [])
        if isinstance(item, dict)
    }
    private_ids = {
        str(item["id"])
        for item in source_ids
        if is_sampler_private_handle(str(item["id"]))
        or (
            str(item.get("type") or "") in opaque_types
            and not record_exposes_entity_reference(
                domain,
                str(item.get("type") or ""),
                str(item["id"]),
                records_by_key.get(
                    (str(item.get("type") or ""), str(item["id"])), {}
                ),
            )
        )
    }

    preselected = live_context.get("query_grounding_summaries")
    if isinstance(preselected, list) and preselected:
        return {
            PUBLIC_ENTITY_SUMMARIES_KEY: [
                _redact_private_ids(str(summary), private_ids)
                for summary in preselected[:50]
            ]
        }

    summaries = list(live_context.get("entity_summaries", []))
    grouped: dict[str, list[tuple[dict[str, Any], str]]] = {}
    for index, item in enumerate(source_ids):
        entity_type = str(item.get("type") or "entity")
        summary = str(summaries[index]) if index < len(summaries) else ""
        grouped.setdefault(entity_type, []).append((item, summary))

    selected: list[tuple[dict[str, Any], str]] = []
    offset = 0
    while len(selected) < 50:
        added = False
        for entity_type in sorted(grouped):
            bucket = grouped[entity_type]
            if offset < len(bucket):
                selected.append(bucket[offset])
                added = True
        if not added:
            break
        offset += 1

    rendered: list[str] = []
    for item, summary in selected:
        entity_id = str(item.get("id") or "")
        entity_type = str(item.get("type") or "entity")
        if entity_id in private_ids:
            record = records_by_key.get((entity_type, entity_id), {})
            selectors = _public_selectors(record, private_ids)
            if not selectors:
                continue
            rendered.append(
                f"  grounded {entity_type} candidate: "
                f"{json.dumps(selectors, ensure_ascii=False, sort_keys=True)}"
            )
        else:
            rendered.append(_redact_private_ids(summary, private_ids))
    return {PUBLIC_ENTITY_SUMMARIES_KEY: rendered}


def _required_types(contract: ToolContract) -> set[str]:
    required = set(contract.required_entity_types)
    required.update(
        predicate.subject.entity_type
        for group in contract.precondition_groups
        for predicate in group
        if predicate.subject.source == "argument"
        and predicate.observed_entity_required
    )
    return required


def _target_type_for_argument(
    target: ToolContract,
    argument: str,
) -> str | None:
    return next((
        binding.entity_type
        for binding in target.input_entities
        if binding.name == argument
    ), None)


def _provider_constraints(
    contracts: list[ToolContract],
    probe_results_by_tool: dict[str, dict[str, Any]],
) -> tuple[dict[str, set[str]], set[str]]:
    """Return observed provider IDs and types supplied only during execution."""
    allowed: dict[str, set[str]] = {}
    supplied_later: set[str] = set()
    for source_index, source in enumerate(contracts[:-1]):
        required_before = {
            entity_type
            for earlier in contracts[:source_index + 1]
            for entity_type in _required_types(earlier)
        }
        for target in contracts[source_index + 1:]:
            bindings = value_bindings(source.domain, source, target)
            for output_field, target_argument in bindings:
                entity_type = _target_type_for_argument(target, target_argument)
                if entity_type is None:
                    continue
                if source.readonly:
                    values = (
                        probe_results_by_tool.get(source.name, {})
                        .get("output_field_values", {})
                        .get(output_field, ())
                    )
                    bucket = allowed.setdefault(entity_type, set())
                    bucket.update(str(value) for value in values if str(value))
                elif entity_type not in required_before:
                    supplied_later.add(entity_type)
    return allowed, supplied_later


GroundingAugmenter = Callable[
    [list[dict[str, Any]]], list[str]
]


def _shopping_categories(records: list[dict[str, Any]]) -> list[str]:
    categories = sorted({
        str((item.get("data") or {}).get("category") or "").strip()
        for item in records
        if isinstance(item, dict)
        and str(item.get("type") or "") == "product"
        and isinstance(item.get("data"), dict)
        and str((item.get("data") or {}).get("category") or "").strip()
    })
    if not categories:
        return []
    return ["  available product categories: " + ", ".join(categories)]


GROUNDING_AUGMENTERS: dict[tuple[str, str], tuple[GroundingAugmenter, ...]] = {
    ("shopping", "get_recommendations"): (_shopping_categories,),
}


def compact_sampling_context(live_context: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": live_context.get("source", "live_readonly_probe"),
        "entity_ids": list(live_context.get("entity_ids", []))[:50],
        "entity_summaries": list(live_context.get("entity_summaries", []))[:50],
        "entity_records": list(live_context.get("entity_records", []))[:50],
        "entity_types": list(live_context.get("entity_types", [])),
        "probe_results": list(live_context.get("probe_results", [])),
    }


def live_context_to_prompt_state(
    live_context: dict[str, Any],
) -> dict[str, Any]:
    """Convert observed entities into the compact state formatter shape."""
    entity_source = live_context.get("entity_ids", [])
    summaries = list(live_context.get("entity_summaries", []))
    prompt_state: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(entity_source):
        if not isinstance(item, dict):
            continue
        entity_id = str(item.get("id") or "")
        entity_type = str(item.get("type") or "entity")
        if not entity_id:
            continue
        container = f"live_probe_{entity_type}s"
        prompt_state.setdefault(container, {})[entity_id] = {
            "id": entity_id,
            "type": entity_type,
            "source": "live_readonly_probe",
            "summary": summaries[index] if index < len(summaries) else "",
        }
    return prompt_state


def generation_query_prompt_state(
    live_context: dict[str, Any],
    domain: str,
    *,
    natural_selector: bool,
) -> dict[str, Any]:
    """Select the one profile-owned live-state view used by Query states."""
    if natural_selector:
        return public_query_prompt_state(live_context, domain)
    return live_context_to_prompt_state(live_context)


def extract_chain_context(
    chain: list[str],
    domain: str,
    live_context: dict[str, Any],
    server_tools: list[dict[str, Any]],
) -> dict[str, Any]:
    """Select records required by a chain without exposing backend-only IDs."""
    if not live_context or not chain:
        return {}
    if not server_tools:
        raise ValueError("server_tools are required for canonical chain context")

    registry = build_contract_registry({domain: server_tools})
    contracts = [registry.get(domain, name) for name in chain]
    probe_results = _aggregate_probe_results_by_tool(
        list(live_context.get("probe_results") or [])
    )
    provider_ids, supplied_later = _provider_constraints(
        contracts, probe_results,
    )
    relevant_types = {
        entity_type
        for contract in contracts
        for entity_type in _required_types(contract)
    } - supplied_later
    opaque_types = relevant_types & set(
        DOMAIN_OPAQUE_ENTITY_TYPES.get(domain, frozenset())
    )

    source_ids = live_context.get("entity_ids", [])
    source_records = live_context.get("entity_records", [])
    summaries_by_key = {
        (str(item.get("type") or ""), str(item.get("id") or "")): str(summary)
        for item, summary in zip(
            live_context.get("entity_ids", []),
            live_context.get("entity_summaries", []),
        )
        if isinstance(item, dict)
    }
    records_by_key = {
        (str(item.get("type") or ""), str(item.get("id") or "")): (
            item.get("data") if isinstance(item.get("data"), dict) else {}
        )
        for item in live_context.get("entity_records", [])
        if isinstance(item, dict)
    }
    hidden_entity_ids = {
        str(item.get("id") or "")
        for item in source_ids
        if isinstance(item, dict)
        and str(item.get("id") or "")
        and (
            is_sampler_private_handle(str(item.get("id") or ""))
            or (
                str(item.get("type") or "") in opaque_types
                and not record_exposes_entity_reference(
                    domain,
                    str(item.get("type") or ""),
                    str(item.get("id") or ""),
                    records_by_key.get(
                        (
                            str(item.get("type") or ""),
                            str(item.get("id") or ""),
                        ),
                        {},
                    ),
                )
            )
        )
    }
    hidden_types = {
        str(item.get("type") or "")
        for item in source_ids
        if isinstance(item, dict)
        and str(item.get("id") or "") in hidden_entity_ids
    }

    entity_ids: list[dict[str, str]] = []
    entity_summaries: list[str] = []
    entity_records: list[dict[str, Any]] = []
    grounding_summaries: list[str] = []
    seen: set[tuple[str, str]] = set()
    for item in source_ids:
        if not isinstance(item, dict):
            continue
        entity_id = str(item.get("id") or "")
        entity_type = str(item.get("type") or "")
        key = (entity_type, entity_id)
        if not entity_id or entity_type not in relevant_types or key in seen:
            continue
        if entity_type in provider_ids and entity_id not in provider_ids[entity_type]:
            continue
        record = records_by_key.get(key, {})
        if not record_satisfies_chain(
            registry, domain, chain, entity_type, record,
        ):
            continue
        seen.add(key)
        summary = summaries_by_key.get(key, f"  {entity_id} ({entity_type})")
        normalized = {"id": entity_id, "type": entity_type}
        entity_ids.append(normalized)
        entity_summaries.append(summary)
        entity_records.append({**normalized, "data": record})
        if entity_id in hidden_entity_ids:
            selector = _public_selectors(record, hidden_entity_ids)
            if selector:
                grounding_summaries.append(
                    f"  grounded {entity_type} candidate: {selector}"
                )
        else:
            grounding_summaries.append(summary)

    for tool_name in chain:
        for augmenter in GROUNDING_AUGMENTERS.get((domain, tool_name), ()):
            grounding_summaries.extend(augmenter(list(source_records)))
    return {
        "entity_ids": entity_ids[:30],
        "entity_summaries": entity_summaries[:30],
        "entity_records": entity_records[:30],
        "query_grounding_summaries": grounding_summaries[:30],
        "opaque_id_hidden_types": sorted(hidden_types),
        "opaque_id_hidden_ids": sorted(hidden_entity_ids),
        "relevant_types": sorted(relevant_types),
        "source": "live_readonly_probe",
    }
