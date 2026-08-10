"""Declarative readonly detail probes for PROVE Step-2 supporting data."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable
from typing import Any

from src.live_mcp.dependency_value_flow import (
    _dependency_value_key,
    _field_values,
)
from src.live_mcp.domain_contracts.outputs import DOMAIN_VALUE_OUTPUT_FIELDS
from src.live_mcp.types import ToolCall


@dataclass(frozen=True)
class DetailProbe:
    entity_type: str
    tool_name: str
    id_argument: str
    observation_key: str = ""
    project_entities: bool = False


DETAIL_PROBES: dict[str, tuple[DetailProbe, ...]] = {
    "shopping": (
        DetailProbe("product", "get_product", "product_id", "product"),
        DetailProbe("product", "get_reviews", "product_id"),
        DetailProbe("return", "get_return_status", "return_id", "return"),
    ),
    "food_delivery": (
        DetailProbe("restaurant", "get_menu", "restaurant_id"),
    ),
    "email": (
        DetailProbe("thread", "get_thread", "thread_id"),
    ),
    "team_chat": (
        DetailProbe("thread", "get_thread", "thread_id", "thread"),
    ),
    "crm": (
        DetailProbe(
            "deal", "get_deal", "deal_id", "deal",
            project_entities=True,
        ),
    ),
    "filesystem": (
        DetailProbe("file", "stat", "path"),
    ),
}


def enrich_readonly_entity_records(
    *,
    executor: Any,
    session_id: str,
    server_name: str,
    entity_records: list[dict[str, Any]],
    add_entity: Callable[[str, str, dict[str, Any] | None], None] | None = None,
) -> list[dict[str, Any]]:
    """Merge public readonly detail observations into discovered records.

    Every call is made through the normal executor and rejected if it mutates
    state.  Returned audit entries make the additional supporting-data probes
    visible to downstream diagnostics.
    """
    audits: list[dict[str, Any]] = []
    for spec in DETAIL_PROBES.get(server_name, ()):
        for record in list(entity_records):
            if str(record.get("type") or "") != spec.entity_type:
                continue
            entity_id = str(record.get("id") or "")
            if not entity_id:
                continue
            arguments = {spec.id_argument: entity_id}
            result = executor.execute(
                session_id,
                ToolCall(
                    spec.tool_name,
                    arguments,
                    call_id=f"detail_{spec.tool_name}_{entity_id}",
                ),
                domain=server_name,
            )
            audit = {
                "tool": spec.tool_name,
                "arguments": arguments,
                "entity_type": spec.entity_type,
                "entity_id": entity_id,
                "success": bool(result.success),
                "state_changed": bool(result.state_changed),
                "error_type": result.error_type,
            }
            # Populate output_field_counts/values so chain_is_feasible can
            # join detail probe outputs to downstream required arguments.
            observation = result.observation if result.success and not result.state_changed else {}
            output_field_map: dict[str, list[Any]] = {}
            for field_name in DOMAIN_VALUE_OUTPUT_FIELDS.get(server_name, {}).get(spec.tool_name, ()):
                unique: dict[str, Any] = {}
                for value in _field_values(observation, field_name):
                    if isinstance(value, (str, int, float, bool)):
                        unique.setdefault(_dependency_value_key(value), value)
                output_field_map[field_name] = list(unique.values())[:100]
            audit["output_field_counts"] = {
                field: len(values) for field, values in output_field_map.items()
            }
            audit["output_field_values"] = output_field_map
            audits.append(audit)
            if not result.success or result.state_changed:
                continue
            observation = result.observation
            if spec.project_entities and add_entity is not None:
                from src.live_mcp.live_state_projection import extract_probe_entities
                extract_probe_entities(
                    observation,
                    add_entity,
                    domain=server_name,
                    tool_name=spec.tool_name,
                )
            detail = (
                observation.get(spec.observation_key)
                if spec.observation_key and isinstance(observation, dict)
                else observation
            )
            if not isinstance(detail, dict):
                continue
            data = record.get("data")
            if not isinstance(data, dict):
                data = {}
                record["data"] = data
            data.update(detail)
    return audits


def _readonly_audit(
    result: Any,
    *,
    tool_name: str,
    arguments: dict[str, Any],
    domain: str,
) -> dict[str, Any]:
    output_values: dict[str, list[Any]] = {}
    for field_name in DOMAIN_VALUE_OUTPUT_FIELDS.get(domain, {}).get(
        tool_name, (),
    ):
        unique: dict[str, Any] = {}
        for value in _field_values(result.observation, field_name):
            if isinstance(value, (str, int, float, bool)):
                unique.setdefault(_dependency_value_key(value), value)
        output_values[field_name] = list(unique.values())[:100]
    return {
        "tool": tool_name,
        "arguments": arguments,
        "success": bool(result.success),
        "state_changed": bool(result.state_changed),
        "error_type": result.error_type,
        "output_field_counts": {
            field: len(values) for field, values in output_values.items()
        },
        "output_field_values": output_values,
    }


def _enrich_shopping(
    *,
    executor: Any,
    session_id: str,
    domain: str,
    tool_schemas: list[dict[str, Any]],
    entity_records: list[dict[str, Any]],
    probe_results: list[dict[str, Any]],
    add_entity: Callable[[str, str, dict[str, Any] | None], None],
) -> list[dict[str, Any]]:
    """Derive shopping membership/review facts from public readonly calls."""
    if any(
        item.get("tool") == "get_wishlist" and item.get("success")
        for item in probe_results
    ):
        for record in entity_records:
            if record.get("type") == "product":
                data = record.setdefault("data", {})
                if isinstance(data, dict):
                    data.setdefault("wishlist_member", False)

    if not any(
        str(schema.get("name") or "") == "get_order"
        for schema in tool_schemas
    ):
        return []

    product_names = {
        str(record.get("id") or ""): str(
            (record.get("data") or {}).get("name") or ""
        )
        for record in entity_records
        if record.get("type") == "product"
        and isinstance(record.get("data"), dict)
    }
    order_ids = [
        str(record.get("id") or "")
        for record in entity_records
        if record.get("type") == "order" and record.get("id")
    ]
    review_eligible: set[str] = set()
    audits: list[dict[str, Any]] = []
    for order_id in order_ids:
        arguments = {"order_id": order_id}
        result = executor.execute(
            session_id,
            ToolCall(
                "get_order", arguments,
                call_id=f"live_probe_get_order_{order_id}",
            ),
            domain=domain,
        )
        audits.append(_readonly_audit(
            result,
            tool_name="get_order",
            arguments=arguments,
            domain=domain,
        ))
        if not result.success or result.state_changed:
            continue
        detail = (
            result.observation.get("order")
            if isinstance(result.observation, dict)
            else None
        )
        if not isinstance(detail, dict):
            continue
        items = [
            item for item in (detail.get("items") or ())
            if isinstance(item, dict)
        ]
        if str(detail.get("status") or "") in {
            "shipped", "returning", "returned",
        }:
            review_eligible.update(
                str(item.get("product_id") or "")
                for item in items
                if str(item.get("product_id") or "")
            )
        item_names = [
            product_names.get(str(item.get("product_id") or ""), "")
            for item in items
        ]
        add_entity(order_id, "order", {
            **detail,
            "item_count": sum(
                int(item.get("quantity", 0) or 0) for item in items
            ),
            "item_names": [name for name in item_names if name],
        })
    for record in entity_records:
        if record.get("type") != "product":
            continue
        data = record.setdefault("data", {})
        if isinstance(data, dict):
            data["review_eligible"] = str(record.get("id") or "") in review_eligible
    return audits


DomainEnricher = Callable[..., list[dict[str, Any]]]

DOMAIN_ENRICHERS: dict[str, tuple[DomainEnricher, ...]] = {
    "shopping": (_enrich_shopping,),
}


def enrich_domain_entity_records(
    *,
    executor: Any,
    session_id: str,
    domain: str,
    tool_schemas: list[dict[str, Any]],
    entity_records: list[dict[str, Any]],
    probe_results: list[dict[str, Any]],
    add_entity: Callable[[str, str, dict[str, Any] | None], None],
) -> list[dict[str, Any]]:
    """Run registered readonly enrichers without domain branches upstream."""
    audits: list[dict[str, Any]] = []
    for enricher in DOMAIN_ENRICHERS.get(domain, ()):
        audits.extend(enricher(
            executor=executor,
            session_id=session_id,
            domain=domain,
            tool_schemas=tool_schemas,
            entity_records=entity_records,
            probe_results=probe_results,
            add_entity=add_entity,
        ))
    return audits
