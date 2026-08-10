"""Project readonly MCP observations into canonical entity records.

The traversal is domain-neutral.  Handler-specific response shapes are
registered as small projectors instead of branching in the orchestrator.
"""

from __future__ import annotations

import posixpath
from collections.abc import Callable
from typing import Any

from src.live_mcp.domain_contracts.probes import (
    _COMMON_ENTITY_SUMMARY_FIELDS,
    _DOMAIN_ENTITY_ID_FIELD_TYPES,
    _DOMAIN_ENTITY_SUMMARY_FIELDS,
    _ENTITY_ID_FIELD_TYPES,
)
from src.live_mcp.domain_contracts.requirements import (
    _DOMAIN_PROBE_PRIMARY_ENTITY_TYPES,
    _DOMAIN_TOOL_RELEVANT,
)


AddEntity = Callable[[str, str, dict[str, Any] | None], None]
Projector = Callable[[dict[str, Any], AddEntity], None]


def _project_payment(record: dict[str, Any], add: AddEntity) -> None:
    payment_id = record.get("payment_id")
    if isinstance(payment_id, str) and payment_id:
        add(payment_id, "payment", {
            "payment_id": payment_id,
            "invoice_id": record.get("invoice_id", ""),
            "status": record.get("payment_status", record.get("status", "")),
            "amount": record.get("amount"),
        })


def _project_filesystem(record: dict[str, Any], add: AddEntity) -> None:
    for field in ("cwd", "path"):
        path = record.get(field)
        if isinstance(path, str) and path:
            add(path, "file", record)
    parent = record.get("path")
    entries = record.get("entries")
    if not isinstance(parent, str) or not isinstance(entries, list):
        return
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        if isinstance(name, str) and name:
            child = posixpath.normpath(posixpath.join(parent, name))
            add(child, "file", {**entry, "path": child})


def _project_team_chat(record: dict[str, Any], add: AddEntity) -> None:
    message_id = record.get("message_id")
    if isinstance(message_id, str) and message_id:
        add(message_id, "message", {
            **record,
            "thread_id": record.get("thread_id"),
        })
    for member in record.get("members") or ():
        if isinstance(member, str) and member:
            add(member, "user", None)
    statuses = record.get("statuses")
    if isinstance(statuses, dict):
        for user_id, status in statuses.items():
            add(str(user_id), "user", {"status": status})


def _project_issue_users(record: dict[str, Any], add: AddEntity) -> None:
    for field in ("assignee", "user", "author"):
        value = record.get(field)
        if isinstance(value, str) and value:
            add(value, "user", None)
    for watcher in record.get("watchers") or ():
        if isinstance(watcher, str) and watcher:
            add(watcher, "user", None)


def _project_cart(record: dict[str, Any], add: AddEntity) -> None:
    for item in record.get("cart") or ():
        if isinstance(item, dict) and item.get("product_id"):
            data = {
                **item,
                "cart_member": True,
            }
            add(str(item["product_id"]), "cart_item", data)
            add(str(item["product_id"]), "product", data)


def _project_wishlist(record: dict[str, Any], add: AddEntity) -> None:
    for item in record.get("wishlist") or ():
        if isinstance(item, dict) and item.get("product_id"):
            add(str(item["product_id"]), "product", {
                **item,
                "wishlist_member": True,
            })


DOMAIN_PROJECTORS: dict[str, tuple[Projector, ...]] = {
    "payments": (_project_payment,),
    "filesystem": (_project_filesystem,),
    "team_chat": (_project_team_chat,),
    "issue_tracker": (_project_issue_users,),
}

TOOL_PROJECTORS: dict[tuple[str, str], tuple[Projector, ...]] = {
    ("shopping", "get_cart"): (_project_cart,),
    ("shopping", "get_wishlist"): (_project_wishlist,),
}


def extract_probe_entities(
    observation: Any,
    add_entity: AddEntity,
    *,
    domain: str,
    tool_name: str,
) -> None:
    """Recursively extract identities and attach data only to primary entities."""
    if isinstance(observation, list):
        for item in observation:
            extract_probe_entities(
                item, add_entity, domain=domain, tool_name=tool_name,
            )
        return
    if not isinstance(observation, dict):
        return

    primary_types = _DOMAIN_PROBE_PRIMARY_ENTITY_TYPES.get(domain, {}).get(
        tool_name,
        _DOMAIN_TOOL_RELEVANT.get(domain, {}).get(tool_name, set()),
    )
    field_types = {
        **_ENTITY_ID_FIELD_TYPES,
        **_DOMAIN_ENTITY_ID_FIELD_TYPES.get(domain, {}),
    }
    for field, entity_type in field_types.items():
        value = observation.get(field)
        if isinstance(value, str) and value:
            add_entity(
                value,
                entity_type,
                observation if entity_type in primary_types else None,
            )
    for projector in DOMAIN_PROJECTORS.get(domain, ()):
        projector(observation, add_entity)
    for projector in TOOL_PROJECTORS.get((domain, tool_name), ()):
        projector(observation, add_entity)
    for value in observation.values():
        if isinstance(value, (dict, list)):
            extract_probe_entities(
                value, add_entity, domain=domain, tool_name=tool_name,
            )


def _remaining_refundable(record: dict[str, Any]) -> float | None:
    amount = record.get("amount")
    refunded = record.get("total_refunded", 0)
    if not isinstance(amount, (int, float)) or not isinstance(
        refunded, (int, float),
    ):
        return None
    return max(0.0, float(amount) - float(refunded))


_FOOD_ORDER_TRANSITIONS = {
    "placed": ["confirmed", "cancelled"],
    "confirmed": ["preparing", "cancelled"],
    "preparing": ["delivering"],
    "delivering": ["delivered"],
    "delivered": [],
    "cancelled": [],
}


def _allowed_food_order_transitions(record: dict[str, Any]) -> list[str] | None:
    status = str(record.get("status") or "")
    return _FOOD_ORDER_TRANSITIONS.get(status)


DERIVED_SUMMARY_FIELDS: dict[
    tuple[str, str], dict[str, Callable[[dict[str, Any]], Any]]
] = {
    ("payments", "invoice"): {
        "remaining_refundable": _remaining_refundable,
    },
    ("food_delivery", "order"): {
        "allowed_next_status": _allowed_food_order_transitions,
    },
}


def format_entity_summary(
    entity_id: str,
    entity_type: str,
    data: dict[str, Any] | None = None,
    *,
    domain: str = "",
) -> str:
    key_fields: dict[str, Any] = {}
    if isinstance(data, dict):
        fields = (
            *_COMMON_ENTITY_SUMMARY_FIELDS,
            *_DOMAIN_ENTITY_SUMMARY_FIELDS.get((domain, entity_type), ()),
        )
        for field in dict.fromkeys(fields):
            if field not in data:
                continue
            value = data[field]
            if isinstance(value, str) and len(value) > 40:
                value = value[:37] + "..."
            elif isinstance(value, list) and len(value) > 8:
                value = [*value[:8], f"... ({len(value) - 8} more)"]
            key_fields[field] = value
        for field, derive in DERIVED_SUMMARY_FIELDS.get(
            (domain, entity_type), {},
        ).items():
            value = derive(data)
            if value is not None:
                key_fields[field] = value
    if key_fields:
        return f"  {entity_id} ({entity_type}): {key_fields}"
    return f"  {entity_id} ({entity_type})"
