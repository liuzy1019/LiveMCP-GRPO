"""User-visible entity reference facts for the local trainable profile.

PROVE may expose any grounded live-state identifier.  This registry is the
stricter local policy that distinguishes a business reference a user can
reasonably quote from a sampler/backend handle used only for tool execution.
"""

from __future__ import annotations

import re
from typing import Any, Iterable


_SAMPLER_HANDLE_RE = re.compile(r"^[A-Za-z][A-Za-z0-9]*_s\d+_\d+$")

# Entity types whose canonical IDs are execution references unless a binding
# field explicitly declares a separate public business reference.
DOMAIN_OPAQUE_ENTITY_TYPES: dict[str, frozenset[str]] = {
    "banking": frozenset({"account", "scheduled_transfer", "transaction"}),
    "calendar": frozenset({"event"}),
    "crm": frozenset({"contact", "deal", "lead", "note", "task"}),
    "email": frozenset({"draft", "email", "thread"}),
    "food_delivery": frozenset({"order", "restaurant", "ticket"}),
    "issue_tracker": frozenset({"issue", "sprint", "time_entry"}),
    "payments": frozenset({"invoice", "payment", "refund", "webhook"}),
    "shopping": frozenset({"order", "product", "return"}),
    "team_chat": frozenset({"channel", "dm", "message", "thread"}),
}
# Values are visibility modes, keyed by canonical entity type and binding
# field. Both modes require a non-sampler value. A deterministic sampler
# handle is execution provenance, never a public business reference.
PUBLIC_REFERENCE_FIELDS: dict[
    str, dict[tuple[str, str], str]
] = {
    "banking": {
        ("transaction", "txn_id"): "business",
    },
    "crm": {
        ("deal", "deal_id"): "business",
        ("lead", "lead_id"): "business",
        ("note", "note_id"): "business",
        ("task", "task_id"): "business",
    },
    "food_delivery": {
        ("order", "order_id"): "business",
        ("ticket", "ticket_id"): "business",
    },
    "filesystem": {
        ("file", "path"): "natural",
    },
    "issue_tracker": {
        ("issue", "issue_id"): "business",
        ("sprint", "sprint_id"): "business",
        ("subtask", "subtask_id"): "business",
    },
    "payments": {
        ("dispute", "dispute_id"): "business",
        ("invoice", "invoice_id"): "business",
        ("payment", "payment_id"): "business",
        ("refund", "refund_id"): "business",
    },
    "shopping": {
        ("order", "order_id"): "business",
        ("return", "return_id"): "business",
        ("review", "review_id"): "business",
    },
    "team_chat": {
        ("channel", "channel_id"): "natural",
    },
}


def is_public_entity_reference(
    domain: str,
    entity_type: str,
    field: str,
    value: Any,
) -> bool:
    """Return whether one typed binding is a user-facing reference."""
    if not isinstance(value, str) or not value:
        return False
    if is_sampler_private_handle(value):
        return False
    mode = PUBLIC_REFERENCE_FIELDS.get(domain, {}).get((entity_type, field))
    return mode in {"business", "natural"}


def is_sampler_private_handle(value: Any) -> bool:
    """Return whether a value carries deterministic sampler provenance."""
    return bool(
        isinstance(value, str) and _SAMPLER_HANDLE_RE.fullmatch(value)
    )


def record_exposes_entity_reference(
    domain: str,
    entity_type: str,
    entity_id: str,
    record: dict[str, Any],
) -> bool:
    """Apply the same field contract to a live-state record."""
    return any(
        value == entity_id
        and is_public_entity_reference(domain, entity_type, field, value)
        for field, value in record.items()
    )


def public_entity_reference_ids_from_records(
    domain: str,
    records: Iterable[dict[str, Any]],
    *,
    entity_types: set[str] | frozenset[str] | None = None,
) -> set[str]:
    """Return backend IDs explicitly declared as public business references.

    The current Live-State is authoritative even when an early-stop trace has
    not yet echoed the reference through a successful tool call.
    """
    allowed_types = set(entity_types or ())
    public_ids: set[str] = set()
    for record in records:
        if not isinstance(record, dict):
            continue
        entity_type = str(record.get("type") or "")
        entity_id = str(record.get("id") or "")
        data = record.get("data")
        if (
            not entity_type
            or not entity_id
            or not isinstance(data, dict)
            or (allowed_types and entity_type not in allowed_types)
        ):
            continue
        if record_exposes_entity_reference(
            domain, entity_type, entity_id, data,
        ):
            public_ids.add(entity_id)
    return public_ids


__all__ = [
    "DOMAIN_OPAQUE_ENTITY_TYPES",
    "PUBLIC_REFERENCE_FIELDS",
    "is_public_entity_reference",
    "is_sampler_private_handle",
    "public_entity_reference_ids_from_records",
    "record_exposes_entity_reference",
]
