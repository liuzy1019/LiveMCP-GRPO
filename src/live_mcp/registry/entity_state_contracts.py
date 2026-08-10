"""Readonly entity-state knowledge required before target-tool selection.

Each inner tuple is a group of equivalent observation fields; at least one
field in every group must be present before the target's state is considered
known.  Value suitability remains the handler-aligned predicate in the
orchestrator.  Keeping knowledge and suitability separate prevents a missing
field from being interpreted as a valid state.
"""

from __future__ import annotations

from typing import Any


_STATE_FIELD_GROUPS: dict[
    str, dict[str, dict[str, tuple[tuple[str, ...], ...]]]
] = {
    "banking": {
        "withdraw": {"account": (("frozen",), ("balance",))},
        "transfer": {"account": (("frozen",), ("balance",))},
        "wire_transfer": {"account": (("balance",),)},
        "bill_pay": {"account": (("balance",),)},
        "deposit": {"account": (("frozen",),)},
        "unfreeze_account": {"account": (("frozen",),)},
    },
    "calendar": {
        "get_recurring_info": {"event": (("recurrence",),)},
        "remove_attendee": {"event": (("attendees",),)},
        "respond_to_event": {"event": (("attendees",),)},
    },
    "crm": {
        "complete_task": {"task": (("status",),)},
    },
    "email": {
        "remove_label": {"email": (("labels",),)},
        "mark_read": {"email": (("read",),)},
        "mark_unread": {"email": (("read",),)},
        "archive_email": {"email": (("archived",),)},
    },
    "filesystem": {
        "readlink": {"file": (("type",),)},
        "tar_extract": {"file": (("type",),)},
        "join": {"file": (("type",),)},
    },
    "food_delivery": {
        "create_order": {"restaurant": (("menu", "items"),)},
        "track_rider": {"order": (("status",),)},
        "cancel_order": {"order": (("status",),)},
        "rate_order": {"order": (("status",),)},
        "add_tip": {"order": (("tip",),)},
    },
    "issue_tracker": {
        "remove_watcher": {"issue": (("watchers",),)},
        "remove_label": {"issue": (("labels",),)},
        "remove_from_sprint": {"issue": (("sprint_id",),)},
        "transition_issue": {"issue": (("state",),)},
    },
    "payments": {
        "pay_invoice": {"invoice": (("status",),)},
        "refund_invoice": {"invoice": (("status",),)},
        "dispute_invoice": {"invoice": (("status",),)},
        "cancel_payment": {"payment": (("status",),)},
    },
    "shopping": {
        "add_to_cart": {"product": (("stock", "available", "in_stock"),)},
        "checkout": {"cart_item": (("product_id", "id"),)},
        "update_cart_quantity": {"cart_item": (("product_id", "id"),)},
        "remove_from_cart": {"cart_item": (("product_id", "id"),)},
        "return_order": {"order": (("status",),)},
        "add_review": {"product": (("review_eligible",),)},
        "add_to_wishlist": {"product": (("wishlist_member",),)},
        "remove_from_wishlist": {"product": (("wishlist_member",),)},
    },
    "team_chat": {
        "send_message": {"channel": (("archived",),)},
    },
}


def required_state_field_groups(
    domain: str,
    tool_name: str,
    entity_type: str,
) -> tuple[tuple[str, ...], ...]:
    return _STATE_FIELD_GROUPS.get(domain, {}).get(tool_name, {}).get(
        entity_type, ()
    )


def entity_state_is_known(
    domain: str,
    tool_name: str,
    entity_type: str,
    record: dict[str, Any],
) -> bool:
    """Return whether all target-relevant state fields were observed."""
    groups = required_state_field_groups(domain, tool_name, entity_type)
    return all(any(field in record for field in alternatives) for alternatives in groups)


def state_contract_tools(domain: str) -> set[str]:
    """Expose covered tools for audits without leaking the mutable registry."""
    return set(_STATE_FIELD_GROUPS.get(domain, {}))
