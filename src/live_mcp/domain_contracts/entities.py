"""Tool-to-entity creation and reference facts."""

from __future__ import annotations

_TOOL_ENTITY_OVERRIDE: dict[str, str] = {
    # Shopping domain
    "checkout": "order",          # checkout completes an order
    "get_cart": "order",          # cart IS the order-in-progress
    "clear_cart": "order",
    "add_to_cart": "order",
    "remove_from_cart": "order",
    "update_cart_quantity": "order",
    "rate_order": "order",
    "return_order": "order",
    "reorder": "order",
    "apply_coupon": "order",
    # Banking domain
    "get_balance": "account",     # balance is a property of account
    "get_history": "account",     # transaction history belongs to account
    "get_statement": "account",   # statement belongs to account
    "transfer": "account",        # transfer moves money between accounts
    "wire_transfer": "account",
    "deposit": "account",
    "withdraw": "account",
    "apply_loan": "account",
    "bill_pay": "account",
    # Payments domain
    "pay_invoice": "invoice",
    "get_invoice": "invoice",
    "refund_invoice": "invoice",
    "cancel_payment": "payment",
    # Calendar / email domains
    "add_to_wishlist": "wishlist",  # shopping: wishlist entity (not in keyword list)
    "move_to_thread": "email",    # email threading
    "list_inbox": "email",        # inbox listing resolves email entities
    "get_thread": "email",        # thread lookup resolves email entities (for reply/forward)
    "get_attachments": "email",   # attachment lookup resolves email entities
    "mark_read": "email",         # operates on email entities
    "mark_unread": "email",       # operates on email entities
    "add_label": "email",         # labels an email entity
    "remove_label": "email",      # removes label from email entity
    # Filesystem domain — cmd tools operate on file/directory entities
    "chmod": "file",
    "chown": "file",
    "cp": "file",
    "rm": "file",
    "mv": "file",
    "mkdir": "file",
    "touch": "file",
}

_DOMAIN_TOOL_ENTITY_OVERRIDE: dict[str, dict[str, str]] = {
    "banking": {
        "schedule_transfer": "scheduled_transfer",
        "cancel_transfer": "scheduled_transfer",
    },
    "issue_tracker": {
        "add_label": "issue",
        "remove_label": "issue",
        "add_watcher": "issue",
        "remove_watcher": "issue",
        "set_milestone": "issue",
        "time_track": "issue",
        "add_to_sprint": "issue",
        "remove_from_sprint": "issue",
        "create_subtask": "issue",
    },
    "team_chat": {
        "get_thread": "thread",
        "get_user_status": "user",
        "send_dm": "user",
    },
    "calendar": {
        "add_attendee": "event",
        "remove_attendee": "event",
        "set_reminder": "event",
        "create_recurring": "event",
    },
    "food_delivery": {
        "get_menu": "restaurant",
        "filter_by_dietary": "restaurant",
        "get_popular_items": "restaurant",
        "add_tip": "order",
        "contact_support": "order",
        "rate_order": "order",
    },
    "filesystem": {
        "ls": "file",
        "cat": "file",
        "stat": "file",
        "head": "file",
        "tail": "file",
        "find": "file",
        "grep": "file",
        "tree": "file",
        "pwd": "file",
        "du": "file",
        "df": "file",
    },
}


# ── Secondary entity resolution ──
# Some "get" tools return JOINed entities — e.g. get_deal returns
# {deal, contact, lead}.  These tools resolve more than their primary
# entity, so _detect_missing_dependency should credit them as valid
# preceding reads for all resolved entities.
_TOOL_SECONDARY_ENTITIES: dict[str, set[str]] = {
    # CRM: get_deal returns linked contact + lead alongside the deal itself
    "get_deal": {"lead", "contact"},
}

_DOMAIN_TOOL_SECONDARY_ENTITIES: dict[str, dict[str, set[str]]] = {
    # Email get_thread returns the full email records in that thread.  The
    # same-named team_chat tool returns messages and must not inherit this.
    "email": {"get_thread": {"email"}},
}


def _tool_entity(name: str, domain: str = "") -> str:
    """Extract the conceptual entity a tool operates on.

    Checks override map first, then entity keyword list, then fallback.
    """
    tool = name.lower()
    server = domain.lower()
    if server and tool in _DOMAIN_TOOL_ENTITY_OVERRIDE.get(server, {}):
        return _DOMAIN_TOOL_ENTITY_OVERRIDE[server][tool]
    if tool in _TOOL_ENTITY_OVERRIDE:
        return _TOOL_ENTITY_OVERRIDE[tool]
    for et in ("event", "order", "account", "email", "invoice",
                "issue", "lead", "deal", "product", "restaurant",
                "channel", "message", "file", "contact", "payment",
                "menu", "cart", "transfer", "transaction"):
        if et in name:
            return et
    return name.split("_")[-1] if "_" in name else name


def _format_graph_hints(graph: dict) -> str:
    if not graph:
        return ""
    lines = ["## Tool Dependency Hints"]
    for tool, deps in sorted(graph.items()):
        parts = []
        if deps.get("explicit"):
            parts.append("→ " + ", ".join(deps["explicit"]))
        if deps.get("implicit") and not deps.get("explicit"):
            parts.append("~ " + ", ".join(deps["implicit"][:3]))
        if parts:
            lines.append(f"  {tool} {'; '.join(parts)}")
    return "\n".join(lines)


_CREATED_ENTITY_BY_TOOL: dict[str, set[str]] = {
    "create_event": {"event"},
    "create_recurring": {"event"},
    "create_invoice": {"invoice"},
    "pay_invoice": {"payment"},
    "refund_invoice": {"refund"},
    "dispute_invoice": {"dispute"},
    "schedule_transfer": {"scheduled_transfer"},
    "create_webhook": {"webhook"},
    # NOTE: send_email intentionally omitted. It produces an outgoing message,
    # not an inbox email that subsequent tools (get_email/reply/archive/mark_read/…)
    # can operate on. Including it would create invalid outgoing-message edges.
    # send_email → {get_email, reply_email, forward_email, …} edges which are
    # semantically reversed. Chains needing "send then observe" are not
    # expressible in this domain's tool surface.
    "create_draft": {"draft"},
    "create_filter": {"filter"},
    "mkdir": {"file"},
    "touch": {"file"},
    "cp": {"file"},
    "mv": {"file"},
    "create_lead": {"lead"},
    "create_contact": {"contact"},
    "convert_lead": {"contact"},
    "create_deal": {"deal"},
    "create_task": {"task"},
    "add_note": {"note"},
    "create_issue": {"issue"},
    "create_sprint": {"sprint"},
    "create_subtask": {"subtask"},
    "time_track": {"time_entry"},
    "add_to_cart": {"cart_item"},
    "checkout": {"order"},
    "create_order": {"order"},
    "reorder": {"order"},
    "return_order": {"return"},
    "create_channel": {"channel"},
    "send_message": {"message"},
    "create_thread": {"thread"},
    "send_dm": {"dm"},
    "contact_support": {"ticket"},
}

# Verified handler-output fields supplied only to dependency classification.
# They do not change the Teacher/Policy-visible MCP schemas.
