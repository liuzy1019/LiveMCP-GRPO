"""Readonly discovery arguments and public entity summary fields."""

from __future__ import annotations

from typing import Any

_DISCOVERY_TOOL_PREFIXES = (
    "list_", "search_", "get_free_busy", "check_conflicts",
    "get_working_hours", "export_calendar",
    "get_exchange_rate", "list_categories", "get_coupons",
    "get_wishlist", "get_cart", "get_time_report", "get_user_status",
    "pwd", "ls", "cat", "stat", "head", "tail", "find", "grep",
    "tree", "du", "df", "file_info", "md5sum", "sha256sum", "wc",
    "xxd", "sort", "uniq", "cut", "sed", "awk", "diff",
)


_ENTITY_ID_FIELD_TYPES: dict[str, str] = {
    "event_id": "event", "account_id": "account", "from_account": "account",
    "to_account": "account", "invoice_id": "invoice", "payment_id": "payment",
    "webhook_id": "webhook", "refund_id": "refund", "dispute_id": "dispute",
    "email_id": "email", "thread_id": "thread", "draft_id": "draft",
    "filter_id": "filter", "lead_id": "lead", "contact_id": "contact",
    "deal_id": "deal", "task_id": "task", "note_id": "note",
    "issue_id": "issue", "sprint_id": "sprint", "subtask_id": "subtask",
    "entry_id": "time_entry", "restaurant_id": "restaurant", "order_id": "order",
    "return_id": "return", "product_id": "product", "channel_id": "channel",
    "message_id": "message", "dm_id": "dm", "ticket_id": "ticket",
    "user_id": "user",
    "transfer_id": "scheduled_transfer", "scheduled_txn_id": "scheduled_transfer",
}

_DOMAIN_ENTITY_ID_FIELD_TYPES: dict[str, dict[str, str]] = {
    "filesystem": {
        "path": "file", "source": "file", "target": "file",
        "file1": "file", "file2": "file", "archive": "file",
        "link_path": "file",
    },
    "issue_tracker": {"assignee": "user", "user": "user"},
    "team_chat": {"recipient": "user"},
}


_READONLY_REQUIRED_PROBE_ARGS: dict[str, dict[str, dict[str, Any]]] = {
    "calendar": {
        "search_events": {"query": ""},
        "get_free_busy": {
            "emails": ["current_user@example.com"],
            "start_time": "2026-06-20T00:00",
            "end_time": "2026-06-30T23:59",
        },
        "check_conflicts": {
            "start_time": "2026-06-25T10:00",
            "end_time": "2026-06-25T11:00",
        },
    },
    "team_chat": {
        "search_messages": {"query": ""},
    },
    "food_delivery": {
        "search_restaurants": {"query": ""},
    },
}



_COMMON_ENTITY_SUMMARY_FIELDS = (
    "name", "title", "subject", "owner", "status", "type",
    "balance", "amount", "price", "stage", "priority",
    "author", "content", "timestamp", "created_at", "updated_at",
    "sender", "recipient", "channel_id", "thread_id",
    "root_message_id", "restaurant_id", "invoice_id", "payment_id",
    "lead_id", "contact_id", "frozen", "stock", "quantity",
    "currency", "due_date", "cuisine", "rating", "total",
    "member_count", "start_time", "end_time", "location",
    "reminders", "recurrence", "attendees", "labels", "watchers",
    "members", "wishlist_member", "cart_member",
)

# Handler-dependent facts belong to typed projections.  A single ever-growing
# generic whitelist hides which business invariant each field supports and
# Entity summaries include all qualified facts returned by discovery.
_DOMAIN_ENTITY_SUMMARY_FIELDS: dict[tuple[str, str], tuple[str, ...]] = {
    ("filesystem", "file"): ("permissions", "size", "path"),
    ("email", "email"): ("read", "archived", "date", "labels", "thread_id"),
    ("shopping", "product"): ("wishlist_member", "review_eligible"),
    ("shopping", "cart_item"): ("cart_member",),
    ("shopping", "order"): ("item_count", "item_names"),
    ("issue_tracker", "issue"): (
        "state", "assignee", "sprint_id", "milestone",
    ),
    ("team_chat", "channel"): ("archived", "description"),
    ("team_chat", "message"): ("reactions", "thread_id", "channel_id"),
    ("food_delivery", "order"): (
        "tip", "subtotal", "delivery_fee", "delivery_address",
    ),
    ("payments", "invoice"): (
        "total_refunded", "refund_id", "payment_status",
    ),
}
