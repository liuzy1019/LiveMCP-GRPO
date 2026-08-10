"""Per-tool existing, relevant, and primary probe entity types."""

from __future__ import annotations

_DOMAIN_TOOL_REQUIREMENTS: dict[str, dict[str, set[str]]] = {
    "calendar": {
        "get_event": {"event"},
        "update_event": {"event"},
        "delete_event": {"event"},
        "add_attendee": {"event"},
        "remove_attendee": {"event"},
        "set_reminder": {"event"},
        "respond_to_event": {"event"},
        "get_recurring_info": {"event"},
    },
    "banking": {
        "get_balance": {"account"},
        "get_history": {"account"},
        "get_statement": {"account"},
        "deposit": {"account"},
        "withdraw": {"account"},
        "transfer": {"account"},
        "wire_transfer": {"account"},
        "bill_pay": {"account"},
        "schedule_transfer": {"account"},
        "freeze_account": {"account"},
        "unfreeze_account": {"account"},
        "cancel_transfer": {"scheduled_transfer"},
    },
    "payments": {
        "get_invoice": {"invoice"},
        "list_invoices": {"invoice"},
        "pay_invoice": {"invoice"},
        "refund_invoice": {"invoice"},
        "dispute_invoice": {"invoice"},
        "cancel_payment": {"payment"},
        "list_webhooks": {"webhook"},
        "delete_webhook": {"webhook"},
    },
    "email": {
        "get_email": {"email"},
        "list_inbox": {"email"},
        "search_emails": {"email"},
        "mark_read": {"email"},
        "mark_unread": {"email"},
        "archive_email": {"email"},
        "add_label": {"email"},
        "remove_label": {"email"},
        "reply_email": {"email"},
        "forward_email": {"email"},
        "get_attachments": {"email"},
        "move_to_thread": {"email", "thread"},
        "get_thread": {"thread"},
    },
    "filesystem": {
        "cat": {"file"},
        "stat": {"file"},
        "head": {"file"},
        "tail": {"file"},
        "grep": {"file"},
        "file_info": {"file"},
        "md5sum": {"file"},
        "sha256sum": {"file"},
        "wc": {"file"},
        "xxd": {"file"},
        "chmod": {"file"},
        "chown": {"file"},
        "rm": {"file"},
        "cp": {"file"},
        "mv": {"file"},
        "truncate": {"file"},
        "split": {"file"},
        "tar_create": {"file"},
        "tar_extract": {"file"},
        "zip": {"file"},
        "unzip": {"file"},
    },
    "crm": {
        "list_leads": {"lead"},
        "update_lead": {"lead"},
        "convert_lead": {"lead"},
        "delete_lead": {"lead"},
        "list_deals": {"deal"},
        "get_deal": {"deal"},
        "update_deal": {"deal"},
        "list_tasks": {"task"},
        "complete_task": {"task"},
    },
    "issue_tracker": {
        "get_issue": {"issue"},
        "list_issues": {"issue"},
        "assign_issue": {"issue", "user"},
        "comment_issue": {"issue"},
        "transition_issue": {"issue"},
        "add_label": {"issue"},
        "remove_label": {"issue"},
        "add_watcher": {"issue", "user"},
        "remove_watcher": {"issue", "user"},
        "set_milestone": {"issue"},
        "add_to_sprint": {"issue", "sprint"},
        "remove_from_sprint": {"issue"},
        "create_subtask": {"issue"},
        "list_subtasks": {"issue"},
        "time_track": {"issue"},
        "list_sprints": {"sprint"},
    },
    "shopping": {
        "get_product": {"product"},
        "compare_products": {"product"},
        "add_to_cart": {"product"},
        "update_cart_quantity": {"cart_item"},
        "remove_from_cart": {"cart_item"},
        "clear_cart": {"cart_item"},
        "checkout": {"cart_item"},
        "get_order": {"order"},
        "cancel_order": {"order"},
        "return_order": {"order"},
        "get_return_status": {"return"},
        "add_review": {"product"},
        "get_reviews": {"product"},
        "add_to_wishlist": {"product"},
        "remove_from_wishlist": {"product"},
    },
    "team_chat": {
        "get_channel": {"channel"},
        "send_message": {"channel"},
        "archive_channel": {"channel"},
        "react_message": {"channel", "message"},
        "create_thread": {"channel", "message"},
        "get_thread": {"thread"},
        "send_dm": {"user"},
    },
    "food_delivery": {
        "get_restaurant": {"restaurant"},
        "get_menu": {"restaurant"},
        "filter_by_dietary": {"restaurant"},
        "get_popular_items": {"restaurant"},
        "create_order": {"restaurant"},
        "get_order": {"order"},
        "list_orders": {"order"},
        "get_estimated_time": {"order"},
        "track_rider": {"order"},
        "cancel_order": {"order"},
        "add_tip": {"order"},
        "reorder": {"order"},
        "rate_order": {"order"},
        "contact_support": {"order"},
    },
}


# ── Domain-specific entity data-quality predicates ──────────────────
# Each predicate returns (qualified: bool, reason: str).  Only domains
# with explicit predicates are filtered; domains not listed here use a
# conservative all-pass fallback.

_DOMAIN_TOOL_RELEVANT: dict[str, dict[str, set[str]]] = {
    "calendar": {"list_events": {"event"}, "search_events": {"event"}, "get_free_busy": {"event"}, "check_conflicts": {"event"}},
    "banking": {"list_accounts": {"account"}, "list_scheduled_transfers": {"scheduled_transfer", "account"}, "get_exchange_rate": {"account"}, "apply_loan": {"account"}},
    "payments": {"get_invoice": {"invoice", "payment"}, "list_invoices": {"invoice", "payment"}, "list_webhooks": {"webhook"}},
    "email": {"list_inbox": {"email"}, "search_emails": {"email"}, "create_draft": {"email", "draft"}},
    "filesystem": {name: {"file"} for name in (
        "pwd", "ls", "find", "tree", "du", "df", "sort", "uniq", "cut",
        "sed", "awk", "split", "diff", "readlink", "truncate",
        "tar_create", "tar_extract", "zip", "unzip", "symlink",
    )},
    "crm": {
        "list_leads": {"lead"},
        "list_deals": {"deal"}, "list_tasks": {"task"},
        "create_deal": {"lead", "contact", "deal"},
        "create_task": {"deal", "contact", "task"},
        "add_note": {"lead", "contact", "deal", "note"},
    },
    "issue_tracker": {"list_issues": {"issue"}, "list_sprints": {"sprint"}, "list_members": {"user"}, "create_subtask": {"issue", "user"}},
    "shopping": {
        "search_products": {"product"}, "list_categories": {"product"},
        "get_coupons": {"product"}, "apply_coupon": {"cart_item"},
        "get_cart": {"cart_item", "product"},
        "get_wishlist": {"wishlist", "product"},
        "list_orders": {"order"},
    },
    "team_chat": {"list_channels": {"channel"}, "get_user_status": {"user"}, "search_messages": {"channel", "message"}},
    "food_delivery": {"search_restaurants": {"restaurant"}, "list_orders": {"order"}},
}


# Entity types whose full record is directly returned by a readonly discovery
# tool. Other IDs in the same record are foreign-key references and must not be
# enriched with the source record's fields. For example, list_deals returns a
# deal record containing contact_id; that ID establishes contact identity but
# the deal's name/amount/stage are not contact facts.
_DOMAIN_PROBE_PRIMARY_ENTITY_TYPES: dict[str, dict[str, set[str]]] = {
    "calendar": {
        "list_events": {"event"}, "search_events": {"event"},
        "get_free_busy": {"event"}, "check_conflicts": {"event"},
    },
    "banking": {
        "list_accounts": {"account"},
        "list_scheduled_transfers": {"scheduled_transfer"},
    },
    "payments": {
        "list_invoices": {"invoice"}, "list_webhooks": {"webhook"},
    },
    "email": {
        "list_inbox": {"email"}, "search_emails": {"email"},
    },
    "filesystem": {
        name: {"file"} for name in (
            "pwd", "ls", "find", "tree", "du", "df", "sort", "uniq",
            "cut", "sed", "awk", "split", "diff", "readlink",
        )
    },
    "crm": {
        "list_leads": {"lead"},
        "list_deals": {"deal"}, "list_tasks": {"task"},
        "get_deal": {"deal", "contact", "lead"},
    },
    "issue_tracker": {
        "list_issues": {"issue"},
        "list_sprints": {"sprint"}, "list_members": {"user"},
    },
    "shopping": {
        "search_products": {"product"}, "get_cart": {"cart_item"},
        "get_wishlist": {"wishlist"}, "list_orders": {"order"},
    },
    "team_chat": {
        "list_channels": {"channel"}, "get_user_status": {"user"},
        "search_messages": {"message"},
    },
    "food_delivery": {
        "search_restaurants": {"restaurant"}, "list_orders": {"order"},
    },
}
