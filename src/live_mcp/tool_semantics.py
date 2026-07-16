"""Canonical cross-layer semantics for public LiveMCP tools."""

from __future__ import annotations

import importlib
from functools import lru_cache
from dataclasses import dataclass
from typing import Any, Literal


ToolOperation = Literal["query", "create", "update", "delete"]


_MUTATION_OPERATIONS: dict[str, dict[str, ToolOperation]] = {
    "banking": {name: "update" for name in (
        "transfer", "wire_transfer", "deposit", "withdraw", "bill_pay",
        "schedule_transfer", "cancel_transfer", "freeze_account",
        "unfreeze_account", "apply_loan",
    )},
    "calendar": {
        "create_event": "create", "update_event": "update",
        "delete_event": "delete", "create_recurring": "create",
        "add_attendee": "update", "remove_attendee": "update",
        "set_reminder": "update", "change_timezone": "update",
        "respond_to_event": "update",
    },
    "crm": {
        "create_lead": "create", "update_lead": "update",
        "convert_lead": "update", "delete_lead": "delete",
        "create_contact": "create", "update_contact": "update",
        "delete_contact": "delete", "create_deal": "create",
        "update_deal": "update", "create_task": "create",
        "complete_task": "update", "add_note": "create",
    },
    "email": {
        "send_email": "create", "create_draft": "create",
        "forward_email": "create", "reply_email": "create",
        "add_label": "update", "remove_label": "update",
        "move_to_thread": "update", "archive_email": "update",
        "mark_read": "update", "mark_unread": "update",
        "create_filter": "create",
    },
    "filesystem": {
        "cd": "update", "mkdir": "create", "touch": "create",
        "mv": "update", "cp": "create", "rm": "delete",
        "chmod": "update", "chown": "update", "umask": "update",
        "symlink": "create", "tar_create": "create",
        "tar_extract": "create", "zip": "create", "unzip": "create",
        "truncate": "update", "split": "create",
    },
    "food_delivery": {
        "create_order": "create", "update_order_status": "update",
        "cancel_order": "update", "rate_order": "update",
        "add_tip": "update", "reorder": "create",
        "contact_support": "create",
    },
    "issue_tracker": {
        "create_issue": "create", "update_issue": "update",
        "assign_issue": "update", "transition_issue": "update",
        "comment_issue": "update", "add_label": "update",
        "remove_label": "update", "add_watcher": "update",
        "remove_watcher": "update", "create_sprint": "create",
        "add_to_sprint": "update", "remove_from_sprint": "update",
        "create_subtask": "create", "time_track": "create",
        "set_milestone": "update",
    },
    "payments": {
        "create_invoice": "create", "pay_invoice": "update",
        "refund_invoice": "update", "cancel_payment": "update",
        "dispute_invoice": "update", "create_webhook": "create",
        "delete_webhook": "delete",
    },
    "shopping": {
        "add_to_cart": "update", "update_cart_quantity": "update",
        "remove_from_cart": "update", "clear_cart": "update",
        "apply_coupon": "update", "checkout": "create",
        "return_order": "create", "add_review": "create",
        "add_to_wishlist": "update", "remove_from_wishlist": "update",
    },
    "team_chat": {
        "create_channel": "create", "archive_channel": "update",
        "send_message": "create", "send_dm": "create",
        "create_thread": "create", "react_message": "update",
    },
}


@dataclass(frozen=True)
class ToolSemantics:
    domain: str
    name: str
    operation: ToolOperation
    sensitive_params: tuple[str, ...]
    allowed_state_roots: tuple[str, ...]


def _same_roots(names: tuple[str, ...], *roots: str) -> dict[str, tuple[str, ...]]:
    return {name: tuple(roots) for name in names}


_MUTATION_STATE_ROOTS: dict[str, dict[str, tuple[str, ...]]] = {
    "banking": {
        **_same_roots(("transfer", "wire_transfer", "deposit", "withdraw", "bill_pay"), "accounts", "transactions", "next_txn_num"),
        **_same_roots(("schedule_transfer",), "scheduled_transfers", "next_txn_num"),
        **_same_roots(("cancel_transfer",), "scheduled_transfers"),
        **_same_roots(("freeze_account", "unfreeze_account"), "accounts", "freeze_log"),
        **_same_roots(("apply_loan",), "loans", "next_txn_num"),
    },
    "calendar": {
        **_same_roots(("create_event", "create_recurring"), "events", "next_event_num"),
        **_same_roots(("update_event", "delete_event", "add_attendee", "remove_attendee", "set_reminder", "respond_to_event"), "events"),
        **_same_roots(("change_timezone",), "timezone"),
    },
    "crm": {
        **_same_roots(("create_lead",), "leads", "next_lead_num"),
        **_same_roots(("update_lead", "delete_lead"), "leads"),
        **_same_roots(("convert_lead",), "leads", "contacts", "next_contact_num"),
        **_same_roots(("create_contact",), "contacts", "next_contact_num"),
        **_same_roots(("update_contact", "delete_contact"), "contacts"),
        **_same_roots(("create_deal",), "deals", "next_deal_num"),
        **_same_roots(("update_deal",), "deals"),
        **_same_roots(("create_task",), "tasks", "next_task_num"),
        **_same_roots(("complete_task",), "tasks"),
        **_same_roots(("add_note",), "notes", "next_note_num"),
    },
    "email": {
        **_same_roots(("send_email", "forward_email", "reply_email"), "emails", "threads", "inbox_order", "next_email_num", "next_thread_num"),
        **_same_roots(("create_draft",), "drafts", "next_email_num"),
        **_same_roots(("add_label", "remove_label", "archive_email", "mark_read", "mark_unread"), "emails", "inbox_order"),
        **_same_roots(("move_to_thread",), "emails", "threads"),
        **_same_roots(("create_filter",), "filters"),
    },
    "filesystem": {
        **_same_roots(("cd",), "cwd"),
        **_same_roots(("umask",), "umask"),
        **_same_roots(("mkdir", "touch", "mv", "cp", "rm", "chmod", "chown", "symlink", "tar_create", "tar_extract", "zip", "unzip", "truncate", "split"), "fs", "cwd"),
    },
    "food_delivery": {
        **_same_roots(("create_order", "reorder"), "orders", "next_order_num"),
        **_same_roots(("update_order_status", "cancel_order", "rate_order", "add_tip"), "orders"),
        **_same_roots(("contact_support",), "support_tickets", "next_ticket_num"),
    },
    "issue_tracker": {
        **_same_roots(("create_issue",), "issues", "next_issue_num"),
        **_same_roots(("create_subtask",), "subtasks", "next_subtask_num"),
        **_same_roots(("update_issue", "assign_issue", "transition_issue", "comment_issue", "add_label", "remove_label", "add_watcher", "remove_watcher", "add_to_sprint", "remove_from_sprint", "set_milestone"), "issues", "sprints"),
        **_same_roots(("create_sprint",), "sprints", "next_sprint_num"),
        **_same_roots(("time_track",), "time_entries", "next_time_entry_num"),
    },
    "payments": {
        **_same_roots(("create_invoice",), "invoices", "next_inv_num"),
        **_same_roots(("pay_invoice",), "invoices", "payments", "next_pay_num"),
        **_same_roots(("refund_invoice",), "invoices", "payments", "refunds", "next_ref_num"),
        **_same_roots(("cancel_payment",), "payments", "invoices"),
        **_same_roots(("dispute_invoice",), "invoices", "disputes", "next_inv_num"),
        **_same_roots(("create_webhook",), "webhooks", "next_wh_num"),
        **_same_roots(("delete_webhook",), "webhooks"),
    },
    "shopping": {
        **_same_roots(("add_to_cart", "update_cart_quantity", "remove_from_cart", "clear_cart"), "cart", "products", "applied_coupon"),
        **_same_roots(("apply_coupon",), "applied_coupon"),
        **_same_roots(("checkout",), "cart", "orders", "products", "applied_coupon", "next_order_num"),
        **_same_roots(("return_order",), "orders", "returns", "next_order_num"),
        **_same_roots(("add_review",), "reviews", "next_order_num"),
        **_same_roots(("add_to_wishlist", "remove_from_wishlist"), "wishlist"),
    },
    "team_chat": {
        **_same_roots(("create_channel",), "channels", "next_ch_num"),
        **_same_roots(("archive_channel",), "channels"),
        **_same_roots(("react_message",), "channels", "threads"),
        **_same_roots(("send_message",), "channels", "threads", "next_msg_num"),
        **_same_roots(("create_thread",), "channels", "threads", "next_thread_num"),
        **_same_roots(("send_dm",), "dms", "next_msg_num"),
    },
}


def build_tool_semantics(
    domain: str, tools: list[dict[str, Any]],
) -> dict[str, ToolSemantics]:
    semantics_by_name: dict[str, ToolSemantics] = {}
    mutations = _MUTATION_OPERATIONS.get(domain, {})
    for tool in tools:
        name = str(tool.get("name") or "")
        annotations = tool.get("annotations") or {}
        readonly = bool(annotations.get("readonly"))
        mutating = bool(annotations.get("mutating"))
        if readonly == mutating:
            raise ValueError(
                f"{domain}.{name} must be exactly one of readonly/mutating"
            )
        operation: ToolOperation
        if readonly:
            operation = "query"
        else:
            operation = mutations.get(name)  # type: ignore[assignment]
            if operation is None:
                raise ValueError(f"missing mutation operation for {domain}.{name}")
        allowed_roots = (
            () if readonly
            else _MUTATION_STATE_ROOTS.get(domain, {}).get(name)
        )
        if allowed_roots is None:
            raise ValueError(f"missing mutation footprint for {domain}.{name}")
        sensitive = annotations.get("sensitive_params", [])
        if isinstance(sensitive, bool) or not isinstance(sensitive, list):
            raise ValueError(
                f"{domain}.{name} sensitive_params must be a field-name list"
            )
        semantics_by_name[name] = ToolSemantics(
            domain=domain,
            name=name,
            operation=operation,
            sensitive_params=tuple(str(field) for field in sensitive),
            allowed_state_roots=tuple(allowed_roots),
        )
    return semantics_by_name


@lru_cache(maxsize=None)
def _public_tool_semantics(domain: str) -> dict[str, ToolSemantics]:
    domain_name = str(domain).lower()
    if domain_name not in _MUTATION_OPERATIONS:
        raise ValueError(f"unknown LiveMCP domain: {domain!r}")
    module = importlib.import_module(
        f"src.live_mcp.servers.{domain_name}.server"
    )
    return build_tool_semantics(domain_name, module.TOOLS)


def is_mutating_tool(name: str, domain: str | None = None) -> bool:
    """Resolve mutation semantics from the exact public tool definition."""
    return resolve_tool_operation(name, domain) != "query"


def resolve_tool_operation(
    name: str, domain: str | None = None,
) -> ToolOperation:
    """Resolve an exact formal operation; unknown or ambiguous tools fail."""
    tool_name = str(name).lower()
    if domain:
        domain_name = str(domain).lower()
        semantics = _public_tool_semantics(domain_name).get(tool_name)
        if semantics is None:
            raise ValueError(
                f"unknown public tool semantics: {domain_name}.{tool_name}"
            )
        return semantics.operation
    operations: set[ToolOperation] = set()
    matched_domains: list[str] = []
    for domain_name in _MUTATION_OPERATIONS:
        semantics = _public_tool_semantics(domain_name).get(tool_name)
        if semantics is not None:
            operations.add(semantics.operation)
            matched_domains.append(domain_name)
    if len(operations) == 1:
        return next(iter(operations))
    if not operations:
        raise ValueError(f"unknown public tool semantics: {name!r}")
    raise ValueError(
        f"ambiguous cross-domain tool operation: {name!r} in {matched_domains}"
    )


__all__ = [
    "ToolSemantics", "ToolOperation", "build_tool_semantics",
    "is_mutating_tool", "resolve_tool_operation",
]
