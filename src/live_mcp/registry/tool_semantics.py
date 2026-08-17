"""Canonical cross-layer semantics for public LiveMCP tools."""

from __future__ import annotations

import importlib
import json
from functools import lru_cache
from dataclasses import dataclass
from typing import Any, Literal


ToolOperation = Literal["query", "create", "update", "delete"]
ToolExecutionSemantics = Literal[
    "readonly", "state_transition", "action_execution",
]

# Successful writes whose externally visible effect may not be represented by
# a replay state delta in every transport implementation.
SELF_CONTAINED_WRITE_TOOLS: frozenset[str] = frozenset({
    "send_email", "send_message", "send_dm", "reply_email", "forward_email",
})


# These tools report a successful execution even when replaying the action
# produces no net state delta (for example, extracting archive entries that
# already exist with identical contents).  That is materially different from
# an idempotent state-transition request such as adding an existing attendee.
_ACTION_EXECUTION_TOOLS: dict[str, frozenset[str]] = {
    "filesystem": frozenset({"tar_extract", "unzip"}),
}


# Prefer the resource whose state is being mutated over routing/destination
# IDs such as thread_id, sprint_id, or channel_id.  This lets a retry correct
# a destination while still requiring it to operate on the same user target.
_MUTATION_TARGET_FIELD_PRIORITY: tuple[str, ...] = (
    "email_id", "issue_id", "message_id", "event_id", "order_id",
    "invoice_id", "payment_id", "product_id", "account_id", "lead_id",
    "contact_id", "deal_id", "task_id", "webhook_id", "transfer_id",
    "restaurant_id", "id",
)


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
        "cancel_order": "update",
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
    execution_semantics: ToolExecutionSemantics
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
        **_same_roots(("cancel_order",), "orders", "products"),
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


# Read footprints are certified for all production domains. Unknown domains
# fail closed through the conservative branch below.
_READ_STATE_ROOTS: dict[str, dict[str, tuple[str, ...]]] = {
    "banking": {
        "list_accounts": ("accounts",),
        "get_account_info": ("accounts",),
        "get_balance": ("accounts",),
        "get_history": ("accounts", "transactions"),
        "get_statement": ("accounts", "transactions"),
        "list_scheduled_transfers": ("scheduled_transfers",),
        "verify_account": ("accounts",),
        "get_exchange_rate": (),
    },
    "calendar": {
        "list_events": ("events",),
        "search_events": ("events",),
        "get_event": ("events",),
        "get_free_busy": ("events",),
        "check_conflicts": ("events",),
        "get_working_hours": ("timezone",),
        "export_calendar": ("events",),
        "get_recurring_info": ("events",),
    },
    "crm": {
        "list_contacts": ("contacts",),
        "list_leads": ("leads",),
        "list_deals": ("deals",),
        "list_tasks": ("tasks",),
        "get_deal": ("deals", "contacts", "leads"),
    },
    "email": {
        "list_inbox": ("emails", "inbox_order"),
        "search_emails": ("emails",),
        "get_email": ("emails",),
        "get_thread": ("emails", "threads"),
        "list_filters": ("filters",),
        "get_attachments": ("emails",),
    },
    "filesystem": {
        "ls": ("fs", "cwd"),
        "pwd": ("cwd",),
        "cat": ("fs", "cwd"),
        "head": ("fs", "cwd"),
        "tail": ("fs", "cwd"),
        "wc": ("fs", "cwd"),
        "stat": ("fs", "cwd"),
        "find": ("fs", "cwd"),
        "grep": ("fs", "cwd"),
        "tree": ("fs", "cwd"),
        "du": ("fs", "cwd"),
        "df": ("fs",),
        "readlink": ("fs", "cwd"),
        "diff": ("fs", "cwd"),
        "sort": ("fs", "cwd"),
        "uniq": ("fs", "cwd"),
        "cut": ("fs", "cwd"),
        "sed": ("fs", "cwd"),
        "awk": ("fs", "cwd"),
        "md5sum": ("fs", "cwd"),
        "sha256sum": ("fs", "cwd"),
        "file_info": ("fs", "cwd"),
        "xxd": ("fs", "cwd"),
        "join": ("fs", "cwd"),
    },
    "food_delivery": {
        "list_restaurants": ("restaurants",),
        "search_restaurants": ("restaurants",),
        "get_restaurant": ("restaurants",),
        "get_menu": ("restaurants",),
        "filter_by_dietary": ("restaurants",),
        "get_popular_items": ("restaurants",),
        "get_order": ("orders",),
        "list_orders": ("orders",),
        "get_estimated_time": ("orders",),
        "track_rider": ("orders",),
    },
    "issue_tracker": {
        "get_issue": ("issues",),
        "list_issues": ("issues",),
        "list_members": ("members",),
        "list_sprints": ("sprints",),
        "list_subtasks": ("subtasks", "issues"),
        "get_time_report": ("time_entries", "issues"),
    },
    "payments": {
        "get_invoice": ("invoices", "payments", "disputes"),
        "list_invoices": ("invoices", "payments", "disputes"),
        "list_webhooks": ("webhooks",),
    },
    "shopping": {
        "search_products": ("products",),
        "get_product": ("products",),
        "list_categories": (),
        "compare_products": ("products",),
        "get_recommendations": (),
        "get_cart": ("cart", "applied_coupon"),
        "get_coupons": (),
        "get_order": ("orders",),
        "list_orders": ("orders",),
        "get_return_status": ("returns",),
        "get_reviews": ("reviews",),
        "get_wishlist": ("wishlist",),
    },
    "team_chat": {
        "list_channels": ("channels",),
        "get_channel": ("channels",),
        "get_thread": ("threads",),
        "search_messages": ("channels",),
        "get_user_status": ("channels",),
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
            execution_semantics=(
                "readonly"
                if readonly
                else (
                    "action_execution"
                    if name in _ACTION_EXECUTION_TOOLS.get(domain, frozenset())
                    else "state_transition"
                )
            ),
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


def resolve_tool_execution_semantics(
    name: str, domain: str,
) -> ToolExecutionSemantics:
    """Resolve whether a successful no-delta call is still a required action."""
    domain_name = str(domain).lower()
    tool_name = str(name).lower()
    semantics = _public_tool_semantics(domain_name).get(tool_name)
    if semantics is None:
        raise ValueError(
            f"unknown public tool semantics: {domain_name}.{tool_name}"
        )
    return semantics.execution_semantics


def resolve_tool_state_roots(
    name: str, domain: str,
) -> tuple[str, ...] | None:
    """Return audited state roots that can invalidate one tool observation.

    Mutations already have a complete footprint contract.  Read footprints are
    rolled out only after per-domain semantic audit; ``None`` means the domain
    has not yet been certified for resource-scoped invalidation.
    """
    domain_name = str(domain).lower()
    tool_name = str(name).lower()
    semantics = _public_tool_semantics(domain_name).get(tool_name)
    if semantics is None:
        raise ValueError(
            f"unknown public tool semantics: {domain_name}.{tool_name}"
        )
    if semantics.operation != "query":
        return semantics.allowed_state_roots
    domain_roots = _READ_STATE_ROOTS.get(domain_name)
    if domain_roots is None:
        return None
    roots = domain_roots.get(tool_name)
    if roots is None:
        raise ValueError(
            f"missing audited read footprint for {domain_name}.{tool_name}"
        )
    return roots


def tool_call_invalidated_by_state_changes(
    name: str,
    arguments: Any,
    domain: str,
    later_events: list[dict[str, Any]],
) -> bool:
    """Whether later factual deltas can change this exact call's outcome."""
    changed_events = [
        event
        for event in later_events
        if (
            isinstance(event, dict)
            and bool(event.get("state_changed"))
            and (
                not str(event.get("server_name") or "")
                or str(event.get("server_name")).lower() == str(domain).lower()
            )
        )
    ]
    if not changed_events:
        return False
    roots = resolve_tool_state_roots(name, domain)
    if roots is None:
        # Unknown or unaudited domains fail closed.
        return True

    if (
        str(domain).lower() == "shopping"
        and str(name).lower() == "search_products"
        and not (
            isinstance(arguments, dict)
            and arguments.get("in_stock_only") is True
        )
    ):
        # The current handler projects only stable catalog descriptors unless
        # stock is an explicit filter. Shopping mutations change stock, not
        # catalog membership/name/category/price/description.
        roots = ()

    relevant_changes = [
        (str(path), event)
        for event in changed_events
        for path in (event.get("state_delta_paths") or [])
        if isinstance(path, str)
        and (
            path == "<root>"
            or any(path == root or path.startswith(f"{root}.") for root in roots)
        )
    ]
    if not relevant_changes:
        return False

    identifiers: set[str] = set()
    if isinstance(arguments, dict):
        for field_name, value in arguments.items():
            if not (
                str(field_name).endswith("_id")
                or str(field_name).endswith("_ids")
            ):
                continue
            values = value if isinstance(value, list) else [value]
            identifiers.update(
                str(item).strip().lower()
                for item in values
                if str(item).strip()
            )
    if not identifiers:
        return True
    for path, event in relevant_changes:
        if path == "<root>":
            return True
        parts = {part.lower() for part in path.split(".")}
        if identifiers & parts:
            return True
        if path not in roots:
            continue
        later_arguments = event.get("arguments")
        later_identifiers = {
            str(item).strip().lower()
            for field_name, value in (
                later_arguments.items()
                if isinstance(later_arguments, dict)
                else []
            )
            if (
                str(field_name).endswith("_id")
                or str(field_name).endswith("_ids")
            )
            for item in (value if isinstance(value, list) else [value])
            if str(item).strip()
        }
        if not later_identifiers or identifiers & later_identifiers:
            return True
    return False


def mutation_target_identity(
    tool_name: str, arguments: Any,
) -> tuple[tuple[str, str], ...] | None:
    """Return stable explicit resource identity for a mutating call."""
    try:
        if resolve_tool_operation(tool_name) == "query":
            return None
    except ValueError:
        return None
    if not isinstance(arguments, dict):
        return None
    for name in _MUTATION_TARGET_FIELD_PRIORITY:
        if name in arguments:
            return ((
                name,
                json.dumps(
                    arguments[name], sort_keys=True, ensure_ascii=False,
                    default=str,
                ),
            ),)
    return None


def _observation_links_target_identities(
    events: list[dict[str, Any]],
    failed_identity: tuple[tuple[str, str], ...] | None,
    successful_identity: tuple[tuple[str, str], ...] | None,
) -> bool:
    """Prove a natural selector and canonical target identify one record."""
    if (
        failed_identity is None
        or successful_identity is None
        or len(failed_identity) != 1
        or len(successful_identity) != 1
        or failed_identity[0][0] != successful_identity[0][0]
    ):
        return False
    field_name = failed_identity[0][0]
    try:
        failed_value = json.loads(failed_identity[0][1])
        successful_value = json.loads(successful_identity[0][1])
    except (TypeError, json.JSONDecodeError):
        return False
    if (
        not isinstance(failed_value, str)
        or not isinstance(successful_value, str)
        or failed_value == successful_value
    ):
        return False

    def linked(value: Any) -> bool:
        if isinstance(value, dict):
            if value.get(field_name) == successful_value and any(
                key != field_name and item == failed_value
                for key, item in value.items()
            ):
                return True
            return any(linked(item) for item in value.values())
        if isinstance(value, list):
            return any(linked(item) for item in value)
        return False

    return any(
        event.get("success") is True and linked(event.get("observation"))
        for event in events
        if isinstance(event, dict)
    )


def unresolved_failed_tool_names(
    execution_history: list[dict[str, Any]],
) -> set[str]:
    """Return failed capabilities not repaired by a matching success."""
    unresolved: set[
        tuple[str, tuple[tuple[str, str], ...] | None]
    ] = set()
    for event_index, event in enumerate(execution_history):
        if not isinstance(event, dict):
            continue
        tool_name = str(event.get("tool_name") or "")
        if not tool_name:
            continue
        target_identity = mutation_target_identity(
            tool_name, event.get("arguments", {}),
        )
        failure_key = (tool_name, target_identity)
        if event.get("success") is True:
            unresolved.discard(failure_key)
            # Readonly and identity-free calls recover at capability level.
            unresolved.discard((tool_name, None))
            for unresolved_key in tuple(unresolved):
                failed_tool, failed_identity = unresolved_key
                if (
                    failed_tool == tool_name
                    and _observation_links_target_identities(
                        execution_history[:event_index + 1],
                        failed_identity,
                        target_identity,
                    )
                ):
                    unresolved.discard(unresolved_key)
        elif event.get("success") is False:
            unresolved.add(failure_key)
    return {tool_name for tool_name, _ in unresolved}


__all__ = [
    "ToolSemantics", "ToolOperation", "ToolExecutionSemantics",
    "build_tool_semantics", "is_mutating_tool", "resolve_tool_operation",
    "resolve_tool_execution_semantics", "resolve_tool_state_roots",
    "tool_call_invalidated_by_state_changes", "mutation_target_identity",
    "unresolved_failed_tool_names",
]
