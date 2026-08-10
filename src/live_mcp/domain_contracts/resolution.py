"""Resolve tool inputs, outputs, and conceptual entity types."""

from __future__ import annotations

from typing import Any

from src.live_mcp.domain_contracts.entities import _CREATED_ENTITY_BY_TOOL
from src.live_mcp.domain_contracts.probes import _DISCOVERY_TOOL_PREFIXES, _ENTITY_ID_FIELD_TYPES
from src.live_mcp.domain_contracts.requirements import _DOMAIN_TOOL_RELEVANT, _DOMAIN_TOOL_REQUIREMENTS


def _crm_add_note_requirements(arguments: dict[str, Any]) -> set[str]:
    entity_type = str(arguments.get("entity_type") or "").lower()
    return {entity_type} if entity_type in {"lead", "contact", "deal"} else set()


_DYNAMIC_REQUIREMENT_RESOLVERS = {
    ("crm", "add_note"): _crm_add_note_requirements,
}

# Some tools create an output entity of the same conceptual type that they
# consume as input.  These are explicit contract facts, not name heuristics.
_CREATED_ENTITY_INPUT_OVERLAPS: dict[tuple[str, str], set[str]] = {
    ("filesystem", "cp"): {"file"},
    ("filesystem", "mv"): {"file"},
    ("filesystem", "readlink"): {"file"},
    ("food_delivery", "reorder"): {"order"},
}

def _tool_existing_entity_requirements(
    tool_name: str,
    server_name: str = "",
    arguments: dict[str, Any] | None = None,
) -> set[str]:
    tool = tool_name.lower()
    server = server_name.lower()
    arguments = arguments or {}

    resolver = _DYNAMIC_REQUIREMENT_RESOLVERS.get((server, tool))
    if resolver is not None:
        return resolver(arguments)

    if server:
        domain_requirements = _DOMAIN_TOOL_REQUIREMENTS.get(server, {})
        if tool in domain_requirements:
            requirements = set(domain_requirements[tool])
            created = set(_CREATED_ENTITY_BY_TOOL.get(tool, set()))
            created.difference_update(
                _CREATED_ENTITY_INPUT_OVERLAPS.get((server, tool), set())
            )
            requirements.difference_update(created)
            return requirements
    if tool.startswith(_DISCOVERY_TOOL_PREFIXES):
        return set()
    if tool.startswith("create_") and tool not in {
        "create_task", "create_deal", "create_thread", "create_subtask", "create_order"
    }:
        return set()
    if tool in {"send_email", "send_dm", "mkdir", "touch"}:
        return set()

    requirements: set[str] = set()
    if "event" in tool or "attendee" in tool or "reminder" in tool or "recurring" in tool:
        requirements.add("event")
    if "account" in tool or tool in {
        "get_balance", "get_history", "get_statement", "deposit", "withdraw",
        "transfer", "bill_pay", "freeze_account", "unfreeze_account", "apply_loan",
        "schedule_transfer",
    }:
        requirements.add("account")
    if "invoice" in tool or tool in {"pay_invoice", "dispute_invoice"}:
        requirements.add("invoice")
    if tool in {"refund_invoice", "cancel_payment"}:
        requirements.add("payment")
    if "webhook" in tool and not tool.startswith(("create_", "list_")):
        requirements.add("webhook")
    if "email" in tool or tool in {
        "mark_read", "mark_unread", "archive_email", "reply_email",
        "forward_email", "get_attachments",
    }:
        requirements.add("email")
    if "draft" in tool and not tool.startswith("create_"):
        requirements.add("draft")
    if "filter" in tool and not tool.startswith(("create_", "list_")):
        requirements.add("filter")
    if "lead" in tool or tool == "convert_lead":
        requirements.add("lead")
    if "contact" in tool:
        requirements.add("contact")
    if "deal" in tool:
        requirements.add("deal")
    if tool in {"update_task", "complete_task"}:
        requirements.add("task")
    if "issue" in tool or tool in {
        "assign_issue", "comment_issue", "transition_issue", "add_watcher",
        "remove_watcher", "set_milestone", "add_label", "remove_label",
    }:
        requirements.add("issue")
    if tool in {"assign_issue", "add_watcher", "remove_watcher"}:
        requirements.add("user")
    if "sprint" in tool:
        requirements.add("sprint")
    if tool == "add_to_sprint":
        requirements.update({"issue", "sprint"})
    if "subtask" in tool and not tool.startswith("create_"):
        requirements.add("subtask")
    if "product" in tool or tool in {
        "add_to_cart", "compare_products", "add_review", "get_reviews",
    }:
        requirements.add("product")
    if tool in {"checkout", "update_cart_quantity", "remove_from_cart", "clear_cart"}:
        requirements.add("cart_item")
    if "order" in tool or tool in {
        "return_order", "get_estimated_time", "add_tip",
        "cancel_order", "reorder", "rate_order", "contact_support",
    }:
        requirements.add("order")
    if "restaurant" in tool or "menu" in tool or tool in {
        "filter_by_dietary", "get_popular_items", "create_order",
    }:
        requirements.add("restaurant")
    if "channel" in tool or tool in {"send_message", "archive_channel", "create_thread"}:
        requirements.add("channel")
    if tool in {"react_message", "create_thread"}:
        requirements.add("message")
    if "message" in tool and tool not in {"send_message", "search_messages"}:
        requirements.add("message")
    if tool in {"chmod", "chown", "rm", "cp", "mv"}:
        requirements.add("file")
    if tool in {"grep", "sed", "awk", "wc", "sort", "uniq", "cut", "head", "tail",
                "cat", "stat", "md5sum", "sha256sum", "xxd", "diff", "split",
                "join", "readlink", "file_info"}:
        requirements.add("file")

    created = set(_CREATED_ENTITY_BY_TOOL.get(tool, set()))
    created.difference_update(
        _CREATED_ENTITY_INPUT_OVERLAPS.get((server, tool), set())
    )
    requirements.difference_update(created)
    return requirements


def _tool_relevant_entity_types(tool_name: str, server_name: str = "") -> set[str]:
    tool = tool_name.lower()
    server = server_name.lower()
    relevant = set(_tool_existing_entity_requirements(tool, server_name))
    relevant.update(_DOMAIN_TOOL_RELEVANT.get(server, {}).get(tool, set()))
    relevant.update(_DOMAIN_TOOL_REQUIREMENTS.get(server, {}).get(tool, set()))
    relevant.update(_CREATED_ENTITY_BY_TOOL.get(tool, set()))

    if tool.startswith("list_") or tool.startswith("search_"):
        noun = tool.removeprefix("list_").removeprefix("search_")
        if noun.endswith("ies"):
            relevant.add(noun[:-3] + "y")
        elif noun.endswith("s"):
            relevant.add(noun[:-1])
        else:
            relevant.add(noun)

    if "product" in tool or tool in {"add_to_cart", "get_wishlist", "checkout"}:
        relevant.add("product")
    if "order" in tool or tool in {
        "checkout", "cancel_order", "return_order", "add_tip", "contact_support",
    }:
        relevant.add("order")
    if any(k in tool for k in ("restaurant", "menu", "popular", "dietary")):
        relevant.add("restaurant")
    if "deal" in tool:
        relevant.add("deal")
    if "contact" in tool:
        relevant.add("contact")
    if "lead" in tool:
        relevant.add("lead")
    if tool in {"create_task", "add_note"}:
        relevant.add("deal")
    if "issue" in tool or tool in {"add_to_sprint", "remove_from_sprint"}:
        relevant.add("issue")
    if tool in {"assign_issue", "add_watcher", "remove_watcher"}:
        relevant.add("user")
    if "sprint" in tool or tool in {"add_to_sprint", "remove_from_sprint"}:
        relevant.add("sprint")
    if "channel" in tool or tool in {"send_message", "create_thread", "react_message"}:
        relevant.add("channel")
    if "message" in tool or tool in {"create_thread", "react_message"}:
        relevant.add("message")
    return relevant


def _dependency_target_entity_type(
    server_name: str,
    target_name: str,
    field_name: str,
) -> str | None:
    """Resolve an ID field to the target's actual state container.

    ``product_id`` identifies a catalog product in most shopping tools, but
    cart mutations require membership in the cart, not mere catalog existence.
    This distinction is necessary for Step-2 identity joins.
    """
    if (
        server_name == "shopping"
        and field_name == "product_id"
        and target_name in {"remove_from_cart", "update_cart_quantity"}
    ):
        return "cart_item"
    return _ENTITY_ID_FIELD_TYPES.get(field_name)


