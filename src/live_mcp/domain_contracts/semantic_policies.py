"""Domain-owned deterministic label policies.

The corpus merger supplies persisted query/trace facts and remains domain
agnostic.  A policy may reject only a contradiction that those facts prove;
these are local consumability gates, not PROVE paper filters.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from src.live_mcp.registry.tool_semantics import is_mutating_tool

LabelPolicy = Callable[[str, str, list[dict[str, Any]], int], str]

_FILESYSTEM_PERSISTENCE_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bremove\b.+\bfrom\s+(?:/|the file\b|this file\b)",
        r"\breplace\b.+\b(?:in|inside)\s+(?:/|it\b|them\b|the file\b)",
        r"\breverse\s+the\s+sort\s+order\s+of\b",
        r"\bchange\s+the\s+line\s+in\b",
        r"\badd\s+a\s+line\s+to\b",
    )
)
_FOOD_SIZE_WORDS = frozenset({"small", "medium", "large", "xl", "xxl"})


def _crm_label_issue(
    query: str,
    terminal: str,
    history: list[dict[str, Any]],
    round_idx: int,
) -> str:
    if terminal not in {"ask_clarification", "report_error"}:
        return ""
    existing_task = re.search(r"\btask[_-][a-z0-9]+\b", query, re.IGNORECASE)
    update_intent = re.search(
        r"\b(?:update|change|set|make|mark|assign|priority|due)\b",
        query,
        re.IGNORECASE,
    )
    create_intent = re.search(
        r"\b(?:create|add|new)\b.{0,20}\btask\b",
        query,
        re.IGNORECASE,
    )
    created_task = any(
        event.get("tool_name") == "create_task"
        and event.get("success") is True
        and event.get("state_changed") is True
        for event in history
    )
    if existing_task and update_intent and not create_intent and created_task:
        return (
            "deterministic_label_quarantine:create_as_update:"
            f"round={round_idx}:tool=create_task"
        )
    return ""


def _filesystem_label_issue(
    query: str,
    terminal: str,
    history: list[dict[str, Any]],
    round_idx: int,
) -> str:
    if terminal != "final_answer" or not any(
        pattern.search(query) for pattern in _FILESYSTEM_PERSISTENCE_PATTERNS
    ):
        return ""
    successful = [event for event in history if event.get("success") is True]
    if not successful:
        return ""
    try:
        mutation_seen = any(
            is_mutating_tool(str(event.get("tool_name") or ""), "filesystem")
            for event in successful
        )
    except ValueError:
        return ""
    if not mutation_seen:
        return (
            "deterministic_label_quarantine:readonly_persistence:"
            f"round={round_idx}"
        )
    return ""


def _normalized_item_names(event: dict[str, Any]) -> set[str]:
    return {
        " ".join(str(item.get("name") or "").lower().split())
        for item in (event.get("arguments") or {}).get("items", [])
        if isinstance(item, dict)
    }


def _food_delivery_label_issue(
    query: str,
    terminal: str,
    history: list[dict[str, Any]],
    round_idx: int,
) -> str:
    del query
    if terminal != "final_answer":
        return ""
    failed = [
        event for event in history
        if event.get("tool_name") == "create_order"
        and event.get("success") is False
    ]
    successful = [
        event for event in history
        if event.get("tool_name") == "create_order"
        and event.get("success") is True
    ]
    for event in failed:
        failed_names = _normalized_item_names(event)
        for successful_event in successful:
            successful_tokens = [
                set(re.findall(r"[a-z0-9]+", name))
                for name in _normalized_item_names(successful_event)
            ]
            for failed_name in failed_names:
                explicit_sizes = (
                    set(re.findall(r"[a-z0-9]+", failed_name))
                    & _FOOD_SIZE_WORDS
                )
                if explicit_sizes and all(
                    not explicit_sizes.issubset(tokens)
                    for tokens in successful_tokens
                ):
                    return (
                        "deterministic_label_quarantine:food_size_downgrade:"
                        f"round={round_idx}:tool=create_order"
                    )
    return ""


DOMAIN_LABEL_POLICIES: dict[str, LabelPolicy] = {
    "crm": _crm_label_issue,
    "filesystem": _filesystem_label_issue,
    "food_delivery": _food_delivery_label_issue,
}


def evaluate_domain_label_issue(
    domain: str,
    query: str,
    terminal: str,
    history: list[dict[str, Any]],
    round_idx: int,
) -> str:
    """Evaluate the registered domain policy, if one exists."""
    policy = DOMAIN_LABEL_POLICIES.get(domain)
    return policy(query, terminal, history, round_idx) if policy else ""


__all__ = ["DOMAIN_LABEL_POLICIES", "evaluate_domain_label_issue"]
