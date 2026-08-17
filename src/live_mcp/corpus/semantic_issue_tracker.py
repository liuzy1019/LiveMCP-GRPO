"""Fact-backed semantic checks for issue-tracker bounded-set follow-ups."""

from __future__ import annotations

import re
from typing import Any

from src.live_mcp.corpus.semantic_core import SemanticQuarantineIssue, _terminal


_BOUNDED_SET_QUERY_RE = re.compile(
    r"\b(?:which|what)\s+(?:one\s+|ones\s+)?of\s+(?:these|those)\b",
    re.IGNORECASE,
)
_LIST_ITEM_RE = re.compile(r"^\s*(?:[-*]|\d+[.)])\s+(.+?)\s*$")


def _listed_items(text: str) -> list[str]:
    items: list[str] = []
    for line in text.splitlines():
        match = _LIST_ITEM_RE.match(line)
        if not match:
            continue
        value = re.sub(r"[*_`]", "", match.group(1)).strip().rstrip(".:")
        if value:
            items.append(" ".join(value.casefold().split()))
    return items


def _matches_prior_item(item: str, allowed: set[str]) -> bool:
    if item in allowed:
        return True
    return any(
        item.startswith(f"{prior}{separator}")
        for prior in allowed
        for separator in (" (", " - ", " — ", ": ")
    )


def issue_tracker_bounded_set_terminal_issue(
    rounds: list[dict[str, Any]],
) -> SemanticQuarantineIssue | None:
    """Reject a follow-up list that adds items outside its prior explicit list.

    This deliberately handles only an exact, locally provable contract: the
    user says "which/what of these/those", the immediately preceding assistant
    answer contains an explicit multi-item list, and the new final answer also
    contains an explicit list.  Open-ended issue facts are outside this gate.
    """
    for position in range(1, len(rounds)):
        round_trace = rounds[position]
        user_query = str(round_trace.get("user_query") or "")
        if not _BOUNDED_SET_QUERY_RE.search(user_query):
            continue
        previous_action, previous_text = _terminal(rounds[position - 1])
        current_action, current_text = _terminal(round_trace)
        if previous_action != "final_answer" or current_action != "final_answer":
            continue
        allowed = set(_listed_items(previous_text))
        selected = _listed_items(current_text)
        if len(allowed) < 2 or not selected:
            continue
        outside = [item for item in selected if not _matches_prior_item(item, allowed)]
        if not outside:
            continue
        return SemanticQuarantineIssue(
            reason_code="issue_tracker_bounded_set_terminal_expansion",
            round_idx=int(round_trace.get("round_idx", position)),
            user_evidence=user_query,
            trace_evidence={
                "prior_list_items": sorted(allowed),
                "current_list_items": selected,
                "outside_prior_list": outside,
            },
        )
    return None


__all__ = ["issue_tracker_bounded_set_terminal_issue"]
