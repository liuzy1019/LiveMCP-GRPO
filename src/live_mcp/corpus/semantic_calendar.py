"""Fact-backed semantic checks for calendar terminal claims."""

from __future__ import annotations

import re
from typing import Any

from src.live_mcp.corpus.semantic_core import (
    SemanticQuarantineIssue,
    _json_dict,
    _successful_history,
    _terminal,
)


_INVITATION_SENT_CLAIM_RE = re.compile(
    r"\b(?:invite|invites|invitation|invitations|notification|notifications)"
    r"\b.{0,40}\b(?:sent|delivered|notified)\b",
    re.IGNORECASE,
)

_NOTIFICATION_EVIDENCE_KEYS = frozenset({
    "attendee_notifications_sent",
    "invitations_sent",
    "notifications_sent",
})


def _contains_notification_evidence(value: Any) -> bool:
    if isinstance(value, dict):
        for key, child in value.items():
            if key in _NOTIFICATION_EVIDENCE_KEYS and child is not False and child is not None:
                return True
            if _contains_notification_evidence(child):
                return True
    elif isinstance(value, (list, tuple)):
        return any(_contains_notification_evidence(child) for child in value)
    return False


def calendar_external_side_effect_claim_issue(
    round_trace: dict[str, Any],
    *,
    round_idx: int,
    evidence_rounds: list[dict[str, Any]] | None = None,
) -> SemanticQuarantineIssue | None:
    """Reject positive invitation-delivery claims absent from MCP evidence."""
    action, terminal_text = _terminal(round_trace)
    if action != "final_answer" or not _INVITATION_SENT_CLAIM_RE.search(terminal_text):
        return None
    relevant_rounds = evidence_rounds or [round_trace]
    history = [
        event
        for evidence_round in relevant_rounds
        for event in _successful_history(evidence_round)
    ]
    if any(
        _contains_notification_evidence(_json_dict(event.get("observation")))
        for event in history
    ):
        return None
    return SemanticQuarantineIssue(
        reason_code="calendar_invitation_delivery_claim_without_evidence",
        round_idx=round_idx,
        user_evidence=str(round_trace.get("user_query") or ""),
        trace_evidence={
            "terminal_text": terminal_text,
            "successful_tool_names": [
                str(event.get("tool_name") or "") for event in history
            ],
            "required_observation_keys": sorted(_NOTIFICATION_EVIDENCE_KEYS),
        },
    )


__all__ = ["calendar_external_side_effect_claim_issue"]
