"""Canonical persisted-candidate quality contract for local training rows."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date
from typing import Any

from src.live_mcp.corpus.semantic_core import resolve_semantic_gate_profile
from src.live_mcp.corpus.semantic_quarantine import evaluate_semantic_quarantine
from src.live_mcp.domain_contracts.semantic_policies import (
    evaluate_domain_label_issue,
)
from src.live_mcp.registry.tool_semantics import (
    is_mutating_tool,
    unresolved_failed_tool_names,
)


_ACTION_DATE_FIELDS = frozenset({
    "date", "due_date", "execute_date", "scheduled_date", "start_time",
})
_ANY_WEEKDAY_RE = re.compile(
    r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    re.IGNORECASE,
)
_ISO_DATE_RE = re.compile(r"\b(20\d{2}-\d{2}-\d{2})(?=T|\b)")
_QUESTION_VALUE_RE = re.compile(
    r"^(?:what|which|who|where|when|how|can you|could you|would you|"
    r"please (?:provide|tell|confirm|specify))\b|\?$",
    re.IGNORECASE,
)
_TARGET_WEEKDAY_RE = re.compile(
    r"\b(?:next|this|on|for|every|by|due)\s+"
    r"(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b"
    r"|\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)"
    r"\s*,",
    re.IGNORECASE,
)
_WEEKDAY_INDEX = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}


@dataclass(frozen=True)
class PersistedQualityFinding:
    reason_code: str
    quality_issue: str
    stage: str


def _json_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if hasattr(value, "tolist"):
        return list(value.tolist())
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []
        return value if isinstance(value, list) else []
    return list(value) if isinstance(value, tuple) else []


def _unresolved_failure_issue(extra: dict[str, Any]) -> str:
    for round_pos, round_trace in enumerate(
        _json_list(extra.get("teacher_round_trace", []))
    ):
        if not isinstance(round_trace, dict):
            continue
        history = [
            event
            for event in _json_list(round_trace.get("execution_history", []))
            if isinstance(event, dict)
        ]
        unresolved = unresolved_failed_tool_names(history)
        terminal = _round_terminal(round_trace)
        if unresolved and terminal == "final_answer":
            round_idx = int(round_trace.get("round_idx", round_pos))
            return (
                f"unresolved_failed_action_final_answer:round={round_idx}:"
                f"tools={sorted(unresolved)}"
            )
    return ""


def _nested_strings(value: Any) -> list[str]:
    if isinstance(value, dict):
        return [item for child in value.values() for item in _nested_strings(child)]
    if isinstance(value, (list, tuple)) or hasattr(value, "tolist"):
        raw = value.tolist() if hasattr(value, "tolist") else value
        return [item for child in raw for item in _nested_strings(child)]
    return [str(value)] if isinstance(value, str) else []


def _action_dates(arguments: dict[str, Any]) -> list[str]:
    found: list[str] = []

    def visit(value: Any, field: str = "") -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                visit(child, str(key))
            return
        if isinstance(value, (list, tuple)) or hasattr(value, "tolist"):
            raw = value.tolist() if hasattr(value, "tolist") else value
            for child in raw:
                visit(child, field)
            return
        if field in _ACTION_DATE_FIELDS:
            found.extend(_ISO_DATE_RE.findall(str(value)))

    visit(arguments)
    return found


def _round_terminal(round_trace: dict[str, Any]) -> str:
    return next((
        str(call.get("action") or "")
        for call in reversed(_json_list(round_trace.get("oracle_calls", [])))
        if isinstance(call, dict)
        and str(call.get("action") or "tool_call") != "tool_call"
    ), "")


def _deterministic_label_issue(extra: dict[str, Any]) -> str:
    """Return a label defect directly proved by persisted query/trace facts."""
    domain = str(extra.get("domain") or "")
    for round_pos, round_trace in enumerate(
        _json_list(extra.get("teacher_round_trace", []))
    ):
        if not isinstance(round_trace, dict):
            continue
        round_idx = int(round_trace.get("round_idx", round_pos))
        query = str(
            round_trace.get("user_query") or round_trace.get("query") or ""
        )
        query_normalized = " ".join(query.lower().split())
        oracle_calls = [
            call for call in _json_list(round_trace.get("oracle_calls", []))
            if isinstance(call, dict)
        ]
        history = [
            event
            for event in _json_list(round_trace.get("execution_history", []))
            if isinstance(event, dict)
        ]
        terminal = _round_terminal(round_trace)

        mentioned_weekdays = {
            (match.group(1) or match.group(2)).lower()
            for match in _TARGET_WEEKDAY_RE.finditer(query)
        }
        all_weekdays = {
            match.group(1).lower() for match in _ANY_WEEKDAY_RE.finditer(query)
        }
        if len(mentioned_weekdays) == 1 and len(all_weekdays) == 1:
            expected_weekday = _WEEKDAY_INDEX[next(iter(mentioned_weekdays))]
            for call in oracle_calls:
                if str(call.get("action") or "tool_call") != "tool_call":
                    continue
                tool_name = str(call.get("tool_name") or "")
                try:
                    mutating = is_mutating_tool(tool_name, domain)
                except ValueError:
                    continue
                if not mutating:
                    continue
                arguments = call.get("arguments") or {}
                if not isinstance(arguments, dict):
                    continue
                for iso_date in _action_dates(arguments):
                    try:
                        actual_weekday = date.fromisoformat(iso_date).weekday()
                    except ValueError:
                        continue
                    if actual_weekday != expected_weekday:
                        return (
                            "deterministic_label_quarantine:weekday_mismatch:"
                            f"round={round_idx}:tool={tool_name}:date={iso_date}"
                        )

        if terminal == "ask_clarification":
            for event in history:
                if (
                    event.get("success") is not True
                    or event.get("state_changed") is not True
                ):
                    continue
                tool_name = str(event.get("tool_name") or "")
                try:
                    mutating = is_mutating_tool(tool_name, domain)
                except ValueError:
                    continue
                if not mutating:
                    continue
                for value in _nested_strings(event.get("arguments") or {}):
                    normalized_value = " ".join(value.lower().split())
                    if (
                        _QUESTION_VALUE_RE.search(value.strip())
                        and normalized_value not in query_normalized
                    ):
                        return (
                            "deterministic_label_quarantine:"
                            "question_placeholder_mutation:"
                            f"round={round_idx}:tool={tool_name}"
                        )

        domain_issue = evaluate_domain_label_issue(
            domain, query, terminal, history, round_idx,
        )
        if domain_issue:
            return domain_issue
    return ""


def _profile_local_trace_issue(extra: dict[str, Any]) -> str:
    """Apply local trace and label gates only under deterministic_v1."""
    try:
        profile = resolve_semantic_gate_profile(extra)
    except ValueError as exc:
        return f"invalid_semantic_gate_profile:{exc}"
    if profile != "deterministic_v1":
        return ""
    return _unresolved_failure_issue(extra) or _deterministic_label_issue(extra)


def _reason_code(quality_issue: str) -> str:
    return quality_issue.split(":round=", 1)[0]


def evaluate_persisted_candidate_quality(
    extra: dict[str, Any],
) -> PersistedQualityFinding | None:
    """Run the one local hard-quality contract shared by producer/consumer."""
    local_issue = _profile_local_trace_issue(extra)
    if local_issue:
        return PersistedQualityFinding(
            reason_code=_reason_code(local_issue),
            quality_issue=local_issue,
            stage="local_quality",
        )
    if resolve_semantic_gate_profile(extra) != "deterministic_v1":
        return None
    semantic_issue = evaluate_semantic_quarantine(extra)
    if semantic_issue is None or not semantic_issue.hard_gate:
        return None
    return PersistedQualityFinding(
        reason_code=semantic_issue.reason_code,
        quality_issue=semantic_issue.quality_issue,
        stage="semantic_quarantine",
    )


__all__ = [
    "PersistedQualityFinding",
    "evaluate_persisted_candidate_quality",
]
