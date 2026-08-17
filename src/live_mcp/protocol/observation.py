"""Shared loss-aware projection for Teacher and Policy tool observations."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from src.live_mcp.types import ToolExecutionResult


OBSERVATION_SCHEMA_VERSION = "live-mcp-observation-v1"
OBSERVATION_PROJECTION_VERSION = "loss-aware-v2"
TRAJECTORY_SCHEMA_VERSION = "live-mcp-fact-contract-trajectory-v2"
DEFAULT_OBSERVATION_CHARS = 4096
# Teacher and Policy share one projection contract.  Call sites may still pass
# an explicit larger budget, but component-specific implicit defaults are not
# allowed to silently expose different facts.
DEFAULT_TEACHER_OBSERVATION_CHARS = DEFAULT_OBSERVATION_CHARS
DEFAULT_POLICY_OBSERVATION_CHARS = DEFAULT_OBSERVATION_CHARS

_FACT_KEYS = (
    "success", "execution_status", "error_type", "error_message",
    "state_changed", "schema_valid", "count", "total", "total_count",
    "next_cursor", "has_more", "partial", "warning", "error", "message",
    "status", "state", "amount", "balance", "remaining",
    "remaining_refundable", "name", "title", "subject", "type",
)

# The canonical MCP envelope plus one nested entity collection reaches an
# entity record at container depth five. Preserve that record's scalar facts;
# only containers nested inside the record cross this boundary.
_MAX_CONTAINER_DEPTH = 6


def compute_server_schema_hash(tools: list[dict[str, Any]]) -> str:
    """Hash the public executable schema independently of graph semantics."""
    payload = [
        {
            "name": tool.get("name"),
            "description": tool.get("description", ""),
            "input_schema": tool.get("input_schema", {}),
            "annotations": tool.get("annotations", {}),
        }
        for tool in sorted(tools, key=lambda item: str(item.get("name", "")))
    ]
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def project_observation(observation: Any, max_chars: int) -> str:
    """Render JSON with stable semantic priority instead of prefix truncation."""
    budget = max(256, int(max_chars))
    original = json.dumps(observation, ensure_ascii=False, default=str)

    def compact(value: Any, depth: int = 0) -> Any:
        # Depth limits protect recursive containers, not scalar facts. Checking
        # scalars first prevents a complete entity record from turning all of
        # its IDs and fields into ``_truncated_depth`` markers.
        if isinstance(value, str):
            if len(value) > 500:
                omitted = len(value) - 480
                return (
                    value[:240]
                    + f"... [truncated {omitted} chars] ..."
                    + value[-240:]
                )
            return value
        if not isinstance(value, (dict, list)):
            return value
        if depth >= _MAX_CONTAINER_DEPTH:
            return {"_truncated_depth": True}
        if isinstance(value, dict):
            keys = list(value)
            id_keys = [key for key in keys if key == "id" or key.endswith("_id")]
            priority = list(dict.fromkeys([*id_keys, *_FACT_KEYS]))
            chosen = [key for key in priority if key in value]
            if len(keys) <= 12:
                chosen.extend(key for key in keys if key not in chosen)
            else:
                chosen.extend(
                    [key for key in keys if key not in chosen][:4]
                )
            result = {key: compact(value[key], depth + 1) for key in chosen}
            if len(chosen) < len(keys):
                result["_omitted_fields"] = len(keys) - len(chosen)
            return result
        if isinstance(value, list):
            if len(value) <= 24:
                return [compact(item, depth + 1) for item in value]
            return [
                *[compact(item, depth + 1) for item in value[:12]],
                {"_omitted_items": len(value) - 24},
                *[compact(item, depth + 1) for item in value[-12:]],
            ]
        raise AssertionError("unreachable observation container type")

    projected = compact(observation)
    rendered = json.dumps(projected, ensure_ascii=False, default=str)
    if len(rendered) <= budget:
        if len(rendered) < len(original):
            wrapped = json.dumps(
                {
                    "data": projected,
                    "_compacted": True,
                    "_original_chars": len(original),
                },
                ensure_ascii=False,
                default=str,
            )
            if len(wrapped) <= budget:
                return wrapped
        return rendered

    facts: list[dict[str, Any]] = []

    def collect(value: Any) -> None:
        if isinstance(value, dict):
            fact = {
                key: child
                for key, child in value.items()
                if (
                    key == "id"
                    or key.endswith("_id")
                    or key in _FACT_KEYS
                )
                and not isinstance(child, (dict, list))
            }
            if fact:
                facts.append(fact)
            for child in value.values():
                collect(child)
        elif isinstance(value, list):
            for child in value:
                collect(child)

    collect(observation)
    total_facts = len(facts)
    if len(facts) > 48:
        facts = [*facts[:24], *facts[-24:]]

    def minimal_fact(fact: dict[str, Any]) -> dict[str, Any]:
        id_items = {
            key: value
            for key, value in fact.items()
            if key == "id" or key.endswith("_id")
        }
        detail_keys = (
            "success", "execution_status", "error_type", "error_message",
            "status", "state", "name", "title", "subject", "type",
            "count", "total", "has_more", "next_cursor",
        )
        for key in detail_keys:
            if key in fact and key not in id_items:
                id_items[key] = fact[key]
            if len(id_items) >= max(2, len([
                item for item in id_items if item == "id" or item.endswith("_id")
            ]) + 2):
                break
        return id_items

    base = {
        "_compacted": True,
        "_original_chars": len(original),
        "_sha256": hashlib.sha256(original.encode("utf-8")).hexdigest()[:16],
    }
    def render_facts(retained: list[dict[str, Any]]) -> str:
        envelope = {
            "summary_facts": retained,
            **base,
            "_omitted_facts": total_facts - len(retained),
        }
        return json.dumps(envelope, ensure_ascii=False, default=str)

    rendered = render_facts(facts)
    if len(rendered) <= budget:
        return rendered

    retained = [minimal_fact(fact) for fact in facts]
    rendered = render_facts(retained)
    if len(rendered) <= budget:
        return rendered

    # Remove from the middle so the first and most recently returned entities
    # survive even at very small budgets.
    while len(retained) > 2:
        retained.pop(len(retained) // 2)
        rendered = render_facts(retained)
        if len(rendered) <= budget:
            return rendered

    if retained:
        rendered = render_facts(retained)
        if len(rendered) <= budget:
            return rendered
    return json.dumps(base, ensure_ascii=False, default=str)


def tool_result_envelope(result: ToolExecutionResult) -> dict[str, Any]:
    """Return one audience-neutral execution envelope for success and failure."""
    return {
        "success": bool(result.success),
        "execution_status": str(result.execution_status),
        "error_type": result.error_type,
        "error_message": str(result.error_message or ""),
        "state_changed": bool(result.state_changed),
        "schema_valid": bool(result.schema_valid),
        "observation": result.observation,
    }


def serialize_tool_result(result: ToolExecutionResult, max_chars: int) -> str:
    return project_observation(tool_result_envelope(result), max_chars=max_chars)


def serialize_execution_error(
    error_type: str,
    error_message: str,
    max_chars: int,
    observation: Any = None,
) -> str:
    envelope = {
        "success": False,
        "execution_status": "FAILURE",
        "error_type": error_type,
        "error_message": error_message,
        "state_changed": False,
        "schema_valid": False,
        "observation": observation,
    }
    return project_observation(envelope, max_chars=max_chars)


__all__ = [
    "DEFAULT_POLICY_OBSERVATION_CHARS",
    "DEFAULT_TEACHER_OBSERVATION_CHARS",
    "DEFAULT_OBSERVATION_CHARS",
    "TRAJECTORY_SCHEMA_VERSION",
    "OBSERVATION_SCHEMA_VERSION",
    "OBSERVATION_PROJECTION_VERSION",
    "compute_server_schema_hash",
    "project_observation",
    "serialize_execution_error",
    "serialize_tool_result",
    "tool_result_envelope",
]
