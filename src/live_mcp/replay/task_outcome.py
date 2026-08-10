"""Outcome metadata derived from accepted generation trajectories."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from src.live_mcp.registry.tool_semantics import build_tool_semantics
from src.live_mcp.types import OracleCall


_IDENTITY_POLICY_BY_DOMAIN: dict[str, str] = {
    "calendar": "preserve",
    "banking": "preserve",
    "payments": "preserve",
    "crm": "preserve",
    "issue_tracker": "preserve",
    "email": "append_only",
    "team_chat": "append_only",
    "shopping": "create_new",
    "food_delivery": "create_new",
    "filesystem": "domain_defined",
}


def stable_state_hash(state: dict[str, Any]) -> str:
    raw = json.dumps(state, sort_keys=True, ensure_ascii=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


def identity_policy_for_domain(domain: str) -> str:
    return _IDENTITY_POLICY_BY_DOMAIN.get(domain, "domain_defined")


def oracle_target_ids(calls: list[OracleCall]) -> list[str]:
    ids: set[str] = set()
    for call in calls:
        for key, value in (call.arguments or {}).items():
            key_lower = key.lower()
            if not (
                key_lower.endswith("_id")
                or key_lower
                in {"path", "source", "destination", "from_account", "to_account"}
            ):
                continue
            if isinstance(value, str) and value:
                ids.add(value)
    return sorted(ids)


def oracle_deleted_target_ids(
    calls: list[OracleCall], domain: str, tools: list[dict[str, Any]],
) -> list[str]:
    contracts = build_tool_semantics(domain, tools)
    ids: set[str] = set()
    for call in calls:
        contract = contracts.get(call.tool_name)
        if contract is None:
            raise ValueError(f"missing tool contract for {domain}.{call.tool_name}")
        if contract.operation == "delete":
            ids.update(oracle_target_ids([call]))
    return sorted(ids)


def protected_fields_by_resource(
    initial_state: dict[str, Any],
    target_ids: list[str],
    success_criteria: list[dict[str, Any]],
) -> dict[str, list[str]]:
    intended: dict[str, set[str]] = {
        resource_id: set() for resource_id in target_ids
    }
    for criterion in success_criteria:
        path = str(criterion.get("path", ""))
        path_parts = criterion.get("path_parts")
        parts = path_parts if isinstance(path_parts, list) else path.split(".")
        for resource_id in target_ids:
            if resource_id in parts and len(parts) > parts.index(resource_id) + 1:
                intended[resource_id].add(parts[-1])

    protected: dict[str, list[str]] = {}
    for container in initial_state.values():
        if not isinstance(container, dict):
            continue
        for resource_id in target_ids:
            entity = container.get(resource_id)
            if not isinstance(entity, dict):
                continue
            fields = {
                field
                for field in entity
                if not field.endswith("_id")
                and field not in intended[resource_id]
            }
            if fields:
                protected[resource_id] = sorted(fields)
    return protected


def attribute_success_criteria(
    success_criteria: list[dict[str, Any]],
    execution_history: list[dict[str, Any]],
    domain: str = "",
) -> list[dict[str, Any]]:
    """Link each state criterion to factual successful-call delta paths."""
    successful_events = [
        event
        for event in execution_history
        if (
            bool(event.get("success"))
            and event.get("state_delta_paths")
            and (
                not domain
                or not str(event.get("server_name") or "")
                or str(event.get("server_name")).lower() == domain.lower()
            )
        )
    ]
    attributed: list[dict[str, Any]] = []
    for index, criterion in enumerate(success_criteria):
        path_parts = criterion.get("path_parts")
        criterion_path = (
            ".".join(str(part) for part in path_parts)
            if isinstance(path_parts, list)
            else str(criterion.get("path") or "")
        )
        sources: list[dict[str, Any]] = []
        for event_index, event in enumerate(successful_events):
            matching_paths = [
                str(delta_path)
                for delta_path in event.get("state_delta_paths", [])
                if criterion_path == str(delta_path)
                or criterion_path.startswith(f"{delta_path}.")
                or str(delta_path).startswith(f"{criterion_path}.")
            ]
            if matching_paths:
                sources.append({
                    "event_index": event_index,
                    "tool_name": str(event.get("tool_name") or ""),
                    "server_name": str(event.get("server_name") or domain),
                    "arguments": dict(event.get("arguments") or {}),
                    "state_delta_paths": matching_paths,
                })
        attributed.append({
            "criterion_index": index,
            "criterion_path": criterion_path,
            "source_calls": sources,
        })
    return attributed
