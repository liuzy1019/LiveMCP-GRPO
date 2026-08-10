"""Pure artifact-to-reward task projection.

This module is the shared parsing boundary for corpus validation, rollout and
reward.  It deliberately has no dependency on training configuration or reward
implementations, so importing a read-only corpus tool cannot initialize a
training profile.
"""

from __future__ import annotations

import json
from typing import Any


class ArtifactIntegrityError(RuntimeError):
    """A serialized task artifact violates the canonical data contract."""


def _json_value(value: Any, field: str) -> Any:
    if hasattr(value, "as_py"):
        value = value.as_py()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError) as exc:
            raise ArtifactIntegrityError(f"{field} contains invalid JSON") from exc
    return value


def _required_json_list(extra_info: dict[str, Any], field: str) -> list[Any]:
    raw = extra_info.get(field)
    if not isinstance(raw, str):
        raise ArtifactIntegrityError(f"{field} must be canonical JSON text")
    parsed = _json_value(raw, field)
    if not isinstance(parsed, list):
        raise ArtifactIntegrityError(f"{field} JSON must contain a list")
    return parsed


def _list_value(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if hasattr(value, "tolist"):
        converted = value.tolist()
        return converted if isinstance(converted, list) else [converted]
    if isinstance(value, tuple):
        return list(value)
    return [value]


def parse_round_contracts(extra_info: dict[str, Any]) -> list[dict[str, Any]]:
    raw = _json_value(extra_info.get("round_contracts"), "round_contracts")
    if not isinstance(raw, list) or not raw:
        raise ArtifactIntegrityError(
            "round_contracts must contain a non-empty list"
        )
    if not all(isinstance(item, dict) for item in raw):
        raise ArtifactIntegrityError(
            "round_contracts contains a non-object entry"
        )
    return raw


def parse_dependency_edges(
    extra_info: dict[str, Any],
) -> list[tuple[int, int]]:
    raw = extra_info.get("dependency_edges")
    ground_truth = extra_info.get("ground_truth")
    if raw is None and isinstance(ground_truth, dict):
        raw = ground_truth.get("dependency_edges")
    if raw is None:
        return []
    raw = _json_value(raw, "dependency_edges")
    if not isinstance(raw, list):
        raise ArtifactIntegrityError("dependency_edges must contain a list")
    edges: list[tuple[int, int]] = []
    for edge in raw:
        if not isinstance(edge, (list, tuple)) or len(edge) != 2:
            raise ArtifactIntegrityError(f"invalid dependency edge: {edge!r}")
        try:
            source, target = int(edge[0]), int(edge[1])
        except (TypeError, ValueError) as exc:
            raise ArtifactIntegrityError(
                f"invalid dependency edge: {edge!r}"
            ) from exc
        if source < 0 or target < 0 or source == target:
            raise ArtifactIntegrityError(f"invalid dependency edge: {edge!r}")
        edges.append((source, target))
    return edges


def _allowed_terminals(extra_info: dict[str, Any], task_id: str) -> list[str]:
    raw = _json_value(
        extra_info.get("allowed_terminal_actions"),
        "allowed_terminal_actions",
    )
    if not isinstance(raw, list) or not raw:
        raise ArtifactIntegrityError(
            f"task {task_id} is missing allowed_terminal_actions"
        )
    allowed = [str(action) for action in raw]
    invalid = sorted(
        set(allowed)
        - {"final_answer", "ask_clarification", "report_error"}
    )
    if invalid:
        raise ArtifactIntegrityError(
            f"task {task_id} has invalid allowed terminal actions: {invalid}"
        )
    return allowed


def _sensitive_params(extra_info: dict[str, Any]) -> dict[str, list[str]]:
    tools = _json_value(
        extra_info.get("clean_visible_tools", "[]"),
        "clean_visible_tools",
    )
    if not isinstance(tools, list):
        raise ArtifactIntegrityError("clean_visible_tools must contain a list")
    result: dict[str, list[str]] = {}
    for tool in tools:
        if not isinstance(tool, dict):
            raise ArtifactIntegrityError(
                "clean_visible_tools contains a non-object entry"
            )
        name = str(tool.get("name") or "")
        annotations = tool.get("annotations") or {}
        sensitive = annotations.get("sensitive_params", [])
        if isinstance(sensitive, bool) or not isinstance(sensitive, list):
            raise ArtifactIntegrityError(
                f"tool {name!r} has invalid sensitive_params metadata"
            )
        if name and sensitive:
            result[name] = [str(value) for value in sensitive]
    return result


def _user_queries(extra_info: dict[str, Any]) -> list[str]:
    trace = _json_value(
        extra_info.get("teacher_round_trace", "[]"),
        "teacher_round_trace",
    )
    queries = []
    if isinstance(trace, list):
        queries = [
            str(item.get("user_query") or item.get("query") or "")
            for item in trace
            if isinstance(item, dict)
        ]
    return queries or [str(extra_info.get("user_query", ""))]


def build_reward_task(extra_info: dict[str, Any]) -> dict[str, Any]:
    """Build the canonical task dictionary consumed by rollout and reward."""
    task_id = str(extra_info.get("task_id", "unknown"))
    oracle_calls = _required_json_list(extra_info, "oracle_calls")
    success_criteria = _required_json_list(extra_info, "success_criteria")
    real_calls = [
        call
        for call in oracle_calls
        if isinstance(call, dict)
        and call.get("action", "tool_call") == "tool_call"
    ]
    terminal_actions = [
        call.get("action")
        for call in oracle_calls
        if isinstance(call, dict)
        and call.get("action")
        in {"final_answer", "ask_clarification", "report_error"}
    ]
    terminal_action = terminal_actions[-1] if terminal_actions else ""

    round_contracts = parse_round_contracts(extra_info)
    contract_names: list[str] = []
    call_rounds: list[int] = []
    for expected_round, contract in enumerate(round_contracts):
        if contract.get("round_idx") != expected_round:
            raise ArtifactIntegrityError(
                f"task {task_id} has non-canonical round_idx at "
                f"round_contracts[{expected_round}]"
            )
        names = contract.get("required_tools", [])
        if not isinstance(names, list) or not all(
            isinstance(name, str) for name in names
        ):
            raise ArtifactIntegrityError(
                f"task {task_id} has invalid required_tools in round "
                f"{expected_round}"
            )
        contract_names.extend(names)
        call_rounds.extend([expected_round] * len(names))
    oracle_names = [str(call.get("tool_name", "")) for call in real_calls]
    if contract_names != oracle_names:
        raise ArtifactIntegrityError(
            f"task {task_id} round contracts do not align with canonical "
            "oracle calls"
        )

    scenario = str(extra_info.get("scenario_type", ""))
    is_abstention = scenario in {"irrelevant", "no_tool_or_abstention"}
    missing_terminal_without_calls = (
        scenario in {"missing_function", "clarification_required"}
        and terminal_action in {"ask_clarification", "report_error"}
        and not real_calls
    )
    if is_abstention or missing_terminal_without_calls:
        required_calls: list[dict[str, Any]] = []
    elif real_calls:
        required_calls = [
            {
                "tool_name": call["tool_name"],
                "arguments": call.get("arguments", {}),
            }
            for call in real_calls
        ]
    else:
        raise ArtifactIntegrityError(
            f"task {task_id} has no canonical oracle tool calls"
        )

    protected = _json_value(
        extra_info.get("protected_fields_by_resource", {}),
        "protected_fields_by_resource",
    )
    if not isinstance(protected, dict):
        protected = {}

    return {
        "task_id": task_id,
        "required_tool_calls": required_calls,
        "required_call_rounds": call_rounds if required_calls else [],
        "identity_policy": extra_info.get("identity_policy", "domain_defined"),
        "budget": extra_info.get("budget", 8),
        "allowed_terminal_actions": _allowed_terminals(extra_info, task_id),
        "success_criteria": success_criteria,
        "target_resource_ids": _list_value(
            extra_info.get("target_resource_ids", [])
        ),
        "protected_resources": _list_value(
            extra_info.get("protected_resources", [])
        ),
        "protected_fields": _list_value(extra_info.get("protected_fields", [])),
        "protected_fields_by_resource": protected,
        "user_query": str(extra_info.get("user_query", "")),
        "user_queries": _user_queries(extra_info),
        "sensitive_params_by_tool": _sensitive_params(extra_info),
        "scenario_type": scenario,
        "final_state": extra_info.get("final_state", {}),
        "round_contracts": round_contracts,
        "dependency_edges": parse_dependency_edges(extra_info),
    }


__all__ = [
    "ArtifactIntegrityError",
    "build_reward_task",
    "parse_dependency_edges",
    "parse_round_contracts",
]
