"""Oracle validation for LLM teacher replay checks."""
from __future__ import annotations

from typing import Any


def criterion_satisfied(final_state: dict[str, Any], criterion: dict[str, Any]) -> bool:
    kind = criterion.get("type")
    server = criterion.get("server")
    state = final_state.get(server, {})
    if kind == "state_equals":
        actual = _get_path(state, criterion.get("path_parts", str(criterion["path"])))
        if actual is None and str(criterion["path"]).endswith(".messages_count"):
            messages = _get_path(
                state, str(criterion["path"]).removesuffix("_count")
            )
            actual = len(messages) if isinstance(messages, list) else None
        expected = criterion.get("value")
        op = criterion.get("op", "eq")
        if op == "gt":
            return isinstance(actual, (int, float)) and isinstance(expected, (int, float)) and actual > expected
        if op == "lt":
            return isinstance(actual, (int, float)) and isinstance(expected, (int, float)) and actual < expected
        if op == "gte":
            return isinstance(actual, (int, float)) and isinstance(expected, (int, float)) and actual >= expected
        if op == "lte":
            return isinstance(actual, (int, float)) and isinstance(expected, (int, float)) and actual <= expected
        if op == "neq":
            return actual != expected
        return actual == expected
    if kind == "state_exists":
        path = criterion.get("path_parts", criterion.get("path", ""))
        return _get_path(state, path) is not None
    if kind == "state_absent":
        path = criterion.get("path_parts", criterion.get("path", ""))
        return _get_path(state, path) is None
    return False


def _get_path(data: dict[str, Any], path: str | list[str]) -> Any:
    value: Any = data
    parts = path if isinstance(path, list) else path.split(".")
    for part in parts:
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value
