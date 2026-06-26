"""Tool schema registry with server-prefixed names.

All schemas stored under '{server}::{tool}' to avoid name collisions across domains.
Schema validation and server resolution try all matching schemas when a tool name
is ambiguous (e.g. 'add_label' exists in both email and issue_tracker).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SchemaValidationResult:
    valid: bool
    missing_required: list[str] = field(default_factory=list)
    unexpected_keys: list[str] = field(default_factory=list)
    type_errors: list[str] = field(default_factory=list)
    enum_errors: list[str] = field(default_factory=list)


class SchemaRegistry:
    _PREFIX_SEP = "::"

    def __init__(self) -> None:
        self._schemas: dict[str, dict[str, Any]] = {}
        self._name_map: dict[str, str] = {}

    def register_tools(
        self,
        server_name: str,
        tools: list[dict[str, Any]],
        name_map: dict[str, str] | None = None,
    ) -> None:
        self._name_map.update(name_map or {})
        for schema in tools:
            name = schema.get("name")
            if not isinstance(name, str) or not name:
                continue
            key = f"{server_name}{self._PREFIX_SEP}{name}"
            self._schemas[key] = schema

    def _matching_keys(self, tool_name: str) -> list[str]:
        """Return all schema keys matching tool_name."""
        if self._PREFIX_SEP in tool_name and tool_name in self._schemas:
            return [tool_name]
        canonical = self._name_map.get(tool_name, tool_name)
        if self._PREFIX_SEP in canonical and canonical in self._schemas:
            return [canonical]
        suffix = f"{self._PREFIX_SEP}{tool_name}"
        return [k for k in self._schemas if k.endswith(suffix)]

    def _server_from_key(self, key: str) -> str:
        return key.split(self._PREFIX_SEP, 1)[0]

    def get_schema(self, tool_name: str) -> dict[str, Any] | None:
        keys = self._matching_keys(tool_name)
        return self._schemas.get(keys[0]) if keys else None

    def server_for_tool(self, tool_name: str, arguments: dict[str, Any] | None = None) -> str | None:
        """Return server name for a tool. Disambiguates by argument validation if needed."""
        keys = self._matching_keys(tool_name)
        if not keys:
            return None
        if len(keys) == 1:
            return self._server_from_key(keys[0])
        # Multiple matches — try to disambiguate by validating arguments
        if arguments:
            for key in keys:
                schema = self._schemas[key]
                if _validate_args(schema, arguments):
                    return self._server_from_key(key)
        return self._server_from_key(keys[0])

    def canonical_name(self, visible_name: str) -> str:
        return self._name_map.get(visible_name, visible_name)

    def all_tools(self) -> list[dict[str, Any]]:
        return list(self._schemas.values())

    def server_tools(self, server_name: str) -> list[dict[str, Any]]:
        prefix = f"{server_name}{self._PREFIX_SEP}"
        return [s for k, s in self._schemas.items() if k.startswith(prefix)]

    def validate_arguments(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> SchemaValidationResult:
        keys = self._matching_keys(tool_name)
        if not keys:
            return SchemaValidationResult(valid=False, type_errors=["arguments must be object"]) if not isinstance(arguments, dict) else SchemaValidationResult(valid=False)
        # Try all matching schemas; return the first valid result
        best: SchemaValidationResult | None = None
        for key in keys:
            schema = self._schemas[key]
            # Fast check: if required args are missing, skip
            required = (schema.get("input_schema") or schema.get("parameters") or {}).get("required", [])
            if required and not all(k in arguments for k in required):
                if best is None:
                    best = SchemaValidationResult(valid=False, missing_required=[k for k in required if k not in arguments])
                continue
            result = self._validate_one(schema, arguments)
            if result.valid:
                return result
            if best is None:
                best = result
        return best or SchemaValidationResult(valid=False)

    def _validate_one(self, schema: dict[str, Any], arguments: dict[str, Any]) -> SchemaValidationResult:
        if not isinstance(arguments, dict):
            return SchemaValidationResult(valid=False, type_errors=["arguments must be object"])
        input_schema = schema.get("input_schema") or schema.get("parameters") or {}
        properties = input_schema.get("properties", {})
        required = input_schema.get("required", [])
        missing = [key for key in required if key not in arguments]
        unexpected = [key for key in arguments if key not in properties]
        type_errors: list[str] = []
        enum_errors: list[str] = []
        for key, value in arguments.items():
            prop = properties.get(key)
            if not isinstance(prop, dict):
                continue
            expected_type = prop.get("type")
            if expected_type and not _type_matches(value, expected_type):
                type_errors.append(key)
            enum_values = prop.get("enum")
            if enum_values is not None and value not in enum_values:
                enum_errors.append(key)
        return SchemaValidationResult(
            valid=not (missing or unexpected or type_errors or enum_errors),
            missing_required=missing,
            unexpected_keys=unexpected,
            type_errors=type_errors,
            enum_errors=enum_errors,
        )


def _validate_args(schema: dict[str, Any], arguments: dict[str, Any]) -> bool:
    """Quick check: do arguments satisfy the required fields of this schema?"""
    input_schema = schema.get("input_schema") or schema.get("parameters") or {}
    required = input_schema.get("required", [])
    return all(k in arguments for k in required)


def _type_matches(value: Any, expected_type: str | list[str]) -> bool:
    expected = expected_type if isinstance(expected_type, list) else [expected_type]
    for item in expected:
        if item == "string" and isinstance(value, str):
            return True
        if item == "integer" and isinstance(value, int) and not isinstance(value, bool):
            return True
        if item == "number" and isinstance(value, (int, float)) and not isinstance(value, bool):
            return True
        if item == "boolean" and isinstance(value, bool):
            return True
        if item == "object" and isinstance(value, dict):
            return True
        if item == "array" and isinstance(value, list):
            return True
        if item == "null" and value is None:
            return True
    return False
