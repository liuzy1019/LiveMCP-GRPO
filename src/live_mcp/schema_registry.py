"""Tool schema registry and bounded top-level argument validation."""

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
    def __init__(self) -> None:
        self._schemas: dict[str, dict[str, Any]] = {}
        self._server_by_tool: dict[str, str] = {}
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
            self._schemas[name] = schema
            self._server_by_tool[name] = server_name

    def get_schema(self, tool_name: str) -> dict[str, Any] | None:
        return self._schemas.get(self.canonical_name(tool_name))

    def server_for_tool(self, tool_name: str) -> str | None:
        return self._server_by_tool.get(self.canonical_name(tool_name))

    def canonical_name(self, visible_name: str) -> str:
        return self._name_map.get(visible_name, visible_name)

    def all_tools(self) -> list[dict[str, Any]]:
        return list(self._schemas.values())

    def server_tools(self, server_name: str) -> list[dict[str, Any]]:
        """返回指定 server 的所有注册工具 schema。"""
        return [
            schema for name, schema in self._schemas.items()
            if self._server_by_tool.get(name) == server_name
        ]

    def validate_arguments(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> SchemaValidationResult:
        schema = self.get_schema(tool_name)
        if schema is None or not isinstance(arguments, dict):
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
