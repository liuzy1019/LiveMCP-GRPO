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

    def _matching_keys(self, tool_name: str, domain: str | None = None) -> list[str]:
        """Return all schema keys matching tool_name."""
        def filter_domain(keys: list[str]) -> list[str]:
            if not domain:
                return keys
            return [key for key in keys if self._server_from_key(key) == domain]

        if self._PREFIX_SEP in tool_name and tool_name in self._schemas:
            return filter_domain([tool_name])
        canonical = self._name_map.get(tool_name, tool_name)
        if self._PREFIX_SEP in canonical and canonical in self._schemas:
            return filter_domain([canonical])
        suffix = f"{self._PREFIX_SEP}{tool_name}"
        keys = [k for k in self._schemas if k.endswith(suffix)]
        if domain:
            return filter_domain(keys)
        return keys

    def _server_from_key(self, key: str) -> str:
        return key.split(self._PREFIX_SEP, 1)[0]

    def get_schema(self, tool_name: str, domain: str | None = None) -> dict[str, Any] | None:
        keys = self._matching_keys(tool_name, domain=domain)
        return self._schemas.get(keys[0]) if keys else None

    def server_for_tool(self, tool_name: str, arguments: dict[str, Any] | None = None, domain: str | None = None) -> str | None:
        """Return server name for a tool. Disambiguates by argument validation or domain hint if needed."""
        keys = self._matching_keys(tool_name, domain=domain)
        if not keys:
            return None
        if len(keys) == 1:
            return self._server_from_key(keys[0])
        # Domain hint: if caller knows the domain, use it to disambiguate
        if domain:
            for key in keys:
                if self._server_from_key(key) == domain:
                    return domain
        # Multiple matches — disambiguate only when exactly one schema accepts
        # the arguments. Silently picking registration order can execute a
        # same-name tool in the wrong domain (for example get_thread).
        if arguments:
            valid_keys = [
                key for key in keys
                if _validate_args(self._schemas[key], arguments)
            ]
            if len(valid_keys) == 1:
                return self._server_from_key(valid_keys[0])
        return None

    def canonical_name(self, visible_name: str) -> str:
        canonical = self._name_map.get(visible_name, visible_name)
        if self._PREFIX_SEP in canonical:
            return canonical.split(self._PREFIX_SEP, 1)[1]
        return canonical

    def all_tools(self) -> list[dict[str, Any]]:
        return list(self._schemas.values())

    def all_tools_with_servers(self) -> list[dict[str, Any]]:
        """Return schema copies annotated with their executable owner server."""
        return [
            {**schema, "_server_name": self._server_from_key(key)}
            for key, schema in self._schemas.items()
        ]

    def server_tools(self, server_name: str) -> list[dict[str, Any]]:
        prefix = f"{server_name}{self._PREFIX_SEP}"
        return [s for k, s in self._schemas.items() if k.startswith(prefix)]

    def validate_arguments(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        domain: str | None = None,
    ) -> SchemaValidationResult:
        keys = self._matching_keys(tool_name, domain=domain)
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
        missing: list[str] = []
        unexpected: list[str] = []
        type_errors: list[str] = []
        enum_errors: list[str] = []
        _validate_schema_value(
            input_schema,
            arguments,
            path="",
            missing=missing,
            unexpected=unexpected,
            type_errors=type_errors,
            enum_errors=enum_errors,
        )
        return SchemaValidationResult(
            valid=not (missing or unexpected or type_errors or enum_errors),
            missing_required=missing,
            unexpected_keys=unexpected,
            type_errors=type_errors,
            enum_errors=enum_errors,
        )


def _validate_schema_value(
    schema: dict[str, Any],
    value: Any,
    *,
    path: str,
    missing: list[str],
    unexpected: list[str],
    type_errors: list[str],
    enum_errors: list[str],
) -> None:
    """Validate the JSON-schema subset exposed by the local MCP tools.

    The previous validator stopped at top-level objects, allowing unsupported
    mutation fields to reach handlers and be silently ignored.  This recursive
    subset intentionally covers only constraints used by this repository; it
    is not presented as a general JSON Schema implementation.
    """
    expected_type = schema.get("type")
    label = path or "arguments"
    if expected_type and not _type_matches(value, expected_type):
        type_errors.append(f"{label}: expected {expected_type}")
        return

    enum_values = schema.get("enum")
    if enum_values is not None and value not in enum_values:
        enum_errors.append(f"{label}: {value!r} not in {enum_values!r}")

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            type_errors.append(f"{label}: must be >= {schema['minimum']}")
        if "exclusiveMinimum" in schema and value <= schema["exclusiveMinimum"]:
            type_errors.append(f"{label}: must be > {schema['exclusiveMinimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            type_errors.append(f"{label}: must be <= {schema['maximum']}")

    if isinstance(value, dict):
        properties = schema.get("properties") or {}
        required = schema.get("required") or []
        if "minProperties" in schema and len(value) < schema["minProperties"]:
            type_errors.append(
                f"{label}: requires at least {schema['minProperties']} field(s)"
            )
        for key in required:
            if key not in value:
                missing.append(f"{path}.{key}" if path else str(key))
        # Tool argument objects have always rejected unknown top-level keys.
        # Nested free-form objects remain open unless their schema explicitly
        # closes them with additionalProperties=false.
        allow_extra = schema.get("additionalProperties", path != "")
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            child_schema = properties.get(key)
            if not isinstance(child_schema, dict):
                if allow_extra is False:
                    unexpected.append(child_path)
                continue
            _validate_schema_value(
                child_schema,
                child,
                path=child_path,
                missing=missing,
                unexpected=unexpected,
                type_errors=type_errors,
                enum_errors=enum_errors,
            )

    if isinstance(value, list):
        if "minItems" in schema and len(value) < schema["minItems"]:
            type_errors.append(f"{label}: requires at least {schema['minItems']} item(s)")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            type_errors.append(f"{label}: allows at most {schema['maxItems']} item(s)")
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for index, child in enumerate(value):
                _validate_schema_value(
                    item_schema,
                    child,
                    path=f"{path}[{index}]" if path else f"[{index}]",
                    missing=missing,
                    unexpected=unexpected,
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
