"""Live MCP tool executor."""

from __future__ import annotations

import time
from typing import Any

from src.live_mcp import errors
from src.live_mcp.protocol.manager import LiveMCPManager
from src.live_mcp.registry.schemas import SchemaRegistry
from src.live_mcp.protocol.transport import TransportError
from src.live_mcp.types import ToolCall, ToolExecutionResult


def _format_schema_error(validation, tool_name: str) -> str:
    """Format schema validation errors into an actionable message for the LLM."""
    from src.live_mcp.registry.schemas import SchemaValidationResult

    parts = [f"schema validation failed for '{tool_name}'"]
    if validation.missing_required:
        parts.append(
            f"Missing required argument(s): {', '.join(validation.missing_required)}"
        )
    if validation.unexpected_keys:
        parts.append(
            f"Unexpected argument(s): {', '.join(validation.unexpected_keys)}"
        )
    if validation.type_errors:
        parts.append(f"Type error(s): {'; '.join(validation.type_errors)}")
    if validation.enum_errors:
        parts.append(f"Enum error(s): {'; '.join(validation.enum_errors)}")
    return ". ".join(parts) + "."


class LiveMCPExecutor:
    def __init__(
        self,
        manager: LiveMCPManager,
        schema_registry: SchemaRegistry,
        timeout_s: float = 10.0,
    ):
        self.manager = manager
        self.schema_registry = schema_registry
        self.timeout_s = timeout_s

    def execute(self, session_id: str, tool_call: ToolCall, blocked_tools: set[str] | None = None, domain: str | None = None) -> ToolExecutionResult:
        started = time.monotonic()
        if blocked_tools and tool_call.name in blocked_tools:
            return self._result(
                tool_call, tool_call.name, session_id, started,
                False,
                {"error": f"Tool '{tool_call.name}' is not available for this task"},
                errors.UNKNOWN_TOOL, "tool blocked (missing function)",
                False, False,
            )
        canonical = self.schema_registry.canonical_name(tool_call.name)
        schema = self.schema_registry.get_schema(tool_call.name, domain=domain)
        if schema is None:
            return self._result(
                tool_call,
                canonical,
                session_id,
                started,
                False,
                None,
                errors.UNKNOWN_TOOL,
                "unknown tool",
                False,
                False,
            )
        validation = self.schema_registry.validate_arguments(tool_call.name, tool_call.arguments, domain=domain)
        if not validation.valid:
            return self._result(
                tool_call,
                canonical,
                session_id,
                started,
                False,
                {
                    "missing_required": validation.missing_required,
                    "unexpected_keys": validation.unexpected_keys,
                    "type_errors": validation.type_errors,
                    "enum_errors": validation.enum_errors,
                },
                errors.SCHEMA_INVALID,
                _format_schema_error(validation, tool_call.name),
                False,
                False,
            )
        server_name = self.schema_registry.server_for_tool(tool_call.name, tool_call.arguments, domain=domain)
        if server_name is None:
            return self._result(
                tool_call,
                canonical,
                session_id,
                started,
                False,
                None,
                errors.SERVER_UNAVAILABLE,
                "tool has no server",
                True,
                False,
            )
        try:
            response = self.manager.call_tool(server_name, session_id, canonical, tool_call.arguments)
        except TransportError as exc:
            if exc.error_type == errors.TIMEOUT:
                self.manager.quarantine_session(
                    session_id,
                    f"unknown commit after {server_name}::{canonical} timeout",
                )
            return self._result(
                tool_call,
                canonical,
                session_id,
                started,
                False,
                None,
                exc.error_type,
                str(exc),
                True,
                False,
            )
        success = bool(response.get("success"))
        error_type = response.get("error_type")
        return self._result(
            tool_call,
            canonical,
            session_id,
            started,
            success,
            response.get("observation"),
            None if success else str(error_type or errors.EXECUTION_ERROR),
            str(response.get("error_message") or ""),
            True,
            bool(response.get("state_changed")),
            metadata={
                "server_name": server_name,
                "state_delta_paths": list(response.get("state_delta_paths") or []),
            },
        )

    def execute_many(
        self,
        session_id: str,
        tool_calls: list[ToolCall],
        mode: str = "sequential",
        blocked_tools: set[str] | None = None,
        domain: str | None = None,
    ) -> list[ToolExecutionResult]:
        if mode != "sequential":
            return [
                ToolExecutionResult(
                    success=False,
                    tool_name=call.name,
                    canonical_tool_name=self.schema_registry.canonical_name(call.name),
                    call_id=call.call_id,
                    session_id=session_id,
                    observation=None,
                    error_type=errors.PARALLEL_NOT_SUPPORTED,
                    error_message=f"unsupported execute_many mode: {mode}",
                    schema_valid=False,
                    state_changed=False,
                    latency_ms=0,
                )
                for call in tool_calls
            ]
        return [self.execute(session_id, call, blocked_tools=blocked_tools, domain=domain) for call in tool_calls]

    def _result(
        self,
        tool_call: ToolCall,
        canonical: str,
        session_id: str,
        started: float,
        success: bool,
        observation: dict[str, Any] | str | None,
        error_type: str | None,
        error_message: str,
        schema_valid: bool,
        state_changed: bool,
        metadata: dict[str, Any] | None = None,
    ) -> ToolExecutionResult:
        # Classify execution into SUCCESS / PARTIAL_SUCCESS / FAILURE.
        # PARTIAL_SUCCESS: tool returned success=True but observation signals partial
        # results (empty list, empty dict, or explicit "partial"/"warning" keys).
        if not success:
            execution_status = "FAILURE"
        elif _is_partial_observation(observation, tool_name=tool_call.name):
            execution_status = "PARTIAL_SUCCESS"
        else:
            execution_status = "SUCCESS"
        return ToolExecutionResult(
            success=success,
            tool_name=tool_call.name,
            canonical_tool_name=canonical,
            call_id=tool_call.call_id,
            session_id=session_id,
            observation=observation,
            error_type=error_type,
            error_message=error_message,
            schema_valid=schema_valid,
            state_changed=state_changed,
            latency_ms=int((time.monotonic() - started) * 1000),
            metadata=metadata or {},
            execution_status=execution_status,
        )


def _is_partial_observation(observation: Any, tool_name: str = "") -> bool:
    """Detect PARTIAL_SUCCESS from observation content.

    A successful tool call is PARTIAL_SUCCESS when:
    - The observation is an empty list (no results found) — only for read/list tools.
    - The observation dict contains a "partial" or "warning" key.
    - A list-valued *primary result key* contains zero items — only for read/list tools.

    Write-class tools (create_*, update_*, delete_*, pay_*, etc.) may return
    dicts with empty list fields (e.g., {"event_id": "evt_001", "attendees": []})
    as a normal successful response. These must NOT be flagged as PARTIAL_SUCCESS.

    None means the server returned no body — treated as SUCCESS for write-class
    tools that return no payload.
    """
    if observation is None:
        return False

    # Determine if this is a read/discovery tool (list_/search_/get_/find_/check_)
    lowered_tool = tool_name.lower()
    is_read_tool = bool(tool_name) and (
        lowered_tool in {"find", "grep", "tree", "du", "df", "pwd"}
        or any(
        lowered_tool.startswith(p)
        for p in ("list_", "search_", "get_", "find_", "check_", "lookup_",
                  "view_", "browse_", "ls", "cat", "stat", "head", "tail",
                  "get_cart", "get_wishlist", "get_coupons")
        )
    )

    if isinstance(observation, list):
        # Empty list result is PARTIAL_SUCCESS only for read tools.
        return is_read_tool and len(observation) == 0

    if isinstance(observation, dict):
        if not observation:
            # Completely empty dict: PARTIAL_SUCCESS only for read tools.
            return is_read_tool
        if "partial" in observation or "warning" in observation:
            return True
        # Check if the primary result list is empty — only for read tools.
        # Write-class tools may have empty list fields (attendees, labels, etc.)
        # that are normal parts of a newly created entity.
        if is_read_tool:
            for value in observation.values():
                if isinstance(value, list) and len(value) == 0:
                    return True
    return False
