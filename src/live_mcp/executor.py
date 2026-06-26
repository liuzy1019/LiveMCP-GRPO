"""Live MCP tool executor."""

from __future__ import annotations

import time
from typing import Any

from src.live_mcp import errors
from src.live_mcp.manager import LiveMCPManager
from src.live_mcp.schema_registry import SchemaRegistry
from src.live_mcp.transport import TransportError
from src.live_mcp.types import ToolCall, ToolExecutionResult


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

    def execute(self, session_id: str, tool_call: ToolCall) -> ToolExecutionResult:
        started = time.monotonic()
        canonical = self.schema_registry.canonical_name(tool_call.name)
        schema = self.schema_registry.get_schema(tool_call.name)
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
        validation = self.schema_registry.validate_arguments(tool_call.name, tool_call.arguments)
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
                "schema validation failed",
                False,
                False,
            )
        server_name = self.schema_registry.server_for_tool(tool_call.name, tool_call.arguments)
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
            metadata={"server_name": server_name},
        )

    def execute_many(
        self,
        session_id: str,
        tool_calls: list[ToolCall],
        mode: str = "sequential",
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
        return [self.execute(session_id, call) for call in tool_calls]

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
        )
