"""Normalized Live MCP error taxonomy."""

from __future__ import annotations

from typing import Any


class CandidateGenerationError(RuntimeError):
    """One candidate was rejected by a named generation-stage contract."""

    def __init__(
        self,
        message: str,
        *,
        stage: str,
        reason: str,
        details: dict[str, Any] | None = None,
        rejection_history: list[dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(message)
        self.stage = stage
        self.reason = reason
        self.details = dict(details or {})
        self.rejection_history = list(rejection_history or [])


class ActionDecisionError(RuntimeError):
    """Action Teacher exhausted retries for one exact decision contract."""

    def __init__(
        self,
        message: str,
        *,
        reason: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.reason = reason
        self.details = dict(details or {})


class TurnLoopError(RuntimeError):
    """One round failed while preserving the partial factual trajectory."""

    def __init__(
        self,
        message: str,
        *,
        reason: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.reason = reason
        self.details = dict(details or {})

PARSE_ERROR = "parse_error"
UNKNOWN_TOOL = "unknown_tool"
SCHEMA_INVALID = "schema_invalid"
ARGUMENT_INVALID = "argument_invalid"
PRECONDITION_FAILED = "precondition_failed"
EXECUTION_ERROR = "execution_error"
TIMEOUT = "timeout"
PERMISSION_DENIED = "permission_denied"
STATE_CONFLICT = "state_conflict"
SERVER_UNAVAILABLE = "server_unavailable"

ERROR_TYPES = {
    PARSE_ERROR,
    UNKNOWN_TOOL,
    SCHEMA_INVALID,
    ARGUMENT_INVALID,
    PRECONDITION_FAILED,
    EXECUTION_ERROR,
    TIMEOUT,
    PERMISSION_DENIED,
    STATE_CONFLICT,
    SERVER_UNAVAILABLE,
}
