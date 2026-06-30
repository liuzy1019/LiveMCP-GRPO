"""Shared dataclasses for the Live MCP MVP."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class SessionSpec:
    session_id: str
    suite_name: str
    server_names: list[str]
    seed: int
    created_at: str
    max_turns: int = 8
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]
    call_id: str
    raw_text: str = ""


@dataclass
class ToolExecutionResult:
    success: bool
    tool_name: str
    canonical_tool_name: str
    call_id: str
    session_id: str
    observation: dict[str, Any] | str | None
    error_type: str | None
    error_message: str
    schema_valid: bool
    state_changed: bool
    latency_ms: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class OracleCall:
    tool_name: str
    arguments: dict[str, Any]
    save_as: str = ""
    action: str = "tool_call"  # "tool_call" | "clarification" | "final_answer" | "report_error"


@dataclass
class OracleProgram:
    task_id: str
    calls: list[OracleCall]
    success_criteria: list[dict[str, Any]]


@dataclass
class LiveTask:
    task_id: str
    source: str
    suite_name: str
    user_prompt: str
    session_id: str
    session_seed: int
    target_servers: list[str]
    visible_tools: list[dict[str, Any]]
    required_tools: list[str]
    expected_outcome: dict[str, Any]
    success_criteria: list[dict[str, Any]]
    oracle_program: OracleProgram
    sampling_context: dict[str, Any]
    max_turns: int
    difficulty: str = "easy"
    task_type: str = ""
    hidden_tools: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    conversation_queries: list[str] = field(default_factory=list)  # PROVE CONTINUATION: all user queries across rounds
    oracle_calls_per_round: list[list] = field(default_factory=list)  # per-round oracle calls for prompt construction
    execution_history_per_round: list[list] = field(default_factory=list)  # per-round execution results


@dataclass
class TraceTurn:
    turn_idx: int
    prompt_hash: str
    model_output: str
    parsed_action_type: str
    tool_calls: list[ToolCall]
    execution_results: list[ToolExecutionResult]
    observation_text: str
    done: bool


@dataclass
class RolloutTrace:
    trace_id: str
    task_id: str
    session_id: str
    model_name: str
    started_at: str
    ended_at: str | None
    turns: list[TraceTurn]
    final_status: str
    reward: dict[str, float]
    metadata: dict[str, Any] = field(default_factory=dict)


def to_plain(value: Any) -> Any:
    """Convert nested dataclasses into JSON-serializable primitives."""
    if hasattr(value, "__dataclass_fields__"):
        return {k: to_plain(v) for k, v in asdict(value).items()}
    if isinstance(value, list):
        return [to_plain(v) for v in value]
    if isinstance(value, dict):
        return {str(k): to_plain(v) for k, v in value.items()}
    return value


def oracle_program_from_dict(data: dict[str, Any]) -> OracleProgram:
    calls = [OracleCall(**call) for call in data.get("calls", [])]
    return OracleProgram(
        task_id=data["task_id"],
        calls=calls,
        success_criteria=list(data.get("success_criteria", [])),
    )


_LIVE_TASK_REQUIRED = {
    "task_id", "source", "suite_name", "user_prompt", "session_id",
    "session_seed", "target_servers", "visible_tools", "required_tools",
    "expected_outcome", "success_criteria", "oracle_program",
    "sampling_context", "max_turns",
}


def live_task_from_dict(data: dict[str, Any]) -> LiveTask:
    payload = dict(data)
    missing = _LIVE_TASK_REQUIRED - set(payload.keys())
    if missing:
        raise KeyError(
            f"live_task_from_dict: missing required fields: {sorted(missing)}"
        )
    payload["oracle_program"] = oracle_program_from_dict(payload["oracle_program"])
    return LiveTask(**payload)
