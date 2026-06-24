"""AuditWrapper: wraps LiveMCPExecutor to produce structured event logs.

OVAL-MCP §3: audit_wrapper records model action, MCP observation/error,
state diff, and normalizes through DomainAdapter before appending to trajectory event log.

Why audit_wrapper is necessary:
  delete(target) -> create(similar_target)
may make final state look like an update, but intermediate unsafe side effects
are only detectable from the event log, not from final state alone.
"""

from __future__ import annotations

from typing import Any

from src.live_mcp.executor import LiveMCPExecutor
from src.live_mcp.manager import LiveMCPManager
from src.live_mcp.types import ToolCall, ToolExecutionResult
from src.oval_mcp.envs.domain_adapter import DomainAdapter, get_adapter
from src.oval_mcp.verifier.events import (
    AuditEvent,
    EventLog,
    TrajectoryEventLog,
    compute_state_hash,
    diff_state_keys,
)


class AuditWrapper:
    """Wraps LiveMCP executor to produce audited event logs.

    Usage:
        wrapper = AuditWrapper(executor, manager, domain_adapter)
        traj_log = wrapper.start(session_id, task_id)
        for turn in rollout:
            event = wrapper.audit_step(session_id, action_type, tool_call, ...)
        wrapper.finish(traj_log)
    """

    def __init__(
        self,
        executor: LiveMCPExecutor,
        manager: LiveMCPManager,
        adapter: DomainAdapter | None = None,
        domain_name: str = "calendar",
    ):
        self.executor = executor
        self.manager = manager
        self.adapter = adapter or get_adapter(domain_name)
        self._event_counter = 0

    def start(self, session_id: str, task_id: str) -> TrajectoryEventLog:
        """Begin a new trajectory event log with pre-state snapshot."""
        self._event_counter = 0
        pre_state = self._get_state_safe(session_id)
        return TrajectoryEventLog(
            event_log=EventLog(session_id=session_id, task_id=task_id),
            pre_state=pre_state,
            post_state=None,
        )

    def audit_step(
        self,
        session_id: str,
        action_type: str,
        tool_calls: list[ToolCall],
        execution_results: list[ToolExecutionResult],
        model_output: str = "",
    ) -> AuditEvent:
        """Record one step: capture pre/post state, normalize, produce AuditEvent.

        For tool_call: executes via LiveMCPExecutor, captures state diff.
        For terminal actions: records without state transition.
        """
        self._event_counter += 1
        event_id = f"evtlog_{session_id}_{self._event_counter:06d}"

        if action_type != "tool_call" or not tool_calls:
            return self._make_terminal_event(
                event_id=event_id,
                session_id=session_id,
                action_type=action_type,
                model_output=model_output,
            )

        # Tool call: capture pre/post state for first call
        call = tool_calls[0]
        result = execution_results[0] if execution_results else None

        pre_state = self._get_state_safe(session_id)
        post_state = self._get_state_safe(session_id)

        normalized = self.adapter.normalize_event(
            action_type="tool_call",
            tool_name=call.name,
            tool_arguments=call.arguments,
            observation=result.observation if result else None,
            execution_success=result.success if result else False,
            state_changed=result.state_changed if result else False,
            before_state=pre_state,
            after_state=post_state,
        )

        before_hash = compute_state_hash(pre_state)
        after_hash = compute_state_hash(post_state)
        changed_fields = (
            diff_state_keys(pre_state, post_state)
            if normalized.get("changed_fields") is None
            else normalized["changed_fields"]
        )

        return AuditEvent(
            event_id=event_id,
            session_id=session_id,
            step=self._event_counter,
            action_type="tool_call",
            tool_name=call.name,
            tool_arguments=call.arguments,
            terminal_action=None,
            operation=normalized.get("operation", ""),
            target_type=normalized.get("target_type", ""),
            target_id=normalized.get("target_id", ""),
            before_hash=before_hash,
            after_hash=after_hash,
            changed_fields=changed_fields,
            created_ids=normalized.get("created_ids", []),
            deleted_ids=normalized.get("deleted_ids", []),
            duplicate_of=normalized.get("duplicate_of"),
            identity_violation=normalized.get("identity_violation", ""),
            forbidden_transition=normalized.get("forbidden_transition", ""),
            observation=result.observation if result else None,
            execution_success=result.success if result else False,
            error_type=result.error_type if result else None,
            error_message=result.error_message if result else "",
            schema_valid=result.schema_valid if result else False,
            state_changed=result.state_changed if result else False,
            latency_ms=result.latency_ms if result else 0,
        )

    def finish(self, traj_log: TrajectoryEventLog) -> None:
        """Capture post-trajectory state snapshot."""
        traj_log.post_state = self._get_state_safe(traj_log.event_log.session_id)

    def _make_terminal_event(
        self,
        event_id: str,
        session_id: str,
        action_type: str,
        model_output: str,
    ) -> AuditEvent:
        """Create an audit event for a terminal action."""
        normalized = self.adapter.normalize_event(
            action_type=action_type,
            tool_name="",
            tool_arguments={},
            observation=model_output,
            execution_success=True,
            state_changed=False,
            before_state=None,
            after_state=None,
        )

        return AuditEvent(
            event_id=event_id,
            session_id=session_id,
            step=self._event_counter,
            action_type=action_type,
            tool_name="",
            tool_arguments={},
            terminal_action=model_output,
            operation=normalized.get("operation", "terminal"),
            target_type="",
            target_id="",
        )

    def _get_state_safe(self, session_id: str) -> dict[str, Any] | None:
        """Get server state, returning None if unavailable."""
        try:
            return self.manager.get_state(session_id)
        except Exception:
            return None


__all__ = ["AuditWrapper"]
