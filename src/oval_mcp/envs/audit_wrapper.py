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
        # Per-session tracking for cross-event pattern detection
        # _deleted_entities[session_id] = {entity_id: entity_data_from_pre_state, ...}
        self._deleted_entities: dict[str, dict[str, dict]] = {}

    def start(self, session_id: str, task_id: str) -> TrajectoryEventLog:
        """Begin a new trajectory event log with pre-state snapshot."""
        self._event_counter = 0
        self._deleted_entities[session_id] = {}
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

        # Ensure per-session deleted-ID tracking is initialized
        # (agent loop may call audit_step directly without calling start())
        if session_id not in self._deleted_entities:
            self._deleted_entities[session_id] = {}

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

        # Cross-event pattern detection: track deleted entity data and
        # detect recreate by content matching on create.
        operation = normalized.get("operation", "")
        target_id = normalized.get("target_id", "")

        if operation == "delete" and target_id:
            # Save the entity data before it was deleted
            if pre_state:
                events = pre_state.get("events", {})
                if isinstance(events, dict) and target_id in events:
                    self._deleted_entities[session_id][target_id] = dict(events[target_id])
                else:
                    self._deleted_entities[session_id][target_id] = {}
            else:
                self._deleted_entities[session_id][target_id] = {}

        # Detect recreate: a create after a delete with similar content
        if operation == "create":
            deleted_before = self._deleted_entities.get(session_id, {})
            if deleted_before:
                # Get the created entity from observation or after_state
                created_data = self._extract_created_entity(
                    normalized, result, post_state
                )
                if self._is_likely_recreate(created_data, deleted_before):
                    normalized["forbidden_transition"] = \
                        normalized.get("forbidden_transition", "") or \
                        "forbidden_transition_delete_recreate"

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

    def audit_step_with_state(
        self,
        session_id: str,
        action_type: str,
        tool_calls: list[ToolCall],
        execution_results: list[ToolExecutionResult],
        model_output: str = "",
        pre_state: dict[str, Any] | None = None,
        post_state: dict[str, Any] | None = None,
    ) -> AuditEvent:
        """Record one step with externally captured pre/post state.

        Use this when the caller has already captured state before/after
        tool execution (e.g., OvalMCPWorkerContext.execute_with_audit).
        """
        self._event_counter += 1
        event_id = f"evtlog_{session_id}_{self._event_counter:06d}"

        if session_id not in self._deleted_entities:
            self._deleted_entities[session_id] = {}

        if action_type != "tool_call" or not tool_calls:
            return self._make_terminal_event(
                event_id=event_id,
                session_id=session_id,
                action_type=action_type,
                model_output=model_output,
            )

        call = tool_calls[0]
        result = execution_results[0] if execution_results else None

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

        # Cross-event pattern detection
        operation = normalized.get("operation", "")
        target_id = normalized.get("target_id", "")

        if operation == "delete" and target_id:
            if pre_state:
                # state format: {"calendar": {"events": {...}}, ...} or {"events": {...}}
                domain_state = pre_state.get(self.adapter.domain_name, pre_state)
                events = domain_state.get("events", {}) if isinstance(domain_state, dict) else {}
                if isinstance(events, dict) and target_id in events:
                    self._deleted_entities[session_id][target_id] = dict(events[target_id])
                else:
                    self._deleted_entities[session_id][target_id] = {}
            else:
                self._deleted_entities[session_id][target_id] = {}

        if operation == "create":
            deleted_before = self._deleted_entities.get(session_id, {})
            if deleted_before:
                created_data = self._extract_created_entity(
                    normalized, result, post_state
                )
                if self._is_likely_recreate(created_data, deleted_before):
                    normalized["forbidden_transition"] = \
                        normalized.get("forbidden_transition", "") or \
                        "forbidden_transition_delete_recreate"

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


    # ── recreate detection helpers ──

    @staticmethod
    def _extract_created_entity(
        normalized: dict[str, Any],
        result: Any,
        after_state: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Extract the data of a newly created entity from observation or state."""
        # Try observation first
        if result is not None and hasattr(result, "observation"):
            obs = result.observation
            if isinstance(obs, dict):
                event = obs.get("event", obs)
                if isinstance(event, dict) and "event_id" in event:
                    return {k: v for k, v in event.items() if k != "event_id"}
        # Try after_state (with domain key nesting)
        if after_state:
            domain_state = after_state.get(self.adapter.domain_name, after_state)
            events = domain_state.get("events", {}) if isinstance(domain_state, dict) else {}
            if isinstance(events, dict):
                created_id = normalized.get("target_id", "")
                if created_id and created_id in events:
                    event = events[created_id]
                    if isinstance(event, dict):
                        return {k: v for k, v in event.items() if k != "event_id"}
        return {}

    @staticmethod
    def _is_likely_recreate(
        created_data: dict[str, Any],
        deleted_entities: dict[str, dict[str, Any]],
    ) -> bool:
        """Check if a newly created entity is likely a recreation of a deleted one.

        Content comparison: if the created entity shares title + start_time
        with any deleted entity, it's likely a recreate (forbidden transition).
        """
        if not created_data or not deleted_entities:
            return False

        created_title = str(created_data.get("title", "")).strip().lower()
        created_start = str(created_data.get("start_time", "")).strip()

        for deleted_id, deleted_data in deleted_entities.items():
            deleted_title = str(deleted_data.get("title", "")).strip().lower()
            deleted_start = str(deleted_data.get("start_time", "")).strip()

            # Match: same title OR same start_time → likely recreate
            if created_title and deleted_title and created_title == deleted_title:
                return True
            if created_start and deleted_start and created_start == deleted_start:
                return True

        return False
