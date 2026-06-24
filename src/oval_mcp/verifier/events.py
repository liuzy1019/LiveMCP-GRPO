"""Audit event dataclasses matching OVAL-MCP §3 schema.

Each event records a single step in the trajectory: tool_call or terminal action.
safety verifier reads the event_log, not just final state.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AuditEvent:
    """Single audited event in a trajectory.

    Matches the JSON schema in OVAL-MCP §3:
    {
      "event_id": "evtlog_000001",
      "session_id": "sess_...",
      "step": 3,
      "action_type": "tool_call",
      "tool_name": "update_event",
      "terminal_action": null,
      "operation": "update",
      "target_type": "domain_resource_type",
      "target_id": "evt_102",
      "before_hash": "sha256:...",
      "after_hash": "sha256:...",
      "changed_fields": [...],
      "created_ids": [],
      "deleted_ids": [],
      "duplicate_of": null,
      "provenance": "audit_wrapper"
    }
    """

    event_id: str
    session_id: str
    step: int

    # action classification
    action_type: str  # tool_call | final_answer | ask_clarification | report_error
    tool_name: str = ""
    tool_arguments: dict[str, Any] = field(default_factory=dict)
    terminal_action: str | None = None  # null for tool_call, text for terminal

    # domain-normalized operation
    operation: str = ""  # create | update | delete | query | terminal
    target_type: str = ""  # domain resource type: "calendar_event", "shopping_cart", etc.
    target_id: str = ""

    # state change evidence (from get_state if available)
    before_hash: str = ""
    after_hash: str = ""
    changed_fields: list[str] = field(default_factory=list)
    created_ids: list[str] = field(default_factory=list)
    deleted_ids: list[str] = field(default_factory=list)

    # safety-relevant
    duplicate_of: str | None = None  # non-null if this is a duplicate side effect
    identity_violation: str = ""  # non-empty if identity_policy violated
    forbidden_transition: str = ""  # non-empty if forbidden state transition detected

    # execution metadata
    observation: dict[str, Any] | str | None = None
    execution_success: bool = False
    error_type: str | None = None
    error_message: str = ""
    schema_valid: bool = False
    state_changed: bool = False
    latency_ms: int = 0

    provenance: str = "audit_wrapper"

    def to_dict(self) -> dict[str, Any]:
        result = {
            "event_id": self.event_id,
            "session_id": self.session_id,
            "step": self.step,
            "action_type": self.action_type,
            "tool_name": self.tool_name,
            "terminal_action": self.terminal_action,
            "operation": self.operation,
            "target_type": self.target_type,
            "target_id": self.target_id,
            "before_hash": self.before_hash,
            "after_hash": self.after_hash,
            "changed_fields": self.changed_fields,
            "created_ids": self.created_ids,
            "deleted_ids": self.deleted_ids,
            "duplicate_of": self.duplicate_of,
            "provenance": self.provenance,
        }
        return result


@dataclass
class EventLog:
    """Ordered list of AuditEvents for one trajectory."""

    events: list[AuditEvent] = field(default_factory=list)
    session_id: str = ""
    task_id: str = ""

    def append(self, event: AuditEvent) -> None:
        self.events.append(event)

    def __len__(self) -> int:
        return len(self.events)

    def __iter__(self):
        return iter(self.events)

    @property
    def tool_call_events(self) -> list[AuditEvent]:
        return [e for e in self.events if e.action_type == "tool_call"]

    @property
    def terminal_events(self) -> list[AuditEvent]:
        return [e for e in self.events if e.action_type != "tool_call"]

    @property
    def successful_calls(self) -> list[AuditEvent]:
        return [e for e in self.tool_call_events if e.execution_success]

    @property
    def failed_calls(self) -> list[AuditEvent]:
        return [e for e in self.tool_call_events if not e.execution_success]

    def has_any_forbidden_event(self) -> bool:
        """Check if any event contains a safety violation marker."""
        for e in self.events:
            if e.forbidden_transition:
                return True
            if e.duplicate_of:
                return True
            if e.identity_violation:
                return True
        return False

    def forbidden_event_types(self) -> list[str]:
        """Collect distinct forbidden event type names."""
        types: list[str] = []
        for e in self.events:
            if e.forbidden_transition:
                types.append(e.forbidden_transition)
            if e.duplicate_of:
                types.append("duplicate_or_inconsistent_side_effect")
            if e.identity_violation:
                types.append("identity_or_provenance_violation")
        return list(set(types))


@dataclass
class TrajectoryEventLog:
    """Complete trajectory event log with pre/post state snapshots."""

    event_log: EventLog = field(default_factory=EventLog)
    pre_state: dict[str, Any] | None = None
    post_state: dict[str, Any] | None = None

    @classmethod
    def empty(cls, session_id: str, task_id: str) -> "TrajectoryEventLog":
        log = EventLog(session_id=session_id, task_id=task_id)
        return cls(event_log=log)


def compute_state_hash(state: dict[str, Any] | None) -> str:
    """Compute deterministic hash of a state dict."""
    if state is None:
        return ""
    canonical = json.dumps(state, sort_keys=True, ensure_ascii=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def diff_state_keys(
    before: dict[str, Any] | None,
    after: dict[str, Any] | None,
) -> list[str]:
    """Return list of top-level keys that changed between two states."""
    if before is None or after is None:
        return []
    all_keys = set(before.keys()) | set(after.keys())
    changed: list[str] = []
    for key in sorted(all_keys):
        b_val = before.get(key)
        a_val = after.get(key)
        if json.dumps(b_val, sort_keys=True, default=str) != json.dumps(a_val, sort_keys=True, default=str):
            changed.append(key)
    return changed
