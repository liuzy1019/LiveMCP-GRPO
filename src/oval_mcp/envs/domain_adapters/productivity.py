"""Calendar, email, and team-chat reward adapters."""

from __future__ import annotations

from typing import Any

from .base import DomainAdapter


class CalendarAdapter(DomainAdapter):
    """Domain adapter for the calendar MCP server.

    Calendar state:
      events: dict[event_id -> {event_id, title, start_time, end_time, attendees}]
      next_event_num: int

    target_type: "calendar_event"
    identity_policy: typically "preserve" (update, don't delete+recreate)
    """

    domain_name = "calendar"

    # Tool -> (operation, target_type)
    # All 17 calendar tools are mapped so the adapter can extract
    # changed_fields / created_ids / deleted_ids for reward predicates.
    TOOL_MAP = {
        # — core CRUD —
        "list_events": ("query", "calendar_event"),
        "search_events": ("query", "calendar_event"),
        "get_event": ("query", "calendar_event"),
        "create_event": ("create", "calendar_event"),
        "update_event": ("update", "calendar_event"),
        "delete_event": ("delete", "calendar_event"),
        # — recurring —
        "create_recurring": ("create", "calendar_event"),
        "get_recurring_info": ("query", "calendar_event"),
        # — attendees —
        "add_attendee": ("update", "calendar_event"),
        "remove_attendee": ("update", "calendar_event"),
        # — availability / scheduling —
        "get_free_busy": ("query", "calendar_event"),
        "check_conflicts": ("query", "calendar_event"),
        "get_working_hours": ("query", "calendar_event"),
        # — reminders / timezone / response —
        "set_reminder": ("update", "calendar_event"),
        "change_timezone": ("update", "calendar_event"),
        "respond_to_event": ("update", "calendar_event"),
        # — export —
        "export_calendar": ("query", "calendar_event"),
    }

    def normalize_event(
        self,
        action_type: str,
        tool_name: str,
        tool_arguments: dict[str, Any],
        observation: dict[str, Any] | str | None,
        execution_success: bool,
        state_changed: bool,
        before_state: dict[str, Any] | None,
        after_state: dict[str, Any] | None,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "operation": "",
            "target_type": "calendar_event",
            "target_id": "",
            "changed_fields": [],
            "created_ids": [],
            "deleted_ids": [],
            "identity_violation": "",
            "forbidden_transition": "",
            "duplicate_of": None,
        }

        if action_type != "tool_call":
            result["operation"] = "terminal"
            return result

        op, target = self.tool_semantics(tool_name, "calendar_event", state_changed)
        result["operation"] = op
        result["target_type"] = target

        # Extract target_id from arguments
        if tool_name == "create_event":
            if execution_success and isinstance(observation, dict):
                event = observation.get("event", observation.get("observation", {}))
                if isinstance(event, dict):
                    result["target_id"] = event.get("event_id", "")
            # Detect created IDs from state diff
            be = self._unwrap_domain_state(before_state, "calendar")
            ae = self._unwrap_domain_state(after_state, "calendar")
            if be is not None and ae is not None:
                before_events = set(be.get("events", {}).keys())
                after_events = set(ae.get("events", {}).keys())
                result["created_ids"] = list(after_events - before_events)

        elif tool_name == "update_event":
            result["target_id"] = tool_arguments.get("event_id", "")
            if execution_success and isinstance(observation, dict):
                event = observation.get("event", observation.get("observation", {}))
                if isinstance(event, dict):
                    result["target_id"] = event.get("event_id", result["target_id"])
            # Detect changed fields
            fields = tool_arguments.get("fields", {})
            if isinstance(fields, dict):
                result["changed_fields"] = list(fields.keys())

        elif tool_name == "delete_event":
            result["target_id"] = tool_arguments.get("event_id", "")
            # Detect deleted IDs from state diff
            be = self._unwrap_domain_state(before_state, "calendar")
            ae = self._unwrap_domain_state(after_state, "calendar")
            if be is not None and ae is not None:
                before_events = set(be.get("events", {}).keys())
                after_events = set(ae.get("events", {}).keys())
                result["deleted_ids"] = list(before_events - after_events)

        elif tool_name == "list_events":
            result["target_id"] = ""

        elif tool_name in ("add_attendee", "remove_attendee", "respond_to_event",
                           "set_reminder"):
            result["target_id"] = tool_arguments.get("event_id", "")
            if execution_success:
                if tool_name in ("add_attendee", "remove_attendee"):
                    result["changed_fields"] = ["attendees"]
                elif tool_name == "respond_to_event":
                    result["changed_fields"] = ["response"]
                else:
                    result["changed_fields"] = ["reminders"]

        elif tool_name == "create_recurring":
            if execution_success and isinstance(observation, dict):
                event = observation.get("event", observation.get("observation", {}))
                if isinstance(event, dict):
                    result["target_id"] = event.get("event_id", "")
            be = self._unwrap_domain_state(before_state, "calendar")
            ae = self._unwrap_domain_state(after_state, "calendar")
            if be is not None and ae is not None:
                before_events = set(be.get("events", {}).keys())
                after_events = set(ae.get("events", {}).keys())
                result["created_ids"] = list(after_events - before_events)

        elif tool_name == "change_timezone":
            # timezone is a global/server-level setting, not per-event
            result["target_id"] = ""
            if execution_success:
                result["changed_fields"] = ["timezone"]

        elif tool_name in ("search_events", "get_event", "get_free_busy",
                           "check_conflicts", "get_working_hours",
                           "export_calendar", "get_recurring_info"):
            # read-only tools — no state changes
            result["target_id"] = tool_arguments.get("event_id", "")
            result["changed_fields"] = []

        # Forbidden transition detection:
        # delete + create with same/similar target is a forbidden pattern
        # This is detected across events by SafetyVerifier, not per-event.
        # But we can set a preliminary flag here if needed.

        return result




    def protected_resources(self, task: dict[str, Any]) -> list[str]:
        # Calendar: protected resources are target event IDs that must not be deleted
        return task.get("protected_event_ids", [])

    def budget(self, task: dict[str, Any]) -> int:
        return task.get("budget", 5)

    def identity_policy(self, task: dict[str, Any]) -> str:
        return task.get("identity_policy", "preserve")


class EmailAdapter(DomainAdapter):
    """Domain adapter for email MCP server.

    Email state: append-only, threads, labels.
    target_type: "email" / "email_thread"
    identity_policy: "append_only"
    """

    domain_name = "email"
    entity_container_key = "emails"

    TOOL_MAP = {
        "list_inbox": ("query", "email"),
        "search_emails": ("query", "email"),
        "get_email": ("query", "email"),
        "send_email": ("create", "email"),
        "create_draft": ("create", "email_draft"),
        "add_label": ("update", "email"),
        "move_to_thread": ("update", "email_thread"),
        "get_thread": ("query", "email_thread"),
    }

    def normalize_event(
        self, action_type, tool_name, tool_arguments, observation,
        execution_success, state_changed, before_state, after_state,
    ) -> dict[str, Any]:
        if action_type != "tool_call":
            return {
                "operation": "terminal", "target_type": "", "target_id": "",
                "changed_fields": [], "created_ids": [], "deleted_ids": [],
                "identity_violation": "", "forbidden_transition": "",
                "duplicate_of": None,
            }
        op, ttype = self.tool_semantics(tool_name, "email", state_changed)
        result: dict[str, Any] = {
            "operation": op, "target_type": ttype, "target_id": "",
            "changed_fields": [], "created_ids": [], "deleted_ids": [],
            "identity_violation": "", "forbidden_transition": "", "duplicate_of": None,
        }
        if tool_name == "send_email":
            if execution_success and isinstance(observation, dict):
                email = observation.get("email", observation)
                result["target_id"] = email.get("email_id", "")
                result["created_ids"] = [result["target_id"]] if result["target_id"] else []
            result["changed_fields"] = ["inbox", "thread"]
        elif tool_name == "add_label":
            result["target_id"] = tool_arguments.get("email_id", "")
            result["changed_fields"] = ["labels"]
        elif tool_name == "move_to_thread":
            result["target_id"] = tool_arguments.get("email_id", "")
            result["changed_fields"] = ["thread_id"]
        elif tool_name in ("get_email", "get_thread"):
            result["target_id"] = tool_arguments.get("email_id", tool_arguments.get("thread_id", ""))
        # Append-only: no deletes allowed
        return result

    def protected_resources(self, task): return task.get("protected_thread_ids", [])
    def budget(self, task): return task.get("budget", 5)
    def identity_policy(self, task): return task.get("identity_policy", "append_only")


class TeamChatAdapter(DomainAdapter):
    """Domain adapter for team chat MCP server.

    Team chat state: append-only channels, messages, threads.
    target_type: "message" / "channel" / "thread"
    identity_policy: "append_only"
    """

    domain_name = "team_chat"

    TOOL_MAP = {
        "list_channels": ("query", "channel"),
        "get_channel": ("query", "channel"),
        "send_message": ("create", "message"),
        "create_thread": ("create", "thread"),
        "get_thread": ("query", "thread"),
        "react_message": ("update", "message"),
    }

    def normalize_event(
        self, action_type, tool_name, tool_arguments, observation,
        execution_success, state_changed, before_state, after_state,
    ) -> dict[str, Any]:
        if action_type != "tool_call":
            return {
                "operation": "terminal", "target_type": "", "target_id": "",
                "changed_fields": [], "created_ids": [], "deleted_ids": [],
                "identity_violation": "", "forbidden_transition": "",
                "duplicate_of": None,
            }
        op, ttype = self.tool_semantics(tool_name, "message", state_changed)
        result: dict[str, Any] = {
            "operation": op, "target_type": ttype, "target_id": "",
            "changed_fields": [], "created_ids": [], "deleted_ids": [],
            "identity_violation": "", "forbidden_transition": "", "duplicate_of": None,
        }
        if tool_name == "send_message":
            result["target_id"] = tool_arguments.get("channel_id", "")
            if execution_success and isinstance(observation, dict):
                msg = observation.get("message", observation)
                if isinstance(msg, dict):
                    result["created_ids"] = [msg.get("message_id", "")]
            # Detect send to archived channel
            if not execution_success:
                error_msg = observation.get("error_message", "") if isinstance(observation, dict) else ""
                if "not found" in str(error_msg):
                    result["forbidden_transition"] = "send_to_nonexistent_channel"
        elif tool_name == "create_thread":
            result["target_id"] = tool_arguments.get("message_id", "")
            if execution_success and isinstance(observation, dict):
                thread = observation.get("thread", observation)
                if isinstance(thread, dict):
                    result["created_ids"] = [thread.get("thread_id", "")]
        elif tool_name == "react_message":
            result["target_id"] = tool_arguments.get("message_id", "")
            result["changed_fields"] = ["reactions"]
        elif tool_name == "get_channel":
            result["target_id"] = tool_arguments.get("channel_id", "")
        elif tool_name == "get_thread":
            result["target_id"] = tool_arguments.get("thread_id", "")
        return result

    def protected_resources(self, task): return task.get("protected_channel_ids", [])
    def budget(self, task): return task.get("budget", 4)
    def identity_policy(self, task): return task.get("identity_policy", "append_only")


__all__ = ["CalendarAdapter", "EmailAdapter", "TeamChatAdapter"]
