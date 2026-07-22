"""Stateful calendar server with 17 tools.
Features: events, recurring rules, attendees, reminders, free/busy, timezone, conflicts.
"""

from __future__ import annotations
from datetime import datetime
import re
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
from src.live_mcp.server_base import StatefulToolServer, _result, serve


_EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_EMAIL_SCHEMA = {
    "type": "string",
    "format": "email",
    "description": "Exact attendee email address, not a display name.",
}
_EMAILS_SCHEMA = {
    "type": "array",
    "items": _EMAIL_SCHEMA,
    "uniqueItems": True,
}

TOOLS = [
    {"name": "list_events", "description": "List calendar events with optional filters.", "input_schema": {"type": "object", "properties": {"date_range": {"type": "string"}, "attendee": {"type": "string"}, "keyword": {"type": "string"}}, "required": []}, "annotations": {"readonly": True, "mutating": False}},
    {"name": "search_events", "description": "Search events by keyword in title/description.", "input_schema": {"type": "object", "properties": {"query": {"type": "string"}, "start_after": {"type": "string"}, "start_before": {"type": "string"}}, "required": ["query"]}, "annotations": {"readonly": True, "mutating": False}},
    {"name": "get_event", "description": "Get a single event by id with full details.", "input_schema": {"type": "object", "properties": {"event_id": {"type": "string"}}, "required": ["event_id"]}, "annotations": {"readonly": True, "mutating": False}},
    {"name": "create_event", "description": "Create a single calendar event. Attendees must be deliverable email addresses, not display names.", "input_schema": {"type": "object", "properties": {"title": {"type": "string"}, "start_time": {"type": "string"}, "end_time": {"type": "string"}, "description": {"type": "string"}, "location": {"type": "string"}, "attendees": _EMAILS_SCHEMA, "reminders": {"type": "array"}}, "required": ["title", "start_time", "end_time"]}, "annotations": {"mutating": True}},
    {"name": "update_event", "description": "Update at least one supported field of an existing event (preserves identity). Attendees must be deliverable email addresses, not display names.", "input_schema": {"type": "object", "properties": {"event_id": {"type": "string"}, "fields": {"type": "object", "properties": {"title": {"type": "string"}, "start_time": {"type": "string"}, "end_time": {"type": "string"}, "description": {"type": "string"}, "location": {"type": "string"}, "attendees": _EMAILS_SCHEMA, "reminders": {"type": "array"}}, "additionalProperties": False, "minProperties": 1}}, "required": ["event_id", "fields"], "additionalProperties": False}, "annotations": {"mutating": True}},
    {"name": "delete_event", "description": "Delete an event by id.", "input_schema": {"type": "object", "properties": {"event_id": {"type": "string"}}, "required": ["event_id"]}, "annotations": {"mutating": True}},
    {"name": "create_recurring", "description": "Create a recurring event with an RRULE-style rule. When recurrence contains BYDAY, start_time must fall on one of those weekdays. Attendees must be deliverable email addresses.", "input_schema": {"type": "object", "properties": {"title": {"type": "string"}, "start_time": {"type": "string"}, "end_time": {"type": "string"}, "recurrence": {"type": "string"}, "attendees": _EMAILS_SCHEMA, "until": {"type": "string"}, "count": {"type": "integer", "minimum": 1}}, "required": ["title", "start_time", "end_time", "recurrence"]}, "annotations": {"mutating": True}},
    {"name": "add_attendee", "description": "Add an attendee by exact email address and optionally set the initial invitation response.", "input_schema": {"type": "object", "properties": {"event_id": {"type": "string"}, "email": _EMAIL_SCHEMA, "response_status": {"type": "string", "enum": ["accepted", "declined", "tentative"]}}, "required": ["event_id", "email"]}, "annotations": {"mutating": True}},
    {"name": "remove_attendee", "description": "Remove an attendee by exact email address.", "input_schema": {"type": "object", "properties": {"event_id": {"type": "string"}, "email": _EMAIL_SCHEMA}, "required": ["event_id", "email"]}, "annotations": {"mutating": True}},
    {"name": "get_free_busy", "description": "Get free/busy slots for exact attendee email addresses in a time range.", "input_schema": {"type": "object", "properties": {"emails": {**_EMAILS_SCHEMA, "minItems": 1}, "start_time": {"type": "string"}, "end_time": {"type": "string"}}, "required": ["emails", "start_time", "end_time"]}, "annotations": {"readonly": True, "mutating": False}},
    {"name": "check_conflicts", "description": "Check if a proposed event conflicts with existing events.", "input_schema": {"type": "object", "properties": {"start_time": {"type": "string"}, "end_time": {"type": "string"}, "exclude_event_id": {"type": "string"}}, "required": ["start_time", "end_time"]}, "annotations": {"readonly": True, "mutating": False}},
    {"name": "set_reminder", "description": "Set a reminder for an event.", "input_schema": {"type": "object", "properties": {"event_id": {"type": "string"}, "minutes_before": {"type": "integer", "minimum": 1, "description": "Positive number of minutes before the event."}, "method": {"type": "string"}}, "required": ["event_id", "minutes_before"]}, "annotations": {"mutating": True}},
    {"name": "get_working_hours", "description": "Get working hours for specified working days.", "input_schema": {"type": "object", "properties": {"date": {"type": "string"}}, "required": []}, "annotations": {"readonly": True, "mutating": False}},
    {"name": "change_timezone", "description": "Change the calendar timezone preference. Existing event timestamps remain stored as supplied.", "input_schema": {"type": "object", "properties": {"timezone": {"type": "string"}}, "required": ["timezone"]}, "annotations": {"mutating": True}},
    {"name": "respond_to_event", "description": "Record an invitation response for an email that is already in the event's current attendees list.", "input_schema": {"type": "object", "properties": {"event_id": {"type": "string"}, "email": _EMAIL_SCHEMA, "response": {"type": "string", "enum": ["accepted", "declined", "tentative"]}}, "required": ["event_id", "email", "response"]}, "annotations": {"mutating": True}},
    {"name": "export_calendar", "description": "Export events in a date range to iCal/JSON format.", "input_schema": {"type": "object", "properties": {"start_date": {"type": "string"}, "end_date": {"type": "string"}, "format": {"type": "string"}}, "required": []}, "annotations": {"readonly": True, "mutating": False}},
    {"name": "get_recurring_info", "description": "Get recurrence metadata for a recurring event.", "input_schema": {"type": "object", "properties": {"event_id": {"type": "string"}}, "required": ["event_id"]}, "annotations": {"readonly": True, "mutating": False}},
]

ALLOWED_FIELDS = {"title", "start_time", "end_time", "description", "location", "attendees", "reminders"}

class CalendarServer(StatefulToolServer):
    def __init__(self) -> None:
        super().__init__("calendar", TOOLS)
        self.handlers = {t["name"]: getattr(self, t["name"]) for t in TOOLS}

    def _eid(self, state): eid = f"evt_{state['next_event_num']:03d}"; state["next_event_num"] += 1; return eid
    @staticmethod
    def _email(value: Any, field: str = "email") -> str:
        email = str(value)
        if not _EMAIL_PATTERN.fullmatch(email):
            raise KeyError(f"{field} must be a valid email address")
        return email

    @classmethod
    def _emails(cls, value: Any, field: str = "attendees") -> list[str]:
        if not isinstance(value, list):
            raise KeyError(f"{field} must be an array of email addresses")
        emails = [cls._email(item, field) for item in value]
        if len(set(emails)) != len(emails):
            raise KeyError(f"{field} must not contain duplicate email addresses")
        return emails

    @staticmethod
    def _datetime(value: Any, field: str) -> datetime:
        try:
            return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError as exc:
            raise KeyError(f"{field} must be an ISO-8601 datetime") from exc

    @classmethod
    def _time_range(cls, start: Any, end: Any) -> None:
        start_dt = cls._datetime(start, "start_time")
        end_dt = cls._datetime(end, "end_time")
        if start_dt >= end_dt:
            raise KeyError("start_time must be before end_time")

    @classmethod
    def _recurrence_start(cls, start: Any, recurrence: Any) -> None:
        """Validate the concrete first occurrence against an RRULE BYDAY."""
        fields: dict[str, str] = {}
        for item in str(recurrence).upper().split(";"):
            if "=" not in item:
                continue
            key, value = item.split("=", 1)
            fields[key.strip()] = value.strip()
        byday = fields.get("BYDAY")
        if not byday:
            return
        weekday_by_code = {
            "MO": 0, "TU": 1, "WE": 2, "TH": 3,
            "FR": 4, "SA": 5, "SU": 6,
        }
        codes = [code.strip() for code in byday.split(",") if code.strip()]
        if not codes or any(code not in weekday_by_code for code in codes):
            raise KeyError("recurrence BYDAY must contain valid weekday codes")
        start_weekday = cls._datetime(start, "start_time").weekday()
        if start_weekday not in {weekday_by_code[code] for code in codes}:
            raise KeyError("start_time weekday must match recurrence BYDAY")

    def _evt(self, state, eid):
        if eid not in state["events"]: raise KeyError(f"event not found: {eid}")
        return state["events"][eid]

    def list_events(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id); events = list(state["events"].values())
        dr = arguments.get("date_range"); attendee = arguments.get("attendee"); kw = arguments.get("keyword", "").lower()
        if dr: events = [e for e in events if e.get("start_time", "")[:10] in dr or dr in str(e.get("start_time", ""))]
        if attendee: events = [e for e in events if attendee in e.get("attendees", [])]
        if kw: events = [e for e in events if kw in e.get("title", "").lower() or kw in e.get("description", "").lower()]
        return _result(True, {"events": events, "count": len(events)}, None, "", False)

    def search_events(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id); q = arguments["query"].lower()
        events = [e for e in state["events"].values() if q in e.get("title", "").lower() or q in e.get("description", "").lower()]
        sa = arguments.get("start_after"); sb = arguments.get("start_before")
        if sa: events = [e for e in events if e["start_time"] >= sa]
        if sb: events = [e for e in events if e["start_time"] <= sb]
        return _result(True, {"events": events, "count": len(events)}, None, "", False)

    def get_event(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        return _result(True, {"event": self._evt(self._state(session_id), arguments["event_id"])}, None, "", False)

    def create_event(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        self._time_range(arguments["start_time"], arguments["end_time"])
        attendees = self._emails(arguments.get("attendees", []))
        state = self._state(session_id); eid = self._eid(state)
        event = {"event_id": eid, "title": arguments["title"], "start_time": arguments["start_time"], "end_time": arguments["end_time"], "description": arguments.get("description", ""), "location": arguments.get("location", ""), "attendees": attendees, "reminders": list(arguments.get("reminders", [])), "recurrence": None}
        state["events"][eid] = event
        return _result(True, {"event": event}, None, "", True)

    def update_event(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id); evt = self._evt(state, arguments["event_id"])
        fields = arguments["fields"]
        if not fields: raise KeyError("at least one event field is required")
        unsupported = sorted(set(fields) - ALLOWED_FIELDS)
        if unsupported: raise KeyError(f"unsupported event field(s): {', '.join(unsupported)}")
        start_time = fields.get("start_time", evt["start_time"])
        end_time = fields.get("end_time", evt["end_time"])
        self._time_range(start_time, end_time)
        if "attendees" in fields:
            fields = {**fields, "attendees": self._emails(fields["attendees"])}
        changed = False
        for k, v in fields.items():
            if k in ALLOWED_FIELDS and evt.get(k) != v:
                evt[k] = v
                changed = True
        return _result(True, {"event": evt}, None, "", changed)

    def delete_event(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id); eid = arguments["event_id"]; evt = self._evt(state, eid)
        del state["events"][eid]
        return _result(True, {"deleted_event": evt}, None, "", True)

    def create_recurring(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        self._time_range(arguments["start_time"], arguments["end_time"])
        self._recurrence_start(arguments["start_time"], arguments["recurrence"])
        attendees = self._emails(arguments.get("attendees", []))
        if arguments.get("count") is not None and arguments["count"] < 1:
            raise KeyError("count must be at least 1")
        state = self._state(session_id); eid = self._eid(state)
        event = {"event_id": eid, "title": arguments["title"], "start_time": arguments["start_time"], "end_time": arguments["end_time"], "description": "", "location": "", "attendees": attendees, "reminders": [], "recurrence": arguments["recurrence"], "recurrence_until": arguments.get("until"), "recurrence_count": arguments.get("count")}
        state["events"][eid] = event
        return _result(True, {"event": event, "recurrence": arguments["recurrence"]}, None, "", True)

    def add_attendee(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        evt = self._evt(self._state(session_id), arguments["event_id"]); email = self._email(arguments["email"])
        if email in evt.setdefault("attendees", []):
            old = evt.setdefault("responses", {}).get(email)
            response = arguments.get("response_status")
            if response is not None:
                evt["responses"][email] = response
            return _result(True, {"event_id": evt["event_id"], "attendees": evt["attendees"], "response": response}, None, "", response is not None and old != response)
        evt["attendees"].append(email)
        response = arguments.get("response_status")
        if response is not None:
            evt.setdefault("responses", {})[email] = response
        return _result(True, {"event_id": evt["event_id"], "attendees": evt["attendees"], "response": response}, None, "", True)

    def remove_attendee(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        evt = self._evt(self._state(session_id), arguments["event_id"])
        email = self._email(arguments["email"])
        attendees = evt.get("attendees", [])
        if email not in attendees:
            return _result(True, {"event_id": evt["event_id"], "attendees": attendees}, None, "", False)
        evt["attendees"] = [a for a in attendees if a != email]
        evt.setdefault("responses", {}).pop(email, None)
        return _result(True, {"event_id": evt["event_id"], "attendees": evt["attendees"]}, None, "", True)

    def get_free_busy(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id); emails = self._emails(arguments["emails"], "emails"); st, et = arguments["start_time"], arguments["end_time"]
        if not emails: raise KeyError("emails must contain at least one email address")
        busy = {}
        for email in emails:
            busy[email] = []
            for e in state["events"].values():
                if email in e.get("attendees", []):
                    if e["start_time"] < et and e["end_time"] > st:
                        busy[email].append({"start": e["start_time"], "end": e["end_time"], "event_id": e["event_id"], "title": e["title"]})
        return _result(True, {"busy": busy, "query_range": {"start": st, "end": et}}, None, "", False)

    def check_conflicts(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id); st, et = arguments["start_time"], arguments["end_time"]; exclude = arguments.get("exclude_event_id")
        conflicts = [{"event_id": e["event_id"], "title": e["title"], "start": e["start_time"], "end": e["end_time"]} for e in state["events"].values() if e["event_id"] != exclude and e["start_time"] < et and e["end_time"] > st]
        return _result(True, {"has_conflicts": len(conflicts) > 0, "conflicts": conflicts}, None, "", False)

    def set_reminder(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        evt = self._evt(self._state(session_id), arguments["event_id"])
        rid = f"rem_{len(evt.get('reminders', [])) + 1}"; mins = int(arguments["minutes_before"]); method = arguments.get("method", "popup")
        if mins <= 0: raise KeyError("minutes_before must be positive")
        evt.setdefault("reminders", []).append({"id": rid, "minutes_before": mins, "method": method})
        return _result(True, {"event_id": evt["event_id"], "reminders": evt["reminders"]}, None, "", True)

    def get_working_hours(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        return _result(True, {"working_hours": {"monday": "09:00-18:00", "tuesday": "09:00-18:00", "wednesday": "09:00-18:00", "thursday": "09:00-18:00", "friday": "09:00-17:00", "saturday": None, "sunday": None}, "timezone": self._state(session_id).get("timezone", "America/New_York")}, None, "", False)

    def change_timezone(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id); tz = arguments["timezone"]; old = state.get("timezone")
        try:
            ZoneInfo(tz)
        except ZoneInfoNotFoundError as exc:
            raise KeyError(f"unknown IANA timezone: {tz}") from exc
        state["timezone"] = tz
        return _result(True, {"timezone": tz, "message": f"timezone changed to {tz}"}, None, "", old != tz)

    def respond_to_event(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        evt = self._evt(self._state(session_id), arguments["event_id"]); email = self._email(arguments["email"]); resp = arguments["response"]
        if email not in evt.get("attendees", []): raise KeyError(f"attendee not found: {email}")
        if resp not in ("accepted", "declined", "tentative"): raise KeyError(f"invalid response: {resp}")
        old = evt.setdefault("responses", {}).get(email)
        evt.setdefault("responses", {})[email] = resp
        return _result(True, {"event_id": evt["event_id"], "email": email, "response": resp}, None, "", old != resp)

    def export_calendar(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id); fmt = arguments.get("format", "json")
        events = list(state["events"].values())
        sd = arguments.get("start_date"); ed = arguments.get("end_date")
        if sd: events = [e for e in events if e["start_time"][:10] >= sd]
        if ed: events = [e for e in events if e["start_time"][:10] <= ed]
        return _result(True, {"format": fmt, "events": events, "count": len(events)}, None, "", False)

    def get_recurring_info(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        evt = self._evt(self._state(session_id), arguments["event_id"])
        if not evt.get("recurrence"): raise KeyError("not a recurring event")
        return _result(True, {"event_id": evt["event_id"], "recurrence": evt["recurrence"], "until": evt.get("recurrence_until"), "count": evt.get("recurrence_count")}, None, "", False)


if __name__ == "__main__":
    serve(CalendarServer())
