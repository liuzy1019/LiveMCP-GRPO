"""Stateful calendar server — 17 tools (PROVE-aligned).
Features: events, recurring rules, attendees, reminders, free/busy, timezone, conflicts.
"""

from __future__ import annotations
from typing import Any
from src.live_mcp.server_base import StatefulToolServer, _result, serve

TOOLS = [
    {"name": "list_events", "description": "List calendar events with optional filters.", "input_schema": {"type": "object", "properties": {"date_range": {"type": "string"}, "attendee": {"type": "string"}, "keyword": {"type": "string"}}, "required": []}, "annotations": {"readonly": True, "mutating": False}},
    {"name": "search_events", "description": "Search events by keyword in title/description.", "input_schema": {"type": "object", "properties": {"query": {"type": "string"}, "start_after": {"type": "string"}, "start_before": {"type": "string"}}, "required": ["query"]}, "annotations": {"readonly": True, "mutating": False}},
    {"name": "get_event", "description": "Get a single event by id with full details.", "input_schema": {"type": "object", "properties": {"event_id": {"type": "string"}}, "required": ["event_id"]}, "annotations": {"readonly": True, "mutating": False}},
    {"name": "create_event", "description": "Create a single calendar event.", "input_schema": {"type": "object", "properties": {"title": {"type": "string"}, "start_time": {"type": "string"}, "end_time": {"type": "string"}, "description": {"type": "string"}, "location": {"type": "string"}, "attendees": {"type": "array"}, "reminders": {"type": "array"}}, "required": ["title", "start_time", "end_time"]}, "annotations": {"mutating": True}},
    {"name": "update_event", "description": "Update fields of an existing event (preserves identity).", "input_schema": {"type": "object", "properties": {"event_id": {"type": "string"}, "fields": {"type": "object"}}, "required": ["event_id", "fields"]}, "annotations": {"mutating": True}},
    {"name": "delete_event", "description": "Delete an event by id.", "input_schema": {"type": "object", "properties": {"event_id": {"type": "string"}}, "required": ["event_id"]}, "annotations": {"mutating": True}},
    {"name": "create_recurring", "description": "Create a recurring event with rule.", "input_schema": {"type": "object", "properties": {"title": {"type": "string"}, "start_time": {"type": "string"}, "end_time": {"type": "string"}, "recurrence": {"type": "string"}, "attendees": {"type": "array"}, "until": {"type": "string"}, "count": {"type": "integer"}}, "required": ["title", "start_time", "end_time", "recurrence"]}, "annotations": {"mutating": True}},
    {"name": "add_attendee", "description": "Add an attendee to an event.", "input_schema": {"type": "object", "properties": {"event_id": {"type": "string"}, "email": {"type": "string"}, "response_status": {"type": "string"}}, "required": ["event_id", "email"]}, "annotations": {"mutating": True}},
    {"name": "remove_attendee", "description": "Remove an attendee from an event.", "input_schema": {"type": "object", "properties": {"event_id": {"type": "string"}, "email": {"type": "string"}}, "required": ["event_id", "email"]}, "annotations": {"mutating": True}},
    {"name": "get_free_busy", "description": "Get free/busy slots for attendees in a time range.", "input_schema": {"type": "object", "properties": {"emails": {"type": "array"}, "start_time": {"type": "string"}, "end_time": {"type": "string"}}, "required": ["emails", "start_time", "end_time"]}, "annotations": {"readonly": True, "mutating": False}},
    {"name": "check_conflicts", "description": "Check if a proposed event conflicts with existing events.", "input_schema": {"type": "object", "properties": {"start_time": {"type": "string"}, "end_time": {"type": "string"}, "exclude_event_id": {"type": "string"}}, "required": ["start_time", "end_time"]}, "annotations": {"readonly": True, "mutating": False}},
    {"name": "set_reminder", "description": "Set a reminder for an event.", "input_schema": {"type": "object", "properties": {"event_id": {"type": "string"}, "minutes_before": {"type": "integer"}, "method": {"type": "string"}}, "required": ["event_id", "minutes_before"]}, "annotations": {"mutating": True}},
    {"name": "get_working_hours", "description": "Get working hours for specified working days.", "input_schema": {"type": "object", "properties": {"date": {"type": "string"}}, "required": []}, "annotations": {"readonly": True, "mutating": False}},
    {"name": "change_timezone", "description": "Change the display timezone for event times.", "input_schema": {"type": "object", "properties": {"timezone": {"type": "string"}}, "required": ["timezone"]}, "annotations": {"mutating": True}},
    {"name": "respond_to_event", "description": "Respond to an event invitation (accept/decline/tentative).", "input_schema": {"type": "object", "properties": {"event_id": {"type": "string"}, "email": {"type": "string"}, "response": {"type": "string"}}, "required": ["event_id", "email", "response"]}, "annotations": {"mutating": True}},
    {"name": "export_calendar", "description": "Export events in a date range to iCal/JSON format.", "input_schema": {"type": "object", "properties": {"start_date": {"type": "string"}, "end_date": {"type": "string"}, "format": {"type": "string"}}, "required": []}, "annotations": {"readonly": True, "mutating": False}},
    {"name": "get_recurring_info", "description": "Get recurrence metadata for a recurring event.", "input_schema": {"type": "object", "properties": {"event_id": {"type": "string"}}, "required": ["event_id"]}, "annotations": {"readonly": True, "mutating": False}},
]

ALLOWED_FIELDS = {"title", "start_time", "end_time", "description", "location", "attendees", "reminders"}

class CalendarServer(StatefulToolServer):
    def __init__(self) -> None:
        super().__init__("calendar", TOOLS)
        self.handlers = {t["name"]: getattr(self, t["name"]) for t in TOOLS}

    def _eid(self, state): eid = f"evt_{state['next_event_num']:03d}"; state["next_event_num"] += 1; return eid
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
        state = self._state(session_id); eid = self._eid(state)
        event = {"event_id": eid, "title": arguments["title"], "start_time": arguments["start_time"], "end_time": arguments["end_time"], "description": arguments.get("description", ""), "location": arguments.get("location", ""), "attendees": list(arguments.get("attendees", [])), "reminders": list(arguments.get("reminders", [])), "recurrence": None}
        state["events"][eid] = event
        return _result(True, {"event": event}, None, "", True)

    def update_event(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id); evt = self._evt(state, arguments["event_id"])
        for k, v in arguments["fields"].items():
            if k in ALLOWED_FIELDS: evt[k] = v
        return _result(True, {"event": evt}, None, "", True)

    def delete_event(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id); eid = arguments["event_id"]; evt = state["events"].pop(eid)
        return _result(True, {"deleted_event": evt}, None, "", True)

    def create_recurring(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id); eid = self._eid(state)
        event = {"event_id": eid, "title": arguments["title"], "start_time": arguments["start_time"], "end_time": arguments["end_time"], "description": "", "location": "", "attendees": list(arguments.get("attendees", [])), "reminders": [], "recurrence": arguments["recurrence"], "recurrence_until": arguments.get("until"), "recurrence_count": arguments.get("count")}
        state["events"][eid] = event
        return _result(True, {"event": event, "recurrence": arguments["recurrence"]}, None, "", True)

    def add_attendee(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        evt = self._evt(self._state(session_id), arguments["event_id"]); email = arguments["email"]
        if email not in evt.setdefault("attendees", []): evt["attendees"].append(email)
        return _result(True, {"event_id": evt["event_id"], "attendees": evt["attendees"]}, None, "", True)

    def remove_attendee(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        evt = self._evt(self._state(session_id), arguments["event_id"])
        email = arguments["email"]; evt["attendees"] = [a for a in evt.get("attendees", []) if a != email]
        return _result(True, {"event_id": evt["event_id"], "attendees": evt["attendees"]}, None, "", True)

    def get_free_busy(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id); emails = arguments["emails"]; st, et = arguments["start_time"], arguments["end_time"]
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
        evt.setdefault("reminders", []).append({"id": rid, "minutes_before": mins, "method": method})
        return _result(True, {"event_id": evt["event_id"], "reminders": evt["reminders"]}, None, "", True)

    def get_working_hours(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        return _result(True, {"working_hours": {"monday": "09:00-18:00", "tuesday": "09:00-18:00", "wednesday": "09:00-18:00", "thursday": "09:00-18:00", "friday": "09:00-17:00", "saturday": None, "sunday": None}, "timezone": "America/New_York"}, None, "", False)

    def change_timezone(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id); tz = arguments["timezone"]; state["timezone"] = tz
        return _result(True, {"timezone": tz, "message": f"timezone changed to {tz}"}, None, "", True)

    def respond_to_event(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        evt = self._evt(self._state(session_id), arguments["event_id"]); email = arguments["email"]; resp = arguments["response"]
        if resp not in ("accepted", "declined", "tentative"): raise KeyError(f"invalid response: {resp}")
        evt.setdefault("responses", {})[email] = resp
        return _result(True, {"event_id": evt["event_id"], "email": email, "response": resp}, None, "", True)

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
