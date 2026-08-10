"""Deterministic state builder for calendar."""

from __future__ import annotations

import datetime as _datetime
import random
from typing import Any

from src.live_mcp.state_seeders.common import (
    _reference_datetime,
    _sample_entities,
    _seed_scoped_id,
)
_CALENDAR_EVENT_TEMPLATES: list[tuple[str, str, str, str]] = [
    ("Team Sync", "alex@example.com", "Weekly team sync up", "Room 101"),
    ("Design Review", "sam@example.com", "Review new UI mockups", "Room 202"),
    ("Budget Check", "alex@example.com", "Q2 budget review", "Room 303"),
    ("Customer Call", "alex@example.com", "Onboarding call with new client", "Room 404"),
    ("Sprint Planning", "charlie@example.com", "Plan sprint 25 scope", "Room 105"),
    ("1-on-1 Check-in", "sam@example.com", "Weekly 1-on-1", "Cafe"),
    ("Product Demo", "dana@example.com", "Demo new features to stakeholders", "Boardroom"),
    ("Retrospective", "alex@example.com", "Sprint retrospective", "Room 202"),
    ("All Hands", "ceo@example.com", "Quarterly all-hands meeting", "Auditorium"),
    ("Interview", "hr@example.com", "Technical interview panel", "Room 106"),
    ("Lunch & Learn", "dana@example.com", "Knowledge sharing session", "Room 303"),
    ("Release Planning", "charlie@example.com", "Roadmap review", "Room 105"),
    ("Vendor Meeting", "alex@example.com", "Contract negotiation", "Boardroom"),
    ("Standup", "charlie@example.com", "Daily standup", "Virtual"),
    ("Workshop", "dana@example.com", "Design thinking workshop", "Room 404"),
    ("Code Review", "bob@example.com", "Review PR #342", "Room 101"),
    ("Client Onboarding", "alex@example.com", "New client kickoff", "Boardroom"),
    ("OKR Review", "ceo@example.com", "Mid-quarter OKR check-in", "Room 202"),
    ("Training Session", "hr@example.com", "Compliance training", "Room 105"),
    ("Town Hall", "ceo@example.com", "Monthly town hall", "Auditorium"),
    ("Architecture Review", "bob@example.com", "System design review", "Room 101"),
    ("Security Briefing", "hr@example.com", "Quarterly security update", "Room 303"),
    ("Partner Sync", "dana@example.com", "Sync with external partner", "Virtual"),
    ("Hiring Panel", "hr@example.com", "Final round interview", "Room 106"),
    ("Post-mortem", "charlie@example.com", "Incident post-mortem", "Room 202"),
]

def _calendar_state(seed: int) -> dict[str, Any]:
    # Calendar entities and the Teacher/Policy temporal anchor must describe
    # the same world. A fixed June fixture made prompts such as "today" refer
    # to November while grounded events still lived in June.
    rng = random.Random(seed)
    events: dict[str, dict[str, Any]] = {}
    selected = _sample_entities(
        rng, _CALENDAR_EVENT_TEMPLATES, target_count=20, id_prefix="evt",
    )
    reference_date = _reference_datetime(seed).date()
    attendee_pool = ["alex@example.com", "sam@example.com", "charlie@example.com",
                     "dana@example.com", "bob@example.com", "erin@example.com", "frank@example.com"]
    for idx, entry in enumerate(selected):
        if len(entry) == 4:
            title, lead, desc, location = entry
        else:
            title, lead, desc = entry[1:4]
            location = f"Room {100 + idx}"
        event_date = reference_date + _datetime.timedelta(days=(idx % 9) - 4)
        hour = 9 + (idx + rng.randint(0, 3)) % 6
        eid = _seed_scoped_id("evt", seed, idx, width=3)
        recurrence = None
        recurrence_until = None
        recurrence_count = None
        if idx % 3 == 0:
            weekday = ("MO", "TU", "WE", "TH", "FR", "SA", "SU")[
                event_date.weekday()
            ]
            recurrence = f"FREQ=WEEKLY;BYDAY={weekday}"
            recurrence_count = 4 + idx % 5
        reminders = []
        if idx % 3 == 0 or idx % 4 == 0:
            reminders = [{
                "id": "rem_1",
                "minutes_before": (15, 30, 60)[idx % 3],
                "method": ("popup", "email")[idx % 2],
            }]
        events[eid] = {
            "event_id": eid, "title": title,
            "start_time": f"{event_date.isoformat()}T{hour:02d}:00",
            "end_time": f"{event_date.isoformat()}T{hour + 1:02d}:00",
            "description": desc, "location": location,
            "attendees": rng.sample(attendee_pool, k=1 + idx % 3),
            "reminders": reminders,
            "recurrence": recurrence,
            "recurrence_until": recurrence_until,
            "recurrence_count": recurrence_count,
        }
    return {"events": events, "next_event_num": len(events) + 1,
            "timezone": "America/New_York", "current_date": reference_date.isoformat()}
