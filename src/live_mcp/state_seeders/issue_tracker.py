"""Deterministic state builder for issue_tracker."""

from __future__ import annotations

import datetime as _datetime
import random
from typing import Any

from src.live_mcp.state_seeders.common import (
    _reference_datetime,
    _sample_entities,
    _seed_scoped_id,
)
_ISSUE_TEMPLATES: list[tuple[str, str, str, str, str, str, str | None]] = [
    ("iss_0001", "Login timeout on mobile", "Users report 30s timeout on iOS.", "high", "bug", "open", None),
    ("iss_0002", "Add dark mode support", "Feature request for dark mode.", "medium", "feature", "in_progress", "bob"),
    ("iss_0003", "Fix PDF export layout", "Tables misaligned in PDF export.", "high", "bug", "in_review", "alice"),
    ("iss_0004", "Update dependencies", "Security audit flagged outdated packages.", "medium", "maintenance", "resolved", "charlie"),
    ("iss_0005", "API rate limiting", "Add rate limiting to public API.", "high", "feature", "open", None),
    ("iss_0006", "Cache invalidation bug", "Stale cache after user profile update.", "critical", "bug", "in_progress", "bob"),
    ("iss_0007", "Search pagination", "Search results missing page controls.", "medium", "bug", "open", None),
    ("iss_0008", "Email notification delay", "Some users not receiving notifications.", "high", "bug", "in_review", "dana"),
    ("iss_0009", "i18n support Phase 1", "Add i18n framework for UI strings.", "low", "feature", "open", None),
    ("iss_0010", "Database migration script", "Schema migration for v3.0.", "medium", "maintenance", "resolved", "charlie"),
    ("iss_0011", "WebSocket reconnection", "Clients disconnect on network flap.", "high", "bug", "in_progress", "alice"),
    ("iss_0012", "Dashboard refresh", "Real-time dashboard widget updates.", "medium", "feature", "open", None),
    ("iss_0013", "Memory leak in worker", "Worker process grows unbounded over 24h.", "critical", "bug", "open", None),
    ("iss_0014", "Add CSV export", "Users want to export table data as CSV.", "low", "feature", "open", None),
    ("iss_0015", "Fix timezone handling", "Events show wrong time in non-UTC zones.", "high", "bug", "in_progress", "dana"),
    ("iss_0016", "Improve error messages", "Generic 500 errors need user-friendly text.", "medium", "improvement", "open", None),
    ("iss_0017", "Add 2FA support", "Implement TOTP-based two-factor auth.", "high", "feature", "open", None),
    ("iss_0018", "Slow query on reports", "Report page takes 8s to load.", "high", "performance", "in_review", "charlie"),
    ("iss_0019", "Broken link in docs", "API docs link to 404 page.", "low", "bug", "resolved", "bob"),
    ("iss_0020", "Add audit log", "Track all admin actions in audit log.", "medium", "feature", "open", None),
]

_ISSUE_TRACKER_MEMBER_TEMPLATES: list[tuple[str, str, str]] = [
    ("alice", "Alice", "developer"),
    ("bob", "Bob", "senior developer"),
    ("charlie", "Charlie", "tech lead"),
    ("dana", "Dana", "designer"),
    ("eric", "Eric", "backend engineer"),
    ("fiona", "Fiona", "QA engineer"),
    ("george", "George", "devops engineer"),
    ("hannah", "Hannah", "product manager"),
    ("ivan", "Ivan", "data scientist"),
    ("julia", "Julia", "frontend developer"),
    ("kevin", "Kevin", "security engineer"),
    ("laura", "Laura", "technical writer"),
    ("mike", "Mike", "mobile developer"),
    ("nina", "Nina", "UX researcher"),
    ("oscar", "Oscar", "SRE"),
]

def _issue_tracker_state(seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    reference_date = _reference_datetime(seed).date()
    member_selected = _sample_entities(rng, _ISSUE_TRACKER_MEMBER_TEMPLATES, target_count=12, id_prefix="usr")
    member_id_map = {
        base: _seed_scoped_id("usr", seed, idx, width=3)
        for idx, (base, _name, _role) in enumerate(member_selected)
    }
    members = {
        member_id_map[base]: {
            "user_id": member_id_map[base],
            "name": name,
            "role": role,
        }
        for base, name, role in member_selected
    }
    member_ids = list(members.keys())
    states = ["open", "in_progress", "in_review", "resolved"]
    issues = {}
    issue_ids: list[str] = []
    for idx, (_iid, title, desc, priority, labels_str, state, assignee) in enumerate(_ISSUE_TEMPLATES[:20]):
        iid = _seed_scoped_id("iss", seed, idx, width=4)
        issue_ids.append(iid)
        sprint_id = _seed_scoped_id("spr", seed, 0, width=4) if idx < 6 else (_seed_scoped_id("spr", seed, 1, width=4) if idx < 12 else (_seed_scoped_id("spr", seed, 2, width=4) if idx < 18 else None))
        issues[iid] = {
            "issue_id": iid, "title": title, "description": desc,
            "priority": priority, "labels": [labels_str],
            "state": state,
            "assignee": member_id_map.get(assignee) if assignee else None,
            "watchers": rng.sample(member_ids, k=rng.randint(0, 2)),
            "sprint_id": sprint_id,
            "milestone": "v2.5" if idx < 6 else None,
            "comments": [],
            "created_at": (reference_date - _datetime.timedelta(days=idx % 8 + 1)).isoformat(),
        }
    sprint_id = _seed_scoped_id("spr", seed, 0, width=4)
    sprint_id_2 = _seed_scoped_id("spr", seed, 1, width=4)
    sprint_id_3 = _seed_scoped_id("spr", seed, 2, width=4)
    sprints = {
        sprint_id: {"sprint_id": sprint_id, "name": "Sprint 24",
                     "start_date": (reference_date - _datetime.timedelta(days=7)).isoformat(),
                     "end_date": (reference_date + _datetime.timedelta(days=7)).isoformat(),
                     "goal": "Bug fixes and feature work", "status": "active",
                     "issues": issue_ids[:6]},
        sprint_id_2: {"sprint_id": sprint_id_2, "name": "Sprint 25",
                       "start_date": (reference_date + _datetime.timedelta(days=7)).isoformat(),
                       "end_date": (reference_date + _datetime.timedelta(days=21)).isoformat(),
                       "goal": "Performance improvements and refactoring", "status": "active",
                       "issues": issue_ids[6:12]},
        sprint_id_3: {"sprint_id": sprint_id_3, "name": "Sprint 26",
                       "start_date": (reference_date + _datetime.timedelta(days=21)).isoformat(),
                       "end_date": (reference_date + _datetime.timedelta(days=35)).isoformat(),
                       "goal": "Security hardening and compliance", "status": "planning",
                       "issues": issue_ids[12:18]},
    }
    # Seed subtasks and time entries so list_subtasks/get_time_report have
    # material data without requiring create_subtask/time_track first.
    member_ids_list = list(members.keys())
    subtask_count = min(5, len(issue_ids))
    subtasks = {
        "sub_0001": {
            "subtask_id": "sub_0001", "issue_id": issue_ids[0],
            "title": "Write unit tests", "assignee": member_ids_list[0] if member_ids_list else None,
            "status": "open",
        },
        "sub_0002": {
            "subtask_id": "sub_0002", "issue_id": issue_ids[1],
            "title": "Update documentation", "assignee": member_ids_list[1] if len(member_ids_list) > 1 else None,
            "status": "in_progress",
        },
        "sub_0003": {
            "subtask_id": "sub_0003", "issue_id": issue_ids[2],
            "title": "Review PR changes", "assignee": member_ids_list[2] if len(member_ids_list) > 2 else None,
            "status": "resolved",
        },
        "sub_0004": {
            "subtask_id": "sub_0004", "issue_id": issue_ids[3],
            "title": "Design API schema", "assignee": member_ids_list[3] if len(member_ids_list) > 3 else None,
            "status": "open",
        },
        "sub_0005": {
            "subtask_id": "sub_0005", "issue_id": issue_ids[4],
            "title": "Performance profiling", "assignee": member_ids_list[4] if len(member_ids_list) > 4 else None,
            "status": "in_progress",
        },
    }
    time_entries = [
        {
            "entry_id": "time_0001", "issue_id": issue_ids[0],
            "hours": 3.5, "description": "Initial investigation",
            "date": (reference_date - _datetime.timedelta(days=2)).isoformat(),
            "user": "current_user",
        },
        {
            "entry_id": "time_0002", "issue_id": issue_ids[1],
            "hours": 1.5, "description": "Code review",
            "date": (reference_date - _datetime.timedelta(days=1)).isoformat(),
            "user": "current_user",
        },
        {
            "entry_id": "time_0003", "issue_id": issue_ids[2],
            "hours": 5.0, "description": "Bug fix implementation",
            "date": (reference_date - _datetime.timedelta(days=3)).isoformat(),
            "user": "current_user",
        },
        {
            "entry_id": "time_0004", "issue_id": issue_ids[0],
            "hours": 2.0, "description": "Writing test cases",
            "date": (reference_date - _datetime.timedelta(days=4)).isoformat(),
            "user": "current_user",
        },
        {
            "entry_id": "time_0005", "issue_id": issue_ids[3],
            "hours": 4.0, "description": "Architecture design review",
            "date": (reference_date - _datetime.timedelta(days=5)).isoformat(),
            "user": "current_user",
        },
    ]
    return {"issues": issues, "members": members, "sprints": sprints,
            "subtasks": subtasks, "time_entries": time_entries,
            "next_issue_num": len(issues) + 1, "next_sprint_num": 4,
            "next_subtask_num": 6, "next_time_entry_num": 6,
            "current_date": reference_date.isoformat()}
