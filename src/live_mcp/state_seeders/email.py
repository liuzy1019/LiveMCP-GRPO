"""Deterministic state builder for email."""

from __future__ import annotations

import datetime as _datetime
import random
from typing import Any

from src.live_mcp.state_seeders.common import (
    _reference_datetime,
    _seed_scoped_id,
)
def _email_state(seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    reference_date = _reference_datetime(seed).date()
    sender_pool = ["boss@example.com", "alice@example.com", "charlie@example.com",
                   "dana@example.com", "hr@example.com", "client@acme.com",
                   "support@vendor.io", "no-reply@saaplatform.com"]
    threads: dict[str, list[str]] = {}
    emails: dict[str, dict[str, Any]] = {}
    inbox: list[str] = []
    labels_pool = [[], ["work"], ["work", "urgent"], ["personal"], ["finance"], ["newsletter"]]
    subjects = [
        "Q2 Review", "Sprint Planning", "Budget approval needed",
        "Meeting follow-up", "New hire announcement", "Security update",
        "Invoice attached", "Project timeline", "Vacation request",
        "Quarterly report", "Server migration", "Client feedback",
    ]
    for idx in range(25):
        eid = _seed_scoped_id("eml", seed, idx, width=4)
        sender = rng.choice(sender_pool)
        subj = subjects[idx % len(subjects)]
        emails[eid] = {
            "email_id": eid, "to": "current_user@example.com",
            "cc": "", "sender": sender,
            "subject": subj,
            "body": f"This is regarding {subj.lower()}. Please review at your earliest convenience.",
            "labels": list(rng.choice(labels_pool)),
            "thread_id": _seed_scoped_id(
                "thd", seed, idx % len(subjects), width=3,
            ),
            "status": "received",
            "date": (reference_date - _datetime.timedelta(days=idx % 10)).isoformat(),
            "read": rng.random() < 0.5,
            "archived": False,
            "attachments": [],
        }
        inbox.append(eid)
        tid = emails[eid]["thread_id"]
        threads.setdefault(tid, []).append(eid)
    # Seed filters so list_filters chains are diverse without create_filter first.
    filters = {
        "flt_0001": {
            "filter_id": "flt_0001", "field": "sender",
            "pattern": "boss@example.com", "action": "label",
            "label": "work",
        },
        "flt_0002": {
            "filter_id": "flt_0002", "field": "subject",
            "pattern": "invoice", "action": "label",
            "label": "finance",
        },
        "flt_0003": {
            "filter_id": "flt_0003", "field": "sender",
            "pattern": "no-reply@saaplatform.com", "action": "archive",
        },
        "flt_0004": {
            "filter_id": "flt_0004", "field": "subject",
            "pattern": "newsletter", "action": "label",
            "label": "newsletter",
        },
    }
    # Seed drafts so create_draft chains are feasible without creating a draft first.
    draft_1 = _seed_scoped_id("eml", seed, 25, width=4)
    draft_2 = _seed_scoped_id("eml", seed, 26, width=4)
    drafts = {
        draft_1: {
            "email_id": draft_1,
            "to": "boss@example.com",
            "subject": "Q2 budget proposal",
            "body": "Hi, attached is the Q2 budget proposal. Let me know your thoughts.",
            "date": (reference_date - _datetime.timedelta(days=1)).isoformat(),
        },
        draft_2: {
            "email_id": draft_2,
            "to": "client@acme.com",
            "subject": "Contract renewal",
            "body": "Following up on our call last week. Please review the updated terms.",
            "date": reference_date.isoformat(),
        },
    }
    return {"emails": emails, "drafts": drafts, "threads": threads,
            "inbox_order": inbox, "next_email_num": 28, "next_thread_num": len(threads) + 1,
            "filters": filters, "current_date": reference_date.isoformat()}
