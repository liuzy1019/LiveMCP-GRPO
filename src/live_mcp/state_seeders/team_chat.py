"""Deterministic state builder for team_chat."""

from __future__ import annotations

import datetime as _datetime
import random
from typing import Any

from src.live_mcp.state_seeders.common import (
    _reference_datetime,
    _sample_entities,
    _seed_scoped_id,
)
_TEAM_CHAT_CHANNEL_TEMPLATES: list[tuple[str, str, str, list[str], bool, list[tuple[str, str]]]] = [
    ("ch_t01", "general", "General discussion",
     ["alice", "bob", "charlie"], False,
     [("Welcome everyone!", "alice"), ("Thanks Alice!", "bob")]),
    ("ch_t02", "engineering", "Engineering team",
     ["bob", "charlie", "dana"], False,
     [("Deploy scheduled for 6pm", "charlie")]),
    ("ch_t03", "design", "Design sync",
     ["dana", "alice"], False,
     [("Mockups ready for review", "dana"), ("Looking great!", "alice")]),
    ("ch_t04", "releases", "Release announcements",
     ["charlie", "bob", "dana", "alice"], False,
     [("v2.5 shipped!", "charlie")]),
    ("ch_t05", "random", "Water cooler chat",
     ["alice", "bob", "charlie", "dana"], False,
     [("Happy Friday! 🎉", "alice")]),
    ("ch_t06", "support", "Customer support triage",
     ["alice", "charlie"], False,
     [("New ticket #4521 assigned", "alice"), ("Taking a look", "charlie")]),
    ("ch_t07", "backend", "Backend services",
     ["bob", "charlie"], False,
     [("API response time spiking", "bob"), ("Checking the load balancer", "charlie")]),
    ("ch_t08", "frontend", "Frontend & UI",
     ["dana", "alice"], False,
     [("New component library merged", "dana")]),
    ("ch_t09", "hr-announcements", "HR updates",
     ["alice"], False,
     [("Benefits enrollment opens Monday", "alice")]),
    ("ch_t10", "devops", "Infrastructure & DevOps",
     ["bob", "charlie", "dana"], False,
     [("CI pipeline upgraded to v3", "bob"), ("Finally!", "dana")]),
    ("ch_t11", "sales-updates", "Sales team updates",
     ["alice", "bob"], False,
     [("Closed the Acme deal!", "alice"), ("Nice work!", "bob")]),
    ("ch_t12", "oncall", "On-call incident response",
     ["charlie", "bob", "dana", "alice"], False,
     [("PagerDuty alert: payment service down", "charlie"), ("I'm on it", "bob")]),
    ("ch_t13", "docs", "Documentation & knowledge base",
     ["dana", "alice"], False,
     [("API docs updated for v2.5", "dana")]),
    ("ch_t14", "offtopic", "Off-topic & memes",
     ["alice", "bob", "charlie", "dana"], False,
     [("Look at this cat 🐱", "alice"), ("Made my day", "charlie")]),
    ("ch_t15", "product-feedback", "Product feedback",
     ["dana", "bob"], False,
     [("User survey results are in", "dana"), ("Let's review in standup", "bob")]),
    ("ch_t16", "archived-2024", "Archived Q4 2024 planning",
     ["alice", "bob"], True, []),
    ("ch_t17", "ml-research", "ML research & experiments",
     ["charlie", "alice"], False,
     [("New embedding model benchmark results", "charlie"), ("Share the notebook?", "alice")]),
    ("ch_t18", "social", "Social committee",
     ["dana", "alice", "bob"], False,
     [("Team outing next Thursday!", "dana"), ("I'm in!", "alice"), ("Count me in too", "bob")]),
]

def _team_chat_state(seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    reference_date = _reference_datetime(seed).date()
    selected = _sample_entities(rng, _TEAM_CHAT_CHANNEL_TEMPLATES, target_count=15, id_prefix="ch")
    channels = {}
    msg_num = 1
    reactions_pool = [["wave"], ["rocket"], ["+1"], ["fire"], []]
    all_members: set[str] = set()
    for idx_ch, (_cid, name, desc, members, archived, msgs) in enumerate(selected):
        cid = _seed_scoped_id("ch", seed, idx_ch, width=3)
        messages = []
        for content, author in msgs:
            mid = _seed_scoped_id("msg", seed, msg_num - 1, width=4)
            messages.append({
                "message_id": mid, "channel_id": cid,
                "content": content, "author": author, "thread_id": None,
                "reactions": list(rng.choice(reactions_pool)),
                "timestamp": (
                    f"{(reference_date - _datetime.timedelta(days=msg_num % 5)).isoformat()}"
                    f"T{9 + msg_num % 8:02d}:00:00"
                ),
            })
            msg_num += 1
        all_members.update(members)
        channels[cid] = {
            "channel_id": cid, "name": name, "description": desc,
            "members": members, "archived": archived, "messages": messages,
        }

    # Enrich: channels with few seed messages get extra follow-up exchanges so
    # the Teacher can build followup chains (search, react, thread, etc.).
    active_channels = [c for c in channels.values() if not c.get("archived")]
    enrichment_messages = [
        ("Any updates on this?", "bob"),
        ("Can someone take a look?", "dana"),
        ("+1, seeing the same issue", "charlie"),
        ("Fixed in the latest PR, closing this out.", "alice"),
        ("Good catch, thanks for flagging!", "bob"),
        ("Let's discuss in the standup tomorrow.", "dana"),
        ("I'll pick this up next sprint.", "charlie"),
        ("Updated the docs to reflect the new behavior.", "alice"),
        ("Please review when you get a chance.", "bob"),
        ("Deployed to staging, please test.", "charlie"),
        ("Status update: blocked on external dependency.", "dana"),
        ("Rolled back due to regression.", "alice"),
    ]
    enriched: set[str] = set()
    for ch in active_channels:
        if len(ch["messages"]) >= 4:
            continue
        available = [
            m for m in enrichment_messages
            if m[1] in ch["members"] and m[0] not in enriched
        ]
        if not available:
            continue
        extra = rng.choice(available)
        mid = _seed_scoped_id("msg", seed, msg_num - 1, width=4)
        ch["messages"].append({
            "message_id": mid, "channel_id": ch["channel_id"],
            "content": extra[0], "author": extra[1], "thread_id": None,
            "reactions": list(rng.choice(reactions_pool)),
            "timestamp": (
                f"{(reference_date - _datetime.timedelta(days=msg_num % 5)).isoformat()}"
                f"T{9 + msg_num % 8:02d}:00:00"
            ),
        })
        enriched.add(extra[0])
        msg_num += 1

    # Seed threads: pick messages from active channels with at least 1
    # prior message (so the thread root has conversation context), create a
    # thread from each, then add 1-3 reply messages in the thread.
    threads: dict[str, Any] = {}
    thread_seed_msgs = [
        ("Let me elaborate on that point.", "alice"),
        ("Here's a more detailed analysis.", "bob"),
        ("I see what you mean — here's my take.", "dana"),
        ("Could you clarify what you meant by that?", "charlie"),
        ("Good question. The short answer is yes.", "alice"),
        ("This looks correct based on the latest data.", "bob"),
        ("I have a different perspective on this.", "dana"),
    ]
    thread_num = 1
    for ch in active_channels:
        if len(threads) >= 6:
            break
        # Thread root must be a message with index >= 1 so there's prior context.
        candidates = [
            (i, m) for i, m in enumerate(ch["messages"])
            if i >= 1 and not m.get("thread_id")
        ]
        if not candidates:
            continue
        root_idx, root_msg = rng.choice(candidates)
        tid = f"thd_s{seed}_{thread_num:04d}"
        thread_num += 1
        root_msg["thread_id"] = tid
        replies = []
        for reply_idx in range(rng.randint(1, 3)):
            reply = rng.choice(thread_seed_msgs)
            if reply[1] in ch["members"]:
                rmid = _seed_scoped_id("msg", seed, msg_num - 1, width=4)
                replies.append({
                    "message_id": rmid, "channel_id": ch["channel_id"],
                    "content": reply[0], "author": reply[1],
                    "thread_id": tid,
                    "reactions": list(rng.choice(reactions_pool)),
                    "timestamp": (
                        f"{(reference_date - _datetime.timedelta(days=msg_num % 5)).isoformat()}"
                        f"T{9 + msg_num % 8:02d}:00:00"
                    ),
                })
                msg_num += 1
        threads[tid] = {
            "thread_id": tid,
            "root_message_id": root_msg["message_id"],
            "channel_id": ch["channel_id"],
            "messages": replies,
        }

    # Seed DMs between members who appear together in channels.
    dm_candidates: list[tuple[str, str]] = []
    for ch in active_channels:
        for i, m1 in enumerate(ch["members"]):
            for m2 in ch["members"][i + 1:]:
                pair = tuple(sorted([m1, m2]))
                if pair not in dm_candidates:
                    dm_candidates.append(pair)  # type: ignore[arg-type]
    rng.shuffle(dm_candidates)
    dm_pool = [
        ("alice", "bob", "Hey, got a minute to chat about the upcoming release?"),
        ("bob", "charlie", "Can you review my PR when you get a chance?"),
        ("dana", "alice", "The design specs are in the shared folder."),
        ("charlie", "bob", "Quick question about the deployment schedule."),
        ("alice", "dana", "I updated the roadmap — mind taking a look?"),
    ]
    dms: list[dict[str, Any]] = []
    dm_seeded: set[tuple[str, str]] = set()
    for sender, receiver, content in rng.sample(dm_pool, min(2, len(dm_pool))):
        for pair in dm_candidates:
            if sender in pair and receiver in pair and pair not in dm_seeded:
                mid = _seed_scoped_id("msg", seed, msg_num - 1, width=4)
                dms.append({
                    "message_id": mid, "sender": sender,
                    "recipient": receiver, "content": content,
                    "timestamp": (
                        f"{(reference_date - _datetime.timedelta(days=msg_num % 3 + 1)).isoformat()}"
                        f"T{10 + msg_num % 6:02d}:00:00"
                    ),
                })
                dm_seeded.add(pair)
                msg_num += 1
                break

    next_thread_num = thread_num
    return {"channels": channels, "threads": threads, "dms": dms,
            "next_msg_num": msg_num, "next_thread_num": next_thread_num,
            "next_ch_num": len(channels) + 1,
            "current_date": reference_date.isoformat()}
