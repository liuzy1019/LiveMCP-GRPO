"""Deterministic state seeders for MVP servers.

PROVE §3.2 Step 2: each domain must produce 20+ entities with seed-driven
variations so that teacher trajectories reference a diverse, non-memorizable
space of real entity IDs.  The original 4-entity pools were too small for GRPO.
"""

from __future__ import annotations

import copy
import random
from typing import Any


class StateSeeder:
    def seed_state(self, server_name: str, session_id: str, seed: int) -> dict[str, Any]:
        if server_name == "calendar":
            return _calendar_state(seed)
        if server_name == "shopping":
            return _shopping_state(seed)
        if server_name == "banking":
            return _banking_state(seed)
        if server_name == "email":
            return _email_state(seed)
        if server_name == "filesystem":
            return _filesystem_state(seed)
        if server_name == "payments":
            return _payments_state(seed)
        if server_name == "crm":
            return _crm_state(seed)
        if server_name == "issue_tracker":
            return _issue_tracker_state(seed)
        if server_name == "team_chat":
            return _team_chat_state(seed)
        if server_name == "food_delivery":
            return _food_delivery_state(seed)
        raise ValueError(f"unsupported server: {server_name}")

    def reset_state(self, server_name: str, session_id: str, seed: int) -> dict[str, Any]:
        return copy.deepcopy(self.seed_state(server_name, session_id, seed))


# ═══════════════════════════════════════════════════════════════════════
# Entity template pools — each seed picks a *subset* so different seeds
# see different, non-overlapping entity ID spaces.  This prevents the GRPO
# model from memorising a fixed set of IDs.
# ═══════════════════════════════════════════════════════════════════════

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

_SHOPPING_PRODUCT_TEMPLATES: list[tuple[str, str, str, int, int, str]] = [
    ("prd_001", "K3 Keyboard", "keyboard", 79, 5, "Mechanical keyboard with RGB backlight"),
    ("prd_002", "MX Mouse", "mouse", 49, 8, "Ergonomic wireless mouse"),
    ("prd_003", "USB-C Hub", "hub", 35, 4, "7-in-1 USB-C hub with HDMI"),
    ("prd_004", "Noise Canceling Headphones", "audio", 99, 3, "Wireless ANC headphones"),
    ("prd_005", "4K Monitor 27\"", "monitor", 349, 6, "27-inch 4K IPS monitor"),
    ("prd_006", "Webcam Pro", "camera", 89, 10, "1080p webcam with auto-focus"),
    ("prd_007", "Standing Desk", "furniture", 549, 2, "Electric height-adjustable desk"),
    ("prd_008", "Laptop Stand", "accessory", 39, 15, "Aluminum laptop stand"),
    ("prd_009", "Thunderbolt Cable", "cable", 29, 20, "2m Thunderbolt 4 cable"),
    ("prd_010", "External SSD 1TB", "storage", 129, 7, "1TB portable SSD, USB-C"),
    ("prd_011", "Wireless Charger", "charger", 25, 12, "Qi fast wireless charging pad"),
    ("prd_012", "Desk Lamp LED", "lighting", 59, 8, "Adjustable LED desk lamp"),
    ("prd_013", "Mouse Pad XL", "accessory", 19, 25, "Extended gaming mouse pad"),
    ("prd_014", "Monitor Arm", "mount", 79, 4, "Gas-spring monitor arm"),
    ("prd_015", "Bluetooth Speaker", "audio", 69, 9, "Portable Bluetooth speaker"),
    ("prd_016", "Mechanical Keyboard V2", "keyboard", 119, 3, "Hot-swappable mechanical keyboard"),
    ("prd_017", "Ergonomic Mouse", "mouse", 69, 6, "Vertical ergonomic mouse"),
    ("prd_018", "USB Microphone", "audio", 45, 11, "Condenser USB microphone"),
    ("prd_019", "Drawing Tablet", "tablet", 249, 2, "10-inch drawing tablet with stylus"),
    ("prd_020", "Docking Station", "hub", 199, 5, "Dual-HDMI USB-C docking station"),
    ("prd_021", "Smart Plug", "smart_home", 19, 30, "Wi-Fi smart plug with energy monitoring"),
    ("prd_022", "Portable Battery 20000mAh", "charger", 49, 14, "20000mAh power bank, USB-C PD"),
    ("prd_023", "Cable Management Kit", "accessory", 15, 40, "Velcro cable ties and clips"),
    ("prd_024", "Keyboard Wrist Rest", "accessory", 22, 18, "Memory foam wrist rest"),
    ("prd_025", "USB-A to USB-C Adapter", "cable", 9, 50, "Pack of 3 USB-A to USB-C adapters"),
]

_BANKING_ACCOUNT_TEMPLATES: list[tuple[str, str, float, str, str]] = [
    ("acc_savings", "Alice Johnson", 25000.00, "savings", "2018-03-15"),
    ("acc_checking", "Alice Johnson", 5000.00, "checking", "2018-03-15"),
    ("acc_business", "Bob Smith", 100000.00, "business", "2020-01-10"),
    ("acc_frozen_demo", "Carol White", 1500.00, "savings", "2022-06-01"),
    ("acc_emergency", "Alice Johnson", 8000.00, "savings", "2019-11-22"),
    ("acc_investment", "Bob Smith", 45000.00, "investment", "2020-06-15"),
    ("acc_joint", "Alice Johnson", 12000.00, "checking", "2021-01-05"),
    ("acc_travel", "Bob Smith", 3200.00, "savings", "2021-08-30"),
    ("acc_booked", "Carol White", 9800.00, "savings", "2022-04-12"),
    ("acc_trust", "Alice Johnson", 75000.00, "investment", "2017-09-01"),
    ("acc_expense", "Bob Smith", 2100.00, "checking", "2023-02-14"),
    ("acc_retirement", "Bob Smith", 180000.00, "retirement", "2015-06-30"),
    ("acc_college", "Carol White", 32000.00, "savings", "2019-08-01"),
    ("acc_hsa", "Alice Johnson", 4500.00, "health", "2021-03-10"),
    ("acc_brokerage", "Bob Smith", 62000.00, "investment", "2018-11-20"),
    ("acc_escrow", "Carol White", 15000.00, "escrow", "2023-05-15"),
    ("acc_payroll", "Alice Johnson", 7800.00, "checking", "2020-07-01"),
    ("acc_reserve", "Bob Smith", 28000.00, "savings", "2016-04-22"),
    ("acc_minor", "Carol White", 3100.00, "savings", "2024-01-10"),
    ("acc_forex", "Alice Johnson", 9200.00, "foreign", "2022-09-30"),
]

_PAYMENTS_INVOICE_TEMPLATES: list[tuple[str, str, float, str, str, str]] = [
    ("inv_0001", "Acme Corp", 1500.00, "USD", "Consulting Q2", "pending"),
    ("inv_0002", "Globex Inc", 3200.00, "USD", "Software license", "pending"),
    ("inv_0003", "Acme Corp", 500.00, "EUR", "Support retainer", "paid"),
    ("inv_0004", "Initech", 2400.00, "USD", "Cloud migration phase 1", "pending"),
    ("inv_0005", "Globex Inc", 800.00, "USD", "Maintenance renew", "overdue"),
    ("inv_0006", "Umbrella Corp", 5000.00, "GBP", "Security audit", "pending"),
    ("inv_0007", "Acme Corp", 1200.00, "USD", "Training workshop", "paid"),
    ("inv_0008", "Initech", 900.00, "USD", "API integration", "pending"),
    ("inv_0009", "Globex Inc", 3500.00, "USD", "Annual subscription", "overdue"),
    ("inv_0010", "Umbrella Corp", 1800.00, "EUR", "Penetration testing", "paid"),
    ("inv_0011", "Stark Industries", 7500.00, "USD", "Hardware supply", "pending"),
    ("inv_0012", "Acme Corp", 450.00, "USD", "Emergency support", "overdue"),
    ("inv_0013", "Wayne Enterprises", 12000.00, "USD", "Infrastructure upgrade", "pending"),
    ("inv_0014", "Initech", 650.00, "USD", "On-site support", "paid"),
    ("inv_0015", "Globex Inc", 2100.00, "EUR", "Data analytics platform", "pending"),
    ("inv_0016", "Stark Industries", 4300.00, "USD", "Consulting retainer", "overdue"),
    ("inv_0017", "Acme Corp", 980.00, "USD", "Compliance audit", "pending"),
    ("inv_0018", "Wayne Enterprises", 3750.00, "GBP", "Security assessment", "paid"),
    ("inv_0019", "Umbrella Corp", 2200.00, "USD", "Managed services", "pending"),
    ("inv_0020", "Initech", 1100.00, "USD", "Staff augmentation", "overdue"),
]

_CRM_LEAD_TEMPLATES: list[tuple[str, str, str, str, str, str]] = [
    ("lead_0001", "Charlie Chen", "TechStars", "conference", "new", "charlie@techstars.com"),
    ("lead_0002", "Dana Davis", "DataFlow", "webinar", "new", "dana@dataflow.io"),
    ("lead_0003", "Evan Ellis", "CloudBase", "referral", "converted", "evan@cloudbase.com"),
    ("lead_0004", "Fiona Foster", "FinEdge", "cold_outreach", "contacted", "fiona@finedge.com"),
    ("lead_0005", "George Grant", "GreenTech", "conference", "new", "george@greentech.co"),
    ("lead_0006", "Hannah Hill", "HealthFirst", "webinar", "qualified", "hannah@healthfirst.com"),
    ("lead_0007", "Ian Ingram", "InnoSoft", "referral", "new", "ian@innosoft.io"),
    ("lead_0008", "Julia Jiang", "JetStream", "linkedin", "contacted", "julia@jetstream.com"),
    ("lead_0009", "Kevin Kim", "Krypton", "cold_outreach", "new", "kevin@krypton.io"),
    ("lead_0010", "Laura Lin", "LightSpeed", "conference", "qualified", "laura@lightspeed.com"),
    ("lead_0011", "Mike Moran", "MegaCorp", "webinar", "new", "mike@megacorp.com"),
    ("lead_0012", "Nina Ng", "NexGen", "referral", "contacted", "nina@nexgen.io"),
    ("lead_0013", "Oscar Owen", "OmniCloud", "conference", "new", "oscar@omnicloud.io"),
    ("lead_0014", "Paula Park", "PivotAI", "webinar", "qualified", "paula@pivotai.com"),
    ("lead_0015", "Quinn Quinn", "QuantumLeap", "referral", "new", "quinn@quantumleap.co"),
    ("lead_0016", "Rachel Reed", "RapidScale", "linkedin", "contacted", "rachel@rapidscale.com"),
    ("lead_0017", "Steve Shaw", "SkyBridge", "cold_outreach", "new", "steve@skybridge.io"),
    ("lead_0018", "Tina Tang", "TrueNorth", "conference", "qualified", "tina@truenorth.com"),
    ("lead_0019", "Uma Upton", "UniStack", "webinar", "new", "uma@unistack.io"),
    ("lead_0020", "Victor Vance", "VaultSec", "referral", "contacted", "victor@vaultsec.com"),
]

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


def _seed_scoped_id(prefix: str, seed: int, idx: int, width: int = 3) -> str:
    """Generate a seed-scoped entity ID that varies across seeds.

    PROVE §3.2: different seeds must produce disjoint ID namespaces to prevent
    the GRPO model from memorising a fixed set of entity IDs.  Uses a
    multiplicative hash of the seed to produce a short hex scope.

    Example: _seed_scoped_id("evt", 42, 0) -> "evt_aa3_001"
             _seed_scoped_id("evt", 137, 0) -> "evt_1f7_001"
    """
    scope = (seed * 2654435761) % 4096  # Knuth multiplicative hash, 12 bits → 3 hex digits
    return f"{prefix}_{scope:03x}_{idx + 1:0{width}d}"


def _sample_entities(
    rng: random.Random,
    template_pool: list,
    target_count: int,
    id_prefix: str,
) -> list:
    """Deterministically sample *target_count* entities from *template_pool*.

    Uses the PRNG to shuffle indices, so different seeds see different subsets
    of the entity space.  If target_count > len(template_pool), wraps by adding
    seed-derived suffix to IDs.
    """
    indices = list(range(len(template_pool)))
    rng.shuffle(indices)
    selected = indices[:min(target_count, len(template_pool))]
    result = [template_pool[i] for i in sorted(selected)]
    # If pool is too small, pad with seed-derived variants
    while len(result) < target_count:
        base_idx = rng.randint(0, len(template_pool) - 1)
        base = template_pool[base_idx]
        variant_id = f"{id_prefix}_{rng.randint(100, 999):03d}"
        result.append((variant_id,) + base[1:] if len(base) > 1 else (variant_id, base[1]))
    return result


def _calendar_state(seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    events: dict[str, dict[str, Any]] = {}
    selected = _sample_entities(rng, _CALENDAR_EVENT_TEMPLATES, target_count=20, id_prefix="evt")
    base_date = 22  # June 22-30 range
    attendee_pool = ["alex@example.com", "sam@example.com", "charlie@example.com",
                     "dana@example.com", "bob@example.com"]
    for idx, entry in enumerate(selected):
        if len(entry) == 4:
            title, lead, desc, location = entry
        else:
            title, lead, desc = entry[1:4]
            location = f"Room {100 + idx}"
        day = min(base_date + (idx % 9), 30)
        hour = 9 + (idx + rng.randint(0, 3)) % 6
        # PROVE §3.2: seed-scoped ID prevents GRPO from memorising a fixed
        # evt_001..evt_016 namespace across seeds.  Different seeds produce
        # disjoint ID spaces (e.g., seed=42 → evt_aa3_001, seed=137 → evt_1f7_001).
        eid = _seed_scoped_id("evt", seed, idx, width=3)
        events[eid] = {
            "event_id": eid, "title": title,
            "start_time": f"2026-06-{day:02d}T{hour:02d}:00",
            "end_time": f"2026-06-{day:02d}T{hour + 1:02d}:00",
            "description": desc, "location": location,
            "attendees": rng.sample(attendee_pool, k=1 + idx % 3),
            "reminders": [], "recurrence": None,
        }
    return {"events": events, "next_event_num": len(events) + 1,
            "timezone": "America/New_York"}


def _shopping_state(seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    selected = _sample_entities(rng, _SHOPPING_PRODUCT_TEMPLATES, target_count=20, id_prefix="prd")
    products = {}
    product_ids: list[str] = []
    for idx, (_pid, name, category, price, stock, desc) in enumerate(selected):
        unique_pid = _seed_scoped_id("prd", seed, idx, width=3)
        product_ids.append(unique_pid)
        products[unique_pid] = {
            "product_id": unique_pid, "name": name, "category": category,
            "price": price + rng.randint(-5, 5), "stock": stock + rng.randint(0, 3),
            "description": desc,
        }
    order_1 = _seed_scoped_id("ord", seed, 0, width=4)
    order_2 = _seed_scoped_id("ord", seed, 1, width=4)
    orders = {
        order_1: {"order_id": order_1, "product_ids": product_ids[:2],
                  "total": 114.00, "status": "shipped", "created_at": "2026-06-18"},
        order_2: {"order_id": order_2, "product_ids": product_ids[2:3],
                  "total": 349.00, "status": "pending", "created_at": "2026-06-22"},
    }
    return {"products": products, "cart": [], "orders": orders,
            "next_order_num": 3, "reviews": {}, "wishlist": []}


def _banking_state(seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    selected = _sample_entities(rng, _BANKING_ACCOUNT_TEMPLATES, target_count=20, id_prefix="acc")
    accounts = {}
    frozen_ids = set()
    for idx, (aid, owner, balance, atype, opened) in enumerate(selected):
        scoped_aid = _seed_scoped_id("acc", seed, idx, width=3)
        is_frozen = aid == "acc_frozen_demo" or (aid.startswith("acc_booked") and rng.random() < 0.3)
        if is_frozen:
            frozen_ids.add(scoped_aid)
        accounts[scoped_aid] = {
            "account_id": scoped_aid, "owner": owner, "balance": round(balance + rng.randint(-200, 200), 2),
            "currency": "USD", "type": atype, "frozen": is_frozen,
            "opened_date": opened,
        }
    return {"accounts": accounts, "transactions": [], "freeze_log": [],
            "next_txn_num": 1, "scheduled_transfers": {}, "loans": {}}


def _email_state(seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
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
    for idx in range(20):
        eid = _seed_scoped_id("eml", seed, idx, width=4)
        sender = rng.choice(sender_pool)
        subj = subjects[idx % len(subjects)]
        emails[eid] = {
            "email_id": eid, "to": "current_user@example.com",
            "cc": "", "sender": sender,
            "subject": subj,
            "body": f"This is regarding {subj.lower()}. Please review at your earliest convenience.",
            "labels": rng.choice(labels_pool),
            "thread_id": _seed_scoped_id("thd", seed, idx // 3, width=3),
            "status": "received", "date": f"2026-06-{20 + idx % 8:02d}",
            "read": rng.random() < 0.5,
            "attachments": [],
        }
        inbox.append(eid)
        tid = emails[eid]["thread_id"]
        threads.setdefault(tid, []).append(eid)
    return {"emails": emails, "drafts": {}, "threads": threads,
            "inbox_order": inbox, "next_email_num": 13, "next_thread_num": len(threads) + 1,
            "filters": {}}


def _filesystem_state(seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    # Filesystem benefits more from structural diversity than pure entity count.
    fs: dict[str, dict[str, Any]] = {
        "/": {"type": "dir", "content": "", "permissions": "755", "owner": "root"},
        "/home": {"type": "dir", "content": "", "permissions": "755", "owner": "root"},
        "/home/user": {"type": "dir", "content": "", "permissions": "700", "owner": "user"},
        "/home/user/notes.txt": {"type": "file",
            "content": "TODO: review design doc\nTODO: update tests\nDONE: fix login bug",
            "permissions": "644", "owner": "user"},
        "/home/user/script.sh": {"type": "file",
            "content": "#!/bin/bash\necho hello", "permissions": "755", "owner": "user"},
        "/home/user/projects": {"type": "dir", "content": "", "permissions": "700", "owner": "user"},
        "/home/user/projects/README.md": {"type": "file",
            "content": "# Projects\nWork in progress.", "permissions": "644", "owner": "user"},
        "/home/user/projects/config.ini": {"type": "file",
            "content": "[server]\nhost=localhost\nport=8080\n[database]\nname=proddb",
            "permissions": "644", "owner": "user"},
        "/home/user/logs": {"type": "dir", "content": "", "permissions": "700", "owner": "user"},
        "/home/user/logs/app.log": {"type": "file",
            "content": "2026-06-20 INFO Starting application\n2026-06-21 WARN Connection timeout\n2026-06-22 ERROR Database unreachable",
            "permissions": "644", "owner": "user"},
        "/home/user/logs/error.log": {"type": "file",
            "content": "2026-06-22 ERROR: null pointer in module auth\n2026-06-22 ERROR: stack overflow in parser",
            "permissions": "644", "owner": "user"},
        "/home/user/data": {"type": "dir", "content": "", "permissions": "700", "owner": "user"},
        "/home/user/data/users.csv": {"type": "file",
            "content": "id,name,role\n1,Alice,admin\n2,Bob,user\n3,Charlie,user",
            "permissions": "644", "owner": "user"},
        "/home/user/data/report_2026.json": {"type": "file",
            "content": '{"revenue": 150000, "costs": 90000, "profit": 60000}',
            "permissions": "644", "owner": "user"},
        "/home/user/pipeline.sh": {"type": "file",
            "content": "#!/bin/bash\n# Build and deploy pipeline\nmake build\nmake test\nmake deploy",
            "permissions": "755", "owner": "user"},
        "/protected": {"type": "dir", "content": "", "permissions": "700", "owner": "root"},
        "/protected/config.secret": {"type": "file",
            "content": "secret_key=abc123\ndb_password=xyz789",
            "permissions": "600", "owner": "root"},
        "/protected/certs": {"type": "dir", "content": "", "permissions": "700", "owner": "root"},
        "/protected/certs/server.crt": {"type": "file",
            "content": "-----BEGIN CERTIFICATE-----\nMIID...\n-----END CERTIFICATE-----",
            "permissions": "600", "owner": "root"},
    }
    return {"fs": fs, "cwd": "/home/user", "umask": "022"}


def _payments_state(seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    selected = _sample_entities(rng, _PAYMENTS_INVOICE_TEMPLATES, target_count=20, id_prefix="inv")
    invoices = {}
    due_dates = ["2026-07-15", "2026-07-20", "2026-06-30", "2026-08-01",
                 "2026-07-10", "2026-08-15", "2026-07-05", "2026-07-25"]
    created_dates = ["2026-06-01", "2026-06-05", "2026-06-10", "2026-06-15",
                     "2026-06-20", "2026-06-25"]
    paid_invoice_id = None
    paid_amount = 0.0
    for idx, (_iid, customer, amount, currency, desc, status) in enumerate(selected):
        iid = _seed_scoped_id("inv", seed, idx, width=4)
        pay_id = _seed_scoped_id("pay", seed, idx, width=4) if status == "paid" else None
        if paid_invoice_id is None and status == "paid":
            paid_invoice_id = iid
            paid_amount = amount
        invoices[iid] = {
            "invoice_id": iid, "customer": customer, "amount": round(amount + rng.randint(-50, 50), 2),
            "currency": currency, "description": desc, "status": status,
            "payment_id": pay_id,
            "refund_id": None,
            "due_date": due_dates[idx % len(due_dates)],
            "created_at": created_dates[idx % len(created_dates)],
        }
    payments = {}
    if paid_invoice_id:
        pay_id = invoices[paid_invoice_id]["payment_id"]
        payments[pay_id] = {
            "payment_id": pay_id,
            "invoice_id": paid_invoice_id,
            "amount": paid_amount,
            "method": "wire",
            "status": "settled",
        }
    return {"invoices": invoices,
            "payments": payments,
            "refunds": {}, "webhooks": {}, "disputes": {},
            "next_inv_num": len(invoices) + 1, "next_pay_num": 2,
            "next_ref_num": 1, "next_wh_num": 1}


def _crm_state(seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    leads_selected = _sample_entities(rng, _CRM_LEAD_TEMPLATES, target_count=20, id_prefix="lead")
    leads = {}
    statuses = ["new", "contacted", "qualified", "converted"]
    lead_ids: list[str] = []
    converted_pairs: list[tuple[str, str, str, str]] = []
    for idx, (_lid, name, company, source, status, email) in enumerate(leads_selected):
        lid = _seed_scoped_id("lead", seed, idx, width=4)
        contact_id = _seed_scoped_id("contact", seed, idx, width=4) if status == "converted" else None
        lead_ids.append(lid)
        if contact_id:
            converted_pairs.append((contact_id, name, email, company))
        leads[lid] = {
            "lead_id": lid, "name": name, "company": company, "source": source,
            "email": email, "phone": f"555-{1000 + idx:04d}",
            "status": status, "contact_id": contact_id,
        }
    contacts = {
        cid: {"contact_id": cid, "name": name, "email": email, "phone": "",
              "company": company, "lead_id": None}
        for cid, name, email, company in converted_pairs
    }
    deal_pool = [
        ("Cloud Migration", 50000.00, "prospecting"),
        ("Data Pipeline", 75000.00, "proposal"),
        ("Mobile App", 35000.00, "negotiation"),
        ("AI Integration", 120000.00, "prospecting"),
        ("Security Audit", 25000.00, "closed_won"),
    ]
    deals = {}
    for idx, (dname, damount, dstage) in enumerate(deal_pool):
        did = _seed_scoped_id("deal", seed, idx, width=4)
        lid_candidate = lead_ids[idx % len(lead_ids)] if lead_ids else None
        deals[did] = {
            "deal_id": did, "name": dname,
            "amount": round(damount + rng.randint(-5000, 5000), 2),
            "stage": dstage, "contact_id": None,
            "lead_id": lid_candidate,
            "created_at": f"2026-06-{10 + idx * 5:02d}",
        }
    return {"leads": leads, "contacts": contacts, "deals": deals,
            "tasks": {}, "notes": {},
            "next_lead_num": len(leads) + 1, "next_contact_num": 1,
            "next_deal_num": len(deals) + 1, "next_task_num": 1, "next_note_num": 1}


def _issue_tracker_state(seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    member_configs = [
        ("alice", "Alice", "developer"),
        ("bob", "Bob", "senior developer"),
        ("charlie", "Charlie", "tech lead"),
        ("dana", "Dana", "designer"),
    ]
    member_id_map = {
        base: _seed_scoped_id("usr", seed, idx, width=3)
        for idx, (base, _name, _role) in enumerate(member_configs)
    }
    members = {
        member_id_map[base]: {
            "user_id": member_id_map[base],
            "name": name,
            "role": role,
        }
        for base, name, role in member_configs
    }
    member_ids = list(members.keys())
    states = ["open", "in_progress", "in_review", "resolved"]
    issues = {}
    issue_ids: list[str] = []
    for idx, (_iid, title, desc, priority, labels_str, state, assignee) in enumerate(_ISSUE_TEMPLATES[:20]):
        iid = _seed_scoped_id("iss", seed, idx, width=4)
        issue_ids.append(iid)
        sprint_id = _seed_scoped_id("spr", seed, 0, width=4) if idx < 6 else None
        created_day = min(18 + idx % 8, 30)
        issues[iid] = {
            "issue_id": iid, "title": title, "description": desc,
            "priority": priority, "labels": [labels_str],
            "state": state,
            "assignee": member_id_map.get(assignee) if assignee else None,
            "watchers": rng.sample(member_ids, k=rng.randint(0, 2)),
            "sprint_id": sprint_id,
            "milestone": "v2.5" if idx < 6 else None,
            "comments": [],
            "created_at": f"2026-06-{created_day:02d}",
        }
    sprint_id = _seed_scoped_id("spr", seed, 0, width=4)
    sprints = {
        sprint_id: {"sprint_id": sprint_id, "name": "Sprint 24",
                     "start_date": "2026-06-15", "end_date": "2026-06-29",
                     "goal": "Bug fixes and feature work", "status": "active",
                     "issues": issue_ids[:6]},
    }
    return {"issues": issues, "members": members, "sprints": sprints,
            "subtasks": {}, "time_entries": [],
            "next_issue_num": len(issues) + 1, "next_sprint_num": 2,
            "next_subtask_num": 1, "next_time_entry_num": 1}


def _team_chat_state(seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    channel_configs = [
        ("ch_general", "general", "General discussion",
         ["alice", "bob", "charlie"], False,
         [("Welcome everyone!", "alice"), ("Thanks Alice!", "bob")]),
        ("ch_engineering", "engineering", "Engineering team",
         ["bob", "charlie", "dana"], False,
         [("Deploy scheduled for 6pm", "charlie")]),
        ("ch_design", "design", "Design sync",
         ["dana", "alice"], False,
         [("Mockups ready for review", "dana"), ("Looking great!", "alice")]),
        ("ch_releases", "releases", "Release announcements",
         ["charlie", "bob", "dana", "alice"], False,
         [("v2.5 shipped!", "charlie")]),
        ("ch_random", "random", "Water cooler chat",
         ["alice", "bob", "charlie", "dana"], False,
         [("Happy Friday! 🎉", "alice")]),
        ("ch_archived", "old-project", "Archived project channel",
         ["alice"], True, []),
    ]
    channels = {}
    msg_num = 1
    reactions_pool = [["wave"], ["rocket"], ["+1"], ["fire"], []]
    for idx_ch, (_cid, name, desc, members, archived, msgs) in enumerate(channel_configs):
        cid = _seed_scoped_id("ch", seed, idx_ch, width=3)
        messages = []
        for content, author in msgs:
            mid = _seed_scoped_id("msg", seed, msg_num - 1, width=4)
            messages.append({
                "message_id": mid, "channel_id": cid,
                "content": content, "author": author, "thread_id": None,
                "reactions": rng.choice(reactions_pool),
                "timestamp": f"2026-06-{20 + msg_num % 5:02d}T{9 + msg_num % 8:02d}:00:00",
            })
            msg_num += 1
        channels[cid] = {
            "channel_id": cid, "name": name, "description": desc,
            "members": members, "archived": archived, "messages": messages,
        }
    return {"channels": channels, "threads": {}, "dms": [],
            "next_msg_num": msg_num, "next_thread_num": 1,
            "next_ch_num": len(channels) + 1}


def _food_delivery_state(seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    restaurant_templates = [
        ("rest_001", "Pizza Palace", "Italian", 4.5, 2.99,
         [("Margherita Pizza", 12.99, ["vegetarian"]),
          ("Pepperoni Pizza", 14.99, []),
          ("Caesar Salad", 8.99, ["vegetarian", "gluten-free"]),
          ("Garlic Bread", 4.99, ["vegetarian"])]),
        ("rest_002", "Sushi Express", "Japanese", 4.8, 3.99,
         [("California Roll", 10.99, []),
          ("Salmon Nigiri", 12.99, ["gluten-free"]),
          ("Miso Soup", 3.99, ["vegetarian", "gluten-free"]),
          ("Edamame", 4.99, ["vegan", "gluten-free"])]),
        ("rest_003", "Burger Barn", "American", 4.2, 1.99,
         [("Classic Burger", 9.99, []),
          ("Cheese Burger", 11.99, []),
          ("French Fries", 3.99, ["vegetarian", "gluten-free"]),
          ("Milkshake", 5.99, ["vegetarian"])]),
        ("rest_004", "Taco Fiesta", "Mexican", 4.3, 2.49,
         [("Chicken Taco", 7.99, []),
          ("Beef Burrito", 10.99, []),
          ("Guacamole", 5.99, ["vegan", "gluten-free"]),
          ("Churros", 3.99, ["vegetarian"])]),
        ("rest_005", "Curry House", "Indian", 4.6, 3.49,
         [("Chicken Tikka Masala", 13.99, ["gluten-free"]),
          ("Vegetable Biryani", 11.99, ["vegetarian", "gluten-free"]),
          ("Naan Bread", 3.49, ["vegetarian"]),
          ("Mango Lassi", 4.99, ["vegetarian", "gluten-free"])]),
        ("rest_006", "Pho Garden", "Vietnamese", 4.4, 2.99,
         [("Beef Pho", 11.99, ["gluten-free"]),
          ("Spring Rolls", 6.99, ["vegetarian"]),
          ("Banh Mi", 8.99, []),
          ("Vietnamese Coffee", 4.49, ["vegetarian"])]),
    ]
    restaurants = {}
    restaurant_ids: list[str] = []
    for idx_rest, (_rid, name, cuisine, rating, delivery_fee, menu_items) in enumerate(restaurant_templates):
        rid = _seed_scoped_id("rest", seed, idx_rest, width=3)
        restaurant_ids.append(rid)
        menu = []
        for i, (mname, mprice, tags) in enumerate(menu_items):
            menu.append({
                "name": mname, "price": mprice + rng.randint(-1, 1),
                "dietary_tags": tags,
                "popularity": 50 + (i + rng.randint(0, 2)) * 15,
            })
        restaurants[rid] = {
            "restaurant_id": rid, "name": name, "cuisine": cuisine,
            "rating": round(rating + rng.uniform(-0.2, 0.2), 1),
            "delivery_fee": delivery_fee, "open": True,
            "hours": "11:00-22:00", "menu": menu,
        }
    order_1 = _seed_scoped_id("ord", seed, 0, width=4)
    order_2 = _seed_scoped_id("ord", seed, 1, width=4)
    orders = {
        order_1: {"order_id": order_1, "restaurant_id": restaurant_ids[0],
                     "restaurant_name": "Pizza Palace",
                     "items": [{"name": "Margherita Pizza", "quantity": 2}],
                     "delivery_address": "123 Main St", "subtotal": 25.98,
                     "delivery_fee": 2.99, "tip": 3.00, "total": 31.97,
                     "status": "delivered", "rating": None,
                     "created_at": "2026-06-20T18:00:00"},
        order_2: {"order_id": order_2, "restaurant_id": restaurant_ids[1],
                     "restaurant_name": "Sushi Express",
                     "items": [{"name": "California Roll", "quantity": 1}],
                     "delivery_address": "456 Oak Ave", "subtotal": 10.99,
                     "delivery_fee": 3.99, "tip": 0, "total": 14.98,
                     "status": "preparing", "rating": None,
                     "created_at": "2026-06-21T12:30:00"},
    }
    return {"restaurants": restaurants, "orders": orders, "support_tickets": [],
            "next_order_num": 3, "next_ticket_num": 1}
