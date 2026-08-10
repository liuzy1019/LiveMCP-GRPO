"""Deterministic state builder for crm."""

from __future__ import annotations

import datetime as _datetime
import random
from typing import Any

from src.live_mcp.state_seeders.common import (
    _reference_datetime,
    _sample_entities,
    _seed_scoped_id,
)
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

_CRM_DEAL_TEMPLATES: list[tuple[str, float, str]] = [
    ("Cloud Migration", 50000.00, "prospecting"),
    ("Data Pipeline", 75000.00, "proposal"),
    ("Mobile App Redesign", 35000.00, "negotiation"),
    ("AI Integration", 120000.00, "prospecting"),
    ("Security Audit", 25000.00, "closed_won"),
    ("ERP Rollout", 200000.00, "prospecting"),
    ("CRM Upgrade", 45000.00, "proposal"),
    ("Analytics Dashboard", 60000.00, "negotiation"),
    ("DevOps Toolchain", 85000.00, "prospecting"),
    ("Compliance Automation", 95000.00, "proposal"),
    ("E-commerce Platform", 180000.00, "negotiation"),
    ("Customer Portal", 70000.00, "closed_won"),
    ("Inventory System", 110000.00, "prospecting"),
    ("HR Onboarding Suite", 40000.00, "proposal"),
    ("Marketing Automation", 55000.00, "closed_won"),
    ("Logistics Dashboard", 130000.00, "negotiation"),
    ("Payment Gateway Refresh", 65000.00, "prospecting"),
    ("Data Warehouse Migration", 160000.00, "proposal"),
    ("Mobile Payment SDK", 90000.00, "closed_won"),
    ("Support Ticket System", 30000.00, "prospecting"),
]

def _crm_state(seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    reference_date = _reference_datetime(seed).date()
    leads_selected = _sample_entities(rng, _CRM_LEAD_TEMPLATES, target_count=20, id_prefix="lead")
    leads = {}
    statuses = ["new", "contacted", "qualified", "converted"]
    lead_ids: list[str] = []
    converted_pairs: list[tuple[str, str, str, str, str]] = []
    for idx, (_lid, name, company, source, status, email) in enumerate(leads_selected):
        lid = _seed_scoped_id("lead", seed, idx, width=4)
        contact_id = _seed_scoped_id("contact", seed, idx, width=4) if status == "converted" else None
        lead_ids.append(lid)
        if contact_id:
            converted_pairs.append((contact_id, name, email, company, lid))
        leads[lid] = {
            "lead_id": lid, "name": name, "company": company, "source": source,
            "email": email, "phone": f"555-{1000 + idx:04d}",
            "status": status, "contact_id": contact_id,
        }
    contacts = {
        cid: {"contact_id": cid, "name": name, "email": email, "phone": "",
              "company": company, "lead_id": lid}
        for cid, name, email, company, lid in converted_pairs
    }
    # Directly-seeded contacts (not from conversion) so delete_contact and
    # update_contact have targets independent of lead conversion status.
    direct_contacts_selected = _sample_entities(
        rng, _CRM_LEAD_TEMPLATES, target_count=5, id_prefix="contact",
    )
    for idx, (_lid, name, company, _source, _status, email) in enumerate(direct_contacts_selected):
        cid = _seed_scoped_id("contact", seed, idx + 100, width=4)
        contacts[cid] = {
            "contact_id": cid, "name": name, "email": email, "phone": "",
            "company": company, "lead_id": "",
        }
    contact_ids = list(contacts.keys())
    # Seed 3 deletable contacts not linked to any deal so delete_contact
    # chains are feasible. They must exist before the task loop so a task
    # can reference them, making them discoverable via list_tasks.
    deletable_specs = [
        ("Vivian Voss", "vivian@freelance.io", "FreeLabs"),
        ("Walter Webb", "walter@webbconsulting.com", "Webb Consulting"),
        ("Xena Xu", "xena@startupx.co", "StartupX"),
    ]
    deletable_contact_ids: list[str] = []
    for d_idx, (name, email, company) in enumerate(deletable_specs):
        cid = _seed_scoped_id("contact", seed, d_idx + 200, width=4)
        contacts[cid] = {
            "contact_id": cid, "name": name, "email": email, "phone": "",
            "company": company, "lead_id": "",
        }
        deletable_contact_ids.append(cid)
    # contact_ids_for_deal excludes deletable contacts so deals never
    # reference them, keeping them eligible for deletion.
    contact_ids_for_deal = [c for c in contact_ids]
    deals_selected = _sample_entities(rng, _CRM_DEAL_TEMPLATES, target_count=15, id_prefix="deal")
    deals = {}
    for idx, (dname, damount, dstage) in enumerate(deals_selected):
        did = _seed_scoped_id("deal", seed, idx, width=4)
        lid_candidate = lead_ids[idx % len(lead_ids)] if lead_ids else None
        cid_candidate = contact_ids_for_deal[idx % len(contact_ids_for_deal)] if contact_ids_for_deal else None
        deals[did] = {
            "deal_id": did, "name": dname,
            "amount": round(damount + rng.randint(-5000, 5000), 2),
            "stage": dstage, "contact_id": cid_candidate,
            "lead_id": lid_candidate,
            "created_at": (reference_date - _datetime.timedelta(days=idx * 5 + 1)).isoformat(),
        }
    # Seed enough tasks so list_tasks/update_task chains are diverse without
    # forcing every chain to create a task first.
    task_templates = [
        ("Follow up on proposal", "medium", 0),
        ("Schedule demo call", "high", 1),
        ("Send contract draft", "low", 2),
        ("Prepare quarterly report", "high", 3),
        ("Update CRM pipeline", "medium", 4),
        ("Call new lead", "high", 5),
        ("Review deal terms", "low", 6),
        ("Send onboarding docs", "medium", 7),
    ]
    tasks = {}
    for task_idx, (desc, priority, deal_offset) in enumerate(task_templates):
        tid = _seed_scoped_id("task", seed, task_idx, width=4)
        deal_ids = list(deals.keys())
        deal_id = deal_ids[deal_offset % len(deal_ids)]
        tasks[tid] = {
            "task_id": tid, "title": desc, "priority": priority,
            "status": "pending",
            "deal_id": deal_id,
            "contact_id": contact_ids[deal_offset % len(contact_ids)],
            "due_date": (reference_date + _datetime.timedelta(days=task_idx * 3 + 2)).isoformat(),
        }
    # Seed notes so get_notes chains have material data.
    note_1 = _seed_scoped_id("note", seed, 0, width=4)
    note_2 = _seed_scoped_id("note", seed, 1, width=4)
    deal_ids = list(deals.keys())
    notes = {
        note_1: {
            "note_id": note_1,
            "deal_id": deal_ids[0],
            "content": "Client requested additional demo before signing.",
            "created_at": (reference_date - _datetime.timedelta(days=2)).isoformat(),
        },
        note_2: {
            "note_id": note_2,
            "contact_id": contact_ids[0],
            "content": "Spoke about timeline extension for Q3 delivery.",
            "created_at": (reference_date - _datetime.timedelta(days=1)).isoformat(),
        },
    }
    # Make one task reference a deletable contact so list_tasks discovers
    # it and delete_contact chains become feasible.
    if deletable_contact_ids:
        last_task_key = list(tasks.keys())[-1]
        tasks[last_task_key]["contact_id"] = deletable_contact_ids[0]
    return {"leads": leads, "contacts": contacts, "deals": deals,
            "tasks": tasks, "notes": notes,
            "next_lead_num": len(leads) + 1, "next_contact_num": 1,
            "next_deal_num": len(deals) + 1, "next_task_num": len(tasks) + 1,
            "next_note_num": len(notes) + 1,
            "current_date": reference_date.isoformat()}
