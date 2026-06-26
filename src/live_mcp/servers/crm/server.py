"""Stateful CRM server — 16 tools (PROVE-aligned).
Relational state: leads, contacts, deals, tasks, notes.
Safety: identity_policy=preserve, reference integrity.
"""

from __future__ import annotations
from typing import Any
from src.live_mcp.server_base import StatefulToolServer, _result, serve

TOOLS = [
    {"name": "create_lead", "description": "Create a new lead.", "input_schema": {"type": "object", "properties": {"name": {"type": "string"}, "company": {"type": "string"}, "source": {"type": "string"}, "email": {"type": "string"}, "phone": {"type": "string"}}, "required": ["name", "company"]}, "annotations": {"mutating": True}},
    {"name": "update_lead", "description": "Update lead fields.", "input_schema": {"type": "object", "properties": {"lead_id": {"type": "string"}, "fields": {"type": "object"}}, "required": ["lead_id", "fields"]}, "annotations": {"mutating": True}},
    {"name": "convert_lead", "description": "Convert a lead into a contact.", "input_schema": {"type": "object", "properties": {"lead_id": {"type": "string"}}, "required": ["lead_id"]}, "annotations": {"mutating": True}},
    {"name": "delete_lead", "description": "Delete a lead (only if not converted).", "input_schema": {"type": "object", "properties": {"lead_id": {"type": "string"}}, "required": ["lead_id"]}, "annotations": {"mutating": True}},
    {"name": "list_leads", "description": "List leads by status, source, or company.", "input_schema": {"type": "object", "properties": {"status": {"type": "string"}, "source": {"type": "string"}, "company": {"type": "string"}}, "required": []}, "annotations": {"readonly": True, "mutating": False}},
    {"name": "create_contact", "description": "Create a contact directly.", "input_schema": {"type": "object", "properties": {"name": {"type": "string"}, "email": {"type": "string"}, "phone": {"type": "string"}, "company": {"type": "string"}}, "required": ["name", "email"]}, "annotations": {"mutating": True}},
    {"name": "update_contact", "description": "Update contact fields.", "input_schema": {"type": "object", "properties": {"contact_id": {"type": "string"}, "fields": {"type": "object"}}, "required": ["contact_id", "fields"]}, "annotations": {"mutating": True}},
    {"name": "delete_contact", "description": "Delete a contact (fails if referenced by deals).", "input_schema": {"type": "object", "properties": {"contact_id": {"type": "string"}}, "required": ["contact_id"]}, "annotations": {"mutating": True}},
    {"name": "create_deal", "description": "Create a deal linked to contact/lead.", "input_schema": {"type": "object", "properties": {"name": {"type": "string"}, "amount": {"type": "number"}, "contact_id": {"type": "string"}, "lead_id": {"type": "string"}, "stage": {"type": "string"}}, "required": ["name", "amount"]}, "annotations": {"mutating": True}},
    {"name": "update_deal", "description": "Update deal stage or amount.", "input_schema": {"type": "object", "properties": {"deal_id": {"type": "string"}, "stage": {"type": "string"}, "amount": {"type": "number"}}, "required": ["deal_id"]}, "annotations": {"mutating": True}},
    {"name": "list_deals", "description": "List deals by stage/contact/lead.", "input_schema": {"type": "object", "properties": {"stage": {"type": "string"}, "contact_id": {"type": "string"}, "lead_id": {"type": "string"}}, "required": []}, "annotations": {"readonly": True, "mutating": False}},
    {"name": "get_deal", "description": "Get full deal details with linked contact/lead.", "input_schema": {"type": "object", "properties": {"deal_id": {"type": "string"}}, "required": ["deal_id"]}, "annotations": {"readonly": True, "mutating": False}},
    {"name": "create_task", "description": "Create a task related to a deal or contact.", "input_schema": {"type": "object", "properties": {"title": {"type": "string"}, "deal_id": {"type": "string"}, "contact_id": {"type": "string"}, "due_date": {"type": "string"}, "priority": {"type": "string"}}, "required": ["title"]}, "annotations": {"mutating": True}},
    {"name": "list_tasks", "description": "List tasks by status, deal, or priority.", "input_schema": {"type": "object", "properties": {"status": {"type": "string"}, "deal_id": {"type": "string"}, "priority": {"type": "string"}}, "required": []}, "annotations": {"readonly": True, "mutating": False}},
    {"name": "complete_task", "description": "Mark a task as completed.", "input_schema": {"type": "object", "properties": {"task_id": {"type": "string"}}, "required": ["task_id"]}, "annotations": {"mutating": True}},
    {"name": "add_note", "description": "Add a note to a deal, contact, or lead.", "input_schema": {"type": "object", "properties": {"entity_type": {"type": "string"}, "entity_id": {"type": "string"}, "content": {"type": "string"}}, "required": ["entity_type", "entity_id", "content"]}, "annotations": {"mutating": True}},
]

VALID_STAGES = ["prospecting", "qualification", "proposal", "negotiation", "closed_won", "closed_lost"]

class CRMServer(StatefulToolServer):
    def __init__(self) -> None:
        super().__init__("crm", TOOLS)
        self.handlers = {t["name"]: getattr(self, t["name"]) for t in TOOLS}

    def create_lead(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id); lid = f"lead_{state['next_lead_num']:04d}"; state["next_lead_num"] += 1
        lead = {"lead_id": lid, "name": arguments["name"], "company": arguments["company"], "source": arguments.get("source", ""), "email": arguments.get("email", ""), "phone": arguments.get("phone", ""), "status": "new", "contact_id": None}
        state["leads"][lid] = lead
        return _result(True, {"lead": lead}, None, "", True)

    def update_lead(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id); lead = state["leads"].get(arguments["lead_id"])
        if not lead: raise KeyError(f"lead not found: {arguments['lead_id']}")
        for k, v in arguments["fields"].items():
            if k in ("name", "company", "source", "email", "phone"): lead[k] = v
        return _result(True, {"lead": lead}, None, "", True)

    def convert_lead(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id); lid = arguments["lead_id"]; lead = state["leads"].get(lid)
        if not lead: raise KeyError(f"lead not found: {lid}")
        if lead["status"] == "converted": raise KeyError("lead already converted")
        if lead["status"] == "lost": raise KeyError("cannot convert lost lead")
        cid = f"contact_{state['next_contact_num']:04d}"; state["next_contact_num"] += 1
        contact = {"contact_id": cid, "name": lead["name"], "email": lead.get("email", ""), "phone": lead.get("phone", ""), "company": lead["company"], "lead_id": lid}
        state["contacts"][cid] = contact; lead["status"] = "converted"; lead["contact_id"] = cid
        return _result(True, {"lead": lead, "contact": contact}, None, "", True)

    def delete_lead(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id); lid = arguments["lead_id"]; lead = state["leads"].get(lid)
        if not lead: raise KeyError(f"lead not found: {lid}")
        if lead["status"] == "converted": raise KeyError("cannot delete converted lead")
        state["leads"].pop(lid)
        return _result(True, {"deleted_lead": lead}, None, "", True)

    def list_leads(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id); leads = list(state["leads"].values())
        if arguments.get("status"): leads = [l for l in leads if l["status"] == arguments["status"]]
        if arguments.get("source"): leads = [l for l in leads if l["source"] == arguments["source"]]
        if arguments.get("company"): leads = [l for l in leads if l["company"] == arguments["company"]]
        return _result(True, {"leads": leads, "count": len(leads)}, None, "", False)

    def create_contact(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id); cid = f"contact_{state['next_contact_num']:04d}"; state["next_contact_num"] += 1
        contact = {"contact_id": cid, "name": arguments["name"], "email": arguments["email"], "phone": arguments.get("phone", ""), "company": arguments.get("company", ""), "lead_id": None}
        state["contacts"][cid] = contact
        return _result(True, {"contact": contact}, None, "", True)

    def update_contact(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id); contact = state["contacts"].get(arguments["contact_id"])
        if not contact: raise KeyError(f"contact not found: {arguments['contact_id']}")
        for k, v in arguments["fields"].items():
            if k in ("name", "email", "phone", "company"): contact[k] = v
        return _result(True, {"contact": contact}, None, "", True)

    def delete_contact(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id); cid = arguments["contact_id"]
        if cid not in state["contacts"]: raise KeyError(f"contact not found: {cid}")
        refs = [d for d in state["deals"].values() if d.get("contact_id") == cid]
        if refs: raise KeyError(f"contact referenced by {len(refs)} deal(s)")
        state["contacts"].pop(cid)
        return _result(True, {"deleted_contact_id": cid}, None, "", True)

    def create_deal(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id); did = f"deal_{state['next_deal_num']:04d}"; state["next_deal_num"] += 1
        cid = arguments.get("contact_id"); lid = arguments.get("lead_id")
        if cid and cid not in state["contacts"]: raise KeyError(f"contact not found: {cid}")
        if lid and lid not in state["leads"]: raise KeyError(f"lead not found: {lid}")
        deal = {"deal_id": did, "name": arguments["name"], "amount": float(arguments["amount"]), "stage": arguments.get("stage", "prospecting"), "contact_id": cid, "lead_id": lid, "created_at": "2026-06-24"}
        if deal["stage"] not in VALID_STAGES: raise KeyError(f"invalid stage: {deal['stage']}")
        state["deals"][did] = deal
        return _result(True, {"deal": deal}, None, "", True)

    def update_deal(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id); deal = state["deals"].get(arguments["deal_id"])
        if not deal: raise KeyError(f"deal not found: {arguments['deal_id']}")
        if "stage" in arguments:
            if arguments["stage"] not in VALID_STAGES: raise KeyError(f"invalid stage: {arguments['stage']}")
            deal["stage"] = arguments["stage"]
        if "amount" in arguments: deal["amount"] = float(arguments["amount"])
        return _result(True, {"deal": deal}, None, "", True)

    def list_deals(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id); deals = list(state["deals"].values())
        if arguments.get("stage"): deals = [d for d in deals if d["stage"] == arguments["stage"]]
        if arguments.get("contact_id"): deals = [d for d in deals if d.get("contact_id") == arguments["contact_id"]]
        if arguments.get("lead_id"): deals = [d for d in deals if d.get("lead_id") == arguments["lead_id"]]
        return _result(True, {"deals": deals, "count": len(deals)}, None, "", False)

    def get_deal(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id); deal = state["deals"].get(arguments["deal_id"])
        if not deal: raise KeyError(f"deal not found: {arguments['deal_id']}")
        contact = state["contacts"].get(deal.get("contact_id")) if deal.get("contact_id") else None
        lead = state["leads"].get(deal.get("lead_id")) if deal.get("lead_id") else None
        return _result(True, {"deal": deal, "contact": contact, "lead": lead}, None, "", False)

    def create_task(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id); tid = f"task_{state['next_task_num']:04d}"; state["next_task_num"] += 1
        task = {"task_id": tid, "title": arguments["title"], "deal_id": arguments.get("deal_id"), "contact_id": arguments.get("contact_id"), "due_date": arguments.get("due_date", ""), "priority": arguments.get("priority", "medium"), "status": "open"}
        state.setdefault("tasks", {})[tid] = task
        return _result(True, {"task": task}, None, "", True)

    def list_tasks(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id); tasks = list(state.get("tasks", {}).values())
        if arguments.get("status"): tasks = [t for t in tasks if t["status"] == arguments["status"]]
        if arguments.get("deal_id"): tasks = [t for t in tasks if t.get("deal_id") == arguments["deal_id"]]
        if arguments.get("priority"): tasks = [t for t in tasks if t["priority"] == arguments["priority"]]
        return _result(True, {"tasks": tasks, "count": len(tasks)}, None, "", False)

    def complete_task(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id); tid = arguments["task_id"]
        if tid not in state.get("tasks", {}): raise KeyError(f"task not found: {tid}")
        state["tasks"][tid]["status"] = "completed"
        return _result(True, {"task": state["tasks"][tid]}, None, "", True)

    def add_note(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id); etype, eid = arguments["entity_type"], arguments["entity_id"]
        entities = {"lead": state["leads"], "contact": state["contacts"], "deal": state["deals"]}
        if etype not in entities or eid not in entities[etype]: raise KeyError(f"{etype} not found: {eid}")
        nid = f"note_{state['next_note_num']:04d}"; state["next_note_num"] += 1
        note = {"note_id": nid, "entity_type": etype, "entity_id": eid, "content": arguments["content"], "created_at": "2026-06-24"}
        state.setdefault("notes", {})[nid] = note
        return _result(True, {"note": note}, None, "", True)

if __name__ == "__main__":
    serve(CRMServer())
