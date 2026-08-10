"""Stateful CRM server with 16 tools.
Relational state: leads, contacts, deals, tasks, notes.
Safety: identity_policy=preserve, reference integrity.
"""

from __future__ import annotations
from typing import Any
from src.live_mcp.server_base import StatefulToolServer, _result, serve

TOOLS = [
    {"name": "create_lead", "description": "Create a new lead.", "input_schema": {"type": "object", "properties": {"name": {"type": "string"}, "company": {"type": "string"}, "source": {"type": "string"}, "email": {"type": "string"}, "phone": {"type": "string"}}, "required": ["name", "company"]}, "annotations": {"mutating": True}},
    {"name": "update_lead", "description": "Update an existing lead's profile or pipeline status. Use convert_lead for the converted status.", "input_schema": {"type": "object", "properties": {"lead_id": {"type": "string"}, "fields": {"type": "object", "properties": {"name": {"type": "string"}, "company": {"type": "string"}, "source": {"type": "string"}, "email": {"type": "string"}, "phone": {"type": "string"}, "status": {"type": "string", "enum": ["new", "contacted", "qualified", "lost"], "description": "Pipeline status; converted is only produced by convert_lead."}}, "additionalProperties": False, "minProperties": 1}}, "required": ["lead_id", "fields"], "additionalProperties": False}, "annotations": {"mutating": True}},
    {"name": "convert_lead", "description": "Convert a lead into a contact.", "input_schema": {"type": "object", "properties": {"lead_id": {"type": "string"}}, "required": ["lead_id"]}, "annotations": {"mutating": True}},
    {"name": "delete_lead", "description": "Delete a lead (only if not converted).", "input_schema": {"type": "object", "properties": {"lead_id": {"type": "string"}}, "required": ["lead_id"]}, "annotations": {"mutating": True}},
    {"name": "list_leads", "description": "List leads by status, source, or company.", "input_schema": {"type": "object", "properties": {"status": {"type": "string"}, "source": {"type": "string"}, "company": {"type": "string"}}, "required": []}, "annotations": {"readonly": True, "mutating": False}},
    {"name": "create_contact", "description": "Create a contact directly.", "input_schema": {"type": "object", "properties": {"name": {"type": "string"}, "email": {"type": "string"}, "phone": {"type": "string"}, "company": {"type": "string"}}, "required": ["name", "email"]}, "annotations": {"mutating": True}},
    {"name": "update_contact", "description": "Update an existing contact's name, email, phone, or company.", "input_schema": {"type": "object", "properties": {"contact_id": {"type": "string"}, "fields": {"type": "object", "properties": {"name": {"type": "string"}, "email": {"type": "string"}, "phone": {"type": "string"}, "company": {"type": "string"}}, "additionalProperties": False, "minProperties": 1}}, "required": ["contact_id", "fields"], "additionalProperties": False}, "annotations": {"mutating": True}},
    {"name": "delete_contact", "description": "Delete a contact only when it is not referenced by any deal or converted lead.", "input_schema": {"type": "object", "properties": {"contact_id": {"type": "string", "description": "Existing contact_id with no deal or converted-lead references."}}, "required": ["contact_id"]}, "annotations": {"mutating": True}},
    {"name": "create_deal", "description": "Create a deal linked to at least one existing contact or lead. The amount must be greater than zero.", "input_schema": {"type": "object", "properties": {"name": {"type": "string"}, "amount": {"type": "number", "exclusiveMinimum": 0, "description": "Positive deal amount; must be greater than zero."}, "contact_id": {"type": "string", "description": "Existing contact_id; at least contact_id or lead_id is required."}, "lead_id": {"type": "string", "description": "Existing lead_id; at least contact_id or lead_id is required."}, "stage": {"type": "string", "enum": ["prospecting", "qualification", "proposal", "negotiation", "closed_won", "closed_lost"]}}, "required": ["name", "amount"]}, "annotations": {"mutating": True}},
    {"name": "update_deal", "description": "Update at least one of deal stage or amount. Any supplied amount must be greater than zero.", "input_schema": {"type": "object", "properties": {"deal_id": {"type": "string"}, "stage": {"type": "string", "enum": ["prospecting", "qualification", "proposal", "negotiation", "closed_won", "closed_lost"]}, "amount": {"type": "number", "exclusiveMinimum": 0, "description": "Positive deal amount; must be greater than zero."}}, "required": ["deal_id"]}, "annotations": {"mutating": True}},
    {"name": "list_deals", "description": "List deals by stage/contact/lead.", "input_schema": {"type": "object", "properties": {"stage": {"type": "string"}, "contact_id": {"type": "string"}, "lead_id": {"type": "string"}}, "required": []}, "annotations": {"readonly": True, "mutating": False}},
    {"name": "get_deal", "description": "Get full deal details with linked contact/lead.", "input_schema": {"type": "object", "properties": {"deal_id": {"type": "string"}}, "required": ["deal_id"]}, "annotations": {"readonly": True, "mutating": False}},
    {"name": "create_task", "description": "Create a task. At least one of deal_id or contact_id must reference an existing entity.", "input_schema": {"type": "object", "properties": {"title": {"type": "string"}, "deal_id": {"type": "string", "description": "Existing deal_id. At least one of deal_id or contact_id is required."}, "contact_id": {"type": "string", "description": "Existing contact_id. At least one of deal_id or contact_id is required."}, "due_date": {"type": "string"}, "priority": {"type": "string"}}, "required": ["title"]}, "annotations": {"mutating": True}},
    {"name": "list_tasks", "description": "List tasks by status, deal, or priority.", "input_schema": {"type": "object", "properties": {"status": {"type": "string"}, "deal_id": {"type": "string"}, "priority": {"type": "string"}}, "required": []}, "annotations": {"readonly": True, "mutating": False}},
    {"name": "complete_task", "description": "Mark a task as completed.", "input_schema": {"type": "object", "properties": {"task_id": {"type": "string"}}, "required": ["task_id"]}, "annotations": {"mutating": True}},
    {"name": "add_note", "description": "Add a note to a deal, contact, or lead.", "input_schema": {"type": "object", "properties": {"entity_type": {"type": "string"}, "entity_id": {"type": "string"}, "content": {"type": "string"}}, "required": ["entity_type", "entity_id", "content"]}, "annotations": {"mutating": True}},
]

VALID_STAGES = ["prospecting", "qualification", "proposal", "negotiation", "closed_won", "closed_lost"]

class CRMServer(StatefulToolServer):
    def __init__(self) -> None:
        super().__init__("crm", TOOLS)
        self.handlers = {t["name"]: getattr(self, t["name"]) for t in TOOLS}

    @staticmethod
    def _lead_deletable(state: dict[str, Any], lead_id: str) -> bool:
        lead = state["leads"].get(lead_id)
        return bool(lead) and lead.get("status") != "converted" and not any(
            deal.get("lead_id") == lead_id
            for deal in state["deals"].values()
        ) and not any(
            note.get("entity_type") == "lead"
            and note.get("entity_id") == lead_id
            for note in state.get("notes", {}).values()
        )

    @staticmethod
    def _contact_deletable(state: dict[str, Any], contact_id: str) -> bool:
        references = (
            (state["deals"].values(), "contact_id"),
            (state["leads"].values(), "contact_id"),
            (state.get("tasks", {}).values(), "contact_id"),
        )
        return contact_id in state["contacts"] and not any(
            item.get(field) == contact_id
            for items, field in references
            for item in items
        ) and not any(
            note.get("entity_type") == "contact"
            and note.get("entity_id") == contact_id
            for note in state.get("notes", {}).values()
        )

    def create_lead(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id); lid = f"lead_{state['next_lead_num']:04d}"; state["next_lead_num"] += 1
        lead = {"lead_id": lid, "name": arguments["name"], "company": arguments["company"], "source": arguments.get("source", ""), "email": arguments.get("email", ""), "phone": arguments.get("phone", ""), "status": "new", "contact_id": None}
        state["leads"][lid] = lead
        return _result(True, {"lead": lead}, None, "", True)

    def update_lead(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id); lead = state["leads"].get(arguments["lead_id"])
        if not lead: raise KeyError(f"lead not found: {arguments['lead_id']}")
        changed = False
        allowed = {"name", "company", "source", "email", "phone", "status"}
        unsupported = sorted(set(arguments["fields"]) - allowed)
        if unsupported:
            raise KeyError(f"unsupported lead field(s): {', '.join(unsupported)}")
        if arguments["fields"].get("status") == "converted":
            raise KeyError("use convert_lead to convert a lead")
        for k, v in arguments["fields"].items():
            if k in allowed and lead.get(k) != v:
                lead[k] = v
                changed = True
        return _result(True, {"lead": lead}, None, "", changed)

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
        if not self._lead_deletable(state, lid):
            raise KeyError("lead is converted or referenced")
        state["leads"].pop(lid)
        return _result(True, {"deleted_lead": lead}, None, "", True)

    def list_leads(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id); leads = [
            {**lead, "deletable": self._lead_deletable(state, lead_id)}
            for lead_id, lead in state["leads"].items()
        ]
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
        changed = False
        allowed = {"name", "email", "phone", "company"}
        unsupported = sorted(set(arguments["fields"]) - allowed)
        if unsupported:
            raise KeyError(f"unsupported contact field(s): {', '.join(unsupported)}")
        for k, v in arguments["fields"].items():
            if k in allowed and contact.get(k) != v:
                contact[k] = v
                changed = True
        return _result(True, {"contact": contact}, None, "", changed)

    def delete_contact(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id); cid = arguments["contact_id"]
        if cid not in state["contacts"]: raise KeyError(f"contact not found: {cid}")
        if not self._contact_deletable(state, cid):
            raise KeyError("contact is referenced")
        state["contacts"].pop(cid)
        return _result(True, {"deleted_contact_id": cid}, None, "", True)

    def create_deal(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id)
        cid = arguments.get("contact_id"); lid = arguments.get("lead_id")
        if not cid and not lid: raise KeyError("contact_id or lead_id is required")
        if cid and cid not in state["contacts"]: raise KeyError(f"contact not found: {cid}")
        if lid and lid not in state["leads"]: raise KeyError(f"lead not found: {lid}")
        amount = float(arguments["amount"])
        if amount <= 0: raise KeyError("amount must be positive")
        stage = arguments.get("stage", "prospecting")
        if stage not in VALID_STAGES: raise KeyError(f"invalid stage: '{stage}'. Valid stages: {VALID_STAGES}")
        did = f"deal_{state['next_deal_num']:04d}"; state["next_deal_num"] += 1
        deal = {"deal_id": did, "name": arguments["name"], "amount": amount, "stage": stage, "contact_id": cid, "lead_id": lid, "created_at": state["current_date"]}
        state["deals"][did] = deal
        return _result(True, {"deal": deal}, None, "", True)

    def update_deal(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id); deal = state["deals"].get(arguments["deal_id"])
        if not deal: raise KeyError(f"deal not found: {arguments['deal_id']}")
        if "stage" not in arguments and "amount" not in arguments:
            raise KeyError("stage or amount is required")
        changed = False
        if "stage" in arguments:
            if arguments["stage"] not in VALID_STAGES: raise KeyError(f"invalid stage: {arguments['stage']}")
            if deal.get("stage") != arguments["stage"]:
                deal["stage"] = arguments["stage"]
                changed = True
        if "amount" in arguments:
            amount = float(arguments["amount"])
            if amount <= 0: raise KeyError("amount must be positive")
            if deal.get("amount") != amount:
                deal["amount"] = amount
                changed = True
        return _result(True, {"deal": deal}, None, "", changed)

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
        if contact is not None:
            contact = {
                **contact,
                "deletable": self._contact_deletable(
                    state, str(contact["contact_id"]),
                ),
            }
        lead = state["leads"].get(deal.get("lead_id")) if deal.get("lead_id") else None
        return _result(True, {"deal": deal, "contact": contact, "lead": lead}, None, "", False)

    def create_task(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id)
        if not arguments.get("deal_id") and not arguments.get("contact_id"):
            raise KeyError("deal_id or contact_id is required")
        if arguments.get("deal_id") and arguments["deal_id"] not in state["deals"]:
            raise KeyError(f"deal not found: {arguments['deal_id']}")
        if arguments.get("contact_id") and arguments["contact_id"] not in state["contacts"]:
            raise KeyError(f"contact not found: {arguments['contact_id']}")
        tid = f"task_{state['next_task_num']:04d}"; state["next_task_num"] += 1
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
        changed = state["tasks"][tid].get("status") != "completed"
        state["tasks"][tid]["status"] = "completed"
        return _result(True, {"task": state["tasks"][tid]}, None, "", changed)

    def add_note(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id); etype, eid = arguments["entity_type"], arguments["entity_id"]
        entities = {"lead": state["leads"], "contact": state["contacts"], "deal": state["deals"]}
        if etype not in entities or eid not in entities[etype]: raise KeyError(f"{etype} not found: {eid}")
        nid = f"note_{state['next_note_num']:04d}"; state["next_note_num"] += 1
        note = {"note_id": nid, "entity_type": etype, "entity_id": eid, "content": arguments["content"], "created_at": state["current_date"]}
        state.setdefault("notes", {})[nid] = note
        return _result(True, {"note": note}, None, "", True)

if __name__ == "__main__":
    serve(CRMServer())
