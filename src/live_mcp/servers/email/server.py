"""Stateful email server — 17 tools (PROVE-aligned).
Append-only: inbox, drafts, sent, threads, labels, filters, attachments, forwarding.
"""

from __future__ import annotations
from typing import Any
from src.live_mcp.server_base import StatefulToolServer, _result, serve

TOOLS = [
    {"name": "list_inbox", "description": "List emails in inbox with optional label/category filter.", "input_schema": {"type": "object", "properties": {"label": {"type": "string"}, "category": {"type": "string"}, "limit": {"type": "integer"}, "unread_only": {"type": "boolean"}}, "required": []}, "annotations": {"readonly": True, "mutating": False}},
    {"name": "search_emails", "description": "Full-text search across sender, subject, body.", "input_schema": {"type": "object", "properties": {"sender": {"type": "string"}, "subject_contains": {"type": "string"}, "keyword": {"type": "string"}, "after_date": {"type": "string"}, "before_date": {"type": "string"}}, "required": []}, "annotations": {"readonly": True, "mutating": False}},
    {"name": "get_email", "description": "Get full email by id including headers, body, attachments.", "input_schema": {"type": "object", "properties": {"email_id": {"type": "string"}}, "required": ["email_id"]}, "annotations": {"readonly": True, "mutating": False}},
    {"name": "send_email", "description": "Send email. Sensitive params. Appends to thread if thread_id given.", "input_schema": {"type": "object", "properties": {"to": {"type": "string"}, "cc": {"type": "string"}, "bcc": {"type": "string"}, "subject": {"type": "string"}, "body": {"type": "string"}, "thread_id": {"type": "string"}, "attachments": {"type": "array"}}, "required": ["to", "subject", "body"]}, "annotations": {"mutating": True, "sensitive_params": True}},
    {"name": "create_draft", "description": "Create an email draft.", "input_schema": {"type": "object", "properties": {"to": {"type": "string"}, "cc": {"type": "string"}, "subject": {"type": "string"}, "body": {"type": "string"}}, "required": ["to", "subject", "body"]}, "annotations": {"mutating": True}},
    {"name": "forward_email", "description": "Forward an existing email.", "input_schema": {"type": "object", "properties": {"email_id": {"type": "string"}, "to": {"type": "string"}, "additional_note": {"type": "string"}}, "required": ["email_id", "to"]}, "annotations": {"mutating": True}},
    {"name": "reply_email", "description": "Reply to an existing email in thread.", "input_schema": {"type": "object", "properties": {"email_id": {"type": "string"}, "body": {"type": "string"}}, "required": ["email_id", "body"]}, "annotations": {"mutating": True}},
    {"name": "add_label", "description": "Add a label to an email or draft.", "input_schema": {"type": "object", "properties": {"email_id": {"type": "string"}, "label": {"type": "string"}}, "required": ["email_id", "label"]}, "annotations": {"mutating": True}},
    {"name": "remove_label", "description": "Remove a label from an email.", "input_schema": {"type": "object", "properties": {"email_id": {"type": "string"}, "label": {"type": "string"}}, "required": ["email_id", "label"]}, "annotations": {"mutating": True}},
    {"name": "move_to_thread", "description": "Move an email to a thread.", "input_schema": {"type": "object", "properties": {"email_id": {"type": "string"}, "thread_id": {"type": "string"}}, "required": ["email_id", "thread_id"]}, "annotations": {"mutating": True}},
    {"name": "get_thread", "description": "Get all emails in a thread.", "input_schema": {"type": "object", "properties": {"thread_id": {"type": "string"}}, "required": ["thread_id"]}, "annotations": {"readonly": True, "mutating": False}},
    {"name": "archive_email", "description": "Archive an email (remove from inbox).", "input_schema": {"type": "object", "properties": {"email_id": {"type": "string"}}, "required": ["email_id"]}, "annotations": {"mutating": True}},
    {"name": "mark_read", "description": "Mark email as read.", "input_schema": {"type": "object", "properties": {"email_id": {"type": "string"}}, "required": ["email_id"]}, "annotations": {"mutating": True}},
    {"name": "mark_unread", "description": "Mark email as unread.", "input_schema": {"type": "object", "properties": {"email_id": {"type": "string"}}, "required": ["email_id"]}, "annotations": {"mutating": True}},
    {"name": "create_filter", "description": "Create an email filter rule.", "input_schema": {"type": "object", "properties": {"field": {"type": "string"}, "pattern": {"type": "string"}, "action": {"type": "string"}, "label": {"type": "string"}}, "required": ["field", "pattern", "action"]}, "annotations": {"mutating": True}},
    {"name": "list_filters", "description": "List active email filters.", "input_schema": {"type": "object", "properties": {}, "required": []}, "annotations": {"readonly": True, "mutating": False}},
    {"name": "get_attachments", "description": "List attachments on an email.", "input_schema": {"type": "object", "properties": {"email_id": {"type": "string"}}, "required": ["email_id"]}, "annotations": {"readonly": True, "mutating": False}},
]

class EmailServer(StatefulToolServer):
    def __init__(self) -> None:
        super().__init__("email", TOOLS)
        self.handlers = {t["name"]: getattr(self, t["name"]) for t in TOOLS}

    def _nxt_eml(self, state): eid = f"eml_{state['next_email_num']:04d}"; state["next_email_num"] += 1; return eid

    def list_inbox(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id); label = arguments.get("label"); limit = int(arguments.get("limit", 20))
        emails = [state["emails"][eid] for eid in state["inbox_order"] if eid in state["emails"]]
        if arguments.get("unread_only"): emails = [e for e in emails if not e.get("read", False)]
        if label: emails = [e for e in emails if label in e.get("labels", [])]
        if arguments.get("category"): emails = [e for e in emails if e.get("category") == arguments["category"]]
        return _result(True, {"emails": emails[-limit:], "total": len(emails)}, None, "", False)

    def search_emails(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id); results = list(state["emails"].values())
        if arguments.get("sender"): s = arguments["sender"].lower(); results = [e for e in results if s in e["sender"].lower()]
        if arguments.get("subject_contains"): s = arguments["subject_contains"].lower(); results = [e for e in results if s in e["subject"].lower()]
        if arguments.get("keyword"): kw = arguments["keyword"].lower(); results = [e for e in results if kw in e["subject"].lower() or kw in e["body"].lower()]
        if arguments.get("after_date"): d = arguments["after_date"]; results = [e for e in results if e.get("date", "") >= d]
        if arguments.get("before_date"): d = arguments["before_date"]; results = [e for e in results if e.get("date", "") <= d]
        return _result(True, {"emails": results[:20], "count": len(results)}, None, "", False)

    def get_email(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        eml = self._state(session_id)["emails"].get(arguments["email_id"])
        if not eml: raise KeyError(f"email not found: {arguments['email_id']}")
        return _result(True, {"email": eml}, None, "", False)

    def send_email(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id); eid = self._nxt_eml(state)
        tid = arguments.get("thread_id") or f"thd_{state['next_thread_num']:03d}"
        if not arguments.get("thread_id"):
            state["next_thread_num"] += 1
        if tid not in state["threads"]: state["threads"][tid] = []
        email = {"email_id": eid, "to": arguments["to"], "cc": arguments.get("cc", ""), "bcc": arguments.get("bcc", ""), "sender": "current_user@example.com", "subject": arguments["subject"], "body": arguments["body"], "labels": [], "thread_id": tid, "status": "sent", "date": "2026-06-24", "read": True, "attachments": arguments.get("attachments", [])}
        state["emails"][eid] = email; state["inbox_order"].append(eid); state["threads"][tid].append(eid)
        return _result(True, {"email": email}, None, "", True)

    def create_draft(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id); eid = f"dft_{state['next_email_num']:04d}"; state["next_email_num"] += 1
        draft = {"email_id": eid, "to": arguments["to"], "cc": arguments.get("cc", ""), "subject": arguments["subject"], "body": arguments["body"], "labels": [], "status": "draft", "date": "2026-06-24"}
        state["drafts"][eid] = draft
        return _result(True, {"draft": draft}, None, "", True)

    def forward_email(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id); orig = state["emails"].get(arguments["email_id"])
        if not orig: raise KeyError(f"email not found: {arguments['email_id']}")
        eid = self._nxt_eml(state); note = arguments.get("additional_note", "")
        body = f"---------- Forwarded message ----------\nFrom: {orig['sender']}\nSubject: {orig['subject']}\n\n{orig['body']}"
        if note: body = note + "\n\n" + body
        email = {"email_id": eid, "to": arguments["to"], "cc": "", "sender": "current_user@example.com", "subject": f"Fwd: {orig['subject']}", "body": body, "labels": [], "thread_id": f"thd_{state['next_thread_num']:03d}", "status": "sent", "date": "2026-06-24", "read": True}
        state["emails"][eid] = email; state["inbox_order"].append(eid)
        state["threads"].setdefault(email["thread_id"], []).append(eid); state["next_thread_num"] += 1
        return _result(True, {"email": email}, None, "", True)

    def reply_email(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id); orig = state["emails"].get(arguments["email_id"])
        if not orig: raise KeyError(f"email not found: {arguments['email_id']}")
        eid = self._nxt_eml(state); tid = orig.get("thread_id", f"thd_{state['next_thread_num']:03d}")
        email = {"email_id": eid, "to": orig["sender"], "cc": "", "sender": "current_user@example.com", "subject": f"Re: {orig['subject']}", "body": arguments["body"], "labels": [], "thread_id": tid, "status": "sent", "date": "2026-06-24", "read": True}
        state["emails"][eid] = email; state["inbox_order"].append(eid)
        state["threads"].setdefault(tid, []).append(eid)
        return _result(True, {"email": email}, None, "", True)

    def add_label(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id); eid = arguments["email_id"]; label = arguments["label"]
        email = state["emails"].get(eid) or state["drafts"].get(eid)
        if not email: raise KeyError(f"email not found: {eid}")
        if label not in email["labels"]: email["labels"].append(label)
        return _result(True, {"email_id": eid, "labels": email["labels"]}, None, "", True)

    def remove_label(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id); eid = arguments["email_id"]; label = arguments["label"]
        email = state["emails"].get(eid) or state["drafts"].get(eid)
        if not email: raise KeyError(f"email not found: {eid}")
        email["labels"] = [l for l in email["labels"] if l != label]
        return _result(True, {"email_id": eid, "labels": email["labels"]}, None, "", True)

    def move_to_thread(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id); eid, tid = arguments["email_id"], arguments["thread_id"]
        if eid not in state["emails"]: raise KeyError(f"email not found: {eid}")
        old_tid = state["emails"][eid].get("thread_id")
        if old_tid and old_tid in state["threads"] and eid in state["threads"][old_tid]: state["threads"][old_tid].remove(eid)
        state["emails"][eid]["thread_id"] = tid
        state["threads"].setdefault(tid, []).append(eid)
        return _result(True, {"email_id": eid, "thread_id": tid}, None, "", True)

    def get_thread(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id); tid = arguments["thread_id"]
        if tid not in state["threads"]: raise KeyError(f"thread not found: {tid}")
        emails = [state["emails"][eid] for eid in state["threads"][tid] if eid in state["emails"]]
        return _result(True, {"thread_id": tid, "emails": emails, "count": len(emails)}, None, "", False)

    def archive_email(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id); eid = arguments["email_id"]
        if eid not in state["emails"]: raise KeyError(f"email not found: {eid}")
        state["emails"][eid]["archived"] = True
        if eid in state["inbox_order"]: state["inbox_order"].remove(eid)
        return _result(True, {"email_id": eid, "archived": True}, None, "", True)

    def mark_read(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id); eid = arguments["email_id"]
        if eid not in state["emails"] and eid not in state["drafts"]: raise KeyError(f"email not found: {eid}")
        (state["emails"].get(eid) or state["drafts"].get(eid))["read"] = True
        return _result(True, {"email_id": eid, "read": True}, None, "", True)

    def mark_unread(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id); eid = arguments["email_id"]
        if eid not in state["emails"] and eid not in state["drafts"]: raise KeyError(f"email not found: {eid}")
        (state["emails"].get(eid) or state["drafts"].get(eid))["read"] = False
        return _result(True, {"email_id": eid, "read": False}, None, "", True)

    def create_filter(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id); fid = f"flt_{len(state.setdefault('filters', {})) + 1:04d}"
        filt = {"filter_id": fid, "field": arguments["field"], "pattern": arguments["pattern"], "action": arguments["action"], "label": arguments.get("label", "")}
        state["filters"][fid] = filt
        return _result(True, {"filter": filt}, None, "", True)

    def list_filters(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        filts = list(self._state(session_id).get("filters", {}).values())
        return _result(True, {"filters": filts, "count": len(filts)}, None, "", False)

    def get_attachments(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        eml = self._state(session_id)["emails"].get(arguments["email_id"])
        if not eml: raise KeyError(f"email not found: {arguments['email_id']}")
        return _result(True, {"email_id": eml["email_id"], "attachments": eml.get("attachments", []), "count": len(eml.get("attachments", []))}, None, "", False)


if __name__ == "__main__":
    serve(EmailServer())
