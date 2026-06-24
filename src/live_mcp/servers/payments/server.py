"""Stateful payments server — 10 tools (PROVE-aligned).
Transactional state: invoices, payments, refunds, webhooks, disputes.
Safety: double-payment, refund-without-payment, dispute constraints.
"""

from __future__ import annotations
from typing import Any
from src.live_mcp.server_base import StatefulToolServer, _result, serve

TOOLS = [
    {"name": "create_invoice", "description": "Create a new invoice.", "input_schema": {"type": "object", "properties": {"customer": {"type": "string"}, "amount": {"type": "number"}, "currency": {"type": "string"}, "description": {"type": "string"}, "due_date": {"type": "string"}}, "required": ["customer", "amount"]}, "annotations": {"mutating": True}},
    {"name": "get_invoice", "description": "Get invoice details.", "input_schema": {"type": "object", "properties": {"invoice_id": {"type": "string"}}, "required": ["invoice_id"]}, "annotations": {"readonly": True, "mutating": False}},
    {"name": "list_invoices", "description": "List invoices by status, customer, or date range.", "input_schema": {"type": "object", "properties": {"status": {"type": "string"}, "customer": {"type": "string"}, "from_date": {"type": "string"}, "to_date": {"type": "string"}}, "required": []}, "annotations": {"readonly": True, "mutating": False}},
    {"name": "pay_invoice", "description": "Pay an invoice. Sensitive param on amount.", "input_schema": {"type": "object", "properties": {"invoice_id": {"type": "string"}, "amount": {"type": "number"}, "method": {"type": "string"}}, "required": ["invoice_id", "amount"]}, "annotations": {"mutating": True, "sensitive_params": True}},
    {"name": "refund_invoice", "description": "Refund a paid invoice (full or partial).", "input_schema": {"type": "object", "properties": {"invoice_id": {"type": "string"}, "amount": {"type": "number"}, "reason": {"type": "string"}}, "required": ["invoice_id", "amount"]}, "annotations": {"mutating": True, "sensitive_params": True}},
    {"name": "cancel_payment", "description": "Cancel a payment before settlement.", "input_schema": {"type": "object", "properties": {"payment_id": {"type": "string"}, "reason": {"type": "string"}}, "required": ["payment_id"]}, "annotations": {"mutating": True}},
    {"name": "dispute_invoice", "description": "File a dispute on an invoice.", "input_schema": {"type": "object", "properties": {"invoice_id": {"type": "string"}, "reason": {"type": "string"}, "evidence": {"type": "string"}}, "required": ["invoice_id", "reason"]}, "annotations": {"mutating": True}},
    {"name": "create_webhook", "description": "Register a webhook endpoint.", "input_schema": {"type": "object", "properties": {"url": {"type": "string"}, "events": {"type": "array"}}, "required": ["url", "events"]}, "annotations": {"mutating": True}},
    {"name": "list_webhooks", "description": "List registered webhooks.", "input_schema": {"type": "object", "properties": {}, "required": []}, "annotations": {"readonly": True, "mutating": False}},
    {"name": "delete_webhook", "description": "Delete a webhook registration.", "input_schema": {"type": "object", "properties": {"webhook_id": {"type": "string"}}, "required": ["webhook_id"]}, "annotations": {"mutating": True}},
]

class PaymentsServer(StatefulToolServer):
    def __init__(self) -> None:
        super().__init__("payments", TOOLS)
        self.handlers = {t["name"]: getattr(self, t["name"]) for t in TOOLS}

    def create_invoice(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id); inv_id = f"inv_{state['next_inv_num']:04d}"; state["next_inv_num"] += 1
        inv = {"invoice_id": inv_id, "customer": arguments["customer"], "amount": float(arguments["amount"]), "currency": arguments.get("currency", "USD"), "description": arguments.get("description", ""), "due_date": arguments.get("due_date", ""), "status": "pending", "payment_id": None, "refund_id": None, "created_at": "2026-06-24"}
        state["invoices"][inv_id] = inv
        return _result(True, {"invoice": inv}, None, "", True)

    def get_invoice(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        inv = self._state(session_id)["invoices"].get(arguments["invoice_id"])
        if not inv: raise KeyError(f"invoice not found: {arguments['invoice_id']}")
        return _result(True, {"invoice": inv}, None, "", False)

    def list_invoices(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id); invs = list(state["invoices"].values())
        st = arguments.get("status"); cust = arguments.get("customer"); fd = arguments.get("from_date"); td = arguments.get("to_date")
        if st: invs = [i for i in invs if i["status"] == st]
        if cust: invs = [i for i in invs if i["customer"] == cust]
        if fd: invs = [i for i in invs if i.get("created_at", "") >= fd]
        if td: invs = [i for i in invs if i.get("created_at", "") <= td]
        return _result(True, {"invoices": invs, "count": len(invs)}, None, "", False)

    def pay_invoice(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id); inv_id = arguments["invoice_id"]; inv = state["invoices"].get(inv_id)
        if not inv: raise KeyError(f"invoice not found: {inv_id}")
        if inv["status"] == "paid": raise KeyError("invoice already paid")
        if inv["status"] == "refunded": raise KeyError("invoice already refunded")
        amount = float(arguments["amount"])
        if abs(amount - inv["amount"]) > 0.01: raise KeyError(f"amount mismatch: {amount} vs {inv['amount']}")
        method = arguments.get("method", "card"); pid = f"pay_{state['next_pay_num']:04d}"; state["next_pay_num"] += 1
        inv["status"] = "paid"; inv["payment_id"] = pid
        state["payments"][pid] = {"payment_id": pid, "invoice_id": inv_id, "amount": amount, "method": method, "status": "settled"}
        return _result(True, {"invoice": inv, "payment": state["payments"][pid]}, None, "", True)

    def refund_invoice(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id); inv_id = arguments["invoice_id"]; inv = state["invoices"].get(inv_id)
        if not inv: raise KeyError(f"invoice not found: {inv_id}")
        if inv["status"] != "paid": raise KeyError(f"cannot refund invoice in status: {inv['status']}")
        amount = float(arguments["amount"])
        if amount > inv["amount"]: raise KeyError(f"refund exceeds invoice: {amount} > {inv['amount']}")
        rid = f"ref_{state['next_ref_num']:04d}"; state["next_ref_num"] += 1
        inv["status"] = "refunded"; inv["refund_id"] = rid
        state["refunds"][rid] = {"refund_id": rid, "invoice_id": inv_id, "amount": amount, "reason": arguments.get("reason", "")}
        return _result(True, {"invoice": inv, "refund": state["refunds"][rid]}, None, "", True)

    def cancel_payment(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id); pid = arguments["payment_id"]; pmt = state["payments"].get(pid)
        if not pmt: raise KeyError(f"payment not found: {pid}")
        if pmt["status"] != "settled": raise KeyError(f"payment already {pmt['status']}")
        pmt["status"] = "cancelled"; pmt["cancel_reason"] = arguments.get("reason", "")
        inv = state["invoices"][pmt["invoice_id"]]; inv["status"] = "pending"; inv["payment_id"] = None
        return _result(True, {"payment": pmt, "invoice_status": "pending"}, None, "", True)

    def dispute_invoice(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id); inv_id = arguments["invoice_id"]; inv = state["invoices"].get(inv_id)
        if not inv: raise KeyError(f"invoice not found: {inv_id}")
        if inv["status"] not in ("paid", "pending"): raise KeyError(f"cannot dispute invoice in status: {inv['status']}")
        did = f"dis_{state['next_inv_num']:04d}"; state["next_inv_num"] += 1
        dispute = {"dispute_id": did, "invoice_id": inv_id, "reason": arguments["reason"], "evidence": arguments.get("evidence", ""), "status": "open"}
        state.setdefault("disputes", {})[did] = dispute; inv["status"] = "disputed"
        return _result(True, {"dispute": dispute, "invoice_status": "disputed"}, None, "", True)

    def create_webhook(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id); wid = f"wh_{state['next_wh_num']:04d}"; state["next_wh_num"] += 1
        wh = {"webhook_id": wid, "url": arguments["url"], "events": arguments["events"], "active": True}
        state["webhooks"][wid] = wh
        return _result(True, {"webhook": wh}, None, "", True)

    def list_webhooks(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        whs = list(self._state(session_id)["webhooks"].values())
        return _result(True, {"webhooks": whs, "count": len(whs)}, None, "", False)

    def delete_webhook(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id); wid = arguments["webhook_id"]
        if wid not in state["webhooks"]: raise KeyError(f"webhook not found: {wid}")
        state["webhooks"][wid]["active"] = False
        return _result(True, {"webhook_id": wid, "deleted": True}, None, "", True)


if __name__ == "__main__":
    serve(PaymentsServer())
