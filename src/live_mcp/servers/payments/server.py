"""Stateful payments server with 10 tools.
Transactional state: invoices, payments, refunds, webhooks, disputes.
Safety: double-payment, refund-without-payment, dispute constraints.
"""

from __future__ import annotations
from typing import Any
from src.live_mcp.server_base import StatefulToolServer, _result, serve

TOOLS = [
    {"name": "create_invoice", "description": "Create a new invoice with a positive amount.", "input_schema": {"type": "object", "properties": {"customer": {"type": "string"}, "amount": {"type": "number", "exclusiveMinimum": 0, "description": "Positive invoice amount; must be greater than zero."}, "currency": {"type": "string"}, "description": {"type": "string"}, "due_date": {"type": "string"}}, "required": ["customer", "amount"]}, "annotations": {"mutating": True}},
    {"name": "get_invoice", "description": "Get invoice details.", "input_schema": {"type": "object", "properties": {"invoice_id": {"type": "string"}}, "required": ["invoice_id"]}, "annotations": {"readonly": True, "mutating": False}},
    {"name": "list_invoices", "description": "List invoices by status, customer, or date range.", "input_schema": {"type": "object", "properties": {"status": {"type": "string"}, "customer": {"type": "string"}, "from_date": {"type": "string"}, "to_date": {"type": "string"}}, "required": []}, "annotations": {"readonly": True, "mutating": False}},
    {"name": "pay_invoice", "description": "Pay an unpaid pending or overdue invoice for exactly its full invoice amount. The invoice must have no linked payment or open dispute and must not be paid, disputed, refunded, or partially refunded. Returns a settled payment with a payment_id. Sensitive param on amount.", "input_schema": {"type": "object", "properties": {"invoice_id": {"type": "string"}, "amount": {"type": "number", "exclusiveMinimum": 0, "description": "Positive amount exactly equal to the invoice amount."}, "method": {"type": "string"}}, "required": ["invoice_id", "amount"]}, "annotations": {"mutating": True, "sensitive_params": ["invoice_id", "amount"]}},
    {"name": "refund_invoice", "description": "Refund a paid or partially_refunded invoice whose linked payment is settled. Amount must be positive and no greater than the invoice's remaining refundable amount.", "input_schema": {"type": "object", "properties": {"invoice_id": {"type": "string"}, "amount": {"type": "number", "exclusiveMinimum": 0, "description": "Positive amount no greater than the remaining refundable amount."}, "reason": {"type": "string"}}, "required": ["invoice_id", "amount"]}, "annotations": {"mutating": True, "sensitive_params": ["invoice_id", "amount"]}},
    {"name": "cancel_payment", "description": "Cancel an existing pending payment before settlement. Requires a payment_id (pay_...), not an invoice_id. Settled or refunded payments cannot be cancelled.", "input_schema": {"type": "object", "properties": {"payment_id": {"type": "string"}, "reason": {"type": "string"}}, "required": ["payment_id"]}, "annotations": {"mutating": True}},
    {"name": "dispute_invoice", "description": "File a dispute on an invoice only when its current status is paid or pending. Requires a non-empty reason.", "input_schema": {"type": "object", "properties": {"invoice_id": {"type": "string"}, "reason": {"type": "string"}, "evidence": {"type": "string"}}, "required": ["invoice_id", "reason"]}, "annotations": {"mutating": True}},
    {"name": "create_webhook", "description": "Register a webhook endpoint.", "input_schema": {"type": "object", "properties": {"url": {"type": "string"}, "events": {"type": "array"}}, "required": ["url", "events"]}, "annotations": {"mutating": True}},
    {"name": "list_webhooks", "description": "List registered webhooks.", "input_schema": {"type": "object", "properties": {}, "required": []}, "annotations": {"readonly": True, "mutating": False}},
    {"name": "delete_webhook", "description": "Delete a webhook registration.", "input_schema": {"type": "object", "properties": {"webhook_id": {"type": "string"}}, "required": ["webhook_id"]}, "annotations": {"mutating": True}},
]

class PaymentsServer(StatefulToolServer):
    def __init__(self) -> None:
        super().__init__("payments", TOOLS)
        self.handlers = {t["name"]: getattr(self, t["name"]) for t in TOOLS}

    def create_invoice(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id)
        amount = float(arguments["amount"])
        if amount <= 0: raise KeyError("amount must be positive")
        inv_id = f"inv_{state['next_inv_num']:04d}"; state["next_inv_num"] += 1
        inv = {"invoice_id": inv_id, "customer": arguments["customer"], "amount": amount, "currency": arguments.get("currency", "USD"), "description": arguments.get("description", ""), "due_date": arguments.get("due_date", ""), "status": "pending", "payment_id": None, "refund_id": None, "created_at": state["current_date"]}
        state["invoices"][inv_id] = inv
        return _result(True, {"invoice": inv}, None, "", True)

    def get_invoice(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id)
        inv = state["invoices"].get(arguments["invoice_id"])
        if not inv: raise KeyError(f"invoice not found: {arguments['invoice_id']}")
        visible = dict(inv)
        if inv.get("payment_id") and inv["payment_id"] in state["payments"]:
            visible["payment_status"] = state["payments"][inv["payment_id"]]["status"]
        return _result(True, {"invoice": visible}, None, "", False)

    def list_invoices(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id)
        invs = []
        for inv in state["invoices"].values():
            visible = dict(inv)
            if inv.get("payment_id") and inv["payment_id"] in state["payments"]:
                visible["payment_status"] = state["payments"][inv["payment_id"]]["status"]
            invs.append(visible)
        st = arguments.get("status"); cust = arguments.get("customer"); fd = arguments.get("from_date"); td = arguments.get("to_date")
        if st: invs = [i for i in invs if i["status"] == st]
        if cust: invs = [i for i in invs if i["customer"] == cust]
        if fd: invs = [i for i in invs if i.get("created_at", "") >= fd]
        if td: invs = [i for i in invs if i.get("created_at", "") <= td]
        return _result(True, {"invoices": invs, "count": len(invs)}, None, "", False)

    def pay_invoice(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id); inv_id = arguments["invoice_id"]; inv = state["invoices"].get(inv_id)
        if not inv: raise KeyError(f"invoice not found: {inv_id}")
        if inv["status"] not in ("pending", "overdue"):
            raise KeyError(f"cannot pay invoice in status: {inv['status']}")
        open_dispute = next(
            (
                dispute for dispute in state.get("disputes", {}).values()
                if dispute.get("invoice_id") == inv_id
                and dispute.get("status") == "open"
            ),
            None,
        )
        if open_dispute is not None:
            raise KeyError(
                f"invoice has open dispute: {open_dispute.get('dispute_id', 'unknown')}"
            )
        if inv.get("payment_id"):
            linked = state["payments"].get(inv["payment_id"])
            linked_status = linked.get("status") if linked else "missing"
            raise KeyError(f"invoice already has linked payment in status: {linked_status}")
        amount = float(arguments["amount"])
        if amount <= 0: raise KeyError("amount must be positive")
        if abs(amount - inv["amount"]) > 0.01: raise KeyError(f"amount mismatch: {amount} vs {inv['amount']}")
        method = arguments.get("method", "card"); pid = f"pay_{state['next_pay_num']:04d}"; state["next_pay_num"] += 1
        inv["status"] = "paid"; inv["payment_id"] = pid
        state["payments"][pid] = {"payment_id": pid, "invoice_id": inv_id, "amount": amount, "method": method, "status": "settled"}
        return _result(True, {"invoice": inv, "payment": state["payments"][pid]}, None, "", True)

    def refund_invoice(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id); inv_id = arguments["invoice_id"]; inv = state["invoices"].get(inv_id)
        if not inv: raise KeyError(f"invoice not found: {inv_id}")
        if inv["status"] not in ("paid", "partially_refunded"): raise KeyError(f"cannot refund invoice in status: {inv['status']}")
        payment_id = inv.get("payment_id")
        payment = state["payments"].get(payment_id) if payment_id else None
        if payment is None: raise KeyError("cannot refund invoice without a linked payment")
        if payment.get("status") != "settled":
            raise KeyError(f"cannot refund payment in status: {payment.get('status')}")
        amount = float(arguments["amount"])
        if amount <= 0: raise KeyError("amount must be positive")
        if amount > inv["amount"]: raise KeyError(f"refund exceeds invoice: {amount} > {inv['amount']}")
        # Track cumulative refunds to prevent over-refunding
        total_refunded = inv.get("total_refunded", 0.0)
        if amount + total_refunded > inv["amount"]:
            raise KeyError(f"cumulative refunds ({amount} + {total_refunded}) exceed invoice amount {inv['amount']}")
        rid = f"ref_{state['next_ref_num']:04d}"; state["next_ref_num"] += 1
        new_total = round(total_refunded + amount, 2)
        inv["status"] = "refunded" if abs(new_total - inv["amount"]) <= 0.01 else "partially_refunded"
        inv["refund_id"] = rid; inv["total_refunded"] = new_total
        state["refunds"][rid] = {"refund_id": rid, "invoice_id": inv_id, "amount": amount, "reason": arguments.get("reason", "")}
        return _result(True, {"invoice": inv, "refund": state["refunds"][rid]}, None, "", True)

    def cancel_payment(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id); pid = arguments["payment_id"]; pmt = state["payments"].get(pid)
        if not pmt: raise KeyError(f"payment not found: {pid}")
        if pmt["status"] == "settled": raise KeyError("cannot cancel settled payment")
        if pmt["status"] != "pending": raise KeyError(f"payment already {pmt['status']}")
        inv = state["invoices"][pmt["invoice_id"]]
        if inv.get("refund_id"): raise KeyError(f"cannot cancel refunded invoice: {inv['refund_id']}")
        pmt["status"] = "cancelled"; pmt["cancel_reason"] = arguments.get("reason", "")
        inv["status"] = "pending"; inv["payment_id"] = None
        return _result(True, {"payment": pmt, "invoice_status": "pending"}, None, "", True)

    def dispute_invoice(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id); inv_id = arguments["invoice_id"]; inv = state["invoices"].get(inv_id)
        if not inv: raise KeyError(f"invoice not found: {inv_id}")
        if inv["status"] not in ("paid", "pending"): raise KeyError(f"cannot dispute invoice in status: {inv['status']}")
        if not arguments["reason"].strip(): raise KeyError("reason must be non-empty")
        did = f"dis_{state['next_inv_num']:04d}"; state["next_inv_num"] += 1
        dispute = {"dispute_id": did, "invoice_id": inv_id, "reason": arguments["reason"], "evidence": arguments.get("evidence", ""), "status": "open"}
        state.setdefault("disputes", {})[did] = dispute; inv["status"] = "disputed"
        return _result(True, {"dispute": dispute, "invoice_status": "disputed"}, None, "", True)

    def create_webhook(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if not arguments["url"].strip(): raise KeyError("url must be non-empty")
        if not arguments["events"]: raise KeyError("events must be non-empty")
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
        deleted = state["webhooks"].pop(wid)
        return _result(
            True,
            {"webhook_id": wid, "deleted": True, "webhook": deleted},
            None,
            "",
            True,
        )


if __name__ == "__main__":
    serve(PaymentsServer())
