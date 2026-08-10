"""Deterministic state builder for payments."""

from __future__ import annotations

import datetime as _datetime
import random
from typing import Any

from src.live_mcp.state_seeders.common import (
    _reference_datetime,
    _sample_entities,
    _seed_scoped_id,
)
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

def _payments_state(
    seed: int,
    state_profile: str = "baseline",
) -> dict[str, Any]:
    rng = random.Random(seed)
    reference_date = _reference_datetime(seed).date()
    selected = _sample_entities(rng, _PAYMENTS_INVOICE_TEMPLATES, target_count=20, id_prefix="inv")
    invoices = {}
    for idx, (_iid, customer, amount, currency, desc, status) in enumerate(selected):
        iid = _seed_scoped_id("inv", seed, idx, width=4)
        pay_id = _seed_scoped_id("pay", seed, idx, width=4) if status == "paid" else None
        due_offset = -(idx % 20 + 1) if status == "overdue" else idx % 20 + 1
        due_date = reference_date + _datetime.timedelta(days=due_offset)
        invoices[iid] = {
            "invoice_id": iid, "customer": customer, "amount": round(amount + rng.randint(-50, 50), 2),
            "currency": currency, "description": desc, "status": status,
            "payment_id": pay_id,
            "refund_id": None,
            "due_date": due_date.isoformat(),
            "created_at": (due_date - _datetime.timedelta(days=30)).isoformat(),
        }
    payments = {}
    pending_payment_assigned = False
    for iid, inv in invoices.items():
        if not inv.get("payment_id"):
            continue
        status = "settled"
        if not pending_payment_assigned:
            status = "pending"
            pending_payment_assigned = True
            # Invoice and payment lifecycle must agree: a payment that has not
            # settled cannot make its invoice refundable as "paid".
            inv["status"] = "pending"
        payments[inv["payment_id"]] = {
            "payment_id": inv["payment_id"],
            "invoice_id": iid,
            "amount": inv["amount"],
            "method": "wire",
            "status": status,
        }
    webhook_id = _seed_scoped_id("wh", seed, 0, width=3)
    webhook_id_2 = _seed_scoped_id("wh", seed, 1, width=3)
    webhook_id_3 = _seed_scoped_id("wh", seed, 2, width=3)
    webhooks = {
        webhook_id: {
            "webhook_id": webhook_id,
            "url": "https://example.com/hooks/payments",
            "events": ["invoice.paid", "invoice.refunded"],
            "active": True,
        },
        webhook_id_2: {
            "webhook_id": webhook_id_2,
            "url": "https://example.com/hooks/refunds",
            "events": ["invoice.refunded"],
            "active": True,
        },
        webhook_id_3: {
            "webhook_id": webhook_id_3,
            "url": "https://example.com/hooks/failures",
            "events": ["payment.failed", "payment.cancelled"],
            "active": False,
        },
    }
    # Seed refunds (up to 3) so refund_invoice chains are feasible.
    settled_payments = [
        (pid, p) for pid, p in payments.items() if p["status"] == "settled"
    ]
    refunds = {}
    for i, (settled_pid, settled_payment) in enumerate(settled_payments[:3]):
        inv = invoices[settled_payment["invoice_id"]]
        refund_id = _seed_scoped_id("ref", seed, i, width=4)
        refund_amount = round(max(0.01, inv["amount"] / 4), 2)
        inv["status"] = "partially_refunded"
        inv["refund_id"] = refund_id
        inv["total_refunded"] = refund_amount
        refunds[refund_id] = {
            "refund_id": refund_id,
            "invoice_id": inv["invoice_id"],
            "amount": refund_amount,
            "reason": "partial service credit",
        }
    state = {"invoices": invoices,
             "payments": payments,
             "refunds": refunds, "webhooks": webhooks, "disputes": {},
             "next_inv_num": len(invoices) + 1, "next_pay_num": len(payments) + 1,
             "next_ref_num": len(refunds) + 1, "next_wh_num": 4,
             "current_date": reference_date.isoformat()}
    if state_profile == "baseline":
        return state
    if state_profile != "payments_rare_state_v1":
        raise ValueError(f"unsupported payments state profile: {state_profile}")

    # Keep the invoice count fixed. The profile enriches executable lifecycle
    # states that can unlock cancel/refund/webhook paths, rather than adding
    # more interchangeable invoice IDs.
    unlinked_pending = [
        inv for inv in invoices.values()
        if inv["status"] == "pending" and not inv.get("payment_id")
    ]
    for offset, inv in enumerate(unlinked_pending[:3], start=len(invoices)):
        payment_id = _seed_scoped_id("pay", seed, offset, width=4)
        inv["payment_id"] = payment_id
        payments[payment_id] = {
            "payment_id": payment_id,
            "invoice_id": inv["invoice_id"],
            "amount": inv["amount"],
            "method": "card",
            "status": "pending",
        }

    settled = [
        payment for payment in payments.values()
        if payment["status"] == "settled"
    ]
    if settled:
        payment = settled[0]
        inv = invoices[payment["invoice_id"]]
        refund_id = _seed_scoped_id("ref", seed, 0, width=4)
        refund_amount = round(max(0.01, inv["amount"] / 4), 2)
        inv["status"] = "partially_refunded"
        inv["refund_id"] = refund_id
        inv["total_refunded"] = refund_amount
        state["refunds"][refund_id] = {
            "refund_id": refund_id,
            "invoice_id": inv["invoice_id"],
            "amount": refund_amount,
            "reason": "partial service credit",
        }

    webhook_specs = [
        (
            "https://example.com/hooks/refunds",
            ["invoice.refunded"],
            True,
        ),
        (
            "https://example.com/hooks/failures",
            ["payment.failed", "payment.cancelled"],
            False,
        ),
    ]
    for offset, (url, events, active) in enumerate(webhook_specs, start=1):
        webhook_id = _seed_scoped_id("wh", seed, offset, width=3)
        webhooks[webhook_id] = {
            "webhook_id": webhook_id,
            "url": url,
            "events": events,
            "active": active,
        }
    state["next_pay_num"] = len(payments) + 1
    state["next_ref_num"] = len(state["refunds"]) + 1
    state["next_wh_num"] = len(webhooks) + 1
    return state
