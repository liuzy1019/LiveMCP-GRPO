"""Payments state facts audited against ``servers/payments/server.py``."""

from src.live_mcp.domain_contracts.states.common import arg, facts, out


PAYMENTS_STATE_FACTS = {
    "create_invoice": facts(post=(
        out("invoice", "invoice_id", "invoice.exists"),
        out("invoice", "invoice_id", "invoice.status", "pending"),
        out("invoice", "invoice_id", "invoice.payment_linked", False),
        out("invoice", "invoice_id", "invoice.dispute_open", False),
        out("invoice", "invoice_id", "invoice.payable", True),
        out("invoice", "invoice_id", "invoice.refundable", False),
        out("invoice", "invoice_id", "invoice.disputable", True),
    )),
    "get_invoice": facts(pre=(arg("invoice", "invoice_id", "invoice.exists"),)),
    "list_invoices": facts(),
    "pay_invoice": facts(
        pre=(
            arg("invoice", "invoice_id", "invoice.exists"),
            arg("invoice", "invoice_id", "invoice.payable"),
            arg("invoice", "invoice_id", "invoice.payment_linked", False),
            arg("invoice", "invoice_id", "invoice.dispute_open", False),
        ),
        post=(
            arg("invoice", "invoice_id", "invoice.status", "paid"),
            arg("invoice", "invoice_id", "invoice.payment_linked", True),
            arg("invoice", "invoice_id", "invoice.payable", False),
            arg("invoice", "invoice_id", "invoice.refundable", True),
            arg("invoice", "invoice_id", "invoice.payment_settled", True),
            out("payment", "payment_id", "payment.exists"),
            out("payment", "payment_id", "payment.status", "settled"),
        ),
    ),
    "refund_invoice": facts(
        pre=(
            arg("invoice", "invoice_id", "invoice.exists"),
            arg("invoice", "invoice_id", "invoice.refundable"),
            arg("invoice", "invoice_id", "invoice.payment_settled"),
        ),
        post=(out("refund", "refund_id", "refund.exists"),),
    ),
    "cancel_payment": facts(
        pre=(
            arg("payment", "payment_id", "payment.exists"),
            arg("payment", "payment_id", "payment.status", "pending"),
        ),
        post=(arg("payment", "payment_id", "payment.status", "cancelled"),),
    ),
    "dispute_invoice": facts(
        pre=(
            arg("invoice", "invoice_id", "invoice.exists"),
            arg("invoice", "invoice_id", "invoice.disputable"),
        ),
        post=(
            arg("invoice", "invoice_id", "invoice.status", "disputed"),
            arg("invoice", "invoice_id", "invoice.dispute_open", True),
            out("dispute", "dispute_id", "dispute.exists"),
        ),
    ),
    "create_webhook": facts(post=(out("webhook", "webhook_id", "webhook.exists"),)),
    "list_webhooks": facts(),
    "delete_webhook": facts(
        pre=(arg("webhook", "webhook_id", "webhook.exists"),),
        post=(arg("webhook", "webhook_id", "webhook.exists", False),),
    ),
}
