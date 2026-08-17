"""Declarative constraints for dependency-chain eligibility.

These facts describe domain state semantics.  The generic evaluator lives in
``dependency_chain_policy``; generation code must not branch on domain names.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ChainConstraint:
    kind: str
    targets: frozenset[str]
    sources: frozenset[str] = frozenset()
    bridges: frozenset[str] = frozenset()
    code: str = "chain_contract_violation"


_DOMAIN_CHAIN_CONSTRAINTS: dict[str, tuple[ChainConstraint, ...]] = {
    "banking": (
        ChainConstraint(
            "forbidden_order", frozenset({"cancel_transfer"}),
            frozenset({"schedule_transfer"}),
            code="synthetic_scheduled_transfer_reversal",
        ),
    ),
    "calendar": (
        ChainConstraint(
            "forbidden_order", frozenset({"get_recurring_info"}),
            frozenset({"create_event"}), code="new_event_is_not_recurring",
        ),
    ),
    "crm": (
        ChainConstraint(
            "forbidden_order", frozenset({"delete_contact"}),
            frozenset({"convert_lead"}),
            code="converted_lead_retains_contact_reference",
        ),
    ),
    "payments": (
        ChainConstraint(
            "forbidden_order", frozenset({"cancel_payment"}),
            frozenset({"pay_invoice"}), code="settled_payment_cannot_be_cancelled",
        ),
        ChainConstraint(
            "require_between", frozenset({"refund_invoice"}),
            frozenset({"create_invoice"}), frozenset({"pay_invoice"}),
            "new_invoice_must_be_paid_before_refund",
        ),
    ),
    "team_chat": (
        ChainConstraint(
            "require_between", frozenset({"create_thread", "react_message"}),
            frozenset({"create_channel"}), frozenset({"send_message"}),
            "new_channel_needs_message_before_message_operation",
        ),
    ),
    "shopping": (
        ChainConstraint(
            "requires_predecessor", frozenset({"add_review"}),
            frozenset({"get_order", "return_order", "get_return_status"}),
            code="review_has_no_order_lifecycle_source",
        ),
        ChainConstraint(
            "forbidden_prior", frozenset({"add_review"}),
            frozenset({"checkout"}), code="checkout_does_not_establish_review_eligibility",
        ),
        ChainConstraint(
            "forbidden_order", frozenset({"get_product", "get_reviews"}),
            frozenset({"return_order"}),
            code="return_does_not_establish_product_read_result",
        ),
        ChainConstraint(
            "forbidden_cooccurrence",
            frozenset({"return_order", "get_return_status"}),
            frozenset({"add_to_cart", "checkout", "cancel_order"}),
            code="chain_mixes_return_and_purchase_lifecycles",
        ),
        ChainConstraint(
            "requires_predecessor",
            frozenset({"remove_from_cart", "update_cart_quantity"}),
            frozenset({"get_cart"}), code="cart_membership_not_observed",
        ),
        ChainConstraint(
            "requires_predecessor", frozenset({"remove_from_wishlist"}),
            frozenset({"get_wishlist"}), code="wishlist_membership_not_observed",
        ),
        ChainConstraint(
            "forbidden_predecessor", frozenset({"add_to_wishlist"}),
            frozenset({"get_wishlist"}), code="wishlist_membership_already_exists",
        ),
        ChainConstraint(
            "forbidden_order",
            frozenset({"remove_from_cart", "update_cart_quantity"}),
            frozenset({"add_to_cart"}), code="synthetic_cart_reversal",
        ),
        ChainConstraint(
            "forbidden_order", frozenset({"remove_from_wishlist"}),
            frozenset({"add_to_wishlist"}), code="synthetic_wishlist_reversal",
        ),
        ChainConstraint(
            "forbidden_order", frozenset({"return_order"}),
            frozenset({"checkout", "get_return_status"}),
            code="return_state_precondition_not_met",
        ),
        ChainConstraint(
            "require_between",
            frozenset({"remove_from_cart", "update_cart_quantity"}),
            frozenset({"checkout", "clear_cart"}), frozenset({"add_to_cart"}),
            "cart_must_be_repopulated_after_clear",
        ),
    ),
    "food_delivery": (
        ChainConstraint(
            "forbidden_order", frozenset({"reorder"}),
            frozenset({"create_order"}),
            code="new_order_cannot_be_reordered_as_history",
        ),
        ChainConstraint(
            "require_between", frozenset({"track_rider"}),
            frozenset({"create_order", "reorder"}),
            frozenset({"update_order_status"}),
            "new_order_needs_status_transition_before_tracking",
        ),
        ChainConstraint(
            "forbidden_order", frozenset({"rate_order"}),
            frozenset({"create_order", "reorder"}),
            code="new_order_cannot_reach_delivered_within_chain_budget",
        ),
        ChainConstraint(
            "forbidden_order",
            frozenset({"rate_order", "track_rider", "update_order_status"}),
            frozenset({"cancel_order"}), code="cancelled_order_is_terminal",
        ),
    ),
}
