from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.live_mcp.corpus.merge_dedup import _write_semantic_quarantine_report
from src.live_mcp.corpus.semantic_quarantine import (
    evaluate_semantic_quarantine,
)
from src.live_mcp.registry.environment_metadata import validate_semantic_gate_evidence
from src.live_mcp.servers.payments.server import TOOLS as PAYMENT_TOOLS
from src.live_mcp.protocol.observation import TRAJECTORY_SCHEMA_VERSION
from src.live_mcp.generation.irrelevance import (
    IRRELEVANCE_PROOF_VERSION,
    _tool_inventory_sha256,
)


def _terminal(text: str, action: str = "final_answer") -> dict:
    return {
        "action": action,
        "tool_name": action,
        "arguments": {"text": text},
    }


def _event(
    tool_name: str,
    *,
    arguments: dict | None = None,
    observation: dict | None = None,
    state_changed: bool = False,
) -> dict:
    return {
        "tool_name": tool_name,
        "arguments": arguments or {},
        "observation": observation or {},
        "success": True,
        "state_changed": state_changed,
    }


def _extra(query: str, history: list[dict], terminal_text: str) -> dict:
    return {
        "domain": "shopping",
        "task_id": "shopping-semantic-test",
        "teacher_round_trace": [{
            "round_idx": 0,
            "user_query": query,
            "execution_history": history,
            "oracle_calls": [
                {
                    "action": "tool_call",
                    "tool_name": event["tool_name"],
                    "arguments": event.get("arguments", {}),
                }
                for event in history
            ] + [_terminal(terminal_text)],
        }],
    }


def _executed_amount_extra(*, amount: int) -> dict:
    query = "Create an invoice for Acme for 1500 dollars."
    history = [_event(
        "create_invoice",
        arguments={"customer": "Acme", "amount": amount},
        observation={"invoice": {"customer": "Acme", "amount": amount}},
        state_changed=True,
    )]
    return {
        "domain": "payments",
        "trajectory_schema_version": TRAJECTORY_SCHEMA_VERSION,
        "continuation_goal_specs": [],
        "success_criteria": [],
        "success_criteria_provenance": [],
        "teacher_round_trace": [{
            "round_idx": 0,
            "user_query": query,
            "execution_history": history,
            "oracle_calls": [{
                "action": "tool_call",
                "tool_name": "create_invoice",
                "arguments": {"customer": "Acme", "amount": amount},
            }, _terminal("The invoice was created.")],
        }],
    }


def test_generic_semantic_gate_accepts_executed_user_amount() -> None:
    assert evaluate_semantic_quarantine(_executed_amount_extra(amount=1500)) is None


def test_generic_semantic_gate_does_not_claim_argument_provenance() -> None:
    issue = evaluate_semantic_quarantine(_executed_amount_extra(amount=1600))

    assert issue is None


def test_irrelevance_requires_capability_inventory_proof() -> None:
    query = "Can you give me the live sports score for tonight's game?"
    extra = {
        "domain": "payments",
        "prompt_profile": "local_trainable_v1",
        "trajectory_schema_version": TRAJECTORY_SCHEMA_VERSION,
        "user_query": query,
        "clean_visible_tools": PAYMENT_TOOLS,
        "irrelevance_capability_proof": {
            "proof_version": IRRELEVANCE_PROOF_VERSION,
            "unavailable_capability_class": "live_sports_score",
            "query_evidence_span": "live sports score",
            "available_tool_inventory_sha256": _tool_inventory_sha256(
                PAYMENT_TOOLS
            ),
        },
        "teacher_round_trace": [{
            "round_idx": 0,
            "user_query": query,
            "execution_history": [],
            "oracle_calls": [_terminal("I cannot provide live sports scores.", "report_error")],
        }],
    }

    assert evaluate_semantic_quarantine(extra) is None
    extra["irrelevance_capability_proof"]["query_evidence_span"] = "tonight"
    issue = evaluate_semantic_quarantine(extra)
    assert issue is not None
    assert issue.reason_code == "irrelevance_query_evidence_class_mismatch"


def test_rejects_social_only_continuation_and_visible_id_reask() -> None:
    social = _extra("Find the K3 Keyboard.", [], "Found it.")
    social["teacher_round_trace"].append({
        "round_idx": 1,
        "user_query": "Actually, I'm all set. Thanks again!",
        "execution_history": [],
        "oracle_calls": [_terminal("You're welcome.")],
    })
    issue = evaluate_semantic_quarantine(social)
    assert issue is not None
    assert issue.reason_code == "shopping_non_actionable_social_continuation"

    reask = _extra(
        "Buy the K3 Keyboard.",
        [_event("checkout", arguments={
            "shipping_address": "123 Maple St",
            "payment_method": "Visa",
        }, observation={"order": {"order_id": "ord_s1_0003"}}, state_changed=True)],
        "Your order ID is ord_s1_0003.",
    )
    reask["teacher_round_trace"].append({
        "round_idx": 1,
        "user_query": "What is the order ID for my K3 Keyboard?",
        "execution_history": [_event("list_orders", observation={"orders": []})],
        "oracle_calls": [_terminal("It is ord_s1_0003.")],
    })
    issue = evaluate_semantic_quarantine(reask)
    assert issue is not None
    assert issue.reason_code == "terminal_exposes_private_entity_id"
    assert issue.hard_gate is True


@pytest.mark.parametrize(("domain", "private_id"), [
    ("banking", "acc_s43_004"),
    ("calendar", "evt_s43_004"),
    ("crm", "deal_s43_004"),
    ("email", "email_s43_004"),
    ("filesystem", "file_s43_004"),
    ("food_delivery", "ord_s43_004"),
    ("issue_tracker", "iss_s43_004"),
    ("payments", "inv_s43_004"),
    ("shopping", "ord_s43_004"),
    ("team_chat", "ch_s43_004"),
])
def test_rejects_sampler_private_id_in_initial_query_for_all_domains(
    domain: str, private_id: str,
) -> None:
    extra = {
        "domain": domain,
        "task_id": f"{domain}-private-id",
        "teacher_round_trace": [{
            "round_idx": 0,
            "user_query": f"Act on {private_id} now.",
            "execution_history": [],
            "oracle_calls": [_terminal(
                "Please provide the authorization code.",
                action="ask_clarification",
            )],
        }],
    }

    issue = evaluate_semantic_quarantine(extra)

    assert issue is not None
    assert issue.reason_code == "initial_query_exposes_private_entity_id"
    assert issue.trace_evidence["leaked_ids"] == [private_id]


def test_calendar_rejects_invitation_delivery_claim_without_observation() -> None:
    extra = {
        "domain": "calendar",
        "teacher_round_trace": [{
            "round_idx": 0,
            "user_query": "Move the post-mortem to Monday.",
            "execution_history": [_event(
                "update_event",
                observation={"event": {"event_id": "evt_1"}},
                state_changed=True,
            )],
            "oracle_calls": [_terminal(
                "The event was moved and updated invites have been sent."
            )],
        }],
    }

    issue = evaluate_semantic_quarantine(extra)

    assert issue is not None
    assert issue.reason_code == (
        "calendar_invitation_delivery_claim_without_evidence"
    )


def test_calendar_allows_invitation_delivery_claim_with_observation() -> None:
    extra = {
        "domain": "calendar",
        "teacher_round_trace": [{
            "round_idx": 0,
            "user_query": "Move the post-mortem to Monday.",
            "execution_history": [_event(
                "update_event",
                observation={
                    "event": {"event_id": "evt_1"},
                    "attendee_notifications_sent": True,
                },
                state_changed=True,
            )],
            "oracle_calls": [_terminal(
                "The event was moved and updated invites have been sent."
            )],
        }],
    }

    assert evaluate_semantic_quarantine(extra) is None


def test_paper_profile_accepts_real_id_grounded_by_prove_sampling_context() -> None:
    extra = {
        "domain": "banking",
        "prompt_profile": "paper_generation_baseline_v1",
        "semantic_gate_profile": "diagnostic_only",
        "teacher_round_trace": [{
            "round_idx": 0,
            "user_query": "Unfreeze account acc_s43_004 now.",
            "execution_history": [],
            "oracle_calls": [_terminal(
                "Please provide the authorization code.",
                action="ask_clarification",
            )],
        }],
    }

    assert evaluate_semantic_quarantine(extra) is None


def test_rejects_sampler_private_id_in_terminal_and_continuation_query() -> None:
    terminal_leak = {
        "domain": "banking",
        "teacher_round_trace": [{
            "round_idx": 0,
            "user_query": "Show my business account details.",
            "execution_history": [],
            "oracle_calls": [_terminal(
                "Your business account ID is acc_s42_003."
            )],
        }],
    }
    issue = evaluate_semantic_quarantine(terminal_leak)
    assert issue is not None
    assert issue.reason_code == "terminal_exposes_private_entity_id"
    assert issue.round_idx == 0
    assert issue.trace_evidence["leaked_ids"] == ["acc_s42_003"]

    continuation_leak = {
        "domain": "banking",
        "teacher_round_trace": [{
            "round_idx": 0,
            "user_query": "Show my business account details.",
            "execution_history": [],
            "oracle_calls": [_terminal("Your business account is active.")],
        }, {
            "round_idx": 1,
            "user_query": "Show the history for acc_s42_003.",
            "execution_history": [],
            "oracle_calls": [_terminal("Here are the recent transactions.")],
        }],
    }
    issue = evaluate_semantic_quarantine(continuation_leak)
    assert issue is not None
    assert issue.reason_code == "continuation_query_exposes_private_entity_id"
    assert issue.round_idx == 1


def test_allows_runtime_created_business_reference_after_reload() -> None:
    extra = {
        "domain": "payments",
        "prompt_profile": "local_trainable_v1",
        "semantic_gate_profile": "deterministic_v1",
        "clean_visible_tools": PAYMENT_TOOLS,
        "teacher_round_trace": [{
            "round_idx": 0,
            "user_query": "Bill Acme $1,500.",
            "execution_history": [_event(
                "create_invoice",
                arguments={"customer": "Acme", "amount": 1500},
                observation={
                    "invoice": {
                        "invoice_id": "inv_0021", "customer": "Acme",
                    },
                },
                state_changed=True,
            )],
            "oracle_calls": [
                {
                    "action": "tool_call",
                    "tool_name": "create_invoice",
                    "arguments": {"customer": "Acme", "amount": 1500},
                },
                _terminal("Invoice inv_0021 was created for Acme."),
            ],
        }],
    }

    issue = evaluate_semantic_quarantine(extra)

    assert issue is None


def test_allows_runtime_private_entity_id_only_inside_tool_trace() -> None:
    extra = {
        "domain": "payments",
        "prompt_profile": "local_trainable_v1",
        "semantic_gate_profile": "deterministic_v1",
        "clean_visible_tools": PAYMENT_TOOLS,
        "teacher_round_trace": [{
            "round_idx": 0,
            "user_query": "Pay the Acme invoice.",
            "execution_history": [_event(
                "pay_invoice",
                arguments={
                    "invoice_id": "inv_0021", "amount": 1500,
                    "method": "card",
                },
                observation={
                    "invoice": {"invoice_id": "inv_0021", "status": "paid"},
                },
                state_changed=True,
            )],
            "oracle_calls": [
                {
                    "action": "tool_call",
                    "tool_name": "pay_invoice",
                    "arguments": {
                        "invoice_id": "inv_0021", "amount": 1500,
                        "method": "card",
                    },
                },
                _terminal("The Acme invoice was paid."),
            ],
        }],
    }

    assert evaluate_semantic_quarantine(extra) is None


def test_rejects_clarification_after_prior_round_fixed_order_focus() -> None:
    extra = _extra(
        "What is the status of my order for the Noise Canceling Headphones?",
        [_event(
            "get_order",
            arguments={"order_id": "ord_pending"},
            observation={"order": {
                "order_id": "ord_pending",
                "status": "pending",
            }},
        )],
        "Your Noise Canceling Headphones order (ord_pending) is pending.",
    )
    extra["teacher_round_trace"].append({
        "round_idx": 1,
        "user_query": "Can you track this order for me?",
        "execution_history": [],
        "oracle_calls": [_terminal(
            "Which order do you mean: ord_pending or ord_returning?",
            action="ask_clarification",
        )],
    })

    issue = evaluate_semantic_quarantine(extra)
    assert issue is not None
    assert issue.reason_code == "shopping_reasks_resolved_order_focus"


def test_allows_order_clarification_without_unique_prior_focus() -> None:
    extra = _extra(
        "Show my orders.",
        [_event("list_orders", observation={"orders": [
            {"order_id": "ord_one"}, {"order_id": "ord_two"},
        ]})],
        "You have orders ord_one and ord_two.",
    )
    extra["teacher_round_trace"].append({
        "round_idx": 1,
        "user_query": "Can you track the order for me?",
        "execution_history": [],
        "oracle_calls": [_terminal(
            "Which order would you like me to track?",
            action="ask_clarification",
        )],
    })

    assert evaluate_semantic_quarantine(extra) is None


def test_rejects_generic_checkout_payment_but_accepts_concrete_method() -> None:
    generic = _extra(
        "Checkout to 123 Maple St using my credit card.",
        [_event("checkout", arguments={
            "shipping_address": "123 Maple St",
            "payment_method": "credit card",
        }, observation={"order": {"order_id": "ord_s1_0003"}}, state_changed=True)],
        "Order placed.",
    )
    issue = evaluate_semantic_quarantine(generic)
    assert issue is not None
    assert issue.reason_code == "shopping_checkout_generic_payment_method"

    concrete = _extra(
        "Checkout to 123 Maple St using Visa.",
        [_event("checkout", arguments={
            "shipping_address": "123 Maple St",
            "payment_method": "Visa",
        }, observation={"order": {"order_id": "ord_s1_0003"}}, state_changed=True)],
        "Order placed.",
    )
    assert evaluate_semantic_quarantine(concrete) is None


def test_review_coverage_accumulates_prior_round_evidence() -> None:
    supported = _extra(
        "Show reviews for the Noise Canceling Headphones.",
        [_event(
            "get_reviews",
            arguments={"product_id": "prd_s1_003"},
            observation={"product_id": "prd_s1_003", "reviews": []},
        )],
        "There are no reviews.",
    )
    supported["teacher_round_trace"].append({
        "round_idx": 1,
        "user_query": "Are there any other headphones with reviews?",
        "execution_history": [_event(
            "search_products",
            observation={"products": [{"product_id": "prd_s1_003"}]},
        )],
        "oracle_calls": [_terminal("There are no other headphones with reviews.")],
    })
    assert evaluate_semantic_quarantine(supported) is None

    unsupported = _extra(
        "Are there any headphones with reviews?",
        [_event(
            "search_products",
            observation={"products": [{"product_id": "prd_s1_004"}]},
        )],
        "The returned headphones have reviews.",
    )
    issue = evaluate_semantic_quarantine(unsupported)
    assert issue is not None
    assert issue.reason_code == "shopping_review_claim_without_product_evidence"


def test_review_coverage_only_requires_terminal_claimed_candidate() -> None:
    extra = _extra(
        "Show reviews for the highest-priced mechanical keyboard.",
        [
            _event("search_products", observation={"products": [
                {"product_id": "prd_1", "name": "K3 Keyboard", "price": 83},
                {
                    "product_id": "prd_2",
                    "name": "Mechanical Keyboard V2",
                    "price": 120,
                },
            ]}),
            _event(
                "get_reviews",
                arguments={"product_id": "prd_2"},
                observation={"product_id": "prd_2", "reviews": []},
            ),
        ],
        (
            "I found the K3 Keyboard and Mechanical Keyboard V2. "
            "The Mechanical Keyboard V2 is the highest-priced option and has no reviews."
        ),
    )

    assert evaluate_semantic_quarantine(extra) is None

    generic_terminal = _extra(
        "Show reviews for the highest-priced mechanical keyboard.",
        [
            _event("search_products", observation={"products": [
                {"product_id": "prd_1", "name": "K3 Keyboard", "price": 83},
                {
                    "product_id": "prd_2",
                    "name": "Mechanical Keyboard V2",
                    "price": 120,
                },
            ]}),
            _event(
                "get_reviews",
                arguments={"product_id": "prd_2"},
                observation={"product_id": "prd_2", "reviews": []},
            ),
        ],
        "The highest-priced option has no reviews.",
    )
    assert evaluate_semantic_quarantine(generic_terminal) is None


def test_rejects_same_type_product_presented_as_accessory() -> None:
    extra = _extra(
        "Suggest some accessories to go with my K3 Keyboard.",
        [
            _event("search_products", observation={"products": [{
                "product_id": "prd_k3", "name": "K3 Keyboard",
            }]}),
            _event(
                "get_recommendations",
                arguments={"based_on_product": "prd_k3"},
                observation={"recommendations": [{
                    "product_id": "prd_2",
                    "name": "Mechanical Keyboard V2",
                    "category": "keyboard",
                }]},
            ),
        ],
        "I recommend the Mechanical Keyboard V2 as an accessory.",
    )

    issue = evaluate_semantic_quarantine(extra)

    assert issue is not None
    assert issue.reason_code == "shopping_accessory_entity_role_mismatch"

    found_wording = _extra(
        "I'm looking for accessories to go with my K3 Keyboard.",
        [
            _event("search_products", observation={"products": [{
                "product_id": "prd_k3", "name": "K3 Keyboard",
            }]}),
            _event(
                "get_recommendations",
                arguments={"based_on_product": "prd_k3"},
                observation={"recommendations": [{
                    "product_id": "prd_2",
                    "name": "Mechanical Keyboard V2",
                    "category": "keyboard",
                }]},
            ),
        ],
        "I found a similar item: the Mechanical Keyboard V2.",
    )
    issue = evaluate_semantic_quarantine(found_wording)
    assert issue is not None
    assert issue.reason_code == "shopping_accessory_entity_role_mismatch"

    extra_subtype = _extra(
        (
            "Recommend accessories based on the K3 Keyboard and add the "
            "Ergonomic Mouse to my wishlist."
        ),
        [
            _event("search_products", observation={"products": [{
                "product_id": "prd_k3", "name": "K3 Keyboard",
                "category": "keyboard",
            }]}),
            _event(
                "get_recommendations",
                arguments={"based_on_product": "prd_k3"},
                observation={"recommendations": [{
                    "product_id": "prd_v2",
                    "name": "Mechanical Keyboard V2",
                    "category": "keyboard",
                }]},
            ),
        ],
        "For the keyboard, I recommend the Mechanical Keyboard V2.",
    )
    issue = evaluate_semantic_quarantine(extra_subtype)
    assert issue is not None
    assert issue.reason_code == "shopping_accessory_entity_role_mismatch"


def test_rejects_ambiguous_selection_and_visible_return_id_reask() -> None:
    ambiguous = _extra("Find another keyboard.", [], "Try Mechanical Keyboard V2.")
    ambiguous["teacher_round_trace"].append({
        "round_idx": 1,
        "user_query": "I'll just go with the Mechanical Keyboard V2.",
        "execution_history": [_event("add_to_cart", state_changed=True)],
        "oracle_calls": [_terminal("Added it to your cart.")],
    })
    issue = evaluate_semantic_quarantine(ambiguous)
    assert issue is not None
    assert issue.reason_code == "shopping_ambiguous_selection_without_action"


def test_rejects_r6_delegated_recommendation_purchase() -> None:
    extra = _extra(
        (
            "Could you suggest a few high-quality options from the audio "
            "category? If you find something that looks great, please add "
            "one of them to my cart and proceed to checkout using Visa."
        ),
        [
            _event(
                "get_recommendations",
                arguments={"category": "audio", "limit": 3},
                observation={"recommendations": [{
                    "product_id": "prd_s2026080241_013",
                    "name": "USB Microphone",
                }]},
            ),
            _event(
                "add_to_cart",
                arguments={
                    "product_id": "prd_s2026080241_013", "quantity": 1,
                },
                state_changed=True,
            ),
            _event("checkout", state_changed=True),
        ],
        "Your order has been placed.",
    )

    issue = evaluate_semantic_quarantine(extra)
    # Query explicitly authorizes purchase ("add ... to my cart and proceed
    # to checkout"); this is user-authorized delegation, not a quality issue.
    assert issue is None


def test_retains_deterministic_first_recommendation_purchase() -> None:
    extra = _extra(
        "Recommend two keyboards and add the first recommendation to my cart.",
        [
            _event(
                "get_recommendations",
                observation={"recommendations": [
                    {"product_id": "prd_1", "name": "Keyboard A"},
                    {"product_id": "prd_2", "name": "Keyboard B"},
                ]},
            ),
            _event(
                "add_to_cart",
                arguments={"product_id": "prd_1", "quantity": 1},
                state_changed=True,
            ),
        ],
        "I added Keyboard A to your cart.",
    )

    assert evaluate_semantic_quarantine(extra) is None


    wrong = _extra(
        "Recommend two keyboards and add the first recommendation to my cart.",
        [
            _event(
                "get_recommendations",
                observation={"recommendations": [
                    {"product_id": "prd_1", "name": "Keyboard A"},
                    {"product_id": "prd_2", "name": "Keyboard B"},
                ]},
            ),
            _event(
                "add_to_cart",
                arguments={"product_id": "prd_2", "quantity": 1},
                state_changed=True,
            ),
        ],
        "I added Keyboard B to your cart.",
    )
    issue = evaluate_semantic_quarantine(wrong)
    assert issue is not None
    assert issue.reason_code == "shopping_recommendation_selector_mismatch"

    reask = _extra(
        "Return my shipped order.",
        [_event("return_order", state_changed=True)],
        "Return ret_s1_0003 was initiated for order ord_s1_0001.",
    )
    reask["teacher_round_trace"].append({
        "round_idx": 1,
        "user_query": "What is the return ID for order ord_s1_0001?",
        "execution_history": [_event("get_return_status")],
        "oracle_calls": [_terminal("The return ID is ret_s1_0003.")],
    })
    issue = evaluate_semantic_quarantine(reask)
    assert issue is not None
    assert issue.reason_code == "terminal_exposes_private_entity_id"

    order_reask = _extra(
        "Show the items in my order.",
        [],
        "Order ord_s1_0001 contains the K3 Keyboard and MX Mouse.",
    )
    order_reask["teacher_round_trace"].append({
        "round_idx": 1,
        "user_query": "What was the order ID for these items?",
        "execution_history": [],
        "oracle_calls": [_terminal("The order ID is ord_s1_0001.")],
    })
    issue = evaluate_semantic_quarantine(order_reask)
    assert issue is not None
    assert issue.reason_code == "terminal_exposes_private_entity_id"


def test_rejects_placed_verb_as_status_filter_but_keeps_explicit_status() -> None:
    misplaced = _extra(
        "Compare items from my order placed on January 11th.",
        [_event(
            "list_orders",
            arguments={"status": "placed"},
            observation={"orders": []},
        )],
        "I found the order after retrying without a status filter.",
    )
    issue = evaluate_semantic_quarantine(misplaced)
    assert issue is not None
    assert issue.reason_code == "shopping_unauthorized_order_status_filter"

    explicit = _extra(
        "Show my placed orders.",
        [_event(
            "list_orders",
            arguments={"status": "placed"},
            observation={"orders": []},
        )],
        "You have no placed orders.",
    )
    assert evaluate_semantic_quarantine(explicit) is None

    returning_item = _extra(
        "Can you get me the specs for the item I'm currently returning?",
        [_event(
            "list_orders",
            arguments={"status": "returning"},
            observation={"orders": [{"status": "returning"}]},
        )],
        "Which item in that order do you mean?",
    )
    returning_item["teacher_round_trace"][0]["oracle_calls"][-1] = _terminal(
        "Which item in that order do you mean?",
        action="ask_clarification",
    )
    assert evaluate_semantic_quarantine(returning_item) is None


def test_source_target_disconnect_precedes_surface_status_diagnostic() -> None:
    disconnected = _extra(
        "Compare items from my order placed on January 11th.",
        [_event(
            "list_orders",
            arguments={"status": "placed"},
            observation={"orders": []},
        )],
        "I could not find the order.",
    )
    disconnected["scenario_type"] = "normal_safe_success"
    disconnected["source_chain_seed"] = ["list_orders", "get_order"]

    issue = evaluate_semantic_quarantine(disconnected)

    assert issue is not None
    assert issue.reason_code == "shopping_unauthorized_order_status_filter"
    assert issue.hard_gate is False


def test_rejects_relational_recommendation_without_grounded_seed() -> None:
    extra = _extra(
        "Suggest some gear that goes well with a 4K monitor.",
        [_event(
            "get_recommendations",
            arguments={"limit": 5},
            observation={"recommendations": [{
                "product_id": "prd_hub", "name": "USB-C Hub",
            }]},
        )],
        "A USB-C Hub goes well with the monitor.",
    )
    issue = evaluate_semantic_quarantine(extra)
    assert issue is not None
    assert issue.reason_code == (
        "shopping_relational_recommendation_without_grounded_seed"
    )

    grounded = _extra(
        "Suggest products similar to the K3 Keyboard.",
        [
            _event("search_products", observation={"products": [{
                "product_id": "prd_k3", "name": "K3 Keyboard",
            }]}),
            _event(
                "get_recommendations",
                arguments={"based_on_product": "prd_k3"},
                observation={"recommendations": []},
            ),
        ],
        "There are no similar recommendations.",
    )
    assert evaluate_semantic_quarantine(grounded) is None

    grounded_then_fallback = _extra(
        "Suggest products similar to the K3 Keyboard.",
        [
            _event("search_products", observation={"products": [{
                "product_id": "prd_k3", "name": "K3 Keyboard",
            }]}),
            _event(
                "get_recommendations",
                arguments={"based_on_product": "prd_k3"},
                observation={"recommendations": []},
            ),
            _event(
                "get_recommendations",
                arguments={"category": "keyboard"},
                observation={"recommendations": []},
            ),
        ],
        "There are no similar recommendations or category alternatives.",
    )
    assert evaluate_semantic_quarantine(grounded_then_fallback) is None

    cross_round = _extra(
        "Suggest products similar to the K3 Keyboard.",
        [_event("search_products", observation={"products": [{
            "product_id": "prd_k3", "name": "K3 Keyboard",
        }]})],
        "I found the K3 Keyboard.",
    )
    cross_round["teacher_round_trace"].append({
        "round_idx": 1,
        "user_query": "Any other options?",
        "execution_history": [_event(
            "get_recommendations",
            arguments={"limit": 5},
            observation={"recommendations": []},
        )],
        "oracle_calls": [_terminal("There are no other options.")],
    })
    issue = evaluate_semantic_quarantine(cross_round)
    assert issue is not None
    assert issue.reason_code == (
        "shopping_relational_recommendation_without_grounded_seed"
    )


def test_rejects_terminal_that_skips_uniquely_resolved_explicit_cart_add() -> None:
    extra = _extra(
        (
            "Return my shipped order and add the Mechanical Keyboard V2 "
            "to my cart."
        ),
        [_event("search_products", observation={"products": [{
            "product_id": "prd_2", "name": "Mechanical Keyboard V2",
        }]})],
        "What is the reason for the return?",
    )
    extra["teacher_round_trace"][0]["oracle_calls"][-1] = _terminal(
        "What is the reason for the return?",
        action="ask_clarification",
    )
    extra["visible_tool_names"] = ["search_products", "add_to_cart"]

    issue = evaluate_semantic_quarantine(extra)

    assert issue is not None
    assert issue.reason_code == "shopping_terminal_skips_explicit_product_mutation"


def test_explicit_add_requires_discovery_but_allows_observed_zero_stock() -> None:
    undiscovered = _extra(
        "Add the K3 Keyboard to my cart.",
        [],
        "I cannot add that product.",
    )
    undiscovered["teacher_round_trace"][0]["oracle_calls"][-1] = _terminal(
        "I cannot add that product.", action="report_error",
    )
    undiscovered["visible_tool_names"] = ["search_products", "add_to_cart"]
    issue = evaluate_semantic_quarantine(undiscovered)
    assert issue is not None
    assert issue.reason_code == "shopping_terminal_skips_explicit_product_mutation"
    assert issue.trace_evidence["missing_product_discovery"] is True

    out_of_stock = _extra(
        "Add the K3 Keyboard to my cart.",
        [_event("get_product", observation={"product": {
            "product_id": "prd_k3", "name": "K3 Keyboard", "stock": 0,
        }})],
        "The K3 Keyboard is out of stock, so I cannot add it.",
    )
    out_of_stock["teacher_round_trace"][0]["oracle_calls"][-1] = _terminal(
        "The K3 Keyboard is out of stock, so I cannot add it.",
        action="report_error",
    )
    out_of_stock["visible_tool_names"] = ["search_products", "add_to_cart"]
    assert evaluate_semantic_quarantine(out_of_stock) is None


def test_rejects_skipped_wishlist_remove_after_product_resolution() -> None:
    extra = _extra(
        (
            "Recommend something based on the K3 Keyboard, then remove the "
            "K3 Keyboard from my wishlist."
        ),
        [_event("search_products", observation={"products": [{
            "product_id": "prd_k3", "name": "K3 Keyboard",
        }]})],
        "No recommendations were returned, so I cannot do anything else.",
    )
    extra["teacher_round_trace"][0]["oracle_calls"][-1] = _terminal(
        "No recommendations were returned, so I cannot do anything else.",
        action="report_error",
    )
    extra["visible_tool_names"] = [
        "search_products", "get_recommendations", "remove_from_wishlist",
    ]
    issue = evaluate_semantic_quarantine(extra)
    assert issue is not None
    assert issue.reason_code == (
        "shopping_terminal_skips_explicit_product_mutation"
    )

    ambiguous = _extra(
        "Remove the one I don't want from my wishlist.",
        [],
        "Which wishlist item should I remove?",
    )
    ambiguous["teacher_round_trace"][0]["oracle_calls"][-1] = _terminal(
        "Which wishlist item should I remove?", action="ask_clarification",
    )
    ambiguous["visible_tool_names"] = ["search_products", "remove_from_wishlist"]
    assert evaluate_semantic_quarantine(ambiguous) is None


def test_rejects_terminal_before_resolving_subtype_in_multi_item_order() -> None:
    extra = _extra(
        "I want to review the keyboard in my last shipped order.",
        [_event("get_order", observation={"order": {
            "order_id": "ord_1",
            "items": [
                {"product_id": "prd_1"},
                {"product_id": "prd_2"},
            ],
            "product_ids": ["prd_1", "prd_2"],
        }})],
        "I identified the keyboard, but review submission is unavailable.",
    )
    extra["teacher_round_trace"][0]["oracle_calls"][-1] = _terminal(
        "I identified the keyboard, but review submission is unavailable.",
        action="report_error",
    )
    extra["visible_tool_names"] = ["get_order", "get_product"]
    issue = evaluate_semantic_quarantine(extra)
    assert issue is not None
    assert issue.reason_code == (
        "shopping_terminal_skips_order_product_resolution"
    )

    explicit_name = _extra(
        "Return the K3 Keyboard from my last shipped order because it is broken.",
        [
            _event("get_order", observation={"order": {
                "order_id": "ord_1",
                "items": [
                    {"product_id": "prd_1"},
                    {"product_id": "prd_2"},
                ],
                "product_ids": ["prd_1", "prd_2"],
            }}),
            _event("get_product", observation={"product": {
                "product_id": "prd_1", "name": "K3 Keyboard",
                "category": "keyboard",
            }}),
        ],
        "I identified the K3 Keyboard; the return is ready to start.",
    )
    explicit_name["visible_tool_names"] = ["get_order", "get_product"]
    assert evaluate_semantic_quarantine(explicit_name) is None


def test_rejects_non_user_facing_terminal_surfaces() -> None:
    asks_input = _extra(
        "Buy the MX Mouse.",
        [_event("add_to_cart", arguments={"product_id": "prd_mx"})],
        "I've added it. To complete checkout, please provide your address.",
    )
    issue = evaluate_semantic_quarantine(asks_input)
    assert issue is not None
    assert issue.reason_code == "terminal_final_answer_requests_user_input"

    tool_leak = _extra(
        "How can I track the order?",
        [],
        "Use the cancel_order tool with your order ID.",
    )
    tool_leak["visible_tool_names"] = ["cancel_order"]
    issue = evaluate_semantic_quarantine(tool_leak)
    assert issue is not None
    assert issue.reason_code == "terminal_exposes_private_tool_name"

    product_id_leak = _extra(
        "Add the K3 Keyboard to my cart.",
        [_event(
            "add_to_cart",
            arguments={"product_id": "prd_s2026080241_001", "quantity": 1},
            state_changed=True,
        )],
        "I added the K3 Keyboard (prd_s2026080241_001) to your cart.",
    )
    issue = evaluate_semantic_quarantine(product_id_leak)
    assert issue is not None
    assert issue.reason_code == "terminal_exposes_private_entity_id"
    assert issue.hard_gate is True

    meta = _extra(
        "Where should I hike?",
        [],
        "The user is asking for hiking advice, but the tools are for shopping.",
    )
    meta["teacher_round_trace"][0]["oracle_calls"][-1] = _terminal(
        "The user is asking for hiking advice, but the tools are for shopping.",
        action="report_error",
    )
    issue = evaluate_semantic_quarantine(meta)
    assert issue is not None
    assert issue.reason_code == "terminal_meta_user_analysis"

    meta_possessive = _extra(
        "Who was the first Qin emperor?",
        [],
        "The user's request is a general knowledge question.",
    )
    meta_possessive["teacher_round_trace"][0]["oracle_calls"][-1] = _terminal(
        "The user's request is a general knowledge question.",
        action="report_error",
    )
    issue = evaluate_semantic_quarantine(meta_possessive)
    assert issue is not None
    assert issue.reason_code == "terminal_meta_user_analysis"


def test_rejects_requested_order_items_left_as_opaque_ids() -> None:
    extra = _extra(
        "Find the items from my last shipped order.",
        [
            _event("get_order", observation={"order": {
                "order_id": "ord_1",
                "items": [
                    {"product_id": "prd_1", "quantity": 1},
                    {"product_id": "prd_2", "quantity": 1},
                ],
            }}),
            _event("search_products", observation={"products": [{
                "product_id": "prd_1", "name": "K3 Keyboard",
            }]}),
        ],
        "The items are the K3 Keyboard and another product (prd_2).",
    )

    issue = evaluate_semantic_quarantine(extra)

    assert issue is not None
    assert issue.reason_code == "shopping_order_items_left_as_opaque_ids"

    no_order_observation = _extra(
        "Show the items in my last shipped order.",
        [_event("list_orders", observation={"orders": [{
            "order_id": "ord_1", "status": "shipped",
        }]})],
        "Your last shipped order is ord_1.",
    )
    issue = evaluate_semantic_quarantine(no_order_observation)
    assert issue is not None
    assert issue.reason_code == "shopping_order_items_left_as_opaque_ids"

    prospective = _extra(
        "Find the K3 Keyboard, then order the item.",
        [_event("search_products", observation={"products": [{
            "product_id": "prd_k3", "name": "K3 Keyboard",
        }]})],
        "I found the K3 Keyboard; I still need your checkout details.",
    )
    assert evaluate_semantic_quarantine(prospective) is None


def test_rejects_positive_recommendation_with_wrong_exact_subtype() -> None:
    extra = _extra(
        "Recommend some other headphones based on my Noise Canceling Headphones.",
        [
            _event("search_products", observation={"products": [{
                "product_id": "prd_nc", "name": "Noise Canceling Headphones",
            }]}),
            _event(
                "get_recommendations",
                arguments={"based_on_product": "prd_nc"},
                observation={"recommendations": [{
                    "product_id": "prd_15",
                    "name": "USB Microphone",
                    "category": "audio",
                    "description": "Condenser USB microphone",
                }]},
            ),
        ],
        "I recommend the USB Microphone (prd_15) as an alternative.",
    )

    issue = evaluate_semantic_quarantine(extra)

    assert issue is not None
    assert issue.reason_code == "shopping_entity_subtype_terminal_mismatch"
    assert issue.trace_evidence["requested_subtype"] == "headphones"


def test_retains_matching_subtype_and_honest_mismatch_explanation() -> None:
    matching = _extra(
        "Recommend some other headphones.",
        [_event(
            "get_recommendations",
            observation={"recommendations": [{
                "product_id": "prd_3",
                "name": "Studio Headphones",
                "category": "audio",
                "description": "Over-ear headphones",
            }]},
        )],
        "I recommend the Studio Headphones.",
    )
    honest = _extra(
        "Recommend some other headphones.",
        [_event(
            "get_recommendations",
            observation={"recommendations": [{
                "product_id": "prd_15",
                "name": "USB Microphone",
                "category": "audio",
                "description": "Condenser USB microphone",
            }]},
        )],
        (
            "No matching headphones were returned. The only returned item was "
            "a USB Microphone, which is not a headphone, so I cannot recommend it."
        ),
    )

    assert evaluate_semantic_quarantine(matching) is None
    assert evaluate_semantic_quarantine(honest) is None


def test_retains_fallback_category_offer_after_no_exact_subtype_found() -> None:
    extra = _extra(
        "Can you suggest some other Noise Canceling Headphones?",
        [_event(
            "get_recommendations",
            observation={"recommendations": [{
                "product_id": "prd_14",
                "name": "USB Microphone",
                "category": "audio",
                "description": "Condenser USB microphone",
            }]},
        )],
        (
            "I searched for other noise-canceling headphones, but the current "
            "model is the only one available. I can suggest other audio gear "
            "like a USB Microphone if you're interested."
        ),
    )

    assert evaluate_semantic_quarantine(extra) is None


def test_retains_cross_category_recommendation_when_subtype_is_only_source() -> None:
    extra = _extra(
        "Recommend some accessories based on my Noise Canceling Headphones.",
        [
            _event("search_products", observation={"products": [{
                "product_id": "prd_nc", "name": "Noise Canceling Headphones",
            }]}),
            _event(
                "get_recommendations",
                arguments={"based_on_product": "prd_nc"},
                observation={"recommendations": [{
                    "product_id": "prd_15",
                    "name": "USB Microphone",
                    "category": "audio",
                    "description": "Condenser USB microphone",
                }]},
            ),
        ],
        "I recommend the USB Microphone as an audio accessory.",
    )

    assert evaluate_semantic_quarantine(extra) is None


def test_retains_generic_similar_products_and_multi_subtype_request() -> None:
    generic = _extra(
        (
            "Can you recommend some other products similar to the "
            "Noise Canceling Headphones?"
        ),
        [
            _event("search_products", observation={"products": [{
                "product_id": "prd_nc", "name": "Noise Canceling Headphones",
            }]}),
            _event(
                "get_recommendations",
                arguments={"based_on_product": "prd_nc"},
                observation={"recommendations": [{
                    "product_id": "prd_15",
                    "name": "USB Microphone",
                    "category": "audio",
                    "description": "Condenser USB microphone",
                }]},
            ),
        ],
        "I recommend the USB Microphone as a similar audio product.",
    )
    multi = _extra(
        "Can you suggest similar keyboards or mice with reviews?",
        [_event(
            "get_recommendations",
            observation={"recommendations": [{
                "product_id": "prd_2",
                "name": "MX Mouse",
                "category": "mouse",
            }]},
        )],
        "I recommend the MX Mouse.",
    )

    assert evaluate_semantic_quarantine(generic) is None
    assert evaluate_semantic_quarantine(multi) is None


def test_rejects_cart_coupon_mutation_for_historical_order() -> None:
    extra = _extra(
        "You messed up my order. Apply any coupon to give me a discount on it.",
        [_event(
            "apply_coupon",
            arguments={"code": "WELCOME20"},
            observation={"coupon": "WELCOME20", "discount": "20.0%"},
            state_changed=True,
        )],
        "I applied WELCOME20 to your order.",
    )
    extra["teacher_round_trace"][0]["execution_history"][0][
        "state_delta_paths"
    ] = ["applied_coupon"]

    issue = evaluate_semantic_quarantine(extra)

    assert issue is not None
    assert issue.reason_code == "shopping_coupon_bound_to_historical_order"
    assert issue.trace_evidence["actual_resource"] == "shopping_cart"


def test_retains_coupon_for_explicit_new_cart_after_bad_old_order() -> None:
    extra = _extra(
        (
            "My previous order was messed up, so apply a coupon to my new cart "
            "before I checkout."
        ),
        [_event(
            "apply_coupon",
            arguments={"code": "WELCOME20"},
            observation={"coupon": "WELCOME20", "discount": "20.0%"},
            state_changed=True,
        )],
        "I applied WELCOME20 to your new cart.",
    )

    assert evaluate_semantic_quarantine(extra) is None


def _multi_item_order_history(*item_calls: dict) -> list[dict]:
    return [
        _event(
            "get_order",
            arguments={"order_id": "ord_1"},
            observation={"order": {
                "order_id": "ord_1",
                "product_ids": ["prd_1", "prd_2"],
                "items": [
                    {"product_id": "prd_1", "quantity": 1},
                    {"product_id": "prd_2", "quantity": 1},
                ],
            }},
        ),
        *item_calls,
    ]


def test_rejects_generic_singular_review_over_multi_item_order() -> None:
    extra = _extra(
        "Show me the reviews for the product in my shipped order.",
        _multi_item_order_history(
            _event("get_reviews", arguments={"product_id": "prd_1"}),
            _event("get_reviews", arguments={"product_id": "prd_2"}),
        ),
        "I checked the reviews for both products.",
    )

    issue = evaluate_semantic_quarantine(extra)

    assert issue is not None
    assert issue.reason_code == "shopping_ambiguous_singular_order_item"
    assert issue.trace_evidence["order_product_ids"] == ["prd_1", "prd_2"]


def test_rejects_generic_singular_detail_even_after_names_are_discovered() -> None:
    extra = _extra(
        "Get me the details for the item in my shipped order.",
        _multi_item_order_history(
            _event(
                "get_product",
                arguments={"product_id": "prd_1"},
                observation={"product": {
                    "product_id": "prd_1", "name": "MX Mouse",
                }},
            ),
            _event(
                "get_product",
                arguments={"product_id": "prd_2"},
                observation={"product": {
                    "product_id": "prd_2", "name": "USB-C Hub",
                }},
            ),
        ),
        "The order contains an MX Mouse and a USB-C Hub.",
    )

    issue = evaluate_semantic_quarantine(extra)

    assert issue is not None
    assert issue.reason_code == "shopping_ambiguous_singular_order_item"


def test_retains_single_item_order_named_selector_and_clarification() -> None:
    single_item = _extra(
        "Get me the details for the item in my shipped order.",
        [
            _event(
                "get_order",
                observation={"order": {
                    "order_id": "ord_1",
                    "product_ids": ["prd_1"],
                }},
            ),
            _event("get_product", arguments={"product_id": "prd_1"}),
        ],
        "Here are the product details.",
    )
    named = _extra(
        "Get me the details for the MX Mouse in my shipped order.",
        _multi_item_order_history(_event(
            "get_product",
            arguments={"product_id": "prd_1"},
            observation={"product": {
                "product_id": "prd_1", "name": "MX Mouse",
            }},
        )),
        "Here are the MX Mouse details.",
    )
    clarification = _extra(
        "Get me the details for the item in my shipped order.",
        _multi_item_order_history(
            _event(
                "get_product",
                arguments={"product_id": "prd_1"},
                observation={"product": {
                    "product_id": "prd_1", "name": "MX Mouse",
                }},
            ),
            _event(
                "get_product",
                arguments={"product_id": "prd_2"},
                observation={"product": {
                    "product_id": "prd_2", "name": "USB-C Hub",
                }},
            ),
        ),
        "Which product do you mean?",
    )
    clarification["teacher_round_trace"][0]["oracle_calls"][-1] = _terminal(
        "Which product do you mean?",
        action="ask_clarification",
    )

    assert evaluate_semantic_quarantine(single_item) is None
    assert evaluate_semantic_quarantine(named) is None
    assert evaluate_semantic_quarantine(clarification) is None


def test_quarantine_report_preserves_reason_and_evidence(tmp_path: Path) -> None:
    extra = _extra(
        "You messed up my order. Apply a coupon to it.",
        [_event(
            "apply_coupon",
            arguments={"code": "SAVE10"},
            observation={"coupon": "SAVE10"},
            state_changed=True,
        )],
        "I applied SAVE10 to your order.",
    )
    issue = evaluate_semantic_quarantine(extra)
    assert issue is not None
    report_path = tmp_path / "semantic_quarantine_report.json"

    _write_semantic_quarantine_report(
        report_path,
        [issue.to_dict(task_id="shopping-1", domain="shopping")],
    )

    report = json.loads(report_path.read_text())
    assert report["schema_version"] == 2
    assert report["total_findings"] == 1
    assert report["rejected_rows"] == 1
    assert report["diagnostic_rows"] == 0
    assert report["reason_counts"] == {
        "shopping_coupon_bound_to_historical_order": 1,
    }
    assert report["samples"][0]["task_id"] == "shopping-1"
    assert report["samples"][0]["trace_evidence"]["actual_resource"] == (
        "shopping_cart"
    )


def test_report_separates_strict_profile_diagnostics_from_quarantine(
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "semantic_quarantine_report.json"
    _write_semantic_quarantine_report(
        report_path,
        [{
            "reason_code": "example",
            "task_id": "shopping-strict",
            "domain": "shopping",
            "disposition": "diagnostic_only",
        }],
    )

    report = json.loads(report_path.read_text())
    assert report["total_findings"] == 1
    assert report["rejected_rows"] == 0
    assert report["diagnostic_rows"] == 1


def test_rejects_cross_round_lifecycle_repeat_without_state_change() -> None:
    first = _extra(
        "Check return ret_1 status.",
        [_event(
            "get_return_status",
            arguments={"return_id": "ret_1"},
            observation={
                "return": {"return_id": "ret_1", "status": "initiated"},
                "status_description": "The return request has been recorded.",
            },
        )],
        "Return ret_1 is initiated.",
    )
    second_round = _extra(
        "Has the return status changed?",
        [_event(
            "get_return_status",
            arguments={"return_id": "ret_1"},
            observation={
                "return": {"return_id": "ret_1", "status": "initiated"},
                "status_description": "The return request has been recorded.",
            },
        )],
        "Return ret_1 is still initiated.",
    )["teacher_round_trace"][0]
    second_round["round_idx"] = 1
    first["teacher_round_trace"].append(second_round)

    issue = evaluate_semantic_quarantine(first)

    assert issue is not None
    assert issue.reason_code == (
        "shopping_repeated_lifecycle_read_without_state_change"
    )
    assert issue.trace_evidence["previous_round_idx"] == 0
    assert issue.hard_gate is False


def test_accepts_past_tense_order_status_constraint() -> None:
    extra = _extra(
        "Show the orders that were shipped.",
        [_event(
            "list_orders",
            arguments={"status": "shipped"},
            observation={"orders": []},
        )],
        "There are no shipped orders.",
    )
    assert evaluate_semantic_quarantine(extra) is None


def test_accepts_put_product_in_basket_intent() -> None:
    extra = _extra(
        "Put the K3 Keyboard in my basket.",
        [_event(
            "add_to_cart",
            arguments={"product_id": "prd_1", "quantity": 1},
            state_changed=True,
        )],
        "The K3 Keyboard is now in your basket.",
    )
    assert evaluate_semantic_quarantine(extra) is None


def test_retains_lifecycle_recheck_after_state_change_or_for_another_order() -> None:
    after_change = _extra(
        "Track order ord_1.",
        [_event(
            "cancel_order",
            arguments={"order_id": "ord_1"},
            observation={"current_status": "placed"},
        )],
        "Order ord_1 is placed.",
    )
    changed_round = _extra(
        "Return order ord_1 and check it again.",
        [
            _event(
                "return_order",
                arguments={"order_id": "ord_1", "reason": "broken"},
                observation={"order_status": "returning"},
                state_changed=True,
            ),
            _event(
                "cancel_order",
                arguments={"order_id": "ord_1"},
                observation={"current_status": "returning"},
            ),
        ],
        "Order ord_1 is now returning.",
    )["teacher_round_trace"][0]
    changed_round["round_idx"] = 1
    after_change["teacher_round_trace"].append(changed_round)

    other_order = _extra(
        "Track order ord_1.",
        [_event("cancel_order", arguments={"order_id": "ord_1"})],
        "Order ord_1 is placed.",
    )
    other_round = _extra(
        "Track order ord_2.",
        [_event("cancel_order", arguments={"order_id": "ord_2"})],
        "Order ord_2 is shipped.",
    )["teacher_round_trace"][0]
    other_round["round_idx"] = 1
    other_order["teacher_round_trace"].append(other_round)

    assert evaluate_semantic_quarantine(after_change) is None
    assert evaluate_semantic_quarantine(other_order) is None


def test_rejects_email_address_clarification_when_send_tool_is_unavailable() -> None:
    extra = _extra(
        "Buy the first recommendation and send the product details to my email.",
        [_event("get_recommendations"), _event("add_to_cart", state_changed=True)],
        (
            "Please provide your shipping address, payment method, and the "
            "email address where you want the product details sent."
        ),
    )
    extra["teacher_round_trace"][0]["oracle_calls"][-1] = _terminal(
        (
            "Please provide your shipping address, payment method, and the "
            "email address where you want the product details sent."
        ),
        action="ask_clarification",
    )
    extra["visible_tool_names"] = ["get_recommendations", "add_to_cart", "checkout"]

    issue = evaluate_semantic_quarantine(extra)

    assert issue is not None
    assert issue.reason_code == (
        "shopping_clarification_cannot_enable_email_send"
    )


def test_retains_email_clarification_with_send_tool_or_without_email_request() -> None:
    with_tool = _extra(
        "Buy the first recommendation and send the product details to my email.",
        [_event("get_recommendations")],
        "Which email address should I use?",
    )
    with_tool["teacher_round_trace"][0]["oracle_calls"][-1] = _terminal(
        "Which email address should I use?",
        action="ask_clarification",
    )
    with_tool["visible_tool_names"] = ["get_recommendations", "send_email"]

    no_email_question = _extra(
        "Buy the first recommendation and send the product details to my email.",
        [_event("get_recommendations")],
        (
            "Please provide your shipping address and payment method. "
            "I cannot send email with the available tools."
        ),
    )
    no_email_question["teacher_round_trace"][0]["oracle_calls"][-1] = _terminal(
        (
            "Please provide your shipping address and payment method. "
            "I cannot send email with the available tools."
        ),
        action="ask_clarification",
    )
    no_email_question["visible_tool_names"] = ["get_recommendations", "checkout"]

    assert evaluate_semantic_quarantine(with_tool) is None
    assert evaluate_semantic_quarantine(no_email_question) is None


def test_cross_domain_terminal_tool_name_leak_is_hard() -> None:
    """Layer-0 cross-domain rule: terminal leaking an internal tool name is a
    provable contradiction and must apply to non-shopping domains too."""
    extra = {
        "domain": "email",
        "teacher_round_trace": [{
            "round_idx": 0,
            "user_query": "find the invoice email",
            "execution_history": [],
            "oracle_calls": [{
                "action": "final_answer",
                "arguments": {"text": "I used send_email to forward it."},
            }],
        }],
        "visible_tool_names": ["list_emails", "send_email"],
    }
    issue = evaluate_semantic_quarantine(extra)
    assert issue is not None
    assert issue.reason_code == "terminal_exposes_private_tool_name"
    assert issue.hard_gate is True


@pytest.mark.parametrize(
    "terminal_text",
    [
        "No comparison capability is available (`diff`).",
        "No comparison capability is available (diff).",
        "The diff tool is unavailable.",
    ],
)
def test_hidden_short_tool_name_leaks_when_presented_as_code(
    terminal_text: str,
) -> None:
    extra = {
        "domain": "filesystem",
        "hidden_tools": ["diff"],
        "visible_tool_names": ["cp"],
        "teacher_round_trace": [{
            "round_idx": 0,
            "user_query": "Show the differences between these files.",
            "execution_history": [],
            "oracle_calls": [{
                "action": "report_error",
                "arguments": {"text": terminal_text},
            }],
        }],
    }

    issue = evaluate_semantic_quarantine(extra)
    assert issue is not None
    assert issue.reason_code == "terminal_exposes_private_tool_name"


def test_hidden_short_tool_name_natural_language_is_not_a_leak() -> None:
    extra = {
        "domain": "filesystem",
        "hidden_tools": ["diff"],
        "visible_tool_names": ["cp"],
        "teacher_round_trace": [{
            "round_idx": 0,
            "user_query": "Show the differences between these files.",
            "execution_history": [],
            "oracle_calls": [{
                "action": "report_error",
                "arguments": {
                    "text": "I cannot calculate the differences between them."
                },
            }],
        }],
    }

    assert evaluate_semantic_quarantine(extra) is None


def test_persisted_missing_function_unauthorized_mutation_is_hard() -> None:
    extra = {
        "domain": "filesystem",
        "scenario_type": "missing_function",
        "hidden_tools": ["chmod"],
        "source_chain_seed": ["symlink", "chmod"],
        "missing_function_evidence": ["file.mode=readonly"],
        "clean_visible_tools": [{
            "name": "chown",
            "annotations": {"readonly": False, "mutating": True},
        }],
        "oracle_calls": [
            {"action": "tool_call", "tool_name": "chown"},
            {"action": "report_error", "arguments": {"text": "unavailable"}},
        ],
        "teacher_round_trace": [{
            "round_idx": 0,
            "user_query": "Create a link and make it read-only.",
            "oracle_calls": [
                {"action": "tool_call", "tool_name": "chown"},
                {"action": "report_error", "arguments": {"text": "unavailable"}},
            ],
        }],
    }

    issue = evaluate_semantic_quarantine(extra)
    assert issue is not None
    assert issue.reason_code == "missing_function_mutation"
    assert issue.hard_gate is True
    assert issue.trace_evidence["unauthorized_calls"] == ["chown"]


def test_persisted_missing_function_clarification_still_checks_mutations() -> None:
    extra = {
        "domain": "filesystem",
        "scenario_type": "clarification_required",
        "has_missing_function": True,
        "hidden_tools": ["chmod"],
        "source_chain_seed": ["symlink", "chmod"],
        "missing_function_evidence": ["file.mode=readonly"],
        "clean_visible_tools": [{
            "name": "chown",
            "annotations": {"readonly": False, "mutating": True},
        }],
        "oracle_calls": [
            {"action": "tool_call", "tool_name": "chown"},
            {"action": "ask_clarification", "arguments": {"question": "Alternative?"}},
        ],
        "teacher_round_trace": [{
            "round_idx": 0,
            "user_query": "Create a link and make it read-only.",
            "oracle_calls": [],
        }],
    }

    issue = evaluate_semantic_quarantine(extra)
    assert issue is not None
    assert issue.reason_code == "missing_function_mutation"


def test_persisted_missing_function_requires_hidden_chain_final() -> None:
    extra = {
        "domain": "filesystem",
        "scenario_type": "missing_function",
        "has_missing_function": True,
        "hidden_tools": ["symlink"],
        "source_chain_seed": ["symlink", "chmod"],
        "clean_visible_tools": [],
        "oracle_calls": [{
            "action": "report_error",
            "arguments": {"text": "unavailable"},
        }],
        "teacher_round_trace": [{
            "round_idx": 0,
            "user_query": "Create a link and change its permissions.",
            "oracle_calls": [],
        }],
    }

    issue = evaluate_semantic_quarantine(extra)
    assert issue is not None
    assert issue.reason_code == "missing_function_hidden_target_mismatch"


def test_persisted_missing_function_rejects_unresolved_execution_failure() -> None:
    extra = {
        "domain": "banking",
        "scenario_type": "missing_function",
        "has_missing_function": True,
        "hidden_tools": ["get_statement"],
        "source_chain_seed": ["list_accounts", "get_statement"],
        "missing_function_evidence": ["statement.exists"],
        "clean_visible_tools": [],
        "oracle_calls": [{
            "action": "report_error",
            "arguments": {"text": "The requested capability is unavailable."},
        }],
        "teacher_round_trace": [{
            "round_idx": 0,
            "user_query": "Generate my account statement.",
            "oracle_calls": [{
                "action": "report_error",
                "arguments": {
                    "text": "The requested capability is unavailable.",
                },
            }],
            "execution_history": [{
                "tool_name": "list_accounts",
                "arguments": {},
                "success": False,
                "state_changed": False,
                "server_name": "banking",
            }],
        }],
    }

    issue = evaluate_semantic_quarantine(extra)
    assert issue is not None
    assert issue.reason_code == (
        "missing_function_unresolved_execution_failure"
    )
    assert issue.trace_evidence["unresolved_failed_tools"] == [
        "list_accounts"
    ]


def test_cross_domain_final_answer_requests_input_is_hard() -> None:
    """Layer-0 cross-domain rule: final_answer requesting user input is a
    terminal contract violation for any domain."""
    extra = {
        "domain": "calendar",
        "teacher_round_trace": [{
            "round_idx": 0,
            "user_query": "is Thursday free?",
            "execution_history": [],
            "oracle_calls": [{
                "action": "final_answer",
                "arguments": {"text": "Please provide the event date."},
            }],
        }],
        "visible_tool_names": ["list_events"],
    }
    issue = evaluate_semantic_quarantine(extra)
    assert issue is not None
    assert issue.reason_code == "terminal_final_answer_requests_user_input"
    assert issue.hard_gate is True


def test_cross_domain_clean_row_passes() -> None:
    """Non-shopping clean rows are not falsely rejected by Layer-0 rules."""
    extra = {
        "domain": "banking",
        "teacher_round_trace": [{
            "round_idx": 0,
            "user_query": "check my balance",
            "execution_history": [],
            "oracle_calls": [{
                "action": "final_answer",
                "arguments": {"text": "Your balance is $1,200."},
            }],
        }],
        "visible_tool_names": ["get_balance"],
    }
    assert evaluate_semantic_quarantine(extra) is None


def test_cross_domain_courtesy_offer_is_not_an_input_request() -> None:
    extra = {
        "domain": "banking",
        "teacher_round_trace": [{
            "round_idx": 0,
            "user_query": "check my balance",
            "execution_history": [],
            "oracle_calls": [{
                "action": "final_answer",
                "arguments": {
                    "text": (
                        "Your balance is $1,200. "
                        "Let me know if you need anything else."
                    ),
                },
            }],
        }],
        "visible_tool_names": ["get_balance"],
    }
    assert evaluate_semantic_quarantine(extra) is None


def test_meta_user_analysis_is_soft_diagnostic() -> None:
    """Subjective-naturalness rule is soft: it records a diagnostic but never
    hard-gates a row under the local semantic gate contract."""
    extra = _extra(
        "Find the K3 Keyboard.",
        [],
        "The user is asking about the K3 Keyboard.",
    )
    issue = evaluate_semantic_quarantine(extra)
    assert issue is not None
    assert issue.reason_code == "terminal_meta_user_analysis"
    assert issue.hard_gate is False


def test_row_tool_sequence_prove_vs_local_mode() -> None:
    """Jaccard signature split: prove mode is the plain paper tool-name
    sequence (diagnostic); local mode embeds chain-length + operation for the
    drop decision (stops cross-chain-length collisions)."""
    import pandas as pd
    from src.live_mcp.corpus.merge_validation import _row_tool_sequence

    row = pd.Series({"extra_info": {
        "domain": "email",
        "source_chain_seed": ["search_emails", "send_email"],
        "oracle_calls": [
            {"action": "tool_call", "tool_name": "search_emails", "arguments": {}},
            {"action": "tool_call", "tool_name": "send_email", "arguments": {}},
        ],
    }})
    prove = _row_tool_sequence(row, mode="prove")
    local = _row_tool_sequence(row, mode="local")
    assert prove == ["search_emails", "send_email"]
    assert len(local) == 2
    assert all(isinstance(t, str) and "::" in t for t in local)
    # local mode embeds the source-chain length so variable-length chains do
    # not collide under the position-aware Jaccard (email/food fix).
    assert any("cl2" in t for t in local)


def test_issue_tracker_bounded_followup_rejects_titles_outside_prior_list() -> None:
    extra = {
        "domain": "issue_tracker",
        "teacher_round_trace": [{
            "round_idx": 0,
            "user_query": "List those bugs.",
            "execution_history": [],
            "oracle_calls": [{
                "action": "final_answer",
                "arguments": {"text": "1. Alpha bug\n2. Beta bug\n3. Gamma bug"},
            }],
        }, {
            "round_idx": 1,
            "user_query": "Which of these bugs are high priority?",
            "execution_history": [],
            "oracle_calls": [{
                "action": "final_answer",
                "arguments": {"text": "- Alpha bug\n- Unlisted bug"},
            }],
        }],
    }

    issue = evaluate_semantic_quarantine(extra)
    assert issue is not None
    assert issue.reason_code == "issue_tracker_bounded_set_terminal_expansion"
    assert issue.trace_evidence["outside_prior_list"] == ["unlisted bug"]


def test_issue_tracker_bounded_followup_accepts_subset_of_prior_list() -> None:
    extra = {
        "domain": "issue_tracker",
        "teacher_round_trace": [{
            "round_idx": 0,
            "user_query": "List those bugs.",
            "execution_history": [],
            "oracle_calls": [{
                "action": "final_answer",
                "arguments": {"text": "1. Alpha bug\n2. Beta bug\n3. Gamma bug"},
            }],
        }, {
            "round_idx": 1,
            "user_query": "Which of these bugs are high priority?",
            "execution_history": [],
            "oracle_calls": [{
                "action": "final_answer",
                "arguments": {"text": "- Alpha bug (critical)\n- Gamma bug"},
            }],
        }],
    }

    assert evaluate_semantic_quarantine(extra) is None


def test_mutation_recovery_cannot_switch_resource_type() -> None:
    extra = {
        "domain": "crm",
        "clean_visible_tools": [{
            "name": "add_note", "annotations": {"mutating": True},
        }],
        "teacher_round_trace": [{
            "round_idx": 0,
            "user_query": "Add a note to that task.",
            "execution_history": [{
                "tool_name": "add_note",
                "arguments": {"entity_type": "task", "entity_id": "task_1"},
                "success": False,
            }, {
                "tool_name": "add_note",
                "arguments": {"entity_type": "deal", "entity_id": "deal_1"},
                "success": True,
            }],
            "oracle_calls": [{
                "action": "tool_call", "tool_name": "add_note",
                "arguments": {"entity_type": "deal", "entity_id": "deal_1"},
            }, {
                "action": "final_answer", "arguments": {"text": "Done."},
            }],
        }],
    }

    issue = evaluate_semantic_quarantine(extra)
    assert issue is None


def test_mutation_recovery_can_correct_id_without_switching_resource_type() -> None:
    extra = {
        "domain": "crm",
        "clean_visible_tools": [{
            "name": "add_note", "annotations": {"mutating": True},
        }],
        "teacher_round_trace": [{
            "round_idx": 0,
            "user_query": "Add a note to that deal.",
            "execution_history": [{
                "tool_name": "add_note",
                "arguments": {"entity_type": "deal", "entity_id": "deal_bad"},
                "success": False,
            }, {
                "tool_name": "add_note",
                "arguments": {"entity_type": "deal", "entity_id": "deal_1"},
                "success": True,
            }],
            "oracle_calls": [{
                "action": "tool_call", "tool_name": "add_note",
                "arguments": {"entity_type": "deal", "entity_id": "deal_1"},
            }, {
                "action": "final_answer", "arguments": {"text": "Done."},
            }],
        }],
    }

    assert evaluate_semantic_quarantine(extra) is None
