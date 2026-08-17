from __future__ import annotations

import pytest

from src.live_mcp.fsm import RobustnessPlan
from src.live_mcp.generation.candidate_trace_validation import (
    validate_early_candidate_trace,
)
from src.live_mcp.generation.teacher_contracts import (
    typed_entity_reference_visibility_from_rounds,
    user_visible_private_id_exposure,
)
from src.live_mcp.replay.gates import provenance_check
from src.live_mcp.types import OracleCall
from src.live_mcp.servers.banking.server import TOOLS as BANKING_TOOLS
from src.live_mcp.servers.payments.server import TOOLS as PAYMENT_TOOLS
from src.live_mcp.servers.crm.server import TOOLS as CRM_TOOLS
from src.live_mcp.servers.email.server import TOOLS as EMAIL_TOOLS
from src.live_mcp.servers.shopping.server import TOOLS as SHOPPING_TOOLS
from src.live_mcp.servers.team_chat.server import TOOLS as TEAM_CHAT_TOOLS
from src.live_mcp.registry.tool_semantics import unresolved_failed_tool_names


@pytest.mark.parametrize("query_amount", ["$1,000", "1,000 USD"])
def test_provenance_treats_thousands_separators_as_numeric_formatting(
    query_amount: str,
) -> None:
    call = OracleCall("transfer", {
        "from_account": "acc_source",
        "to_account": "acc_target",
        "amount": 1000,
        "currency": "USD",
    }, server_name="banking")
    passed, violations = provenance_check(
        [call],
        user_query=(
            f"Transfer {query_amount} from acc_source to acc_target in USD."
        ),
        aligned_observations=[{}],
        tool_schemas=BANKING_TOOLS,
        domain="banking",
    )

    assert passed is True
    assert violations == []


def test_provenance_does_not_conflate_different_or_partial_numbers() -> None:
    call = OracleCall("transfer", {
        "from_account": "acc_source",
        "to_account": "acc_target",
        "amount": 1000,
        "currency": "USD",
    }, server_name="banking")
    passed, violations = provenance_check(
        [call],
        user_query="Transfer $10,000 from acc_source to acc_target in USD.",
        aligned_observations=[{}],
        tool_schemas=BANKING_TOOLS,
        domain="banking",
    )

    assert passed is False
    assert [(item["param"], item["value"]) for item in violations] == [
        ("amount", "1000"),
    ]


def _validate(
    calls: list[OracleCall],
    *,
    plan: RobustnessPlan | None = None,
    source_chain: list[str] | None = None,
):
    return validate_early_candidate_trace(
        domain="calendar",
        difficulty="complete",
        plan=plan or RobustnessPlan(),
        oracle_calls=calls,
        execution_history=[{
            "success": True,
            "tool_name": call.tool_name,
            "server_name": "calendar",
            "state_changed": False,
        } for call in calls if call.action == "tool_call"],
        conversation_queries=["show my events"],
        oracle_calls_per_round=[calls],
        source_chain_seed=source_chain,
        server_tools=[],
        paper_baseline=True,
    )


def test_accepts_realized_readonly_source_chain() -> None:
    calls = [
        OracleCall("list_events", {}),
        OracleCall("", {"text": "done"}, action="final_answer"),
    ]
    result = _validate(calls, source_chain=["list_events"])

    assert result.accepted is True
    assert result.scenario_type == "normal_safe_success"
    assert result.terminal_action == "final_answer"


def test_local_profile_rejects_initial_mutation_absent_from_query_source() -> None:
    calls = [
        OracleCall("search_emails", {"sender": "client@example.com"}),
        OracleCall("get_email", {"email_id": "email-public-1"}),
        OracleCall("archive_email", {"email_id": "email-public-1"}),
        OracleCall("", {"text": "Here is the email."}, action="final_answer"),
    ]

    result = validate_early_candidate_trace(
        domain="email",
        difficulty="complete",
        plan=RobustnessPlan(),
        oracle_calls=calls,
        execution_history=[],
        conversation_queries=["show me the full email from the client"],
        oracle_calls_per_round=[calls],
        source_chain_seed=["search_emails", "get_email"],
        server_tools=EMAIL_TOOLS,
        paper_baseline=False,
    )

    assert result.accepted is False
    assert result.reason == (
        "initial_round_unauthorized_mutation:tools=['archive_email']"
    )


def test_local_profile_accepts_all_source_authorized_mutations() -> None:
    calls = [
        OracleCall("return_order", {
            "order_id": "order-public-1", "reason": "broken",
        }),
        OracleCall("add_to_wishlist", {"product_id": "product-public-1"}),
        OracleCall("", {"text": "The return and wishlist update are done."},
                   action="final_answer"),
    ]

    result = validate_early_candidate_trace(
        domain="shopping",
        difficulty="complete",
        plan=RobustnessPlan(),
        oracle_calls=calls,
        execution_history=[{
            "success": True,
            "tool_name": call.tool_name,
            "server_name": "shopping",
            "state_changed": True,
        } for call in calls if call.action == "tool_call"],
        conversation_queries=["return the broken monitor arm and wishlist it"],
        oracle_calls_per_round=[calls],
        source_chain_seed=[
            "return_order", "get_return_status", "add_to_wishlist",
        ],
        server_tools=SHOPPING_TOOLS,
        paper_baseline=False,
    )

    assert result.reason != "initial_round_unauthorized_mutation"


def test_local_profile_rejects_private_id_in_assistant_terminal() -> None:
    calls = [
        OracleCall("list_accounts", {"type": "business"}),
        OracleCall("get_account_info", {"account_id": "acc_s42_003"}),
        OracleCall("", {
            "text": "Your business account ID is acc_s42_003.",
        }, action="final_answer"),
    ]
    result = validate_early_candidate_trace(
        domain="banking",
        difficulty="complete",
        plan=RobustnessPlan(),
        oracle_calls=calls,
        execution_history=[{
            "success": True,
            "tool_name": call.tool_name,
            "server_name": "banking",
            "state_changed": False,
        } for call in calls if call.action == "tool_call"],
        conversation_queries=["show my business account details"],
        oracle_calls_per_round=[calls],
        source_chain_seed=["list_accounts", "get_account_info"],
        server_tools=[],
        paper_baseline=False,
    )

    assert result.accepted is False
    assert result.reason == (
        "user_visible_private_entity_id:round=0:"
        "surface=assistant_terminal:ids=['acc_s42_003']"
    )


def test_local_profile_rejects_seeded_id_despite_public_override() -> None:
    public_invoice_id = "inv_s2771120365517000001_0007"
    terminal = OracleCall(
        "", {"text": "This capability is unavailable."},
        action="report_error",
    )

    result = validate_early_candidate_trace(
        domain="payments",
        difficulty="complete",
        plan=RobustnessPlan(irrelevance=True),
        oracle_calls=[terminal],
        execution_history=[],
        conversation_queries=[f"show invoice {public_invoice_id}"],
        oracle_calls_per_round=[[terminal]],
        oracle_observations_per_round=[[{}]],
        source_chain_seed=None,
        server_tools=PAYMENT_TOOLS,
        paper_baseline=False,
        live_state_public_entity_ids={public_invoice_id},
    )

    assert result.accepted is False
    assert result.reason.startswith("user_visible_private_entity_id:")


def test_banking_transaction_id_is_a_typed_public_business_reference() -> None:
    calls = [[OracleCall(
        "bill_pay",
        {
            "account_id": "acc_s42_003",
            "amount": 45,
            "payee": "FreshFoods",
        },
    )]]
    observations = [[{
        "transaction": {
            "txn_id": "txn_0016",
            "type": "bill_pay",
            "payee": "FreshFoods",
        },
    }]]

    private_ids, public_ids = typed_entity_reference_visibility_from_rounds(
        domain="banking",
        calls_per_round=calls,
        observations_per_round=observations,
        server_tools=BANKING_TOOLS,
        entity_types={"transaction"},
    )

    assert private_ids == set()
    assert public_ids == {"txn_0016"}
    assert user_visible_private_id_exposure(
        ["what is the transaction ID?"],
        [[OracleCall(
            "", {"text": "The transaction ID is txn_0016."},
            action="final_answer",
        )]],
        private_entity_ids=private_ids,
        public_entity_ids=public_ids,
    ) is None


@pytest.mark.parametrize("alias", ["_004", "...004"])
def test_local_profile_rejects_derived_private_handle_suffix(alias: str) -> None:
    exposure = user_visible_private_id_exposure(
        ["show my savings account"],
        [[OracleCall("", {
            "text": f"Your savings account ending in {alias} is active.",
        }, action="final_answer")]],
        private_entity_ids={"acc_s3712943567317000004_004"},
    )

    assert exposure is not None
    assert exposure.surface == "assistant_terminal"
    assert exposure.leaked_ids == (alias,)


def test_local_profile_rejects_private_id_in_continuation_query() -> None:
    first_round = [
        OracleCall("list_accounts", {"type": "business"}),
        OracleCall("get_account_info", {"account_id": "acc_s42_003"}),
        OracleCall("", {"text": "Your business account is active."},
                   action="final_answer"),
    ]
    second_round = [
        OracleCall("get_history", {"account_id": "acc_s42_003"}),
        OracleCall("", {"text": "Here are the recent transactions."},
                   action="final_answer"),
    ]
    calls = first_round + second_round
    result = validate_early_candidate_trace(
        domain="banking",
        difficulty="complete",
        plan=RobustnessPlan(),
        oracle_calls=calls,
        execution_history=[{
            "success": True,
            "tool_name": call.tool_name,
            "server_name": "banking",
            "state_changed": False,
        } for call in calls if call.action == "tool_call"],
        conversation_queries=[
            "show my business account details",
            "show the history for acc_s42_003",
        ],
        oracle_calls_per_round=[first_round, second_round],
        source_chain_seed=["list_accounts", "get_account_info"],
        server_tools=[],
        paper_baseline=False,
    )

    assert result.accepted is False
    assert result.reason == (
        "user_visible_private_entity_id:round=1:"
        "surface=user_query:ids=['acc_s42_003']"
    )


def test_private_id_gate_keeps_internal_arguments_and_paper_boundary() -> None:
    private_terminal_calls = [
        OracleCall("list_accounts", {"type": "business"}),
        OracleCall("get_account_info", {"account_id": "acc_s42_003"}),
        OracleCall("", {"text": "The account ID is acc_s42_003."},
                   action="final_answer"),
    ]
    common = dict(
        domain="banking",
        difficulty="complete",
        plan=RobustnessPlan(),
        execution_history=[{
            "success": True,
            "tool_name": "list_accounts",
            "server_name": "banking",
            "state_changed": False,
        }, {
            "success": True,
            "tool_name": "get_account_info",
            "server_name": "banking",
            "state_changed": False,
        }],
        conversation_queries=["show my business account details"],
        source_chain_seed=["list_accounts", "get_account_info"],
        server_tools=[],
    )
    paper = validate_early_candidate_trace(
        **common,
        oracle_calls=private_terminal_calls,
        oracle_calls_per_round=[private_terminal_calls],
        paper_baseline=True,
    )
    assert paper.accepted is True

    internal_only_calls = [
        private_terminal_calls[0],
        private_terminal_calls[1],
        OracleCall("", {"text": "Your business account is active."},
                   action="final_answer"),
    ]
    local = validate_early_candidate_trace(
        **common,
        oracle_calls=internal_only_calls,
        oracle_calls_per_round=[internal_only_calls],
        paper_baseline=False,
    )
    assert local.accepted is True


def test_local_profile_allows_runtime_created_business_reference_in_terminal() -> None:
    calls = [
        OracleCall("create_invoice", {
            "customer": "Acme", "amount": 1500,
        }),
        OracleCall("", {
            "text": "Invoice inv_0021 was created for Acme.",
        }, action="final_answer"),
    ]
    result = validate_early_candidate_trace(
        domain="payments",
        difficulty="complete",
        plan=RobustnessPlan(),
        oracle_calls=calls,
        execution_history=[{
            "success": True,
            "tool_name": "create_invoice",
            "server_name": "payments",
            "state_changed": True,
        }],
        conversation_queries=["Bill Acme $1,500."],
        oracle_calls_per_round=[calls],
        oracle_observations_per_round=[[
            {"invoice": {"invoice_id": "inv_0021", "customer": "Acme"}},
            {},
        ]],
        source_chain_seed=["create_invoice"],
        server_tools=PAYMENT_TOOLS,
        paper_baseline=False,
    )

    assert result.accepted is True


def test_local_profile_still_rejects_backend_webhook_handle_in_terminal() -> None:
    calls = [
        OracleCall("list_webhooks", {}),
        OracleCall("", {
            "text": "The webhook ID is wh_s42_003.",
        }, action="final_answer"),
    ]
    result = validate_early_candidate_trace(
        domain="payments",
        difficulty="complete",
        plan=RobustnessPlan(),
        oracle_calls=calls,
        execution_history=[{
            "success": True,
            "tool_name": "list_webhooks",
            "server_name": "payments",
            "state_changed": False,
        }],
        conversation_queries=["Show my registered webhooks."],
        oracle_calls_per_round=[calls],
        oracle_observations_per_round=[[
            {"webhooks": [{"webhook_id": "wh_s42_003"}]},
            {},
        ]],
        source_chain_seed=["list_webhooks"],
        server_tools=PAYMENT_TOOLS,
        paper_baseline=False,
    )

    assert result.accepted is False
    assert "ids=['wh_s42_003']" in result.reason


def test_local_profile_keeps_runtime_id_inside_tool_arguments() -> None:
    calls = [
        OracleCall("create_invoice", {
            "customer": "Acme", "amount": 1500,
        }),
        OracleCall("pay_invoice", {
            "invoice_id": "inv_0021", "amount": 1500, "method": "card",
        }),
        OracleCall("", {
            "text": "The Acme invoice was created and paid.",
        }, action="final_answer"),
    ]
    result = validate_early_candidate_trace(
        domain="payments",
        difficulty="complete",
        plan=RobustnessPlan(),
        oracle_calls=calls,
        execution_history=[
            {
                "success": True, "tool_name": "create_invoice",
                "server_name": "payments", "state_changed": True,
            },
            {
                "success": True, "tool_name": "pay_invoice",
                "server_name": "payments", "state_changed": True,
            },
        ],
        conversation_queries=["Bill Acme $1,500 and pay it by card."],
        oracle_calls_per_round=[calls],
        oracle_observations_per_round=[[
            {"invoice": {"invoice_id": "inv_0021", "customer": "Acme"}},
            {"invoice": {"invoice_id": "inv_0021", "status": "paid"}},
            {},
        ]],
        source_chain_seed=["create_invoice", "pay_invoice"],
        server_tools=PAYMENT_TOOLS,
        paper_baseline=False,
    )

    assert result.accepted is True


def test_local_profile_does_not_classify_natural_name_as_private_id() -> None:
    calls = [
        OracleCall("list_channels", {}),
        OracleCall("", {
            "text": "The releases channel has four members.",
        }, action="final_answer"),
    ]
    result = validate_early_candidate_trace(
        domain="team_chat",
        difficulty="complete",
        plan=RobustnessPlan(),
        oracle_calls=calls,
        execution_history=[{
            "success": True,
            "tool_name": "list_channels",
            "server_name": "team_chat",
            "state_changed": False,
        }],
        conversation_queries=["How many people are in releases?"],
        oracle_calls_per_round=[calls],
        oracle_observations_per_round=[[
            {"channels": [{
                "channel_id": "ch_s42_003",
                "name": "releases",
                "member_count": 4,
            }]},
            {},
        ]],
        source_chain_seed=["list_channels"],
        server_tools=TEAM_CHAT_TOOLS,
        paper_baseline=False,
    )

    assert result.accepted is True


def test_local_profile_rejects_runtime_private_id_in_terminal() -> None:
    calls = [
        OracleCall("search_emails", {"query": "release"}),
        OracleCall("", {
            "text": "The matching internal email ID is eml_0028.",
        }, action="final_answer"),
    ]
    result = validate_early_candidate_trace(
        domain="email",
        difficulty="complete",
        plan=RobustnessPlan(),
        oracle_calls=calls,
        execution_history=[{
            "success": True,
            "tool_name": "search_emails",
            "server_name": "email",
            "state_changed": False,
        }],
        conversation_queries=["Find the release email."],
        oracle_calls_per_round=[calls],
        oracle_observations_per_round=[[
            {"emails": [{"email_id": "eml_0028", "subject": "Release"}]},
            {},
        ]],
        source_chain_seed=["search_emails"],
        server_tools=EMAIL_TOOLS,
        paper_baseline=False,
    )

    assert result.accepted is False
    assert "ids=['eml_0028']" in result.reason


def test_local_profile_allows_idempotent_mutation_after_real_change() -> None:
    calls = [
        OracleCall("search_emails", {"subject_contains": "migration"}),
        OracleCall("add_label", {"email_id": "eml_0011", "label": "urgent"}),
        OracleCall("add_label", {"email_id": "eml_0023", "label": "urgent"}),
        OracleCall("", {
            "text": "Both migration emails are labeled urgent.",
        }, action="final_answer"),
    ]
    result = validate_early_candidate_trace(
        domain="email",
        difficulty="minimal",
        plan=RobustnessPlan(),
        oracle_calls=calls,
        execution_history=[
            {
                "success": True,
                "tool_name": "search_emails",
                "server_name": "email",
                "state_changed": False,
            },
            {
                "success": True,
                "tool_name": "add_label",
                "server_name": "email",
                "state_changed": True,
            },
            {
                "success": True,
                "tool_name": "add_label",
                "server_name": "email",
                "state_changed": False,
            },
        ],
        conversation_queries=["Label the migration emails urgent."],
        oracle_calls_per_round=[calls],
        oracle_observations_per_round=[[
            {"emails": [
                {"email_id": "eml_0011", "labels": ["personal"]},
                {"email_id": "eml_0023", "labels": ["work", "urgent"]},
            ]},
            {"email_id": "eml_0011", "labels": ["personal", "urgent"]},
            {"email_id": "eml_0023", "labels": ["work", "urgent"]},
            {},
        ]],
        source_chain_seed=["search_emails", "add_label"],
        server_tools=EMAIL_TOOLS,
        paper_baseline=False,
    )

    assert result.accepted is True


def test_local_profile_rejects_hidden_tool_name_in_report_error() -> None:
    calls = [OracleCall(
        "", {"text": "The add_to_wishlist tool is unavailable."},
        action="report_error",
    )]
    result = validate_early_candidate_trace(
        domain="shopping",
        difficulty="complete",
        plan=RobustnessPlan(
            missing_function=True,
            hidden_tool="add_to_wishlist",
        ),
        oracle_calls=calls,
        execution_history=[],
        conversation_queries=["Add the Webcam Pro to my wishlist."],
        oracle_calls_per_round=[calls],
        source_chain_seed=["add_to_wishlist"],
        server_tools=SHOPPING_TOOLS,
        teacher_visible_tools=[
            tool for tool in SHOPPING_TOOLS
            if tool["name"] != "add_to_wishlist"
        ],
        paper_baseline=False,
    )

    assert result.accepted is False
    assert result.reason == (
        "user_visible_private_tool_name:round=0:"
        "tools=['add_to_wishlist']"
    )


def test_paper_profile_does_not_apply_local_tool_name_gate() -> None:
    calls = [OracleCall(
        "", {"text": "The add_to_wishlist tool is unavailable."},
        action="report_error",
    )]
    result = validate_early_candidate_trace(
        domain="shopping",
        difficulty="complete",
        plan=RobustnessPlan(
            missing_function=True,
            hidden_tool="add_to_wishlist",
        ),
        oracle_calls=calls,
        execution_history=[],
        conversation_queries=["Add the Webcam Pro to my wishlist."],
        oracle_calls_per_round=[calls],
        source_chain_seed=["add_to_wishlist"],
        server_tools=SHOPPING_TOOLS,
        teacher_visible_tools=[
            tool for tool in SHOPPING_TOOLS
            if tool["name"] != "add_to_wishlist"
        ],
        paper_baseline=True,
    )

    assert result.accepted is True


def test_local_profile_rejects_visible_tool_name_in_final_answer() -> None:
    leaked_calls = [
        OracleCall("list_webhooks", {}),
        OracleCall(
            "", {"text": "I used list_webhooks to check registrations."},
            action="final_answer",
        ),
    ]
    leaked = validate_early_candidate_trace(
        domain="payments",
        difficulty="complete",
        plan=RobustnessPlan(),
        oracle_calls=leaked_calls,
        execution_history=[{
            "success": True, "tool_name": "list_webhooks",
            "server_name": "payments", "state_changed": False,
        }],
        conversation_queries=["Show my registered webhooks."],
        oracle_calls_per_round=[leaked_calls],
        source_chain_seed=["list_webhooks"],
        server_tools=PAYMENT_TOOLS,
        teacher_visible_tools=PAYMENT_TOOLS,
        paper_baseline=False,
    )
    clean_calls = [
        leaked_calls[0],
        OracleCall(
            "", {"text": "I checked your registered webhooks."},
            action="final_answer",
        ),
    ]
    clean = validate_early_candidate_trace(
        domain="payments",
        difficulty="complete",
        plan=RobustnessPlan(),
        oracle_calls=clean_calls,
        execution_history=[{
            "success": True, "tool_name": "list_webhooks",
            "server_name": "payments", "state_changed": False,
        }],
        conversation_queries=["Show my registered webhooks."],
        oracle_calls_per_round=[clean_calls],
        source_chain_seed=["list_webhooks"],
        server_tools=PAYMENT_TOOLS,
        teacher_visible_tools=PAYMENT_TOOLS,
        paper_baseline=False,
    )

    assert leaked.accepted is False
    assert leaked.reason == (
        "user_visible_private_tool_name:round=0:tools=['list_webhooks']"
    )
    assert clean.accepted is True

def _validate_payment_continuation(
    first_round: list[OracleCall],
    second_round: list[OracleCall],
    observations: list[list[dict]],
    *,
    paper_baseline: bool = False,
):
    calls = first_round + second_round
    return validate_early_candidate_trace(
        domain="payments",
        difficulty="complete",
        plan=RobustnessPlan(),
        oracle_calls=calls,
        execution_history=[{
            "success": True,
            "tool_name": call.tool_name,
            "server_name": "payments",
            "state_changed": True,
        } for call in calls if call.action == "tool_call"],
        conversation_queries=["pay invoice A", "handle the same invoice"],
        oracle_calls_per_round=[first_round, second_round],
        oracle_observations_per_round=observations,
        source_chain_seed=[first_round[0].tool_name],
        server_tools=PAYMENT_TOOLS,
        paper_baseline=paper_baseline,
    )


def test_local_continuation_rejects_switch_to_different_typed_entity() -> None:
    first_round = [
        OracleCall("pay_invoice", {
            "invoice_id": "inv_A", "amount": 100, "method": "card",
        }),
        OracleCall("", {"text": "Invoice A was paid."}, action="final_answer"),
    ]
    second_round = [
        OracleCall("pay_invoice", {
            "invoice_id": "inv_B", "amount": 200, "method": "card",
        }),
        OracleCall("", {"text": "Invoice B was paid."}, action="final_answer"),
    ]

    result = _validate_payment_continuation(
        first_round,
        second_round,
        [[{"invoice_id": "inv_A", "payment_id": "pay_A"}, {}],
         [{"invoice_id": "inv_B", "payment_id": "pay_B"}, {}]],
    )

    assert result.accepted is True
    assert result.continuation_link_evidence[-1]["verification"] == (
        "diagnostic_unproven"
    )
    assert "no_exact_typed_entity_reuse" in (
        result.continuation_link_evidence[-1]["reason"]
    )


def test_local_continuation_accepts_same_typed_entity_and_records_evidence() -> None:
    first_round = [
        OracleCall("pay_invoice", {
            "invoice_id": "inv_A", "amount": 100, "method": "card",
        }),
        OracleCall("", {"text": "Invoice A was paid."}, action="final_answer"),
    ]
    second_round = [
        OracleCall("refund_invoice", {
            "invoice_id": "inv_A", "amount": 20,
        }),
        OracleCall("", {"text": "Invoice A was refunded."}, action="final_answer"),
    ]

    result = _validate_payment_continuation(
        first_round,
        second_round,
        [[{"invoice_id": "inv_A", "payment_id": "pay_A"}, {}],
         [{"invoice_id": "inv_A", "refund_id": "ref_A"}, {}]],
    )

    assert result.accepted is True
    assert result.continuation_link_evidence == [{
        "previous_round_idx": 0,
        "current_round_idx": 1,
        "entity_type": "invoice",
        "value": "inv_A",
        "previous_capability": "pay_invoice",
        "previous_field": "invoice_id",
        "previous_surface": "argument",
        "current_capability": "refund_invoice",
        "current_field": "invoice_id",
    }]


def test_local_continuation_accepts_entity_created_by_previous_observation() -> None:
    first_round = [
        OracleCall("create_invoice", {
            "customer": "Acme", "amount": 100,
        }),
        OracleCall("", {"text": "The invoice was created."}, action="final_answer"),
    ]
    second_round = [
        OracleCall("pay_invoice", {
            "invoice_id": "inv_new", "amount": 100, "method": "card",
        }),
        OracleCall("", {"text": "The invoice was paid."}, action="final_answer"),
    ]

    result = _validate_payment_continuation(
        first_round,
        second_round,
        [[{"invoice_id": "inv_new", "amount": 100}, {}],
         [{"invoice_id": "inv_new", "payment_id": "pay_new"}, {}]],
    )

    assert result.accepted is True
    assert result.continuation_link_evidence[0]["previous_surface"] == "observation"


def test_paper_continuation_does_not_apply_local_entity_link_gate() -> None:
    first_round = [
        OracleCall("pay_invoice", {
            "invoice_id": "inv_A", "amount": 100, "method": "card",
        }),
        OracleCall("", {"text": "Invoice A was paid."}, action="final_answer"),
    ]
    second_round = [
        OracleCall("pay_invoice", {
            "invoice_id": "inv_B", "amount": 200, "method": "card",
        }),
        OracleCall("", {"text": "Invoice B was paid."}, action="final_answer"),
    ]

    result = _validate_payment_continuation(
        first_round,
        second_round,
        [[{"invoice_id": "inv_A", "payment_id": "pay_A"}, {}],
         [{"invoice_id": "inv_B", "payment_id": "pay_B"}, {}]],
        paper_baseline=True,
    )

    assert result.accepted is True
    assert result.continuation_link_evidence == []


def test_local_continuation_links_polymorphic_entity_type_and_id_pair() -> None:
    first_round = [
        OracleCall("update_lead", {
            "lead_id": "lead_A", "fields": {"status": "contacted"},
        }),
        OracleCall("", {"text": "The lead was updated."}, action="final_answer"),
    ]
    second_round = [
        OracleCall("add_note", {
            "entity_type": "lead", "entity_id": "lead_A",
            "content": "Requested a demo.",
        }),
        OracleCall("", {"text": "The note was added."}, action="final_answer"),
    ]
    calls = first_round + second_round

    result = validate_early_candidate_trace(
        domain="crm",
        difficulty="complete",
        plan=RobustnessPlan(),
        oracle_calls=calls,
        execution_history=[{
            "success": True,
            "tool_name": call.tool_name,
            "server_name": "crm",
            "state_changed": True,
        } for call in calls if call.action == "tool_call"],
        conversation_queries=["update the lead", "add a note to that lead"],
        oracle_calls_per_round=[first_round, second_round],
        oracle_observations_per_round=[
            [{"lead_id": "lead_A"}, {}],
            [{"note_id": "note_A"}, {}],
        ],
        source_chain_seed=["update_lead"],
        server_tools=CRM_TOOLS,
        paper_baseline=False,
    )

    assert result.accepted is True
    assert result.continuation_link_evidence[0]["entity_type"] == "lead"


def test_local_continuation_keeps_lineage_across_direct_answer_round() -> None:
    first_round = [
        OracleCall("pay_invoice", {
            "invoice_id": "inv_A", "amount": 100, "method": "card",
        }),
        OracleCall("", {"text": "Invoice A was paid."}, action="final_answer"),
    ]
    direct_answer_round = [
        OracleCall("", {"text": "It is paid."}, action="final_answer"),
    ]
    third_round = [
        OracleCall("refund_invoice", {
            "invoice_id": "inv_A", "amount": 20,
        }),
        OracleCall("", {"text": "Invoice A was refunded."}, action="final_answer"),
    ]
    calls = first_round + direct_answer_round + third_round

    result = validate_early_candidate_trace(
        domain="payments",
        difficulty="complete",
        plan=RobustnessPlan(),
        oracle_calls=calls,
        execution_history=[{
            "success": True,
            "tool_name": call.tool_name,
            "server_name": "payments",
            "state_changed": True,
        } for call in calls if call.action == "tool_call"],
        conversation_queries=["pay invoice A", "is it paid?", "refund it"],
        oracle_calls_per_round=[
            first_round, direct_answer_round, third_round,
        ],
        oracle_observations_per_round=[
            [{"invoice_id": "inv_A", "payment_id": "pay_A"}, {}],
            [{}],
            [{"invoice_id": "inv_A", "refund_id": "ref_A"}, {}],
        ],
        source_chain_seed=["pay_invoice"],
        server_tools=PAYMENT_TOOLS,
        paper_baseline=False,
    )

    assert result.accepted is True
    assert result.continuation_link_evidence[0]["current_round_idx"] == 2


def test_local_continuation_does_not_invent_entity_mismatch_for_global_query() -> None:
    first_round = [
        OracleCall("delete_webhook", {"webhook_id": "wh_A"}),
        OracleCall("", {"text": "The webhook was deleted."}, action="final_answer"),
    ]
    second_round = [
        OracleCall("list_webhooks", {}),
        OracleCall("", {"text": "Here are the remaining webhooks."}, action="final_answer"),
    ]
    calls = first_round + second_round

    result = validate_early_candidate_trace(
        domain="payments",
        difficulty="complete",
        plan=RobustnessPlan(),
        oracle_calls=calls,
        execution_history=[{
            "success": True,
            "tool_name": call.tool_name,
            "server_name": "payments",
            "state_changed": call.tool_name == "delete_webhook",
        } for call in calls if call.action == "tool_call"],
        conversation_queries=["delete webhook A", "list what remains"],
        oracle_calls_per_round=[first_round, second_round],
        oracle_observations_per_round=[
            [{"webhook_id": "wh_A", "deleted": True}, {}],
            [{"webhooks": []}, {}],
        ],
        source_chain_seed=["delete_webhook"],
        server_tools=PAYMENT_TOOLS,
        paper_baseline=False,
    )

    assert result.accepted is True
    assert result.continuation_link_evidence[-1]["verification"] == (
        "not_applicable_no_typed_entity_input"
    )


def test_rejects_successful_distractor_from_ground_truth() -> None:
    calls = [
        OracleCall("list_events", {}),
        OracleCall("", {"text": "done"}, action="final_answer"),
    ]
    result = _validate(
        calls,
        plan=RobustnessPlan(
            inject_distractors=True,
            distractor_tools=[{"name": "list_events"}],
        ),
        source_chain=["list_events"],
    )

    assert result.accepted is False
    assert result.reason.startswith("successful_distractor_calls:")


def test_rejects_complete_task_with_no_tool_and_final_answer() -> None:
    result = _validate([
        OracleCall("", {"text": "done"}, action="final_answer"),
    ])

    assert result.accepted is False
    assert result.reason == "invalid_zero_tool_terminal"


def test_paper_profile_rejects_unauthorized_missing_function_mutation() -> None:
    calls = [
        OracleCall("chown", {"path": "/tmp/a", "owner": "everyone"}),
        OracleCall("", {"text": "chmod is unavailable"}, action="report_error"),
    ]
    result = validate_early_candidate_trace(
        domain="filesystem",
        difficulty="complete",
        plan=RobustnessPlan(
            missing_function=True, hidden_tool="chmod",
            missing_function_evidence=("file.mode=readonly",),
        ),
        oracle_calls=calls,
        execution_history=[{
            "success": True,
            "tool_name": "chown",
            "server_name": "filesystem",
            "state_changed": True,
        }],
        conversation_queries=["make the file executable"],
        oracle_calls_per_round=[calls],
        source_chain_seed=["symlink", "chmod"],
        server_tools=[{
            "name": "chown",
            "annotations": {"readonly": False, "mutating": True},
        }],
        paper_baseline=True,
    )

    assert result.accepted is False
    assert result.reason == "missing_function_mutation"


def test_missing_function_rejects_source_chain_prefix_mutation() -> None:
    calls = [
        OracleCall("symlink", {
            "target": "/tmp/source", "link_path": "/tmp/link",
        }),
        OracleCall("", {"text": "permission change unavailable"}, action="report_error"),
    ]
    result = validate_early_candidate_trace(
        domain="filesystem",
        difficulty="complete",
        plan=RobustnessPlan(
            missing_function=True, hidden_tool="chmod",
            missing_function_evidence=("file.mode=readonly",),
        ),
        oracle_calls=calls,
        execution_history=[{
            "success": True,
            "tool_name": "symlink",
            "server_name": "filesystem",
            "state_changed": True,
        }],
        conversation_queries=["create a link and make it read-only"],
        oracle_calls_per_round=[calls],
        source_chain_seed=["symlink", "chmod"],
        server_tools=[{
            "name": "symlink",
            "annotations": {"readonly": False, "mutating": True},
        }],
        paper_baseline=True,
    )

    assert result.accepted is False
    assert result.reason == "missing_function_mutation"


def test_missing_function_requires_hidden_tool_to_be_chain_final() -> None:
    calls = [
        OracleCall("", {"text": "unavailable"}, action="report_error"),
    ]
    result = validate_early_candidate_trace(
        domain="filesystem",
        difficulty="complete",
        plan=RobustnessPlan(
            missing_function=True, hidden_tool="symlink",
            missing_function_evidence=("file.link=true",),
        ),
        oracle_calls=calls,
        execution_history=[],
        conversation_queries=["Create a link and change its permissions."],
        oracle_calls_per_round=[calls],
        source_chain_seed=["symlink", "chmod"],
        server_tools=[],
        paper_baseline=True,
    )

    assert result.accepted is False
    assert result.reason == "missing_function_hidden_target_mismatch"


def test_missing_function_rejects_unresolved_visible_execution_failure() -> None:
    calls = [
        OracleCall("list_accounts", {}),
        OracleCall("", {"text": "The requested capability is unavailable."},
                   action="report_error"),
    ]
    result = validate_early_candidate_trace(
        domain="banking",
        difficulty="complete",
        plan=RobustnessPlan(
            missing_function=True, hidden_tool="get_statement",
        ),
        oracle_calls=calls,
        execution_history=[{
            "tool_name": "list_accounts",
            "arguments": {},
            "success": False,
            "state_changed": False,
            "server_name": "banking",
        }],
        conversation_queries=["Generate my account statement."],
        oracle_calls_per_round=[calls],
        source_chain_seed=["list_accounts", "get_statement"],
        server_tools=BANKING_TOOLS,
        paper_baseline=True,
    )

    assert result.accepted is False
    assert result.reason == (
        "missing_function_unresolved_execution_failure:"
        "tools=['list_accounts']"
    )


def test_missing_function_allows_recovered_visible_execution_failure() -> None:
    calls = [
        OracleCall("list_accounts", {}),
        OracleCall("", {"text": "The requested capability is unavailable."},
                   action="report_error"),
    ]
    result = validate_early_candidate_trace(
        domain="banking",
        difficulty="complete",
        plan=RobustnessPlan(
            missing_function=True, hidden_tool="get_statement",
        ),
        oracle_calls=calls,
        execution_history=[{
            "tool_name": "list_accounts",
            "arguments": {},
            "success": False,
            "state_changed": False,
            "server_name": "banking",
        }, {
            "tool_name": "list_accounts",
            "arguments": {},
            "success": True,
            "state_changed": False,
            "server_name": "banking",
        }],
        conversation_queries=["Generate my account statement."],
        oracle_calls_per_round=[calls],
        source_chain_seed=["list_accounts", "get_statement"],
        server_tools=BANKING_TOOLS,
        paper_baseline=True,
    )

    assert result.accepted is True


def test_failure_recovery_uses_observed_selector_to_canonical_id_alias() -> None:
    history = [{
        "tool_name": "deposit",
        "arguments": {"account_id": "1861", "amount": 500},
        "success": False,
        "server_name": "banking",
    }, {
        "tool_name": "list_accounts",
        "arguments": {},
        "success": True,
        "server_name": "banking",
        "observation": {"accounts": [{
            "account_id": "acc_private_003",
            "account_last4": "1861",
        }]},
    }, {
        "tool_name": "deposit",
        "arguments": {"account_id": "acc_private_003", "amount": 500},
        "success": True,
        "server_name": "banking",
    }]

    assert unresolved_failed_tool_names(history) == set()


def test_failure_recovery_does_not_cross_unrelated_resource_ids() -> None:
    history = [{
        "tool_name": "deposit",
        "arguments": {"account_id": "1861", "amount": 500},
        "success": False,
        "server_name": "banking",
    }, {
        "tool_name": "deposit",
        "arguments": {"account_id": "acc_private_999", "amount": 500},
        "success": True,
        "server_name": "banking",
    }]

    assert unresolved_failed_tool_names(history) == {"deposit"}


def test_non_missing_report_error_keeps_real_failure_recovery_path() -> None:
    calls = [
        OracleCall("list_accounts", {}),
        OracleCall("", {"text": "The account service rejected the request."},
                   action="report_error"),
    ]
    result = validate_early_candidate_trace(
        domain="banking",
        difficulty="complete",
        plan=RobustnessPlan(),
        oracle_calls=calls,
        execution_history=[{
            "tool_name": "list_accounts",
            "arguments": {},
            "success": True,
            "state_changed": False,
            "server_name": "banking",
        }, {
            "tool_name": "get_balance",
            "arguments": {"account_id": "acc_001"},
            "success": False,
            "state_changed": False,
            "server_name": "banking",
        }],
        conversation_queries=["Show my accounts."],
        oracle_calls_per_round=[calls],
        source_chain_seed=["list_accounts"],
        server_tools=BANKING_TOOLS,
        paper_baseline=True,
    )

    assert result.accepted is True
    assert result.scenario_type == "tool_error_recovery"


def test_non_missing_report_error_accepts_empty_read_result_evidence() -> None:
    calls = [
        OracleCall("list_accounts", {}),
        OracleCall("", {"text": "No matching account was found."},
                   action="report_error"),
    ]
    result = validate_early_candidate_trace(
        domain="banking",
        difficulty="complete",
        plan=RobustnessPlan(),
        oracle_calls=calls,
        execution_history=[{
            "tool_name": "list_accounts",
            "arguments": {},
            "success": True,
            "execution_status": "PARTIAL_SUCCESS",
            "state_changed": False,
            "server_name": "banking",
            "observation": {"accounts": []},
        }],
        conversation_queries=["Find my business account."],
        oracle_calls_per_round=[calls],
        source_chain_seed=["list_accounts"],
        server_tools=BANKING_TOOLS,
        paper_baseline=True,
    )

    assert result.accepted is True
    assert result.terminal_action == "report_error"


def test_missing_function_ignores_non_execution_rejection_marker() -> None:
    calls = [
        OracleCall("", {"text": "The requested capability is unavailable."},
                   action="report_error"),
    ]
    result = validate_early_candidate_trace(
        domain="banking",
        difficulty="complete",
        plan=RobustnessPlan(
            missing_function=True, hidden_tool="get_statement",
        ),
        oracle_calls=calls,
        execution_history=[{
            "tool_name": "__reject__",
            "arguments": {},
            "success": False,
            "state_changed": False,
            "server_name": "banking",
        }],
        conversation_queries=["Generate my account statement."],
        oracle_calls_per_round=[calls],
        source_chain_seed=["list_accounts", "get_statement"],
        server_tools=BANKING_TOOLS,
        paper_baseline=True,
    )

    assert result.accepted is True


def test_local_minimal_allows_clarification_for_missing_non_entity_value() -> None:
    calls = [
        OracleCall("list_accounts", {"type": "business"}),
        OracleCall(
            "ask_clarification",
            {"question": "Where should the deposit come from?"},
            action="ask_clarification",
        ),
    ]
    result = validate_early_candidate_trace(
        domain="banking",
        difficulty="minimal",
        plan=RobustnessPlan(),
        oracle_calls=calls,
        execution_history=[{
            "success": True,
            "tool_name": "list_accounts",
            "server_name": "banking",
            "state_changed": False,
        }],
        conversation_queries=["put $1,200 into my business account"],
        oracle_calls_per_round=[calls],
        source_chain_seed=["list_accounts", "deposit"],
        server_tools=[],
        paper_baseline=False,
        oracle_observations=[{
            "accounts": [{"account_id": "acc_003", "type": "business"}],
        }, {}],
        dependency_contracts=[{
            "source_capability": "list_accounts",
            "target_capability": "deposit",
            "target_argument": "account_id",
            "source_output_field": "account_id",
        }],
    )

    assert result.accepted is True


def test_local_minimal_retains_clarification_for_multiple_dependency_values() -> None:
    calls = [
        OracleCall("list_accounts", {}),
        OracleCall(
            "ask_clarification",
            {"question": "Which savings and checking accounts?"},
            action="ask_clarification",
        ),
    ]
    result = validate_early_candidate_trace(
        domain="banking",
        difficulty="minimal",
        plan=RobustnessPlan(),
        oracle_calls=calls,
        execution_history=[{
            "success": True,
            "tool_name": "list_accounts",
            "server_name": "banking",
            "state_changed": False,
        }],
        conversation_queries=[
            "schedule $100 from my savings to checking next Friday"
        ],
        oracle_calls_per_round=[calls],
        source_chain_seed=["list_accounts", "schedule_transfer"],
        server_tools=[],
        paper_baseline=False,
        oracle_observations=[{
            "accounts": [
                {"account_id": "acc_s1", "type": "savings"},
                {"account_id": "acc_s2", "type": "savings"},
                {"account_id": "acc_c1", "type": "checking"},
            ],
        }, {}],
        dependency_contracts=[{
            "source_capability": "list_accounts",
            "target_capability": "schedule_transfer",
            "target_argument": "from_account",
            "source_output_field": "account_id",
        }, {
            "source_capability": "list_accounts",
            "target_capability": "schedule_transfer",
            "target_argument": "to_account",
            "source_output_field": "account_id",
        }],
    )

    assert result.accepted is True
    assert result.scenario_type == "clarification_required"


@pytest.mark.parametrize(
    ("difficulty", "paper_baseline"),
    [("missing", False), ("minimal", True)],
)
def test_unique_dependency_clarification_gate_respects_profile_boundary(
    difficulty: str, paper_baseline: bool,
) -> None:
    calls = [
        OracleCall("list_accounts", {"type": "business"}),
        OracleCall(
            "ask_clarification", {"question": "Please provide one detail."},
            action="ask_clarification",
        ),
    ]
    result = validate_early_candidate_trace(
        domain="banking",
        difficulty=difficulty,
        plan=RobustnessPlan(),
        oracle_calls=calls,
        execution_history=[{
            "success": True,
            "tool_name": "list_accounts",
            "server_name": "banking",
            "state_changed": False,
        }],
        conversation_queries=["put money into my business account"],
        oracle_calls_per_round=[calls],
        source_chain_seed=["list_accounts", "deposit"],
        server_tools=[],
        paper_baseline=paper_baseline,
        oracle_observations=[{
            "accounts": [{"account_id": "acc_003"}],
        }, {}],
        dependency_contracts=[{
            "source_capability": "list_accounts",
            "target_capability": "deposit",
            "target_argument": "account_id",
            "source_output_field": "account_id",
        }],
    )

    assert result.accepted is True
