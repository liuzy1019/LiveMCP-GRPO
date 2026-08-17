from types import SimpleNamespace

import src.live_mcp.generation.context_provider as context_provider
import src.live_mcp.dependency_value_flow as dependency_value_flow
import src.live_mcp.generation.candidate_prepare as candidate_prepare
from src.live_mcp.contracts.factory import build_contract_registry
from src.live_mcp.dependency_chain_policy import (
    chain_contract_issue,
    goal_coherence_issue,
    scenario_chain_issue,
)
from src.live_mcp.dependency_value_flow import (
    _filter_relation_verifiable_chains,
)
from src.live_mcp.generation.context_provider import GenerationContextMixin
from src.live_mcp.servers.shopping.server import TOOLS as SHOPPING_TOOLS
from src.live_mcp.servers.calendar.server import TOOLS as CALENDAR_TOOLS
from src.live_mcp.servers.crm.server import TOOLS as CRM_TOOLS
from src.live_mcp.servers.food_delivery.server import TOOLS as FOOD_TOOLS
from src.live_mcp.servers.filesystem.server import TOOLS as FILESYSTEM_TOOLS
from src.live_mcp.servers.payments.server import TOOLS as PAYMENT_TOOLS
from src.live_mcp.servers.team_chat.server import TOOLS as TEAM_CHAT_TOOLS
from src.live_mcp.servers.issue_tracker.server import TOOLS as ISSUE_TOOLS


def test_candidate_prepare_uses_canonical_dependency_contract_builder() -> None:
    assert candidate_prepare._operational_dependency_contracts is (
        dependency_value_flow._operational_dependency_contracts
    )


def test_banking_rejects_synthetic_schedule_then_cancel_goal() -> None:
    assert chain_contract_issue(
        "banking", ["schedule_transfer", "cancel_transfer"],
    ) == "synthetic_scheduled_transfer_reversal"
    assert chain_contract_issue(
        "banking", ["list_scheduled_transfers", "cancel_transfer"],
    ) is None


def test_shopping_mixed_lifecycle_chain_is_rejected_by_contract() -> None:
    chain = [
        "get_return_status",
        "add_to_cart",
        "checkout",
        "get_order",
        "remove_from_wishlist",
    ]

    assert chain_contract_issue("shopping", chain) == (
        "chain_mixes_return_and_purchase_lifecycles"
    )


def test_adjacent_wishlist_chain_remains_eligible() -> None:
    assert chain_contract_issue(
        "shopping", ["get_wishlist", "remove_from_wishlist"],
    ) is None


def test_goal_coherence_rejects_mutation_unrelated_to_readonly_target() -> None:
    registry = build_contract_registry({"payments": PAYMENT_TOOLS})
    assert goal_coherence_issue(
        registry,
        "payments",
        ["cancel_payment", "list_invoices"],
    ) == "independent_prior_mutation:cancel_payment"


def test_goal_coherence_rejects_state_reversal() -> None:
    registry = build_contract_registry({"calendar": CALENDAR_TOOLS})
    assert goal_coherence_issue(
        registry,
        "calendar",
        ["list_events", "add_attendee", "remove_attendee"],
    ) == "mutation_reversal:add_attendee->remove_attendee"


def test_goal_coherence_rejects_hidden_state_on_new_entity() -> None:
    registry = build_contract_registry({"calendar": CALENDAR_TOOLS})

    assert goal_coherence_issue(
        registry, "calendar", ["create_event", "remove_attendee"],
    ) == "created_entity_hidden_precondition:remove_attendee:attendee.member"
    assert goal_coherence_issue(
        registry, "calendar", ["create_recurring", "respond_to_event"],
    ) == "created_entity_hidden_precondition:respond_to_event:attendee.member"
    assert goal_coherence_issue(
        registry, "calendar", ["create_event", "add_attendee"],
    ) is None


def test_goal_coherence_rejects_independent_prior_mutation() -> None:
    registry = build_contract_registry({"payments": PAYMENT_TOOLS})
    assert goal_coherence_issue(
        registry,
        "payments",
        ["cancel_payment", "create_invoice", "pay_invoice"],
    ) == "independent_prior_mutation:cancel_payment"


def test_goal_coherence_keeps_contract_linked_mutation_workflows() -> None:
    crm = build_contract_registry({"crm": CRM_TOOLS})
    payments = build_contract_registry({"payments": PAYMENT_TOOLS})
    assert goal_coherence_issue(
        crm, "crm", ["create_lead", "convert_lead"],
    ) is None
    assert goal_coherence_issue(
        payments,
        "payments",
        ["create_invoice", "pay_invoice", "refund_invoice"],
    ) is None
    food = build_contract_registry({"food_delivery": FOOD_TOOLS})
    assert goal_coherence_issue(
        food,
        "food_delivery",
        ["create_order", "get_order", "get_menu"],
    ) is None


def test_goal_coherence_rejects_redundant_create_then_update() -> None:
    crm = build_contract_registry({"crm": CRM_TOOLS})

    assert goal_coherence_issue(
        crm, "crm", ["create_deal", "update_deal"],
    ) == "redundant_create_update:create_deal->update_deal"
    assert goal_coherence_issue(
        crm, "crm", ["create_lead", "convert_lead"],
    ) is None


def test_goal_coherence_rejects_target_effect_already_set_by_creator() -> None:
    issue_tracker = build_contract_registry({"issue_tracker": ISSUE_TOOLS})

    assert goal_coherence_issue(
        issue_tracker,
        "issue_tracker",
        ["create_issue", "remove_from_sprint"],
    ) == (
        "created_entity_effect_already_satisfied:"
        "remove_from_sprint:issue.in_sprint"
    )


def test_missing_function_scenario_rejects_mutating_prefix() -> None:
    payments = build_contract_registry({"payments": PAYMENT_TOOLS})

    assert scenario_chain_issue(
        payments,
        "payments",
        ["list_invoices", "cancel_payment", "dispute_invoice"],
        difficulty="complete",
        missing_function=True,
    ) == "missing_function_mutating_prefix:cancel_payment"
    assert scenario_chain_issue(
        payments,
        "payments",
        ["list_invoices", "dispute_invoice"],
        difficulty="complete",
        missing_function=True,
    ) is None


def test_incomplete_scenario_rejects_multiple_mutations_only() -> None:
    payments = build_contract_registry({"payments": PAYMENT_TOOLS})
    chain = ["cancel_payment", "get_invoice", "pay_invoice"]

    assert scenario_chain_issue(
        payments,
        "payments",
        chain,
        difficulty="minimal",
        missing_function=False,
    ) == "incomplete_multi_mutation:cancel_payment,pay_invoice"
    assert scenario_chain_issue(
        payments,
        "payments",
        chain,
        difficulty="complete",
        missing_function=False,
    ) is None


def test_food_delivery_rejects_reordering_a_just_created_order() -> None:
    food = build_contract_registry({"food_delivery": FOOD_TOOLS})

    assert chain_contract_issue(
        "food_delivery", ["create_order", "reorder", "cancel_order"],
    ) == "new_order_cannot_be_reordered_as_history"
    assert goal_coherence_issue(
        food, "food_delivery", ["create_order", "reorder"],
    ) is None


def test_chain_prefix_echo_does_not_create_a_new_dependency_contract() -> None:
    contracts = dependency_value_flow._operational_dependency_contracts(
        ["create_order", "get_order", "get_menu"],
        "food_delivery",
        FOOD_TOOLS,
    )

    assert not any(
        item["source_capability"] == "get_order"
        and item["target_capability"] == "get_menu"
        for item in contracts
    )


def test_chain_prefix_echo_does_not_justify_an_unrelated_mutation() -> None:
    registry = build_contract_registry({"team_chat": TEAM_CHAT_TOOLS})

    assert goal_coherence_issue(
        registry,
        "team_chat",
        ["send_message", "create_thread", "archive_channel"],
    ) == "independent_prior_mutation:create_thread"


def test_filesystem_copy_then_delete_created_target_is_a_reversal() -> None:
    registry = build_contract_registry({"filesystem": FILESYSTEM_TOOLS})

    assert goal_coherence_issue(
        registry, "filesystem", ["cp", "rm"],
    ) == "mutation_reversal:cp->rm"


def test_canonical_feasibility_filter_executes_domain_chain_contract(
    monkeypatch,
) -> None:
    subject = GenerationContextMixin()
    subject.manager = SimpleNamespace(
        registry=SimpleNamespace(server_tools=lambda _domain: SHOPPING_TOOLS),
    )
    monkeypatch.setattr(
        context_provider,
        "simulate_symbolic_chain",
        lambda *_args, **_kwargs: ({}, []),
    )
    monkeypatch.setattr(
        context_provider,
        "chain_is_feasible",
        lambda *_args, **_kwargs: (True, ""),
    )
    mixed = [
        "get_return_status",
        "add_to_cart",
        "checkout",
        "get_order",
        "remove_from_wishlist",
    ]
    valid = ["get_wishlist", "remove_from_wishlist"]

    retained = subject._filter_feasible_chains(
        [mixed, valid],
        "shopping",
        {"entity_ids": [], "observed_entity_count": 0},
    )

    assert retained == [valid]


def test_relation_precheck_keeps_implicit_edge_without_value_contract(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        dependency_value_flow,
        "_operational_dependency_contracts",
        lambda *_args, **_kwargs: [],
    )
    implicit_chain = ["create_resource", "read_resource"]
    explicit_chain = ["discover_resource", "read_resource"]
    graph = {
        "create_resource": {
            "explicit": [],
            "implicit": ["read_resource"],
        },
        "discover_resource": {
            "explicit": ["read_resource"],
            "implicit": [],
        },
    }

    retained, issues = _filter_relation_verifiable_chains(
        [implicit_chain, explicit_chain], graph, "filesystem", [],
    )

    assert retained == [implicit_chain]
    assert issues == {
        "explicit_edge_without_value_contract:discover_resource->read_resource": 1,
    }


def test_relation_precheck_requires_adjacent_explicit_contract(
    monkeypatch,
) -> None:
    chain = ["discover_resource", "read_resource"]
    graph = {
        "discover_resource": {
            "explicit": ["read_resource"],
            "implicit": [],
        },
    }
    monkeypatch.setattr(
        dependency_value_flow,
        "_operational_dependency_contracts",
        lambda *_args, **_kwargs: [{
            "source_capability": "discover_resource",
            "target_capability": "read_resource",
            "target_argument": "path",
            "source_output_field": "path",
        }],
    )

    retained, issues = _filter_relation_verifiable_chains(
        [chain], graph, "filesystem", [],
    )

    assert retained == [chain]
    assert issues == {}
