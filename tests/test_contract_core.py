from __future__ import annotations

import importlib
import random
from types import SimpleNamespace

import pytest

from src.live_mcp.contracts.abstract_state import AbstractState, simulate_contract_chain
from src.live_mcp.contracts.models import (
    ArgumentValue,
    EntityBinding,
    StatePredicate,
    ToolContract,
)
from src.live_mcp.contracts.registry import ContractRegistry
from src.live_mcp.contracts.chain_simulator import simulate_symbolic_chain
from src.live_mcp.contracts.factory import build_contract_registry
from src.live_mcp.domain_contracts.states import DOMAIN_STATE_FACTS
from src.live_mcp.dependency_value_flow import (
    _difficulty_vector_for_chain,
    _operational_dependency_contracts,
)
from src.live_mcp.generation.chain_scheduler import ChainSchedulerMixin


DOMAINS = (
    "banking", "calendar", "crm", "email", "filesystem",
    "food_delivery", "issue_tracker", "payments", "shopping", "team_chat",
)


def _order_status(value: str, source: str = "argument") -> StatePredicate:
    return StatePredicate(
        slot="order.status",
        subject=EntityBinding("order", "order_id", source),
        value=value,
    )


def test_state_simulator_rejects_conflicting_downstream_precondition() -> None:
    create = ToolContract(
        domain="probe",
        name="create_order",
        readonly=False,
        mutating=True,
        output_entities=(EntityBinding("order", "order_id", "output"),),
        output_fields=frozenset({"order_id"}),
        postconditions=(_order_status("placed", "output"),),
    )
    return_order = ToolContract(
        domain="probe",
        name="return_order",
        readonly=False,
        mutating=True,
        required_arguments=frozenset({"order_id"}),
        input_entities=(EntityBinding("order", "order_id"),),
        preconditions=(_order_status("shipped"),),
    )

    _, issues = simulate_contract_chain(
        [create, return_order],
        [{"order_id": "symbol:created-order"}, {"order_id": "symbol:created-order"}],
    )

    assert len(issues) == 1
    assert issues[0].reason == "contradicted"


def test_state_simulator_accepts_matching_transition() -> None:
    create = ToolContract(
        domain="probe",
        name="create_order",
        readonly=False,
        mutating=True,
        output_entities=(EntityBinding("order", "order_id", "output"),),
        output_fields=frozenset({"order_id"}),
        postconditions=(_order_status("shipped", "output"),),
    )
    return_order = ToolContract(
        domain="probe",
        name="return_order",
        readonly=False,
        mutating=True,
        required_arguments=frozenset({"order_id"}),
        input_entities=(EntityBinding("order", "order_id"),),
        preconditions=(_order_status("shipped"),),
    )

    _, issues = simulate_contract_chain(
        [create, return_order],
        [{"order_id": "symbol:created-order"}, {"order_id": "symbol:created-order"}],
    )

    assert issues == ()


def test_symbolic_simulator_keeps_argument_driven_transition_unknown() -> None:
    update = ToolContract(
        domain="probe",
        name="update_order_status",
        readonly=False,
        mutating=True,
        arguments=frozenset({"order_id", "status"}),
        postconditions=(StatePredicate(
            slot="order.status",
            subject=EntityBinding("order", "order_id"),
            value=ArgumentValue("status"),
        ),),
    )
    track = ToolContract(
        domain="probe",
        name="track_rider",
        readonly=True,
        mutating=False,
        arguments=frozenset({"order_id"}),
        preconditions=(_order_status("delivering"),),
    )

    registry = ContractRegistry([update, track])
    _, issues = simulate_symbolic_chain(
        registry, "probe", ["update_order_status", "track_rider"],
    )

    assert issues == ()


def test_concrete_simulator_resolves_argument_driven_transition() -> None:
    update = ToolContract(
        domain="probe",
        name="update_order_status",
        readonly=False,
        mutating=True,
        postconditions=(StatePredicate(
            slot="order.status",
            subject=EntityBinding("order", "order_id"),
            value=ArgumentValue("status"),
        ),),
    )
    track = ToolContract(
        domain="probe",
        name="track_rider",
        readonly=True,
        mutating=False,
        preconditions=(_order_status("delivering"),),
    )

    _, issues = simulate_contract_chain(
        [update, track],
        [
            {"order_id": "order:1", "argument:status": "delivering"},
            {"order_id": "order:1"},
        ],
    )

    assert issues == ()


def test_registry_fails_closed_on_missing_or_duplicate_contract() -> None:
    contract = ToolContract(
        domain="probe", name="read", readonly=True, mutating=False,
    )
    registry = ContractRegistry([contract])
    with pytest.raises(ValueError, match="Duplicate"):
        registry.register(contract)
    with pytest.raises(KeyError, match="Missing tool contract"):
        registry.get("probe", "missing")


def test_registry_audits_live_schema_annotations_and_required_arguments() -> None:
    registry = ContractRegistry([
        ToolContract(
            domain="probe",
            name="read",
            readonly=True,
            mutating=False,
            required_arguments=frozenset({"item_id"}),
            input_entities=(EntityBinding("item", "item_id"),),
        ),
    ])

    assert registry.audit_schema("probe", [{
        "name": "read",
        "input_schema": {"required": ["item_id"]},
        "annotations": {"readonly": True, "mutating": False},
    }]) == ()


def test_all_190_tools_have_one_registered_state_contract() -> None:
    domain_tools = {
        domain: importlib.import_module(
            f"src.live_mcp.servers.{domain}.server"
        ).TOOLS
        for domain in DOMAINS
    }
    registry = build_contract_registry(domain_tools)

    assert sum(registry.coverage().values()) == 190
    for domain, tools in domain_tools.items():
        names = {tool["name"] for tool in tools}
        assert set(DOMAIN_STATE_FACTS[domain]) == names
        assert registry.audit_schema(domain, tools) == ()


def test_prove_dependency_vector_consumes_the_supplied_live_schemas() -> None:
    tools = importlib.import_module(
        "src.live_mcp.servers.food_delivery.server"
    ).TOOLS
    chain = [
        "create_order", "update_order_status", "track_rider",
    ]

    contracts = _operational_dependency_contracts(
        chain, "food_delivery", tools,
    )
    vector = _difficulty_vector_for_chain(
        chain=chain,
        server_name="food_delivery",
        server_tools=tools,
        chain_context={"entity_records": [], "opaque_id_hidden_types": []},
        feasible_chains=[chain],
        distractor_count=0,
    )

    assert isinstance(contracts, list)
    assert vector.oracle_tool_count == len(chain)


def test_operational_dependency_excludes_optional_argument_echoes() -> None:
    tools = importlib.import_module(
        "src.live_mcp.servers.filesystem.server"
    ).TOOLS

    assert _operational_dependency_contracts(
        ["ls", "touch"], "filesystem", tools,
    ) == []
    assert _operational_dependency_contracts(
        ["tar_create", "tar_extract"], "filesystem", tools,
    ) == []


def test_paper_chain_scheduler_binds_static_fingerprint_correctly() -> None:
    scheduler = ChainSchedulerMixin()
    scheduler.prompt_profile = SimpleNamespace(paper_baseline=True)
    scheduler._chain_sampling_stats = {}
    scheduler._chain_sampling_sequences = {}
    import threading
    scheduler._chain_sampling_lock = threading.RLock()

    chain, fingerprint, attempt, novel = scheduler._select_feasible_chain(
        "banking", [["list_accounts", "transfer"]], random.Random(42),
    )

    assert chain == ["list_accounts", "transfer"]
    assert fingerprint == scheduler._chain_fingerprint("banking", chain)
    assert (attempt, novel) == (1, False)


def test_paper_chain_scheduler_exhausts_unattempted_paths_before_repeat() -> None:
    scheduler = ChainSchedulerMixin()
    scheduler.prompt_profile = SimpleNamespace(paper_baseline=True)
    scheduler._chain_sampling_stats = {}
    scheduler._chain_sampling_sequences = {}
    import threading
    scheduler._chain_sampling_lock = threading.RLock()
    chains = [
        ["list_accounts", "transfer"],
        ["get_balance", "transfer"],
        ["list_accounts", "get_statement"],
    ]

    selected = [
        tuple(scheduler._select_feasible_chain(
            "banking", chains, random.Random(seed),
        )[0])
        for seed in range(3)
    ]
    assert len(set(selected)) == 3

    fingerprint = scheduler._chain_fingerprint("banking", list(selected[0]))
    scheduler._record_chain_rejected("banking", fingerprint, "goal_unsat")
    assert scheduler._chain_sampling_summary("banking")[
        "rejected_goal_total"
    ] == 1


def test_entity_requirements_are_derived_from_state_contracts() -> None:
    domain_tools = {
        domain: importlib.import_module(
            f"src.live_mcp.servers.{domain}.server"
        ).TOOLS
        for domain in DOMAINS
    }
    registry = build_contract_registry(domain_tools)

    assert registry.get("payments", "list_invoices").required_entity_types == frozenset()
    assert registry.get("shopping", "checkout").required_entity_types == frozenset()
    assert registry.get("shopping", "update_cart_quantity").required_entity_types == frozenset({"product"})
    assert dict(registry.get("banking", "transfer").minimum_entity_counts) == {"account": 2}
    assert dict(registry.get("filesystem", "join").minimum_entity_counts) == {"file": 2}
    assert dict(registry.get("shopping", "compare_products").minimum_entity_counts) == {"product": 2}
    assert [
        {predicate.subject.entity_type for predicate in group}
        for group in registry.get("crm", "create_deal").precondition_groups
    ] == [{"contact", "lead"}]


@pytest.mark.parametrize(
    ("domain", "chain", "blocked_tool"),
    (
        ("calendar", ["create_event", "get_recurring_info"], "get_recurring_info"),
        ("payments", ["pay_invoice", "cancel_payment"], "cancel_payment"),
        ("shopping", ["checkout", "return_order"], "return_order"),
        ("food_delivery", ["create_order", "track_rider"], "track_rider"),
    ),
)
def test_symbolic_state_simulator_rejects_cross_domain_state_conflicts(
    domain: str,
    chain: list[str],
    blocked_tool: str,
) -> None:
    tools = {
        item: importlib.import_module(
            f"src.live_mcp.servers.{item}.server"
        ).TOOLS
        for item in DOMAINS
    }
    registry = build_contract_registry(tools)

    _, issues = simulate_symbolic_chain(registry, domain, chain)

    assert issues
    assert issues[0].tool_name == blocked_tool
