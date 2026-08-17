from __future__ import annotations

import importlib
import random
from types import SimpleNamespace

import pytest

from src.live_mcp.contracts.models import (
    ArgumentValue,
    EntityBinding,
    StatePredicate,
    ToolContract,
)
from src.live_mcp.contracts.registry import ContractRegistry
from src.live_mcp.contracts.record_evaluator import record_state_is_known
from src.live_mcp.contracts.chain_simulator import simulate_symbolic_chain
from src.live_mcp.contracts.factory import build_contract_registry
from src.live_mcp.contracts.value_flow import chain_novel_output_fields
from src.live_mcp.domain_contracts.states import DOMAIN_STATE_FACTS
from src.live_mcp.dependency_value_flow import (
    _operational_dependency_contracts,
)
from src.live_mcp.generation.chain_scheduler import ChainSchedulerMixin


DOMAINS = (
    "banking", "calendar", "crm", "email", "filesystem",
    "food_delivery", "issue_tracker", "payments", "shopping", "team_chat",
)


def _domain_registry(*domains: str) -> ContractRegistry:
    return build_contract_registry({
        domain: importlib.import_module(
            f"src.live_mcp.servers.{domain}.server"
        ).TOOLS
        for domain in domains
    })


def test_state_sensitive_target_fails_closed_when_field_is_missing() -> None:
    contract = _domain_registry("food_delivery").get(
        "food_delivery", "create_order",
    )
    assert not record_state_is_known(
        "food_delivery", contract, "restaurant", {"name": "Cafe"},
    )


def test_equivalent_observation_field_marks_state_as_known() -> None:
    contract = _domain_registry("food_delivery").get(
        "food_delivery", "create_order",
    )
    assert record_state_is_known(
        "food_delivery", contract, "restaurant", {"items": []},
    )


def test_existence_only_target_does_not_invent_state_requirement() -> None:
    contract = _domain_registry("crm").get("crm", "update_contact")
    assert record_state_is_known("crm", contract, "contact", {})


def test_chain_novel_outputs_preserve_typed_state_created_identity() -> None:
    prior = ToolContract(
        domain="probe",
        name="inspect_optional_thread",
        readonly=True,
        mutating=False,
        arguments=frozenset({"thread_id"}),
        input_entities=(EntityBinding("thread", "thread_id"),),
    )
    creator = ToolContract(
        domain="probe",
        name="create_thread",
        readonly=False,
        mutating=True,
        arguments=frozenset({"message_id"}),
        output_entities=(EntityBinding("thread", "thread_id", "output"),),
        output_fields=frozenset({"thread_id"}),
        created_output_fields=frozenset({"thread_id"}),
    )

    assert chain_novel_output_fields([prior, creator], 1) == {"thread_id"}


def test_chain_novel_outputs_still_remove_readonly_prefix_echo() -> None:
    prior = ToolContract(
        domain="probe",
        name="select_thread",
        readonly=True,
        mutating=False,
        arguments=frozenset({"thread_id"}),
        input_entities=(EntityBinding("thread", "thread_id"),),
    )
    detail = ToolContract(
        domain="probe",
        name="get_thread",
        readonly=True,
        mutating=False,
        arguments=frozenset({"channel_id"}),
        output_entities=(EntityBinding("thread", "thread_id", "output"),),
        output_fields=frozenset({"thread_id"}),
    )

    assert chain_novel_output_fields([prior, detail], 1) == set()


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
        created_output_fields=frozenset({"order_id"}),
        postconditions=(_order_status("placed", "output"),),
    )
    return_order = ToolContract(
        domain="probe",
        name="return_order",
        readonly=False,
        mutating=True,
        arguments=frozenset({"order_id"}),
        required_arguments=frozenset({"order_id"}),
        input_entities=(EntityBinding("order", "order_id"),),
        preconditions=(_order_status("shipped"),),
    )

    _, issues = simulate_symbolic_chain(
        ContractRegistry([create, return_order]),
        "probe",
        ["create_order", "return_order"],
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
        created_output_fields=frozenset({"order_id"}),
        postconditions=(_order_status("shipped", "output"),),
    )
    return_order = ToolContract(
        domain="probe",
        name="return_order",
        readonly=False,
        mutating=True,
        arguments=frozenset({"order_id"}),
        required_arguments=frozenset({"order_id"}),
        input_entities=(EntityBinding("order", "order_id"),),
        preconditions=(_order_status("shipped"),),
    )

    _, issues = simulate_symbolic_chain(
        ContractRegistry([create, return_order]),
        "probe",
        ["create_order", "return_order"],
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


def test_all_191_tools_have_one_registered_state_contract() -> None:
    domain_tools = {
        domain: importlib.import_module(
            f"src.live_mcp.servers.{domain}.server"
        ).TOOLS
        for domain in DOMAINS
    }
    registry = build_contract_registry(domain_tools)

    assert sum(registry.coverage().values()) == 191
    for domain, tools in domain_tools.items():
        names = {tool["name"] for tool in tools}
        assert set(DOMAIN_STATE_FACTS[domain]) == names
        assert registry.audit_schema(domain, tools) == ()


def test_operational_dependency_contracts_consume_supplied_live_schemas() -> None:
    tools = importlib.import_module(
        "src.live_mcp.servers.food_delivery.server"
    ).TOOLS
    chain = [
        "create_order", "update_order_status", "track_rider",
    ]

    contracts = _operational_dependency_contracts(
        chain, "food_delivery", tools,
    )

    assert isinstance(contracts, list)


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


@pytest.mark.parametrize("archive_creator", ("tar_create", "zip"))
@pytest.mark.parametrize(
    "text_tool",
    (
        "cat", "head", "tail", "wc", "sort", "uniq", "cut", "sed",
        "awk", "truncate", "split", "diff", "join",
    ),
)
def test_filesystem_archive_creator_cannot_feed_text_tool(
    archive_creator: str,
    text_tool: str,
) -> None:
    tools = {
        domain: importlib.import_module(
            f"src.live_mcp.servers.{domain}.server"
        ).TOOLS
        for domain in DOMAINS
    }
    registry = build_contract_registry(tools)

    _, issues = simulate_symbolic_chain(
        registry, "filesystem", [archive_creator, text_tool],
    )

    assert issues
    assert issues[0].tool_name == text_tool
    assert issues[0].predicate.slot == "filesystem.archive"
    assert issues[0].reason == "contradicted"


def test_filesystem_plain_file_creator_can_feed_text_tool() -> None:
    tools = {
        domain: importlib.import_module(
            f"src.live_mcp.servers.{domain}.server"
        ).TOOLS
        for domain in DOMAINS
    }
    registry = build_contract_registry(tools)

    _, issues = simulate_symbolic_chain(
        registry, "filesystem", ["touch", "sort"],
    )

    assert issues == ()


@pytest.mark.parametrize(
    ("archive_creator", "extractor", "expect_issue"),
    (
        ("tar_create", "tar_extract", False),
        ("zip", "unzip", False),
        ("tar_create", "unzip", True),
        ("zip", "tar_extract", True),
    ),
)
def test_filesystem_archive_format_controls_extractor_dependency(
    archive_creator: str,
    extractor: str,
    expect_issue: bool,
) -> None:
    tools = {
        domain: importlib.import_module(
            f"src.live_mcp.servers.{domain}.server"
        ).TOOLS
        for domain in DOMAINS
    }
    registry = build_contract_registry(tools)

    _, issues = simulate_symbolic_chain(
        registry, "filesystem", [archive_creator, extractor],
    )

    assert bool(issues) is expect_issue
    if issues:
        assert issues[0].predicate.slot == "filesystem.archive_format"
        assert issues[0].reason == "contradicted"


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
