from __future__ import annotations

import importlib
import asyncio
import json
import os
import subprocess
import sys

import pytest

import src.reward.oval_reward_fn as reward_module

from src.agent_loop.livemcp_oval_loop import (
    LiveMCPOvalLoop,
    _validate_environment_metadata,
)
from src.agent_loop.oval_mcp_worker import OvalMCPWorkerContext
from src.live_mcp import errors
from src.live_mcp.executor import LiveMCPExecutor
from src.live_mcp.observation import (
    TRAJECTORY_SCHEMA_VERSION,
    OBSERVATION_SCHEMA_VERSION,
    OBSERVATION_PROJECTION_VERSION,
    compute_server_schema_hash,
    serialize_tool_result,
)
from src.live_mcp.environment_metadata import (
    build_environment_metadata,
    compute_reward_fingerprint,
    compute_transition_fingerprint,
)
from src.live_mcp.orchestrator import (
    _attribute_success_criteria,
    _detect_duplicate_side_effect,
    _detect_missing_dependency,
)
from src.live_mcp.config import load_suite_config
from src.live_mcp.state_seeder import StateSeeder
from src.live_mcp.types import OracleCall, ToolCall, ToolExecutionResult
from src.oval_mcp.envs.domain_adapter import get_adapter
from src.training.livemcp_hyperparams import LiveMCPHyperparams
from src.training.trainer_config import TrainerConfig
from src.training.run_grpo import _bind_profile_estimator
from src.training.hooks import (
    update_lambda_safe,
    validate_livemcp_non_tensor_batch,
)
from src.reward.oval_reward_fn import (
    _apply_round_contract_penalty,
    _validate_round_contracts,
)
from src.live_mcp.servers.banking.server import BankingServer
from src.live_mcp.servers.calendar.server import CalendarServer
from src.live_mcp.servers.crm.server import CRMServer
from src.live_mcp.servers.filesystem.server import FilesystemServer
from src.live_mcp.servers.food_delivery.server import FoodDeliveryServer
from src.live_mcp.servers.email.server import EmailServer
from src.live_mcp.servers.issue_tracker.server import IssueTrackerServer
from src.live_mcp.servers.payments.server import PaymentsServer
from src.live_mcp.servers.shopping.server import ShoppingServer
from src.live_mcp.servers.team_chat.server import TeamChatServer
from src.live_mcp.server_base import StatefulToolServer, _result as server_result
from src.live_mcp.tool_semantics import build_tool_semantics, resolve_tool_operation
from src.live_mcp.schema_registry import SchemaRegistry
from src.live_mcp.transport import TransportError
from src.oval_mcp.training.lambda_state import LambdaState
from src.oval_mcp.verifier.events import AuditEvent, EventLog


DOMAINS = (
    "calendar", "shopping", "banking", "email", "filesystem",
    "payments", "crm", "issue_tracker", "team_chat", "food_delivery",
)


def _result(**overrides) -> ToolExecutionResult:
    payload = {
        "success": False,
        "tool_name": "update_event",
        "canonical_tool_name": "update_event",
        "call_id": "call-1",
        "session_id": "session-1",
        "observation": {
            "missing_required": ["event_id"],
            "unexpected_keys": [],
            "type_errors": [],
            "enum_errors": [],
        },
        "error_type": "SCHEMA_INVALID",
        "error_message": "missing event_id",
        "schema_valid": False,
        "state_changed": False,
        "latency_ms": 1,
        "execution_status": "FAILURE",
    }
    payload.update(overrides)
    return ToolExecutionResult(**payload)


def test_policy_result_serializer_preserves_structured_failure() -> None:
    rendered = serialize_tool_result(_result(), max_chars=4096)
    parsed = json.loads(rendered)
    assert parsed["success"] is False
    assert parsed["execution_status"] == "FAILURE"
    assert parsed["error_type"] == "SCHEMA_INVALID"
    assert parsed["observation"]["missing_required"] == ["event_id"]
    assert len(rendered) <= 4096


def test_strict_prove_profile_forces_pure_task_reward() -> None:
    cfg = LiveMCPHyperparams(
        reward_profile="prove_baseline",
        i_shape=1,
        i_process=1,
        lambda_safe_default=2.0,
    )
    assert cfg.i_shape == 0
    assert cfg.i_process == 0
    assert cfg.lambda_safe_default == 0.0
    assert cfg.objective_formula == "J = R_task"


def test_reward_worker_import_honors_strict_prove_profile() -> None:
    env = dict(os.environ)
    env.update({
        "OVAL_REWARD_PROFILE": "prove_baseline",
        "OVAL_I_SHAPE": "1",
        "OVAL_I_PROCESS": "1",
        "OVAL_LAMBDA_SAFE_DEFAULT": "2.0",
    })
    code = """
from src.reward import oval_reward_fn as reward
assert reward._REWARD_PROFILE == 'prove_baseline'
assert reward._I_SHAPE == 0
assert reward._I_PROCESS == 0
assert reward._LAMBDA_SAFE_DEFAULT == 0.0
result = reward.compute_score('live', '', {}, {
    'domain': 'calendar',
    'oracle_calls': '[{\"action\":\"tool_call\",\"tool_name\":\"list_events\",\"arguments\":{}}]',
    'success_criteria': '[]',
    'allowed_terminal_actions': ['final_answer'],
    'round_contracts': [{'round_idx': 0, 'required_tools': ['list_events'], 'allowed_terminal_actions': ['final_answer']}],
    'audit_events': [{
        'event_id': 'terminal', 'session_id': 's', 'step': 0,
        'action_type': 'final_answer', 'terminal_action': 'done',
    }],
})
assert result['reward_profile'] == 'prove_baseline'
assert result['lambda_safe'] == 0.0
"""
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=os.getcwd(),
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )


def test_invalid_reward_profile_fails_fast() -> None:
    with pytest.raises(ValueError, match="reward_profile"):
        LiveMCPHyperparams(reward_profile="implicit_mix")


def test_three_deterministic_scenario_misclassifications_are_fixed() -> None:
    crm = [OracleCall(
        "add_note",
        {"entity_type": "lead", "entity_id": "lead_001", "content": "note"},
    )]
    watcher = [OracleCall(
        "add_watcher", {"issue_id": "iss_001", "user": "user_001"},
    )]
    archive_then_label = [
        OracleCall("archive_email", {"email_id": "eml_001"}),
        OracleCall("add_label", {"email_id": "eml_001", "label": "done"}),
    ]
    assert not _detect_missing_dependency(crm, "crm")
    assert not _detect_missing_dependency(watcher, "issue_tracker")
    assert not _detect_duplicate_side_effect(archive_then_label)


def test_all_formal_tools_use_schema_bound_operations_without_fallback() -> None:
    for domain in DOMAINS:
        module = importlib.import_module(
            f"src.live_mcp.servers.{domain}.server"
        )
        adapter = get_adapter(domain)
        adapter.register_tool_schemas(module.TOOLS)
        for tool in module.TOOLS:
            operation, _target = adapter.tool_semantics(
                tool["name"], "domain_resource", state_changed=False,
            )
            annotations = tool["annotations"]
            if annotations.get("readonly"):
                assert operation == "query"
            else:
                assert operation in {"create", "update", "delete"}
        with pytest.raises(ValueError, match="unregistered tool contract"):
            adapter.tool_semantics("obsolete_tool", "domain_resource")


def _contract_maps(
    tools_by_domain: dict[str, list[dict]],
    *,
    profiles: list[str] | None = None,
) -> dict:
    return {
        "server_schema_hashes": {
            domain: compute_server_schema_hash(tools)
            for domain, tools in tools_by_domain.items()
        },
        "transition_fingerprints": {
            domain: compute_transition_fingerprint(domain, tools)
            for domain, tools in tools_by_domain.items()
        },
        "initial_state_hashes": {
            domain: f"initial-{domain}" for domain in tools_by_domain
        },
        "reward_fingerprint": compute_reward_fingerprint(),
        "max_observation_chars": 4096,
        "reward_profile_compatibility": profiles or ["prove_baseline", "oval_full"],
    }


def test_environment_metadata_accepts_exact_schema_and_versions() -> None:
    tools = [{
        "name": "list_items",
        "description": "List items.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
        "annotations": {"readonly": True, "mutating": False},
    }]
    extra = {
        "domain": "calendar",
        "server_schema_hash": compute_server_schema_hash(tools),
        "initial_state_hash": "initial-calendar",
        **_contract_maps({"calendar": tools}),
        "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
        "observation_projection_version": OBSERVATION_PROJECTION_VERSION,
        "trajectory_schema_version": TRAJECTORY_SCHEMA_VERSION,
    }
    _validate_environment_metadata(
        extra,
        tools,
        "prove_baseline",
        current_tools_by_domain={"calendar": tools},
        required_owner_domains={"calendar"},
    )


def test_environment_metadata_rejects_unbound_distractor_owner() -> None:
    tools = [{
        "name": "list_items",
        "description": "List items.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
        "annotations": {"readonly": True, "mutating": False},
    }]
    extra = {
        "server_schema_hash": compute_server_schema_hash(tools),
        **_contract_maps({"calendar": tools}, profiles=["prove_baseline"]),
        "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
        "observation_projection_version": OBSERVATION_PROJECTION_VERSION,
        "trajectory_schema_version": TRAJECTORY_SCHEMA_VERSION,
    }
    with pytest.raises(RuntimeError, match="missing executable owner"):
        _validate_environment_metadata(
            extra,
            tools,
            "prove_baseline",
            current_tools_by_domain={"calendar": tools},
            required_owner_domains={"calendar", "shopping"},
        )


def test_environment_metadata_rejects_distractor_schema_drift() -> None:
    primary = [{
        "name": "list_items", "description": "List items.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
        "annotations": {"readonly": True, "mutating": False},
    }]
    distractor = [{
        "name": "search_orders", "description": "Search orders.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
        "annotations": {"readonly": True, "mutating": False},
    }]
    extra = {
        "domain": "calendar",
        "server_schema_hash": compute_server_schema_hash(primary),
        "initial_state_hash": "initial-calendar",
        **_contract_maps({
            "calendar": primary,
            "shopping": distractor,
        }, profiles=["prove_baseline"]),
        "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
        "observation_projection_version": OBSERVATION_PROJECTION_VERSION,
        "trajectory_schema_version": TRAJECTORY_SCHEMA_VERSION,
    }
    drifted = [{**distractor[0], "description": "Changed schema."}]
    with pytest.raises(RuntimeError, match=r"server_schema_hashes\['shopping'\] mismatch"):
        _validate_environment_metadata(
            extra,
            primary,
            "prove_baseline",
            current_tools_by_domain={
                "calendar": primary,
                "shopping": drifted,
            },
            required_owner_domains={"calendar", "shopping"},
        )


def test_environment_metadata_rejects_legacy_rows() -> None:
    with pytest.raises(RuntimeError, match="observation_schema_version"):
        _validate_environment_metadata({}, [], "prove_baseline")


def test_environment_metadata_rejects_observation_budget_drift() -> None:
    tools = [{
        "name": "list_items",
        "description": "List items.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
        "annotations": {"readonly": True, "mutating": False},
    }]
    extra = {
        "server_schema_hash": compute_server_schema_hash(tools),
        **_contract_maps({"calendar": tools}, profiles=["prove_baseline"]),
        "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
        "observation_projection_version": OBSERVATION_PROJECTION_VERSION,
        "trajectory_schema_version": TRAJECTORY_SCHEMA_VERSION,
        "max_observation_chars": 4000,
    }
    with pytest.raises(RuntimeError, match="max_observation_chars mismatch"):
        _validate_environment_metadata(
            extra,
            tools,
            "prove_baseline",
            current_tools_by_domain={"calendar": tools},
            required_owner_domains={"calendar"},
            runtime_max_observation_chars=4096,
        )


def test_all_task_subtypes_share_environment_metadata_builder() -> None:
    tools = [{
        "name": "list_items",
        "description": "List items.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
        "annotations": {"readonly": True, "mutating": False},
    }]
    metadata = build_environment_metadata(
        load_suite_config("configs/live_mcp/ten_domain_suite.yaml"),
        tools,
        primary_server_name="calendar",
        owner_server_tools={"calendar": tools},
    )
    assert metadata["server_schema_hash"] == compute_server_schema_hash(tools)
    assert metadata["server_schema_hashes"] == {
        "calendar": compute_server_schema_hash(tools),
    }
    assert metadata["observation_schema_version"] == OBSERVATION_SCHEMA_VERSION
    assert metadata["observation_projection_version"] == OBSERVATION_PROJECTION_VERSION
    assert metadata["trajectory_schema_version"] == TRAJECTORY_SCHEMA_VERSION
    assert metadata["max_observation_chars"] == 4096
    assert metadata["reward_profile_compatibility"] == [
        "prove_baseline", "oval_full",
    ]


def test_round_contract_gate_rejects_incomplete_trajectory_in_all_profiles() -> None:
    components = (0.8, 0.7, 0.6, 0.5)
    assert _apply_round_contract_penalty(
        components,
        round_contract_ok=False,
        reward_profile="prove_baseline",
    ) == (0.0, 0.0, 0.0, 0.0)
    assert _apply_round_contract_penalty(
        components,
        round_contract_ok=False,
        reward_profile="oval_full",
    ) == (0.0, 0.0, 0.0, 0.0)


def test_prove_reward_has_no_formula_external_terminal_or_identity_penalty() -> None:
    events = EventLog(events=[
        AuditEvent(
            event_id="tool", session_id="s", step=0,
            action_type="tool_call", tool_name="update_event",
            tool_arguments={}, tool_name_known=True, schema_valid=True,
            execution_success=True, identity_violation="wrong_resource",
        ),
        AuditEvent(
            event_id="terminal", session_id="s", step=1,
            action_type="report_error", terminal_action="failed",
        ),
    ])
    task = {
        "required_tool_calls": [
            {"tool_name": "update_event", "arguments": {}},
        ],
        "allowed_terminal_actions": ["final_answer"],
        "identity_policy": "preserve",
    }

    baseline = reward_module._task_reward.compute(events, task)
    assert baseline.r_validity == 1.0
    assert baseline.r_coverage == 1.0

    oval = reward_module._task_reward.compute(events, {
        **task,
        "apply_terminal_validity_penalty": True,
        "apply_identity_coverage_penalty": True,
    })
    assert oval.r_validity == 0.5
    assert oval.r_coverage == 0.0


def test_argument_equality_keeps_boolean_and_number_types_distinct() -> None:
    equal = reward_module._task_reward._args_equal
    assert equal(True, "true") is True
    assert equal(False, "false") is True
    assert equal(True, 1) is False
    assert equal(False, 0) is False


def test_reward_profile_binds_the_advantage_estimator(monkeypatch) -> None:
    monkeypatch.delenv("OVAL_ADV_ESTIMATOR", raising=False)
    monkeypatch.setenv("OVAL_REWARD_PROFILE", "prove_baseline")
    assert TrainerConfig.from_env().adv_estimator == "grpo"

    monkeypatch.setenv("OVAL_REWARD_PROFILE", "oval_full")
    assert TrainerConfig.from_env().adv_estimator == "livemcp_grpo"

    monkeypatch.setenv("OVAL_ADV_ESTIMATOR", "grpo")
    with pytest.raises(ValueError, match="requires adv_estimator"):
        TrainerConfig.from_env()


def test_direct_training_entry_binds_and_rejects_estimator_drift() -> None:
    argv = ["run_grpo.py"]
    assert _bind_profile_estimator("prove_baseline", argv) == "grpo"
    assert argv[-1] == "algorithm.adv_estimator=grpo"

    with pytest.raises(ValueError, match="requires adv_estimator"):
        _bind_profile_estimator(
            "prove_baseline",
            ["run_grpo.py", "+algorithm.adv_estimator=livemcp_grpo"],
        )


def test_seeded_mutable_entity_fields_do_not_alias() -> None:
    seeder = StateSeeder()
    email_state = seeder.seed_state("email", "session", 42)
    labels = [email["labels"] for email in email_state["emails"].values()]
    assert len({id(value) for value in labels}) == len(labels)

    chat_state = seeder.seed_state("team_chat", "session", 42)
    reactions = [
        message["reactions"]
        for channel in chat_state["channels"].values()
        for message in channel["messages"]
    ]
    assert len({id(value) for value in reactions}) == len(reactions)


def test_all_seeded_mutable_objects_are_isolated() -> None:
    """No seeded list/dict/set may be shared across state paths."""
    def collect(value, path, seen):
        if isinstance(value, (dict, list, set)):
            seen.setdefault(id(value), []).append(path)
            children = value.items() if isinstance(value, dict) else enumerate(value)
            for key, child in children:
                collect(child, (*path, str(key)), seen)

    seeder = StateSeeder()
    for domain in DOMAINS:
        seen = {}
        collect(seeder.seed_state(domain, "alias-audit", 42), (), seen)
        aliases = [paths for paths in seen.values() if len(paths) > 1]
        assert aliases == [], (domain, aliases)


def test_mutation_counter_roots_exist_in_seeded_state() -> None:
    seeder = StateSeeder()
    for domain in DOMAINS:
        module = importlib.import_module(
            f"src.live_mcp.servers.{domain}.server"
        )
        state = seeder.seed_state(domain, "counter-root-audit", 42)
        for semantics in build_tool_semantics(domain, module.TOOLS).values():
            for root in semantics.allowed_state_roots:
                if root.startswith("next_"):
                    assert root in state, (domain, semantics.name, root)


def test_server_boundary_rolls_back_partial_mutation_and_audits_delta() -> None:
    class ProbeServer(StatefulToolServer):
        def __init__(self):
            tools = [
                {"name": "fail_after_write"},
                {"name": "wrong_change_flag"},
                {"name": "valid_write"},
            ]
            super().__init__(
                "banking", tools, enforce_tool_semantics=False,
            )
            self.handlers = {
                "fail_after_write": self.fail_after_write,
                "wrong_change_flag": self.wrong_change_flag,
                "valid_write": self.valid_write,
            }

        def fail_after_write(self, session_id, _arguments):
            self._state(session_id)["transactions"].append({"txn_id": "bad"})
            raise KeyError("forced failure")

        def wrong_change_flag(self, session_id, _arguments):
            self._state(session_id)["transactions"].append({"txn_id": "bad"})
            return server_result(True, {}, None, "", False)

        def valid_write(self, session_id, _arguments):
            self._state(session_id)["transactions"].append({"txn_id": "ok"})
            return server_result(True, {}, None, "", True)

    server = ProbeServer(); sid = _reset(server)
    for name in ("fail_after_write", "wrong_change_flag"):
        result = server._call_tool({"session_id": sid, "name": name, "arguments": {}})
        assert result["success"] is False
        assert server.sessions[sid]["transactions"] == []
    valid = server._call_tool({"session_id": sid, "name": "valid_write", "arguments": {}})
    assert valid["success"] is True
    assert valid["state_delta_paths"] == ["transactions"]


def test_success_criteria_are_attributed_to_factual_call_deltas() -> None:
    criteria = [{
        "type": "state_equals",
        "path_parts": ["accounts", "acc_1", "balance"],
        "value": 90,
    }]
    history = [{
        "success": True,
        "tool_name": "withdraw",
        "arguments": {"account_id": "acc_1", "amount": 10},
        "state_delta_paths": ["accounts.acc_1.balance", "transactions"],
    }]
    attributed = _attribute_success_criteria(criteria, history)
    assert attributed[0]["source_calls"][0]["tool_name"] == "withdraw"
    assert attributed[0]["source_calls"][0]["state_delta_paths"] == [
        "accounts.acc_1.balance",
    ]


def test_reversible_targets_are_live_grounded_and_round_trip() -> None:
    banking = BankingServer(); bank_sid = _reset(banking, seed=42)
    scheduled = banking._call_tool({
        "session_id": bank_sid, "name": "list_scheduled_transfers", "arguments": {},
    })
    scheduled_id = scheduled["observation"]["scheduled_transfers"][0]["scheduled_txn_id"]
    cancelled = banking._call_tool({
        "session_id": bank_sid, "name": "cancel_transfer",
        "arguments": {"scheduled_txn_id": scheduled_id},
    })
    assert cancelled["success"] and cancelled["state_delta_paths"] == [
        f"scheduled_transfers.{scheduled_id}.status",
    ]
    account_id = next(
        aid for aid, account in banking.sessions[bank_sid]["accounts"].items()
        if not account["frozen"]
    )
    assert banking._call_tool({
        "session_id": bank_sid, "name": "freeze_account",
        "arguments": {"account_id": account_id, "reason": "user request"},
    })["success"]
    assert banking._call_tool({
        "session_id": bank_sid, "name": "unfreeze_account",
        "arguments": {"account_id": account_id, "authorization_code": "user-code"},
    })["success"]

    payments = PaymentsServer(); pay_sid = _reset(payments, seed=42)
    webhook_id = next(iter(payments.sessions[pay_sid]["webhooks"]))
    assert payments._call_tool({
        "session_id": pay_sid, "name": "delete_webhook",
        "arguments": {"webhook_id": webhook_id},
    })["success"]
    assert webhook_id not in payments.sessions[pay_sid]["webhooks"]
    repeated_delete = payments._call_tool({
        "session_id": pay_sid, "name": "delete_webhook",
        "arguments": {"webhook_id": webhook_id},
    })
    assert repeated_delete["success"] is False
    assert repeated_delete["state_changed"] is False
    created = payments._call_tool({
        "session_id": pay_sid, "name": "create_webhook",
        "arguments": {"url": "https://example.com/new", "events": ["invoice.paid"]},
    })
    created_id = created["observation"]["webhook"]["webhook_id"]
    assert payments._call_tool({
        "session_id": pay_sid, "name": "delete_webhook",
        "arguments": {"webhook_id": created_id},
    })["success"]
    invoice = next(
        item for item in payments.sessions[pay_sid]["invoices"].values()
        if item["status"] in {"pending", "overdue"} and not item.get("payment_id")
    )
    assert payments._call_tool({
        "session_id": pay_sid, "name": "pay_invoice",
        "arguments": {"invoice_id": invoice["invoice_id"], "amount": invoice["amount"]},
    })["success"]
    assert payments._call_tool({
        "session_id": pay_sid, "name": "refund_invoice",
        "arguments": {"invoice_id": invoice["invoice_id"], "amount": invoice["amount"]},
    })["success"]

    shopping = ShoppingServer(); shop_sid = _reset(shopping, seed=42)
    existing = shopping.sessions[shop_sid]["wishlist"][0]
    assert shopping._call_tool({
        "session_id": shop_sid, "name": "remove_from_wishlist",
        "arguments": {"product_id": existing},
    })["success"]
    assert shopping._call_tool({
        "session_id": shop_sid, "name": "add_to_wishlist",
        "arguments": {"product_id": existing},
    })["success"]
    assert shopping._call_tool({
        "session_id": shop_sid, "name": "remove_from_wishlist",
        "arguments": {"product_id": existing},
    })["success"]


def test_mutating_tools_have_real_footprints() -> None:
    """Exercise representative mutating tools across all ten domains."""
    banking = BankingServer(); sid = _reset(banking)
    account_id = next(iter(banking.sessions[sid]["accounts"]))
    assert banking._call_tool({
        "session_id": sid, "name": "apply_loan",
        "arguments": {
            "account_id": account_id, "amount": 100,
            "term_months": 12, "purpose": "equipment",
        },
    })["state_delta_paths"]

    calendar = CalendarServer(); sid = _reset(calendar)
    assert calendar._call_tool({
        "session_id": sid, "name": "change_timezone",
        "arguments": {"timezone": "UTC"},
    })["state_delta_paths"]
    event_id = next(iter(calendar.sessions[sid]["events"]))
    assert calendar._call_tool({
        "session_id": sid, "name": "delete_event",
        "arguments": {"event_id": event_id},
    })["state_delta_paths"]

    crm = CRMServer(); sid = _reset(crm)
    lead_id = next(
        key for key, value in crm.sessions[sid]["leads"].items()
        if value["status"] != "converted"
        and not any(deal.get("lead_id") == key for deal in crm.sessions[sid]["deals"].values())
    )
    assert crm._call_tool({
        "session_id": sid, "name": "delete_lead",
        "arguments": {"lead_id": lead_id},
    })["state_delta_paths"]
    contact = crm._call_tool({
        "session_id": sid, "name": "create_contact",
        "arguments": {"name": "Audit User", "email": "audit@example.com"},
    })["observation"]["contact"]
    assert crm._call_tool({
        "session_id": sid, "name": "delete_contact",
        "arguments": {"contact_id": contact["contact_id"]},
    })["state_delta_paths"]

    email = EmailServer(); sid = _reset(email)
    email_id = next(iter(email.sessions[sid]["emails"]))
    cases = [
        ("create_draft", {"to": "a@example.com", "subject": "S", "body": "B"}),
        ("create_filter", {"field": "sender", "pattern": "a@example.com", "action": "label", "label": "audit"}),
        ("forward_email", {"email_id": email_id, "to": "b@example.com"}),
    ]
    for name, arguments in cases:
        assert email._call_tool({
            "session_id": sid, "name": name, "arguments": arguments,
        })["state_delta_paths"]

    filesystem = FilesystemServer(); sid = _reset(filesystem)
    fs_cases = [
        ("cd", {"path": "/home/user/data"}),
        ("umask", {"mask": "027"}),
        ("chown", {"path": "/home/user/notes.txt", "owner": "audit"}),
        ("cp", {"source": "/home/user/notes.txt", "target": "/home/user/notes-copy.txt", "recursive": False}),
        ("mkdir", {"path": "/home/user/audit-dir", "parents": False}),
    ]
    for name, arguments in fs_cases:
        assert filesystem._call_tool({
            "session_id": sid, "name": name, "arguments": arguments,
        })["state_delta_paths"]
    assert filesystem._call_tool({
        "session_id": sid, "name": "zip",
        "arguments": {"archive": "/home/user/notes.zip", "paths": ["/home/user/notes.txt"]},
    })["success"]
    assert filesystem._call_tool({
        "session_id": sid, "name": "rm",
        "arguments": {"path": "/home/user/notes.txt", "recursive": False},
    })["success"]
    assert filesystem._call_tool({
        "session_id": sid, "name": "unzip",
        "arguments": {"archive": "/home/user/notes.zip", "target_dir": "/home/user"},
    })["state_delta_paths"]

    issue = IssueTrackerServer(); sid = _reset(issue)
    issue_id = next(iter(issue.sessions[sid]["issues"]))
    assert issue._call_tool({
        "session_id": sid, "name": "create_subtask",
        "arguments": {"issue_id": issue_id, "title": "Audit subtask"},
    })["state_delta_paths"]
    assert issue._call_tool({
        "session_id": sid, "name": "create_sprint",
        "arguments": {
            "name": "Audit Sprint", "start_date": "2026-07-01",
            "end_date": "2026-07-14",
        },
    })["state_delta_paths"]

    payments = PaymentsServer(); sid = _reset(payments)
    pending_id = next(
        key for key, value in payments.sessions[sid]["payments"].items()
        if value["status"] == "pending"
    )
    assert payments._call_tool({
        "session_id": sid, "name": "cancel_payment",
        "arguments": {"payment_id": pending_id, "reason": "user request"},
    })["state_delta_paths"]

    shopping = ShoppingServer(); sid = _reset(shopping)
    product_id = next(iter(shopping.sessions[sid]["products"]))
    assert shopping._call_tool({
        "session_id": sid, "name": "add_review",
        "arguments": {
            "product_id": product_id, "rating": 5,
            "body": "Verified in mutation-footprint audit",
        },
    })["state_delta_paths"]
    assert shopping._call_tool({
        "session_id": sid, "name": "add_to_cart",
        "arguments": {"product_id": product_id, "quantity": 1},
    })["success"]
    assert shopping._call_tool({
        "session_id": sid, "name": "remove_from_cart",
        "arguments": {"product_id": product_id},
    })["state_delta_paths"]

    chat = TeamChatServer(); sid = _reset(chat)
    assert chat._call_tool({
        "session_id": sid, "name": "create_channel",
        "arguments": {"name": "mutation-footprint-audit"},
    })["state_delta_paths"]
    channel_id = next(
        key for key, value in chat.sessions[sid]["channels"].items()
        if not value["archived"]
    )
    message = chat._call_tool({
        "session_id": sid, "name": "send_message",
        "arguments": {"channel_id": channel_id, "content": "audit message"},
    })
    assert message["state_delta_paths"]
    message_id = message["observation"]["message"]["message_id"]
    assert chat._call_tool({
        "session_id": sid, "name": "create_thread",
        "arguments": {"channel_id": channel_id, "message_id": message_id},
    })["state_delta_paths"]
    assert chat._call_tool({
        "session_id": sid, "name": "send_dm",
        "arguments": {"recipient": "alice", "content": "audit dm"},
    })["state_delta_paths"]
    assert chat._call_tool({
        "session_id": sid, "name": "archive_channel",
        "arguments": {"channel_id": channel_id},
    })["state_delta_paths"]


def _reset(server, session_id: str = "test", seed: int = 123) -> str:
    server.handle_request("session/reset", {"session_id": session_id, "seed": seed})
    return session_id


def test_unknown_or_closed_session_does_not_revive() -> None:
    server = BankingServer()
    result = server.handle_request("tools/call", {
        "session_id": "closed",
        "name": "list_accounts",
        "arguments": {},
    })["result"]
    assert result["success"] is False
    assert "unknown or closed session" in result["error_message"]
    assert server.sessions == {}


def test_banking_rejects_self_transfer_and_accepts_user_authorization() -> None:
    server = BankingServer(); sid = _reset(server)
    state = server.sessions[sid]
    account_id = next(iter(state["accounts"]))
    self_transfer = server._call_tool({
        "session_id": sid,
        "name": "transfer",
        "arguments": {
            "from_account": account_id,
            "to_account": account_id,
            "amount": 1,
        },
    })
    assert self_transfer["success"] is False
    past_schedule = server._call_tool({
        "session_id": sid,
        "name": "schedule_transfer",
        "arguments": {
            "from_account": list(state["accounts"])[0],
            "to_account": list(state["accounts"])[1],
            "amount": 1,
            "execute_date": state["current_date"],
        },
    })
    assert past_schedule["success"] is False
    frozen_id = next(
        aid for aid, account in state["accounts"].items() if account["frozen"]
    )
    unfreeze = server._call_tool({
        "session_id": sid,
        "name": "unfreeze_account",
        "arguments": {
            "account_id": frozen_id,
            "authorization_code": "user-provided-code",
        },
    })
    assert unfreeze["success"] is True
    assert state["accounts"][frozen_id]["frozen"] is False


def test_banking_uses_seed_clock_and_rejects_currency_mismatch() -> None:
    server = BankingServer(); sid = _reset(server, seed=4097)
    state = server.sessions[sid]
    accounts = list(state["accounts"])
    result = server._call_tool({
        "session_id": sid,
        "name": "transfer",
        "arguments": {
            "from_account": accounts[0], "to_account": accounts[1],
            "amount": 1, "currency": "EUR",
        },
    })
    assert result["success"] is False

    account_id = next(aid for aid, account in state["accounts"].items() if not account["frozen"])
    frozen = server._call_tool({
        "session_id": sid, "name": "freeze_account",
        "arguments": {"account_id": account_id, "reason": "audit"},
    })
    assert frozen["success"] is True
    assert state["freeze_log"][-1]["timestamp"] == state["current_date"]


def test_calendar_rejects_empty_or_unsupported_update() -> None:
    server = CalendarServer(); sid = _reset(server)
    event_id = next(iter(server.sessions[sid]["events"]))
    for fields in ({}, {"unsupported": "value"}):
        result = server._call_tool({
            "session_id": sid,
            "name": "update_event",
            "arguments": {"event_id": event_id, "fields": fields},
        })
        assert result["success"] is False

    recurring = server._call_tool({
        "session_id": sid,
        "name": "create_recurring",
        "arguments": {
            "title": "Invalid recurrence",
            "start_time": "2026-07-15T10:00:00",
            "end_time": "2026-07-15T11:00:00",
            "recurrence": "daily",
            "count": 0,
        },
    })
    assert recurring["success"] is False

    event_count = len(server.sessions[sid]["events"])
    weekday_mismatch = server._call_tool({
        "session_id": sid,
        "name": "create_recurring",
        "arguments": {
            "title": "Mismatched recurrence",
            "start_time": "2026-07-15T10:00:00",
            "end_time": "2026-07-15T11:00:00",
            "recurrence": "FREQ=WEEKLY;BYDAY=MO",
            "count": 3,
        },
    })
    assert weekday_mismatch["success"] is False
    assert weekday_mismatch["state_changed"] is False
    assert len(server.sessions[sid]["events"]) == event_count


def test_calendar_attendee_response_and_timezone_are_consistent() -> None:
    server = CalendarServer(); sid = _reset(server)
    state = server.sessions[sid]
    event_id = next(iter(state["events"]))
    email = "audit@example.com"
    added = server._call_tool({
        "session_id": sid, "name": "add_attendee",
        "arguments": {
            "event_id": event_id, "email": email,
            "response_status": "accepted",
        },
    })
    assert added["success"] is True
    assert state["events"][event_id]["responses"][email] == "accepted"
    removed = server._call_tool({
        "session_id": sid, "name": "remove_attendee",
        "arguments": {"event_id": event_id, "email": email},
    })
    assert removed["success"] is True
    assert email not in state["events"][event_id].get("responses", {})

    changed = server._call_tool({
        "session_id": sid, "name": "change_timezone",
        "arguments": {"timezone": "Asia/Shanghai"},
    })
    assert changed["success"] is True
    hours = server._call_tool({
        "session_id": sid, "name": "get_working_hours", "arguments": {},
    })
    assert hours["observation"]["timezone"] == "Asia/Shanghai"
    invalid = server._call_tool({
        "session_id": sid, "name": "change_timezone",
        "arguments": {"timezone": "Mars/Olympus"},
    })
    assert invalid["success"] is False


def test_payments_pending_payment_is_not_refundable() -> None:
    server = PaymentsServer(); sid = _reset(server)
    state = server.sessions[sid]
    pending = next(p for p in state["payments"].values() if p["status"] == "pending")
    invoice = state["invoices"][pending["invoice_id"]]
    assert invoice["status"] == "pending"
    result = server._call_tool({
        "session_id": sid,
        "name": "refund_invoice",
        "arguments": {"invoice_id": invoice["invoice_id"], "amount": invoice["amount"]},
    })
    assert result["success"] is False


def test_crm_rejects_unlinked_resources_and_noop_update() -> None:
    server = CRMServer(); sid = _reset(server)
    state = server.sessions[sid]
    cases = [
        ("create_deal", {"name": "Unlinked", "amount": 100}),
        ("create_task", {"title": "Unlinked"}),
        ("update_deal", {"deal_id": next(iter(state["deals"]))}),
    ]
    for name, arguments in cases:
        result = server._call_tool({
            "session_id": sid, "name": name, "arguments": arguments,
        })
        assert result["success"] is False


def test_crm_deletion_rejects_task_and_note_references() -> None:
    server = CRMServer(); sid = _reset(server)
    state = server.sessions[sid]
    contact_id = next(iter(state["contacts"]))
    state.setdefault("tasks", {})["task_audit"] = {
        "task_id": "task_audit", "contact_id": contact_id,
    }
    result = server._call_tool({
        "session_id": sid, "name": "delete_contact",
        "arguments": {"contact_id": contact_id},
    })
    assert result["success"] is False

    lead_id = next(
        lid for lid, lead in state["leads"].items()
        if lead["status"] != "converted"
        and not any(deal.get("lead_id") == lid for deal in state["deals"].values())
    )
    state.setdefault("notes", {})["note_audit"] = {
        "note_id": "note_audit", "entity_type": "lead", "entity_id": lead_id,
    }
    result = server._call_tool({
        "session_id": sid, "name": "delete_lead",
        "arguments": {"lead_id": lead_id},
    })
    assert result["success"] is False


def test_filesystem_archive_round_trip_restores_removed_file() -> None:
    server = FilesystemServer(); sid = _reset(server)
    state = server.sessions[sid]
    source = "/home/user/notes.txt"
    original = dict(state["fs"][source])
    create = server._call_tool({
        "session_id": sid,
        "name": "tar_create",
        "arguments": {"archive": "/home/user/notes.tar", "paths": [source]},
    })
    assert create["success"] is True
    assert create["observation"]["files_count"] == 1
    assert server._call_tool({
        "session_id": sid,
        "name": "rm",
        "arguments": {"path": source, "recursive": False},
    })["success"] is True
    extract = server._call_tool({
        "session_id": sid,
        "name": "tar_extract",
        "arguments": {"archive": "/home/user/notes.tar", "target_dir": "/home/user"},
    })
    assert extract["success"] is True
    assert extract["state_changed"] is True
    assert state["fs"][source] == original


def test_filesystem_path_boundaries_cwd_diff_and_join_are_consistent() -> None:
    server = FilesystemServer(); sid = _reset(server)
    state = server.sessions[sid]
    state["fs"]["/home/user/data2"] = {
        "type": "dir", "content": "", "permissions": "700", "owner": "user",
    }
    state["fs"]["/home/user/data2/outside.txt"] = {
        "type": "file", "content": "needle", "permissions": "600", "owner": "user",
    }
    found = server._call_tool({
        "session_id": sid, "name": "find",
        "arguments": {"path": "/home/user/data", "pattern": "*.txt"},
    })
    assert all(not path.startswith("/home/user/data2") for path in found["observation"]["matches"])

    assert server._call_tool({
        "session_id": sid, "name": "cd", "arguments": {"path": "/home/user/data"},
    })["success"]
    assert server._call_tool({
        "session_id": sid, "name": "mv",
        "arguments": {"source": "/home/user/data", "target": "/home/user/data-moved"},
    })["success"]
    assert state["cwd"] == "/home/user/data-moved"
    assert server._call_tool({
        "session_id": sid, "name": "rm",
        "arguments": {"path": "/home/user/data-moved", "recursive": True},
    })["success"]
    assert state["cwd"] == "/home/user"

    state["fs"]["/home/user/left.txt"] = {
        "type": "file", "content": "a\nleft-tail", "permissions": "600", "owner": "user",
    }
    state["fs"]["/home/user/right.txt"] = {
        "type": "file", "content": "a", "permissions": "600", "owner": "user",
    }
    diff = server._call_tool({
        "session_id": sid, "name": "diff",
        "arguments": {"file1": "/home/user/left.txt", "file2": "/home/user/right.txt"},
    })
    assert diff["observation"]["differences"][-1]["left"] == "left-tail"

    state["fs"]["/home/user/j1.txt"] = {
        "type": "file", "content": "x 1\ny 2", "permissions": "600", "owner": "user",
    }
    state["fs"]["/home/user/j2.txt"] = {
        "type": "file", "content": "p 2\nq 1", "permissions": "600", "owner": "user",
    }
    joined = server._call_tool({
        "session_id": sid, "name": "join",
        "arguments": {
            "file1": "/home/user/j1.txt", "file2": "/home/user/j2.txt",
            "field": 2,
        },
    })
    assert joined["observation"]["joined"] == ["x 1 q", "y 2 p"]


def test_freeship_changes_cart_and_checkout_totals() -> None:
    server = ShoppingServer(); sid = _reset(server)
    state = server.sessions[sid]
    product_id = next(iter(state["products"]))
    assert server._call_tool({
        "session_id": sid, "name": "add_to_cart",
        "arguments": {"product_id": product_id, "quantity": 1},
    })["success"]
    before = server._call_tool({
        "session_id": sid, "name": "get_cart", "arguments": {},
    })["observation"]
    assert before["shipping"] > 0
    assert server._call_tool({
        "session_id": sid, "name": "apply_coupon",
        "arguments": {"code": "FREESHIP"},
    })["success"]
    after = server._call_tool({
        "session_id": sid, "name": "get_cart", "arguments": {},
    })["observation"]
    assert after["shipping"] == 0
    assert after["final_total"] < before["final_total"]
    order = server._call_tool({
        "session_id": sid, "name": "checkout",
        "arguments": {"shipping_address": "1 Audit Way", "payment_method": "card"},
    })["observation"]["order"]
    assert order["shipping"] == 0


def test_track_order_fallback_uses_session_date_without_mutation() -> None:
    server = ShoppingServer(); sid = _reset(server)
    state = server.sessions[sid]
    order = next(iter(state["orders"].values()))
    order.pop("tracking", None)

    result = server._call_tool({
        "session_id": sid,
        "name": "track_order",
        "arguments": {"order_id": order["order_id"]},
    })

    assert result["success"] is True
    assert result["state_changed"] is False
    assert result["state_delta_paths"] == []
    assert result["observation"]["tracking"] == [{
        "status": order["status"],
        "timestamp": state["current_date"],
        "location": "Warehouse",
    }]


def test_readonly_handlers_with_session_timestamps_execute() -> None:
    filesystem = FilesystemServer(); fs_sid = _reset(filesystem)
    fs_state = filesystem.sessions[fs_sid]
    path = next(
        path for path, node in fs_state["fs"].items()
        if node["type"] == "file"
    )
    stat = filesystem._call_tool({
        "session_id": fs_sid, "name": "stat", "arguments": {"path": path},
    })
    assert stat["success"] is True
    assert stat["state_changed"] is False
    assert stat["observation"]["modified"].startswith(fs_state["current_date"])

    delivery = FoodDeliveryServer(); food_sid = _reset(delivery)
    food_state = delivery.sessions[food_sid]
    order = next(iter(food_state["orders"].values()))
    order["status"] = "delivering"
    tracked = delivery._call_tool({
        "session_id": food_sid,
        "name": "track_rider",
        "arguments": {"order_id": order["order_id"]},
    })
    assert tracked["success"] is True
    assert tracked["state_changed"] is False
    assert tracked["observation"]["estimated_arrival"].startswith(food_state["current_date"])


def test_comment_issue_uses_session_date_and_mutates_only_issue() -> None:
    server = IssueTrackerServer(); sid = _reset(server)
    state = server.sessions[sid]
    issue = next(iter(state["issues"].values()))
    result = server._call_tool({
        "session_id": sid,
        "name": "comment_issue",
        "arguments": {"issue_id": issue["issue_id"], "body": "Audit comment"},
    })
    assert result["success"] is True
    assert result["state_delta_paths"]
    assert result["observation"]["issue"]["comments"][-1]["timestamp"] == state["current_date"]


def test_thread_reply_reaction_updates_both_views_without_aliasing() -> None:
    server = TeamChatServer(); sid = _reset(server)
    state = server.sessions[sid]
    channel = next(channel for channel in state["channels"].values() if channel["messages"])
    root = channel["messages"][0]
    thread = server._call_tool({
        "session_id": sid,
        "name": "create_thread",
        "arguments": {"channel_id": channel["channel_id"], "message_id": root["message_id"]},
    })["observation"]["thread"]
    reply = server._call_tool({
        "session_id": sid,
        "name": "send_message",
        "arguments": {
            "channel_id": channel["channel_id"],
            "thread_id": thread["thread_id"],
            "content": "Thread reply",
        },
    })["observation"]["message"]
    channel_reply = next(item for item in channel["messages"] if item["message_id"] == reply["message_id"])
    thread_reply = state["threads"][thread["thread_id"]]["messages"][0]
    assert channel_reply is not thread_reply

    reacted = server._call_tool({
        "session_id": sid,
        "name": "react_message",
        "arguments": {
            "channel_id": channel["channel_id"],
            "message_id": reply["message_id"],
            "reaction": "audit-reaction",
        },
    })
    assert reacted["success"] is True
    assert any(path.startswith("channels") for path in reacted["state_delta_paths"])
    assert any(path.startswith("threads") for path in reacted["state_delta_paths"])
    assert "audit-reaction" in channel_reply["reactions"]
    assert "audit-reaction" in thread_reply["reactions"]


def test_seed_namespaces_do_not_repeat_at_old_modulus_boundary() -> None:
    seeder = StateSeeder()
    banking_a = seeder.seed_state("banking", "a", 1)
    banking_b = seeder.seed_state("banking", "b", 4097)
    assert set(banking_a["accounts"]).isdisjoint(banking_b["accounts"])
    fs_a = seeder.seed_state("filesystem", "a", 1)
    fs_b = seeder.seed_state("filesystem", "b", 4097)
    assert fs_a != fs_b


def test_estimator_metadata_contract_does_not_reconstruct_old_rows() -> None:
    with pytest.raises(ValueError, match="missing canonical metadata"):
        validate_livemcp_non_tensor_batch(
            {"extra_info": [{"group_id": "g"}]}, batch_size=1,
        )
    canonical = {
        "group_id": ["g", "g"],
        "perturbation_level": ["none", "none"],
        "scenario_type": ["normal", "normal"],
    }
    assert validate_livemcp_non_tensor_batch(canonical, 2) is canonical
    with pytest.raises(ValueError, match="length mismatch"):
        validate_livemcp_non_tensor_batch(canonical, 3)


def test_idempotent_mutations_do_not_become_execution_errors() -> None:
    shopping = ShoppingServer(); shopping_sid = _reset(shopping)
    product_id = next(iter(shopping.sessions[shopping_sid]["products"]))
    assert shopping._call_tool({
        "session_id": shopping_sid,
        "name": "add_to_cart",
        "arguments": {"product_id": product_id, "quantity": 1},
    })["success"] is True
    same_quantity = shopping._call_tool({
        "session_id": shopping_sid,
        "name": "update_cart_quantity",
        "arguments": {"product_id": product_id, "quantity": 1},
    })
    assert same_quantity["success"] is True
    assert same_quantity["state_changed"] is False
    assert same_quantity["state_delta_paths"] == []

    filesystem = FilesystemServer(); filesystem_sid = _reset(filesystem)
    cwd = filesystem.sessions[filesystem_sid]["cwd"]
    same_cwd = filesystem._call_tool({
        "session_id": filesystem_sid,
        "name": "cd",
        "arguments": {"path": cwd},
    })
    assert same_cwd["success"] is True
    assert same_cwd["state_changed"] is False
    assert same_cwd["state_delta_paths"] == []

    source = "/home/user/notes.txt"
    split_args = {"path": source, "lines_per_file": 1}
    assert filesystem._call_tool({
        "session_id": filesystem_sid,
        "name": "split",
        "arguments": split_args,
    })["success"] is True
    repeated_split = filesystem._call_tool({
        "session_id": filesystem_sid,
        "name": "split",
        "arguments": split_args,
    })
    assert repeated_split["success"] is True
    assert repeated_split["state_changed"] is False
    assert repeated_split["state_delta_paths"] == []


def test_round_reference_tool_miss_is_diagnostic_not_contract_failure() -> None:
    task = {"round_contracts": [
        {"round_idx": 0, "required_tools": ["reference_tool"], "allowed_terminal_actions": ["final_answer"]},
        {"round_idx": 1, "required_tools": [], "allowed_terminal_actions": ["final_answer"]},
    ]}
    events = [
        {"action_type": "round_tool_diagnostic"},
        {"action_type": "final_answer", "round_idx": 0},
        {"action_type": "final_answer", "round_idx": 1},
    ]
    assert _validate_round_contracts(events, task) == (True, "rounds_ok=2/2")


def test_round_contract_rejects_missing_single_round_terminal() -> None:
    task = {"round_contracts": [{
        "round_idx": 0,
        "required_tools": ["list_accounts"],
        "allowed_terminal_actions": ["final_answer"],
    }]}
    events = [{
        "action_type": "tool_call",
        "tool_name": "list_accounts",
        "round_idx": 0,
    }]
    assert _validate_round_contracts(events, task) == (
        False,
        "terminal count mismatch: got 0 terminals, expected 1 (fewer)",
    )


def test_task_reward_does_not_match_future_round_call_early() -> None:
    reward = reward_module.TaskReward()
    events = EventLog(events=[
        AuditEvent(
            event_id="round0", session_id="s", step=0, round_idx=0,
            action_type="tool_call", tool_name="list_accounts",
            tool_arguments={}, tool_name_known=True, schema_valid=True,
            execution_success=True,
        ),
        AuditEvent(
            event_id="future-early", session_id="s", step=1, round_idx=0,
            action_type="tool_call", tool_name="freeze_account",
            tool_arguments={"account_id": "acc_1"}, tool_name_known=True,
            schema_valid=True, execution_success=True,
        ),
    ])
    result = reward.compute(events, {
        "required_tool_calls": [
            {"tool_name": "list_accounts", "arguments": {}},
            {"tool_name": "freeze_account", "arguments": {"account_id": "acc_1"}},
        ],
        "required_call_rounds": [0, 1],
        "dependency_edges": [],
        "budget": 4,
    })
    assert result.r_coverage == pytest.approx(0.5)
    assert result.aligned_calls == 1


def test_reward_exception_fails_closed(monkeypatch) -> None:
    def boom(*args, **kwargs):
        raise RuntimeError("reward contract exploded")

    monkeypatch.setattr(reward_module._task_reward, "compute", boom)
    with pytest.raises(reward_module.RewardIntegrityError, match="reward contract exploded"):
        reward_module.compute_score(
            "live",
            "",
            {"oracle_calls": json.dumps([{
                "action": "tool_call", "tool_name": "list_accounts", "arguments": {},
            }]), "success_criteria": "[]"},
            {
                "domain": "banking",
                "allowed_terminal_actions": ["final_answer"],
                "round_contracts": [{
                    "round_idx": 0,
                    "required_tools": ["list_accounts"],
                    "allowed_terminal_actions": ["final_answer"],
                }],
                "audit_events": [{
                    "event_id": "terminal",
                    "session_id": "s",
                    "step": 0,
                    "action_type": "final_answer",
                    "terminal_action": "final_answer",
                }],
            },
        )


def test_optional_reward_components_also_fail_closed(monkeypatch) -> None:
    def boom(*args, **kwargs):
        raise RuntimeError("progress contract exploded")

    monkeypatch.setattr(reward_module._progress_tracker, "compute", boom)
    with pytest.raises(
        reward_module.RewardIntegrityError,
        match="progress contract exploded",
    ):
        reward_module._compute_f_gamma(
            reward_module.EventLog(events=[]), {}, domain_adapter=None,
        )


def test_malformed_audit_event_fails_closed() -> None:
    with pytest.raises(reward_module.RewardIntegrityError, match="invalid audit event type"):
        reward_module._parse_audit_events([42])


def test_lambda_state_corruption_is_not_treated_as_default(tmp_path) -> None:
    path = tmp_path / "lambda.json"
    path.write_text("not-json", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        LambdaState.load_or_default(path=str(path))


def test_oval_full_requires_canonical_safety_batch(monkeypatch) -> None:
    monkeypatch.setenv("OVAL_REWARD_PROFILE", "oval_full")
    with pytest.raises(ValueError, match="canonical top-level c_safety"):
        update_lambda_safe({}, 2)
    with pytest.raises(ValueError, match="length mismatch"):
        update_lambda_safe({"c_safety": [0]}, 2)


def test_tool_operation_resolution_rejects_unknown_semantics() -> None:
    assert resolve_tool_operation("remove_attendee", "calendar") == "update"
    assert resolve_tool_operation("cancel_payment", "payments") == "update"
    assert resolve_tool_operation("archive_email", "email") == "update"
    assert resolve_tool_operation("delete_webhook", "payments") == "delete"
    with pytest.raises(ValueError, match="unknown public tool semantics"):
        resolve_tool_operation("legacy_delete_invoice", "payments")


def test_server_rejects_mutation_outside_tool_footprint() -> None:
    server = BankingServer()
    sid = _reset(server)

    def faulty_freeze(session_id, _arguments):
        server._state(session_id)["loans"]["rogue"] = {"status": "approved"}
        return server_result(True, {}, None, "", True)

    server.handlers["freeze_account"] = faulty_freeze
    result = server._call_tool({
        "session_id": sid,
        "name": "freeze_account",
        "arguments": {"account_id": "unused", "reason": "audit"},
    })
    assert result["success"] is False
    assert "disallowed state roots" in result["error_message"]
    assert "rogue" not in server.sessions[sid]["loans"]


def test_rollout_closes_session_when_inner_loop_raises(monkeypatch) -> None:
    loop = object.__new__(LiveMCPOvalLoop)

    class Context:
        closed: list[str] = []

        def close_session(self, session_id: str) -> None:
            self.closed.append(session_id)

    ctx = Context()

    async def fail(_sampling_params, **kwargs):
        kwargs["_session_cleanup"].append((ctx, "leaked-session"))
        raise RuntimeError("unexpected rollout failure")

    monkeypatch.setattr(loop, "_run_impl", fail)
    with pytest.raises(RuntimeError, match="unexpected rollout failure"):
        asyncio.run(loop.run({}))
    assert ctx.closed == ["leaked-session"]


def test_malformed_handler_response_rolls_back_state() -> None:
    server = StatefulToolServer("contract_test", [])
    server.sessions["s"] = {"counter": 0}

    def malformed(session_id, _arguments):
        server.sessions[session_id]["counter"] = 1
        return None

    server.handlers["mutate_then_break"] = malformed
    result = server._call_tool({
        "session_id": "s", "name": "mutate_then_break", "arguments": {},
    })
    assert result["success"] is False
    assert result["error_type"] == errors.EXECUTION_ERROR
    assert server.sessions["s"] == {"counter": 0}


def test_handler_cannot_escape_transaction_by_deleting_session() -> None:
    server = StatefulToolServer("contract_test", [])
    server.sessions["s"] = {"counter": 0}

    def delete_session(session_id, _arguments):
        server.sessions.pop(session_id)
        return server_result(True, {}, None, "", True)

    server.handlers["delete_session"] = delete_session
    result = server._call_tool({
        "session_id": "s", "name": "delete_session", "arguments": {},
    })
    assert result["success"] is False
    assert server.sessions["s"] == {"counter": 0}


def test_executor_timeout_quarantines_unknown_commit_session() -> None:
    registry = SchemaRegistry()
    registry.register_tools("calendar", [{
        "name": "mutate",
        "input_schema": {"type": "object", "properties": {}},
    }])

    class Manager:
        quarantined = []

        def call_tool(self, *_args, **_kwargs):
            raise TransportError(errors.TIMEOUT, "late response")

        def quarantine_session(self, session_id, reason):
            self.quarantined.append((session_id, reason))

    manager = Manager()
    result = LiveMCPExecutor(manager, registry).execute(
        "s-timeout", ToolCall(name="mutate", arguments={}, call_id="c"),
        domain="calendar",
    )
    assert result.success is False
    assert result.error_type == errors.TIMEOUT
    assert manager.quarantined
    assert manager.quarantined[0][0] == "s-timeout"


def test_worker_session_creation_is_atomic_on_discovery_failure() -> None:
    worker = object.__new__(OvalMCPWorkerContext)

    class Manager:
        closed = []

        def create_session(self, seed):
            return type("Session", (), {"session_id": "partial"})()

        def discover_tools(self, _session_id):
            raise RuntimeError("schema discovery failed")

        def close_session(self, session_id):
            self.closed.append(session_id)

    worker.manager = Manager()
    with pytest.raises(RuntimeError, match="schema discovery failed"):
        worker.create_session(seed=7)
    assert worker.manager.closed == ["partial"]


def test_audit_event_roundtrip_preserves_state_evidence_failure() -> None:
    event = AuditEvent(
        event_id="e", session_id="s", step=0, action_type="tool_call",
        pre_state_status="available", post_state_status="error",
        state_evidence_errors=["post_state:TimeoutError:timed out"],
    )
    restored = reward_module._dict_to_audit_event(event.to_dict())
    assert restored.post_state_status == "error"
    assert restored.state_evidence_errors == [
        "post_state:TimeoutError:timed out"
    ]
