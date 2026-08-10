from __future__ import annotations

import importlib
import asyncio
import datetime
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import src.reward.oval_reward_fn as reward_module

from src.agent_loop.livemcp_oval_loop import (
    LiveMCPOvalLoop,
    _derive_sampling_seed,
    _next_identical_action_count,
    _parse_tool_calls_json,
    _resolve_terminal_type,
    _single_conversation_token_ids,
    _unknown_tool_audit_event,
    _invalid_mixed_action_audit_event,
    _validate_environment_metadata,
)
from src.agent_loop.livemcp_oval_worker import OvalMCPWorkerContext
from src.live_mcp import errors
from src.live_mcp.executor import LiveMCPExecutor
from src.live_mcp.protocol.observation import (
    TRAJECTORY_SCHEMA_VERSION,
    OBSERVATION_SCHEMA_VERSION,
    OBSERVATION_PROJECTION_VERSION,
    compute_server_schema_hash,
    serialize_tool_result,
)
from src.live_mcp.registry.environment_metadata import (
    build_environment_metadata,
    compute_initial_state_hashes,
    compute_reward_fingerprint,
    compute_transition_fingerprint,
    normalize_state_profiles,
    state_profiles_for_suite,
)
from src.live_mcp.corpus.shard_row_projection import _tool_owner_domains
from src.live_mcp.registry.environment_metadata import validate_tool_owner_contract
from src.live_mcp.dependency_value_flow import _field_values
from src.live_mcp.generation.scenario import (
    detect_duplicate_side_effect as _detect_duplicate_side_effect,
    detect_missing_dependency as _detect_missing_dependency,
)
from src.live_mcp.replay.task_outcome import (
    attribute_success_criteria as _attribute_success_criteria,
)
from src.live_mcp.config import load_suite_config
from src.live_mcp.protocol.manager import LiveMCPManager
from src.live_mcp.state_seeder import StateSeeder
from src.live_mcp.types import OracleCall, ToolCall, ToolExecutionResult
from src.oval_mcp.envs.domain_adapter import get_adapter
from src.oval_mcp.envs.audit_wrapper import AuditWrapper
from src.training.hyperparams import LiveMCPHyperparams
from src.training.trainer_config import TrainerConfig
from src.training.run_grpo import _bind_profile_estimator
from src.training.hooks import (
    update_lambda_safe,
    validate_livemcp_non_tensor_batch,
)
from src.reward.oval_reward_fn import (
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
from src.live_mcp.registry.tool_semantics import build_tool_semantics, resolve_tool_operation
from src.live_mcp.registry.schemas import SchemaRegistry
from src.live_mcp.protocol.transport import TransportError
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


def test_observation_budget_override_and_suite_default_share_one_resolver() -> None:
    suite_rollout = {"observation_max_chars": 4096}
    assert LiveMCPHyperparams(
        max_observation_chars=0,
    ).resolve_max_observation_chars(suite_rollout) == 4096
    assert LiveMCPHyperparams(
        max_observation_chars=8192,
    ).resolve_max_observation_chars(suite_rollout) == 8192


def test_rollout_contract_controls_are_typed_and_exported(monkeypatch) -> None:
    monkeypatch.setenv("OVAL_PLAIN_FINAL_COMPAT", "1")
    monkeypatch.setenv("OVAL_LAMBDA_STATE_PATH", "/tmp/audit-lambda.json")
    cfg = LiveMCPHyperparams.from_env()
    assert cfg.plain_final_compat is True
    assert cfg.lambda_state_path == "/tmp/audit-lambda.json"

    cfg.plain_final_compat = False
    cfg.export_env()
    assert os.environ["OVAL_PLAIN_FINAL_COMPAT"] == "False"
    assert cfg.to_dict()["lambda_state_path"] == "/tmp/audit-lambda.json"


def test_trainer_defaults_to_available_sdpa_attention_backend() -> None:
    config = TrainerConfig(train_file="missing.parquet")
    assert config.attn_implementation == "sdpa"
    assert config.use_remove_padding is True
    assert (
        "+actor_rollout_ref.model.override_config.attn_implementation=sdpa"
        in config.to_hydra_overrides()
    )
    assert (
        "actor_rollout_ref.model.use_remove_padding=true"
        in config.to_hydra_overrides()
    )


def test_trainer_can_disable_remove_padding(monkeypatch) -> None:
    monkeypatch.setenv("OVAL_USE_REMOVE_PADDING", "false")
    config = TrainerConfig.from_env(train_file="missing.parquet")
    assert config.use_remove_padding is False
    assert (
        "actor_rollout_ref.model.use_remove_padding=false"
        in config.to_hydra_overrides()
    )


def test_trainer_can_disable_redundant_overlong_filter(monkeypatch) -> None:
    monkeypatch.setenv("OVAL_FILTER_OVERLONG_PROMPTS", "false")
    config = TrainerConfig.from_env(train_file="missing.parquet")
    assert config.filter_overlong_prompts is False
    assert "data.filter_overlong_prompts=false" in config.to_hydra_overrides()


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
runtime = reward.RewardRuntime.from_environment()
assert runtime.cfg.reward_profile == 'prove_baseline'
assert runtime.cfg.i_shape == 0
assert runtime.cfg.i_process == 0
assert runtime.cfg.lambda_safe_default == 0.0
result = reward.compute_score('live', '', {}, {
    'reward_profile': 'prove_baseline',
    'prompt_profile': 'local_trainable_v1',
    'semantic_gate_profile': 'deterministic_v1',
    'artifact_purpose': 'training_candidate',
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


def test_get_thread_uses_each_domains_registered_output_entities() -> None:
    email_trace = [
        OracleCall("get_thread", {}),
        OracleCall("reply_email", {}),
    ]
    team_chat_trace = [
        OracleCall("get_thread", {}),
        OracleCall("react_message", {}),
    ]

    assert not _detect_missing_dependency(email_trace, "email")
    # team_chat.get_thread publicly returns channel/message identifiers that
    # can ground react_message. The former same-name exception hid this value
    # flow even though the handler contract exposes it.
    assert not _detect_missing_dependency(team_chat_trace, "team_chat")


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


def test_tool_keyed_domain_contracts_reference_current_schemas_only() -> None:
    from src.live_mcp.domain_contracts.dependency import (
        _DEPENDENCY_TOOL_STATE_POSTCONDITIONS,
        _DEPENDENCY_TOOL_STATE_PRECONDITIONS,
    )
    from src.live_mcp.domain_contracts.entities import _TOOL_ENTITY_OVERRIDE
    from src.live_mcp.domain_contracts.requirements import (
        _DOMAIN_PROBE_PRIMARY_ENTITY_TYPES,
        _DOMAIN_TOOL_RELEVANT,
    )

    all_tool_names: set[str] = set()
    for domain in DOMAINS:
        module = importlib.import_module(
            f"src.live_mcp.servers.{domain}.server"
        )
        live_names = {tool["name"] for tool in module.TOOLS}
        all_tool_names.update(live_names)
        assert set(_DOMAIN_TOOL_RELEVANT.get(domain, {})) <= live_names
        assert set(_DOMAIN_PROBE_PRIMARY_ENTITY_TYPES.get(domain, {})) <= live_names
        assert set(_DEPENDENCY_TOOL_STATE_PRECONDITIONS[domain]) == live_names
        assert set(_DEPENDENCY_TOOL_STATE_POSTCONDITIONS[domain]) == live_names

    assert set(_TOOL_ENTITY_OVERRIDE) <= all_tool_names


def test_shopping_adapter_has_specific_semantics_for_all_public_tools() -> None:
    module = importlib.import_module("src.live_mcp.servers.shopping.server")
    adapter = get_adapter("shopping")
    adapter.register_tool_schemas(module.TOOLS)

    assert set(adapter.TOOL_MAP) == {
        tool["name"] for tool in module.TOOLS
    }
    for tool in module.TOOLS:
        _operation, target_type = adapter.tool_semantics(
            tool["name"], "shopping_resource",
        )
        assert target_type != "shopping_resource"

    returned = adapter.normalize_event(
        action_type="tool_call",
        tool_name="return_order",
        tool_arguments={"order_id": "ord_s42_0001", "reason": "damaged"},
        observation={
            "return": {
                "return_id": "ret_s42_0003",
                "order_id": "ord_s42_0001",
            },
        },
        execution_success=True,
        state_changed=True,
        before_state=None,
        after_state=None,
    )
    assert returned["target_type"] == "shopping_return"
    assert returned["target_id"] == "ord_s42_0001"
    assert returned["created_ids"] == ["ret_s42_0003"]

    reviewed = adapter.normalize_event(
        action_type="tool_call",
        tool_name="add_review",
        tool_arguments={"product_id": "prd_s42_001", "rating": 5, "body": "ok"},
        observation={
            "review": {
                "review_id": "rev_s42_0004",
                "product_id": "prd_s42_001",
            },
        },
        execution_success=True,
        state_changed=True,
        before_state=None,
        after_state=None,
    )
    assert reviewed["target_type"] == "product_review"
    assert reviewed["target_id"] == "prd_s42_001"
    assert reviewed["created_ids"] == ["rev_s42_0004"]


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
        "state_profiles": {
            domain: "baseline" for domain in tools_by_domain
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
        "state_profiles": {"calendar": "baseline", "shopping": "baseline"},
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


def test_policy_visible_tool_owner_contract_is_exact_and_unambiguous() -> None:
    visible = [
        {"name": "list_events"},
        {"name": "search_products", "_server_name": "shopping"},
    ]
    owners = _tool_owner_domains(visible, "calendar")
    extra = {
        "visible_tool_names": ["list_events", "search_products"],
        "tool_owner_domains": json.dumps(owners),
    }

    assert owners == {
        "list_events": "calendar",
        "search_products": "shopping",
    }
    assert validate_tool_owner_contract(extra) == owners


def test_policy_visible_duplicate_or_unbound_tool_owner_fails_closed() -> None:
    with pytest.raises(RuntimeError, match="ambiguous"):
        _tool_owner_domains([
            {"name": "get_order", "_server_name": "shopping"},
            {"name": "get_order", "_server_name": "food_delivery"},
        ], "shopping")

    with pytest.raises(RuntimeError, match="exactly cover"):
        validate_tool_owner_contract({
            "visible_tool_names": ["list_events", "search_products"],
            "tool_owner_domains": {"list_events": "calendar"},
        })

    for owner in (None, 7, ""):
        with pytest.raises(RuntimeError, match="non-empty strings"):
            validate_tool_owner_contract({
                "visible_tool_names": ["list_events"],
                "tool_owner_domains": {"list_events": owner},
            })


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
    assert isinstance(metadata["reward_profile_fingerprints"], dict)
    assert set(metadata["reward_profile_fingerprints"].keys()) == {"prove_baseline", "oval_full"}


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

    task_reward = reward_module.TaskReward(reward_profile="prove_baseline")
    baseline = task_reward.compute(events, task)
    assert baseline.r_validity == 1.0
    assert baseline.r_coverage == 1.0

    oval = task_reward.compute(events, {
        **task,
        "apply_terminal_validity_penalty": True,
        "apply_identity_coverage_penalty": True,
    })
    assert oval.r_validity == 0.5
    assert oval.r_coverage == 0.0


def test_argument_equality_keeps_boolean_and_number_types_distinct() -> None:
    equal = reward_module.TaskReward._args_equal
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


def test_strict_prove_profile_fails_closed_until_sources_exist(
    monkeypatch,
) -> None:
    monkeypatch.setenv("OVAL_EXPERIMENT_PROFILE", "prove_reproduction_v1")
    monkeypatch.setenv("OVAL_REWARD_PROFILE", "prove_baseline")
    with pytest.raises(ValueError, match="is unavailable"):
        TrainerConfig.from_env()


def test_training_run_name_is_injectable_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("OVAL_RUN_NAME", "smoke_seed_41")
    assert TrainerConfig.from_env().generate_run_name() == "smoke_seed_41"


def test_training_seed_reaches_data_sampler_and_vllm_rollout(monkeypatch) -> None:
    from verl.workers.config.rollout import RolloutConfig

    monkeypatch.setenv("OVAL_SEED", "41")
    overrides = TrainerConfig.from_env(train_file="missing.parquet").to_hydra_overrides()
    assert "data.seed=41" in overrides
    assert "+actor_rollout_ref.rollout.seed=41" in overrides
    assert RolloutConfig(name="vllm", seed=41).seed == 41


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


def test_calendar_seed_covers_public_event_state_variants() -> None:
    events = StateSeeder().seed_state(
        "calendar", "calendar-state-audit", 42,
    )["events"].values()

    assert len(events) == 20
    assert any(event.get("recurrence") for event in events)
    assert any(not event.get("recurrence") for event in events)
    assert any(event.get("reminders") for event in events)
    assert any(not event.get("reminders") for event in events)

    server = CalendarServer()
    server.handle_request(
        "session/reset", {"session_id": "calendar-state-audit", "seed": 42},
    )
    recurring_id = next(
        event["event_id"]
        for event in server.sessions["calendar-state-audit"]["events"].values()
        if event.get("recurrence")
    )
    result = server._call_tool({
        "session_id": "calendar-state-audit",
        "name": "get_recurring_info",
        "arguments": {"event_id": recurring_id},
    })
    assert result["success"] is True
    assert result["observation"]["recurrence"]


def test_payments_rare_state_profile_preserves_baseline_and_relations() -> None:
    seed = 2026072801
    seeder = StateSeeder()
    baseline = seeder.seed_state("payments", "baseline-audit", seed)
    canonical = json.dumps(
        baseline, sort_keys=True, separators=(",", ":"), default=str,
    )
    assert hashlib.sha256(canonical.encode()).hexdigest() == (
        "357cec88546a4fabfdbf599914a8d317da30da37b85200e8da12e97b5fec3360"
    )
    assert len(baseline["invoices"]) == 20
    assert len(baseline["payments"]) == 5
    assert len(baseline["refunds"]) == 3
    assert len(baseline["webhooks"]) == 3
    assert all(
        payment["invoice_id"] in baseline["invoices"]
        for payment in baseline["payments"].values()
    )
    assert all(
        refund["invoice_id"] in baseline["invoices"]
        for refund in baseline["refunds"].values()
    )

    rare = seeder.seed_state(
        "payments", "rare-audit", seed, "payments_rare_state_v1"
    )
    assert len(rare["invoices"]) == len(baseline["invoices"]) == 20
    assert len(rare["payments"]) == 8
    assert sum(
        payment["status"] == "pending"
        for payment in rare["payments"].values()
    ) == 4
    assert len(rare["refunds"]) == 3
    assert len(rare["webhooks"]) == 3
    partial = next(
        inv for inv in rare["invoices"].values()
        if inv["status"] == "partially_refunded"
    )
    assert rare["payments"][partial["payment_id"]]["status"] == "settled"
    assert rare["refunds"][partial["refund_id"]]["invoice_id"] == partial["invoice_id"]
    for payment in rare["payments"].values():
        assert payment["invoice_id"] in rare["invoices"]


def test_state_profile_is_explicit_and_missing_rows_fail_closed() -> None:
    owners = {"payments", "calendar"}
    with pytest.raises(RuntimeError, match="missing state_profiles"):
        normalize_state_profiles(None, owners)
    suite = load_suite_config(
        "configs/live_mcp/ten_domain_suite_payments_rare_state_v1.yaml"
    )
    assert state_profiles_for_suite(suite, owners) == {
        "calendar": "baseline",
        "payments": "payments_rare_state_v1",
    }
    baseline_hashes = compute_initial_state_hashes(owners, 2026072801)
    rare_hashes = compute_initial_state_hashes(
        owners,
        2026072801,
        state_profiles_for_suite(suite, owners),
    )
    assert baseline_hashes["calendar"] == rare_hashes["calendar"]
    assert baseline_hashes["payments"] != rare_hashes["payments"]
    manager = LiveMCPManager(suite)
    assert manager._state_profiles(None)["payments"] == "payments_rare_state_v1"
    assert manager._state_profiles({})["payments"] == "baseline"

    with pytest.raises(ValueError, match="unsupported state profile"):
        StateSeeder().seed_state(
            "calendar", "invalid-profile", 42, "payments_rare_state_v1"
        )


def test_manager_scoped_session_resets_only_bound_environment() -> None:
    suite = load_suite_config("configs/live_mcp/ten_domain_suite.yaml")
    manager = LiveMCPManager(suite)
    calls: list[tuple[str, str, dict]] = []

    def request(server_name, method, params, **_kwargs):
        calls.append((server_name, method, dict(params)))
        return {}

    manager._request = request
    session = manager.create_session(
        seed=17,
        server_names=["shopping"],
    )
    try:
        assert session.server_names == ["shopping"]
        assert session.metadata["state_profiles"] == {"shopping": "baseline"}
        reset_calls = [call for call in calls if call[1] == "session/reset"]
        assert [call[0] for call in reset_calls] == ["shopping"]
        assert reset_calls[0][2]["seed"] == 17
    finally:
        manager.close_session(session.session_id)


def test_payments_server_rare_state_paths_are_executable() -> None:
    server = PaymentsServer()
    session_id = "payments-rare"
    server.handle_request(
        "session/reset",
        {
            "session_id": session_id,
            "seed": 2026072801,
            "state_profile": "payments_rare_state_v1",
        },
    )
    state = server.sessions[session_id]
    pending = next(
        payment for payment in state["payments"].values()
        if payment["status"] == "pending"
    )
    cancelled = server._call_tool({
        "session_id": session_id,
        "name": "cancel_payment",
        "arguments": {
            "payment_id": pending["payment_id"],
            "reason": "gray audit",
        },
    })
    assert cancelled["success"] is True

    partial = next(
        invoice for invoice in state["invoices"].values()
        if invoice["status"] == "partially_refunded"
    )
    refunded = server._call_tool({
        "session_id": session_id,
        "name": "refund_invoice",
        "arguments": {
            "invoice_id": partial["invoice_id"],
            "amount": 0.01,
            "reason": "gray audit",
        },
    })
    assert refunded["success"] is True

    webhook_id = next(iter(state["webhooks"]))
    deleted = server._call_tool({
        "session_id": session_id,
        "name": "delete_webhook",
        "arguments": {"webhook_id": webhook_id},
    })
    assert deleted["success"] is True


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
    baseline_transactions = list(server.sessions[sid]["transactions"])
    for name in ("fail_after_write", "wrong_change_flag"):
        result = server._call_tool({"session_id": sid, "name": name, "arguments": {}})
        assert result["success"] is False
        assert server.sessions[sid]["transactions"] == baseline_transactions
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
        "arguments": {
            "invoice_id": invoice["invoice_id"],
            "amount": invoice["amount"],
            "method": "card",
        },
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
    shopping_state = shopping.sessions[sid]
    reviewed = set(shopping_state["reviews"])
    product_id = next(
        item["product_id"]
        for order in shopping_state["orders"].values()
        if order["status"] in {"shipped", "returning", "returned"}
        for item in order["items"]
        if item["product_id"] not in reviewed
    )
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


def test_banking_schedule_respects_freeze_and_verification_minimizes_disclosure() -> None:
    server = BankingServer()
    sid = _reset(server)
    state = server.sessions[sid]
    frozen_id = next(
        aid for aid, account in state["accounts"].items() if account["frozen"]
    )
    other_id = next(aid for aid in state["accounts"] if aid != frozen_id)
    future_date = (
        datetime.date.fromisoformat(state["current_date"])
        + datetime.timedelta(days=1)
    ).isoformat()

    blocked = server._call_tool({
        "session_id": sid,
        "name": "schedule_transfer",
        "arguments": {
            "from_account": frozen_id,
            "to_account": other_id,
            "amount": 1,
            "execute_date": future_date,
        },
    })
    assert blocked["success"] is False
    assert "account frozen" in blocked["error_message"]

    verification = server._call_tool({
        "session_id": sid,
        "name": "verify_account",
        "arguments": {
            "account_id": frozen_id,
            "owner_name": "wrong owner",
        },
    })
    assert verification["success"] is True
    assert verification["observation"]["verified"] is False
    assert "owner" not in verification["observation"]

    adapter = get_adapter("banking")
    normalized = adapter.normalize_event(
        action_type="tool_call",
        tool_name="verify_account",
        tool_arguments={
            "account_id": frozen_id,
            "owner_name": "wrong owner",
        },
        observation=verification["observation"],
        execution_success=True,
        state_changed=False,
        before_state=None,
        after_state=None,
    )
    assert normalized["identity_violation"] == (
        "identity_or_provenance_violation"
    )


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
        "arguments": {"event_id": event_id, "email": email},
    })
    assert added["success"] is True
    before_repeat = json.dumps(state, sort_keys=True)
    repeated = server._call_tool({
        "session_id": sid, "name": "add_attendee",
        "arguments": {"event_id": event_id, "email": email},
    })
    assert repeated["success"] is True
    assert repeated["state_changed"] is False
    assert json.dumps(state, sort_keys=True) == before_repeat

    responded = server._call_tool({
        "session_id": sid, "name": "add_attendee",
        "arguments": {
            "event_id": event_id, "email": email,
            "response_status": "accepted",
        },
    })
    assert responded["success"] is True
    assert responded["state_changed"] is True
    assert state["events"][event_id]["responses"][email] == "accepted"

    first_reminder = server._call_tool({
        "session_id": sid, "name": "set_reminder",
        "arguments": {"event_id": event_id, "minutes_before": 15},
    })
    assert first_reminder["success"] is True
    reminder_count = len(state["events"][event_id]["reminders"])
    updated_reminder = server._call_tool({
        "session_id": sid, "name": "set_reminder",
        "arguments": {
            "event_id": event_id,
            "minutes_before": 15,
            "method": "email",
        },
    })
    assert updated_reminder["success"] is True
    assert updated_reminder["state_changed"] is True
    assert len(state["events"][event_id]["reminders"]) == reminder_count
    assert state["events"][event_id]["reminders"][-1]["method"] == "email"
    unchanged_reminder = server._call_tool({
        "session_id": sid, "name": "set_reminder",
        "arguments": {
            "event_id": event_id,
            "minutes_before": 15,
            "method": "email",
        },
    })
    assert unchanged_reminder["success"] is True
    assert unchanged_reminder["state_changed"] is False
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


def test_calendar_rejects_display_names_in_email_identity_fields() -> None:
    server = CalendarServer(); sid = _reset(server)
    state = server.sessions[sid]
    event_id = next(iter(state["events"]))
    before = json.dumps(state, sort_keys=True)
    cases = [
        ("create_event", {
            "title": "Invalid attendee",
            "start_time": "2026-07-20T10:00:00",
            "end_time": "2026-07-20T11:00:00",
            "attendees": ["Sarah"],
        }),
        ("update_event", {
            "event_id": event_id,
            "fields": {"attendees": ["Sarah"]},
        }),
        ("create_recurring", {
            "title": "Invalid recurring attendee",
            "start_time": "2026-07-20T10:00:00",
            "end_time": "2026-07-20T11:00:00",
            "recurrence": "FREQ=WEEKLY;BYDAY=MO",
            "attendees": ["Sarah"],
        }),
        ("add_attendee", {"event_id": event_id, "email": "Sarah"}),
        ("remove_attendee", {"event_id": event_id, "email": "Sarah"}),
        ("get_free_busy", {
            "emails": ["Sarah"],
            "start_time": "2026-07-20T10:00:00",
            "end_time": "2026-07-20T11:00:00",
        }),
        ("respond_to_event", {
            "event_id": event_id,
            "email": "Sarah",
            "response": "accepted",
        }),
    ]
    for name, arguments in cases:
        result = server._call_tool({
            "session_id": sid,
            "name": name,
            "arguments": arguments,
        })
        assert result["success"] is False, name
        assert result["state_changed"] is False, name
        assert "valid email address" in result["error_message"], name
        assert json.dumps(state, sort_keys=True) == before, name

    schemas = {tool["name"]: tool["input_schema"] for tool in server.tools}
    assert schemas["create_event"]["properties"]["attendees"]["items"]["format"] == "email"
    assert schemas["get_free_busy"]["properties"]["emails"]["minItems"] == 1
    assert schemas["add_attendee"]["properties"]["email"]["format"] == "email"


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


def test_payments_disputed_invoice_cannot_be_paid() -> None:
    server = PaymentsServer(); sid = _reset(server)
    state = server.sessions[sid]
    invoice = next(
        item for item in state["invoices"].values()
        if item["status"] == "pending" and not item.get("payment_id")
    )
    disputed = server._call_tool({
        "session_id": sid,
        "name": "dispute_invoice",
        "arguments": {
            "invoice_id": invoice["invoice_id"],
            "reason": "duplicate charge",
        },
    })
    assert disputed["success"] is True
    assert invoice["status"] == "disputed"
    dispute_id = disputed["observation"]["dispute"]["dispute_id"]
    before = json.dumps(state, sort_keys=True)

    result = server._call_tool({
        "session_id": sid,
        "name": "pay_invoice",
        "arguments": {
            "invoice_id": invoice["invoice_id"],
            "amount": invoice["amount"],
        },
    })
    assert result["success"] is False
    assert result["state_changed"] is False
    assert "cannot pay invoice in status: disputed" in result["error_message"]
    assert json.dumps(state, sort_keys=True) == before
    assert state["disputes"][dispute_id]["status"] == "open"
    assert invoice["status"] == "disputed"
    assert invoice.get("payment_id") is None

    # Defend against an already-inconsistent imported/seeded state too: the
    # open dispute is authoritative even if the invoice status was stale.
    invoice = state["invoices"][invoice["invoice_id"]]
    invoice["status"] = "pending"
    inconsistent_before = json.dumps(state, sort_keys=True)
    open_dispute_result = server._call_tool({
        "session_id": sid,
        "name": "pay_invoice",
        "arguments": {
            "invoice_id": invoice["invoice_id"],
            "amount": invoice["amount"],
        },
    })
    assert open_dispute_result["success"] is False
    assert open_dispute_result["state_changed"] is False
    assert f"invoice has open dispute: {dispute_id}" in open_dispute_result["error_message"]
    assert json.dumps(state, sort_keys=True) == inconsistent_before


def test_payments_require_user_payment_method_and_nonempty_webhook_events() -> None:
    server = PaymentsServer()
    sid = _reset(server)
    state = server.sessions[sid]
    invoice = next(
        item for item in state["invoices"].values()
        if item["status"] in {"pending", "overdue"}
        and not item.get("payment_id")
    )
    before = json.dumps(state, sort_keys=True)

    missing_method = server._call_tool({
        "session_id": sid,
        "name": "pay_invoice",
        "arguments": {
            "invoice_id": invoice["invoice_id"],
            "amount": invoice["amount"],
        },
    })
    assert missing_method["success"] is False
    assert "method" in missing_method["error_message"]
    assert json.dumps(state, sort_keys=True) == before

    empty_events = server._call_tool({
        "session_id": sid,
        "name": "create_webhook",
        "arguments": {
            "url": "https://example.com/hook",
            "events": [],
        },
    })
    assert empty_events["success"] is False
    assert "events must be non-empty" in empty_events["error_message"]


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
    coupon_result = server._call_tool({
        "session_id": sid, "name": "apply_coupon",
        "arguments": {"code": "FREESHIP"},
    })
    assert coupon_result["success"]
    assert coupon_result["observation"]["discount"] == "free shipping"
    after = server._call_tool({
        "session_id": sid, "name": "get_cart", "arguments": {},
    })["observation"]
    assert after["shipping"] == 0
    assert after["final_total"] < before["final_total"]
    checkout_order = server._call_tool({
        "session_id": sid, "name": "checkout",
        "arguments": {"shipping_address": "1 Audit Way", "payment_method": "Visa"},
    })["observation"]["order"]
    assert set(checkout_order) == {"order_id", "status"}
    order = server._call_tool({
        "session_id": sid, "name": "get_order",
        "arguments": {"order_id": checkout_order["order_id"]},
    })["observation"]["order"]
    assert order["shipping"] == 0


def test_shopping_return_rejects_order_that_has_not_shipped() -> None:
    server = ShoppingServer()
    sid = _reset(server)
    pending = next(
        order for order in server.sessions[sid]["orders"].values()
        if order["status"] == "pending"
    )
    result = server._call_tool({
        "session_id": sid,
        "name": "return_order",
        "arguments": {
            "order_id": pending["order_id"],
            "reason": "damaged",
        },
    })
    assert result["success"] is False
    assert "only shipped orders can be returned" in result["error_message"]


def test_shopping_return_rejects_generic_reason_placeholder() -> None:
    server = ShoppingServer()
    sid = _reset(server)
    shipped = next(
        order for order in server.sessions[sid]["orders"].values()
        if order["status"] == "shipped"
    )
    result = server._call_tool({
        "session_id": sid,
        "name": "return_order",
        "arguments": {
            "order_id": shipped["order_id"],
            "reason": "User requested return",
        },
    })
    assert result["success"] is False
    assert "concrete user-provided reason" in result["error_message"]
    assert shipped["status"] == "shipped"


def test_shopping_return_validates_item_membership_and_whole_order_semantics() -> None:
    server = ShoppingServer()
    sid = _reset(server)
    shipped = next(
        order for order in server.sessions[sid]["orders"].values()
        if order["status"] == "shipped"
    )

    invalid = server._call_tool({
        "session_id": sid,
        "name": "return_order",
        "arguments": {
            "order_id": shipped["order_id"],
            "reason": "damaged",
            "items": ["prd_missing"],
        },
    })
    assert invalid["success"] is False
    assert "product not in order" in invalid["error_message"]
    assert shipped["status"] == "shipped"

    whole_order = server._call_tool({
        "session_id": sid,
        "name": "return_order",
        "arguments": {
            "order_id": shipped["order_id"],
            "reason": "damaged",
        },
    })
    assert whole_order["success"] is True
    assert whole_order["observation"]["return"]["items"] == [
        item["product_id"] for item in shipped["items"]
    ]
    assert whole_order["observation"]["return"]["item_details"] == shipped["items"]
    assert _field_values(whole_order["observation"], "product_id") == [
        item["product_id"] for item in shipped["items"]
    ]

    return_id = whole_order["observation"]["return"]["return_id"]
    status = server._call_tool({
        "session_id": sid,
        "name": "get_return_status",
        "arguments": {"return_id": return_id},
    })
    assert status["success"] is True
    assert _field_values(status["observation"], "product_id") == [
        item["product_id"] for item in shipped["items"]
    ]
    assert "request has been recorded" in status["observation"]["status_description"]
    assert "does not expose a shipping label" in status["observation"]["status_description"]


def test_shopping_rejects_semantically_incomplete_checkout_and_comparison() -> None:
    server = ShoppingServer()
    sid = _reset(server)
    product_ids = list(server.sessions[sid]["products"])
    assert server._call_tool({
        "session_id": sid,
        "name": "add_to_cart",
        "arguments": {"product_id": product_ids[0], "quantity": 1},
    })["success"]

    incomplete_checkout = server._call_tool({
        "session_id": sid,
        "name": "checkout",
        "arguments": {},
    })
    assert incomplete_checkout["success"] is False
    assert "shipping_address" in incomplete_checkout["error_message"]

    blank_address = server._call_tool({
        "session_id": sid,
        "name": "checkout",
        "arguments": {"shipping_address": " ", "payment_method": "card"},
    })
    assert blank_address["success"] is False
    assert "shipping_address must be non-empty" in blank_address["error_message"]

    blank_payment_method = server._call_tool({
        "session_id": sid,
        "name": "checkout",
        "arguments": {"shipping_address": "1 Audit Way", "payment_method": " "},
    })
    assert blank_payment_method["success"] is False
    assert "payment_method must be non-empty" in blank_payment_method["error_message"]

    placeholder_address = server._call_tool({
        "session_id": sid,
        "name": "checkout",
        "arguments": {
            "shipping_address": "my home address",
            "payment_method": "Visa card",
        },
    })
    assert placeholder_address["success"] is False
    assert "unresolved placeholder" in placeholder_address["error_message"]

    placeholder_payment = server._call_tool({
        "session_id": sid,
        "name": "checkout",
        "arguments": {
            "shipping_address": "1 Audit Way",
            "payment_method": "card on file",
        },
    })
    assert placeholder_payment["success"] is False
    assert "unresolved placeholder" in placeholder_payment["error_message"]

    one_product = server._call_tool({
        "session_id": sid,
        "name": "compare_products",
        "arguments": {"product_ids": [product_ids[0]]},
    })
    assert one_product["success"] is False
    assert "at least two distinct" in one_product["error_message"]

    unknown_product = server._call_tool({
        "session_id": sid,
        "name": "compare_products",
        "arguments": {"product_ids": [product_ids[0], "prd_missing"]},
    })
    assert unknown_product["success"] is False
    assert "product not found" in unknown_product["error_message"]


def test_shopping_return_rejects_blank_reason_without_mutation() -> None:
    server = ShoppingServer()
    sid = _reset(server)
    shipped = next(
        order for order in server.sessions[sid]["orders"].values()
        if order["status"] == "shipped"
    )

    result = server._call_tool({
        "session_id": sid,
        "name": "return_order",
        "arguments": {
            "order_id": shipped["order_id"],
            "reason": " ",
        },
    })

    assert result["success"] is False
    assert "reason must be non-empty" in result["error_message"]
    assert shipped["status"] == "shipped"


def test_shopping_recommendation_rejects_unknown_explicit_seed() -> None:
    server = ShoppingServer()
    sid = _reset(server)

    result = server._call_tool({
        "session_id": sid,
        "name": "get_recommendations",
        "arguments": {"based_on_product": "prd_missing"},
    })

    assert result["success"] is False
    assert "product not found" in result["error_message"]


def test_shopping_recommendation_excludes_seed_product() -> None:
    server = ShoppingServer()
    sid = _reset(server)
    state = server.sessions[sid]
    seed_product = next(iter(state["products"].values()))

    result = server._call_tool({
        "session_id": sid,
        "name": "get_recommendations",
        "arguments": {
            "based_on_product": seed_product["product_id"],
            "limit": 20,
        },
    })

    assert result["success"] is True
    recommendations = result["observation"]["recommendations"]
    assert all(
        product["product_id"] != seed_product["product_id"]
        for product in recommendations
    )
    assert all(
        product["category"] == seed_product["category"]
        for product in recommendations
    )


def test_shopping_recommendation_preserves_explicit_category_with_seed() -> None:
    server = ShoppingServer()
    sid = _reset(server)
    state = server.sessions[sid]
    keyboard = next(
        product for product in state["products"].values()
        if product["category"] == "keyboard"
    )

    result = server._call_tool({
        "session_id": sid,
        "name": "get_recommendations",
        "arguments": {
            "based_on_product": keyboard["product_id"],
            "category": "mouse",
            "limit": 20,
        },
    })

    assert result["success"] is True
    recommendations = result["observation"]["recommendations"]
    assert recommendations
    assert all(
        product["category"] == "mouse"
        for product in recommendations
    )
    assert all(
        product["product_id"] != keyboard["product_id"]
        for product in recommendations
    )


def test_shopping_rejects_invalid_recommendation_limit_and_unknown_wishlist_id() -> None:
    server = ShoppingServer()
    sid = _reset(server)

    invalid_limit = server._call_tool({
        "session_id": sid,
        "name": "get_recommendations",
        "arguments": {"limit": -1},
    })
    assert invalid_limit["success"] is False
    assert "limit must be between 1 and 20" in invalid_limit["error_message"]

    unknown_product = server._call_tool({
        "session_id": sid,
        "name": "remove_from_wishlist",
        "arguments": {"product_id": "prd_missing"},
    })
    assert unknown_product["success"] is False
    assert "product not found" in unknown_product["error_message"]


def test_shopping_reviews_reject_unknown_explicit_sort_mode() -> None:
    server = ShoppingServer()
    sid = _reset(server)
    product_id = next(iter(server.sessions[sid]["products"]))

    result = server._call_tool({
        "session_id": sid,
        "name": "get_reviews",
        "arguments": {"product_id": product_id, "sort_by": "newest"},
    })

    assert result["success"] is False
    assert "unsupported review sort" in result["error_message"]
    assert result["state_changed"] is False


def test_shopping_rejects_reversed_price_range_and_blank_review_body() -> None:
    server = ShoppingServer()
    sid = _reset(server)
    product_id = next(iter(server.sessions[sid]["products"]))
    reviews_before = json.loads(json.dumps(server.sessions[sid]["reviews"]))

    reversed_range = server._call_tool({
        "session_id": sid,
        "name": "search_products",
        "arguments": {"min_price": 100, "max_price": 10},
    })
    assert reversed_range["success"] is False
    assert "min_price must be less than or equal" in reversed_range["error_message"]
    assert reversed_range["state_changed"] is False

    blank_review = server._call_tool({
        "session_id": sid,
        "name": "add_review",
        "arguments": {
            "product_id": product_id,
            "rating": 5,
            "body": " ",
        },
    })
    assert blank_review["success"] is False
    assert "review body must be non-empty" in blank_review["error_message"]
    assert blank_review["state_changed"] is False
    assert server.sessions[sid]["reviews"] == reviews_before


def test_shopping_dynamic_entity_ids_are_seed_scoped() -> None:
    server = ShoppingServer()
    dynamic_ids: list[set[str]] = []
    for seed in (41, 42):
        sid = _reset(server, seed=seed)
        state = server.sessions[sid]
        reviewed = set(state["reviews"])
        product_id = next(
            item["product_id"]
            for order in state["orders"].values()
            if order["status"] in {"shipped", "returning", "returned"}
            for item in order["items"]
            if item["product_id"] not in reviewed
        )
        shipped = next(
            order for order in state["orders"].values()
            if order["status"] == "shipped"
        )

        assert server._call_tool({
            "session_id": sid,
            "name": "add_to_cart",
            "arguments": {"product_id": product_id, "quantity": 1},
        })["success"]
        checkout = server._call_tool({
            "session_id": sid,
            "name": "checkout",
            "arguments": {
                "shipping_address": "1 Audit Way",
                    "payment_method": "Visa",
            },
        })
        returned = server._call_tool({
            "session_id": sid,
            "name": "return_order",
            "arguments": {
                "order_id": shipped["order_id"],
                "reason": "damaged",
            },
        })
        review = server._call_tool({
            "session_id": sid,
            "name": "add_review",
            "arguments": {
                "product_id": product_id,
                "rating": 5,
                "body": "Works well",
            },
        })
        assert checkout["success"] and returned["success"] and review["success"]
        ids = {
            checkout["observation"]["order"]["order_id"],
            returned["observation"]["return"]["return_id"],
            review["observation"]["review"]["review_id"],
        }
        assert all(f"_s{seed}_" in entity_id for entity_id in ids)
        dynamic_ids.append(ids)

    assert dynamic_ids[0].isdisjoint(dynamic_ids[1])


def test_cancel_order_rejects_shipped_status() -> None:
    server = ShoppingServer(); sid = _reset(server)
    state = server.sessions[sid]
    # Find the shipped order
    shipped = next(o for o in state["orders"].values() if o.get("status") == "shipped")

    result = server._call_tool({
        "session_id": sid,
        "name": "cancel_order",
        "arguments": {"order_id": shipped["order_id"]},
    })

    assert result["success"] is False
    assert "only placed or pending orders can be cancelled" in result["error_message"]


def test_cancel_order_succeeds_for_placed_order() -> None:
    server = ShoppingServer(); sid = _reset(server)
    state = server.sessions[sid]
    # Find the placed (or pending) order
    placed = next(
        o for o in state["orders"].values()
        if o.get("status") in ("placed", "pending")
    )
    old_stock = {
        item["product_id"]: state["products"][item["product_id"]]["stock"]
        for item in placed["items"]
        if item["product_id"] in state["products"]
    }

    result = server._call_tool({
        "session_id": sid,
        "name": "cancel_order",
        "arguments": {"order_id": placed["order_id"]},
    })

    assert result["success"] is True
    assert result["state_changed"] is True
    assert result["observation"]["order"]["status"] == "cancelled"
    # Stock should be returned
    for pid, old_qty in old_stock.items():
        item = next(i for i in placed["items"] if i["product_id"] == pid)
        assert state["products"][pid]["stock"] == old_qty + item["quantity"]


def test_shopping_discovery_summaries_leave_exact_stock_to_get_product() -> None:
    server = ShoppingServer()
    sid = _reset(server)
    state = server.sessions[sid]
    product = next(iter(state["products"].values()))

    searched = server._call_tool({
        "session_id": sid,
        "name": "search_products",
        "arguments": {"query": product["name"]},
    })["observation"]["products"]
    assert len(searched) == 1
    assert "stock" not in searched[0]
    assert searched[0]["product_id"] == product["product_id"]

    recommended = server._call_tool({
        "session_id": sid,
        "name": "get_recommendations",
        "arguments": {"limit": 1},
    })["observation"]["recommendations"]
    assert recommended
    assert "stock" not in recommended[0]

    wishlist = server._call_tool({
        "session_id": sid,
        "name": "get_wishlist",
        "arguments": {},
    })["observation"]["wishlist"]
    assert wishlist
    assert all("stock" not in item for item in wishlist)

    detailed = server._call_tool({
        "session_id": sid,
        "name": "get_product",
        "arguments": {"product_id": product["product_id"]},
    })["observation"]["product"]
    assert detailed["stock"] == product["stock"]


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


def test_oval_task_reward_does_not_match_future_round_call_early() -> None:
    reward = reward_module.TaskReward(reward_profile="oval_full")
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

    monkeypatch.setattr(reward_module.TaskReward, "compute", boom)
    reward_profile = reward_module.get_config().reward_profile
    with pytest.raises(reward_module.RewardIntegrityError, match="reward contract exploded"):
        reward_module.compute_score(
            "live",
            "",
            {"oracle_calls": json.dumps([{
                "action": "tool_call", "tool_name": "list_accounts", "arguments": {},
            }]), "success_criteria": "[]"},
            {
                "reward_profile": reward_profile,
                "prompt_profile": "local_trainable_v1",
                "semantic_gate_profile": "deterministic_v1",
                "artifact_purpose": "training_candidate",
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


def test_rollout_reward_profile_mismatch_fails_before_scoring() -> None:
    with pytest.raises(
        reward_module.RewardIntegrityError,
        match="rollout/reward profile mismatch",
    ):
        reward_module.compute_score(
            "live", "", {}, {"reward_profile": "not-the-loaded-profile"},
        )


def test_rollout_reward_profile_missing_fails_before_scoring() -> None:
    with pytest.raises(
        reward_module.RewardIntegrityError,
        match="missing reward_profile",
    ):
        reward_module.compute_score("live", "", {}, {})


def test_reward_rejects_paper_audit_artifact_before_scoring() -> None:
    reward_profile = reward_module.get_config().reward_profile
    with pytest.raises(
        reward_module.RewardIntegrityError,
        match="not training-consumable",
    ):
        reward_module.compute_score(
            "live",
            "",
            {},
            {
                "reward_profile": reward_profile,
                "prompt_profile": "paper_generation_baseline_v1",
                "semantic_gate_profile": "diagnostic_only",
                "artifact_purpose": "paper_audit",
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
            reward_module.EventLog(events=[]), {}, gamma=1.0,
            domain_adapter=None,
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


def test_rollout_chat_template_tokens_accept_transformers5_batch_encoding() -> None:
    assert _single_conversation_token_ids({
        "input_ids": [11, 12, 13],
        "attention_mask": [1, 1, 1],
    }) == [11, 12, 13]
    assert _single_conversation_token_ids({"input_ids": [[21, 22]]}) == [21, 22]


def test_rollout_chat_template_tokens_reject_ambiguous_batches() -> None:
    with pytest.raises(ValueError, match="exactly one conversation"):
        _single_conversation_token_ids({"input_ids": [[1], [2]]})


def test_rollout_infers_plain_final_only_after_round_contract_is_satisfied() -> None:
    assert _resolve_terminal_type(
        "The K3 Keyboard costs $82 and has RGB backlighting.",
        allowed_terminal_actions=["final_answer"],
        required_tools=["search_products", "get_product"],
        successful_tool_names=["search_products", "get_product"],
    ) == ("final_answer", True)

    # Content-based clarification classification (PROVE-aligned): the model
    # is asking the user for the product ID, so the terminal is
    # ask_clarification even without XML tags and even when the contract
    # only lists final_answer.
    assert _resolve_terminal_type(
        "Please provide the product ID.",
        allowed_terminal_actions=["final_answer"],
        required_tools=["search_products", "get_product"],
        successful_tool_names=[],
    ) == ("ask_clarification", False)

    # allow_plain_final=False only gates the *plain final* fallback;
    # content-marker classification still applies.
    assert _resolve_terminal_type(
        "The K3 Keyboard costs $82 and has RGB backlighting.",
        allowed_terminal_actions=["final_answer"],
        required_tools=["search_products", "get_product"],
        successful_tool_names=["search_products", "get_product"],
        allow_plain_final=False,
    ) == ("unknown", False)


def test_rollout_plain_terminal_fallback_preserves_nonfinal_contracts() -> None:
    # Content-based report_error classification: "I cannot complete" is a
    # refusal, resolved by content marker rather than XML tags.
    assert _resolve_terminal_type(
        "I cannot complete that request.",
        allowed_terminal_actions=["report_error"],
        required_tools=[],
        successful_tool_names=[],
    ) == ("report_error", False)
    # A plain-text request for clarification is NOT coerced into
    # final_answer — content markers preserve the non-final type.
    assert _resolve_terminal_type(
        "Could you clarify which order you mean?",
        allowed_terminal_actions=["final_answer"],
        required_tools=[],
        successful_tool_names=[],
    ) == ("ask_clarification", False)
    assert _resolve_terminal_type(
        "<ask_clarification>Which order?</ask_clarification>",
        allowed_terminal_actions=["ask_clarification"],
        required_tools=[],
        successful_tool_names=[],
    ) == ("ask_clarification", False)


def test_rollout_sampling_seed_is_stable_and_trajectory_specific() -> None:
    trajectory = {
        "step": 7,
        "sample_index": "task-123",
        "rollout_n": 0,
        "validate": False,
    }
    seed = _derive_sampling_seed(42, trajectory, 0)
    assert seed == _derive_sampling_seed(42, dict(trajectory), 0)
    assert 0 <= seed <= 0x7FFFFFFF

    variants = {
        _derive_sampling_seed(42, trajectory, 1),
        _derive_sampling_seed(43, trajectory, 0),
        _derive_sampling_seed(42, {**trajectory, "step": 8}, 0),
        _derive_sampling_seed(42, {**trajectory, "sample_index": "task-456"}, 0),
        _derive_sampling_seed(42, {**trajectory, "rollout_n": 1}, 0),
        _derive_sampling_seed(42, {**trajectory, "validate": True}, 0),
    }
    assert seed not in variants
    assert len(variants) == 6


def test_tool_call_parser_rejects_missing_outer_brace() -> None:
    malformed = (
        '{"name":"update_event","arguments":{"event_id":"evt_1",'
        '"fields":{"start_time":"15:00"}}'
    )
    assert _parse_tool_calls_json(malformed) == []
    assert _parse_tool_calls_json(f"{malformed}}}") == [{
        "name": "update_event",
        "arguments": {
            "event_id": "evt_1",
            "fields": {"start_time": "15:00"},
        },
    }]


def test_identical_invalid_action_counter_resets_on_action_change() -> None:
    assert _next_identical_action_count(None, "bad-a", 0) == 1
    assert _next_identical_action_count("bad-a", "bad-a", 1) == 2
    assert _next_identical_action_count("bad-a", "bad-b", 2) == 1


def test_unknown_model_tool_is_a_scoreable_invalid_call() -> None:
    event = _unknown_tool_audit_event(
        session_id="s",
        turn_idx=2,
        round_idx=1,
        tool_name="echo",
        tool_arguments={"text": "hello"},
    )
    parsed = reward_module._parse_audit_events([event])[0]
    assert parsed.tool_name == "echo"
    assert parsed.tool_name_known is False
    assert parsed.schema_valid is False
    assert parsed.execution_success is False
    assert event["error_type"] == "unknown_tool"


def test_mixed_model_action_is_scoreable_not_an_integrity_exception() -> None:
    event = _invalid_mixed_action_audit_event(
        session_id="s", turn_idx=1, round_idx=0,
    )
    parsed = reward_module._parse_audit_events([event])[0]

    assert parsed.action_type == "tool_call"
    assert parsed.tool_name_known is False
    assert parsed.schema_valid is False
    assert parsed.execution_success is False
    assert event["error_type"] == "invalid_mixed_action"


@pytest.mark.parametrize("domain", DOMAINS)
@pytest.mark.parametrize(
    "action_type", ["final_answer", "report_error", "ask_clarification"],
)
def test_every_domain_normalizes_terminal_without_tool_contract(
    domain: str,
    action_type: str,
) -> None:
    normalized = get_adapter(domain).normalize_event(
        action_type=action_type,
        tool_name="",
        tool_arguments={},
        observation="done",
        execution_success=True,
        state_changed=False,
        before_state=None,
        after_state=None,
    )
    assert normalized["operation"] == "terminal"
    assert normalized["target_id"] == ""


def test_task_reward_direct_sum_perfect_tool_trajectory_is_one_point_three() -> None:
    event_log = EventLog(events=[
        AuditEvent(
            event_id="tool", session_id="s", step=0, round_idx=0,
            action_type="tool_call", tool_name="list_accounts",
            tool_arguments={}, tool_name_known=True, schema_valid=True,
            execution_success=True,
        ),
        AuditEvent(
            event_id="terminal", session_id="s", step=1, round_idx=0,
            action_type="final_answer", terminal_action="done",
            execution_success=True, schema_valid=True,
        ),
    ])
    result = reward_module.TaskReward().compute(event_log, {
        "required_tool_calls": [{
            "tool_name": "list_accounts", "arguments": {},
        }],
        "required_call_rounds": [0],
        "allowed_terminal_actions": ["final_answer"],
    })
    assert result.r_validity == pytest.approx(1.0)
    assert result.r_coverage == pytest.approx(1.0)
    assert result.r_name == pytest.approx(1.0)
    assert result.r_arg == pytest.approx(1.0)
    assert result.r_efficiency == pytest.approx(0.0)
    assert result.r_task == pytest.approx(1.3)


def test_task_reward_direct_sum_applies_extra_valid_call_precision_penalty() -> None:
    events = [
        AuditEvent(
            event_id="required", session_id="s", step=0, round_idx=0,
            action_type="tool_call", tool_name="list_accounts",
            tool_arguments={}, tool_name_known=True, schema_valid=True,
            execution_success=True,
        ),
        AuditEvent(
            event_id="extra", session_id="s", step=1, round_idx=0,
            action_type="tool_call", tool_name="get_balance",
            tool_arguments={}, tool_name_known=True, schema_valid=True,
            execution_success=True,
        ),
        AuditEvent(
            event_id="terminal", session_id="s", step=2, round_idx=0,
            action_type="final_answer", terminal_action="done",
            execution_success=True, schema_valid=True,
        ),
    ]
    result = reward_module.TaskReward().compute(EventLog(events=events), {
        "required_tool_calls": [{
            "tool_name": "list_accounts", "arguments": {},
        }],
        "required_call_rounds": [0],
        "allowed_terminal_actions": ["final_answer"],
    })
    assert result.r_name == pytest.approx(0.5)
    assert result.r_task == pytest.approx(1.2)


def test_task_reward_keeps_no_tool_binary_definition() -> None:
    valid = reward_module.TaskReward().compute(EventLog(events=[AuditEvent(
        event_id="terminal", session_id="s", step=0, round_idx=0,
        action_type="report_error", terminal_action="unsupported",
        execution_success=True, schema_valid=True,
    )]), {
        "required_tool_calls": [],
        "allowed_terminal_actions": ["report_error"],
    })
    assert valid.r_task == pytest.approx(1.0)


def test_prove_no_tool_reward_ignores_terminal_type() -> None:
    result = reward_module.TaskReward(
        reward_profile="prove_baseline",
    ).compute(EventLog(events=[AuditEvent(
        event_id="terminal", session_id="s", step=0, round_idx=0,
        action_type="final_answer", terminal_action="done",
        execution_success=True, schema_valid=True,
    )]), {
        "required_tool_calls": [],
        "allowed_terminal_actions": ["ask_clarification"],
    })
    assert result.r_task == pytest.approx(1.0)


def test_oval_no_tool_reward_preserves_terminal_contract() -> None:
    result = reward_module.TaskReward(
        reward_profile="oval_full",
    ).compute(EventLog(events=[AuditEvent(
        event_id="terminal", session_id="s", step=0, round_idx=0,
        action_type="final_answer", terminal_action="done",
        execution_success=True, schema_valid=True,
    )]), {
        "required_tool_calls": [],
        "allowed_terminal_actions": ["ask_clarification"],
    })
    assert result.r_task == pytest.approx(0.0)


def test_prove_coverage_is_not_execution_gated() -> None:
    result = reward_module.TaskReward(
        reward_profile="prove_baseline",
    ).compute(EventLog(events=[AuditEvent(
        event_id="tool", session_id="s", step=0, round_idx=0,
        action_type="tool_call", tool_name="lookup",
        tool_arguments={"id": "x"}, tool_name_known=True,
        schema_valid=True, execution_success=False,
    )]), {
        "required_tool_calls": [{
            "tool_name": "lookup", "arguments": {"id": "x"},
        }],
        "required_call_rounds": [0],
        "dependency_edges": [],
    })
    assert result.r_validity == pytest.approx(2 / 3)
    assert result.r_coverage == pytest.approx(1.0)
    assert result.r_arg == pytest.approx(1.0)
    assert result.r_task == pytest.approx(17 / 15)


def test_prove_coverage_ignores_conversation_round() -> None:
    result = reward_module.TaskReward(
        reward_profile="prove_baseline",
    ).compute(EventLog(events=[AuditEvent(
        event_id="tool", session_id="s", step=0, round_idx=1,
        action_type="tool_call", tool_name="lookup",
        tool_arguments={"id": "x"}, tool_name_known=True,
        schema_valid=True, execution_success=True,
    )]), {
        "required_tool_calls": [{
            "tool_name": "lookup", "arguments": {"id": "x"},
        }],
        "required_call_rounds": [0],
        "dependency_edges": [],
    })
    assert result.r_coverage == pytest.approx(1.0)
    assert result.r_arg == pytest.approx(1.0)
    assert result.r_task == pytest.approx(1.3)


def test_prove_coverage_allows_reordering_without_dependencies() -> None:
    events = [
        AuditEvent(
            event_id="second", session_id="s", step=0, round_idx=1,
            action_type="tool_call", tool_name="second",
            tool_arguments={"value": 2}, tool_name_known=True,
            schema_valid=True, execution_success=True,
        ),
        AuditEvent(
            event_id="first", session_id="s", step=1, round_idx=1,
            action_type="tool_call", tool_name="first",
            tool_arguments={"value": 1}, tool_name_known=True,
            schema_valid=True, execution_success=True,
        ),
    ]
    result = reward_module.TaskReward(
        reward_profile="prove_baseline",
    ).compute(EventLog(events=events), {
        "required_tool_calls": [
            {"tool_name": "first", "arguments": {"value": 1}},
            {"tool_name": "second", "arguments": {"value": 2}},
        ],
        "required_call_rounds": [0, 0],
        "dependency_edges": [],
    })
    assert result.r_coverage == pytest.approx(1.0)
    assert result.r_arg == pytest.approx(1.0)
    assert result.r_task == pytest.approx(1.3)


def test_prove_coverage_still_enforces_dependency_order() -> None:
    events = [
        AuditEvent(
            event_id="dependent-too-early", session_id="s", step=0, round_idx=1,
            action_type="tool_call", tool_name="second",
            tool_arguments={"value": 2}, tool_name_known=True,
            schema_valid=True, execution_success=True,
        ),
        AuditEvent(
            event_id="predecessor", session_id="s", step=1, round_idx=1,
            action_type="tool_call", tool_name="first",
            tool_arguments={"value": 1}, tool_name_known=True,
            schema_valid=True, execution_success=True,
        ),
    ]
    result = reward_module.TaskReward(
        reward_profile="prove_baseline",
    ).compute(EventLog(events=events), {
        "required_tool_calls": [
            {"tool_name": "first", "arguments": {"value": 1}},
            {"tool_name": "second", "arguments": {"value": 2}},
        ],
        "required_call_rounds": [0, 0],
        "dependency_edges": [(0, 1)],
    })
    assert result.r_coverage == pytest.approx(0.5)
    assert result.r_arg == pytest.approx(1.0)
    assert result.aligned_calls == 1


def test_oval_coverage_remains_execution_and_round_gated() -> None:
    result = reward_module.TaskReward(
        reward_profile="oval_full",
    ).compute(EventLog(events=[AuditEvent(
        event_id="tool", session_id="s", step=0, round_idx=1,
        action_type="tool_call", tool_name="lookup",
        tool_arguments={"id": "x"}, tool_name_known=True,
        schema_valid=True, execution_success=True,
    )]), {
        "required_tool_calls": [{
            "tool_name": "lookup", "arguments": {"id": "x"},
        }],
        "required_call_rounds": [0],
        "dependency_edges": [],
    })
    assert result.r_coverage == pytest.approx(0.0)
    assert result.r_arg == pytest.approx(0.0)
    assert result.r_task == pytest.approx(0.7)


def test_task_reward_does_not_clip_excess_call_penalty() -> None:
    events = [
        AuditEvent(
            event_id=f"bad-{index}", session_id="s", step=index, round_idx=0,
            action_type="tool_call", tool_name="unknown",
            tool_arguments={}, tool_name_known=False, schema_valid=False,
            execution_success=False,
        )
        for index in range(100)
    ]
    events.append(AuditEvent(
        event_id="terminal", session_id="s", step=100, round_idx=0,
        action_type="report_error", terminal_action="failed",
        execution_success=True, schema_valid=True,
    ))
    result = reward_module.TaskReward().compute(EventLog(events=events), {
        "required_tool_calls": [{
            "tool_name": "list_accounts", "arguments": {},
        }],
        "required_call_rounds": [0],
        "allowed_terminal_actions": ["report_error"],
    })
    assert result.r_efficiency < 0
    assert result.r_task < -0.2


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


def test_audit_recreate_detection_can_extract_created_entity() -> None:
    adapter = get_adapter("payments")
    adapter.register_tool_schemas(PaymentsServer().tools)
    wrapper = AuditWrapper(None, None, adapter, "payments")
    wrapper._deleted_entities["s"] = {"wh_old": {"url": "https://old"}}
    call = ToolCall(
        name="create_webhook",
        arguments={"url": "https://new", "events": ["invoice.paid"]},
        call_id="c",
    )
    result = ToolExecutionResult(
        success=True,
        tool_name="create_webhook",
        canonical_tool_name="create_webhook",
        call_id="c",
        session_id="s",
        observation={
            "webhook": {
                "webhook_id": "wh_new",
                "url": "https://new",
                "events": ["invoice.paid"],
            }
        },
        error_type=None,
        error_message="",
        schema_valid=True,
        state_changed=True,
        latency_ms=0,
        execution_status="SUCCESS",
    )

    event = wrapper.audit_step_with_state(
        "s",
        "tool_call",
        [call],
        [result],
        pre_state={"payments": {"invoices": {}}},
        post_state={"payments": {"invoices": {}}},
    )

    assert event.target_id == "https://new"
