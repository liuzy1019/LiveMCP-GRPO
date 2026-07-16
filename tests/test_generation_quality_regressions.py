from __future__ import annotations

import datetime as _datetime
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from scripts.generate_data import (
    _assert_split_integrity,
    _candidate_shard_split,
    _domain_quotas,
    _build_round_contracts,
    _domain_recovery_requests,
    _filter_training_eligible_tasks,
    _is_zero_yield_error,
    _irrelevance_ratio_for_round,
    _zero_yield_is_recoverable,
    _load_generation_checkpoint,
    _minimum_action_budget,
    _required_round_oracle_calls,
    _serialize_training_oracle,
    _stratified_task_split,
    _tasks_to_rows,
    _validate_canonical_rows_replay,
    _validate_task_training_contract,
    _write_generation_checkpoint,
)
from scripts.merge_generation_shards import (
    _dedup_jaccard, _quality_issue, _row_fingerprint,
    _initial_query_key, _isolate_initial_queries, _suggest_topup_count,
    merge_shards,
)
from scripts.validate_generation_pipeline import _stage3_output_issue, _strict_cache_issue
from src.live_mcp.orchestrator import (
    ConversationFSM, FSMStateGroup, TaskOrchestrator,
    RobustnessPlan,
    _CREATED_ENTITY_BY_TOOL,
    _build_teacher_visible_tools,
    _chain_is_feasible,
    _chain_respects_state_preconditions,
    _detect_missing_dependency,
    _entity_record_satisfies_chain,
    _extract_chain_context,
    _extract_probe_entities,
    _format_entity_summary,
    _live_context_to_prompt_state,
    _missing_function_has_nonprefix_mutation,
    _missing_function_original_round_should_abort,
    _zero_tool_terminal_is_valid,
    _teacher_public_action_context,
    _tool_existing_entity_requirements,
    _classify_scenario,
    _compact_sampling_context,
)
from src.live_mcp.observation import (
    TRAJECTORY_SCHEMA_VERSION,
    OBSERVATION_SCHEMA_VERSION,
    OBSERVATION_PROJECTION_VERSION,
    compute_server_schema_hash,
)
from src.live_mcp.schema_registry import SchemaRegistry
from src.live_mcp.executor import _is_partial_observation
from src.live_mcp.llm_client import (
    LLMClient,
    _remaining_context_output_budget,
)
from src.live_mcp.environment_metadata import (
    compute_reward_fingerprint,
    compute_transition_fingerprint,
)
from src.live_mcp.servers.calendar.server import TOOLS as CALENDAR_TOOLS
from src.live_mcp.servers.banking.server import TOOLS as BANKING_TOOLS
from src.live_mcp.servers.crm.server import CRMServer, TOOLS as CRM_TOOLS
from src.live_mcp.servers.filesystem.server import TOOLS as FILESYSTEM_TOOLS
from src.live_mcp.servers.food_delivery.server import (
    FoodDeliveryServer, TOOLS as FOOD_DELIVERY_TOOLS,
)
from src.live_mcp.servers.issue_tracker.server import (
    IssueTrackerServer, TOOLS as ISSUE_TRACKER_TOOLS,
)
from src.live_mcp.servers.payments.server import TOOLS as PAYMENTS_TOOLS
from src.live_mcp.servers.shopping.server import TOOLS as SHOPPING_TOOLS
from src.live_mcp.servers.team_chat.server import TOOLS as TEAM_CHAT_TOOLS
from src.live_mcp.task_planner import (
    ActionPlan, ContinuationPolicy, TaskPlanner, _PERSONA_TEMPLATES,
    derive_success_criteria,
    provenance_check, replay_validate,
)
from src.live_mcp.task_planner import (
    _format_history, _format_state_compact, _format_tools,
    _target_tool_requirement,
)
from src.live_mcp.tool_semantics import is_mutating_tool
from src.live_mcp.observation import (
    DEFAULT_TEACHER_OBSERVATION_CHARS,
    project_observation,
)
from src.live_mcp.types import LiveTask, OracleCall, OracleProgram
from src.reward.oval_reward_fn import _build_task_dict


def test_multi_round_action_budget_covers_tools_and_terminals() -> None:
    oracle = [
        *[
            {"tool_name": f"tool_{i}", "arguments": {}, "action": "tool_call"}
            for i in range(10)
        ],
        {"tool_name": "final_answer", "arguments": {}, "action": "final_answer"},
    ]
    contracts = [
        {"round_idx": i, "required_tools": [], "allowed_terminal_actions": ["final_answer"]}
        for i in range(3)
    ]
    assert _minimum_action_budget(oracle, contracts) == 13


def test_calendar_seeded_events_share_the_generation_reference_date() -> None:
    from src.live_mcp.state_seeder import StateSeeder
    from src.live_mcp.task_planner import reference_datetime_for_seed

    seed = 60
    state = StateSeeder().seed_state("calendar", "temporal-contract", seed)
    reference_date = reference_datetime_for_seed(seed).date()
    event_dates = {
        _datetime.date.fromisoformat(event["start_time"][:10])
        for event in state["events"].values()
    }

    assert reference_date in event_dates
    assert all(abs((event_date - reference_date).days) <= 4 for event_date in event_dates)


def test_stage1_strict_cache_rejects_stale_tool_schema() -> None:
    names = ["a", "b"]
    ledger = [{
        "pair": ["a", "b"],
        "source": "a",
        "target": "b",
        "relation": "explicit",
    }]
    graph = TaskOrchestrator._graph_from_pair_classifications(ledger, names)
    payload = {
        "cache_version": TaskOrchestrator.DEPENDENCY_CACHE_VERSION,
        "dependency_semantics_version": TaskOrchestrator.DEPENDENCY_SEMANTICS_VERSION,
        "tool_names": names,
        "tool_count": 2,
        "pair_classifications": ledger,
        "expected_pair_count": 1,
        "classified_pair_count": 1,
        "classification_complete": True,
        "graph": graph,
        "classifier_contract_hash": "contract",
        "teacher_model_id": "teacher",
        "classifier_prompt_sha256": "prompt",
    }
    assert _strict_cache_issue("probe", payload, names) == ""
    issue = _strict_cache_issue("probe", payload, ["a", "b", "c"])
    assert "tool_names/tool_count" in issue

    tools = [
        {"name": name, "input_schema": {"required": []},
         "annotations": {"readonly": True, "mutating": False}}
        for name in names
    ]
    issue = _strict_cache_issue("probe", payload, names, tools)
    assert "1 条关系定义冲突" in issue


def test_explicit_fsm_preserves_partial_success_as_distinct_outcome() -> None:
    class _Teacher:
        calls = 0

        def decide_action(self, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                return ActionPlan(
                    action="tool_call", tool_name="list_items", arguments={}, text="",
                )
            return ActionPlan(
                action="final_answer", tool_name="", arguments={}, text="No items found.",
            )

    class _Executor:
        def execute(self, *_args, **_kwargs):
            return SimpleNamespace(
                success=True,
                observation=[],
                error_type="",
                error_message="",
                execution_status="PARTIAL_SUCCESS",
                state_changed=False,
                schema_valid=True,
            )

    orchestrator = TaskOrchestrator.__new__(TaskOrchestrator)
    orchestrator.executor = _Executor()
    fsm = ConversationFSM()
    oracle, history, *_ = orchestrator._run_turn_loop(
        teacher=_Teacher(),
        current_query="List the items.",
        server_tools=[{"name": "list_items"}],
        server_name="probe",
        session_id="session",
        difficulty="complete",
        round_idx=0,
        fsm=fsm,
    )

    assert history[0]["execution_status"] == "PARTIAL_SUCCESS"
    assert any(
        transition.get("outcome") == "PARTIAL_SUCCESS"
        and transition["to"] == FSMStateGroup.RESPONSE.value
        for transition in fsm.transitions
    )
    assert [call.action for call in oracle] == ["tool_call", "final_answer"]


def test_filesystem_find_empty_result_is_partial_success() -> None:
    assert _is_partial_observation(
        {"matches": [], "count": 0}, tool_name="find",
    )


def test_teacher_history_preserves_outcome_state_change_and_error() -> None:
    rendered = _format_history([
        {
            "tool_name": "list_items", "arguments": {},
            "observation": {"items": [], "count": 0},
            "success": True, "execution_status": "PARTIAL_SUCCESS",
            "state_changed": False, "schema_valid": True,
        },
        {
            "tool_name": "get_invoice", "arguments": {"invoice_id": "882"},
            "observation": None, "success": False,
            "execution_status": "FAILURE", "state_changed": False,
            "schema_valid": False,
            "error_type": "KeyError", "error_message": "invoice not found: 882",
        },
    ])
    assert "PARTIAL_SUCCESS; state_changed=False" in rendered
    assert "FAILURE; state_changed=False" in rendered
    assert "KeyError: invoice not found: 882" in rendered
    assert '"schema_valid": false' in rendered


def test_long_observation_is_structurally_compacted_not_prefix_sliced() -> None:
    observation = {
        "invoices": [
            {"invoice_id": f"inv_{index:03d}", "status": "open", "amount": index}
            for index in range(60)
        ],
        "count": 60,
    }
    rendered = project_observation(observation, max_chars=1200)
    parsed = json.loads(rendered)
    assert parsed["_compacted"] is True
    assert parsed["_original_chars"] > 1200
    assert parsed["_sha256"]
    facts = parsed["summary_facts"]
    assert any(item.get("invoice_id") == "inv_000" for item in facts)
    assert any(item.get("invoice_id") == "inv_059" for item in facts)
    assert any(item.get("status") == "open" for item in facts)


def test_observation_compactor_handles_wide_nested_entity() -> None:
    entity = {"email_id": "eml_1", "status": "sent"}
    entity.update({f"field_{index}": index for index in range(20)})
    rendered = project_observation(
        {"email": entity}, max_chars=DEFAULT_TEACHER_OBSERVATION_CHARS,
    )
    assert "eml_1" in rendered
    assert "_omitted_fields" in rendered


def test_schema_registry_recursively_rejects_unknown_crm_fields() -> None:
    registry = SchemaRegistry()
    registry.register_tools("crm", CRM_TOOLS)
    invalid = registry.validate_arguments(
        "update_contact",
        {"contact_id": "contact_1", "fields": {"role": "Architect"}},
        domain="crm",
    )
    assert not invalid.valid
    assert invalid.unexpected_keys == ["fields.role"]
    valid = registry.validate_arguments(
        "update_lead",
        {"lead_id": "lead_1", "fields": {"status": "qualified"}},
        domain="crm",
    )
    assert valid.valid


@pytest.mark.parametrize(
    ("domain", "tools", "tool_name", "arguments"),
    [
        ("calendar", CALENDAR_TOOLS, "set_reminder", {"event_id": "evt_1", "minutes_before": 0}),
        ("shopping", SHOPPING_TOOLS, "add_to_cart", {"product_id": "prod_1", "quantity": 0}),
        ("banking", BANKING_TOOLS, "wire_transfer", {"from_account": "acc_1", "routing_number": "1", "recipient_name": "A", "amount": -1}),
        ("payments", PAYMENTS_TOOLS, "refund_invoice", {"invoice_id": "inv_1", "amount": 0}),
        ("issue_tracker", ISSUE_TRACKER_TOOLS, "time_track", {"issue_id": "iss_1", "hours": -0.5}),
        ("filesystem", FILESYSTEM_TOOLS, "truncate", {"path": "/tmp/a", "size": -1}),
    ],
)
def test_public_schema_exposes_handler_numeric_preconditions(
    domain: str,
    tools: list[dict],
    tool_name: str,
    arguments: dict,
) -> None:
    registry = SchemaRegistry()
    registry.register_tools(domain, tools)
    result = registry.validate_arguments(tool_name, arguments, domain=domain)
    assert not result.valid
    assert result.type_errors


def test_issue_update_schema_rejects_workflow_state_noop() -> None:
    registry = SchemaRegistry()
    registry.register_tools("issue_tracker", ISSUE_TRACKER_TOOLS)
    invalid = registry.validate_arguments(
        "update_issue",
        {"issue_id": "iss_1", "fields": {"state": "in_progress"}},
        domain="issue_tracker",
    )
    assert not invalid.valid
    assert invalid.unexpected_keys == ["fields.state"]
    valid = registry.validate_arguments(
        "update_issue",
        {"issue_id": "iss_1", "fields": {"priority": "critical"}},
        domain="issue_tracker",
    )
    assert valid.valid


def test_irrelevance_path_does_not_hardcode_terminal_or_fallback_query() -> None:
    source = Path("src/live_mcp/orchestrator.py").read_text()
    block = source[source.index("def _generate_irrelevant_tasks("):]
    block = block[:block.index("def _generate_irrelevant_query(")]
    assert "self._run_turn_loop(" in block
    assert "_fallback_irrelevant_query" not in block
    assert 'arguments={"text": "No available tool can satisfy this request."}' not in block
    assert "conversation_queries=[query]" in block
    assert "oracle_calls_per_round=[list(oracle_calls)]" in block


def test_export_raises_row_budget_to_reproducible_minimum() -> None:
    round_0 = [
        OracleCall("get_event", {"event_id": "evt_1"}),
        OracleCall("final_answer", {"text": "found"}, action="final_answer"),
    ]
    round_1 = [
        OracleCall("update_event", {"event_id": "evt_1", "fields": {"title": "new"}}),
        OracleCall("final_answer", {"text": "updated"}, action="final_answer"),
    ]
    task = LiveTask(
        task_id="calendar_budget", source="teacher", suite_name="suite",
        user_prompt="find the event", session_id="s1", session_seed=1,
        target_servers=["calendar"],
        visible_tools=[{"name": "get_event"}, {"name": "update_event"}],
        required_tools=["get_event", "update_event"], expected_outcome={},
        success_criteria=[],
        oracle_program=OracleProgram(
            task_id="calendar_budget", calls=round_0 + round_1,
            success_criteria=[],
        ),
        sampling_context={
            "initial_action_context": {
                "entity_summaries": [
                    "  evt_1 (event): {'title': 'Planning'}",
                    "  evt_2 (event): {'title': 'Review'}",
                ]
            }
        }, max_turns=2,
        metadata={
            "scenario_type": "normal_safe_success",
            "generation_mode": "chain_seeded",
            "paper_replay_valid": True,
            "provenance_valid": True,
            "provenance_violation_count": 0,
            "teacher_attempt_count": 3,
            "teacher_failed_attempt_count": 1,
            "teacher_attempt_trace": [
                {
                    "round_idx": 0,
                    "call": {"tool_name": "get_event", "arguments": {}},
                    "observation": {},
                },
                {
                    "round_idx": 1,
                    "call": {"tool_name": "update_event", "arguments": {}},
                    "observation": {},
                },
                {
                    "round_idx": 1,
                    "call": {"tool_name": "update_event", "arguments": {}},
                    "observation": {},
                },
            ],
            "teacher_round_trace": [
                {
                    "round_idx": 0,
                    "user_query": "find the event",
                    "oracle_calls": [
                        {"tool_name": "get_event", "action": "tool_call"},
                        {"tool_name": "final_answer", "action": "final_answer"},
                    ],
                    "execution_history": [],
                },
                {
                    "round_idx": 1,
                    "user_query": "now update it",
                    "oracle_calls": [
                        {"tool_name": "update_event", "action": "tool_call"},
                        {"tool_name": "final_answer", "action": "final_answer"},
                    ],
                    "execution_history": [],
                },
            ],
            "replay_num_calls": 3,
            "replay_num_errors": 1,
        },
        conversation_queries=["find the event", "now update it"],
        oracle_calls_per_round=[round_0, round_1],
        execution_history_per_round=[[], []],
    )
    row = _tasks_to_rows([task], 1)[0]
    assert row["extra_info"]["minimum_action_budget"] == 4
    assert row["extra_info"]["budget"] == 4
    assert row["extra_info"]["teacher_attempt_count"] == 3
    assert row["extra_info"]["teacher_failed_attempt_count"] == 1
    assert row["extra_info"]["replay_num_calls"] == 3
    assert row["extra_info"]["replay_num_errors"] == 1
    system_prompt = json.loads(row["prompt"])[0]["content"]
    assert "appear in the user request or tool results" in system_prompt
    assert "## Current Grounded Entities (Observable Context)" in system_prompt
    assert "evt_1 (event): {'title': 'Planning'}" in system_prompt


def test_completed_dependency_chain_reaches_prove_continuation_decision() -> None:
    source = Path("src/live_mcp/orchestrator.py").read_text()
    continuation_block = source[source.index("# PROVE §3.2 continuation"):]
    continuation_block = continuation_block[:continuation_block.index("# If we broke out")]
    assert "sample_continuation_decision" in continuation_block
    assert "completed_chain_progress >= len(chain_seed)" not in continuation_block
    followup_block = source[source.index("current_query = teacher.generate_followup("):]
    followup_block = followup_block[:followup_block.index("conversation_queries.append")]
    assert "chain_seed=None" in followup_block


def test_continuation_policy_samples_each_round_with_boundary_masks() -> None:
    class FixedRng:
        def __init__(self, value: float):
            self.value = value

        def random(self) -> float:
            return self.value

    assert ContinuationPolicy.sample_continuation_decision(1, FixedRng(0.05)) == "clarification"
    assert ContinuationPolicy.sample_continuation_decision(1, FixedRng(0.50)) == "follow_up"
    assert ContinuationPolicy.sample_continuation_decision(2, FixedRng(0.20)) == "end"
    assert ContinuationPolicy.sample_continuation_decision(2, FixedRng(0.35)) == "clarification"
    assert ContinuationPolicy.sample_continuation_decision(2, FixedRng(0.50)) == "follow_up"
    assert ContinuationPolicy.sample_continuation_decision(3, FixedRng(0.50)) == "end"

    source = Path("src/live_mcp/orchestrator.py").read_text()
    end_branch = source[source.index('if decision == "end":'):]
    end_branch = end_branch[:end_branch.index("break")]
    assert '"continuation_selected"' in end_branch
    assert 'decision="end"' in end_branch


def test_dependency_chain_is_not_a_hidden_oracle_program() -> None:
    source = Path("src/live_mcp/task_planner.py").read_text()
    assert "## Oracle Synthesis Target" not in source
    assert "Do NOT final_answer until ALL remaining chain tools" not in source
    orchestrator_source = Path("src/live_mcp/orchestrator.py").read_text()
    assert "teacher_dep_hints" not in orchestrator_source
    assert 'arguments={"text": "Task completed."}' not in orchestrator_source


def test_query_prompt_requires_complete_chain_final_outcome() -> None:
    class CapturingClient:
        messages = None

        def generate_chat(self, messages, **kwargs):
            self.messages = messages
            return '{"user_query": "copy the file into the new complaints folder", "target_capability": "cp", "chain_supported": true}'

    client = CapturingClient()
    planner = TaskPlanner(client, "filesystem", seed=1)
    generated = planner.generate_query(
        tool_schemas=[
            {"name": "mkdir", "description": "create a directory", "inputSchema": {}},
            {"name": "cp", "description": "copy a file", "inputSchema": {}},
        ],
        grounded_state={},
        difficulty="complete",
        rng=__import__("random").Random(1),
        chain_seed=["mkdir", "cp"],
    )
    assert generated.user_query.startswith("copy the file")
    assert generated.target_capability == "cp"
    assert generated.chain_supported is True
    assert generated.attempts == 1
    prompt = client.messages[1]["content"]
    assert "['mkdir', 'cp']" in prompt
    assert "final outcome" in prompt
    assert "copy a file" in prompt


def test_teacher_stage_token_budget_is_forwarded_and_override_wins() -> None:
    class CapturingClient:
        def __init__(self):
            self.max_tokens = []

        def generate_chat(self, messages, **kwargs):
            self.max_tokens.append(kwargs.get("max_tokens"))
            return "{}"

    client = CapturingClient()
    planner = TaskPlanner(client, "calendar", seed=1)
    messages = [{"role": "user", "content": "probe"}]
    planner._generate_chat("action_decision", messages, temperature=0.0)
    planner._generate_chat(
        "action_decision", messages, temperature=0.0, max_tokens=77,
    )
    assert client.max_tokens == [384, 77]


def test_teacher_observation_budget_is_explicitly_configurable() -> None:
    planner = TaskPlanner(
        object(), "calendar", seed=1, max_observation_chars=4096,
    )
    assert planner.max_observation_chars == 4096


def test_query_generation_retries_mismatched_chain_target() -> None:
    class SequencedClient:
        def __init__(self):
            self.calls = 0

        def generate_chat(self, messages, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return '{"user_query": "dispute inv_1", "target_capability": "dispute_invoice", "chain_supported": true}'
            return '{"user_query": "create an invoice and show me its details", "target_capability": "get_invoice", "chain_supported": true}'

    client = SequencedClient()
    planner = TaskPlanner(client, "payments", seed=1)
    generated = planner.generate_query(
        tool_schemas=[
            {"name": "create_invoice", "description": "create invoice", "inputSchema": {}},
            {"name": "get_invoice", "description": "get invoice", "inputSchema": {}},
        ],
        grounded_state={},
        difficulty="complete",
        rng=__import__("random").Random(1),
        chain_seed=["create_invoice", "get_invoice"],
    )
    assert generated.target_capability == "get_invoice"
    assert generated.attempts == 2
    assert client.calls == 2


def test_query_generation_does_not_retry_explicit_unsat_chain() -> None:
    class UnsatClient:
        def __init__(self) -> None:
            self.calls = 0

        def generate_chat(self, messages, **kwargs):
            self.calls += 1
            return json.dumps({
                "user_query": "UNSAT",
                "target_capability": "refund_invoice",
                "chain_supported": False,
                "mutation_evidence": [],
            })

    client = UnsatClient()
    planner = TaskPlanner(client, "payments", seed=1)
    with pytest.raises(RuntimeError, match="Failed to generate query"):
        planner.generate_query(
            tool_schemas=PAYMENTS_TOOLS,
            grounded_state={},
            difficulty="complete",
            rng=__import__("random").Random(1),
            chain_seed=["create_invoice", "pay_invoice", "refund_invoice"],
        )

    assert client.calls == 1


def test_verify_account_is_a_valid_read_predecessor_for_apply_loan() -> None:
    calls = [
        OracleCall("verify_account", {"account_id": "acc_1", "owner_name": "me"}),
        OracleCall(
            "apply_loan",
            {"account_id": "acc_1", "amount": 5000, "term_months": 24},
        ),
    ]
    assert not _detect_missing_dependency(calls, "banking")


def test_find_is_a_valid_read_predecessor_for_remove_file() -> None:
    calls = [
        OracleCall("find", {"path": "/home/user", "pattern": "pipeline.sh"}),
        OracleCall("rm", {"path": "/home/user/pipeline.sh", "recursive": False}),
    ]
    assert not _detect_missing_dependency(calls, "filesystem")


def test_readonly_schema_overrides_name_based_mutation_heuristic() -> None:
    sed_schema = next(tool for tool in FILESYSTEM_TOOLS if tool["name"] == "sed")
    rendered = _format_tools([sed_schema])
    requirement = _target_tool_requirement([sed_schema], "sed")

    assert not is_mutating_tool("sed", "filesystem")
    assert "Read-only: this tool does not modify server state." in rendered
    assert "read-only and does not modify server state" in requirement


def test_team_chat_thread_reaction_query_receives_empty_thread_fact() -> None:
    class CapturingClient:
        model_name = "test"

        def __init__(self) -> None:
            self.user_prompt = ""

        def generate_chat(self, messages, **kwargs):
            self.user_prompt = messages[-1]["content"]
            return json.dumps({
                "user_query": "Create a thread on msg_1 and react to the root message.",
                "target_capability": "react_message",
                "chain_supported": True,
                "mutation_evidence": [
                    {"capability": "create_thread", "query_span": "Create a thread"},
                    {"capability": "react_message", "query_span": "react to the root message"},
                ],
            })

    client = CapturingClient()
    planner = TaskPlanner(client, "team_chat", seed=1)
    planner.generate_query(
        tool_schemas=TEAM_CHAT_TOOLS,
        grounded_state={},
        difficulty="complete",
        rng=__import__("random").Random(1),
        chain_seed=["create_thread", "get_thread", "react_message"],
    )

    assert "newly created thread starts with no replies" in client.user_prompt
    assert "target the existing root message" in client.user_prompt


def test_query_generation_does_not_require_internal_mutation_evidence() -> None:
    class Client:
        def generate_chat(self, messages, **kwargs):
            return json.dumps({
                "user_query": "buy prd_1 now",
                "target_capability": "checkout",
                "chain_supported": True,
                "mutation_evidence": [
                    {"capability": "checkout", "query_span": "buy prd_1 now"},
                ],
            })

    planner = TaskPlanner(Client(), "shopping", seed=1)
    generated = planner.generate_query(
        tool_schemas=SHOPPING_TOOLS,
        grounded_state={},
        difficulty="complete",
        rng=__import__("random").Random(1),
        chain_seed=["add_to_cart", "checkout"],
    )

    assert generated.user_query == "buy prd_1 now"
    assert generated.attempts == 1


def test_query_generation_does_not_require_explicit_cd_authorization() -> None:
    class Client:
        def generate_chat(self, messages, **kwargs):
            return json.dumps({
                "user_query": "Split /home/user/notes.txt into 10-line chunks.",
                "target_capability": "split",
                "chain_supported": True,
                "mutation_evidence": [{
                    "capability": "split",
                    "query_span": "Split /home/user/notes.txt into 10-line chunks",
                }],
            })

    planner = TaskPlanner(Client(), "filesystem", seed=1)
    generated = planner.generate_query(
        tool_schemas=FILESYSTEM_TOOLS,
        grounded_state={},
        difficulty="complete",
        rng=__import__("random").Random(1),
        chain_seed=["cd", "split"],
    )

    assert generated.attempts == 1
    assert generated.target_capability == "split"


def test_query_generation_requires_cd_authorization_when_cd_is_final_target() -> None:
    class Client:
        def __init__(self) -> None:
            self.calls = 0

        def generate_chat(self, messages, **kwargs):
            self.calls += 1
            evidence = [] if self.calls == 1 else [{
                "capability": "cd",
                "query_span": "Change to /home/user/projects",
            }]
            return json.dumps({
                "user_query": "Change to /home/user/projects.",
                "target_capability": "cd",
                "chain_supported": True,
                "mutation_evidence": evidence,
            })

    planner = TaskPlanner(Client(), "filesystem", seed=1)
    generated = planner.generate_query(
        tool_schemas=FILESYSTEM_TOOLS,
        grounded_state={},
        difficulty="complete",
        rng=__import__("random").Random(1),
        chain_seed=["pwd", "cd"],
    )

    assert generated.attempts == 2
    assert generated.target_capability == "cd"


def test_later_write_with_direct_entity_selectors_is_not_missing_dependency() -> None:
    calls = [
        OracleCall("list_channels", {"archived": False}),
        OracleCall(
            "create_thread",
            {"message_id": "msg_730_0003", "channel_id": "ch_730_002"},
        ),
        OracleCall(
            "send_message",
            {
                "channel_id": "ch_730_002",
                "thread_id": "thd_0001",
                "content": "sounds good",
            },
        ),
    ]
    assert not _detect_missing_dependency(calls, "team_chat")


def test_non_id_member_selector_aliases_bind_user_dependencies() -> None:
    issue_calls = [
        OracleCall(
            "assign_issue",
            {"issue_id": "iss_9ac_0001", "assignee": "usr_9ac_001"},
        ),
    ]
    chat_calls = [
        OracleCall("search_messages", {"query": "mockups"}),
        OracleCall("send_dm", {"recipient": "dana", "content": "reviewing"}),
    ]
    assert not _detect_missing_dependency(issue_calls, "issue_tracker")
    assert not _detect_missing_dependency(chat_calls, "team_chat")


def test_outcome_criteria_are_diagnostic_not_a_prove_filter() -> None:
    task = SimpleNamespace(
        task_id="shopping_bad_outcome",
        metadata={
            "project_outcome_valid": False,
            "paper_replay_valid": True,
            "provenance_valid": True,
        },
        oracle_program=SimpleNamespace(
            calls=[
                OracleCall("report_error", {"text": "unavailable"}, action="report_error"),
            ],
            success_criteria=[],
        ),
        scenario_type="irrelevant",
        task_type="irrelevant",
        conversation_queries=["do something unsupported"],
        oracle_calls_per_round=[[
            OracleCall("report_error", {"text": "unavailable"}, action="report_error"),
        ]],
        user_prompt="do something unsupported",
        hidden_tools=[],
        visible_tools=[],
    )
    assert _filter_training_eligible_tasks([task]) == [task]


def test_paper_replay_invalid_task_is_removed_before_split() -> None:
    task = SimpleNamespace(
        task_id="replay-invalid",
        metadata={"paper_replay_valid": False},
    )
    assert _filter_training_eligible_tasks([task]) == []


def test_missing_function_allows_visible_prefix_calls_before_clarification() -> None:
    prefix = OracleCall("get_event", {"event_id": "evt_1"})
    terminal = OracleCall(
        "ask_clarification", {"question": "The update tool is unavailable."},
        action="ask_clarification",
    )
    task = SimpleNamespace(
        task_id="missing-prefix",
        metadata={
            "scenario_type": "clarification_required",
            "has_missing_function": True,
            "hidden_tool": "update_event",
            "chain_seed": [],
        },
        oracle_program=SimpleNamespace(
            calls=[prefix, terminal], success_criteria=[],
        ),
        scenario_type="clarification_required",
        task_type="missing_function",
        conversation_queries=["move evt_1 to tomorrow"],
        oracle_calls_per_round=[[prefix, terminal]],
        user_prompt="move evt_1 to tomorrow",
        hidden_tools=["update_event"],
        visible_tools=[{"name": "get_event"}],
    )

    _validate_task_training_contract(task)


def test_missing_function_rejects_unfinished_mutating_workaround_only() -> None:
    prefix = OracleCall("create_recurring", {"title": "Weekly sync"})
    read = OracleCall("list_events", {})
    workaround = OracleCall("create_event", {"title": "One-off sync"})
    cleanup = OracleCall("delete_event", {"event_id": "evt_series"})
    terminal = OracleCall(
        "report_error", {"reason": "Cannot edit one occurrence"},
        action="report_error",
    )

    assert not _missing_function_has_nonprefix_mutation(
        True,
        [prefix, read, terminal],
        ["create_recurring", "update_event"],
        "update_event",
        CALENDAR_TOOLS,
    )
    assert _missing_function_has_nonprefix_mutation(
        True,
        [prefix, read, workaround, cleanup, terminal],
        ["create_recurring", "update_event"],
        "update_event",
        CALENDAR_TOOLS,
    )
    assert not _missing_function_has_nonprefix_mutation(
        False,
        [workaround, terminal],
        ["create_recurring", "update_event"],
        "update_event",
        CALENDAR_TOOLS,
    )


def test_missing_function_teacher_can_call_visible_prefix_tool() -> None:
    class PrefixClient:
        def generate_chat(self, messages, **kwargs):
            return json.dumps({
                "action": "tool_call",
                "tool_name": "get_event",
                "arguments": {"event_id": "evt_1"},
            })

    planner = TaskPlanner(PrefixClient(), "calendar", seed=1)
    action = planner.decide_action(
        tool_schemas=[{
            "name": "get_event",
            "description": "Get an event.",
            "input_schema": {
                "type": "object",
                "properties": {"event_id": {"type": "string"}},
                "required": ["event_id"],
            },
        }],
        user_query="move evt_1 to tomorrow",
        execution_history=[],
        missing_function=True,
        blocked_tools={"update_event"},
    )
    assert action.action == "tool_call"
    assert action.tool_name == "get_event"


def test_missing_function_cannot_be_satisfied_by_later_unrelated_round() -> None:
    completed = [
        OracleCall("create_deal", {"stage": "negotiation"}),
        OracleCall("final_answer", {"text": "done"}, action="final_answer"),
    ]
    abstained = [
        OracleCall("get_deal", {"deal_id": "deal_1"}),
        OracleCall("report_error", {"text": "capability missing"}, action="report_error"),
    ]
    assert _missing_function_original_round_should_abort(True, 0, completed)
    assert not _missing_function_original_round_should_abort(True, 0, abstained)
    assert not _missing_function_original_round_should_abort(True, 1, completed)
    assert not _missing_function_original_round_should_abort(False, 0, completed)


def test_round_trace_keeps_true_state_separate_from_teacher_public_context() -> None:
    class Teacher:
        trace_includes_state = True

        def __init__(self):
            self.events = []

        def record_environment_event(self, stage, **payload):
            self.events.append((stage, payload))

    class Manager:
        def get_state(self, session_id, server_name):
            assert session_id == "session-1"
            assert server_name == "crm"
            return {"crm": {"contacts": {"contact_1": {"name": "Alice"}}}}

    teacher = Teacher()
    TaskOrchestrator._record_round_trace(
        SimpleNamespace(manager=Manager()),
        teacher=teacher,
        session_id="session-1",
        server_name="crm",
        round_idx=0,
        phase="input",
        current_query="update the deal",
        visible_tools=[{"name": "update_deal"}],
        public_context={"entity_summaries": ["<hidden-contact-id>"]},
    )
    boundary = next(payload for stage, payload in teacher.events if stage == "round_boundary")
    snapshot = next(payload for stage, payload in teacher.events if stage == "entity_state_snapshot")
    assert boundary["public_context"] == {
        "entity_summaries": ["<hidden-contact-id>"],
    }
    assert "state" not in boundary
    assert snapshot["state"]["contacts"]["contact_1"]["name"] == "Alice"
    assert snapshot["state_hash"]


def test_continuation_action_context_keeps_all_refreshed_entities() -> None:
    live_context = {
        "source": "live_readonly_probe",
        "entity_ids": [
            {"id": "ord_3e3_0001", "type": "order"},
            {"id": "ord_3e3_0002", "type": "order"},
        ],
        "entity_summaries": [
            "ord_3e3_0001 (order): status=delivered",
            "ord_3e3_0002 (order): status=preparing",
        ],
        "entity_records": [],
        "entity_types": ["order"],
        "probe_results": [],
    }
    context = _compact_sampling_context(live_context)
    assert {item["id"] for item in context["entity_ids"]} == {
        "ord_3e3_0001", "ord_3e3_0002",
    }
    assert "status=delivered" in context["entity_summaries"][0]


def test_action_prompt_uses_live_context_without_claiming_it_is_exhaustive() -> None:
    class CapturingClient:
        messages = None

        def generate_chat(self, messages, **kwargs):
            self.messages = messages
            return json.dumps({
                "action": "ask_clarification",
                "question": "What amount should I use?",
            })

    client = CapturingClient()
    planner = TaskPlanner(client, "crm", seed=1)
    action = planner.decide_action(
        tool_schemas=[{
            "name": "create_deal",
            "description": "Create a deal.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "amount": {"type": "number"},
                },
                "required": ["name", "amount"],
            },
            "annotations": {"mutating": True},
        }],
        user_query="convert lead_1 to a deal",
        execution_history=[],
        difficulty="minimal",
        chain_context={
            "entity_summaries": ["lead_1 (lead): status=qualified"],
        },
    )
    prompt = client.messages[1]["content"]
    assert action.action == "ask_clarification"
    assert "Current Grounded Entities" in prompt
    assert "ONLY these IDs exist" not in prompt
    assert "Never invent a user-decided required value" in client.messages[0]["content"]


def test_query_prompt_keeps_duplicate_selector_beyond_first_fifteen() -> None:
    class CapturingClient:
        messages = None
        def generate_chat(self, messages, **kwargs):
            self.messages = messages
            return json.dumps({
                "user_query": "move the Q2 Review from alice@example.com",
                "target_capability": "move_to_thread",
                "chain_supported": True,
            })

    client = CapturingClient()
    planner = TaskPlanner(client, "email", seed=1)
    summaries = [f"grounded email candidate: subject {i}" for i in range(16)]
    summaries.append("grounded email candidate: Q2 Review from alice@example.com")
    planner.generate_query(
        tool_schemas=[
            {"name": "list_inbox", "description": "List email.", "input_schema": {}},
            {"name": "move_to_thread", "description": "Move email.", "input_schema": {}},
        ],
        grounded_state={}, difficulty="complete",
        rng=__import__("random").Random(1),
        chain_seed=["list_inbox", "move_to_thread"],
        chain_context={
            "query_grounding_summaries": summaries,
            "opaque_id_hidden_types": ["email"],
        },
    )
    assert "Q2 Review from alice@example.com" in client.messages[1]["content"]


def test_minimal_followup_cannot_invent_existing_entity_selector() -> None:
    class CapturingClient:
        messages = None

        def generate_chat(self, messages, **kwargs):
            self.messages = messages
            return json.dumps({"user_query": "message alice with the update"})

    client = CapturingClient()
    planner = TaskPlanner(client, "team_chat", seed=1)
    query = planner.generate_followup(
        tool_schemas=[{
            "name": "send_dm",
            "description": "Send a DM to an existing member.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "recipient": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["recipient", "content"],
            },
        }],
        grounded_state={
            "live_probe_users": {
                "alice": {"id": "alice", "type": "user", "status": "online"},
            },
        },
        previous_query="check who is online",
        difficulty="minimal",
        rng=__import__("random").Random(1),
    )
    assert query == "message alice with the update"
    system_prompt = client.messages[0]["content"]
    user_prompt = client.messages[1]["content"]
    assert "Across every difficulty level" in system_prompt
    assert "Staying in the same domain is NOT enough" in system_prompt
    assert "finding shell scripts is unrelated and forbidden" in system_prompt
    assert "missing or minimal difficulty" in user_prompt.lower()
    assert "MUST be copied exactly from Current State" in user_prompt


def test_clarification_continuation_stays_within_visible_domain_capabilities() -> None:
    class CapturingClient:
        messages = None

        def generate_chat(self, messages, **kwargs):
            self.messages = messages
            return json.dumps({"user_query": "Which deal should I update?"})

    client = CapturingClient()
    planner = TaskPlanner(client, "crm", seed=1)
    query = planner.generate_clarification(
        tool_schemas=[{
            "name": "update_deal",
            "description": "Update an existing CRM deal.",
            "input_schema": {
                "type": "object",
                "properties": {"deal_id": {"type": "string"}},
                "required": ["deal_id"],
            },
        }],
        grounded_state={},
        previous_query="Update the deal",
        difficulty="missing",
        rng=__import__("random").Random(1),
        previous_response="Which deal do you mean?",
    )
    assert query == "Which deal should I update?"
    system_prompt = client.messages[0]["content"]
    user_prompt = client.messages[1]["content"]
    assert "Do not introduce a new goal" in system_prompt
    assert "Sales CRM" in user_prompt
    assert "Available Tools and Preconditions" in user_prompt
    assert "update_deal" in user_prompt
    assert "unavailable" in user_prompt
    assert "cross-domain capability" in user_prompt


def test_new_followup_round_does_not_default_to_final_answer_from_prior_history() -> None:
    class CapturingClient:
        messages = None

        def generate_chat(self, messages, **kwargs):
            self.messages = messages
            return json.dumps({
                "action": "tool_call",
                "tool_name": "cancel_order",
                "arguments": {"order_id": "ord_1"},
            })

    client = CapturingClient()
    planner = TaskPlanner(client, "food_delivery", seed=1)
    action = planner.decide_action(
        tool_schemas=[{
            "name": "cancel_order",
            "description": "Cancel an order.",
            "input_schema": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"],
            },
        }],
        user_query="cancel ord_1",
        execution_history=[{
            "tool_name": "add_tip", "arguments": {}, "observation": {},
            "success": True,
        }],
        attempt=0,
        difficulty="complete",
    )
    assert action.action == "tool_call"
    assert "MUST call a tool" in client.messages[1]["content"]


def test_empty_terminal_is_retried() -> None:
    class SequencedClient:
        calls = 0

        def generate_chat(self, messages, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return json.dumps({"action": "final_answer", "text": ""})
            return json.dumps({
                "action": "final_answer", "text": "Payment is already linked."
            })

    client = SequencedClient()
    planner = TaskPlanner(client, "food_delivery", seed=1)
    action = planner.decide_action(
        tool_schemas=[],
        user_query="Do I need to provide payment information?",
        execution_history=[{
            "tool_name": "add_tip", "arguments": {}, "observation": {},
            "success": True,
        }],
        attempt=0,
        difficulty="complete",
        allow_direct_answer=True,
    )
    assert client.calls == 2
    assert action.action == "final_answer"
    assert action.text == "Payment is already linked."


def test_final_answer_question_is_retried_as_clarification() -> None:
    class SequencedClient:
        calls = 0

        def generate_chat(self, messages, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return json.dumps({
                    "action": "final_answer",
                    "text": "Which account should I use?",
                })
            return json.dumps({
                "action": "ask_clarification",
                "question": "Which account should I use?",
            })

    client = SequencedClient()
    planner = TaskPlanner(client, "banking", seed=1)
    action = planner.decide_action(
        tool_schemas=[],
        user_query="move the money",
        execution_history=[],
        attempt=0,
        difficulty="minimal",
        allow_direct_answer=True,
    )
    assert client.calls == 2
    assert action.action == "ask_clarification"
    assert action.text == "Which account should I use?"


def test_followup_prompt_contains_previous_visible_response() -> None:
    class CapturingClient:
        messages = None

        def generate_chat(self, messages, **kwargs):
            self.messages = messages
            return json.dumps({"user_query": "Cancel the new order instead."})

    client = CapturingClient()
    planner = TaskPlanner(client, "food_delivery", seed=1)
    planner.generate_followup(
        tool_schemas=[{
            "name": "cancel_order",
            "description": "Cancel only a placed or confirmed order.",
            "input_schema": {"type": "object", "properties": {}, "required": []},
        }], grounded_state={}, previous_query="Reorder ord_1",
        difficulty="complete", rng=__import__("random").Random(1),
        previous_response="Your new order is ord_3.",
        conversation_context=[
            {
                "round_idx": 0,
                "user_query": "Find the order from Sushi Express.",
                "assistant_response": "The order is ord_1.",
                "terminal_action": "final_answer",
            },
            {
                "round_idx": 1,
                "user_query": "Reorder ord_1.",
                "assistant_response": "Your new order is ord_3.",
                "terminal_action": "final_answer",
            },
        ],
    )
    prompt = client.messages[1]["content"]
    assert "Assistant reply you just received" in prompt
    assert "Your new order is ord_3." in prompt
    assert "All Completed Conversation Rounds" in prompt
    assert "Find the order from Sushi Express." in prompt
    assert "The order is ord_1." in prompt
    assert "Available Tools and Preconditions" in prompt
    assert "Cancel only a placed or confirmed order." in prompt


def test_payments_tool_descriptions_expose_handler_preconditions() -> None:
    descriptions = {tool["name"]: tool["description"] for tool in PAYMENTS_TOOLS}
    assert "exactly its full invoice amount" in descriptions["pay_invoice"]
    assert "remaining refundable amount" in descriptions["refund_invoice"]
    assert "payment_id" in descriptions["cancel_payment"]
    assert "paid or pending" in descriptions["dispute_invoice"]


def test_empty_oracle_round_is_rejected_before_export() -> None:
    task = SimpleNamespace(
        task_id="empty-round",
        oracle_calls_per_round=[
            [OracleCall("get_email", {"email_id": "eml_1"})],
            [],
        ],
    )
    with __import__("pytest").raises(ValueError, match="oracle round 1 is empty"):
        _build_round_contracts(task)


def test_turn_loop_rejects_no_teacher_action() -> None:
    class EmptyTeacher:
        def decide_action(self, **kwargs):
            raise RuntimeError("invalid model output")

    orchestrator = TaskOrchestrator.__new__(TaskOrchestrator)
    with __import__("pytest").raises(RuntimeError, match="produced no action"):
        orchestrator._run_turn_loop(
            teacher=EmptyTeacher(), current_query="label the email",
            server_tools=[], server_name="email", session_id="s",
            difficulty="complete", round_idx=1,
        )


def test_turn_loop_executes_repeated_teacher_calls_against_live_mcp() -> None:
    """PROVE keeps trace actions intact; dedup happens after conversation completion."""

    class SequencedTeacher:
        def __init__(self):
            self.index = 0
            self.histories = []

        def decide_action(self, **kwargs):
            self.histories.append([
                dict(item) for item in kwargs.get("execution_history", [])
            ])
            actions = [
                ActionPlan("tool_call", "list_events", {"date": "today"}),
                ActionPlan("tool_call", "list_events", {"date": "today"}),
                ActionPlan("final_answer", text="One event found."),
            ]
            action = actions[self.index]
            self.index += 1
            return action

    class FakeExecutor:
        calls = 0

        def execute(self, *args, **kwargs):
            self.calls += 1
            return SimpleNamespace(
                success=True, observation={"events": ["evt_1"]},
                execution_status="SUCCESS", error_type="", error_message="",
                state_changed=False, schema_valid=True,
            )

    orchestrator = TaskOrchestrator.__new__(TaskOrchestrator)
    orchestrator.executor = FakeExecutor()
    teacher = SequencedTeacher()
    calls, history, *_ = orchestrator._run_turn_loop(
        teacher=teacher, current_query="what is today",
        server_tools=[{"name": "list_events"}],
        server_name="calendar", session_id="s", difficulty="complete",
        round_idx=0,
    )
    assert orchestrator.executor.calls == 2
    assert [call.tool_name for call in calls] == [
        "list_events", "list_events", "final_answer",
    ]
    assert [item["execution_status"] for item in history] == ["SUCCESS", "SUCCESS"]
    assert "no_progress_warning" in teacher.histories[2][-1]


def test_no_progress_repeat_stays_in_trace_but_not_required_rl_oracle() -> None:
    call = OracleCall("list_events", {"date": "today"})
    terminal = OracleCall(
        "final_answer", {"text": "One event found."}, action="final_answer",
    )
    task = SimpleNamespace(
        task_id="repeat-label",
        task_type="",
        metadata={"scenario_type": "normal_safe_success"},
        oracle_program=OracleProgram(
            task_id="repeat-label",
            calls=[call, call, terminal],
            success_criteria=[],
        ),
        oracle_calls_per_round=[[call, call, terminal]],
        execution_history_per_round=[[
            {
                "tool_name": "list_events", "arguments": {"date": "today"},
                "success": True, "state_changed": False,
                "observation": {"events": ["evt_1"]},
            },
            {
                "tool_name": "list_events", "arguments": {"date": "today"},
                "success": True, "state_changed": False,
                "observation": {"events": ["evt_1"]},
                "no_progress_warning": "exact repeat",
            },
        ]],
    )

    assert len(task.oracle_program.calls) == 3
    serialized = _serialize_training_oracle(task)
    assert [item["tool_name"] for item in serialized] == [
        "list_events", "final_answer",
    ]
    assert _build_round_contracts(task)[0]["required_tools"] == ["list_events"]


def test_no_progress_required_oracle_survives_parquet_reward_roundtrip(
    tmp_path: Path,
) -> None:
    call = OracleCall("list_events", {"date": "today"})
    terminal = OracleCall(
        "final_answer", {"text": "One event found."}, action="final_answer",
    )
    visible_tools = [{
        "name": "list_events", "description": "List events.",
        "input_schema": {"type": "object", "properties": {}},
        "annotations": {"readonly": True, "mutating": False},
    }]
    task = LiveTask(
        task_id="repeat-roundtrip", source="teacher", suite_name="suite",
        user_prompt="what is on today", session_id="s-repeat", session_seed=1,
        target_servers=["calendar"],
        visible_tools=visible_tools,
        required_tools=["list_events"], expected_outcome={},
        success_criteria=[],
        oracle_program=OracleProgram(
            task_id="repeat-roundtrip", calls=[call, call, terminal],
            success_criteria=[],
        ),
        sampling_context={}, max_turns=3,
        metadata={
            "scenario_type": "normal_safe_success",
            "generation_mode": "chain_seeded",
            "paper_replay_valid": True,
            "provenance_valid": True,
            "provenance_violation_count": 0,
            "server_schema_hash": compute_server_schema_hash(visible_tools),
            "server_schema_hashes": {
                "calendar": compute_server_schema_hash(visible_tools),
            },
            "transition_fingerprints": {
                "calendar": compute_transition_fingerprint(
                    "calendar", visible_tools,
                ),
            },
            "initial_state_hash": "test-initial-state",
            "initial_state_hashes": {"calendar": "test-initial-state"},
            "reward_fingerprint": compute_reward_fingerprint(),
            "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
            "observation_projection_version": OBSERVATION_PROJECTION_VERSION,
            "trajectory_schema_version": TRAJECTORY_SCHEMA_VERSION,
            "max_observation_chars": 4096,
            "reward_profile_compatibility": ["prove_baseline", "oval_full"],
        },
        conversation_queries=["what is on today"],
        oracle_calls_per_round=[[call, call, terminal]],
        execution_history_per_round=[[
            {
                "tool_name": "list_events", "arguments": {"date": "today"},
                "success": True, "state_changed": False,
                "observation": {"events": ["evt_1"]},
            },
            {
                "tool_name": "list_events", "arguments": {"date": "today"},
                "success": True, "state_changed": False,
                "observation": {"events": ["evt_1"]},
                "no_progress_warning": "exact repeat",
            },
        ]],
    )
    row = _tasks_to_rows([task], 1)[0]
    path = tmp_path / "repeat.parquet"
    pd.DataFrame([row]).to_parquet(path, index=False)
    read_row = pd.read_parquet(path).iloc[0].to_dict()
    reward_task = _build_task_dict(read_row["extra_info"])

    assert reward_task["required_tool_calls"] == [{
        "tool_name": "list_events", "arguments": {"date": "today"},
    }]
    assert list(read_row["extra_info"]["teacher_trace_tool_sequence"]) == [
        "list_events", "list_events",
    ]
    assert read_row["extra_info"]["server_schema_hash"] == (
        compute_server_schema_hash(visible_tools)
    )
    assert json.loads(read_row["extra_info"]["server_schema_hashes"]) == {
        "calendar": compute_server_schema_hash(visible_tools),
    }
    assert not bool(read_row["extra_info"]["has_state_outcome_oracle"])
    assert read_row["extra_info"]["observation_schema_version"] == (
        OBSERVATION_SCHEMA_VERSION
    )
    assert read_row["extra_info"]["observation_projection_version"] == OBSERVATION_PROJECTION_VERSION
    assert read_row["extra_info"]["trajectory_schema_version"] == (
        TRAJECTORY_SCHEMA_VERSION
    )
    assert read_row["extra_info"]["max_observation_chars"] == 4096
    assert bool(read_row["extra_info"]["provenance_valid"])
    assert read_row["extra_info"]["provenance_violation_count"] == 0
    assert list(read_row["extra_info"]["reward_profile_compatibility"]) == [
        "prove_baseline", "oval_full",
    ]
    duplicate = read_row.copy()
    duplicate["extra_info"] = dict(read_row["extra_info"])
    duplicate["extra_info"]["task_id"] = "repeat-roundtrip-copy"
    deduped, removed = _dedup_jaccard(
        pd.DataFrame([read_row, duplicate]), threshold=0.70,
    )
    assert len(deduped) == 1
    assert removed == 1


def test_shard_jaccard_prefers_full_teacher_trace_sequence() -> None:
    row = _row("check events", "raw-sequence")
    row["extra_info"]["teacher_trace_tool_sequence"] = [
        "list_events", "list_events",
    ]
    row["extra_info"]["oracle_calls"] = json.dumps([
        {"tool_name": "list_events", "arguments": {}, "action": "tool_call"},
        {"tool_name": "final_answer", "arguments": {}, "action": "final_answer"},
    ])
    other = _row("check events again", "raw-sequence-2")
    other["extra_info"]["teacher_trace_tool_sequence"] = [
        "list_events", "list_events",
    ]
    other["extra_info"]["oracle_calls"] = row["extra_info"]["oracle_calls"]

    unique, removed = _dedup_jaccard(pd.DataFrame([row, other]), threshold=0.70)
    assert len(unique) == 1
    assert removed == 1


def test_jaccard_does_not_merge_same_named_tools_across_domains() -> None:
    food = _row("list my deliveries", "food-list", domain="food_delivery")
    food["extra_info"]["teacher_trace_tool_sequence"] = ["list_orders"]
    shopping = _row("list my purchases", "shopping-list", domain="shopping")
    shopping["extra_info"]["teacher_trace_tool_sequence"] = ["list_orders"]

    unique, removed = _dedup_jaccard(
        pd.DataFrame([food, shopping]), threshold=0.70,
    )
    assert len(unique) == 2
    assert removed == 0


def test_reward_preserves_missing_function_visible_prefix_calls() -> None:
    oracle = [
        {
            "tool_name": "get_event",
            "arguments": {"event_id": "evt_1"},
            "action": "tool_call",
        },
        {
            "tool_name": "ask_clarification",
            "arguments": {"question": "The update tool is unavailable."},
            "action": "ask_clarification",
        },
    ]
    task = _build_task_dict({
        "task_id": "missing-prefix",
        "scenario_type": "clarification_required",
        "has_missing_function": True,
        "oracle_calls": json.dumps(oracle),
        "success_criteria": "[]",
        "allowed_terminal_actions": ["ask_clarification"],
        "round_contracts": [{
            "round_idx": 0,
            "required_tools": ["get_event"],
            "allowed_terminal_actions": ["ask_clarification"],
        }],
    })

    assert task["required_tool_calls"] == [{
        "tool_name": "get_event",
        "arguments": {"event_id": "evt_1"},
    }]


def test_reward_accepts_missing_function_report_error_without_tool_calls() -> None:
    task = _build_task_dict({
        "task_id": "missing-zero-prefix",
        "scenario_type": "missing_function",
        "has_missing_function": True,
        "oracle_calls": json.dumps([{
            "tool_name": "report_error",
            "arguments": {"text": "The required tool is unavailable."},
            "action": "report_error",
        }]),
        "success_criteria": "[]",
        "allowed_terminal_actions": ["report_error"],
        "round_contracts": [{
            "round_idx": 0,
            "required_tools": [],
            "allowed_terminal_actions": ["report_error"],
        }],
    })

    assert task["required_tool_calls"] == []


def test_clarification_terminal_controls_scenario_even_with_prefix_calls() -> None:
    scenario = _classify_scenario(
        server_name="payments",
        oracle_calls=[OracleCall("list_invoices", {"status": "pending"})],
        execution_history=[],
        terminal_action="ask_clarification",
        seed=1,
    )
    assert scenario == "clarification_required"


def test_report_error_is_not_labeled_normal_success() -> None:
    scenario = _classify_scenario(
        server_name="team_chat",
        oracle_calls=[
            OracleCall("list_channels", {}),
            OracleCall("report_error", {"reason": "No suitable channel"}, action="report_error"),
        ],
        execution_history=[{"tool_name": "list_channels", "success": True}],
        terminal_action="report_error",
        seed=1,
    )
    assert scenario == "partial_completion_or_abstention"


def test_zero_call_report_error_is_no_tool_abstention() -> None:
    scenario = _classify_scenario(
        server_name="team_chat",
        oracle_calls=[],
        execution_history=[],
        terminal_action="report_error",
        seed=1,
    )
    assert scenario == "no_tool_or_abstention"


def test_zero_tool_terminal_contract_is_shared_across_difficulties() -> None:
    clarification = [OracleCall(
        "ask_clarification", {"question": "Which account?"},
        action="ask_clarification",
    )]
    give_up = [OracleCall(
        "report_error", {"reason": "No available tool can do that."},
        action="report_error",
    )]
    direct_answer = [OracleCall(
        "final_answer", {"text": "Done."}, action="final_answer",
    )]

    assert _zero_tool_terminal_is_valid("missing", clarification)
    assert _zero_tool_terminal_is_valid("minimal", clarification)
    assert not _zero_tool_terminal_is_valid("complete", clarification)
    assert _zero_tool_terminal_is_valid("complete", give_up)
    assert not _zero_tool_terminal_is_valid("minimal", direct_answer)


def test_report_error_after_execution_failure_stays_recovery() -> None:
    scenario = _classify_scenario(
        server_name="filesystem",
        oracle_calls=[OracleCall("report_error", {"reason": "move failed"}, action="report_error")],
        execution_history=[{"tool_name": "mv", "success": False}],
        terminal_action="report_error",
        seed=1,
    )
    assert scenario == "tool_error_recovery"


def test_merge_does_not_filter_outcome_diagnostic_false_values() -> None:
    row = pd.Series(_row("q-bad", "t-bad"))
    row["extra_info"]["project_outcome_valid"] = pd.array([False], dtype="boolean")[0]
    assert _quality_issue(row) == ""


def _environment_extra(domain: str) -> dict:
    from src.live_mcp.environment_metadata import compute_initial_state_hashes

    seed = 42
    module = importlib.import_module(f"src.live_mcp.servers.{domain}.server")
    schema_hash = compute_server_schema_hash(module.TOOLS)
    initial_hash = compute_initial_state_hashes({domain}, seed)[domain]
    return {
        "trajectory_schema_version": TRAJECTORY_SCHEMA_VERSION,
        "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
        "observation_projection_version": OBSERVATION_PROJECTION_VERSION,
        "server_schema_hash": schema_hash,
        "server_schema_hashes": json.dumps({domain: schema_hash}),
        "transition_fingerprints": json.dumps({
            domain: compute_transition_fingerprint(domain, module.TOOLS),
        }),
        "initial_state_hash": initial_hash,
        "initial_state_hashes": json.dumps({domain: initial_hash}),
        "session_seed": seed,
        "reward_fingerprint": compute_reward_fingerprint(),
        "max_observation_chars": 4096,
        "reward_profile_compatibility": ["prove_baseline", "oval_full"],
        "paper_replay_valid": True,
        "provenance_valid": True,
        "replay_error_rate": 0.0,
        "replay_num_calls": 1,
        "replay_num_errors": 0,
        "canonical_replay_valid": True,
        "canonical_replay_error_rate": 0.0,
        "canonical_replay_num_calls": 1,
        "canonical_replay_num_errors": 0,
        "canonical_replay_criteria_ok": True,
        "canonical_replay_criteria_failed": 0,
        "generation_method": "task_planner",
        "conversation_queries": json.dumps(["placeholder"]),
        "teacher_attempt_count": 1,
        "teacher_attempt_trace": json.dumps([{
            "round_idx": 0,
            "call": {"tool_name": "placeholder", "arguments": {}},
            "observation": {},
        }]),
        "teacher_round_trace": json.dumps([{
            "round_idx": 0,
            "user_query": "placeholder",
            "oracle_calls": [{"tool_name": "placeholder", "action": "tool_call"}],
            "execution_history": [],
        }]),
    }


def test_successful_mutating_noop_is_not_a_required_oracle_step() -> None:
    call = OracleCall(
        "remove_watcher",
        {"issue_id": "iss_1", "user_id": "usr_1"},
    )
    terminal = OracleCall(
        "final_answer", {"text": "No change was needed."},
        action="final_answer",
    )
    task = SimpleNamespace(
        target_servers=["issue_tracker"],
        execution_history_per_round=[[{
            "tool_name": "remove_watcher",
            "arguments": {"issue_id": "iss_1", "user_id": "usr_1"},
            "success": True,
            "state_changed": False,
        }]],
    )
    assert _required_round_oracle_calls(task, 0, [call, terminal]) == [terminal]

    task.execution_history_per_round[0][0].pop("state_changed")
    assert _required_round_oracle_calls(task, 0, [call, terminal]) == [
        call, terminal,
    ]


def test_canonical_row_replay_uses_exact_exported_oracle(monkeypatch) -> None:
    captured = {}

    def fake_replay_validate(**kwargs):
        captured.update(kwargs)
        return True, 0.0, 0, 2, True, 0

    monkeypatch.setattr(
        "src.live_mcp.task_planner.replay_validate", fake_replay_validate,
    )
    oracle = [
        {
            "tool_name": "get_event",
            "arguments": {"event_id": "evt_1"},
            "action": "tool_call",
        },
        {
            "tool_name": "final_answer",
            "arguments": {"text": "Found it."},
            "action": "final_answer",
        },
    ]
    rows = [{
        "uid": "calendar_exact_replay",
        "reward_model": {"ground_truth": {
            "oracle_calls": json.dumps(oracle),
            "success_criteria": "[]",
        }},
        "extra_info": {
            "task_id": "calendar_exact_replay",
            "domain": "calendar",
            "session_seed": 42,
            "hidden_tools": json.dumps(["update_event"]),
        },
    }]

    _validate_canonical_rows_replay(rows, manager=object(), executor=object())

    assert [call.tool_name for call in captured["oracle_calls"]] == [
        "get_event", "final_answer",
    ]
    assert captured["blocked_tools"] == {"update_event"}
    assert rows[0]["extra_info"]["canonical_replay_valid"] is True
    assert rows[0]["extra_info"]["canonical_replay_num_calls"] == 2


def _row(query: str, task_id: str, domain: str = "calendar") -> dict:
    oracle = [{
        "tool_name": f"get_event_{query}",
        "arguments": {"event_id": query},
        "action": "tool_call",
    }, {
        "tool_name": "final_answer",
        "arguments": {"text": "done"},
        "action": "final_answer",
    }]
    return {
        "prompt": json.dumps([
            {"role": "system", "content": "calendar tools"},
            {"role": "user", "content": query},
        ]),
        "data_source": "live_mcp_state_machine",
        "reward_model": {
            "style": "rule",
            "ground_truth": {"oracle_calls": json.dumps(oracle)},
        },
        "extra_info": {
            **_environment_extra(domain),
            "domain": domain,
            "user_query": query,
            "oracle_calls": json.dumps(oracle),
            "success_criteria": "[]",
            "allowed_terminal_actions": ["final_answer"],
            "round_contracts": json.dumps([{
                "round_idx": 0,
                "required_tools": [f"get_event_{query}"],
                "allowed_terminal_actions": ["final_answer"],
            }]),
            "dependency_edges": "[]",
            "task_id": task_id,
            "hidden_tools": [],
            "visible_tool_names": [f"get_event_{query}"],
            "conversation_queries": json.dumps([query]),
            "teacher_attempt_trace": json.dumps([{
                "round_idx": 0,
                "call": {"tool_name": f"get_event_{query}", "arguments": {}},
                "observation": {},
            }]),
            "teacher_round_trace": json.dumps([{
                "round_idx": 0,
                "user_query": query,
                "oracle_calls": oracle,
                "execution_history": [],
            }]),
        },
        "uid": task_id,
        "group_id": task_id,
        "perturbation_level": "complete",
        "scenario_type": "normal_safe_success",
    }


def test_merge_backfills_after_fingerprint_and_task_id_overlap(tmp_path: Path) -> None:
    shard_dir = tmp_path / "shards"
    output_dir = tmp_path / "out"
    shard_dir.mkdir()
    pd.DataFrame([
        _row("q1", "t1"),
        _row("q2", "t2"),
        _row("q3", "t3"),
    ]).to_parquet(shard_dir / "shard_0_train.parquet", index=False)
    pd.DataFrame([
        _row("q1", "v-overlap-fingerprint"),
        _row("q4", "t2"),
        _row("q5", "t5"),
        _row("q6", "t6"),
    ]).to_parquet(shard_dir / "shard_0_val.parquet", index=False)

    assert merge_shards(shard_dir, output_dir, count=2, val_count=2) == 0
    train = pd.read_parquet(output_dir / "train.parquet")
    val = pd.read_parquet(output_dir / "val.parquet")
    assert len(train) == 2
    assert len(val) == 2
    assert not (
        {_row_fingerprint(row) for _, row in train.iterrows()}
        & {_row_fingerprint(row) for _, row in val.iterrows()}
    )
    train_ids = {row["extra_info"]["task_id"] for _, row in train.iterrows()}
    val_ids = {row["extra_info"]["task_id"] for _, row in val.iterrows()}
    assert not (train_ids & val_ids)


def test_global_merge_enforces_train_and_val_domain_quotas(tmp_path: Path) -> None:
    shard_dir = tmp_path / "shards"
    output_dir = tmp_path / "out"
    shard_dir.mkdir()
    rows = [
        _row(f"{domain}-{index}", f"{domain}-{index}", domain)
        for domain in ("calendar", "filesystem")
        for index in range(3)
    ]
    pd.DataFrame(rows[:3]).to_parquet(shard_dir / "shard_0_train.parquet", index=False)
    pd.DataFrame(rows[3:]).to_parquet(shard_dir / "shard_0_val.parquet", index=False)

    assert merge_shards(
        shard_dir, output_dir, count=4, val_count=2,
        domains=["calendar", "filesystem"],
    ) == 0
    train = pd.read_parquet(output_dir / "train.parquet")
    val = pd.read_parquet(output_dir / "val.parquet")
    assert train["extra_info"].map(lambda x: x["domain"]).value_counts().to_dict() == {
        "calendar": 2, "filesystem": 2,
    }
    assert val["extra_info"].map(lambda x: x["domain"]).value_counts().to_dict() == {
        "calendar": 1, "filesystem": 1,
    }


def test_global_merge_fails_when_one_domain_cannot_meet_quota(tmp_path: Path) -> None:
    shard_dir = tmp_path / "shards"
    output_dir = tmp_path / "out"
    shard_dir.mkdir()
    pd.DataFrame([
        _row("calendar-0", "calendar-0", "calendar"),
        _row("calendar-1", "calendar-1", "calendar"),
        _row("filesystem-0", "filesystem-0", "filesystem"),
    ]).to_parquet(shard_dir / "shard_0_train.parquet", index=False)
    pd.DataFrame([]).to_parquet(shard_dir / "shard_0_val.parquet", index=False)

    deficits_path = tmp_path / "deficits.json"
    assert merge_shards(
        shard_dir, output_dir, count=2, val_count=2,
        domains=["calendar", "filesystem"],
        deficits_output=deficits_path,
    ) == 1
    report = json.loads(deficits_path.read_text())
    assert report["available_by_domain"] == {"calendar": 2, "filesystem": 1}
    assert report["required_by_domain"] == {"calendar": 2, "filesystem": 2}
    assert report["deficits"] == {"filesystem": 1}


def test_global_merge_reports_retention_aware_topup_size(tmp_path: Path) -> None:
    shard_dir = tmp_path / "shards"
    output_dir = tmp_path / "out"
    shard_dir.mkdir()
    rows = [
        _row(f"calendar-{index}", f"calendar-{index}", "calendar")
        for index in range(2)
    ]
    pd.DataFrame(rows).to_parquet(
        shard_dir / "shard_0_train.parquet", index=False,
    )
    pd.DataFrame([]).to_parquet(
        shard_dir / "shard_0_val.parquet", index=False,
    )

    deficits_path = tmp_path / "deficits.json"
    assert merge_shards(
        shard_dir, output_dir, count=3, val_count=0,
        domains=["calendar"], deficits_output=deficits_path,
    ) == 1
    report = json.loads(deficits_path.read_text())
    assert report["candidate_by_domain"] == {"calendar": 2}
    assert report["jaccard_retention_by_domain"] == {"calendar": 1.0}
    assert report["suggested_topup_by_domain"] == {"calendar": 3}


def test_topup_size_scales_with_observed_jaccard_retention() -> None:
    assert _suggest_topup_count(
        missing=11, available=49, candidates=170,
    ) == 47
    assert _suggest_topup_count(
        missing=3, available=57, candidates=137,
    ) == 10
    assert _suggest_topup_count(
        missing=0, available=60, candidates=100,
    ) == 0


def test_stage3_output_gate_checks_exact_domain_counts(tmp_path: Path) -> None:
    path = tmp_path / "smoke.parquet"
    pd.DataFrame([
        _row("calendar-0", "calendar-0", "calendar"),
        _row("filesystem-0", "filesystem-0", "filesystem"),
    ]).to_parquet(path, index=False)
    assert _stage3_output_issue(path, ["calendar", "filesystem"], 1) == ""
    issue = _stage3_output_issue(path, ["calendar", "filesystem"], 2)
    assert "行数错误" in issue


def test_client_and_recovery_seed_ranges_do_not_overlap() -> None:
    base_seed = 42
    client_stride = 1_000_000
    seeds = {
        base_seed + client * client_stride + recovery_round * 100_000
        for client in range(8)
        for recovery_round in range(3)
    }
    assert len(seeds) == 8 * 3


def test_recovery_targets_only_domains_below_final_quota() -> None:
    tasks = [
        SimpleNamespace(target_servers=[domain])
        for domain in ("calendar", "shopping", "banking")
    ]
    counts, requests = _domain_recovery_requests(
        tasks,
        ["calendar", "shopping", "banking", "filesystem", "food_delivery"],
        {domain: 1 for domain in (
            "calendar", "shopping", "banking", "filesystem", "food_delivery",
        )},
    )
    assert counts["calendar"] == 1
    assert requests == [("filesystem", 2), ("food_delivery", 2)]


def test_recovery_does_not_reinject_irrelevance() -> None:
    assert _irrelevance_ratio_for_round(0.05, 0) == 0.05
    assert _irrelevance_ratio_for_round(0.05, 1) == 0.0
    assert _irrelevance_ratio_for_round(0.05, 5) == 0.0

    source = Path("scripts/generate_data.sh").read_text()
    start = source.index("merge_vllm_with_topups()")
    topup = source[start:source.index("# MODE 1:", start)]
    assert "--irrelevance-ratio 0" in topup


def test_topup_failure_preserves_successful_shards_for_global_retry() -> None:
    source = Path("scripts/generate_data.sh").read_text()
    start = source.index('if [ "${topup_failed}" -gt 0 ]; then')
    block = source[start:source.index("        fi", start) + len("        fi")]
    assert "preserving successful candidate shards" in block
    assert "return 1" not in block


def test_topup_uses_available_generation_client_slots() -> None:
    source = Path("scripts/generate_data.sh").read_text()
    start = source.index("merge_vllm_with_topups()")
    topup = source[start:source.index("# MODE 1:", start)]
    assert 'total_topup_slots="${TOTAL_GEN_CLIENTS:-${NUM_INSTANCES}}"' in topup
    assert 'topup_count / chunk_count' in topup
    assert 'topup_${topup_round}_${topup_domain}_${chunk_index}' in topup


def test_final_split_keeps_identical_initial_queries_on_one_side() -> None:
    def row(task_id: str, query: str) -> dict:
        return {
            "uid": task_id,
            "extra_info": {"task_id": task_id, "user_query": query},
        }

    train = pd.DataFrame([
        row("train_dup", "Pay My Bill"),
        row("train_unique", "show my balance"),
    ])
    val = pd.DataFrame([
        row("val_dup", "  pay   my bill "),
        row("val_unique", "show my history"),
    ])

    isolated = _isolate_initial_queries(train, val)
    assert isolated is not None
    isolated_train, isolated_val = isolated
    train_queries = {
        " ".join(r["extra_info"]["user_query"].lower().split())
        for _, r in isolated_train.iterrows()
    }
    val_queries = {
        " ".join(r["extra_info"]["user_query"].lower().split())
        for _, r in isolated_val.iterrows()
    }
    assert not (train_queries & val_queries)
    assert len(isolated_train) == len(train)
    assert len(isolated_val) == len(val)


def test_final_split_isolates_cross_domain_query_and_preserves_quotas() -> None:
    def row(task_id: str, query: str, domain: str) -> dict:
        return {
            "uid": task_id,
            "extra_info": {
                "task_id": task_id,
                "user_query": query,
                "domain": domain,
            },
        }

    train = pd.DataFrame([
        row("shopping_dup", "remove a red wine stain", "shopping"),
        row("calendar_swap", "show tomorrow meetings", "calendar"),
    ])
    val = pd.DataFrame([
        row("calendar_dup", " remove  a red wine stain ", "calendar"),
        row("shopping_unique", "show my orders", "shopping"),
    ])
    before = {
        split: frame["extra_info"].map(lambda x: x["domain"]).value_counts().to_dict()
        for split, frame in (("train", train), ("val", val))
    }
    isolated = _isolate_initial_queries(train, val)
    assert isolated is not None
    isolated_train, isolated_val = isolated
    after = {
        split: frame["extra_info"].map(lambda x: x["domain"]).value_counts().to_dict()
        for split, frame in (("train", isolated_train), ("val", isolated_val))
    }
    assert after == before
    train_queries = set(isolated_train.apply(_initial_query_key, axis=1))
    val_queries = set(isolated_val.apply(_initial_query_key, axis=1))
    assert not (train_queries & val_queries)


def test_merge_rejects_unrelated_email_label_criteria() -> None:
    row = pd.Series({
        "prompt": json.dumps([{"role": "user", "content": "label it"}]),
        "scenario_type": "normal_safe_success",
        "extra_info": {
            **_environment_extra("email"),
            "domain": "email",
            "paper_replay_valid": True,
            "hidden_tools": [],
            "visible_tool_names": ["add_label"],
            "oracle_calls": json.dumps([{
                "action": "tool_call",
                "tool_name": "add_label",
                "arguments": {"email_id": "eml_1", "label": "urgent"},
            }]),
            "success_criteria": json.dumps([
                {"type": "state_equals", "path_parts": ["emails", "eml_1", "labels"], "value": ["urgent"]},
                {"type": "state_equals", "path_parts": ["emails", "eml_2", "labels"], "value": ["urgent"]},
            ]),
        },
    })
    assert _quality_issue(row).startswith("unrelated_email_label_criteria:")


def test_merge_rejects_unrelated_team_chat_reaction_criteria() -> None:
    from src.live_mcp.state_seeder import StateSeeder

    seed = 42
    state = StateSeeder().seed_state("team_chat", "audit", seed)
    messages = [
        message
        for channel in state["channels"].values()
        for message in channel["messages"]
    ]
    target, unrelated = messages[:2]
    polluted = [dict(message) for message in messages[:2]]
    polluted[0]["reactions"] = [*target["reactions"], "audit-reaction"]
    polluted[1]["reactions"] = [*unrelated["reactions"], "audit-reaction"]
    row = pd.Series({
        "prompt": json.dumps([{"role": "user", "content": "react to it"}]),
        "scenario_type": "normal_safe_success",
        "extra_info": {
            **_environment_extra("team_chat"),
            "domain": "team_chat",
            "session_seed": seed,
            "paper_replay_valid": True,
            "hidden_tools": [],
            "visible_tool_names": ["react_message"],
            "oracle_calls": json.dumps([{
                "action": "tool_call",
                "tool_name": "react_message",
                "arguments": {
                    "channel_id": target["channel_id"],
                    "message_id": target["message_id"],
                    "reaction": "audit-reaction",
                },
            }]),
            "success_criteria": json.dumps([{
                "type": "state_equals",
                "path_parts": ["channels", target["channel_id"], "messages"],
                "value": polluted,
            }]),
        },
    })
    assert _quality_issue(row).startswith(
        "unrelated_team_chat_reaction_criteria:"
    )


def test_merge_rejects_stale_environment_metadata() -> None:
    row = pd.Series({
        "prompt": json.dumps([{"role": "user", "content": "do it"}]),
        "scenario_type": "normal_safe_success",
        "extra_info": {
            "domain": "banking",
            "trajectory_schema_version": "live-mcp-trajectory-v0",
        },
    })
    assert _quality_issue(row) == "stale_trajectory_schema:live-mcp-trajectory-v0"


def test_merge_rejects_missing_environment_metadata() -> None:
    row = pd.Series({
        "prompt": json.dumps([{"role": "user", "content": "do it"}]),
        "scenario_type": "normal_safe_success",
        "extra_info": {"domain": "banking"},
    })
    assert _quality_issue(row) == (
        "missing_environment_metadata:trajectory_schema_version"
    )


def test_recovery_zero_yield_does_not_abort_other_domains() -> None:
    assert _is_zero_yield_error(RuntimeError(
        "generate_many produced 0 tasks (target 2, failures=2)"
    ))
    assert not _is_zero_yield_error(RuntimeError("MCP session crashed"))
    zero = RuntimeError("generate_many produced 0 tasks (target 1, failures=1)")
    assert _zero_yield_is_recoverable(
        zero, recovery_round=0, shard_mode=True,
    )
    assert not _zero_yield_is_recoverable(
        zero, recovery_round=0, shard_mode=False,
    )
    assert _zero_yield_is_recoverable(
        zero, recovery_round=1, shard_mode=False,
    )


def test_replay_counts_not_found_execution_failure_as_error() -> None:
    class Manager:
        def create_session(self, seed):
            return SimpleNamespace(session_id="replay")
        def discover_tools(self, session_id):
            return []
        def get_state(self, session_id):
            return {}
        def close_session(self, session_id):
            pass

    class Executor:
        def execute(self, *args, **kwargs):
            return SimpleNamespace(
                success=False,
                schema_valid=True,
                observation={"error": "thread not found: eml_1"},
            )

    passed, rate, errors, calls, _, _ = replay_validate(
        [OracleCall("get_thread", {"thread_id": "eml_1"})],
        Manager(), Executor(), seed=1, domain="email",
    )
    assert not passed
    assert (rate, errors, calls) == (1.0, 1, 1)


def test_replay_allows_successful_empty_result() -> None:
    class Manager:
        def create_session(self, seed):
            return SimpleNamespace(session_id="replay")
        def discover_tools(self, session_id):
            return []
        def get_state(self, session_id):
            return {}
        def close_session(self, session_id):
            pass

    class Executor:
        def execute(self, *args, **kwargs):
            return SimpleNamespace(success=True, schema_valid=True, observation=[])

    passed, rate, errors, calls, _, _ = replay_validate(
        [OracleCall("search_messages", {"query": "none"})],
        Manager(), Executor(), seed=1, domain="team_chat",
    )
    assert passed
    assert (rate, errors, calls) == (0.0, 0, 1)


def test_replay_rejects_unexpected_success_even_below_error_threshold() -> None:
    class Manager:
        def create_session(self, seed):
            return SimpleNamespace(session_id="replay")
        def discover_tools(self, session_id):
            return []
        def get_state(self, session_id):
            return {}
        def close_session(self, session_id):
            pass

    class Executor:
        def execute(self, *args, **kwargs):
            return SimpleNamespace(
                success=True, schema_valid=True, observation={"created": True},
                execution_status="SUCCESS", state_changed=True,
                error_type=None, error_message="",
            )

    program = [
        OracleCall("create_event", {}, expected_success=False),
        OracleCall("list_events", {}),
        OracleCall("list_events", {}),
        OracleCall("list_events", {}),
    ]
    passed, rate, errors, calls, _, _ = replay_validate(
        program, Manager(), Executor(), seed=1, domain="calendar",
    )
    assert not passed
    assert (rate, errors, calls) == (1.0, 1, 1)


def test_replay_trace_records_fresh_state_call_and_result() -> None:
    class Manager:
        def __init__(self):
            self.state = {"crm": {"deals": {}}}
        def create_session(self, seed):
            return SimpleNamespace(session_id="replay-traced")
        def discover_tools(self, session_id):
            return []
        def get_state(self, session_id):
            return self.state
        def close_session(self, session_id):
            pass

    class Executor:
        def execute(self, *args, **kwargs):
            return SimpleNamespace(
                success=True, schema_valid=True,
                execution_status="SUCCESS", state_changed=False,
                observation={"deals": []}, error_type="", error_message="",
            )

    events = []
    replay_validate(
        [OracleCall("list_deals", {})],
        Manager(), Executor(), seed=7, domain="crm",
        trace_recorder=lambda stage, **payload: events.append((stage, payload)),
        trace_include_state=True,
    )

    assert [stage for stage, _ in events] == [
        "replay_start", "replay_call", "replay_result",
    ]
    assert events[0][1]["initial_state"] == {"crm": {"deals": {}}}
    assert events[1][1]["observation"] == {"deals": []}
    assert events[2][1]["passed"] is True


def test_generation_checkpoint_roundtrip_preserves_internal_task(tmp_path) -> None:
    args = SimpleNamespace(
        count=500, val_count=100, domain="all", model="teacher",
        api_base="http://127.0.0.1:8001/v1", seed=7, suite="suite.yaml",
        pool_oversample_pct=0.5, irrelevance_ratio=0.05,
        distractor_rate=0.4, missing_function_rate=0.2,
    )
    task = LiveTask(
        task_id="calendar_7", source="teacher", suite_name="suite",
        user_prompt="show event", session_id="s7", session_seed=7,
        target_servers=["calendar"], visible_tools=[{"name": "get_event"}],
        required_tools=["get_event"], expected_outcome={"status": "ok"},
        success_criteria=[{"field": "id", "value": "evt_1"}],
        oracle_program=OracleProgram(
            task_id="calendar_7",
            calls=[OracleCall("get_event", {"event_id": "evt_1"})],
            success_criteria=[{"field": "id", "value": "evt_1"}],
            progress_predicates=[{"step": 1, "tool": "get_event"}],
        ),
        sampling_context={"entity": "evt_1"}, max_turns=3,
        metadata={"scenario_type": "normal_safe_success"},
        oracle_calls_per_round=[
            [OracleCall("get_event", {"event_id": "evt_1"})],
            [OracleCall(
                "final_answer", {"text": "done"}, action="final_answer",
            )],
        ],
    )
    path = tmp_path / "pool.json"
    _write_generation_checkpoint(
        path, args, [task], completed_rounds=4,
        round_requests=[("payments", 3)],
    )
    tasks, completed, requests = _load_generation_checkpoint(path, args)
    assert completed == 4
    assert requests == [("payments", 3)]
    assert tasks == [task]
    assert all(
        isinstance(call, OracleCall)
        for round_calls in tasks[0].oracle_calls_per_round
        for call in round_calls
    )

    mismatched = SimpleNamespace(**vars(args))
    mismatched.seed = 8
    try:
        _load_generation_checkpoint(path, mismatched)
    except ValueError as exc:
        assert "config mismatch" in str(exc)
    else:
        raise AssertionError("mismatched checkpoint config must fail")


def test_prove_graceful_give_up_does_not_require_failed_tool_call() -> None:
    task = LiveTask(
        task_id="calendar_report_without_failure", source="teacher",
        suite_name="suite", user_prompt="show event", session_id="s1",
        session_seed=1, target_servers=["calendar"],
        visible_tools=[{"name": "list_events"}], required_tools=["list_events"],
        expected_outcome={}, success_criteria=[],
        oracle_program=OracleProgram(
            task_id="calendar_report_without_failure",
            calls=[
                OracleCall("list_events", {}),
                OracleCall("final_answer", {"text": "done"}, action="final_answer"),
                OracleCall("report_error", {"text": "cannot"}, action="report_error"),
            ], success_criteria=[],
        ),
        sampling_context={}, max_turns=3,
        metadata={
            "scenario_type": "partial_completion_or_abstention",
            "generation_mode": "chain_seeded",
            "chain_seed": [],
            "paper_replay_valid": True,
            "provenance_valid": True,
        },
        conversation_queries=["show event", "move another event"],
        oracle_calls_per_round=[
            [
                OracleCall("list_events", {}),
                OracleCall("final_answer", {"text": "done"}, action="final_answer"),
            ],
            [OracleCall("report_error", {"text": "cannot"}, action="report_error")],
        ],
        execution_history_per_round=[
            [{"tool_name": "list_events", "success": True}],
            [],
        ],
    )
    _validate_task_training_contract(task)
    assert _filter_training_eligible_tasks([task]) == [task]


def test_split_honors_explicit_domain_quotas() -> None:
    tasks = []
    for index, domain in enumerate(("calendar", "calendar", "filesystem")):
        tasks.append(SimpleNamespace(
            task_id=f"quota-{index}",
            target_servers=[domain],
            user_prompt=f"query {index}",
            task_type="normal",
            metadata={},
            oracle_program=SimpleNamespace(calls=[
                OracleCall(f"tool_{index}", {"id": index}),
            ]),
        ))
    train, val = _stratified_task_split(
        tasks, train_count=2, val_count=0, seed=3,
        domain_quotas={"calendar": 1, "filesystem": 1},
    )
    assert not val
    assert {task.target_servers[0] for task in train} == {"calendar", "filesystem"}


def test_split_enforces_train_and_val_domain_quotas_separately() -> None:
    domains = ["calendar", "filesystem"]
    tasks = [
        SimpleNamespace(
            task_id=f"{domain}-{index}", target_servers=[domain],
            user_prompt=f"{domain} query {index}", task_type="normal",
            metadata={}, oracle_program=SimpleNamespace(calls=[
                OracleCall(f"{domain}_tool_{index}", {"id": index}),
            ]),
        )
        for domain in domains for index in range(3)
    ]
    train, val = _stratified_task_split(
        tasks, train_count=4, val_count=2, seed=3,
        domain_quotas={domain: 3 for domain in domains},
    )
    assert _domain_quotas(4, domains) == {"calendar": 2, "filesystem": 2}
    assert [sum(t.target_servers[0] == domain for t in train) for domain in domains] == [2, 2]
    assert [sum(t.target_servers[0] == domain for t in val) for domain in domains] == [1, 1]


def test_shard_mode_does_not_require_each_shard_to_meet_domain_quotas() -> None:
    tasks = [
        SimpleNamespace(
            task_id=f"calendar-{i}", target_servers=["calendar"],
            user_prompt=f"calendar query {i}", task_type="normal",
            metadata={}, oracle_program=SimpleNamespace(calls=[
                OracleCall(f"calendar_tool_{i}", {"id": i}),
            ]),
        )
        for i in range(4)
    ] + [
        SimpleNamespace(
            task_id="filesystem-0", target_servers=["filesystem"],
            user_prompt="filesystem query", task_type="normal",
            metadata={}, oracle_program=SimpleNamespace(calls=[
                OracleCall("filesystem_tool", {"id": 0}),
            ]),
        )
    ]
    train, val = _stratified_task_split(
        tasks, train_count=4, val_count=1, seed=3,
        domain_quotas=None,
    )
    assert len(train) == 4
    assert len(val) == 1


def test_candidate_shard_keeps_locally_similar_eligible_rows() -> None:
    tasks = [
        SimpleNamespace(
            task_id=f"banking-{i}", target_servers=["banking"],
            user_prompt=f"query {i}", task_type="normal",
            metadata={}, oracle_program=SimpleNamespace(calls=[
                OracleCall("list_accounts", {"type": "checking"}),
            ]),
        )
        for i in range(3)
    ]
    train, val = _candidate_shard_split(
        tasks, train_count=2, val_count=1, seed=3,
    )
    assert len(train) == 2
    assert len(val) == 1
    assert {task.task_id for task in train + val} == {
        "banking-0", "banking-1", "banking-2",
    }
    assert all(task.metadata.get("semantic_fingerprint") for task in tasks)


def test_shard_integrity_accepts_partial_nonempty_candidate_output() -> None:
    args = SimpleNamespace(count=3, val_count=1, shard_mode=True)
    train = pd.DataFrame([
        {"extra_info": {"semantic_fingerprint": "a"}},
        {"extra_info": {"semantic_fingerprint": "b"}},
    ])
    _assert_split_integrity(train, pd.DataFrame(), args)


def test_handler_state_preconditions_reject_proven_impossible_chains() -> None:
    assert not _chain_respects_state_preconditions(
        "payments", ["create_invoice", "pay_invoice", "cancel_payment"],
    )
    assert not _chain_respects_state_preconditions(
        "food_delivery", ["create_order", "track_rider"],
    )
    assert not _chain_respects_state_preconditions(
        "food_delivery", ["reorder", "track_rider"],
    )
    assert not _chain_respects_state_preconditions(
        "crm", ["create_lead", "convert_lead", "delete_contact"],
    )


def test_crm_delete_contact_preserves_converted_lead_reference() -> None:
    server = CRMServer()
    server.sessions["crm-reference"] = {
        "leads": {
            "lead_1": {
                "lead_id": "lead_1",
                "status": "converted",
                "contact_id": "contact_1",
            },
        },
        "contacts": {"contact_1": {"contact_id": "contact_1"}},
        "deals": {},
    }
    with __import__("pytest").raises(
        KeyError, match="referenced by 1 converted lead",
    ):
        server.delete_contact("crm-reference", {"contact_id": "contact_1"})
    assert "contact_1" in server.sessions["crm-reference"]["contacts"]


def test_handler_state_preconditions_defer_live_state_to_feasibility_filter() -> None:
    assert _chain_respects_state_preconditions(
        "shopping", ["apply_coupon", "checkout"],
    )
    assert _chain_respects_state_preconditions(
        "shopping", ["checkout", "get_order"],
    )
    assert _chain_respects_state_preconditions(
        "food_delivery", ["list_orders", "cancel_order"],
    )
    assert _chain_respects_state_preconditions(
        "food_delivery", ["update_order_status", "track_rider"],
    )
    assert _chain_respects_state_preconditions(
        "payments", ["list_invoices", "cancel_payment"],
    )


def test_food_delivery_entity_filter_simulates_handler_lifecycle() -> None:
    assert _entity_record_satisfies_chain(
        server_name="food_delivery",
        chain_seed=["list_orders", "update_order_status", "track_rider"],
        etype="order",
        record={"status": "preparing", "tip": 0},
    )
    assert not _entity_record_satisfies_chain(
        server_name="food_delivery",
        chain_seed=["list_orders", "update_order_status", "track_rider"],
        etype="order",
        record={"status": "placed", "tip": 0},
    )
    assert _entity_record_satisfies_chain(
        server_name="food_delivery",
        chain_seed=["list_orders", "update_order_status", "rate_order"],
        etype="order",
        record={"status": "delivering", "tip": 0},
    )
    assert _entity_record_satisfies_chain(
        server_name="food_delivery",
        chain_seed=["reorder", "add_tip"],
        etype="order",
        record={"status": "delivered", "tip": 5},
    )


def test_reorder_consumes_old_order_and_creates_new_order() -> None:
    assert _tool_existing_entity_requirements("reorder", "food_delivery") == {"order"}
    assert _CREATED_ENTITY_BY_TOOL["reorder"] == {"order"}


def test_pay_invoice_entity_filter_rejects_an_existing_linked_payment() -> None:
    chain = ["list_invoices", "pay_invoice"]
    assert not _entity_record_satisfies_chain(
        server_name="payments",
        chain_seed=chain,
        etype="invoice",
        record={"status": "pending", "payment_id": "pay_existing"},
    )
    assert _entity_record_satisfies_chain(
        server_name="payments",
        chain_seed=chain,
        etype="invoice",
        record={"status": "pending", "payment_id": None},
    )


def test_success_criteria_do_not_duplicate_list_or_domain_postconditions() -> None:
    initial = {
        "events": {"evt_1": {"reminders": [], "title": "sync"}},
        "leads": {"lead_1": {"status": "new"}},
    }
    final = {
        "events": {"evt_1": {
            "reminders": [{"id": "rem_1", "minutes_before": 15}],
            "title": "sync",
        }},
        "leads": {"lead_1": {"status": "converted"}},
    }
    criteria = derive_success_criteria(
        initial, final,
        [OracleCall("set_reminder", {"event_id": "evt_1"}),
         OracleCall("convert_lead", {"lead_id": "lead_1"})],
        "crm",
    )
    keys = [
        (c.get("type"), c.get("server"), c.get("path"), json.dumps(c.get("value"), sort_keys=True))
        for c in criteria
    ]
    assert len(keys) == len(set(keys))
    assert sum(c.get("path") == "events.evt_1.reminders" for c in criteria) == 1
    assert sum(c.get("path") == "leads.lead_1.status" for c in criteria) == 1


def test_success_criteria_capture_new_nested_calendar_response() -> None:
    initial = {"events": {"evt_1": {"title": "Sync"}}}
    final = {
        "events": {
            "evt_1": {
                "title": "Sync",
                "responses": {"alice@example.com": "accepted"},
            },
        },
    }
    criteria = derive_success_criteria(
        initial,
        final,
        [OracleCall(
            "respond_to_event",
            {"event_id": "evt_1", "email": "alice@example.com", "response": "accepted"},
        )],
        "calendar",
    )
    assert {
        "type": "state_equals",
        "server": "calendar",
        "path": "events.evt_1.responses.alice@example.com",
        "path_parts": ["events", "evt_1", "responses", "alice@example.com"],
        "value": "accepted",
    } in criteria


def test_checkout_does_not_require_nonempty_final_cart() -> None:
    initial = {"cart": []}
    final = {"cart": [], "orders": {"ord_1": {"status": "confirmed"}}}
    criteria = derive_success_criteria(
        initial,
        final,
        [
            OracleCall("add_to_cart", {"product_id": "prod_1", "quantity": 1}),
            OracleCall("checkout", {}),
        ],
        "shopping",
    )
    assert not any(c.get("type") == "cart_not_empty" for c in criteria)


def test_success_criteria_do_not_emit_pathless_count_aliases() -> None:
    initial = {"emails": {}, "channels": {"ch_1": {"messages": []}}}
    final = {
        "emails": {"eml_1": {"email_id": "eml_1", "subject": "Done"}},
        "channels": {"ch_1": {"messages": [{"message_id": "msg_1"}]}},
    }
    criteria = derive_success_criteria(
        initial, final,
        [
            OracleCall("send_email", {"to": "a@example.com"}),
            OracleCall("send_message", {"channel_id": "ch_1"}),
        ],
        "email",
    )
    assert not any(
        criterion.get("type") in {"email_count_gte", "cart_not_empty"}
        or str(criterion.get("path", "")).endswith(".messages_count")
        for criterion in criteria
    )


def test_issue_tracker_member_ids_are_discoverable_and_accepted() -> None:
    server = IssueTrackerServer()
    session_id = "member-grounding"
    server.handle_request("session/reset", {"session_id": session_id, "seed": 42})
    result = server.list_members(session_id, {"name": "alice"})
    members = result["observation"]["members"]
    assert result["success"] and len(members) == 1
    assert members[0]["name"] == "Alice"
    assert members[0]["user_id"].startswith("usr_")

    issue_id = next(iter(server.sessions[session_id]["issues"]))
    assigned = server.assign_issue(
        session_id,
        {"issue_id": issue_id, "assignee": members[0]["user_id"]},
    )
    assert assigned["success"]
    assert assigned["observation"]["issue"]["assignee"] == members[0]["user_id"]


def test_issue_tracker_assignee_chain_requires_discoverable_user() -> None:
    issue_only = {
        "qualified_entity_ids": [{"type": "issue", "id": "iss_1"}],
    }
    ok, reason = _chain_is_feasible(
        ["list_issues", "assign_issue"], "issue_tracker", issue_only,
    )
    assert not ok and "user(0/1)" in reason

    grounded = {
        "qualified_entity_ids": [
            {"type": "issue", "id": "iss_1"},
            {"type": "user", "id": "usr_1"},
        ],
    }
    assert _chain_is_feasible(
        ["list_members", "list_issues", "assign_issue"],
        "issue_tracker",
        grounded,
    )[0]


def test_feasible_chain_filter_accepts_probe_entity_list() -> None:
    orchestrator = TaskOrchestrator.__new__(TaskOrchestrator)
    live_context = {
        "qualified_entity_ids": [
            {"type": "issue", "id": "iss_1"},
            {"type": "user", "id": "usr_1"},
        ],
        "qualified_entity_records": [],
        "probed_entity_count": 2,
        "qualified_entity_count": 2,
    }
    chain = ["list_members", "list_issues", "assign_issue"]
    assert orchestrator._filter_feasible_chains(
        [chain], "issue_tracker", live_context,
    ) == [chain]


def test_issue_tracker_member_probe_enriches_previously_seen_user() -> None:
    class Executor:
        def execute(self, _session_id, call, domain=None):
            assert domain == "issue_tracker"
            if call.name == "list_issues":
                observation = {
                    "issues": [{"issue_id": "iss_1", "assignee": "usr_1"}],
                }
            else:
                assert call.name == "list_members"
                observation = {
                    "members": [{
                        "user_id": "usr_1",
                        "name": "Alice",
                        "role": "developer",
                    }],
                }
            return SimpleNamespace(
                success=True,
                state_changed=False,
                observation=observation,
                error_type=None,
            )

    readonly_schema = lambda name: {
        "name": name,
        "input_schema": {"type": "object", "properties": {}, "required": []},
        "annotations": {"readonly": True, "mutating": False},
    }
    context = TaskOrchestrator._probe_live_sampling_context(
        SimpleNamespace(executor=Executor()),
        "probe-session",
        "issue_tracker",
        [readonly_schema("list_issues"), readonly_schema("list_members")],
    )
    user_record = next(
        item for item in context["entity_records"]
        if item["type"] == "user" and item["id"] == "usr_1"
    )
    assert user_record["data"]["name"] == "Alice"
    user_summary = next(
        summary for item, summary in zip(
            context["entity_ids"], context["entity_summaries"]
        )
        if item == {"type": "user", "id": "usr_1"}
    )
    assert "Alice" in user_summary


def test_live_probe_does_not_merge_foreign_record_into_referenced_entity() -> None:
    class Executor:
        def execute(self, _session_id, call, domain=None):
            assert domain == "crm"
            if call.name == "list_leads":
                observation = {
                    "leads": [{
                        "lead_id": "lead_1",
                        "contact_id": "contact_1",
                        "name": "Lead Name",
                        "status": "converted",
                    }],
                }
            else:
                assert call.name == "list_deals"
                observation = {
                    "deals": [{
                        "deal_id": "deal_1",
                        "contact_id": "contact_1",
                        "name": "Deal Name",
                        "amount": 12000,
                        "stage": "negotiation",
                    }],
                }
            return SimpleNamespace(
                success=True,
                state_changed=False,
                observation=observation,
                error_type=None,
            )

    readonly_schema = lambda name: {
        "name": name,
        "input_schema": {"type": "object", "properties": {}, "required": []},
        "annotations": {"readonly": True, "mutating": False},
    }
    context = TaskOrchestrator._probe_live_sampling_context(
        SimpleNamespace(executor=Executor()),
        "probe-session",
        "crm",
        [readonly_schema("list_leads"), readonly_schema("list_deals")],
    )
    records = {
        (item["type"], item["id"]): item["data"]
        for item in context["entity_records"]
    }
    assert records[("lead", "lead_1")]["name"] == "Lead Name"
    assert records[("deal", "deal_1")]["name"] == "Deal Name"
    assert records[("contact", "contact_1")] == {}


def test_live_sampling_qualified_summaries_are_keyed_by_type_and_id() -> None:
    source = Path("src/live_mcp/orchestrator.py").read_text()
    probe_block = source[
        source.index("def _probe_live_sampling_context"):source.index("def _filter_feasible_chains")
    ]
    assert "entity_key_to_summary" in probe_block
    assert '(str(q.get("type", "")), str(q.get("id", "")))' in probe_block


def test_chain_feasibility_uses_handler_relevant_entity_records() -> None:
    filesystem_context = {
        "qualified_entity_ids": [{"type": "file", "id": "/home/user"}],
        "qualified_entity_records": [{
            "type": "file", "id": "/home/user", "data": {"type": "directory"},
        }],
    }
    ok, _ = _chain_is_feasible(
        ["ls", "readlink"], "filesystem", filesystem_context,
    )
    assert not ok

    assert not _entity_record_satisfies_chain(
        server_name="filesystem", chain_seed=["ls", "tar_extract"],
        etype="file", record={"type": "directory"},
    )
    assert _entity_record_satisfies_chain(
        server_name="filesystem", chain_seed=["ls", "tar_extract"],
        etype="file", record={"type": "file"},
    )


def test_payment_entity_filter_matches_dispute_handler_statuses() -> None:
    assert not _entity_record_satisfies_chain(
        server_name="payments", chain_seed=["list_invoices", "dispute_invoice"],
        etype="invoice", record={"status": "overdue"},
    )
    assert _entity_record_satisfies_chain(
        server_name="payments", chain_seed=["list_invoices", "dispute_invoice"],
        etype="invoice", record={"status": "pending"},
    )


def test_opaque_banking_selector_preserves_type_owner_and_frozen_state() -> None:
    live_context = {
        "qualified_entity_ids": [{"type": "account", "id": "acc_1"}],
        "qualified_entity_records": [{
            "type": "account", "id": "acc_1", "data": {
                "type": "savings", "owner": "Carol White",
                "balance": 1453.0, "currency": "USD", "frozen": True,
            },
        }],
        "entity_ids": [{"type": "account", "id": "acc_1"}],
        "entity_summaries": ["acc_1 (account): savings frozen"],
        "entity_records": [{
            "type": "account", "id": "acc_1", "data": {
                "type": "savings", "owner": "Carol White",
                "balance": 1453.0, "currency": "USD", "frozen": True,
            },
        }],
    }
    context = _extract_chain_context(
        ["list_accounts", "unfreeze_account", "bill_pay"],
        "banking", live_context,
    )
    rendered = "\n".join(context["query_grounding_summaries"])
    assert "'type': 'savings'" in rendered
    assert "'owner': 'Carol White'" in rendered
    assert "'frozen': True" in rendered


def test_payment_chain_context_exposes_only_legal_typed_entities_and_amounts() -> None:
    live_context = {
        "qualified_entity_ids": [
            {"type": "invoice", "id": "inv_paid"},
            {"type": "invoice", "id": "inv_pending"},
            {"type": "payment", "id": "pay_pending"},
        ],
        "qualified_entity_records": [
            {"type": "invoice", "id": "inv_paid", "data": {
                "status": "partially_refunded", "amount": 100.0,
                "total_refunded": 25.0, "payment_id": "pay_settled",
                "payment_status": "settled",
            }},
            {"type": "invoice", "id": "inv_pending", "data": {
                "status": "pending", "amount": 90.0,
            }},
            {"type": "payment", "id": "pay_pending", "data": {
                "status": "pending", "amount": 90.0,
                "invoice_id": "inv_pending",
            }},
        ],
        "entity_ids": [
            {"type": "invoice", "id": "inv_paid"},
            {"type": "invoice", "id": "inv_pending"},
            {"type": "payment", "id": "pay_pending"},
        ],
        "entity_records": [
            {"type": "invoice", "id": "inv_paid", "data": {
                "status": "partially_refunded", "amount": 100.0,
                "total_refunded": 25.0, "payment_id": "pay_settled",
                "payment_status": "settled",
            }},
            {"type": "invoice", "id": "inv_pending", "data": {
                "status": "pending", "amount": 90.0,
            }},
            {"type": "payment", "id": "pay_pending", "data": {
                "status": "pending", "amount": 90.0,
                "invoice_id": "inv_pending",
            }},
        ],
        "entity_summaries": [
            "  inv_paid (invoice): {'status': 'partially_refunded', 'amount': 100.0}",
            "  inv_pending (invoice): {'status': 'pending', 'amount': 90.0}",
            "  pay_pending (payment): {'status': 'pending', 'amount': 90.0}",
        ],
    }
    refund_context = _extract_chain_context(
        ["list_invoices", "refund_invoice"], "payments", live_context,
    )
    assert refund_context["entity_ids"] == [{"id": "inv_paid", "type": "invoice"}]
    assert "remaining_refundable=75.0" in refund_context["entity_summaries"][0]
    assert refund_context["query_visible_entity_ids"] == []
    assert refund_context["opaque_id_hidden_types"] == ["invoice"]
    assert "inv_paid" not in refund_context["query_grounding_summaries"][0]
    assert "partially_refunded" in refund_context["query_grounding_summaries"][0]

    cancel_context = _extract_chain_context(
        ["cancel_payment"], "payments", live_context,
    )
    assert cancel_context["entity_ids"] == [{"id": "pay_pending", "type": "payment"}]
    assert "resource_type=payment" in cancel_context["entity_summaries"][0]
    assert cancel_context["query_visible_entity_ids"] == [
        {"id": "pay_pending", "type": "payment"}
    ]


def test_chain_context_summary_survives_prompt_state_formatting() -> None:
    context = {
        "entity_ids": [{"type": "invoice", "id": "inv_1"}],
        "entity_summaries": ["inv_1 (invoice): status=paid; payable_amount=100"],
    }
    prompt_state = _live_context_to_prompt_state(context)
    rendered = _format_state_compact(prompt_state)
    assert "status=paid" in rendered
    assert "payable_amount=100" in rendered


def test_initial_query_uses_chain_specific_grounding_state() -> None:
    source = Path("src/live_mcp/orchestrator.py").read_text()
    block = source[source.index("query_chain_context ="):source.index("conversation_fsm.transition(")]
    assert "query_grounding_state = _live_context_to_prompt_state" in block
    assert "grounded_state=query_grounding_state" in block
    assert "grounded_state=teacher_grounding_state" not in block


def test_action_teacher_does_not_receive_sampler_private_ids() -> None:
    context = _teacher_public_action_context({
        "entity_ids": [
            {"id": "inv_private", "type": "invoice"},
            {"id": "pay_public", "type": "payment"},
        ],
        "entity_summaries": [
            "inv_private (invoice): status=paid; amount=50",
            "pay_public (payment): status=pending; amount=50",
        ],
    }, "cancel pay_public")
    rendered = "\n".join(context["entity_summaries"])
    assert "inv_private" not in rendered
    assert "<hidden-invoice-id>" in rendered
    assert "pay_public" in rendered
    assert "status=paid" in rendered


def test_food_chain_context_exposes_only_handler_legal_next_status() -> None:
    context = _extract_chain_context(
        ["list_orders", "update_order_status"],
        "food_delivery",
        {
            "qualified_entity_ids": [{"type": "order", "id": "ord_1"}],
            "qualified_entity_records": [{
                "type": "order", "id": "ord_1", "data": {"status": "preparing"},
            }],
            "entity_ids": [{"type": "order", "id": "ord_1"}],
            "entity_records": [{
                "type": "order", "id": "ord_1", "data": {"status": "preparing"},
            }],
            "entity_summaries": ["  ord_1 (order): {'status': 'preparing'}"],
        },
    )
    assert "allowed_next_status=['delivering']" in context["entity_summaries"][0]


def test_creator_chain_does_not_ground_downstream_on_existing_entity() -> None:
    context = _extract_chain_context(
        ["create_issue", "create_subtask"],
        "issue_tracker",
        {
            "qualified_entity_ids": [{"type": "issue", "id": "iss_existing"}],
            "qualified_entity_records": [{
                "type": "issue", "id": "iss_existing", "data": {"state": "open"},
            }],
            "entity_ids": [{"type": "issue", "id": "iss_existing"}],
            "entity_summaries": ["iss_existing (issue)"],
            "entity_records": [{
                "type": "issue", "id": "iss_existing", "data": {"state": "open"},
            }],
        },
    )
    assert context["entity_ids"] == []


def test_food_create_order_schema_matches_handler_item_contract() -> None:
    schema = next(t for t in FOOD_DELIVERY_TOOLS if t["name"] == "create_order")
    item_schema = schema["input_schema"]["properties"]["items"]["items"]
    assert item_schema["required"] == ["name", "quantity"]
    assert item_schema["properties"]["quantity"]["minimum"] == 1
    rendered = _format_tools([schema])
    assert "items* (array<object{name*: string, quantity*: integer(minimum=1)}>)" in rendered


def test_banking_entity_filter_is_chain_specific() -> None:
    frozen = {"balance": 100, "frozen": True}
    active = {"balance": 100, "frozen": False}
    assert _entity_record_satisfies_chain(
        server_name="banking", chain_seed=["list_accounts", "unfreeze_account"],
        etype="account", record=frozen,
    )
    assert not _entity_record_satisfies_chain(
        server_name="banking", chain_seed=["list_accounts", "unfreeze_account"],
        etype="account", record=active,
    )
    assert not _entity_record_satisfies_chain(
        server_name="banking", chain_seed=["list_accounts", "withdraw"],
        etype="account", record=frozen,
    )


def test_generate_many_stops_submitting_after_domain_quota(monkeypatch) -> None:
    orchestrator = TaskOrchestrator.__new__(TaskOrchestrator)
    orchestrator.manager = SimpleNamespace(server_names=["calendar"])
    submitted: list[int] = []

    def generate(_server, task_seed, *_args):
        submitted.append(task_seed)
        return SimpleNamespace(
            user_prompt=f"query-{task_seed}",
            metadata={"paper_replay_valid": True, "project_outcome_valid": True},
        )

    orchestrator._generate_task_with_postprocess = generate
    orchestrator._generate_irrelevant_tasks = lambda *_args: []
    monkeypatch.setenv("LIVEMCP_GENERATION_MAX_WORKERS", "1")

    tasks = orchestrator.generate_many(
        "calendar", count=2, seed=100, irrelevance_ratio=0,
    )
    assert len(tasks) == 2
    assert len(submitted) == 2


def test_same_name_tool_requires_domain_when_schemas_are_ambiguous() -> None:
    registry = SchemaRegistry()
    schema = {
        "name": "get_thread",
        "input_schema": {
            "type": "object",
            "properties": {"thread_id": {"type": "string"}},
            "required": ["thread_id"],
        },
    }
    registry.register_tools("email", [schema])
    registry.register_tools("team_chat", [schema])
    args = {"thread_id": "thd_001"}

    assert registry.get_schema("get_thread") is None
    assert registry.get_schema("get_thread", domain="email") == schema
    assert registry.server_for_tool("get_thread", args) is None
    assert registry.server_for_tool("get_thread", args, domain="email") == "email"


def test_chain_feasibility_checks_real_entity_cardinality() -> None:
    one_account = {"qualified_entity_ids": [{"type": "account", "id": "acc_1"}]}
    two_accounts = {"qualified_entity_ids": [
        {"type": "account", "id": "acc_1"},
        {"type": "account", "id": "acc_2"},
    ]}
    ok, reason = _chain_is_feasible(["transfer"], "banking", one_account)
    assert not ok and "account(1/2)" in reason
    assert _chain_is_feasible(["transfer"], "banking", two_accounts)[0]

    one_file = {"qualified_entity_ids": [{"type": "file", "id": "/a.txt"}]}
    assert not _chain_is_feasible(["join"], "filesystem", one_file)[0]
    # touch creates the second file required by join; this remains feasible.
    assert _chain_is_feasible(["touch", "join"], "filesystem", one_file)[0]


def test_state_filters_reject_only_handler_proven_invalid_entities() -> None:
    assert not _entity_record_satisfies_chain(
        server_name="calendar", chain_seed=["respond_to_event"], etype="event",
        record={"attendees": []},
    )
    assert _entity_record_satisfies_chain(
        server_name="calendar", chain_seed=["add_attendee", "respond_to_event"],
        etype="event", record={"attendees": []},
    )
    assert not _entity_record_satisfies_chain(
        server_name="email", chain_seed=["mark_read"], etype="email",
        record={"read": True},
    )
    assert _entity_record_satisfies_chain(
        server_name="email", chain_seed=["mark_unread"], etype="email",
        record={"read": True},
    )
    assert not _entity_record_satisfies_chain(
        server_name="crm", chain_seed=["complete_task"], etype="task",
        record={"status": "completed"},
    )
    # Missing probe fields are unknown, not evidence of an invalid entity.
    assert _entity_record_satisfies_chain(
        server_name="calendar", chain_seed=["respond_to_event"], etype="event",
        record={"title": "Sync"},
    )
    assert _entity_record_satisfies_chain(
        server_name="email", chain_seed=["mark_unread"], etype="email",
        record={"subject": "Hello"},
    )
    assert not _entity_record_satisfies_chain(
        server_name="team_chat", chain_seed=["send_message"], etype="channel",
        record={"archived": True},
    )
    assert _entity_record_satisfies_chain(
        server_name="team_chat", chain_seed=["get_channel"], etype="channel",
        record={"archived": True},
    )


def test_chain_entity_summary_keeps_required_supporting_data() -> None:
    summary = _format_entity_summary(
        "evt_1",
        "event",
        {
            "title": "Project sync",
            "attendees": ["alice@example.com"],
            "labels": ["urgent"],
            "watchers": ["usr_1"],
        },
    )
    assert "alice@example.com" in summary
    assert "urgent" in summary
    assert "usr_1" in summary

    relational = _format_entity_summary(
        "msg_1",
        "message",
        {
            "content": "Ship the release",
            "channel_id": "ch_engineering",
            "thread_id": "th_1",
            "created_at": "2026-06-19T09:00:00",
        },
    )
    assert "ch_engineering" in relational
    assert "th_1" in relational
    assert "2026-06-19" in relational

    calendar_outcome = _format_entity_summary(
        "evt_021",
        "event",
        {
            "title": "Performance Review",
            "start_time": "2026-01-16T10:00:00Z",
            "end_time": "2026-01-16T11:00:00Z",
            "location": "Conference Room B",
            "reminders": [
                {"id": "rem_1", "minutes_before": 15, "method": "popup"}
            ],
        },
    )
    assert "start_time" in calendar_outcome
    assert "end_time" in calendar_outcome
    assert "Conference Room B" in calendar_outcome
    assert "minutes_before" in calendar_outcome


def test_domain_entity_projection_keeps_handler_decision_fields() -> None:
    cases = [
        ("filesystem", "file", {"permissions": "755", "size": 12}, ("permissions", "size")),
        ("email", "email", {"read": False, "archived": True}, ("read", "archived")),
        ("issue_tracker", "issue", {"state": "in_progress", "assignee": "usr_1", "sprint_id": "spr_1"}, ("state", "assignee", "sprint_id")),
        ("team_chat", "message", {"reactions": ["👍"]}, ("reactions", "👍")),
        ("food_delivery", "order", {"tip": 5.0, "total": 20.0}, ("tip", "5.0")),
        ("payments", "invoice", {"amount": 100.0, "total_refunded": 40.0}, ("total_refunded", "remaining_refundable", "60.0")),
    ]
    for domain, etype, record, expected in cases:
        summary = _format_entity_summary(
            "entity_1", etype, record, server_name=domain,
        )
        for marker in expected:
            assert marker in summary, (domain, marker, summary)


def test_shopping_list_orders_preserves_primary_order_record() -> None:
    extracted = []

    def add_entity(eid, etype, data=None):
        extracted.append((eid, etype, data))

    order = {
        "order_id": "ord_1",
        "status": "shipped",
        "total": 82,
        "created_at": "2026-06-18",
    }
    _extract_probe_entities(
        {"orders": [order]}, add_entity,
        server_name="shopping", tool_name="list_orders",
    )
    assert extracted == [("ord_1", "order", order)]


def test_food_seed_dates_share_query_reference_anchor() -> None:
    from src.live_mcp.state_seeder import _food_delivery_state
    from src.live_mcp.task_planner import reference_datetime_for_seed

    # This seed selects Saturday, June 20, 2026 in the shared schedule.
    seed = len(_PERSONA_TEMPLATES) * 2
    reference = reference_datetime_for_seed(seed)
    orders = list(_food_delivery_state(seed)["orders"].values())
    created = [
        _datetime.datetime.fromisoformat(order["created_at"])
        for order in orders
    ]
    assert all(value <= reference.replace(hour=23, minute=59) for value in created)
    assert any(value.weekday() == 4 and value.date() < reference.date() for value in created)

    server = FoodDeliveryServer()
    server.handle_request("session/reset", {"session_id": "food-clock", "seed": seed})
    state = server.sessions["food-clock"]
    restaurant = next(iter(state["restaurants"].values()))
    result = server.create_order("food-clock", {
        "restaurant_id": restaurant["restaurant_id"],
        "items": [{"name": restaurant["menu"][0]["name"], "quantity": 1}],
        "delivery_address": "123 Main St",
    })
    assert result["observation"]["order"]["created_at"].startswith(
        reference.date().isoformat()
    )


def test_teacher_visible_schemas_expose_observed_handler_preconditions() -> None:
    def tool(tools, name):
        return next(item for item in tools if item["name"] == name)

    deal = tool(CRM_TOOLS, "create_deal")
    amount_schema = deal["input_schema"]["properties"]["amount"]
    assert amount_schema["exclusiveMinimum"] == 0
    assert "greater than zero" in deal["description"]
    update_deal_amount = tool(CRM_TOOLS, "update_deal")["input_schema"]["properties"]["amount"]
    assert update_deal_amount["exclusiveMinimum"] == 0
    assert "converted lead" in tool(CRM_TOOLS, "delete_contact")["description"]

    invoice_amount = tool(PAYMENTS_TOOLS, "create_invoice")["input_schema"]["properties"]["amount"]
    assert invoice_amount["exclusiveMinimum"] == 0

    rider = tool(FOOD_DELIVERY_TOOLS, "track_rider")
    assert "delivering" in rider["description"]
    status = tool(FOOD_DELIVERY_TOOLS, "update_order_status")
    assert "preparing→delivering" in status["description"]
    assert "in_transit" not in status["input_schema"]["properties"]["status"]["enum"]
    list_orders = tool(FOOD_DELIVERY_TOOLS, "list_orders")
    order_status = list_orders["input_schema"]["properties"]["status"]
    assert set(order_status["enum"]) == {
        "placed", "confirmed", "preparing", "delivering", "delivered", "cancelled",
    }
    assert "in_transit is not a valid status" in order_status["description"]
    rating = tool(FOOD_DELIVERY_TOOLS, "rate_order")["input_schema"]["properties"]["rating"]
    assert rating["minimum"] == 1
    assert rating["maximum"] == 5
    tip_amount = tool(FOOD_DELIVERY_TOOLS, "add_tip")["input_schema"]["properties"]["amount"]
    assert tip_amount["exclusiveMinimum"] == 0

    response = tool(CALENDAR_TOOLS, "respond_to_event")
    assert "current attendees" in response["description"]
    assert "Exact attendee email" in response["input_schema"]["properties"]["email"]["description"]

    dm = tool(TEAM_CHAT_TOOLS, "send_dm")
    assert "existing team member" in dm["description"]
    assert "not an invented display name" in dm["input_schema"]["properties"]["recipient"]["description"]

    channels = tool(TEAM_CHAT_TOOLS, "list_channels")
    assert "archived channels are omitted" in channels["description"]

    move = tool(FILESYSTEM_TOOLS, "mv")
    assert "must not already exist" in move["description"]


def test_distractors_are_visible_to_teacher_generation() -> None:
    domain_tool = {
        "name": "get_event",
        "input_schema": {"properties": {"kind": {"type": "string", "enum": ["a"]}}},
    }
    distractor = {"name": "transfer", "input_schema": {"properties": {}}}
    plan = RobustnessPlan(
        inject_distractors=True,
        distractor_tools=[distractor],
        strip_enums=True,
    )
    visible = _build_teacher_visible_tools([domain_tool], plan)
    assert [tool["name"] for tool in visible] == ["get_event", "transfer"]
    assert "enum" not in visible[0]["input_schema"]["properties"]["kind"]


def test_robustness_plan_sample_remains_class_api() -> None:
    plan = RobustnessPlan.sample(
        seed=1,
        all_tools_pool=[],
        domain_tools=[],
        distractor_rate=0.0,
        strip_enums_rate=0.0,
        missing_function_rate=1.0,
    )
    assert plan.missing_function is True


def test_launcher_skips_row_spotcheck_for_explicitly_empty_split() -> None:
    source = Path("scripts/generate_data.sh").read_text()
    empty_guard = source.index("if df.empty:")
    spotcheck = source.index("# Production readback: every row")
    assert empty_guard < spotcheck


def test_launcher_does_not_spawn_zero_quota_shards() -> None:
    source = Path("scripts/generate_data.sh").read_text()
    assert source.count('if [ $((SHARD_TRAIN + SHARD_VAL)) -eq 0 ]; then') == 2
    assert 'Waiting for ${ACTIVE_GEN_CLIENTS} active generation processes...' in source
    assert 'ERROR: ${FAILED}/${ACTIVE_GEN_CLIENTS} active generation processes failed' in source


def test_enum_stripping_removes_nested_enums() -> None:
    tool = {
        "name": "create_order",
        "input_schema": {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "size": {"type": "string", "enum": ["s", "m"]},
                        },
                    },
                },
            },
        },
    }
    visible = _build_teacher_visible_tools(
        [tool], RobustnessPlan(strip_enums=True),
    )
    size_schema = visible[0]["input_schema"]["properties"]["items"]["items"]["properties"]["size"]
    assert "enum" not in size_schema


def test_provenance_rejects_value_from_future_user_round() -> None:
    from src.live_mcp.servers.banking.server import TOOLS as BANKING_TOOLS

    call = OracleCall("unfreeze_account", {
        "account_id": "acc_s42_001", "authorization_code": "secret-123",
    })
    passed, violations = provenance_check(
        [call],
        user_query="connect account acc_s42_001",
        aligned_observations=[{}],
        tool_schemas=BANKING_TOOLS,
        domain="banking",
        user_queries=["connect account acc_s42_001", "use token secret-123"],
        call_round_indices=[0],
    )
    assert not passed
    assert any(v["param"] == "authorization_code" for v in violations)

    passed, violations = provenance_check(
        [call],
        user_query="connect account acc_s42_001",
        aligned_observations=[{}],
        tool_schemas=BANKING_TOOLS,
        domain="banking",
        user_queries=["connect account acc_s42_001", "use token secret-123"],
        call_round_indices=[1],
    )
    assert passed
    assert violations == []


def test_provenance_rejects_hallucinated_schema_sensitive_amount() -> None:
    from src.live_mcp.servers.banking.server import TOOLS as BANKING_TOOLS

    call = OracleCall("transfer", {
        "from_account": "acc_s42_001",
        "to_account": "acc_s42_002",
        "amount": 987.65,
    })
    passed, violations = provenance_check(
        [call],
        user_query="move money from acc_s42_001 to acc_s42_002",
        aligned_observations=[{}],
        tool_schemas=BANKING_TOOLS,
        domain="banking",
    )

    assert not passed
    assert any(v["param"] == "amount" for v in violations)


def test_success_criteria_cover_top_level_scalars_none_and_deletion() -> None:
    criteria = derive_success_criteria(
        initial_state={"timezone": "UTC", "coupon": "SAVE10", "umask": "022"},
        final_state={"timezone": "Asia/Shanghai", "coupon": None},
        oracle_calls=[],
        domain="calendar",
    )
    indexed = {(item["type"], item["path"]): item for item in criteria}

    assert indexed[("state_equals", "timezone")]["value"] == "Asia/Shanghai"
    assert indexed[("state_equals", "coupon")]["value"] is None
    assert ("state_absent", "umask") in indexed


def test_merge_does_not_prune_recovery_calls_within_a_trace() -> None:
    row = _row("repeat after follow-up", "round-repeat")
    call = {"tool_name": "get_event", "arguments": {"event_id": "evt_1"}, "action": "tool_call"}
    terminal = {"tool_name": "final_answer", "arguments": {"text": "done"}, "action": "final_answer"}

    same_round = pd.Series(row.copy())
    same_round["extra_info"] = dict(row["extra_info"])
    same_round["extra_info"]["oracle_calls"] = json.dumps([call, call, terminal])
    same_round["extra_info"]["round_contracts"] = json.dumps([{
        "round_idx": 0,
        "required_tools": ["get_event", "get_event"],
        "allowed_terminal_actions": ["final_answer"],
    }])
    assert _quality_issue(same_round) == ""

    cross_round = pd.Series(row.copy())
    cross_round["extra_info"] = dict(row["extra_info"])
    cross_round["extra_info"]["oracle_calls"] = json.dumps([call, call, terminal])
    cross_round["extra_info"]["round_contracts"] = json.dumps([
        {
            "round_idx": 0,
            "required_tools": ["get_event"],
            "allowed_terminal_actions": ["final_answer"],
        },
        {
            "round_idx": 1,
            "required_tools": ["get_event"],
            "allowed_terminal_actions": ["final_answer"],
        },
    ])
    assert _quality_issue(cross_round) == ""


def test_scenario_terminal_label_is_not_a_prove_rejection_gate() -> None:
    row = _row("recover gracefully", "graceful")
    row["extra_info"]["oracle_calls"] = json.dumps([{
        "tool_name": "get_event_recover gracefully",
        "arguments": {"event_id": "recover gracefully"},
        "action": "tool_call",
    }, {
        "tool_name": "report_error",
        "arguments": {"text": "The service could not complete the request."},
        "action": "report_error",
    }])
    row["extra_info"]["allowed_terminal_actions"] = ["report_error"]
    row["extra_info"]["round_contracts"] = json.dumps([{
        "round_idx": 0,
        "required_tools": ["get_event_recover gracefully"],
        "allowed_terminal_actions": ["report_error"],
    }])
    row["scenario_type"] = "normal_safe_success"
    assert _quality_issue(pd.Series(row)) == ""


def test_global_jaccard_dedup_uses_tool_order_and_domain_identity() -> None:
    def with_sequence(query: str, domain: str, sequence: list[str]) -> dict:
        row = _row(query, query)
        row["extra_info"]["domain"] = domain
        calls = [
            {"tool_name": name, "arguments": {}, "action": "tool_call"}
            for name in sequence
        ] + [{"tool_name": "final_answer", "arguments": {}, "action": "final_answer"}]
        row["extra_info"]["oracle_calls"] = json.dumps(calls)
        return row

    df = pd.DataFrame([
        with_sequence("a", "calendar", ["list_events", "get_event"]),
        with_sequence("b", "calendar", ["list_events", "get_event"]),
        with_sequence("c", "calendar", ["get_event", "list_events"]),
        with_sequence("d", "email", ["list_events", "get_event"]),
    ])
    unique, removed = _dedup_jaccard(df, threshold=0.70)
    assert removed == 1
    assert list(unique["uid"]) == ["a", "c", "d"]


def test_context_boundary_budget_preserves_input_and_reduces_decode_once() -> None:
    error = RuntimeError(
        "This model's maximum context length is 8192 tokens. However, you "
        "requested 384 output tokens and your prompt contains at least 7809 "
        "input tokens, for a total of at least 8193 tokens."
    )
    assert _remaining_context_output_budget(error, 384) == 383


def test_context_boundary_budget_fails_closed_for_tiny_or_unrelated_errors() -> None:
    tiny = RuntimeError(
        "maximum context length is 8192 tokens; requested 384 output tokens; "
        "prompt contains at least 8160 input tokens"
    )
    assert _remaining_context_output_budget(tiny, 384) is None
    assert _remaining_context_output_budget(RuntimeError("connection reset"), 384) is None


def test_openai_client_retries_context_error_with_remaining_budget() -> None:
    calls: list[int] = []

    class _Completions:
        def create(self, **kwargs):
            calls.append(kwargs["max_tokens"])
            if len(calls) == 1:
                raise RuntimeError(
                    "maximum context length is 8192 tokens; requested 384 "
                    "output tokens; prompt contains at least 7809 input tokens"
                )
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content='{"action":"final_answer"}'))]
            )

    client = LLMClient(mode="openai", model_path="Gemma-4-31B-it")
    client._client = SimpleNamespace(
        chat=SimpleNamespace(completions=_Completions())
    )
    result = client.generate_chat(
        [{"role": "user", "content": "test"}],
        max_tokens=384,
    )
    assert result == '{"action":"final_answer"}'
    assert calls == [384, 383]
