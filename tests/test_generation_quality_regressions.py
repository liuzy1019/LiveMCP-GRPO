from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from scripts.generate_data import (
    _domain_quotas,
    _build_round_contracts,
    _domain_recovery_requests,
    _filter_training_eligible_tasks,
    _is_zero_yield_error,
    _load_generation_checkpoint,
    _minimum_action_budget,
    _stratified_task_split,
    _tasks_to_rows,
    _validate_task_training_contract,
    _write_generation_checkpoint,
)
from scripts.merge_rollout_shards import (
    _dedup_jaccard, _quality_issue, _row_fingerprint, merge_shards,
)
from scripts.validate_pipeline import _stage3_output_issue
from src.live_mcp.orchestrator import (
    TaskOrchestrator,
    RobustnessPlan,
    _CREATED_ENTITY_BY_TOOL,
    _build_teacher_visible_tools,
    _chain_is_feasible,
    _chain_respects_state_preconditions,
    _detect_missing_dependency,
    _entity_record_satisfies_chain,
    _extract_chain_context,
    _tool_existing_entity_requirements,
    _classify_scenario,
    _compact_sampling_context,
)
from src.live_mcp.schema_registry import SchemaRegistry
from src.live_mcp.servers.food_delivery.server import TOOLS as FOOD_DELIVERY_TOOLS
from src.live_mcp.task_planner import (
    ActionPlan, TaskPlanner, derive_success_criteria, provenance_check, replay_validate,
)
from src.live_mcp.task_planner import _format_tools
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
        sampling_context={}, max_turns=2,
        metadata={
            "scenario_type": "normal_safe_success",
            "generation_mode": "chain_seeded",
            "paper_replay_valid": True,
        },
        conversation_queries=["find the event", "now update it"],
        oracle_calls_per_round=[round_0, round_1],
        execution_history_per_round=[[], []],
    )
    row = _tasks_to_rows([task], 1)[0]
    assert row["extra_info"]["minimum_action_budget"] == 4
    assert row["extra_info"]["budget"] == 4


def test_completed_dependency_chain_reaches_prove_continuation_decision() -> None:
    source = Path("src/live_mcp/orchestrator.py").read_text()
    continuation_block = source[source.index("# PROVE §3.2 continuation"):]
    continuation_block = continuation_block[:continuation_block.index("# If we broke out")]
    assert "sample_continuation_decision" in continuation_block
    assert "completed_chain_progress >= len(chain_seed)" not in continuation_block
    followup_block = source[source.index("current_query = teacher.generate_followup("):]
    followup_block = followup_block[:followup_block.index("conversation_queries.append")]
    assert "chain_seed=None" in followup_block


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
            return '{"user_query": "copy the file into the new complaints folder"}'

    client = CapturingClient()
    planner = TaskPlanner(client, "filesystem", seed=1)
    query = planner.generate_query(
        tool_schemas=[
            {"name": "mkdir", "description": "create a directory", "inputSchema": {}},
            {"name": "cp", "description": "copy a file", "inputSchema": {}},
        ],
        grounded_state={},
        difficulty="complete",
        rng=__import__("random").Random(1),
        chain_seed=["mkdir", "cp"],
    )
    assert query.startswith("copy the file")
    prompt = client.messages[1]["content"]
    assert "['mkdir', 'cp']" in prompt
    assert "final outcome" in prompt
    assert "copy a file" in prompt


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


def test_outcome_criteria_are_diagnostic_not_a_prove_filter() -> None:
    task = SimpleNamespace(
        task_id="shopping_bad_outcome",
        metadata={"project_outcome_valid": False, "paper_replay_valid": True},
        oracle_program=SimpleNamespace(
            calls=[
                OracleCall("report_error", {"text": "unavailable"}, action="report_error"),
            ],
            success_criteria=[],
        ),
        scenario_type="irrelevant",
        task_type="irrelevant",
        conversation_queries=["do something unsupported"],
        oracle_calls_per_round=[],
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


def test_followup_prompt_contains_previous_visible_response() -> None:
    class CapturingClient:
        messages = None

        def generate_chat(self, messages, **kwargs):
            self.messages = messages
            return json.dumps({"user_query": "Cancel the new order instead."})

    client = CapturingClient()
    planner = TaskPlanner(client, "food_delivery", seed=1)
    planner.generate_followup(
        tool_schemas=[], grounded_state={}, previous_query="Reorder ord_1",
        difficulty="complete", rng=__import__("random").Random(1),
        previous_response="Your new order is ord_3.",
    )
    prompt = client.messages[1]["content"]
    assert "Assistant reply you just received" in prompt
    assert "Your new order is ord_3." in prompt


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

        def decide_action(self, **kwargs):
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
            )

    orchestrator = TaskOrchestrator.__new__(TaskOrchestrator)
    orchestrator.executor = FakeExecutor()
    calls, history, *_ = orchestrator._run_turn_loop(
        teacher=SequencedTeacher(), current_query="what is today",
        server_tools=[{"name": "list_events"}],
        server_name="calendar", session_id="s", difficulty="complete",
        round_idx=0,
    )
    assert orchestrator.executor.calls == 2
    assert [call.tool_name for call in calls] == [
        "list_events", "list_events", "final_answer",
    ]
    assert [item["execution_status"] for item in history] == ["SUCCESS", "SUCCESS"]


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
        "allowed_terminal_actions": ["ask_clarification"],
    })

    assert task["required_tool_calls"] == [{
        "tool_name": "get_event",
        "arguments": {"event_id": "evt_1"},
    }]


def test_clarification_terminal_controls_scenario_even_with_prefix_calls() -> None:
    scenario = _classify_scenario(
        server_name="payments",
        oracle_calls=[OracleCall("list_invoices", {"status": "pending"})],
        execution_history=[],
        terminal_action="ask_clarification",
        seed=1,
    )
    assert scenario == "clarification_required"


def test_merge_does_not_filter_outcome_diagnostic_false_values() -> None:
    row = pd.Series(_row("q-bad", "t-bad"))
    row["extra_info"]["project_outcome_valid"] = pd.array([False], dtype="boolean")[0]
    assert _quality_issue(row) == ""


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
            "domain": domain,
            "user_query": query,
            "oracle_calls": json.dumps(oracle),
            "task_id": task_id,
            "hidden_tools": [],
            "visible_tool_names": [f"get_event_{query}"],
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

    assert merge_shards(
        shard_dir, output_dir, count=2, val_count=2,
        domains=["calendar", "filesystem"],
    ) == 1


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


def test_recovery_zero_yield_does_not_abort_other_domains() -> None:
    assert _is_zero_yield_error(RuntimeError(
        "generate_many produced 0 tasks (target 2, failures=2)"
    ))
    assert not _is_zero_yield_error(RuntimeError("MCP session crashed"))


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
            "scenario_type": "normal_safe_success",
            "generation_mode": "chain_seeded",
            "chain_seed": [],
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
    call = OracleCall("authenticate", {"api_token": "secret-123"})
    passed, violations = provenance_check(
        [call],
        user_query="connect my account",
        aligned_observations=[{}],
        user_queries=["connect my account", "use token secret-123"],
        call_round_indices=[0],
    )
    assert not passed
    assert violations[0]["param"] == "api_token"

    passed, violations = provenance_check(
        [call],
        user_query="connect my account",
        aligned_observations=[{}],
        user_queries=["connect my account", "use token secret-123"],
        call_round_indices=[1],
    )
    assert passed
    assert violations == []


def test_merge_does_not_prune_recovery_calls_within_a_trace() -> None:
    row = _row("repeat after follow-up", "round-repeat")
    call = {"tool_name": "get_event", "arguments": {"event_id": "evt_1"}, "action": "tool_call"}
    terminal = {"tool_name": "final_answer", "arguments": {"text": "done"}, "action": "final_answer"}

    same_round = pd.Series(row.copy())
    same_round["extra_info"] = dict(row["extra_info"])
    same_round["extra_info"]["oracle_calls"] = json.dumps([call, call, terminal])
    assert _quality_issue(same_round) == ""

    cross_round = pd.Series(row.copy())
    cross_round["extra_info"] = dict(row["extra_info"])
    cross_round["extra_info"]["oracle_calls"] = json.dumps([call, terminal, call, terminal])
    assert _quality_issue(cross_round) == ""


def test_scenario_terminal_label_is_not_a_prove_rejection_gate() -> None:
    row = _row("recover gracefully", "graceful")
    row["extra_info"]["oracle_calls"] = json.dumps([{
        "tool_name": "report_error",
        "arguments": {"text": "The service could not complete the request."},
        "action": "report_error",
    }])
    row["scenario_type"] = "normal_safe_success"
    assert _quality_issue(pd.Series(row)) == ""


def test_global_jaccard_dedup_uses_tool_order_without_domain_exemption() -> None:
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
    assert removed == 2
    assert list(unique["uid"]) == ["a", "c"]
