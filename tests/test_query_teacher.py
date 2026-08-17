from __future__ import annotations

import json
import random

import pytest

from src.live_mcp.generation.query_teacher import QueryGenerationError
from src.live_mcp.fsm import RobustnessPlan
from src.live_mcp.generation.robustness import bind_missing_function_contract
from src.live_mcp.generation.turn_loop import run_turn_loop
from src.live_mcp.generation.teacher_contracts import ActionPlan
from src.live_mcp.errors import (
    ActionDecisionError,
    CandidateGenerationError,
    TurnLoopError,
)
from src.live_mcp.prompt_profiles import PromptProfile
from src.live_mcp.task_planner import TaskPlanner, _resolve_teacher_trace_path
from src.live_mcp.generation.teacher_contracts import _chain_goal_phrase
from src.live_mcp.servers.food_delivery.server import TOOLS as FOOD_TOOLS
from src.live_mcp.servers.shopping.server import TOOLS as SHOPPING_TOOLS
from src.live_mcp.servers.calendar.server import TOOLS as CALENDAR_TOOLS
from src.live_mcp.servers.email.server import TOOLS as EMAIL_TOOLS
from src.live_mcp.servers.issue_tracker.server import TOOLS as ISSUE_TOOLS
from src.live_mcp.servers.payments.server import TOOLS as PAYMENT_TOOLS
from src.live_mcp.types import ToolExecutionResult


def _tools() -> list[dict]:
    return [
        {
            "name": "list_items",
            "input_schema": {"type": "object", "properties": {}},
            "annotations": {"mutating": False},
        },
        {
            "name": "update_item",
            "input_schema": {
                "type": "object",
                "properties": {
                    "item_id": {"type": "string"},
                    "value": {"type": "string"},
                },
                "required": ["item_id", "value"],
            },
            "annotations": {"mutating": True},
        },
    ]


def test_query_generation_error_uses_candidate_error_taxonomy() -> None:
    assert issubclass(QueryGenerationError, CandidateGenerationError)


def test_missing_function_rejects_readonly_target_without_information_contract() -> None:
    bound, reason = bind_missing_function_contract(
        domain="calendar",
        source_chain_seed=["list_events", "get_recurring_info"],
        tool_schemas=CALENDAR_TOOLS,
        plan=RobustnessPlan(missing_function=True),
    )

    assert bound is None
    assert reason == "missing_function_readonly_capability_unproven"


def test_missing_function_rejects_mutating_prefix() -> None:
    bound, reason = bind_missing_function_contract(
        domain="payments",
        source_chain_seed=["get_invoice", "pay_invoice", "refund_invoice"],
        tool_schemas=PAYMENT_TOOLS,
        plan=RobustnessPlan(missing_function=True),
    )

    assert bound is None
    assert reason == "missing_function_mutating_prefix:pay_invoice"


def test_missing_function_binds_unique_effect_after_readonly_prefix() -> None:
    bound, reason = bind_missing_function_contract(
        domain="payments",
        source_chain_seed=["list_invoices", "dispute_invoice"],
        tool_schemas=PAYMENT_TOOLS,
        plan=RobustnessPlan(missing_function=True),
    )

    assert reason == ""
    assert bound is not None
    assert bound.hidden_tool == "dispute_invoice"
    assert any("dispute.exists" in item for item in bound.missing_function_evidence)


def test_missing_function_rejects_mutation_before_mcp_dispatch() -> None:
    class Teacher:
        def __init__(self) -> None:
            self.actions = [
                ActionPlan(
                    action="tool_call", tool_name="pay_invoice",
                    arguments={
                        "invoice_id": "inv_s42_0001", "amount": 10,
                        "method": "card",
                    },
                ),
                ActionPlan(
                    action="report_error",
                    text="The requested operation cannot be completed.",
                ),
            ]

        def decide_action(self, **_kwargs):
            return self.actions.pop(0)

    class Executor:
        def execute(self, *_args, **_kwargs):
            raise AssertionError("mutation reached MCP dispatch")

    owner = type("Owner", (), {"executor": Executor()})()
    calls, history, *_ = run_turn_loop(
        owner,
        teacher=Teacher(),
        current_query="Pay and then refund the invoice.",
        server_tools=PAYMENT_TOOLS,
        server_name="payments",
        session_id="session",
        difficulty="complete",
        round_idx=0,
        missing_function_contract=True,
    )

    assert history == []
    assert [call.action for call in calls] == ["report_error"]


def test_source_chain_rejects_unauthorized_mutation_before_dispatch() -> None:
    class Teacher:
        def __init__(self) -> None:
            self.calls = 0

        def decide_action(self, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                return ActionPlan(
                    action="tool_call", tool_name="list_invoices", arguments={},
                )
            return ActionPlan(
                action="tool_call",
                tool_name="cancel_payment",
                arguments={"payment_id": "pay_1", "reason": "requested"},
            )

    class Executor:
        dispatched: list[str] = []

        def execute(self, _session_id, call, **_kwargs):
            self.dispatched.append(call.name)
            return ToolExecutionResult(
                success=True,
                tool_name=call.name,
                canonical_tool_name=call.name,
                call_id="call",
                session_id="session",
                observation={"invoices": []},
                error_type=None,
                error_message="",
                schema_valid=True,
                state_changed=False,
                latency_ms=1,
                execution_status="SUCCESS",
            )

    executor = Executor()
    owner = type("Owner", (), {"executor": executor})()
    with pytest.raises(TurnLoopError) as exc_info:
        run_turn_loop(
            owner,
            teacher=Teacher(),
            current_query="List invoices, then cancel a payment.",
            server_tools=PAYMENT_TOOLS,
            server_name="payments",
            session_id="session",
            difficulty="complete",
            round_idx=0,
            authorized_mutating_tools={"dispute_invoice"},
        )

    assert executor.dispatched == ["list_invoices"]
    assert exc_info.value.reason == (
        "initial_round_unauthorized_mutation_repeated"
    )
    assert len(exc_info.value.details["pre_dispatch_rejections"]) == 3
    assert exc_info.value.details["tool_calls_dispatched"] == 1


def test_same_name_goal_phrase_comes_from_owner_schema() -> None:
    assert _chain_goal_phrase(FOOD_TOOLS, "cancel_order") == (
        "cancel an order (only before preparing)"
    )
    assert _chain_goal_phrase(SHOPPING_TOOLS, "cancel_order") == (
        "cancel a placed or pending order"
    )


def test_prompt_profile_contains_only_active_generation_controls() -> None:
    assert set(PromptProfile.__dataclass_fields__) == {
        "name",
        "paper_baseline",
        "policy_private",
        "natural_selector",
        "dependency_necessary",
    }


def test_teacher_trace_path_accepts_run_evidence_and_legacy_logs(tmp_path) -> None:
    run_path = tmp_path / "data" / "runs" / "run-a" / "teacher_trace.jsonl"
    log_path = tmp_path / "logs" / "legacy_trace.jsonl"

    assert _resolve_teacher_trace_path(
        str(run_path), project_root=tmp_path,
    ) == run_path
    assert _resolve_teacher_trace_path(
        str(log_path), project_root=tmp_path,
    ) == log_path


def test_teacher_trace_path_rejects_path_outside_evidence_roots(tmp_path) -> None:
    assert _resolve_teacher_trace_path(
        str(tmp_path / "README.md"), project_root=tmp_path,
    ) is None


def test_query_generation_rejects_sampler_private_id_leak() -> None:
    class Client:
        def __init__(self) -> None:
            self.calls = 0

        def generate_chat(self, _messages, **_kwargs) -> str:
            self.calls += 1
            query = (
                "Freeze account acc_s42_003."
                if self.calls == 1
                else "Freeze my only business account."
            )
            return json.dumps({
                "user_query": query,
                "target_capability": "freeze_account",
                "chain_supported": True,
                "mutation_evidence": [{
                    "capability": "freeze_account",
                    "query_span": "Freeze",
                }],
            })

    client = Client()
    planner = TaskPlanner(
        client, "banking", seed=42,
        prompt_profile="local_trainable_v1",
    )
    tools = [{
        "name": "freeze_account",
        "description": "Freeze an account.",
        "input_schema": {
            "type": "object",
            "properties": {"account_id": {"type": "string"}},
            "required": ["account_id"],
        },
        "annotations": {"mutating": True},
    }]

    generated = planner.generate_query(
        tool_schemas=tools,
        grounded_state={},
        difficulty="complete",
        rng=random.Random(42),
        chain_seed=["freeze_account"],
        chain_context={
            "entity_ids": [{"id": "acc_s42_003", "type": "account"}],
            "query_grounding_summaries": [
                "grounded account candidate: {'type': 'business'}"
            ],
            "opaque_id_hidden_types": ["account"],
        },
    )

    assert generated.user_query == "Freeze my only business account."
    assert client.calls == 2


def test_paper_query_generation_accepts_real_id_from_sampling_context() -> None:
    class Client:
        def generate_chat(self, _messages, **_kwargs) -> str:
            return json.dumps({
                "user_query": "Freeze account acc_s42_003.",
                "target_capability": "freeze_account",
                "chain_supported": True,
                "mutation_evidence": [{
                    "capability": "freeze_account",
                    "query_span": "Freeze",
                }],
            })

    tools = [{
        "name": "freeze_account",
        "description": "Freeze an account.",
        "input_schema": {
            "type": "object",
            "properties": {"account_id": {"type": "string"}},
            "required": ["account_id"],
        },
        "annotations": {"mutating": True},
    }]
    generated = TaskPlanner(
        Client(), "banking", seed=42,
        prompt_profile="paper_generation_baseline_v1",
    ).generate_query(
        tool_schemas=tools,
        grounded_state={
            "live_probe_accounts": {
                "acc_s42_003": {"id": "acc_s42_003", "type": "account"},
            },
        },
        difficulty="complete",
        rng=random.Random(42),
        chain_seed=["freeze_account"],
        chain_context={
            "entity_ids": [{"id": "acc_s42_003", "type": "account"}],
            "query_grounding_summaries": [
                "  acc_s42_003 (account): type=business"
            ],
            "opaque_id_hidden_types": ["account"],
        },
    )

    assert generated.user_query == "Freeze account acc_s42_003."


def test_query_teacher_receives_verified_causal_relations_as_internal_guide() -> None:
    class Client:
        def __init__(self) -> None:
            self.prompt = ""

        def generate_chat(self, messages, **_kwargs) -> str:
            self.prompt = "\n".join(
                message["content"] for message in messages
            )
            return json.dumps({
                "user_query": "Update the selected item to blue.",
                "target_capability": "update_item",
                "chain_supported": True,
                "mutation_evidence": [{
                    "capability": "update_item",
                    "query_span": "Update",
                }],
            })

    client = Client()
    TaskPlanner(
        client, "demo", seed=42,
        prompt_profile="local_trainable_v1",
    ).generate_query(
        tool_schemas=_tools(),
        grounded_state={},
        difficulty="complete",
        rng=random.Random(42),
        chain_seed=["list_items", "update_item"],
        chain_context={
            "dependency_relations": [{
                "source_capability": "list_items",
                "target_capability": "update_item",
                "relation": "explicit",
                "value_bindings": [{
                    "source_output_field": "item_id",
                    "target_argument": "item_id",
                }],
                "state_bindings": [],
            }],
        },
    )

    assert "## Verified Causal Relations" in client.prompt
    assert "list_items output field item_id supplies update_item argument item_id" in client.prompt


def test_query_teacher_does_not_duplicate_chain_grounding_records() -> None:
    class Client:
        def __init__(self) -> None:
            self.prompt = ""

        def generate_chat(self, messages, **_kwargs) -> str:
            self.prompt = "\n".join(item["content"] for item in messages)
            return json.dumps({
                "user_query": "Update the only shown item.",
                "target_capability": "update_item",
                "chain_supported": True,
                "mutation_evidence": [{
                    "capability": "update_item",
                    "query_span": "Update the only shown item",
                }],
            })

    marker = "UNIQUE_GROUNDING_RECORD"
    client = Client()
    planner = TaskPlanner(
        client, "banking", seed=42,
        prompt_profile="local_trainable_v1",
    )
    planner.generate_query(
        _tools(),
        {"public_entity_summaries": [marker]},
        "complete",
        random.Random(0),
        chain_seed=["list_items", "update_item"],
        chain_context={
            "query_grounding_summaries": [marker],
            "opaque_id_hidden_types": ["item"],
        },
    )

    assert client.prompt.count(marker) == 1


def test_terminal_opaque_id_instruction_is_local_profile_only() -> None:
    class Client:
        def __init__(self) -> None:
            self.prompt = ""

        def generate_chat(self, messages, **_kwargs) -> str:
            self.prompt = "\n".join(message["content"] for message in messages)
            return json.dumps({"action": "final_answer", "text": "Done."})

    prompts: dict[str, str] = {}
    for profile in (
        "paper_generation_baseline_v1", "local_trainable_v1",
    ):
        client = Client()
        TaskPlanner(
            client, "banking", seed=42, prompt_profile=profile,
        ).decide_action(
            tool_schemas=[],
            user_query="Show my account.",
            execution_history=[],
            attempt=1,
        )
        prompts[profile] = client.prompt

    marker = "Opaque sampler IDs are internal tool arguments"
    assert marker not in prompts["paper_generation_baseline_v1"]
    assert marker in prompts["local_trainable_v1"]


def test_local_terminal_retries_sampler_private_id_without_rewriting() -> None:
    class Client:
        def __init__(self) -> None:
            self.calls = 0
            self.prompts: list[str] = []

        def generate_chat(self, messages, **_kwargs) -> str:
            self.calls += 1
            self.prompts.append("\n".join(item["content"] for item in messages))
            if self.calls == 1:
                return json.dumps({
                    "action": "final_answer",
                    "text": "Updated issue iss_s42_0001.",
                })
            return json.dumps({
                "action": "final_answer",
                "text": "Updated the login timeout issue.",
            })

    client = Client()
    action = TaskPlanner(
        client, "issue_tracker", seed=42,
        prompt_profile="local_trainable_v1",
    ).decide_action(
        tool_schemas=[],
        user_query="Mark the login timeout issue urgent.",
        execution_history=[],
        attempt=1,
    )

    assert client.calls == 2
    assert action.text == "Updated the login timeout issue."
    assert "Previous Output Correction" in client.prompts[1]


def test_local_terminal_retries_exact_private_trace_id() -> None:
    class Client:
        def __init__(self) -> None:
            self.calls = 0

        def generate_chat(self, _messages, **_kwargs) -> str:
            self.calls += 1
            text = (
                "Archived email eml_0028."
                if self.calls == 1
                else "Archived the server migration email."
            )
            return json.dumps({"action": "final_answer", "text": text})

    client = Client()
    action = TaskPlanner(
        client, "email", seed=42, prompt_profile="local_trainable_v1",
    ).decide_action(
        tool_schemas=EMAIL_TOOLS,
        user_query="Archive the server migration email.",
        execution_history=[{
            "tool_name": "search_emails",
            "arguments": {"subject_contains": "server migration"},
            "success": True,
            "observation": {
                "emails": [{"email_id": "eml_0028", "subject": "Server migration"}],
                "count": 1,
            },
        }],
        attempt=1,
    )

    assert client.calls == 2
    assert action.text == "Archived the server migration email."


def test_local_terminal_uses_distractor_owner_for_private_trace_id() -> None:
    class Client:
        def __init__(self) -> None:
            self.calls = 0

        def generate_chat(self, _messages, **_kwargs) -> str:
            self.calls += 1
            text = (
                "Updated event evt_0028."
                if self.calls == 1
                else "Updated the server migration event."
            )
            return json.dumps({"action": "final_answer", "text": text})

    calendar_schema = next(
        dict(tool, _server_name="calendar")
        for tool in CALENDAR_TOOLS
        if tool["name"] == "get_event"
    )
    visible_tools = [*EMAIL_TOOLS, calendar_schema]
    client = Client()
    action = TaskPlanner(
        client, "email", seed=42, prompt_profile="local_trainable_v1",
    ).decide_action(
        tool_schemas=visible_tools,
        user_query="Update the event mentioned in the migration email.",
        execution_history=[{
            "server_name": "calendar",
            "tool_name": "get_event",
            "arguments": {"event_id": "evt_0028"},
            "success": True,
            "observation": {
                "event": {"event_id": "evt_0028", "title": "Server migration"},
            },
        }],
        attempt=1,
    )

    assert client.calls == 2
    assert action.text == "Updated the server migration event."


def test_local_terminal_allows_declared_public_business_reference() -> None:
    class Client:
        def __init__(self) -> None:
            self.calls = 0

        def generate_chat(self, _messages, **_kwargs) -> str:
            self.calls += 1
            return json.dumps({
                "action": "final_answer",
                "text": "Created issue iss_0021.",
            })

    client = Client()
    action = TaskPlanner(
        client, "issue_tracker", seed=42,
        prompt_profile="local_trainable_v1",
    ).decide_action(
        tool_schemas=ISSUE_TOOLS,
        user_query="Create an issue for the timeout.",
        execution_history=[{
            "tool_name": "create_issue",
            "arguments": {"title": "Timeout"},
            "success": True,
            "observation": {"issue": {"issue_id": "iss_0021", "title": "Timeout"}},
        }],
        attempt=1,
    )

    assert client.calls == 1
    assert action.text == "Created issue iss_0021."


def test_local_terminal_retries_internal_tool_name_without_rewriting() -> None:
    class Client:
        def __init__(self) -> None:
            self.calls = 0

        def generate_chat(self, _messages, **_kwargs) -> str:
            self.calls += 1
            text = (
                "The update_event function is unavailable."
                if self.calls == 1
                else "I cannot update that calendar event."
            )
            return json.dumps({"action": "report_error", "text": text})

    client = Client()
    action = TaskPlanner(
        client, "calendar", seed=42,
        prompt_profile="local_trainable_v1",
    ).decide_action(
        tool_schemas=CALENDAR_TOOLS,
        user_query="Update Team Sync.",
        execution_history=[],
        attempt=1,
    )

    assert client.calls == 2
    assert action.text == "I cannot update that calendar event."


def test_terminal_retry_exhaustion_preserves_exact_reason() -> None:
    class Client:
        def generate_chat(self, _messages, **_kwargs) -> str:
            return json.dumps({
                "action": "report_error",
                "text": "The update_event function is unavailable.",
            })

    planner = TaskPlanner(
        Client(), "calendar", seed=42,
        prompt_profile="local_trainable_v1",
    )
    with pytest.raises(ActionDecisionError) as exc_info:
        planner.decide_action(
            tool_schemas=CALENDAR_TOOLS,
            user_query="Update Team Sync.",
            execution_history=[],
            attempt=1,
        )

    assert exc_info.value.reason == "terminal_private_tool_name_exposure"
    assert exc_info.value.details["attempts"] == 3


def test_turn_loop_failure_preserves_partial_execution_trace() -> None:
    class Teacher:
        def __init__(self) -> None:
            self.calls = 0

        def decide_action(self, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                return ActionPlan(
                    action="tool_call",
                    tool_name="list_invoices",
                    arguments={},
                )
            raise ActionDecisionError(
                "terminal retry exhausted",
                reason="terminal_private_tool_name_exposure",
                details={"attempts": 3},
            )

    class Executor:
        def execute(self, *_args, **_kwargs):
            return ToolExecutionResult(
                success=True,
                tool_name="list_invoices",
                canonical_tool_name="list_invoices",
                call_id="call",
                session_id="session",
                observation={"invoices": [{"invoice_id": "inv_1"}]},
                error_type=None,
                error_message="",
                schema_valid=True,
                state_changed=False,
                latency_ms=1,
                execution_status="SUCCESS",
            )

    owner = type("Owner", (), {"executor": Executor()})()
    with pytest.raises(TurnLoopError) as exc_info:
        run_turn_loop(
            owner,
            teacher=Teacher(),
            current_query="List invoices.",
            server_tools=PAYMENT_TOOLS,
            server_name="payments",
            session_id="session",
            difficulty="complete",
            round_idx=0,
        )

    assert exc_info.value.reason == "terminal_private_tool_name_exposure"
    assert [
        call["tool_name"] for call in exc_info.value.details["oracle_calls"]
    ] == ["list_invoices"]
    assert exc_info.value.details["execution_history"][0]["success"] is True


def test_missing_function_contract_persists_after_tool_feedback() -> None:
    class Client:
        def __init__(self) -> None:
            self.prompts: list[str] = []

        def generate_chat(self, messages, **_kwargs) -> str:
            self.prompts.append(
                "\n".join(message["content"] for message in messages)
            )
            return json.dumps({
                "action": "report_error",
                "reason": "No available capability can complete the request.",
            })

    client = Client()
    planner = TaskPlanner(client, "banking", seed=42)
    for attempt in (0, 1):
        planner.decide_action(
            tool_schemas=[],
            user_query="Generate my account statement.",
            execution_history=[{
                "tool_name": "list_accounts",
                "success": True,
                "state_changed": False,
                "observation": {"accounts": []},
            }],
            attempt=attempt,
            missing_function=True,
            blocked_tools={"get_statement"},
        )

    marker = "## Missing-Function Contract"
    assert marker in client.prompts[0]
    assert marker in client.prompts[1]


def test_normal_action_prompt_omits_missing_function_contract() -> None:
    class Client:
        def __init__(self) -> None:
            self.prompt = ""

        def generate_chat(self, messages, **_kwargs) -> str:
            self.prompt = "\n".join(
                message["content"] for message in messages
            )
            return json.dumps({"action": "final_answer", "text": "Done."})

    client = Client()
    TaskPlanner(client, "banking", seed=42).decide_action(
        tool_schemas=[],
        user_query="Show my balance.",
        execution_history=[],
        attempt=1,
        missing_function=False,
    )

    assert "## Missing-Function Contract" not in client.prompt


def test_query_generation_reports_goal_unsat_for_fixed_chain_state() -> None:
    class Client:
        def generate_chat(self, _messages, **_kwargs) -> str:
            return json.dumps({
                "user_query": "UNSAT",
                "target_capability": "update_item",
                "chain_supported": False,
                "mutation_evidence": [],
            })

    planner = TaskPlanner(Client(), "demo", seed=42)
    with pytest.raises(QueryGenerationError) as exc_info:
        planner.generate_query(
            tool_schemas=_tools(),
            grounded_state={},
            difficulty="complete",
            rng=random.Random(42),
            chain_seed=["list_items", "update_item"],
            chain_context={},
        )
    assert exc_info.value.reason == "goal_unsat"


def test_query_generation_treats_explicit_chain_unsupported_as_goal_unsat() -> None:
    class Client:
        def __init__(self) -> None:
            self.calls = 0

        def generate_chat(self, _messages, **_kwargs) -> str:
            self.calls += 1
            return json.dumps({
                "user_query": "Update the item.",
                "target_capability": "update_item",
                "chain_supported": False,
                "mutation_evidence": [],
            })

    client = Client()
    planner = TaskPlanner(client, "demo", seed=42)
    generated = planner.generate_query(
            tool_schemas=_tools(),
            grounded_state={},
            difficulty="complete",
            rng=random.Random(42),
            chain_seed=["list_items", "update_item"],
            chain_context={},
        )

    assert generated.target_capability == "update_item"
    assert client.calls == 1


def test_query_generation_does_not_treat_missing_support_field_as_goal_unsat() -> None:
    class Client:
        def __init__(self) -> None:
            self.calls = 0

        def generate_chat(self, _messages, **_kwargs) -> str:
            self.calls += 1
            return json.dumps({
                "user_query": "Update the item.",
                "target_capability": "update_item",
                "mutation_evidence": [{
                    "capability": "update_item",
                    "query_span": "Update",
                }],
            })

    client = Client()
    planner = TaskPlanner(client, "demo", seed=42)
    generated = planner.generate_query(
            tool_schemas=_tools(),
            grounded_state={},
            difficulty="complete",
            rng=random.Random(42),
            chain_seed=["list_items", "update_item"],
            chain_context={},
        )

    assert generated.target_capability == "update_item"
    assert client.calls == 1


def test_query_generation_requires_evidence_for_every_chain_mutation() -> None:
    class Client:
        def __init__(self) -> None:
            self.calls = 0

        def generate_chat(self, _messages, **_kwargs) -> str:
            self.calls += 1
            evidence = [{
                "capability": "update_item",
                "query_span": "then update it",
            }]
            if self.calls == 2:
                evidence.insert(0, {
                    "capability": "create_item",
                    "query_span": "Create a draft",
                })
            return json.dumps({
                "user_query": "Create a draft, then update it.",
                "target_capability": "update_item",
                "chain_supported": True,
                "mutation_evidence": evidence,
            })

    tools = [
        {
            "name": name,
            "description": name,
            "input_schema": {"type": "object", "properties": {}},
            "annotations": {"mutating": True},
        }
        for name in ("create_item", "update_item")
    ]
    client = Client()
    generated = TaskPlanner(client, "demo", seed=42).generate_query(
        tool_schemas=tools,
        grounded_state={},
        difficulty="complete",
        rng=random.Random(42),
        chain_seed=["create_item", "update_item"],
        chain_context={},
    )

    assert client.calls == 1


def test_query_generation_stops_after_repeated_partial_mutation_goal() -> None:
    class Client:
        def __init__(self) -> None:
            self.calls = 0

        def generate_chat(self, _messages, **_kwargs) -> str:
            self.calls += 1
            return json.dumps({
                "user_query": "Update the item.",
                "target_capability": "update_item",
                "chain_supported": True,
                "mutation_evidence": [{
                    "capability": "update_item",
                    "query_span": "Update",
                }],
            })

    tools = [
        {
            "name": name,
            "input_schema": {"type": "object", "properties": {}},
            "annotations": {"mutating": True},
        }
        for name in ("create_item", "update_item")
    ]
    client = Client()
    generated = TaskPlanner(client, "demo", seed=42).generate_query(
            tool_schemas=tools,
            grounded_state={},
            difficulty="complete",
            rng=random.Random(42),
            chain_seed=["create_item", "update_item"],
            chain_context={},
        )

    assert generated.target_capability == "update_item"
    assert client.calls == 1


def test_query_generation_rejects_mutation_evidence_outside_chain() -> None:
    class Client:
        def __init__(self) -> None:
            self.calls = 0

        def generate_chat(self, _messages, **_kwargs) -> str:
            self.calls += 1
            evidence = [{
                "capability": "update_item",
                "query_span": "update it",
            }]
            if self.calls == 1:
                evidence.append({
                    "capability": "delete_item",
                    "query_span": "update it",
                })
            return json.dumps({
                "user_query": "Find the item and update it.",
                "target_capability": "update_item",
                "chain_supported": True,
                "mutation_evidence": evidence,
            })

    tools = [
        {
            "name": "list_items",
            "input_schema": {"type": "object", "properties": {}},
            "annotations": {"readonly": True, "mutating": False},
        },
        *[
            {
                "name": name,
                "input_schema": {"type": "object", "properties": {}},
                "annotations": {"mutating": True},
            }
            for name in ("update_item", "delete_item")
        ],
    ]
    client = Client()
    generated = TaskPlanner(client, "demo", seed=42).generate_query(
        tool_schemas=tools,
        grounded_state={},
        difficulty="complete",
        rng=random.Random(42),
        chain_seed=["list_items", "update_item"],
        chain_context={},
    )

    assert client.calls == 1


@pytest.mark.parametrize(
    ("difficulty", "required_marker", "forbidden_markers"),
    [
        (
            "complete",
            "Every required field that controls a state change MUST",
            (
                "Exactly ONE critical user-supplied field may be absent",
                "Do NOT require concrete entity selectors or parameters",
            ),
        ),
        (
            "missing",
            "Exactly ONE critical user-supplied field may be absent",
            (
                "Every required field that controls a state change MUST",
                "Do NOT require concrete entity selectors or parameters",
            ),
        ),
        (
            "minimal",
            "Do NOT require concrete entity selectors or parameters",
            (
                "Every required field that controls a state change MUST",
                "Exactly ONE critical user-supplied field may be absent",
            ),
        ),
    ],
)
def test_query_prompt_applies_only_the_selected_difficulty_contract(
    difficulty: str,
    required_marker: str,
    forbidden_markers: tuple[str, ...],
) -> None:
    class Client:
        def __init__(self) -> None:
            self.prompt = ""

        def generate_chat(self, messages, **_kwargs) -> str:
            self.prompt = "\n".join(
                str(message["content"]) for message in messages
            )
            return json.dumps({
                "user_query": "update the item",
                "target_capability": "update_item",
                "chain_supported": True,
                "mutation_evidence": [{
                    "capability": "update_item",
                    "query_span": "update",
                }],
            })

    client = Client()
    generated = TaskPlanner(client, "demo", seed=42).generate_query(
        tool_schemas=_tools(),
        grounded_state={},
        difficulty=difficulty,
        rng=random.Random(42),
        chain_seed=["list_items", "update_item"],
        chain_context={},
    )

    assert generated.user_query == "update the item"
    assert required_marker in client.prompt
    assert all(marker not in client.prompt for marker in forbidden_markers)
    assert "mutation_evidence" not in client.prompt


def test_query_generation_rejects_unknown_difficulty_before_teacher_call() -> None:
    class Client:
        def generate_chat(self, _messages, **_kwargs) -> str:
            raise AssertionError("Teacher must not run for an invalid difficulty")

    with pytest.raises(ValueError, match="unknown query difficulty"):
        TaskPlanner(Client(), "demo", seed=42).generate_query(
            tool_schemas=_tools(),
            grounded_state={},
            difficulty="mixed",
            rng=random.Random(42),
            chain_seed=["list_items", "update_item"],
            chain_context={},
        )
