from __future__ import annotations

import json
import random

import pytest

from src.live_mcp.generation.teacher_contracts import (
    reference_date_for_candidate_state,
    reference_date_for_seed,
)
from src.live_mcp.domain_contracts.reference_visibility import (
    is_public_entity_reference,
    public_entity_reference_ids_from_records,
)
from src.live_mcp.live_state_query_view import (
    generation_query_prompt_state,
    public_query_prompt_state,
)
from src.live_mcp.planner_format import format_state_compact
from src.live_mcp.task_planner import TaskPlanner
from src.live_mcp.servers.banking.server import TOOLS as BANKING_TOOLS


def test_candidate_reference_date_is_owned_by_live_state_seed() -> None:
    generation_seed = 3712943567317000043
    state_seed = 3712943567317000004

    assert reference_date_for_candidate_state(
        generation_seed, state_seed,
    ) == reference_date_for_seed(state_seed)
    assert reference_date_for_candidate_state(
        generation_seed, None,
    ) == reference_date_for_seed(generation_seed)


def _banking_live_context() -> dict:
    return {
        "entity_ids": [
            {"id": "acc_s800000200042_003", "type": "account"},
            {
                "id": "sched_s800000200042_002",
                "type": "scheduled_transfer",
            },
        ],
        "entity_summaries": [
            "  acc_s800000200042_003 (account): type=business, balance=2500",
            "  sched_s800000200042_002 (scheduled_transfer): amount=100, "
            "account_id=acc_s800000200042_003, status=scheduled",
        ],
        "entity_records": [
            {
                "id": "acc_s800000200042_003",
                "type": "account",
                "data": {
                    "type": "business",
                    "balance": 2500,
                    "frozen": False,
                    "account_last4": "4821",
                },
            },
            {
                "id": "sched_s800000200042_002",
                "type": "scheduled_transfer",
                "data": {
                    "amount": 100,
                    "status": "scheduled",
                    "account_id": "acc_s800000200042_003",
                },
            },
        ],
    }


def test_banking_public_query_state_removes_private_ids_but_keeps_selectors() -> None:
    state = public_query_prompt_state(_banking_live_context(), "banking")
    rendered = format_state_compact(state)

    assert "acc_s800000200042_003" not in rendered
    assert "sched_s800000200042_002" not in rendered
    assert '"type": "business"' in rendered
    assert '"balance": 2500' in rendered
    assert '"account_last4": "4821"' in rendered
    assert '"amount": 100' in rendered
    assert '"status": "scheduled"' in rendered
    assert "account_id" not in rendered


def test_public_query_state_preserves_nonopaque_user_identifier() -> None:
    state = public_query_prompt_state({
        "entity_ids": [{"id": "/workspace/report.txt", "type": "file"}],
        "entity_summaries": [
            "  /workspace/report.txt (file): size=12"
        ],
        "entity_records": [],
    }, "filesystem")

    assert "/workspace/report.txt" in format_state_compact(state)


def test_public_query_state_hides_seeded_business_reference() -> None:
    state = public_query_prompt_state({
        "entity_ids": [{"id": "inv_s42_001", "type": "invoice"}],
        "entity_summaries": [
            "  inv_s42_001 (invoice): customer=Acme status=pending"
        ],
        "entity_records": [{
            "id": "inv_s42_001",
            "type": "invoice",
            "data": {
                "invoice_id": "inv_s42_001",
                "customer": "Acme",
                "status": "pending",
            },
        }],
    }, "payments")

    rendered = format_state_compact(state)
    assert "inv_s42_001" not in rendered
    assert '"customer": "Acme"' in rendered


def test_live_state_seeded_business_reference_is_not_a_public_id_fact() -> None:
    records = [{
        "id": "inv_s42_001",
        "type": "invoice",
        "data": {
            "invoice_id": "inv_s42_001",
            "customer": "Acme",
        },
    }, {
        "id": "pay_s42_002",
        "type": "payment",
        "data": {"status": "pending"},
    }]

    assert public_entity_reference_ids_from_records(
        "payments", records,
    ) == set()


@pytest.mark.parametrize(("domain", "entity_type", "field", "value"), [
    ("banking", "transaction", "txn_id", "txn_s42_001"),
    ("calendar", "event", "event_id", "evt_s42_001"),
    ("crm", "deal", "deal_id", "deal_s42_001"),
    ("email", "email", "email_id", "email_s42_001"),
    ("filesystem", "file", "path", "file_s42_001"),
    ("food_delivery", "order", "order_id", "ord_s42_001"),
    ("issue_tracker", "issue", "issue_id", "iss_s42_001"),
    ("payments", "invoice", "invoice_id", "inv_s42_001"),
    ("shopping", "order", "order_id", "ord_s42_001"),
    ("team_chat", "channel", "channel_id", "ch_s42_001"),
])
def test_sampler_handle_is_never_a_public_reference(
    domain: str, entity_type: str, field: str, value: str,
) -> None:
    assert is_public_entity_reference(
        domain, entity_type, field, value,
    ) is False


def test_filesystem_path_is_a_natural_public_reference() -> None:
    assert is_public_entity_reference(
        "filesystem", "file", "path", "/home/user/clean.txt",
    ) is True


def test_public_query_state_hides_seeded_channel_but_keeps_natural_channel() -> None:
    state = public_query_prompt_state({
        "entity_ids": [
            {"id": "ch_s42_001", "type": "channel"},
            {"id": "releases", "type": "channel"},
        ],
        "entity_summaries": ["seeded channel", "releases channel"],
        "entity_records": [
            {
                "id": "ch_s42_001", "type": "channel",
                "data": {"channel_id": "ch_s42_001", "name": "ops"},
            },
            {
                "id": "releases", "type": "channel",
                "data": {"channel_id": "releases", "name": "releases"},
            },
        ],
    }, "team_chat")

    rendered = format_state_compact(state)
    assert "ch_s42_001" not in rendered
    assert "releases" in rendered


def test_public_query_state_omits_opaque_record_without_natural_selector() -> None:
    state = public_query_prompt_state({
        "entity_ids": [{"id": "thr_s42_001", "type": "thread"}],
        "entity_summaries": ["  thr_s42_001 (thread)"],
        "entity_records": [{
            "id": "thr_s42_001",
            "type": "thread",
            "data": {
                "thread_id": "thr_s42_001",
                "root_message_id": "msg_s42_001",
            },
        }],
    }, "team_chat")

    assert state == {"public_entity_summaries": []}


def test_public_query_state_uses_cross_domain_natural_selector_fields() -> None:
    state = public_query_prompt_state({
        "entity_ids": [{"id": "msg_s42_001", "type": "message"}],
        "entity_summaries": ["message"],
        "entity_records": [{
            "id": "msg_s42_001",
            "type": "message",
            "data": {
                "author": "alex",
                "channel_name": "ops",
                "content": "deploy completed",
            },
        }],
    }, "team_chat")

    rendered = format_state_compact(state)
    assert "deploy completed" in rendered
    assert "msg_s42_001" not in rendered

    payment_state = public_query_prompt_state({
        "entity_ids": [{"id": "wh_s42_001", "type": "webhook"}],
        "entity_summaries": ["webhook"],
        "entity_records": [{
            "id": "wh_s42_001",
            "type": "webhook",
            "data": {"url": "https://hooks.example.test/a", "active": True},
        }],
    }, "payments")
    payment_rendered = format_state_compact(payment_state)
    assert "https://hooks.example.test/a" in payment_rendered
    assert "wh_s42_001" not in payment_rendered


def test_profile_state_routing_keeps_prove_ids_and_hides_local_opaque_ids() -> None:
    context = _banking_live_context()
    paper = generation_query_prompt_state(
        context, "banking", natural_selector=False,
    )
    local = generation_query_prompt_state(
        context, "banking", natural_selector=True,
    )

    assert "acc_s800000200042_003" in format_state_compact(paper)
    assert "acc_s800000200042_003" not in format_state_compact(local)


def test_followup_and_clarification_prompts_receive_only_public_state() -> None:
    class Client:
        def __init__(self) -> None:
            self.messages: list[list[dict[str, str]]] = []

        def generate_chat(self, messages, **_kwargs) -> str:
            self.messages.append(messages)
            if "target_capability" in messages[-1]["content"] and (
                "Choose one concrete" in messages[-1]["content"]
            ):
                return json.dumps({
                    "target_capability": "list_scheduled_transfers",
                })
            return json.dumps({
                "user_query": "Check that transfer again.",
                "target_capability": "list_scheduled_transfers",
                "mutation_evidence": [],
                "argument_evidence": [],
            })

    client = Client()
    planner = TaskPlanner(
        client, "banking", seed=800000200042,
        prompt_profile="local_trainable_v1",
    )
    public_state = public_query_prompt_state(_banking_live_context(), "banking")
    kwargs = {
        "tool_schemas": BANKING_TOOLS,
        "grounded_state": public_state,
        "previous_query": "Cancel the $500 transfer.",
        "difficulty": "complete",
        "rng": random.Random(0),
        "previous_response": "The $500 transfer was cancelled.",
    }

    planner.generate_followup(**kwargs)
    planner.generate_clarification(**kwargs)

    rendered_prompts = "\n".join(
        message["content"]
        for messages in client.messages
        for message in messages
    )
    assert "acc_s800000200042_003" not in rendered_prompts
    assert "sched_s800000200042_002" not in rendered_prompts
    assert "Grounded Live State" in rendered_prompts
    assert "Never invent an identifier" in rendered_prompts
