from __future__ import annotations

import json
import random

import pytest

from src.live_mcp.orchestrator import _decision_stratum
from src.live_mcp.generation.query_teacher import QueryGenerationError
from src.live_mcp.task_planner import TaskPlanner
from src.live_mcp.task_spec import DifficultyVector, compile_task_spec


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
    planner = TaskPlanner(client, "banking", seed=42)
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
    assert generated.mutation_evidence == [{
        "capability": "freeze_account",
        "query_span": "Freeze",
    }]


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

    assert client.calls == 2
    assert {item["capability"] for item in generated.mutation_evidence} == {
        "create_item", "update_item",
    }


def test_compile_task_spec_classifies_parameter_provenance() -> None:
    spec = compile_task_spec(
        domain="demo",
        session_seed=42,
        state_profile="baseline",
        state_fingerprint="state-sha",
        difficulty="complete",
        source_chain=["list_items", "update_item"],
        tool_schemas=_tools(),
        dependency_contracts=[{
            "source_capability": "list_items",
            "target_capability": "update_item",
            "target_argument": "item_id",
            "source_output_field": "item_id",
        }],
        natural_selector_types=["item"],
        robustness={"distractors": False},
    )

    assert spec.final_outcome_capability == "update_item"
    assert spec.user_decided_parameters == (("update_item", "value"),)
    assert spec.dependency_bindings[0].provenance_class == "tool_discoverable"
    assert "observation_derived_argument" in spec.static_strata
    assert spec.fingerprint() == spec.fingerprint()
    prompt = spec.to_prompt_contract()
    assert "session_seed" not in prompt
    assert "state_fingerprint" not in prompt
    assert prompt["dependency_bindings"][0]["target_argument"] == "item_id"


def test_compile_task_spec_rejects_non_operational_chain() -> None:
    with pytest.raises(ValueError, match="operational field-level dependency"):
        compile_task_spec(
            domain="demo",
            session_seed=42,
            state_profile="baseline",
            state_fingerprint="state-sha",
            difficulty="complete",
            source_chain=["list_items", "update_item"],
            tool_schemas=_tools(),
            dependency_contracts=[],
            natural_selector_types=[],
            robustness={},
        )


def test_compile_task_spec_persists_decision_difficulty_without_values() -> None:
    vector = DifficultyVector(
        selector_candidate_count=3,
        viable_chain_count=4,
        operational_dependency_count=1,
        observation_derived_argument_count=1,
        post_mutation_recheck_count=0,
        distractor_count=2,
        oracle_tool_count=2,
    )
    spec = compile_task_spec(
        domain="demo",
        session_seed=42,
        state_profile="baseline",
        state_fingerprint="state-sha",
        difficulty="complete",
        source_chain=["list_items", "update_item"],
        tool_schemas=_tools(),
        dependency_contracts=[{
            "source_capability": "list_items",
            "target_capability": "update_item",
            "target_argument": "item_id",
            "source_output_field": "item_id",
        }],
        natural_selector_types=["item"],
        robustness={"distractors": True},
        decision_stratum="discovery",
        difficulty_vector=vector,
    )

    prompt = spec.to_prompt_contract()
    assert spec.decision_stratum == "discovery"
    assert spec.difficulty_vector.selector_candidate_count == 3
    assert "decision_stratum" not in prompt
    assert "difficulty_vector" not in prompt
    assert "session_seed" not in prompt


@pytest.mark.parametrize(
    ("selector_count", "derived_count", "recheck_count", "expected"),
    [
        (1, 1, 0, "direct"),
        (3, 1, 0, "discovery"),
        (1, 2, 0, "dependent"),
        (1, 2, 1, "stateful"),
    ],
)
def test_decision_strata_are_mutually_exclusive(
    selector_count: int,
    derived_count: int,
    recheck_count: int,
    expected: str,
) -> None:
    vector = DifficultyVector(
        selector_candidate_count=selector_count,
        viable_chain_count=2,
        operational_dependency_count=derived_count,
        observation_derived_argument_count=derived_count,
        post_mutation_recheck_count=recheck_count,
        distractor_count=0,
        oracle_tool_count=2,
    )
    assert _decision_stratum(vector) == expected
