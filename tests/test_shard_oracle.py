from types import SimpleNamespace

import pytest

from src.live_mcp.corpus.shard_oracle import (
    _validate_task_training_contract,
)
from src.live_mcp.types import OracleCall


def test_missing_function_artifact_requires_hidden_chain_final() -> None:
    terminal = OracleCall(
        "", {"text": "unavailable"}, action="report_error",
    )
    task = SimpleNamespace(
        task_id="missing-function-hidden-target-mismatch",
        metadata={
            "scenario_type": "missing_function",
            "has_missing_function": True,
            "hidden_tool": "symlink",
            "source_chain_seed": ["symlink", "chmod"],
            "missing_function_evidence": ["file.link=true"],
            "prompt_profile": "local_trainable_v1",
        },
        task_type="missing_function",
        oracle_program=SimpleNamespace(
            calls=[terminal], success_criteria=[],
        ),
        oracle_calls_per_round=[[terminal]],
        execution_history_per_round=[[]],
        hidden_tools=["symlink"],
        visible_tools=[],
        target_servers=["filesystem"],
        conversation_queries=["Create a link and change its permissions."],
    )

    with pytest.raises(
        ValueError,
        match=r"must hide exactly source_chain_seed\[-1\]",
    ):
        _validate_task_training_contract(task)


def test_shard_rejects_initial_mutation_absent_from_query_source() -> None:
    calls = [
        OracleCall("search_emails", {}),
        OracleCall("archive_email", {"email_id": "email-public-1"}),
        OracleCall("", {"text": "done"}, action="final_answer"),
    ]
    task = SimpleNamespace(
        task_id="unauthorized-initial-mutation",
        metadata={
            "scenario_type": "normal_safe_success",
            "generation_mode": "chain_seeded",
            "generation_method": "contract_test",
            "prompt_profile": "local_trainable_v1",
            "source_chain_seed": ["search_emails", "get_email"],
            "chain_seed": [],
        },
        task_type="task_planner",
        oracle_program=SimpleNamespace(calls=calls, success_criteria=[]),
        oracle_calls_per_round=[calls],
        execution_history_per_round=[[]],
        hidden_tools=[],
        visible_tools=[],
        target_servers=["email"],
        conversation_queries=["show me the email"],
        user_prompt="show me the email",
    )

    with pytest.raises(
        ValueError, match="capabilities absent from source_chain_seed",
    ):
        _validate_task_training_contract(task)
