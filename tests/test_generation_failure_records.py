from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.live_mcp.corpus.failure_records import GenerationFailureWriter
from src.live_mcp.generation.batch import BatchGenerationMixin
from src.live_mcp.generation.irrelevance import IrrelevanceGenerationMixin
from src.live_mcp.generation.query_teacher import QueryGenerationError
from src.live_mcp.generation.candidate_pipeline import (
    _candidate_conversation_reason,
)
from src.live_mcp.errors import CandidateGenerationError, TurnLoopError


@pytest.mark.parametrize(
    ("exception", "reason"),
    [
        (RuntimeError("unexpected conversation failure"),
         "candidate_conversation_exception"),
        (
            RuntimeError("Failed to generate followup for filesystem"),
            "continuation_generation_failed",
        ),
        (
            RuntimeError(
                "Teacher exhausted the per-round action budget without a "
                "terminal response"
            ),
            "teacher_action_budget_exhausted",
        ),
        (
            RuntimeError(
                "Teacher repeated a successful no-progress tool call after "
                "three pre-dispatch rejections"
            ),
            "teacher_no_progress_exhausted",
        ),
    ],
)
def test_conversation_failures_use_stable_reason_taxonomy(
    exception: Exception, reason: str,
) -> None:
    assert _candidate_conversation_reason(exception) == reason


def test_structured_turn_failure_reason_takes_precedence() -> None:
    error = TurnLoopError(
        "generic outer message",
        reason="terminal_private_tool_name_exposure",
        details={"oracle_calls": []},
    )

    assert _candidate_conversation_reason(error) == (
        "terminal_private_tool_name_exposure"
    )


def test_failure_writer_appends_once_and_preserves_unclassified_evidence(
    tmp_path: Path,
) -> None:
    path = tmp_path / "failures.jsonl"
    writer = GenerationFailureWriter(path)
    record = {
        "domain": "banking",
        "generation_seed": 42,
        "stage": "candidate_generation",
        "reason_code": "candidate_exception",
        "traceback": "trace one",
    }

    assert writer.append(record) is True
    assert writer.append({
        **record,
        "message": "changed after resume",
        "traceback": "trace two",
    }) is False

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["classification"] == "unclassified"
    assert payload["traceback"] == "trace one"
    assert len(payload["failure_id"]) == 64


def test_failure_writer_rejects_capture_time_classification(tmp_path: Path) -> None:
    writer = GenerationFailureWriter(tmp_path / "failures.jsonl")

    with pytest.raises(ValueError, match="remain unclassified"):
        writer.append({
            "domain": "banking",
            "stage": "candidate_generation",
            "reason_code": "candidate_exception",
            "classification": "system_logic",
        })


def test_batch_records_exception_identity_and_traceback(monkeypatch) -> None:
    class FailingGenerator(BatchGenerationMixin):
        CHAIN_SAMPLING_JACCARD_THRESHOLD = 0.70
        SAMPLING_CONTEXT_REFRESH_K = 10

        def __init__(self) -> None:
            self.manager = SimpleNamespace(server_names=["banking"])

        def _generate_task_with_postprocess(
            self,
            server_name,
            seed,
            state_seed,
            difficulty,
            distractor_rate,
            missing_function_rate,
        ):
            raise KeyError(f"forced-{seed}")

        def _generate_irrelevant_tasks(
            self, count, seed, servers, failure_callback=None,
        ):
            assert count == 0
            assert failure_callback is not None
            return []

    monkeypatch.setenv("LIVEMCP_GENERATION_MAX_WORKERS", "1")
    failures: list[dict] = []

    with pytest.raises(RuntimeError, match="produced 0 tasks"):
        FailingGenerator().generate_many(
            "banking",
            count=1,
            seed=42,
            difficulty_mix={"complete": 1.0},
            irrelevance_ratio=0.0,
            failure_callback=failures.append,
        )

    assert [record["generation_seed"] for record in failures] == [42, 43]
    assert all(record["state_seed"] == 42 for record in failures)
    assert all(record["difficulty"] == "complete" for record in failures)
    assert all(record["reason_code"] == "candidate_exception" for record in failures)
    assert all(record["exception_type"] == "KeyError" for record in failures)
    assert all("forced-" in record["traceback"] for record in failures)


def test_batch_preserves_structured_query_failure_reason(monkeypatch) -> None:
    class FailingGenerator(BatchGenerationMixin):
        CHAIN_SAMPLING_JACCARD_THRESHOLD = 0.70
        SAMPLING_CONTEXT_REFRESH_K = 10

        def __init__(self) -> None:
            self.manager = SimpleNamespace(server_names=["banking"])

        def _generate_task_with_postprocess(self, *args, **kwargs):
            raise QueryGenerationError(
                "fixed chain and state do not support one goal",
                reason="goal_unsat",
            )

        def _generate_irrelevant_tasks(
            self, count, seed, servers, failure_callback=None,
        ):
            return []

    monkeypatch.setenv("LIVEMCP_GENERATION_MAX_WORKERS", "1")
    failures: list[dict] = []

    with pytest.raises(RuntimeError, match="produced 0 tasks"):
        FailingGenerator().generate_many(
            "banking",
            count=1,
            seed=42,
            difficulty_mix={"complete": 1.0},
            irrelevance_ratio=0.0,
            failure_callback=failures.append,
        )

    assert failures
    assert {record["reason_code"] for record in failures} == {"goal_unsat"}
    assert {record["exception_type"] for record in failures} == {
        "QueryGenerationError"
    }
    assert {record["stage"] for record in failures} == {"query_generation"}


def test_batch_preserves_candidate_rejection_stage_and_history(monkeypatch) -> None:
    class FailingGenerator(BatchGenerationMixin):
        CHAIN_SAMPLING_JACCARD_THRESHOLD = 0.70
        SAMPLING_CONTEXT_REFRESH_K = 10

        def __init__(self) -> None:
            self.manager = SimpleNamespace(server_names=["banking"])

        def _generate_task_with_postprocess(self, *args, **kwargs):
            history = [{
                "stage": "early_trace_validation",
                "reason_code": "invalid_zero_tool_terminal",
                "attempt": 1,
            }]
            raise CandidateGenerationError(
                "candidate rejected",
                stage="early_trace_validation",
                reason="invalid_zero_tool_terminal",
                details={"task_id": "banking-42"},
                rejection_history=history,
            )

        def _generate_irrelevant_tasks(
            self, count, seed, servers, failure_callback=None,
        ):
            return []

    monkeypatch.setenv("LIVEMCP_GENERATION_MAX_WORKERS", "1")
    failures: list[dict] = []

    with pytest.raises(RuntimeError, match="produced 0 tasks"):
        FailingGenerator().generate_many(
            "banking",
            count=1,
            seed=42,
            difficulty_mix={"complete": 1.0},
            irrelevance_ratio=0.0,
            failure_callback=failures.append,
        )

    assert failures
    assert {record["stage"] for record in failures} == {
        "early_trace_validation"
    }
    assert {record["reason_code"] for record in failures} == {
        "invalid_zero_tool_terminal"
    }
    assert all(record["details"]["task_id"] == "banking-42" for record in failures)
    assert all(record["rejection_history"] for record in failures)


def test_irrelevance_query_exceptions_are_recorded_per_candidate() -> None:
    class FailingIrrelevanceGenerator(IrrelevanceGenerationMixin):
        def _generate_irrelevant_query(self, *args, **kwargs):
            raise TimeoutError("teacher timeout")

    generator = FailingIrrelevanceGenerator()
    generator.manager = SimpleNamespace(server_names=["banking"])
    generator.suite_config = SimpleNamespace(rollout={})
    generator.client = SimpleNamespace()
    generator.prompt_profile = "paper_generation_baseline_v1"
    failures: list[dict] = []

    tasks = generator._generate_irrelevant_tasks(
        1,
        100,
        ["banking"],
        failure_callback=failures.append,
    )

    assert tasks == []
    assert len(failures) == 5
    assert {record["reason_code"] for record in failures} == {
        "query_generation_exception"
    }
    assert {record["exception_type"] for record in failures} == {"TimeoutError"}
    assert [record["generation_seed"] for record in failures] == [
        100, 101, 102, 103, 104,
    ]


def test_local_irrelevance_query_uses_canonical_json_parser() -> None:
    class Generator(IrrelevanceGenerationMixin):
        pass

    class Teacher:
        def _generate_chat(self, *args, **kwargs):
            assert kwargs["json_mode"] is True
            return (
                '{"user_query":"Can you give me a weather forecast?",'
                '"unavailable_capability_class":"weather_forecast",'
                '"query_evidence_span":"weather forecast"}'
            )

    generator = Generator()
    generator.prompt_profile = SimpleNamespace(name="local_trainable_v1")
    generator.manager = SimpleNamespace(
        registry=SimpleNamespace(
            server_tools=lambda _domain: [{"name": "list_accounts"}],
        ),
    )

    query, proof = generator._generate_irrelevant_query(
        Teacher(), "banking", diversity_key="contract-test",
    )

    assert query == "Can you give me a weather forecast?"
    assert proof["unavailable_capability_class"] == "weather_forecast"
    assert proof["query_evidence_span"] == "weather forecast"


def test_irrelevance_programming_error_is_not_resampled() -> None:
    class Generator(IrrelevanceGenerationMixin):
        def _generate_irrelevant_query(self, *args, **kwargs):
            raise ImportError("broken parser dependency")

    generator = Generator()
    generator.manager = SimpleNamespace(server_names=["banking"])
    generator.suite_config = SimpleNamespace(rollout={})
    generator.client = SimpleNamespace()
    generator.prompt_profile = "paper_generation_baseline_v1"
    failures: list[dict] = []

    with pytest.raises(ImportError, match="broken parser dependency"):
        generator._generate_irrelevant_tasks(
            1,
            100,
            ["banking"],
            failure_callback=failures.append,
        )

    assert len(failures) == 1
    assert failures[0]["reason_code"] == "query_generation_exception"


def test_batch_records_and_reraises_unhandled_irrelevance_failure(
    monkeypatch,
) -> None:
    class Generator(BatchGenerationMixin):
        CHAIN_SAMPLING_JACCARD_THRESHOLD = 0.70
        SAMPLING_CONTEXT_REFRESH_K = 10

        def __init__(self) -> None:
            self.manager = SimpleNamespace(server_names=["banking"])

        def _generate_irrelevant_tasks(
            self, count, seed, servers, failure_callback=None,
        ):
            raise TypeError("forced task serialization failure")

    monkeypatch.setenv("LIVEMCP_GENERATION_MAX_WORKERS", "1")
    failures: list[dict] = []

    with pytest.raises(TypeError, match="serialization failure"):
        Generator().generate_many(
            "banking",
            count=1,
            seed=42,
            irrelevance_count=1,
            failure_callback=failures.append,
        )

    assert len(failures) == 1
    assert failures[0]["stage"] == "irrelevance_batch"
    assert failures[0]["reason_code"] == "unhandled_batch_exception"
    assert failures[0]["exception_type"] == "TypeError"
    assert "serialization failure" in failures[0]["traceback"]
