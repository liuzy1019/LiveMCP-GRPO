from __future__ import annotations

import pytest

from src.live_mcp.artifact import validation
from src.live_mcp.artifact.validation import validate_temporal_anchor
from src.live_mcp.generation.teacher_contracts import reference_date_for_seed


def test_temporal_anchor_rejects_generation_state_seed_drift() -> None:
    state_seed = 3712943567317000004
    good = {
        "sampling_state_seed": state_seed,
        "reference_date": reference_date_for_seed(state_seed),
    }
    validate_temporal_anchor(good)

    with pytest.raises(ValueError, match="reference_date/live-state mismatch"):
        validate_temporal_anchor({
            **good,
            "reference_date": reference_date_for_seed(
                3712943567317000043,
            ),
        })


def test_training_contract_runs_one_canonical_gate_sequence(monkeypatch) -> None:
    calls: list[str] = []

    monkeypatch.setattr(
        validation, "validate_training_artifact_evidence",
        lambda _: calls.append("training"),
    )
    monkeypatch.setattr(
        validation, "validate_ground_truth_consistency",
        lambda _extra, _ground_truth: calls.append("ground_truth"),
    )
    monkeypatch.setattr(
        validation, "validate_temporal_anchor",
        lambda _: calls.append("temporal"),
    )
    monkeypatch.setattr(
        validation, "validate_prove_corpus_evidence",
        lambda _: calls.append("prove"),
    )
    monkeypatch.setattr(
        validation, "validate_teacher_generation_evidence",
        lambda _: calls.append("teacher"),
    )
    monkeypatch.setattr(
        validation, "validate_semantic_gate_evidence",
        lambda _: calls.append("semantic"),
    )
    monkeypatch.setattr(
        validation, "build_reward_task",
        lambda _: calls.append("projection") or {"required_tool_calls": []},
    )

    task = validation.validate_artifact_contract(
        {}, require_training=True, ground_truth={},
    )

    assert task == {"required_tool_calls": []}
    assert calls == [
        "training", "ground_truth", "temporal", "prove", "teacher", "semantic",
        "projection",
    ]


@pytest.mark.parametrize(
    ("gate", "message"),
    [
        ("validate_training_artifact_evidence", "non-training artifact"),
        ("validate_prove_corpus_evidence", "missing positive PROVE replay"),
        ("validate_teacher_generation_evidence", "missing Teacher evidence"),
        ("validate_semantic_gate_evidence", "semantic quarantine"),
    ],
)
def test_training_contract_fails_closed_at_each_gate(
    monkeypatch, gate: str, message: str,
) -> None:
    for name in (
        "validate_training_artifact_evidence",
        "validate_prove_corpus_evidence",
        "validate_teacher_generation_evidence",
        "validate_semantic_gate_evidence",
    ):
        monkeypatch.setattr(validation, name, lambda _: None)
    monkeypatch.setattr(
        validation, gate,
        lambda _: (_ for _ in ()).throw(RuntimeError(message)),
    )
    monkeypatch.setattr(
        validation, "validate_ground_truth_consistency",
        lambda _extra, _ground_truth: None,
    )
    monkeypatch.setattr(validation, "validate_temporal_anchor", lambda _: None)

    with pytest.raises(RuntimeError, match=message):
        validation.validate_artifact_contract(
            {}, require_training=True, ground_truth={},
        )


def test_diagnostic_contract_does_not_require_training_purpose(monkeypatch) -> None:
    monkeypatch.setattr(
        validation, "validate_training_artifact_evidence",
        lambda _: (_ for _ in ()).throw(AssertionError("training gate called")),
    )
    monkeypatch.setattr(validation, "validate_prove_corpus_evidence", lambda _: None)
    monkeypatch.setattr(
        validation, "validate_teacher_generation_evidence", lambda _: None,
    )
    monkeypatch.setattr(validation, "validate_semantic_gate_evidence", lambda _: None)
    monkeypatch.setattr(validation, "validate_temporal_anchor", lambda _: None)
    monkeypatch.setattr(
        validation, "build_reward_task", lambda _: {"required_tool_calls": []},
    )

    validation.validate_artifact_contract({}, require_training=False)


def test_real_training_contract_rejects_missing_prove_evidence() -> None:
    with pytest.raises(RuntimeError, match="missing positive PROVE replay"):
        validation.validate_artifact_contract(
            {
                "prompt_profile": "local_trainable_v1",
                "semantic_gate_profile": "deterministic_v1",
                "artifact_purpose": "training_candidate",
                "sampling_state_seed": 42,
                "reference_date": reference_date_for_seed(42),
            },
            require_training=True,
        )
