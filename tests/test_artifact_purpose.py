import pytest

from src.live_mcp.corpus.semantic_core import (
    expected_artifact_purpose,
    validate_artifact_purpose,
)
from src.live_mcp.registry.environment_metadata import (
    validate_training_artifact_evidence,
)
from src.live_mcp.corpus.profile import _prove_proxy_bucket


def test_profile_pairs_have_explicit_non_overlapping_purposes() -> None:
    assert expected_artifact_purpose({
        "prompt_profile": "paper_generation_baseline_v1",
        "semantic_gate_profile": "diagnostic_only",
    }) == "paper_audit"
    assert expected_artifact_purpose({
        "prompt_profile": "local_trainable_v1",
        "semantic_gate_profile": "deterministic_v1",
    }) == "training_candidate"
    assert expected_artifact_purpose({
        "prompt_profile": "paper_generation_baseline_v1",
        "semantic_gate_profile": "deterministic_v1",
    }) == "experiment"


def test_training_boundary_rejects_paper_audit_rows() -> None:
    row = {
        "prompt_profile": "paper_generation_baseline_v1",
        "semantic_gate_profile": "diagnostic_only",
        "artifact_purpose": "paper_audit",
    }

    with pytest.raises(RuntimeError, match="not training-consumable"):
        validate_training_artifact_evidence(row)


def test_training_boundary_accepts_matching_training_candidate() -> None:
    row = {
        "prompt_profile": "local_trainable_v1",
        "semantic_gate_profile": "deterministic_v1",
        "artifact_purpose": "training_candidate",
    }

    validate_training_artifact_evidence(row)


def test_fixed_attempt_diagnostic_is_never_training_candidate() -> None:
    row = {
        "prompt_profile": "local_trainable_v1",
        "semantic_gate_profile": "deterministic_v1",
        "fixed_attempt_budget": True,
        "artifact_purpose": "experiment",
    }

    assert expected_artifact_purpose(row) == "experiment"
    with pytest.raises(ValueError, match="not training-consumable"):
        validate_artifact_purpose(row, require_training=True)


def test_persisted_purpose_cannot_override_profile_pair() -> None:
    row = {
        "prompt_profile": "paper_generation_baseline_v1",
        "semantic_gate_profile": "diagnostic_only",
        "artifact_purpose": "training_candidate",
    }

    with pytest.raises(ValueError, match="mismatch"):
        validate_artifact_purpose(row, require_training=True)


def test_prove_mix_builder_rejects_audit_row_before_reward_projection() -> None:
    row = {
        "prompt_profile": "paper_generation_baseline_v1",
        "semantic_gate_profile": "diagnostic_only",
        "artifact_purpose": "paper_audit",
    }

    with pytest.raises(RuntimeError, match="not training-consumable"):
        _prove_proxy_bucket(row)
