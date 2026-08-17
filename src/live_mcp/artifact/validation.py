"""Shared static validation for serialized LiveMCP task artifacts.

Runtime-dependent checks such as server schema, initial-state and observation
budget compatibility remain at the consumer that owns that runtime context.
This module owns the row-level contract that must not drift between corpus,
training, rollout and reward processes.
"""

from __future__ import annotations

from typing import Any
import json

from src.live_mcp.artifact.reward_task import (
    build_reward_task,
    validate_ground_truth_consistency,
)
from src.live_mcp.generation.teacher_contracts import reference_date_for_seed
from src.live_mcp.registry.environment_metadata import (
    validate_prove_corpus_evidence,
    validate_semantic_gate_evidence,
    validate_teacher_generation_evidence,
    validate_training_artifact_evidence,
)


_GROUND_TRUTH_UNSET = object()


def _json_value(value: Any, field: str, expected_type: type) -> Any:
    if hasattr(value, "as_py"):
        value = value.as_py()
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid {field} JSON") from exc
    if not isinstance(value, expected_type):
        raise ValueError(f"missing or invalid {field}")
    return value


def validate_temporal_anchor(extra_info: dict[str, Any]) -> None:
    """Require artifact time to match the seed that created Live-State."""
    state_seed = extra_info.get("sampling_state_seed")
    if state_seed is None:
        raise ValueError("missing sampling_state_seed")
    expected = reference_date_for_seed(int(state_seed))
    actual = str(extra_info.get("reference_date") or "")
    if actual != expected:
        raise ValueError(
            "reference_date/live-state mismatch: "
            f"artifact={actual!r}, expected={expected!r}, "
            f"sampling_state_seed={state_seed}"
        )


def validate_artifact_contract(
    extra_info: dict[str, Any],
    *,
    require_training: bool,
    ground_truth: Any = _GROUND_TRUTH_UNSET,
) -> dict[str, Any]:
    """Validate and project one canonical serialized task row."""
    if require_training:
        validate_training_artifact_evidence(extra_info)
    if ground_truth is not _GROUND_TRUTH_UNSET:
        validate_ground_truth_consistency(extra_info, ground_truth)
    validate_temporal_anchor(extra_info)
    validate_prove_corpus_evidence(extra_info)
    validate_teacher_generation_evidence(extra_info)
    validate_semantic_gate_evidence(extra_info)
    return build_reward_task(extra_info)


__all__ = [
    "validate_artifact_contract",
    "validate_temporal_anchor",
]
