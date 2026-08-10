import pytest

from src.live_mcp.artifact.dependency_contract import validate_dependency_artifact


def _row() -> dict:
    return {
        "scenario_type": "normal_safe_success",
        "oracle_calls": [
            {"action": "tool_call", "tool_name": "search"},
            {"action": "tool_call", "tool_name": "detail"},
            {"action": "tool_call", "tool_name": "mutate"},
            {"action": "final_answer", "tool_name": "final_answer"},
        ],
        "source_chain_seed": ["search", "mutate"],
        "source_chain_edges": [{
            "source_capability": "search",
            "target_capability": "mutate",
            "relation": "explicit",
        }],
        "chain_seed": ["search", "mutate"],
        "realized_tool_sequence": ["search", "detail", "mutate"],
        "dependency_call_indices": [0, 2],
        "auxiliary_call_indices": [1],
        "dependency_edges": [[0, 2]],
        "verified_dependency_evidence": [{
            "source_capability": "search",
            "target_capability": "mutate",
            "evidence_type": "explicit_value_binding",
            "source_call_index": 0,
            "target_call_index": 2,
        }],
    }


def test_dependency_artifact_accepts_exact_canonical_partition() -> None:
    validate_dependency_artifact(_row())


def test_dependency_artifact_uses_verified_recovery_source_index() -> None:
    row = _row()
    row["oracle_calls"].insert(
        0, {"action": "tool_call", "tool_name": "search"},
    )
    row["realized_tool_sequence"] = ["search", "search", "detail", "mutate"]
    row["dependency_call_indices"] = [1, 3]
    row["auxiliary_call_indices"] = [0, 2]
    row["dependency_edges"] = [[1, 3]]
    row["verified_dependency_evidence"][0]["source_call_index"] = 1
    row["verified_dependency_evidence"][0]["target_call_index"] = 3

    validate_dependency_artifact(row)


def test_dependency_artifact_rejects_auxiliary_call_as_reward_edge() -> None:
    row = _row()
    row["chain_seed"] = ["search", "detail", "mutate"]
    row["dependency_call_indices"] = [0, 1, 2]
    row["auxiliary_call_indices"] = []
    row["dependency_edges"] = [[0, 1], [1, 2]]

    with pytest.raises(RuntimeError, match="preserve its sampled source chain"):
        validate_dependency_artifact(row)


def test_non_success_artifact_cannot_carry_reward_dependency_edges() -> None:
    row = _row()
    row["scenario_type"] = "missing_function"

    with pytest.raises(RuntimeError, match="must not carry reward dependencies"):
        validate_dependency_artifact(row)
