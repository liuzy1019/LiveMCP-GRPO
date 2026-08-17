from src.live_mcp.domain_contracts.semantic_policies import (
    DOMAIN_LABEL_POLICIES,
    evaluate_domain_label_issue,
)
from src.live_mcp.corpus.local_quality import (
    _deterministic_label_issue,
    _profile_local_trace_issue,
    evaluate_persisted_candidate_quality,
)
from src.live_mcp.prompt_profiles import requires_outcome_replay


def test_domain_label_policy_registry_is_explicit() -> None:
    assert set(DOMAIN_LABEL_POLICIES) == {"crm", "filesystem", "food_delivery"}
    assert evaluate_domain_label_issue("banking", "", "final_answer", [], 0) == ""


def test_crm_policy_rejects_create_as_update() -> None:
    issue = evaluate_domain_label_issue(
        "crm",
        "Update task_123 priority to high",
        "report_error",
        [{
            "tool_name": "create_task",
            "success": True,
            "state_changed": True,
        }],
        0,
    )
    assert "create_as_update" in issue


def test_filesystem_policy_requires_persistent_mutation() -> None:
    issue = evaluate_domain_label_issue(
        "filesystem",
        "Remove the first line from /tmp/a.txt",
        "final_answer",
        [{"tool_name": "head", "success": True}],
        1,
    )
    assert "readonly_persistence" in issue


def test_food_policy_rejects_explicit_size_downgrade() -> None:
    issue = evaluate_domain_label_issue(
        "food_delivery",
        "Order a large pizza",
        "final_answer",
        [
            {
                "tool_name": "create_order",
                "success": False,
                "arguments": {"items": [{"name": "large pizza"}]},
            },
            {
                "tool_name": "create_order",
                "success": True,
                "arguments": {"items": [{"name": "pizza"}]},
            },
        ],
        2,
    )
    assert "food_size_downgrade" in issue


def _filesystem_readonly_persistence_row(
    *, prompt_profile: str, semantic_gate_profile: str,
) -> dict:
    return {
        "domain": "filesystem",
        "prompt_profile": prompt_profile,
        "semantic_gate_profile": semantic_gate_profile,
        "teacher_round_trace": [{
            "round_idx": 0,
            "user_query": "Remove the first line from /tmp/a.txt",
            "oracle_calls": [{"action": "final_answer", "arguments": {}}],
            "execution_history": [{"tool_name": "head", "success": True}],
        }],
    }


def test_paper_diagnostic_profile_does_not_apply_local_label_gate() -> None:
    extra = _filesystem_readonly_persistence_row(
        prompt_profile="paper_generation_baseline_v1",
        semantic_gate_profile="diagnostic_only",
    )
    assert "readonly_persistence" in _deterministic_label_issue(extra)
    assert _profile_local_trace_issue(extra) == ""


def test_local_deterministic_profile_applies_local_label_gate() -> None:
    extra = _filesystem_readonly_persistence_row(
        prompt_profile="local_trainable_v1",
        semantic_gate_profile="deterministic_v1",
    )
    assert "readonly_persistence" in _profile_local_trace_issue(extra)
    finding = evaluate_persisted_candidate_quality(extra)
    assert finding is not None
    assert finding.stage == "local_quality"
    assert "readonly_persistence" in finding.quality_issue


def test_outcome_replay_gate_is_owned_by_local_profile() -> None:
    assert requires_outcome_replay("local_trainable_v1") is True
    assert requires_outcome_replay("paper_generation_baseline_v1") is False
