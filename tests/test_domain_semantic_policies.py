from src.live_mcp.domain_contracts.semantic_policies import (
    DOMAIN_LABEL_POLICIES,
    evaluate_domain_label_issue,
)


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
