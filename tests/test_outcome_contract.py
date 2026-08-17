from src.live_mcp.contracts.outcome import mutation_outcome_issue


def _is_mutating(name: str) -> bool:
    return name == "update_item"


def test_mutation_outcome_requires_state_criteria() -> None:
    assert mutation_outcome_issue(
        tool_names=["list_items", "update_item"],
        success_criteria=[],
        criterion_provenance=[],
        is_mutating=_is_mutating,
    ) == "mutation_success_criteria_missing:tools=['update_item']"


def test_readonly_outcome_can_have_no_state_criteria() -> None:
    assert mutation_outcome_issue(
        tool_names=["list_items"],
        success_criteria=[],
        criterion_provenance=[],
        is_mutating=_is_mutating,
    ) is None


def test_mutation_outcome_requires_attributed_criteria() -> None:
    criteria = [{"path": "items.item_1.value", "value": "new"}]

    assert mutation_outcome_issue(
        tool_names=["update_item"],
        success_criteria=criteria,
        criterion_provenance=[],
        is_mutating=_is_mutating,
    ) == "mutation_success_criteria_provenance_missing:indices=[0]"
    assert mutation_outcome_issue(
        tool_names=["update_item"],
        success_criteria=criteria,
        criterion_provenance=[{
            "criterion_index": 0,
            "source_calls": [{"tool_name": "update_item"}],
        }],
        is_mutating=_is_mutating,
    ) is None
