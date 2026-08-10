from src.live_mcp.registry.entity_state_contracts import entity_state_is_known


def test_state_sensitive_target_fails_closed_when_field_is_missing() -> None:
    assert not entity_state_is_known(
        "food_delivery", "create_order", "restaurant", {"name": "Cafe"}
    )


def test_equivalent_observation_field_satisfies_state_knowledge_contract() -> None:
    assert entity_state_is_known(
        "food_delivery", "create_order", "restaurant", {"items": []}
    )


def test_existence_only_target_does_not_invent_a_state_requirement() -> None:
    assert entity_state_is_known("crm", "update_contact", "contact", {})
