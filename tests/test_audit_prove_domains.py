from scripts.audit_prove_domains import (
    _classify_target_availability,
    _entity_state_distributions,
)


def test_availability_audit_distinguishes_chain_reachable_target() -> None:
    availability = {
        "delete_contact": {"has_usable_entities": False},
        "update_contact": {"has_usable_entities": False},
        "list_contacts": {"has_usable_entities": True},
    }
    baseline, reachable, unreachable = _classify_target_availability(
        availability,
        [["create_contact", "delete_contact"]],
    )

    assert sorted(baseline) == ["delete_contact", "update_contact"]
    assert reachable == ["delete_contact"]
    assert list(unreachable) == ["update_contact"]


def test_entity_state_distributions_are_domain_neutral() -> None:
    records = [
        {"type": "order", "data": {"status": "pending", "total": 5}},
        {"type": "order", "data": {"status": "shipped", "total": 8}},
        {"type": "product", "data": {"stock": 0, "name": "x"}},
        {"type": "product", "data": {"stock": 2, "name": "y"}},
        {"type": "event", "data": {"recurrence": None, "reminders": []}},
        {"type": "event", "data": {"recurrence": "weekly", "reminders": [10]}},
    ]
    assert _entity_state_distributions(records) == {
        "order": {"status": {"pending": 1, "shipped": 1}},
        "product": {"stock_availability": {"available": 1, "empty": 1}},
        "event": {
            "recurrence": {"empty": 1, "present": 1},
            "reminders": {"empty": 1, "present": 1},
        },
    }
