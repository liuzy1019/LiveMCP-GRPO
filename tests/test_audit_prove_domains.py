import sys
from types import SimpleNamespace

import pytest

import scripts.audit_prove_domains as audit_module
from scripts.audit_prove_domains import (
    _classify_target_availability,
    _entity_state_distributions,
    _query_state_view_audit,
)


def test_cli_defaults_to_local_trainable_profile(monkeypatch, capsys) -> None:
    captured: dict[str, object] = {}

    def fake_audit(domains, seed, prompt_profile):
        captured.update(
            domains=domains, seed=seed, prompt_profile=prompt_profile,
        )
        return {}

    monkeypatch.setattr(audit_module, "load_suite_config", lambda _path: SimpleNamespace(
        servers=[SimpleNamespace(name="banking", enabled=True)],
    ))
    monkeypatch.setattr(audit_module, "audit_domains", fake_audit)
    monkeypatch.setattr(sys, "argv", ["audit_prove_domains.py"])

    assert audit_module.main() == 0
    assert captured == {
        "domains": ["banking"],
        "seed": 42,
        "prompt_profile": "local_trainable_v1",
    }
    assert capsys.readouterr().out.strip() == "{}"


def test_cli_rejects_unknown_prompt_profile(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["audit_prove_domains.py", "--prompt-profile", "prove_strict_v1"],
    )
    with pytest.raises(SystemExit, match="2"):
        audit_module.main()


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


def test_query_state_view_audit_distinguishes_paper_and_local_contracts() -> None:
    context = {
        "entity_ids": [{"type": "account", "id": "acc_s42_003"}],
        "entity_summaries": [
            "  acc_s42_003 (account): type=business, balance=200"
        ],
        "entity_records": [{
            "type": "account",
            "id": "acc_s42_003",
            "data": {"type": "business", "balance": 200},
        }],
    }

    paper = _query_state_view_audit(
        context, "banking", natural_selector=False,
    )
    local = _query_state_view_audit(
        context, "banking", natural_selector=True,
    )

    assert paper["exposed_opaque_id_count"] == 1
    assert paper["omitted_opaque_candidate_count"] == 0
    assert paper["profile_contract_satisfied"] is True
    assert local["exposed_opaque_id_count"] == 0
    assert local["empty_selector_count"] == 0
    assert local["omitted_opaque_candidate_count"] == 0
    assert local["profile_contract_satisfied"] is True


def test_query_state_view_audit_treats_seeded_business_field_as_opaque() -> None:
    context = {
        "entity_ids": [{"type": "invoice", "id": "inv_s42_001"}],
        "entity_summaries": [
            "  inv_s42_001 (invoice): customer=Acme status=pending"
        ],
        "entity_records": [{
            "type": "invoice",
            "id": "inv_s42_001",
            "data": {
                "invoice_id": "inv_s42_001",
                "customer": "Acme",
                "status": "pending",
            },
        }],
    }

    local = _query_state_view_audit(
        context, "payments", natural_selector=True,
    )

    assert local["opaque_source_id_count"] == 1
    assert local["public_business_reference_count"] == 0
    assert local["exposed_opaque_id_count"] == 0
    assert local["profile_contract_satisfied"] is True
