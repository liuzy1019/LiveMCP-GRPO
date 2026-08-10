from __future__ import annotations

import copy

import numpy as np
import pytest

import src.live_mcp.corpus.recertify as recertify
from src.live_mcp.corpus.recertify import _list_field, _recertify_row


def _runtime_row() -> dict:
    return {
        "uid": "calendar-recertify",
        "reward_model": {
            "ground_truth": {
                "oracle_calls": "[]",
                "success_criteria": "[]",
            },
        },
        "extra_info": {
            "domain": "calendar",
            "session_seed": 7,
            "hidden_tools": "[]",
            "state_profiles": {"calendar": "baseline"},
            "reward_fingerprint": "legacy-reward",
        },
    }


@pytest.fixture(autouse=True)
def _stub_non_replay_contracts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(recertify, "validate_prove_corpus_evidence", lambda _: None)
    monkeypatch.setattr(recertify, "validate_teacher_generation_evidence", lambda _: None)
    monkeypatch.setattr(recertify, "validate_environment_metadata", lambda *_, **__: None)
    monkeypatch.setattr(recertify, "compute_initial_state_hashes", lambda *_: {})
    monkeypatch.setattr(recertify, "_current_tools", lambda _: {"calendar": []})
    monkeypatch.setattr(
        recertify, "build_reward_task", lambda _: {"required_tool_calls": []},
    )


def test_recertify_row_updates_only_reward_and_fresh_replay_evidence() -> None:
    row = _runtime_row()
    original = copy.deepcopy(row)

    migrated = _recertify_row(
        row,
        expected_source_fingerprint="legacy-reward",
        current_fingerprint="49af6d51611cc335",
        runtime_observation_budget=4096,
        manager=object(),
        executor=object(),
        replay_fn=lambda **_: (True, 0.0, 0, 1, True, 0),
    )

    assert row == original
    assert migrated["extra_info"]["reward_fingerprint"] == "49af6d51611cc335"
    assert migrated["extra_info"]["canonical_replay_num_calls"] == 1
    assert migrated["uid"] == original["uid"]
    assert migrated["reward_model"] == original["reward_model"]
    changed = {
        key for key in migrated["extra_info"]
        if migrated["extra_info"].get(key) != original["extra_info"].get(key)
    }
    assert changed == {
        "reward_fingerprint",
        "canonical_replay_valid",
        "canonical_replay_error_rate",
        "canonical_replay_num_errors",
        "canonical_replay_num_calls",
        "canonical_replay_criteria_ok",
        "canonical_replay_criteria_failed",
        "reward_profile_fingerprints",
    }


def test_recertify_row_rejects_unexpected_source_fingerprint() -> None:
    with pytest.raises(RuntimeError, match="unexpected source reward_fingerprint"):
        _recertify_row(
            _runtime_row(),
            expected_source_fingerprint="another-legacy-reward",
            current_fingerprint="69a7f7786d5935aa",
            runtime_observation_budget=4096,
            manager=object(),
            executor=object(),
            replay_fn=lambda **_: (True, 0.0, 0, 1, True, 0),
        )


def test_recertify_row_rejects_failed_fresh_replay() -> None:
    with pytest.raises(RuntimeError, match="fresh canonical replay rejected"):
        _recertify_row(
            _runtime_row(),
            expected_source_fingerprint="legacy-reward",
            current_fingerprint="69a7f7786d5935aa",
            runtime_observation_budget=4096,
            manager=object(),
            executor=object(),
            replay_fn=lambda **_: (False, 1.0, 1, 1, False, 1),
        )


def test_list_field_accepts_parquet_numpy_arrays() -> None:
    assert _list_field(np.array(["tool_a", "tool_b"]), "hidden_tools") == [
        "tool_a", "tool_b",
    ]
