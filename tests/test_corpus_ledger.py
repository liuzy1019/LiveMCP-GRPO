from __future__ import annotations

import pandas as pd
import pytest

from src.live_mcp.corpus.ledger import CorpusLedger
from src.live_mcp.generation.mix_policy import (
    PROVE_DIFFICULTY_MIX,
    default_difficulty_mix,
)


def _rows(domain: str, difficulty: str, count: int) -> list[dict]:
    return [
        {
            "uid": f"{domain}-{difficulty}-{index}",
            "extra_info": {
                "domain": domain,
                "difficulty": difficulty,
                "oracle_calls": [{
                    "action": "tool_call",
                    "tool_name": f"tool-{difficulty}-{index}",
                }],
            },
        }
        for index in range(count)
    ]


def _irrelevance_rows(domain: str, count: int) -> list[dict]:
    return [
        {
            "uid": f"{domain}-irrelevant-{index}",
            "extra_info": {
                "domain": domain,
                "difficulty": "minimal",
                "generation_method": "irrelevant_teacher_fsm",
                "oracle_calls": [],
            },
        }
        for index in range(count)
    ]


def test_ledger_reports_exact_global_accepted_deficits() -> None:
    pool = pd.DataFrame(
        _rows("calendar", "complete", 5)
        + _rows("calendar", "missing", 8)
        + _rows("calendar", "minimal", 7)
    )
    ledger = CorpusLedger.from_pool(pool, {"calendar": 20})
    assert ledger.targets["calendar"] == {
        "complete": 12,
        "missing": 4,
        "minimal": 4,
    }
    assert ledger.deficits == {"calendar": {"complete": 7}}
    assert ledger.topup_requests() == [("calendar", "complete", 7)]
    assert len(ledger.retained_tool_sequences["calendar"]) == 20
    assert ledger.report()["retained_tool_sequences_by_domain"]["calendar"][0] == [
        "tool-complete-0"
    ]


def test_prove_difficulty_default_is_caller_owned() -> None:
    first = default_difficulty_mix()
    first["complete"] = 0.0
    assert default_difficulty_mix() == dict(PROVE_DIFFICULTY_MIX)
    assert default_difficulty_mix() == {
        "complete": 0.60,
        "missing": 0.20,
        "minimal": 0.20,
    }


def test_ledger_selects_exact_strata_after_topup() -> None:
    pool = pd.DataFrame(
        _rows("calendar", "complete", 14)
        + _rows("calendar", "missing", 8)
        + _rows("calendar", "minimal", 7)
    )
    ledger = CorpusLedger.from_pool(pool, {"calendar": 20})
    assert ledger.complete
    selected = ledger.select(pool)
    counts = selected["extra_info"].map(
        lambda value: value["difficulty"]
    ).value_counts().to_dict()
    assert counts == {"complete": 12, "missing": 4, "minimal": 4}


def test_ledger_applies_difficulty_mix_only_to_non_irrelevance_rows() -> None:
    pool = pd.DataFrame(
        _rows("banking", "complete", 57)
        + _rows("banking", "missing", 19)
        + _rows("banking", "minimal", 19)
        + _irrelevance_rows("banking", 5)
    )

    ledger = CorpusLedger.from_pool(
        pool,
        {"banking": 100},
        irrelevance_count=5,
    )

    assert ledger.targets["banking"] == {
        "complete": 57,
        "missing": 19,
        "minimal": 19,
    }
    assert ledger.irrelevance_targets == {"banking": 5}
    assert ledger.irrelevance_available == {"banking": 5}
    assert ledger.irrelevance_deficits == {}
    assert ledger.complete


def test_ledger_reports_and_selects_exact_irrelevance_deficit() -> None:
    pool = pd.DataFrame(
        _rows("banking", "complete", 58)
        + _rows("banking", "missing", 20)
        + _rows("banking", "minimal", 19)
        + _irrelevance_rows("banking", 3)
    )
    ledger = CorpusLedger.from_pool(
        pool,
        {"banking": 100},
        irrelevance_count=5,
    )

    assert ledger.deficits == {"banking": {}}
    assert ledger.irrelevance_deficits == {"banking": 2}
    assert not ledger.complete
    assert ledger.topup_requests() == [
        ("banking", "irrelevance", 2),
    ]

    complete_pool = pd.concat(
        [pool, pd.DataFrame(_irrelevance_rows("banking", 2))],
        ignore_index=True,
    )
    complete_ledger = CorpusLedger.from_pool(
        complete_pool,
        {"banking": 100},
        irrelevance_count=5,
    )
    selected = complete_ledger.select(complete_pool)
    selected_irrelevance = selected["extra_info"].map(
        lambda extra: extra.get("generation_method")
        == "irrelevant_teacher_fsm"
    )
    assert int(selected_irrelevance.sum()) == 5
    assert len(selected) == 100


def test_ledger_allocates_global_irrelevance_exactly_across_domains() -> None:
    pool = pd.DataFrame(
        _rows("banking", "complete", 2)
        + _rows("calendar", "complete", 1)
        + _irrelevance_rows("banking", 1)
        + _irrelevance_rows("calendar", 1)
    )
    ledger = CorpusLedger.from_pool(
        pool,
        {"banking": 3, "calendar": 2},
        irrelevance_count=2,
        fixed_difficulty="complete",
    )

    assert ledger.irrelevance_targets == {"banking": 1, "calendar": 1}
    assert sum(ledger.irrelevance_targets.values()) == 2
    assert ledger.targets == {
        "banking": {"complete": 2, "missing": 0, "minimal": 0},
        "calendar": {"complete": 1, "missing": 0, "minimal": 0},
    }
    assert ledger.complete


def test_ledger_rejects_impossible_irrelevance_target() -> None:
    with pytest.raises(ValueError, match="irrelevance_count"):
        CorpusLedger.from_pool(
            pd.DataFrame(_irrelevance_rows("banking", 1)),
            {"banking": 1},
            irrelevance_count=2,
        )


def test_ledger_sums_disjoint_difficulty_and_irrelevance_deficits() -> None:
    pool = pd.DataFrame(
        _rows("banking", "complete", 55)
        + _rows("banking", "missing", 19)
        + _rows("banking", "minimal", 19)
        + _irrelevance_rows("banking", 3)
    )
    ledger = CorpusLedger.from_pool(
        pool,
        {"banking": 100},
        irrelevance_count=5,
    )

    assert ledger.deficits == {"banking": {"complete": 2}}
    assert ledger.irrelevance_deficits == {"banking": 2}
    assert ledger.total_deficits_by_domain == {"banking": 4}
    assert ledger.report()["stratum_deficits_by_domain"] == {"banking": 4}


def test_ledger_reports_base_and_candidate_retained_sequences_without_counting_base() -> None:
    candidate_pool = pd.DataFrame(
        _rows("banking", "complete", 1)
    )
    retained_pool = pd.DataFrame([
        {
            "uid": "base-row",
            "extra_info": {
                "domain": "banking",
                "difficulty": "complete",
                "oracle_calls": [{
                    "action": "tool_call",
                    "tool_name": "base_occupied_tool",
                }],
            },
        },
        *candidate_pool.to_dict("records"),
    ])

    ledger = CorpusLedger.from_pool(
        candidate_pool,
        {"banking": 1},
        fixed_difficulty="complete",
        retained_pool=retained_pool,
    )

    assert ledger.available == {
        "banking": {"complete": 1, "missing": 0, "minimal": 0}
    }
    assert ledger.retained_tool_sequences == {
        "banking": [["base_occupied_tool"], ["tool-complete-0"]]
    }


def test_ledger_allocates_measured_candidate_budget_across_exact_strata() -> None:
    pool = pd.DataFrame(
        _rows("banking", "complete", 55)
        + _rows("banking", "missing", 18)
        + _rows("banking", "minimal", 18)
        + _irrelevance_rows("banking", 4)
    )
    ledger = CorpusLedger.from_pool(
        pool,
        {"banking": 100},
        irrelevance_count=5,
    )

    requests = ledger.candidate_topup_requests({"banking": 12})

    assert sum(item["missing"] for item in requests) == 5
    assert sum(item["candidate_count"] for item in requests) == 12
    assert {(item["stratum"], item["missing"]) for item in requests} == {
        ("complete", 2),
        ("missing", 1),
        ("minimal", 1),
        ("irrelevance", 1),
    }
    assert all(
        item["candidate_count"] >= item["missing"] for item in requests
    )
