from __future__ import annotations

import pandas as pd

from src.live_mcp.corpus.ledger import CorpusLedger


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
