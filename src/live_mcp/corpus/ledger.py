"""Persistent corpus allocation contract shared by merge and top-up.

The ledger is intentionally model-agnostic.  It derives exact accepted-row
strata from the current globally deduplicated pool; launchers only execute the
resulting requests and never recompute the distribution independently.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.live_mcp.corpus.merge_validation import _as_extra, _row_tool_sequence


DIFFICULTIES = ("complete", "missing", "minimal")
PAPER_DIFFICULTY_MIX = {
    "complete": 0.60,
    "missing": 0.20,
    "minimal": 0.20,
}


def _largest_remainder(total: int) -> dict[str, int]:
    raw = {
        key: total * PAPER_DIFFICULTY_MIX[key]
        for key in DIFFICULTIES
    }
    quotas = {key: int(raw[key]) for key in DIFFICULTIES}
    remainder = total - sum(quotas.values())
    order = sorted(
        DIFFICULTIES,
        key=lambda key: (-(raw[key] - quotas[key]), DIFFICULTIES.index(key)),
    )
    for key in order[:remainder]:
        quotas[key] += 1
    return quotas


def _difficulty(value: Any) -> str:
    extra = _as_extra(value)
    return str(extra.get("difficulty") or "")


@dataclass(frozen=True)
class CorpusLedger:
    targets: dict[str, dict[str, int]]
    available: dict[str, dict[str, int]]
    deficits: dict[str, dict[str, int]]
    retained_tool_sequences: dict[str, list[list[str]]]

    @classmethod
    def from_pool(
        cls,
        pool: pd.DataFrame,
        required_by_domain: dict[str, int],
    ) -> "CorpusLedger":
        targets = {
            domain: _largest_remainder(required)
            for domain, required in required_by_domain.items()
        }
        available: dict[str, dict[str, int]] = {}
        deficits: dict[str, dict[str, int]] = {}
        for domain in required_by_domain:
            domain_rows = pool.loc[
                pool["extra_info"].map(
                    lambda value: str(_as_extra(value).get("domain") or "")
                    == domain
                )
            ]
            counts = {
                difficulty: int(
                    domain_rows["extra_info"].map(_difficulty).eq(difficulty).sum()
                )
                for difficulty in DIFFICULTIES
            }
            available[domain] = counts
            domain_deficits = {
                difficulty: max(0, targets[domain][difficulty] - counts[difficulty])
                for difficulty in DIFFICULTIES
            }
            deficits[domain] = {
                key: value for key, value in domain_deficits.items() if value
            }
        retained_tool_sequences = {
            domain: [
                sequence
                for _, row in pool.iterrows()
                if str(_as_extra(row["extra_info"]).get("domain") or "") == domain
                if (sequence := _row_tool_sequence(row, mode="prove"))
            ]
            for domain in required_by_domain
        }
        return cls(
            targets=targets,
            available=available,
            deficits=deficits,
            retained_tool_sequences=retained_tool_sequences,
        )

    @property
    def complete(self) -> bool:
        return not any(self.deficits.values())

    def report(self) -> dict[str, Any]:
        return {
            "difficulty_targets_by_domain": self.targets,
            "difficulty_available_by_domain": self.available,
            "difficulty_deficits_by_domain": self.deficits,
            "retained_tool_sequences_by_domain": self.retained_tool_sequences,
        }

    def topup_requests(self) -> list[tuple[str, str, int]]:
        return [
            (domain, difficulty, count)
            for domain in sorted(self.deficits)
            for difficulty, count in self.deficits[domain].items()
            if count > 0
        ]

    def select(self, pool: pd.DataFrame) -> pd.DataFrame:
        if not self.complete:
            raise ValueError("cannot select from a corpus with stratum deficits")
        selected: list[pd.DataFrame] = []
        for domain, targets in self.targets.items():
            domain_mask = pool["extra_info"].map(
                lambda value: str(_as_extra(value).get("domain") or "") == domain
            )
            for difficulty in DIFFICULTIES:
                target = targets[difficulty]
                if not target:
                    continue
                difficulty_mask = pool["extra_info"].map(_difficulty).eq(difficulty)
                selected.append(pool.loc[domain_mask & difficulty_mask].head(target))
        if not selected:
            return pool.iloc[0:0].copy()
        return pd.concat(selected, ignore_index=True)


__all__ = ["CorpusLedger", "DIFFICULTIES", "PAPER_DIFFICULTY_MIX"]
