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
from src.live_mcp.generation.mix_policy import (
    DIFFICULTIES,
    PROVE_DIFFICULTY_MIX,
    largest_remainder_mix_quotas,
)


def _largest_remainder(
    total: int,
    fixed_difficulty: str | None = None,
) -> dict[str, int]:
    if fixed_difficulty is not None:
        if fixed_difficulty not in DIFFICULTIES:
            raise ValueError(f"unknown fixed difficulty: {fixed_difficulty!r}")
        return {
            difficulty: total if difficulty == fixed_difficulty else 0
            for difficulty in DIFFICULTIES
        }
    return largest_remainder_mix_quotas(total, PROVE_DIFFICULTY_MIX)


def _difficulty(value: Any) -> str:
    extra = _as_extra(value)
    return str(extra.get("difficulty") or "")


def _is_irrelevance(value: Any) -> bool:
    extra = _as_extra(value)
    return str(extra.get("generation_method") or "") == "irrelevant_teacher_fsm"


def _allocate_irrelevance(
    total: int,
    required_by_domain: dict[str, int],
) -> dict[str, int]:
    required_total = sum(required_by_domain.values())
    if not 0 <= total <= required_total:
        raise ValueError(
            f"irrelevance_count must be in [0, {required_total}], got {total}"
        )
    if required_total == 0:
        return {domain: 0 for domain in required_by_domain}
    exact = {
        domain: total * required / required_total
        for domain, required in required_by_domain.items()
    }
    allocated = {domain: int(value) for domain, value in exact.items()}
    remainder = total - sum(allocated.values())
    order = sorted(
        required_by_domain,
        key=lambda domain: (
            -(exact[domain] - allocated[domain]),
            domain,
        ),
    )
    for domain in order[:remainder]:
        allocated[domain] += 1
    return allocated


@dataclass(frozen=True)
class CorpusLedger:
    targets: dict[str, dict[str, int]]
    available: dict[str, dict[str, int]]
    deficits: dict[str, dict[str, int]]
    irrelevance_targets: dict[str, int]
    irrelevance_available: dict[str, int]
    irrelevance_deficits: dict[str, int]
    retained_tool_sequences: dict[str, list[list[str]]]

    @classmethod
    def from_pool(
        cls,
        pool: pd.DataFrame,
        required_by_domain: dict[str, int],
        *,
        irrelevance_count: int = 0,
        fixed_difficulty: str | None = None,
        retained_pool: pd.DataFrame | None = None,
    ) -> "CorpusLedger":
        irrelevance_targets = _allocate_irrelevance(
            irrelevance_count, required_by_domain,
        )
        targets = {
            domain: _largest_remainder(
                required - irrelevance_targets[domain],
                fixed_difficulty,
            )
            for domain, required in required_by_domain.items()
        }
        available: dict[str, dict[str, int]] = {}
        deficits: dict[str, dict[str, int]] = {}
        irrelevance_available: dict[str, int] = {}
        for domain in required_by_domain:
            domain_rows = pool.loc[
                pool["extra_info"].map(
                    lambda value: str(_as_extra(value).get("domain") or "")
                    == domain
                )
            ]
            irrelevance_mask = domain_rows["extra_info"].map(_is_irrelevance)
            irrelevance_available[domain] = int(irrelevance_mask.sum())
            normal_rows = domain_rows.loc[~irrelevance_mask]
            counts = {
                difficulty: int(
                    normal_rows["extra_info"].map(_difficulty).eq(difficulty).sum()
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
        irrelevance_deficits = {
            domain: irrelevance_targets[domain] - irrelevance_available[domain]
            for domain in required_by_domain
            if irrelevance_available[domain] < irrelevance_targets[domain]
        }
        sequence_pool = pool if retained_pool is None else retained_pool
        retained_tool_sequences = {
            domain: [
                sequence
                for _, row in sequence_pool.iterrows()
                if str(_as_extra(row["extra_info"]).get("domain") or "") == domain
                if (sequence := _row_tool_sequence(row, mode="prove"))
            ]
            for domain in required_by_domain
        }
        return cls(
            targets=targets,
            available=available,
            deficits=deficits,
            irrelevance_targets=irrelevance_targets,
            irrelevance_available=irrelevance_available,
            irrelevance_deficits=irrelevance_deficits,
            retained_tool_sequences=retained_tool_sequences,
        )

    @property
    def complete(self) -> bool:
        return not any(self.deficits.values()) and not self.irrelevance_deficits

    @property
    def total_deficits_by_domain(self) -> dict[str, int]:
        totals = {
            domain: (
                sum(self.deficits.get(domain, {}).values())
                + int(self.irrelevance_deficits.get(domain, 0))
            )
            for domain in self.targets
        }
        return {domain: count for domain, count in totals.items() if count > 0}

    def report(self) -> dict[str, Any]:
        return {
            "difficulty_targets_by_domain": self.targets,
            "difficulty_available_by_domain": self.available,
            "difficulty_deficits_by_domain": self.deficits,
            "irrelevance_targets_by_domain": self.irrelevance_targets,
            "irrelevance_available_by_domain": self.irrelevance_available,
            "irrelevance_deficits_by_domain": self.irrelevance_deficits,
            "stratum_deficits_by_domain": self.total_deficits_by_domain,
            "retained_tool_sequences_by_domain": self.retained_tool_sequences,
        }

    def topup_requests(self) -> list[tuple[str, str, int]]:
        difficulty_requests = [
            (domain, difficulty, count)
            for domain in sorted(self.deficits)
            for difficulty, count in self.deficits[domain].items()
            if count > 0
        ]
        irrelevance_requests = [
            (domain, "irrelevance", count)
            for domain, count in sorted(self.irrelevance_deficits.items())
            if count > 0
        ]
        return [*difficulty_requests, *irrelevance_requests]

    def candidate_topup_requests(
        self,
        candidate_budget_by_domain: dict[str, int],
    ) -> list[dict[str, Any]]:
        """Allocate a measured domain candidate budget across exact deficits."""
        result: list[dict[str, Any]] = []
        by_domain: dict[str, list[tuple[str, int]]] = {}
        for domain, stratum, missing in self.topup_requests():
            by_domain.setdefault(domain, []).append((stratum, missing))
        for domain in sorted(by_domain):
            strata = by_domain[domain]
            total_missing = sum(missing for _, missing in strata)
            candidate_budget = max(
                total_missing,
                int(candidate_budget_by_domain.get(domain, total_missing)),
            )
            extra = candidate_budget - total_missing
            exact_extra = {
                stratum: extra * missing / total_missing
                for stratum, missing in strata
            }
            allocated_extra = {
                stratum: int(value) for stratum, value in exact_extra.items()
            }
            remainder = extra - sum(allocated_extra.values())
            order = sorted(
                strata,
                key=lambda item: (
                    -(exact_extra[item[0]] - allocated_extra[item[0]]),
                    item[0],
                ),
            )
            for stratum, _ in order[:remainder]:
                allocated_extra[stratum] += 1
            for stratum, missing in strata:
                result.append({
                    "domain": domain,
                    "stratum": stratum,
                    "missing": int(missing),
                    "candidate_count": int(
                        missing + allocated_extra[stratum]
                    ),
                })
        return result

    def select(self, pool: pd.DataFrame) -> pd.DataFrame:
        if not self.complete:
            raise ValueError("cannot select from a corpus with stratum deficits")
        selected: list[pd.DataFrame] = []
        for domain, targets in self.targets.items():
            domain_mask = pool["extra_info"].map(
                lambda value: str(_as_extra(value).get("domain") or "") == domain
            )
            domain_rows = pool.loc[domain_mask]
            irrelevance_mask = domain_rows["extra_info"].map(_is_irrelevance)
            irrelevance_target = self.irrelevance_targets[domain]
            if irrelevance_target:
                selected.append(
                    domain_rows.loc[irrelevance_mask].head(irrelevance_target)
                )
            normal_rows = domain_rows.loc[~irrelevance_mask]
            for difficulty in DIFFICULTIES:
                target = targets[difficulty]
                if not target:
                    continue
                difficulty_mask = normal_rows["extra_info"].map(
                    _difficulty
                ).eq(difficulty)
                selected.append(normal_rows.loc[difficulty_mask].head(target))
        if not selected:
            return pool.iloc[0:0].copy()
        return pd.concat(selected, ignore_index=True)


__all__ = ["CorpusLedger", "DIFFICULTIES", "PROVE_DIFFICULTY_MIX"]
