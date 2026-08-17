"""Published PROVE generation-mixture policy and exact quota allocation."""

from __future__ import annotations

import math
from collections.abc import Mapping


DIFFICULTIES = ("complete", "missing", "minimal")
PROVE_DIFFICULTY_MIX: Mapping[str, float] = {
    "complete": 0.60,
    "missing": 0.20,
    "minimal": 0.20,
}


def default_difficulty_mix() -> dict[str, float]:
    """Return a caller-owned copy of the published PROVE 60/20/20 mix."""
    return dict(PROVE_DIFFICULTY_MIX)


def largest_remainder_mix_quotas(
    target: int,
    weights: Mapping[str, float],
) -> dict[str, int]:
    """Allocate an exact accepted-row target across configured mix weights."""
    if target < 0:
        raise ValueError("target must be non-negative")
    if not weights:
        raise ValueError("mix weights must not be empty")
    normalized = {str(key): float(value) for key, value in weights.items()}
    if any(value < 0 for value in normalized.values()):
        raise ValueError("mix weights must be non-negative")
    total = sum(normalized.values())
    if total <= 0:
        raise ValueError("mix weights must sum to a positive value")
    exact = {key: target * value / total for key, value in normalized.items()}
    quotas = {key: math.floor(value) for key, value in exact.items()}
    remaining = target - sum(quotas.values())
    order = sorted(
        normalized,
        key=lambda key: (-(exact[key] - math.floor(exact[key])), key),
    )
    for key in order[:remaining]:
        quotas[key] += 1
    return quotas


__all__ = [
    "DIFFICULTIES",
    "PROVE_DIFFICULTY_MIX",
    "default_difficulty_mix",
    "largest_remainder_mix_quotas",
]
