"""Domain allocation helpers shared by candidate generation and global merge."""

from __future__ import annotations

import math
from collections.abc import Sequence


def position_aware_jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    """Local scheduling diagnostic; not the PROVE corpus gate."""
    if not a or not b:
        return 0.0
    pos_a = set(enumerate(a))
    pos_b = set(enumerate(b))
    return len(pos_a & pos_b) / len(pos_a | pos_b)


def plain_tool_jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    """PROVE Jaccard over the sets of executed tool names."""
    set_a = {str(item) for item in a if str(item)}
    set_b = {str(item) for item in b if str(item)}
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def jaccard_unique_sequence_count(
    sequences: Sequence[Sequence[str]],
    *,
    threshold: float = 0.70,
) -> int:
    """Count sequences retained by the canonical PROVE Jaccard gate."""
    kept: list[tuple[str, ...]] = []
    for raw_sequence in sequences:
        sequence = tuple(str(item) for item in raw_sequence if str(item))
        if not sequence:
            continue
        if any(
            plain_tool_jaccard(sequence, prior) >= threshold
            for prior in kept
        ):
            continue
        kept.append(sequence)
    return len(kept)


def capacity_weighted_domain_quotas(
    target: int,
    domains: Sequence[str],
    capacities: dict[str, int],
    *,
    minimum_per_domain: int | None = None,
) -> dict[str, int]:
    """Allocate a target by unique-chain capacity while preserving a floor."""
    ordered_domains = list(domains)
    if not ordered_domains:
        return {}
    if target < 0:
        raise ValueError("target must be non-negative")
    if len(ordered_domains) != len(set(ordered_domains)):
        raise ValueError("domains must be unique")
    if minimum_per_domain is None:
        minimum_per_domain = (
            max(1, target // (2 * len(ordered_domains)))
            if target >= len(ordered_domains)
            else 0
        )
    floor = max(0, int(minimum_per_domain))
    if floor * len(ordered_domains) > target:
        raise ValueError(
            f"minimum domain coverage {floor} x {len(ordered_domains)} "
            f"exceeds split target {target}"
        )

    quotas = {domain: floor for domain in ordered_domains}
    remainder = target - floor * len(ordered_domains)
    if remainder == 0:
        return quotas

    weights = {
        domain: max(0, int(capacities.get(domain, 0)))
        for domain in ordered_domains
    }
    total_weight = sum(weights.values())
    if total_weight == 0:
        weights = {domain: 1 for domain in ordered_domains}
        total_weight = len(ordered_domains)

    raw = {
        domain: remainder * weights[domain] / total_weight
        for domain in ordered_domains
    }
    assigned = 0
    for domain in ordered_domains:
        whole = math.floor(raw[domain])
        quotas[domain] += whole
        assigned += whole

    leftover = remainder - assigned
    order = sorted(
        ordered_domains,
        key=lambda domain: (
            -(raw[domain] - math.floor(raw[domain])),
            domain,
        ),
    )
    for domain in order[:leftover]:
        quotas[domain] += 1
    return quotas
