"""Collision-free candidate seed allocation for generation shards."""

from __future__ import annotations

import argparse


DOMAIN_ORDER = (
    "banking",
    "calendar",
    "crm",
    "email",
    "filesystem",
    "food_delivery",
    "issue_tracker",
    "payments",
    "shopping",
    "team_chat",
)
DIFFICULTY_ORDER = ("complete", "missing", "minimal", "mixed")
_CHUNKS_PER_STRATUM = 10_000


def topup_seed(
    *,
    base_seed: int,
    stride: int,
    artifact_round: int,
    domain: str,
    difficulty: str,
    chunk_index: int,
) -> int:
    """Map a top-up identity tuple injectively into the generation seed space."""
    if stride < 1:
        raise ValueError("stride must be positive")
    if artifact_round < 1:
        raise ValueError("artifact_round must be positive")
    if not 0 <= chunk_index < _CHUNKS_PER_STRATUM:
        raise ValueError(
            f"chunk_index must be in [0, {_CHUNKS_PER_STRATUM}), got {chunk_index}"
        )
    try:
        domain_index = DOMAIN_ORDER.index(domain)
    except ValueError as exc:
        raise ValueError(f"unknown domain: {domain!r}") from exc
    normalized_difficulty = difficulty or "mixed"
    try:
        difficulty_index = DIFFICULTY_ORDER.index(normalized_difficulty)
    except ValueError as exc:
        raise ValueError(f"unknown difficulty: {difficulty!r}") from exc

    slot = (
        (
            artifact_round * len(DOMAIN_ORDER)
            + domain_index
        ) * len(DIFFICULTY_ORDER)
        + difficulty_index
    ) * _CHUNKS_PER_STRATUM + chunk_index
    return int(base_seed) + slot * int(stride)


def _main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-seed", type=int, required=True)
    parser.add_argument("--stride", type=int, required=True)
    parser.add_argument("--artifact-round", type=int, required=True)
    parser.add_argument("--domain", choices=DOMAIN_ORDER, required=True)
    parser.add_argument("--difficulty", choices=DIFFICULTY_ORDER, required=True)
    parser.add_argument("--chunk-index", type=int, required=True)
    args = parser.parse_args()
    print(topup_seed(
        base_seed=args.base_seed,
        stride=args.stride,
        artifact_round=args.artifact_round,
        domain=args.domain,
        difficulty=args.difficulty,
        chunk_index=args.chunk_index,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())


__all__ = ["DIFFICULTY_ORDER", "DOMAIN_ORDER", "topup_seed"]
