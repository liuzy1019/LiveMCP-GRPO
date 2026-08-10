import itertools

import pytest

from src.live_mcp.corpus.candidate_identity import topup_seed


def test_topup_seed_is_unique_across_round_domain_difficulty_and_chunk() -> None:
    identities = itertools.product(
        (1, 2, 3),
        ("payments", "shopping"),
        ("complete", "missing", "minimal", "mixed"),
        range(4),
    )
    seeds = {
        topup_seed(
            base_seed=42,
            stride=1_000_000,
            artifact_round=round_index,
            domain=domain,
            difficulty=difficulty,
            chunk_index=chunk_index,
        )
        for round_index, domain, difficulty, chunk_index in identities
    }

    assert len(seeds) == 3 * 2 * 4 * 4


def test_observed_old_collision_no_longer_reuses_seed() -> None:
    missing_round_two = topup_seed(
        base_seed=42,
        stride=1_000_000,
        artifact_round=2,
        domain="payments",
        difficulty="missing",
        chunk_index=0,
    )
    complete_round_three = topup_seed(
        base_seed=42,
        stride=1_000_000,
        artifact_round=3,
        domain="payments",
        difficulty="complete",
        chunk_index=0,
    )

    assert missing_round_two != complete_round_three


def test_topup_seed_rejects_unbounded_chunk_index() -> None:
    with pytest.raises(ValueError, match="chunk_index"):
        topup_seed(
            base_seed=42,
            stride=1_000_000,
            artifact_round=1,
            domain="payments",
            difficulty="complete",
            chunk_index=10_000,
        )
