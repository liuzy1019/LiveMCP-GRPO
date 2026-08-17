import pytest
import pandas as pd

from src.live_mcp.corpus.candidate_identity import candidate_seed


def test_candidate_seed_is_stable_and_distinct_across_full_identity() -> None:
    identities = [
        ("run-a", 0, "all", "initial", 0),
        ("run-a", 0, "all", "initial", 1),
        ("run-a", 1, "payments", "complete", 0),
        ("run-a", 1, "payments", "irrelevance", 0),
        ("run-a", 2, "payments", "complete", 0),
        ("run-a", 1, "shopping", "complete", 0),
        ("run-b", 1, "payments", "complete", 0),
    ]
    seeds = [
        candidate_seed(
            base_seed=42,
            stride=1_000_000,
            run_id=run_id,
            artifact_round=artifact_round,
            domain_scope=domain_scope,
            stratum=stratum,
            chunk_index=chunk_index,
        )
        for run_id, artifact_round, domain_scope, stratum, chunk_index in identities
    ]

    assert len(set(seeds)) == len(identities)
    assert all(seed % 1_000_000 == 0 for seed in seeds)
    assert all(
        abs(left - right) >= 1_000_000
        for index, left in enumerate(seeds)
        for right in seeds[index + 1:]
    )
    assert candidate_seed(
        base_seed=42,
        stride=1_000_000,
        run_id="run-a",
        artifact_round=1,
        domain_scope="payments",
        stratum="complete",
        chunk_index=0,
    ) == seeds[2]


def test_observed_old_collision_no_longer_reuses_seed() -> None:
    missing_round_two = candidate_seed(
        base_seed=42,
        stride=1_000_000,
        run_id="run-a",
        artifact_round=2,
        domain_scope="payments",
        stratum="missing",
        chunk_index=0,
    )
    complete_round_three = candidate_seed(
        base_seed=42,
        stride=1_000_000,
        run_id="run-a",
        artifact_round=3,
        domain_scope="payments",
        stratum="complete",
        chunk_index=0,
    )

    assert missing_round_two != complete_round_three


def test_candidate_seed_and_recovery_offsets_roundtrip_through_parquet(
    tmp_path,
) -> None:
    stride = 1_000_000
    namespace_seed = candidate_seed(
        base_seed=42,
        stride=stride,
        run_id="diagnostic_banking_32_16_20260810_204113",
        artifact_round=0,
        domain_scope="banking",
        stratum="initial",
        chunk_index=0,
    )
    last_recovery_seed = namespace_seed + stride - 1

    assert 0 <= namespace_seed <= last_recovery_seed <= (1 << 63) - 1
    path = tmp_path / "candidate_seed.parquet"
    pd.DataFrame([
        {"extra_info": {"generation_seed": last_recovery_seed}},
    ]).to_parquet(path, index=False)
    restored = pd.read_parquet(path).iloc[0]["extra_info"]
    assert restored["generation_seed"] == last_recovery_seed


def test_candidate_seed_rejects_invalid_identity() -> None:
    with pytest.raises(ValueError, match="chunk_index"):
        candidate_seed(
            base_seed=42,
            stride=1_000_000,
            run_id="run-a",
            artifact_round=1,
            domain_scope="payments",
            stratum="complete",
            chunk_index=10_000,
        )
    with pytest.raises(ValueError, match="run_id"):
        candidate_seed(
            base_seed=42,
            stride=1_000_000,
            run_id="",
            artifact_round=1,
            domain_scope="payments",
            stratum="complete",
            chunk_index=0,
        )
    with pytest.raises(ValueError, match="stride must be <="):
        candidate_seed(
            base_seed=42,
            stride=(1 << 63) + 1,
            run_id="run-a",
            artifact_round=1,
            domain_scope="payments",
            stratum="complete",
            chunk_index=0,
        )
