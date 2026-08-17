"""Stable run-scoped seed namespaces for generation candidates."""

from __future__ import annotations

import argparse
import hashlib
import json


STRATA = (
    "initial",
    "complete",
    "missing",
    "minimal",
    "mixed",
    "irrelevance",
)
_MAX_CHUNKS_PER_STRATUM = 10_000
_MAX_INT64_SEED = (1 << 63) - 1


def _identity_text(name: str, value: str) -> str:
    normalized = str(value)
    if (
        not normalized
        or normalized != normalized.strip()
        or any(ord(character) < 32 for character in normalized)
    ):
        raise ValueError(f"{name} must be a non-empty canonical string")
    return normalized


def candidate_seed(
    *,
    base_seed: int,
    stride: int,
    run_id: str,
    artifact_round: int,
    domain_scope: str,
    stratum: str,
    chunk_index: int,
) -> int:
    """Map the full candidate identity into a recovery-safe seed namespace.

    The digest is mapped onto a stride-aligned signed-int64 namespace so every
    shard-local recovery offset in ``[0, stride)`` remains Parquet-safe. The
    complete identity, including the caller's base seed, is hashed canonically
    and is stable across resume of the same run.
    """
    if stride < 1:
        raise ValueError("stride must be positive")
    namespace_count = (_MAX_INT64_SEED + 1) // int(stride)
    if namespace_count < 1:
        raise ValueError(
            f"stride must be <= {_MAX_INT64_SEED + 1}, got {stride}"
        )
    if artifact_round < 0:
        raise ValueError("artifact_round must be non-negative")
    if not 0 <= chunk_index < _MAX_CHUNKS_PER_STRATUM:
        raise ValueError(
            "chunk_index must be in "
            f"[0, {_MAX_CHUNKS_PER_STRATUM}), got {chunk_index}"
        )
    normalized_run_id = _identity_text("run_id", run_id)
    normalized_domain = _identity_text("domain_scope", domain_scope)
    if stratum not in STRATA:
        raise ValueError(f"unknown stratum: {stratum!r}")
    payload = json.dumps(
        {
            "base_seed": int(base_seed),
            "run_id": normalized_run_id,
            "artifact_round": int(artifact_round),
            "domain_scope": normalized_domain,
            "stratum": stratum,
            "chunk_index": int(chunk_index),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    digest_value = int.from_bytes(
        hashlib.sha256(payload.encode("utf-8")).digest()[:16],
        "big",
    )
    namespace = digest_value % namespace_count
    return namespace * int(stride)


def _main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-seed", type=int, required=True)
    parser.add_argument("--stride", type=int, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--artifact-round", type=int, required=True)
    parser.add_argument("--domain-scope", required=True)
    parser.add_argument("--stratum", choices=STRATA, required=True)
    parser.add_argument("--chunk-index", type=int, required=True)
    args = parser.parse_args()
    print(candidate_seed(
        base_seed=args.base_seed,
        stride=args.stride,
        run_id=args.run_id,
        artifact_round=args.artifact_round,
        domain_scope=args.domain_scope,
        stratum=args.stratum,
        chunk_index=args.chunk_index,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())


__all__ = ["STRATA", "candidate_seed"]
