#!/usr/bin/env python3
"""Build an immutable training selection from a certified run."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.live_mcp.corpus.shard import _validate_parquet_readback
from src.live_mcp.artifact.reward_task import build_reward_task
from src.live_mcp.registry.environment_metadata import (
    validate_training_artifact_evidence,
)
from src.utils import normalize_extra_info, sha256_file


PROVE_PUBLISHED_COUNTS = {
    "mcp_conversation": 10_895,
    "missing_function": 1_500,
    "internal_abstention_proxy": 1_122,
}
PROVE_PUBLISHED_TOTAL = sum(PROVE_PUBLISHED_COUNTS.values())


def _is_no_tool(extra_info: Any) -> bool:
    normalized = normalize_extra_info(extra_info)
    validate_training_artifact_evidence(normalized)
    task = build_reward_task(normalized)
    required = task.get("required_tool_calls")
    if not isinstance(required, list):
        raise RuntimeError("production reward parser returned invalid required_tool_calls")
    return len(required) == 0


def _stable_rank(uid: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{uid}".encode("utf-8")).hexdigest()


def _largest_remainder_ratio_quotas(
    weights: dict[str, int],
    target: int,
) -> dict[str, int]:
    if target < 0:
        raise ValueError("target must be >= 0")
    total_weight = sum(weights.values())
    if total_weight <= 0:
        raise ValueError("ratio weights must sum to a positive value")
    exact = {
        key: target * weight / total_weight
        for key, weight in weights.items()
    }
    quotas = {key: math.floor(value) for key, value in exact.items()}
    remaining = target - sum(quotas.values())
    order = sorted(
        weights,
        key=lambda key: (
            -(exact[key] - math.floor(exact[key])),
            key,
        ),
    )
    for key in order[:remaining]:
        quotas[key] += 1
    return quotas


def _max_capacity_ratio_quotas(
    capacities: dict[str, int],
    weights: dict[str, int],
    *,
    target_rows: int | None = None,
) -> dict[str, int]:
    if set(capacities) != set(weights):
        raise ValueError("capacity and ratio buckets must match")
    upper = sum(capacities.values())
    if target_rows is not None:
        if target_rows <= 0:
            raise ValueError("target_rows must be positive")
        upper = min(upper, target_rows)
    for target in range(upper, 0, -1):
        quotas = _largest_remainder_ratio_quotas(weights, target)
        if all(quotas[key] <= capacities[key] for key in weights):
            return quotas
    raise RuntimeError("source contains no capacity-compatible PROVE composition")


def _prove_proxy_bucket(extra_info: Any) -> tuple[str | None, int]:
    extra = normalize_extra_info(extra_info)
    validate_training_artifact_evidence(extra)
    task = build_reward_task(extra)
    required = task.get("required_tool_calls")
    if not isinstance(required, list):
        raise RuntimeError("production reward parser returned invalid required_tool_calls")
    required_count = len(required)
    scenario = str(extra.get("scenario_type") or "")
    if bool(extra.get("has_missing_function", False)):
        return "missing_function", required_count
    if scenario in ("no_tool_or_abstention", "irrelevant"):
        if required_count:
            raise RuntimeError(
                f"{scenario} row unexpectedly contains required tool calls"
            )
        return "internal_abstention_proxy", required_count
    if required_count:
        return "mcp_conversation", required_count
    return None, required_count


def _largest_remainder_quotas(
    counts: dict[tuple[str, ...], int],
    target: int,
) -> dict[tuple[str, ...], int]:
    total = sum(counts.values())
    if target < 0 or target > total:
        raise ValueError(f"invalid target {target} for population {total}")
    if target == 0:
        return {key: 0 for key in counts}

    exact = {
        key: target * count / total for key, count in counts.items()
    }
    quotas = {
        key: min(counts[key], math.floor(value))
        for key, value in exact.items()
    }
    remaining = target - sum(quotas.values())
    order = sorted(
        counts,
        key=lambda key: (
            -(exact[key] - math.floor(exact[key])),
            key,
        ),
    )
    for key in order:
        if remaining == 0:
            break
        if quotas[key] < counts[key]:
            quotas[key] += 1
            remaining -= 1
    if remaining:
        raise RuntimeError(f"failed to allocate {remaining} selection slots")
    return quotas


def _select_stratified(
    frame: pd.DataFrame,
    indices: list[int],
    *,
    target: int,
    seed: int,
) -> tuple[list[int], dict[str, int]]:
    strata: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for index in indices:
        row = frame.loc[index]
        extra = normalize_extra_info(row["extra_info"])
        _, required_count = _prove_proxy_bucket(extra)
        key = (
            str(extra.get("domain") or "unknown"),
            str(row.get("perturbation_level") or "unknown"),
            str(row.get("scenario_type") or "unknown"),
            "zero_tool" if required_count == 0 else "tool_required",
        )
        strata[key].append(index)
    quotas = _largest_remainder_quotas(
        {key: len(items) for key, items in strata.items()},
        target,
    )
    selected: list[int] = []
    selected_by_stratum: dict[str, int] = {}
    for key, items in sorted(strata.items()):
        ranked = sorted(
            items,
            key=lambda index: _stable_rank(str(frame.at[index, "uid"]), seed),
        )
        chosen = ranked[:quotas[key]]
        selected.extend(chosen)
        selected_by_stratum["::".join(key)] = len(chosen)
    return selected, selected_by_stratum


def build_prove_composition_proxy(
    *,
    input_dir: Path,
    output_dir: Path,
    seed: int,
    target_rows: int | None = None,
) -> dict[str, Any]:
    """Select the paper's 80.60/11.10/8.30 composition using internal abstention.

    This is intentionally named a proxy: the source corpus contains no imported
    When2Call or xLAM-Irrelevance rows, so its internal irrelevance rows cannot
    be reported as an external-source reproduction.
    """
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    if input_dir == output_dir:
        raise RuntimeError("input and output directories must differ")
    if output_dir.exists():
        raise RuntimeError(f"output directory already exists: {output_dir}")

    train_input = input_dir / "train.parquet"
    val_input = input_dir / "val.parquet"
    if not train_input.is_file() or not val_input.is_file():
        raise RuntimeError("input directory must contain train.parquet and val.parquet")

    train = pd.read_parquet(train_input)
    if train["uid"].duplicated().any():
        raise RuntimeError("source train contains duplicate uid values")

    bucket_indices: dict[str, list[int]] = {
        key: [] for key in PROVE_PUBLISHED_COUNTS
    }
    excluded_indices: list[int] = []
    excluded_scenarios: Counter[str] = Counter()
    for index, row in train.iterrows():
        bucket, _ = _prove_proxy_bucket(row["extra_info"])
        if bucket is None:
            excluded_indices.append(index)
            excluded_scenarios[str(row.get("scenario_type") or "unknown")] += 1
        else:
            bucket_indices[bucket].append(index)

    capacities = {
        key: len(indices) for key, indices in bucket_indices.items()
    }
    quotas = _max_capacity_ratio_quotas(
        capacities,
        PROVE_PUBLISHED_COUNTS,
        target_rows=target_rows,
    )

    selected_indices: list[int] = []
    selected_by_bucket_stratum: dict[str, dict[str, int]] = {}
    for offset, key in enumerate(PROVE_PUBLISHED_COUNTS):
        chosen, strata = _select_stratified(
            train,
            bucket_indices[key],
            target=quotas[key],
            seed=seed + offset,
        )
        selected_indices.extend(chosen)
        selected_by_bucket_stratum[key] = strata

    selected_train = train.loc[sorted(selected_indices)].reset_index(drop=True)
    if selected_train["uid"].duplicated().any():
        raise RuntimeError("selected train contains duplicate uid values")
    selected_bucket_counts: Counter[str] = Counter()
    selected_zero_tool = 0
    for extra_info in selected_train["extra_info"]:
        bucket, required_count = _prove_proxy_bucket(extra_info)
        if bucket is None:
            raise RuntimeError("unmapped row entered PROVE composition selection")
        selected_bucket_counts[bucket] += 1
        selected_zero_tool += int(required_count == 0)
    if dict(selected_bucket_counts) != quotas:
        raise RuntimeError(
            "selected PROVE bucket counts do not match allocated quotas: "
            f"selected={dict(selected_bucket_counts)} quotas={quotas}"
        )

    staging = output_dir.with_name(f".{output_dir.name}.tmp-{os.getpid()}")
    staging.mkdir(parents=True)
    try:
        train_output = staging / "train.parquet"
        val_output = staging / "val.parquet"
        selected_train.to_parquet(train_output, index=False)
        shutil.copy2(val_input, val_output)
        _validate_parquet_readback(train_output)
        _validate_parquet_readback(val_output)

        selected_rows = len(selected_train)
        report = {
            "status": "passed",
            "profile": "prove_composition_proxy",
            "strict_external_source_reproduction": False,
            "proxy_limitation": (
                "internal no_tool_or_abstention rows stand in for the paper's "
                "806 When2Call + 316 xLAM-Irrelevance external rows"
            ),
            "source_run": str(input_dir),
            "selection_seed": seed,
            "paper_published_counts": {
                "mcp_conversation": 10_895,
                "missing_function": 1_500,
                "external_abstention": 1_122,
                "total": PROVE_PUBLISHED_TOTAL,
            },
            "paper_target_ratios": {
                "mcp_conversation": 10_895 / PROVE_PUBLISHED_TOTAL,
                "missing_function": 1_500 / PROVE_PUBLISHED_TOTAL,
                "external_abstention": 1_122 / PROVE_PUBLISHED_TOTAL,
            },
            "source": {
                "train_rows": int(len(train)),
                "bucket_capacities": capacities,
                "excluded_unmapped_rows": len(excluded_indices),
                "excluded_unmapped_scenarios": dict(sorted(excluded_scenarios.items())),
                "train_sha256": sha256_file(train_input),
                "val_sha256": sha256_file(val_input),
            },
            "selected": {
                "train_rows": selected_rows,
                "bucket_rows": dict(selected_bucket_counts),
                "bucket_ratios": {
                    key: selected_bucket_counts[key] / selected_rows
                    for key in PROVE_PUBLISHED_COUNTS
                },
                "no_tool_rows": selected_zero_tool,
                "zero_tool_ratio": selected_zero_tool / selected_rows,
                "train_sha256": sha256_file(train_output),
                "val_rows": int(len(pd.read_parquet(val_output))),
                "val_sha256": sha256_file(val_output),
            },
            "selected_by_bucket_stratum": selected_by_bucket_stratum,
        }
        if report["source"]["val_sha256"] != report["selected"]["val_sha256"]:
            raise RuntimeError("validation copy checksum mismatch")
        (staging / "selection_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        staging.rename(output_dir)
        return report
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def build_training_mix(
    *,
    input_dir: Path,
    output_dir: Path,
    zero_tool_ratio: float,
    seed: int,
) -> dict[str, Any]:
    if not 0.0 <= zero_tool_ratio < 1.0:
        raise ValueError("zero_tool_ratio must be in [0, 1)")
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    if input_dir == output_dir:
        raise RuntimeError("input and output directories must differ")
    if output_dir.exists():
        raise RuntimeError(f"output directory already exists: {output_dir}")

    train_input = input_dir / "train.parquet"
    val_input = input_dir / "val.parquet"
    if not train_input.is_file() or not val_input.is_file():
        raise RuntimeError("input directory must contain train.parquet and val.parquet")

    train = pd.read_parquet(train_input)
    if train["uid"].duplicated().any():
        raise RuntimeError("source train contains duplicate uid values")

    no_tool_mask = train["extra_info"].map(_is_no_tool)
    tool_indices = train.index[~no_tool_mask].tolist()
    no_tool_indices = train.index[no_tool_mask].tolist()
    if not tool_indices:
        raise RuntimeError("source train contains no tool-required rows")

    requested_no_tool = round(
        len(tool_indices) * zero_tool_ratio / max(1.0 - zero_tool_ratio, 1e-12),
    )
    selected_no_tool_count = min(requested_no_tool, len(no_tool_indices))

    strata: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index in no_tool_indices:
        row = train.loc[index]
        extra_info = normalize_extra_info(row["extra_info"])
        key = (
            str(extra_info.get("domain") or "unknown"),
            str(row.get("scenario_type") or "unknown"),
        )
        strata[key].append(index)
    quotas = _largest_remainder_quotas(
        {key: len(indices) for key, indices in strata.items()},
        selected_no_tool_count,
    )

    selected_no_tool_indices: list[int] = []
    selected_by_stratum: dict[str, int] = {}
    for key, indices in sorted(strata.items()):
        ranked = sorted(
            indices,
            key=lambda index: _stable_rank(str(train.at[index, "uid"]), seed),
        )
        selected = ranked[:quotas[key]]
        selected_no_tool_indices.extend(selected)
        selected_by_stratum[f"{key[0]}::{key[1]}"] = len(selected)

    selected_indices = sorted(tool_indices + selected_no_tool_indices)
    selected_train = train.loc[selected_indices].reset_index(drop=True)
    selected_mask = selected_train["extra_info"].map(_is_no_tool)
    actual_no_tool = int(selected_mask.sum())
    actual_tool = int(len(selected_train) - actual_no_tool)
    if actual_tool != len(tool_indices):
        raise RuntimeError("tool-required rows were lost during selection")
    if actual_no_tool != selected_no_tool_count:
        raise RuntimeError("selected no-tool count does not match allocation")
    if selected_train["uid"].duplicated().any():
        raise RuntimeError("selected train contains duplicate uid values")

    staging = output_dir.with_name(f".{output_dir.name}.tmp-{os.getpid()}")
    staging.mkdir(parents=True)
    try:
        train_output = staging / "train.parquet"
        val_output = staging / "val.parquet"
        selected_train.to_parquet(train_output, index=False)
        shutil.copy2(val_input, val_output)
        _validate_parquet_readback(train_output)
        _validate_parquet_readback(val_output)

        report = {
            "status": "passed",
            "source_run": str(input_dir),
            "selection_seed": seed,
            "target_zero_tool_ratio": zero_tool_ratio,
            "source": {
                "train_rows": int(len(train)),
                "tool_rows": len(tool_indices),
                "no_tool_rows": len(no_tool_indices),
                "train_sha256": sha256_file(train_input),
                "val_sha256": sha256_file(val_input),
            },
            "selected": {
                "train_rows": int(len(selected_train)),
                "tool_rows": actual_tool,
                "no_tool_rows": actual_no_tool,
                "zero_tool_ratio": actual_no_tool / len(selected_train),
                "train_sha256": sha256_file(train_output),
                "val_rows": int(len(pd.read_parquet(val_output))),
                "val_sha256": sha256_file(val_output),
            },
            "source_no_tool_scenarios": dict(sorted(Counter(
                str(train.at[index, "scenario_type"])
                for index in no_tool_indices
            ).items())),
            "selected_no_tool_scenarios": dict(sorted(Counter(
                str(train.at[index, "scenario_type"])
                for index in selected_no_tool_indices
            ).items())),
            "selected_no_tool_by_domain_scenario": selected_by_stratum,
        }
        if report["source"]["val_sha256"] != report["selected"]["val_sha256"]:
            raise RuntimeError("validation copy checksum mismatch")
        (staging / "selection_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        staging.rename(output_dir)
        return report
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--profile",
        choices=("prove_composition_proxy", "zero_tool"),
        default="prove_composition_proxy",
    )
    parser.add_argument("--zero-tool-ratio", type=float, default=0.20)
    parser.add_argument("--target-rows", type=int)
    parser.add_argument("--seed", type=int, default=20260726)
    args = parser.parse_args()
    if args.profile == "prove_composition_proxy":
        report = build_prove_composition_proxy(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            seed=args.seed,
            target_rows=args.target_rows,
        )
    else:
        if args.target_rows is not None:
            parser.error("--target-rows is only valid for prove_composition_proxy")
        report = build_training_mix(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            zero_tool_ratio=args.zero_tool_ratio,
            seed=args.seed,
        )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
