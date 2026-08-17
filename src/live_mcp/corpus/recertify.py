#!/usr/bin/env python3
"""Re-certify canonical generated data against the current runtime semantics."""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.live_mcp.artifact.readback import validate_parquet_readback
from src.live_mcp.registry.environment_metadata import (
    compute_initial_state_hashes,
    compute_reward_fingerprint,
    compute_transition_fingerprint,
    normalize_state_profiles,
    validate_environment_metadata,
)
from src.live_mcp.generation_runtime import TeacherGenerationRuntime
from src.live_mcp.replay.gates import replay_validate
from src.live_mcp.types import OracleCall
from src.live_mcp.artifact.validation import validate_artifact_contract
from src.utils import normalize_extra_info, normalize_json_field, sha256_file


ReplayFn = Callable[..., tuple[bool, float, int, int, bool, int]]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fresh-replay a canonical run and write a new copy bound to the "
            "current runtime and reward fingerprints"
        ),
    )
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--expected-source-reward-fingerprint",
        required=True,
        help="Exact fingerprint that every input row must contain",
    )
    parser.add_argument(
        "--suite",
        default="configs/live_mcp/ten_domain_suite.yaml",
        help="Current MCP suite configuration",
    )
    return parser


def _owners(extra_info: dict[str, Any]) -> set[str]:
    raw = normalize_json_field(extra_info.get("tool_owner_domains"), default={})
    owners = {str(extra_info.get("domain") or "")}
    if isinstance(raw, dict):
        owners.update(str(owner) for owner in raw.values())
    owners.discard("")
    if not owners:
        raise RuntimeError("row has no executable owner domain")
    return owners


def _current_tools(owners: set[str]) -> dict[str, list[dict[str, Any]]]:
    return {
        owner: list(importlib.import_module(
            f"src.live_mcp.servers.{owner}.server"
        ).TOOLS)
        for owner in sorted(owners)
    }


def _oracle_payload(row: dict[str, Any]) -> tuple[list[OracleCall], list[Any]]:
    reward_model = row.get("reward_model") or {}
    ground_truth = reward_model.get("ground_truth") or {}
    raw_calls = normalize_json_field(
        ground_truth.get("oracle_calls", "[]"), default=[],
    )
    raw_criteria = normalize_json_field(
        ground_truth.get("success_criteria", "[]"), default=[],
    )
    if not isinstance(raw_calls, list) or not isinstance(raw_criteria, list):
        raise RuntimeError("invalid canonical oracle payload")
    calls = [
        OracleCall(
            tool_name=str(call.get("tool_name") or ""),
            arguments=dict(call.get("arguments") or {}),
            save_as=str(call.get("save_as") or ""),
            action=str(call.get("action") or "tool_call"),
            server_name=str(call.get("server_name") or ""),
            expected_success=call.get("expected_success"),
        )
        for call in raw_calls
        if isinstance(call, dict)
    ]
    if len(calls) != len(raw_calls):
        raise RuntimeError("canonical oracle contains a non-object call")
    return calls, raw_criteria


def _list_field(value: Any, field: str) -> list[Any]:
    value = normalize_json_field(value, default=[])
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if not isinstance(value, list):
        raise RuntimeError(f"invalid {field}")
    return value


def _transition_fingerprints(extra_info: dict[str, Any]) -> dict[str, str]:
    value = normalize_json_field(
        extra_info.get("transition_fingerprints"), default={},
    )
    if not isinstance(value, dict) or not value:
        raise RuntimeError("missing transition_fingerprints")
    return {str(owner): str(fingerprint) for owner, fingerprint in value.items()}


def _recertify_row(
    row: dict[str, Any],
    *,
    expected_source_fingerprint: str,
    current_fingerprint: str,
    runtime_observation_budget: int,
    manager: Any,
    executor: Any,
    replay_fn: ReplayFn = replay_validate,
) -> dict[str, Any]:
    migrated = copy.deepcopy(row)
    extra_info = dict(normalize_extra_info(migrated.get("extra_info")))
    source_fingerprint = str(extra_info.get("reward_fingerprint") or "")
    if source_fingerprint != expected_source_fingerprint:
        raise RuntimeError(
            "unexpected source reward_fingerprint: "
            f"data={source_fingerprint!r}, expected={expected_source_fingerprint!r}"
        )
    reward_model = migrated.get("reward_model") or {}
    validate_artifact_contract(
        extra_info,
        require_training=False,
        ground_truth=(
            reward_model.get("ground_truth")
            if isinstance(reward_model, dict) else None
        ),
    )
    owners = _owners(extra_info)
    current_tools = _current_tools(owners)
    source_transition_fingerprints = _transition_fingerprints(extra_info)
    if set(source_transition_fingerprints) != owners:
        raise RuntimeError(
            "source transition_fingerprints must exactly cover owners: "
            f"fingerprints={sorted(source_transition_fingerprints)}, "
            f"owners={sorted(owners)}"
        )
    current_transition_fingerprints = {
        owner: compute_transition_fingerprint(owner, current_tools[owner])
        for owner in sorted(owners)
    }
    if (
        source_fingerprint == current_fingerprint
        and source_transition_fingerprints == current_transition_fingerprints
    ):
        raise RuntimeError(
            "source run already uses the current runtime/reward fingerprints"
        )
    state_profiles = normalize_state_profiles(
        extra_info.get("state_profiles"), owners,
    )

    # Validate every environment contract while substituting only the field
    # whose semantics this fresh replay is explicitly re-certifying.
    validation_metadata = dict(extra_info)
    validation_metadata["reward_fingerprint"] = current_fingerprint
    validation_metadata["transition_fingerprints"] = (
        current_transition_fingerprints
    )
    seed = int(extra_info["session_seed"])
    validate_environment_metadata(
        validation_metadata,
        current_tools_by_domain=current_tools,
        required_owner_domains=owners,
        reward_profile="prove_baseline",
        runtime_max_observation_chars=runtime_observation_budget,
        current_initial_state_hashes=compute_initial_state_hashes(
            owners,
            seed,
            state_profiles,
        ),
    )

    calls, criteria = _oracle_payload(migrated)
    hidden = _list_field(extra_info.get("hidden_tools"), "hidden_tools")
    replay = replay_fn(
        oracle_calls=calls,
        manager=manager,
        executor=executor,
        seed=seed,
        domain=str(extra_info["domain"]),
        success_criteria=criteria,
        blocked_tools={str(name) for name in hidden},
        state_profiles=state_profiles,
    )
    valid, error_rate, num_errors, num_calls, criteria_ok, criteria_failed = replay
    from src.live_mcp.prompt_profiles import requires_outcome_replay

    outcome_required = requires_outcome_replay(
        str(extra_info.get("prompt_profile") or "")
    )
    if not valid or (outcome_required and not criteria_ok):
        raise RuntimeError(
            "fresh canonical replay rejected row: "
            f"valid={valid}, errors={num_errors}/{num_calls}, "
            f"criteria_failed={criteria_failed}"
        )

    extra_info["reward_fingerprint"] = current_fingerprint
    extra_info["transition_fingerprints"] = current_transition_fingerprints
    extra_info["reward_profile_fingerprints"] = {
        profile: compute_reward_fingerprint(profile)
        for profile in ("prove_baseline", "oval_full")
    }
    extra_info["canonical_replay_valid"] = True
    extra_info["canonical_replay_error_rate"] = float(error_rate)
    extra_info["canonical_replay_num_errors"] = int(num_errors)
    extra_info["canonical_replay_num_calls"] = int(num_calls)
    extra_info["canonical_replay_criteria_ok"] = bool(criteria_ok)
    extra_info["canonical_replay_criteria_failed"] = int(criteria_failed)
    task = validate_artifact_contract(extra_info, require_training=False)
    validate_environment_metadata(
        extra_info,
        current_tools_by_domain=current_tools,
        required_owner_domains=owners,
        reward_profile="prove_baseline",
        runtime_max_observation_chars=runtime_observation_budget,
        current_initial_state_hashes=compute_initial_state_hashes(
            owners,
            seed,
            state_profiles,
        ),
    )
    if not isinstance(task, dict) or "required_tool_calls" not in task:
        raise RuntimeError("production reward parser returned an invalid task")
    migrated["extra_info"] = extra_info
    return migrated


def recertify_run(
    *,
    input_dir: Path,
    output_dir: Path,
    expected_source_fingerprint: str,
    suite_path: str | Path,
) -> dict[str, Any]:
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    if input_dir == output_dir:
        raise RuntimeError("input and output directories must differ")
    if output_dir.exists():
        raise RuntimeError(f"output directory already exists: {output_dir}")
    inputs = {
        split: input_dir / f"{split}.parquet" for split in ("train", "val")
    }
    missing = [str(path) for path in inputs.values() if not path.is_file()]
    if missing:
        raise RuntimeError(f"missing input parquet(s): {missing}")

    current_fingerprint = compute_reward_fingerprint("prove_baseline")
    staging = output_dir.with_name(f".{output_dir.name}.tmp-{os.getpid()}")
    if staging.exists():
        raise RuntimeError(f"staging directory already exists: {staging}")
    staging.mkdir(parents=True)
    report: dict[str, Any] = {
        "source_run": str(input_dir),
        "source_reward_fingerprint": expected_source_fingerprint,
        "current_reward_fingerprint": current_fingerprint,
        "source_transition_fingerprints": {},
        "splits": {},
        "rejections": [],
    }
    try:
        with TeacherGenerationRuntime.from_suite(suite_path) as runtime:
            assert runtime.executor is not None
            runtime_budget = int(
                runtime.suite_config.rollout.get("observation_max_chars", 4096)
            )
            for split, input_path in inputs.items():
                frame = pd.read_parquet(input_path)
                accepted: list[dict[str, Any]] = []
                for row_index, row in enumerate(frame.to_dict(orient="records")):
                    extra = normalize_extra_info(row.get("extra_info"))
                    for owner, fingerprint in _transition_fingerprints(extra).items():
                        previous = report["source_transition_fingerprints"].get(owner)
                        if previous is not None and previous != fingerprint:
                            raise RuntimeError(
                                "source run mixes transition fingerprints for "
                                f"{owner!r}: {previous!r} != {fingerprint!r}"
                            )
                        report["source_transition_fingerprints"][owner] = fingerprint
                    try:
                        accepted.append(_recertify_row(
                            row,
                            expected_source_fingerprint=expected_source_fingerprint,
                            current_fingerprint=current_fingerprint,
                            runtime_observation_budget=runtime_budget,
                            manager=runtime.manager,
                            executor=runtime.executor,
                        ))
                    except Exception as exc:
                        report["rejections"].append({
                            "split": split,
                            "row_index": row_index,
                            "task_id": str(extra.get("task_id") or row.get("uid") or ""),
                            "reason": f"{type(exc).__name__}: {exc}",
                        })
                output_path = staging / f"{split}.parquet"
                pd.DataFrame(accepted, columns=frame.columns).to_parquet(
                    output_path, index=False,
                )
                validate_parquet_readback(output_path)
                report["splits"][split] = {
                    "input_rows": int(len(frame)),
                    "accepted_rows": len(accepted),
                    "rejected_rows": int(len(frame) - len(accepted)),
                    "input_sha256": sha256_file(input_path),
                    "output_sha256": sha256_file(output_path),
                }

        report["status"] = "passed" if not report["rejections"] else "rejected"
        (staging / "recertification_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        if report["rejections"]:
            raise RuntimeError(
                f"recertification rejected {len(report['rejections'])} row(s); "
                f"diagnostics retained at {staging}"
            )
        staging.rename(output_dir)
        return report
    except Exception:
        # Passing artifacts are unpublished on any failure. Keep only the
        # diagnostic staging directory when it contains a report.
        if staging.exists() and not (staging / "recertification_report.json").exists():
            shutil.rmtree(staging)
        raise


def main() -> int:
    args = build_arg_parser().parse_args()
    report = recertify_run(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        expected_source_fingerprint=args.expected_source_reward_fingerprint,
        suite_path=args.suite,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
