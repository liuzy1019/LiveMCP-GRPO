#!/usr/bin/env python3
"""Run one or all YAML-defined 30+10 domain generation/rollout checks."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = PROJECT_ROOT / "configs" / "tests" / "per_domain"
DOMAINS = (
    "banking", "calendar", "crm", "email", "filesystem",
    "food_delivery", "issue_tracker", "payments", "shopping", "team_chat",
)


def load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError(f"{path}: expected schema_version=1 mapping")
    domain = str(payload.get("domain") or "")
    if domain not in DOMAINS or path.stem != domain:
        raise ValueError(f"{path}: filename/domain mismatch or unknown domain")
    generation = payload.get("generation")
    rollout = payload.get("rollout")
    if not isinstance(generation, dict) or not isinstance(rollout, dict):
        raise ValueError(f"{path}: generation and rollout mappings are required")
    expected = {
        "model": "models/Google/Gemma-4-31B-it",
        "suite": "configs/live_mcp/ten_domain_suite.yaml",
        "count": 30,
        "val_count": 10,
        "checkpoint_interval": 10,
        "gpu_count": 4,
        "prompt_profile": "local_trainable_v1",
        "semantic_gate_profile": "deterministic_v1",
        "distractor_rate": 0.40,
        "irrelevance_ratio": 0.05,
        "missing_function_rate": 0.1210165389,
    }
    drift = {
        key: (generation.get(key), value)
        for key, value in expected.items()
        if generation.get(key) != value
    }
    if drift:
        raise ValueError(f"{path}: formal test contract drift: {drift}")
    if rollout.get("reward_profile") != "prove_baseline":
        raise ValueError(f"{path}: rollout.reward_profile must be prove_baseline")
    expected_rollout = {
        "reward_profile": "prove_baseline",
        "experiment_profile": "custom",
        "steps": 1,
        "batch_size": 16,
        "rollout_n": 16,
    }
    rollout_drift = {
        key: (rollout.get(key), value)
        for key, value in expected_rollout.items()
        if rollout.get(key) != value
    }
    seed = payload.get("seed")
    if (
        rollout_drift
        or not isinstance(seed, int)
        or rollout.get("seeds") != str(seed)
    ):
        raise ValueError(
            f"{path}: formal rollout contract drift: "
            f"seed={seed} rollout_seeds={rollout.get('seeds')} "
            f"fields={rollout_drift}"
        )
    return payload


def validate_gpus(value: str) -> str:
    parts = value.split(",")
    if (
        len(parts) != 4
        or len(set(parts)) != 4
        or any(not part.isdigit() for part in parts)
    ):
        raise ValueError("--gpus must name exactly four distinct integer GPU IDs")
    return value


def generation_command(config: dict[str, Any], run_id: str) -> list[str]:
    generation = config["generation"]
    return [
        sys.executable, "-m", "src.live_mcp.corpus.cli", "run",
        "--mode", "full",
        "--model", str(generation["model"]),
        "--domain", str(config["domain"]),
        "--suite", str(generation["suite"]),
        "--count", str(generation["count"]),
        "--val-count", str(generation["val_count"]),
        "--seed", str(config["seed"]),
        "--run-id", run_id,
        "--checkpoint-interval", str(generation["checkpoint_interval"]),
        "--prompt-profile", str(generation["prompt_profile"]),
        "--semantic-gate-profile", str(generation["semantic_gate_profile"]),
        "--distractor-rate", str(generation["distractor_rate"]),
        "--irrelevance-ratio", str(generation["irrelevance_ratio"]),
        "--missing-function-rate", str(generation["missing_function_rate"]),
    ]


def recertify_command(
    config: dict[str, Any], run_dir: Path, output_dir: Path, fingerprint: str,
) -> list[str]:
    return [
        sys.executable, "-m", "src.live_mcp.corpus.cli", "recertify",
        "--input", str(run_dir.relative_to(PROJECT_ROOT)),
        "--output", str(output_dir.relative_to(PROJECT_ROOT)),
        "--expected-source-reward-fingerprint", fingerprint,
        "--suite", str(config["generation"]["suite"]),
    ]


def rollout_command(
    config: dict[str, Any], data_dir: Path, gpus: str, artifact_id: str,
) -> list[str]:
    rollout = config["rollout"]
    return [
        "bash", "scripts/smoke_rollout_reward.sh",
        "--gpus", gpus,
        "--reward-profile", str(rollout["reward_profile"]),
        "--experiment-profile", str(rollout["experiment_profile"]),
        "--seeds", str(rollout["seeds"]),
        "--steps", str(rollout["steps"]),
        "--batch-size", str(rollout["batch_size"]),
        "--rollout-n", str(rollout["rollout_n"]),
        "--train-file", str((data_dir / "train.parquet").relative_to(PROJECT_ROOT)),
        "--val-file", str((data_dir / "val.parquet").relative_to(PROJECT_ROOT)),
        "--artifact-id", artifact_id,
    ]


def _run(command: list[str], *, env: dict[str, str] | None = None) -> None:
    completed = subprocess.run(
        command, cwd=PROJECT_ROOT, env=env, check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed with returncode={completed.returncode}: {command}"
        )


def _reward_fingerprint(manifest: dict[str, Any]) -> str:
    values: set[str] = set()
    for split in ("train", "val"):
        values.update(
            manifest.get("outputs", {}).get(split, {}).get("provenance", {}).get(
                "reward_fingerprint", []
            )
        )
    if len(values) != 1:
        raise RuntimeError(f"expected one reward fingerprint, got {sorted(values)}")
    return next(iter(values))


def run_config(
    path: Path, *, gpus: str | None, skip_rollout: bool, cleanup: bool,
    dry_run: bool,
) -> dict[str, Any]:
    config = load_config(path)
    if not skip_rollout:
        if not gpus:
            raise ValueError("--gpus is required unless --skip-rollout is set")
        gpus = validate_gpus(gpus)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_id = f"per_domain_test_{config['domain']}_{stamp}"
    artifact_id = f"pdt_{config['domain']}_{stamp}_{os.getpid()}"
    run_dir = PROJECT_ROOT / "data" / "runs" / run_id
    recertified_dir = PROJECT_ROOT / "data" / "runs" / f"{run_id}_recertified"
    gen_command = generation_command(config, run_id)
    if dry_run:
        return {"domain": config["domain"], "generation_command": gen_command}

    env = os.environ.copy()
    env["GPU_COUNT"] = str(config["generation"]["gpu_count"])
    experiment_root = PROJECT_ROOT / "experiments" / "oval-mcp-grpo"
    experiment_root.mkdir(parents=True, exist_ok=True)
    tmp_root = Path(os.environ.get("TMPDIR", "/tmp")).resolve()
    _run(gen_command, env=env)
    manifest_path = run_dir / "generation_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "completed":
        raise RuntimeError(f"generation manifest is not completed: {manifest_path}")
    fingerprint = _reward_fingerprint(manifest)
    _run(recertify_command(config, run_dir, recertified_dir, fingerprint))

    if not skip_rollout:
        assert gpus is not None
        _run(rollout_command(config, recertified_dir, gpus, artifact_id))

    seeds = [seed for seed in str(config["rollout"]["seeds"]).split(",") if seed]
    owned_experiments = [
        experiment_root / f"smoke_rollout_reward_{artifact_id}_seed{seed}"
        for seed in seeds
    ]
    owned_tmp = [
        tmp_root / f"oval_ray_smoke_{artifact_id}_{seed}" for seed in seeds
    ]
    rollout_reports: dict[str, Any] = {}
    if not skip_rollout:
        from scripts.analyze_rollout_rewards import analyze, load_rows

        for experiment in owned_experiments:
            if not experiment.is_dir():
                raise RuntimeError(f"owned rollout directory missing: {experiment}")
            rollout_paths = sorted((experiment / "rollouts").glob("*.jsonl"))
            rollout_reports[experiment.name] = analyze(
                load_rows(rollout_paths), min_group_std=1e-6,
            )

    report = {
        "domain": config["domain"],
        "config": str(path.relative_to(PROJECT_ROOT)),
        "run_id": run_id,
        "generation_manifest": manifest,
        "recertification_report": json.loads(
            (recertified_dir / "recertification_report.json").read_text(
                encoding="utf-8"
            )
        ),
        "rollout_executed": not skip_rollout,
        "rollout_reports": rollout_reports,
        "cleanup": cleanup,
    }
    report_path = PROJECT_ROOT / "logs" / f"{run_id}_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    if cleanup:
        shutil.rmtree(run_dir)
        shutil.rmtree(recertified_dir)
        for experiment in owned_experiments:
            if experiment.is_dir():
                shutil.rmtree(experiment)
        for path in owned_tmp:
            if path.is_dir():
                shutil.rmtree(path)
    return {**report, "report_path": str(report_path.relative_to(PROJECT_ROOT))}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", choices=DOMAINS)
    parser.add_argument("--gpus", help="physical GPU IDs used by policy rollout")
    parser.add_argument("--skip-rollout", action="store_true")
    parser.add_argument("--cleanup", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    paths = (
        [CONFIG_ROOT / f"{args.domain}.yaml"]
        if args.domain else [CONFIG_ROOT / f"{domain}.yaml" for domain in DOMAINS]
    )
    results = [
        run_config(
            path, gpus=args.gpus, skip_rollout=args.skip_rollout,
            cleanup=args.cleanup, dry_run=args.dry_run,
        )
        for path in paths
    ]
    print(json.dumps(results, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
