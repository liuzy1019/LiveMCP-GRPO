"""Single public CLI for LiveMCP corpus production."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
from typing import Sequence

from src.live_mcp.prompt_profiles import PROMPT_PROFILES
from loguru import logger


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL = "models/Google/Gemma-4-31B-it"
BUCKETS = (
    "mcp_conversation",
    "missing_function",
    "internal_abstention_proxy",
    "external_abstention",
)


class GenerationTerminated(RuntimeError):
    def __init__(self, signum: int):
        self.signum = signum
        super().__init__(f"generation terminated by signal {signum}")


def _run_launcher(
    command: list[str], *, cwd: Path, env: dict[str, str],
) -> subprocess.CompletedProcess:
    """Run the launcher as a process group and propagate operator shutdown."""
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        start_new_session=True,
    )
    previous_sigterm = signal.getsignal(signal.SIGTERM)

    def _raise_terminated(signum, _frame) -> None:
        raise GenerationTerminated(signum)

    signal.signal(signal.SIGTERM, _raise_terminated)
    try:
        returncode = process.wait()
    except (KeyboardInterrupt, GenerationTerminated):
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)
    return subprocess.CompletedProcess(command, returncode)


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _reserve_new_run_directory(output_dir: Path) -> None:
    """Atomically reserve a run id before publishing its first manifest.

    Checking for an existing manifest and creating it are separate filesystem
    operations.  Two launchers using the same fresh run id could therefore both
    pass the check and race on the manifest.  Directory creation is the single
    atomic ownership operation; an existing directory must use an explicit
    resume flow or a different run id, even when it happens to be empty.
    """
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        output_dir.mkdir()
    except FileExistsError as exc:
        raise ValueError(
            f"run-id already contains generation artifacts or is reserved: "
            f"{output_dir}; use a new --run-id or the explicit resume command"
        ) from exc


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _generated_parquet_summary(path: Path) -> dict:
    import pandas as pd

    frame = pd.read_parquet(path)
    difficulties = Counter(str(value) for value in frame.get("perturbation_level", []))
    scenarios = Counter(str(value) for value in frame.get("scenario_type", []))
    domains: Counter[str] = Counter()
    provenance: dict[str, set[str]] = {
        "teacher_model_id": set(),
        "prompt_profile": set(),
        "semantic_gate_profile": set(),
        "artifact_purpose": set(),
        "reward_fingerprint": set(),
        "dependency_classifier_contract_hash": set(),
    }
    for raw_extra in frame.get("extra_info", []):
        extra = raw_extra if isinstance(raw_extra, dict) else {}
        domains[str(extra.get("domain") or "")] += 1
        for key in provenance:
            value = str(extra.get(key) or "")
            if value:
                provenance[key].add(value)
    domains.pop("", None)
    return {
        "path": _relative_display(path),
        "rows": len(frame),
        "sha256": _sha256_file(path),
        "domains": dict(sorted(domains.items())),
        "difficulties": dict(sorted(difficulties.items())),
        "scenarios": dict(sorted(scenarios.items())),
        "provenance": {
            key: sorted(values) for key, values in provenance.items()
        },
    }


def _generation_failure_summary(directory: Path) -> dict:
    stages: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    records = 0
    files = sorted(directory.glob("*.jsonl")) if directory.is_dir() else []
    for path in files:
        for line_number, raw_line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1,
        ):
            if not raw_line.strip():
                continue
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"invalid failure record {path}:{line_number}"
                ) from exc
            records += 1
            stages[str(record.get("stage") or "unknown")] += 1
            reasons[str(record.get("reason_code") or "unknown")] += 1
    return {
        "directory": _relative_display(directory),
        "files": len(files),
        "records": records,
        "stages": dict(sorted(stages.items())),
        "reasons": dict(sorted(reasons.items())),
    }


def _command_owns_run(arguments: list[str], run_id: str) -> bool:
    """Return whether one argv vector is a generation owner for ``run_id``."""
    return any(
        argument == "--run-id"
        and index + 1 < len(arguments)
        and arguments[index + 1] == run_id
        for index, argument in enumerate(arguments)
    ) and (
        any(argument.endswith("src/live_mcp/corpus/launcher.sh") for argument in arguments)
        or (
            "src.live_mcp.corpus.cli" in arguments
            and any(command in arguments for command in ("run", "resume"))
        )
    )


def _run_process_is_active(run_id: str) -> bool:
    """Return whether another live process command line owns this run id."""
    current_pid = os.getpid()
    proc_root = Path("/proc")
    if not proc_root.is_dir():
        return False
    for entry in proc_root.iterdir():
        if not entry.name.isdigit() or int(entry.name) == current_pid:
            continue
        try:
            arguments = [
                item.decode(errors="replace")
                for item in (entry / "cmdline").read_bytes().split(b"\0")
                if item
            ]
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if _command_owns_run(arguments, run_id):
            return True
    return False


def reconcile_run_manifest(output_dir: Path) -> dict:
    """Idempotently converge a recoverable manifest from durable evidence."""
    manifest_path = output_dir / "generation_manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"run manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    status = str(manifest.get("status") or "")
    fixed_attempt = bool(
        (manifest.get("request") or {}).get("fixed_attempt_budget")
    )
    if status != "running" and not (status == "failed" and fixed_attempt):
        return manifest

    run_id = output_dir.name
    accepted = output_dir / "accepted.parquet"
    if accepted.is_file():
        try:
            from src.live_mcp.artifact.readback import validate_parquet_readback

            validate_parquet_readback(accepted)
            outputs = dict(manifest.get("outputs") or {})
            for split in ("train", "val", "accepted"):
                artifact = output_dir / f"{split}.parquet"
                if artifact.is_file():
                    outputs[split] = _generated_parquet_summary(artifact)
            manifest.update({
                "status": "completed",
                "returncode": 0,
                "finished_at": datetime.now().astimezone().isoformat(),
                "outputs": outputs,
                "failure_evidence": _generation_failure_summary(
                    output_dir / "failures"
                ),
                "reconciled_from_durable_evidence": True,
            })
            if fixed_attempt:
                manifest["completion_kind"] = "fixed_attempt_diagnostic"
        except Exception as exc:
            manifest.update({
                "status": "failed",
                "finished_at": datetime.now().astimezone().isoformat(),
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "reconciled_from_durable_evidence": True,
            })
    elif not _run_process_is_active(run_id):
        manifest.update({
            "status": "interrupted",
            "finished_at": datetime.now().astimezone().isoformat(),
            "error_type": "OrphanedRun",
            "error_message": (
                "no active owner process and no accepted artifact"
            ),
            "reconciled_from_durable_evidence": True,
        })
    _write_json_atomic(manifest_path, manifest)
    return manifest


def _status_command(args: argparse.Namespace) -> int:
    output_dir = PROJECT_ROOT / "data" / "runs" / args.run_id
    manifest = reconcile_run_manifest(output_dir)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _relative_display(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def _require_certified_base(value: str | Path) -> Path:
    base = _project_path(value).resolve()
    required = ("train.parquet", "val.parquet")
    missing = [name for name in required if not (base / name).is_file()]
    has_certification_report = any(
        (base / name).is_file()
        for name in ("merge_report.json", "recertification_report.json")
    )
    if not has_certification_report:
        missing.append("merge_report.json|recertification_report.json")
    if missing:
        raise ValueError(
            f"base corpus is incomplete: {base}; missing={missing}"
        )
    return base


def _bucket_generation_args(bucket: str) -> list[str]:
    if bucket == "mcp_conversation":
        return [
            "--tool-required-only",
            "--missing-function-rate",
            "0",
            "--irrelevance-ratio",
            "0",
        ]
    if bucket == "missing_function":
        return [
            "--missing-function-rate",
            "1",
            "--irrelevance-ratio",
            "0",
        ]
    if bucket == "internal_abstention_proxy":
        raise ValueError(
            "internal_abstention_proxy is selection-only until a dedicated "
            "abstention-only generator passes gray testing"
        )
    if bucket == "external_abstention":
        raise ValueError(
            "external_abstention requires an auditable When2Call/xLAM source; "
            "the current repository cannot generate it"
        )
    raise ValueError(f"unknown bucket: {bucket}")


def build_launcher_command(args: argparse.Namespace) -> tuple[list[str], dict[str, str]]:
    """Translate the public run contract to the internal launcher contract."""
    fixed_attempt_budget = bool(
        getattr(args, "fixed_attempt_budget", False)
    )
    if fixed_attempt_budget:
        if args.command != "run" or args.mode != "full":
            raise ValueError(
                "--fixed-attempt-budget is only valid for full run mode"
            )
        if args.publish:
            raise ValueError(
                "--fixed-attempt-budget is diagnostic and cannot publish"
            )
        if args.candidate_budget is not None:
            raise ValueError(
                "--fixed-attempt-budget cannot be combined with "
                "--candidate-budget"
            )
    command = [
        "bash",
        "src/live_mcp/corpus/launcher.sh",
        "--model",
        args.model,
        "--domain",
        args.domain,
        "--suite",
        args.suite,
        "--seed",
        str(args.seed),
        "--run-id",
        args.run_id,
        "--checkpoint-interval",
        str(args.checkpoint_interval),
    ]
    if args.chain_bin_quotas is not None:
        command.extend(["--chain-bin-quotas", args.chain_bin_quotas])
    env: dict[str, str] = {}
    # The public CLI may be invoked through an absolute environment Python
    # without an activated CONDA_PREFIX.  Pin the internal launcher and every
    # child process to that same interpreter unless the caller deliberately
    # supplied a PYTHON_BIN override.
    env["PYTHON_BIN"] = os.environ.get("PYTHON_BIN", sys.executable)
    prompt_profile = getattr(
        args, "prompt_profile", "paper_generation_baseline_v1"
    )
    # Always set the profile explicitly.  Otherwise a stale parent-shell
    # LIVEMCP_PROMPT_PROFILE can silently change the requested formal profile.
    env["LIVEMCP_PROMPT_PROFILE"] = prompt_profile
    env["LIVEMCP_SEMANTIC_GATE_PROFILE"] = args.semantic_gate_profile
    if fixed_attempt_budget:
        env.update({
            "GEN_OVERSAMPLE_PCT": "0",
            "GENERATION_MAX_RECOVERY_ROUNDS": "1",
            "GENERATION_PRESERVE_CANDIDATES": "1",
            "LIVEMCP_FIXED_ATTEMPT_BUDGET": "1",
            "MERGE_TOPUP_ROUNDS": "0",
        })
    if args.difficulty is not None:
        command.extend(["--difficulty", args.difficulty])
    command.extend(["--distractor-rate", str(args.distractor_rate)])
    if args.preserve_candidates:
        env["GENERATION_PRESERVE_CANDIDATES"] = "1"
    if args.mode == "full":
        if args.count is None or args.count < 1:
            raise ValueError("full mode requires --count >= 1")
        if args.val_count < 1:
            raise ValueError("full mode requires --val-count >= 1")
        if args.base is not None or args.net_new is not None or args.bucket:
            raise ValueError(
                "full mode does not accept --base, --net-new, or --bucket"
            )
        if args.chain_bin_quotas is not None:
            raise ValueError(
                "full mode does not accept --chain-bin-quotas"
            )
        command.extend([
            "--count",
            str(args.count),
            "--val-count",
            str(args.val_count),
        ])
        if args.candidate_budget is not None:
            command.extend([
                "--candidate-budget",
                str(args.candidate_budget),
            ])
        if args.tool_required_only:
            if args.irrelevance_ratio not in (None, 0, 0.0):
                raise ValueError(
                    "--tool-required-only is incompatible with non-zero "
                    "--irrelevance-ratio"
                )
            command.extend([
                "--tool-required-only",
                "--irrelevance-ratio",
                str(
                    0
                    if args.irrelevance_ratio is None
                    else args.irrelevance_ratio
                ),
            ])
        elif args.irrelevance_ratio is not None:
            command.extend([
                "--irrelevance-ratio",
                str(args.irrelevance_ratio),
            ])
        if args.missing_function_rate is not None:
            command.extend([
                "--missing-function-rate",
                str(args.missing_function_rate),
            ])
        if args.publish:
            command.append("--publish")
    else:
        if (
            args.tool_required_only
            or args.irrelevance_ratio is not None
            or args.missing_function_rate is not None
        ):
            raise ValueError(
                "supplement mode derives tool/irrelevance/missing-function "
                "settings from --bucket"
            )
        if args.base is None:
            raise ValueError("supplement mode requires --base")
        if args.net_new is None or args.net_new < 1:
            raise ValueError("supplement mode requires --net-new >= 1")
        if not args.bucket:
            raise ValueError("supplement mode requires --bucket")
        if (
            args.chain_bin_quotas is not None
            and args.bucket != "mcp_conversation"
        ):
            raise ValueError(
                "--chain-bin-quotas is only valid for mcp_conversation"
            )
        if args.publish:
            raise ValueError(
                "supplement mode cannot publish directly; use finalize first"
            )
        base = _require_certified_base(args.base)
        candidate_budget = args.candidate_budget or math.ceil(
            args.net_new * 30 / 7
        )
        if candidate_budget < args.net_new:
            raise ValueError("--candidate-budget must be >= --net-new")
        command.extend([
            "--count",
            str(args.net_new),
            "--candidate-budget",
            str(candidate_budget),
            "--val-count",
            "0",
            "--base-train",
            _relative_display(base / "train.parquet"),
            "--base-val",
            _relative_display(base / "val.parquet"),
            *_bucket_generation_args(args.bucket),
        ])
        env["GEN_OVERSAMPLE_PCT"] = "0"
    if args.resume_candidate_dir is not None:
        candidate_dir = _project_path(args.resume_candidate_dir).resolve()
        if not candidate_dir.is_dir():
            raise ValueError(
                f"resume candidate directory not found: {candidate_dir}"
            )
        env["GENERATION_RESUME_CANDIDATE_DIR"] = str(candidate_dir)
    return command, env


def _run_command(args: argparse.Namespace) -> int:
    if args.command == "resume" and args.resume_candidate_dir is None:
        raise ValueError("resume requires --resume-candidate-dir")
    if args.command == "resume" and args.publish:
        raise ValueError("resume cannot publish directly; use finalize first")
    command, env_updates = build_launcher_command(args)
    payload = {
        "command": command,
        "command_display": shlex.join(command),
        "environment": env_updates,
        "dry_run": bool(args.dry_run),
    }
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    if args.dry_run:
        return 0
    if args.command == "resume":
        merge_result = _try_resume_merge_without_teacher(args)
        if merge_result == 0:
            return 0
        if merge_result == 2:
            return 2
        print(
            "resume candidates still have a recoverable deficit; "
            "starting Teacher for top-up",
            flush=True,
        )
    env = os.environ.copy()
    env.update(env_updates)
    output_dir = PROJECT_ROOT / "data" / "runs" / args.run_id
    manifest_path = output_dir / "generation_manifest.json"
    if args.command == "run":
        _reserve_new_run_directory(output_dir)
    started_at = datetime.now().astimezone().isoformat()
    manifest = {
        "schema_version": 1,
        "status": "running",
        "started_at": started_at,
        "request": {
            key: getattr(args, key, None)
            for key in (
                "command", "mode", "model", "domain", "suite", "count",
                "val_count", "seed", "run_id", "candidate_budget",
                "prompt_profile", "semantic_gate_profile", "difficulty",
                "distractor_rate", "tool_required_only", "irrelevance_ratio",
                "missing_function_rate", "checkpoint_interval",
                "fixed_attempt_budget",
            )
        },
        "launcher": {
            "command": command,
            "environment": env_updates,
        },
        "outputs": {},
    }
    _write_json_atomic(manifest_path, manifest)
    try:
        completed = _run_launcher(
            command,
            cwd=PROJECT_ROOT,
            env=env,
        )
    except KeyboardInterrupt:
        manifest.update({
            "status": "interrupted",
            "finished_at": datetime.now().astimezone().isoformat(),
            "error_type": "KeyboardInterrupt",
            "error_message": "generation interrupted by operator",
        })
        _write_json_atomic(manifest_path, manifest)
        raise
    except GenerationTerminated as exc:
        manifest.update({
            "status": "interrupted",
            "finished_at": datetime.now().astimezone().isoformat(),
            "error_type": "Signal",
            "error_message": str(exc),
            "signal": int(exc.signum),
        })
        _write_json_atomic(manifest_path, manifest)
        return 128 + int(exc.signum)
    except Exception as exc:
        manifest.update({
            "status": "failed",
            "finished_at": datetime.now().astimezone().isoformat(),
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        })
        _write_json_atomic(manifest_path, manifest)
        raise
    manifest["status"] = "completed" if completed.returncode == 0 else "failed"
    manifest["returncode"] = int(completed.returncode)
    manifest["finished_at"] = datetime.now().astimezone().isoformat()
    try:
        for split in ("train", "val", "accepted"):
            artifact = output_dir / f"{split}.parquet"
            if artifact.is_file():
                manifest["outputs"][split] = _generated_parquet_summary(artifact)
        if args.fixed_attempt_budget and completed.returncode == 0:
            manifest["completion_kind"] = "fixed_attempt_diagnostic"
            filter_report = output_dir / "candidates" / "merge_deficits.json"
            if filter_report.is_file():
                manifest["outputs"]["filter_report"] = {
                    "path": _relative_display(filter_report),
                    "sha256": _sha256_file(filter_report),
                    "summary": json.loads(
                        filter_report.read_text(encoding="utf-8")
                    ),
                }
        manifest["failure_evidence"] = _generation_failure_summary(
            output_dir / "failures"
        )
    except Exception as exc:
        manifest.update({
            "status": "failed",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        })
        _write_json_atomic(manifest_path, manifest)
        raise
    _write_json_atomic(manifest_path, manifest)
    return completed.returncode


def _try_resume_merge_without_teacher(args: argparse.Namespace) -> int:
    """Finish a sufficient preserved pool without paying Teacher startup."""
    from src.live_mcp.corpus.audit import audit_file

    candidate_dir = _project_path(args.resume_candidate_dir).resolve()
    output_dir = PROJECT_ROOT / "data" / "runs" / args.run_id
    count = args.net_new if args.mode == "supplement" else args.count
    if count is None:
        raise ValueError("resume target count is missing")
    val_count = 0 if args.mode == "supplement" else args.val_count
    deficits = candidate_dir / "resume_merge_deficits.json"
    command = [
        sys.executable,
        "-m",
        "src.live_mcp.corpus.merge",
        "--tmpdir",
        str(candidate_dir),
        "--output-dir",
        str(output_dir),
        "--count",
        str(count),
        "--val-count",
        str(val_count),
        "--domain",
        args.domain,
        "--deficits-output",
        str(deficits),
    ]
    if args.mode == "supplement":
        irrelevance_ratio = 0.0
    elif args.tool_required_only:
        irrelevance_ratio = 0.0
    else:
        irrelevance_ratio = (
            0.05 if args.irrelevance_ratio is None
            else float(args.irrelevance_ratio)
        )
    command.extend([
        "--irrelevance-count",
        str(int((count + val_count) * irrelevance_ratio + 0.5)),
    ])
    if args.difficulty is not None:
        command.extend(["--difficulty", args.difficulty])
    if args.chain_bin_quotas is not None:
        command.extend(["--chain-bin-quotas", args.chain_bin_quotas])
    if args.mode == "supplement":
        base = _require_certified_base(args.base)
        command.extend([
            "--base-train",
            str(base / "train.parquet"),
            "--base-val",
            str(base / "val.parquet"),
        ])
    print(
        json.dumps(
            {
                "resume_merge_command": command,
                "resume_merge_display": shlex.join(command),
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    if completed.returncode != 0:
        return completed.returncode
    train_path = output_dir / "train.parquet"
    val_path = output_dir / "val.parquet"
    print(json.dumps({
        "resume_merge": "completed_without_teacher",
        "train_audit": audit_file(train_path),
        "val_audit": audit_file(val_path),
    }, indent=2, sort_keys=True), flush=True)
    return 0


def _plan_command(args: argparse.Namespace) -> int:
    from src.live_mcp.corpus.planning import build_plan

    base = _require_certified_base(args.base)
    run_id = args.run_id
    plan = build_plan(
        input_dir=base,
        max_net_new=args.max_net_new,
        candidate_numerator=args.candidate_numerator,
        candidate_denominator=args.candidate_denominator,
        seed=args.seed,
        run_id=run_id,
    )
    report = (
        _project_path(args.report).resolve()
        if args.report
        else PROJECT_ROOT / "logs" / (
            f"{datetime.now():%Y%m%d_%H%M%S}_gap_plan.json"
        )
    )
    print(json.dumps(plan, indent=2, sort_keys=True))
    if not args.no_write:
        report.parent.mkdir(parents=True, exist_ok=True)
        temporary = report.with_name(f".{report.name}.tmp")
        temporary.write_text(
            json.dumps(plan, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(report)
        print(f"plan_report={report}")
    return 0


def _finalize_command(args: argparse.Namespace) -> int:
    from src.live_mcp.corpus.finalize import merge_incremental

    base = _require_certified_base(args.base)
    incremental_paths: list[Path] = []
    for value in args.incremental:
        path = _project_path(value).resolve()
        if path.is_dir():
            path = path / "train.parquet"
        if not path.is_file():
            raise ValueError(f"incremental parquet not found: {path}")
        incremental_paths.append(path)
    report = merge_incremental(
        base_train_path=base / "train.parquet",
        base_val_path=base / "val.parquet",
        incremental_paths=incremental_paths,
        output_dir=_project_path(args.output).resolve(),
        publish=args.publish,
        quarantine_invalid_incremental=args.quarantine_invalid_incremental,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _profile_command(args: argparse.Namespace) -> int:
    from src.live_mcp.corpus.profile import (
        build_prove_composition_proxy,
        build_training_mix,
    )

    input_dir = _require_certified_base(args.input)
    output_dir = _project_path(args.output).resolve()
    if args.profile == "prove_composition_proxy":
        report = build_prove_composition_proxy(
            input_dir=input_dir,
            output_dir=output_dir,
            seed=args.seed,
            target_rows=args.target_rows,
        )
    else:
        if args.target_rows is not None:
            raise ValueError(
                "--target-rows is only valid for prove_composition_proxy"
            )
        report = build_training_mix(
            input_dir=input_dir,
            output_dir=output_dir,
            zero_tool_ratio=args.zero_tool_ratio,
            seed=args.seed,
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _recertify_command(args: argparse.Namespace) -> int:
    from src.live_mcp.corpus.recertify import recertify_run

    report = recertify_run(
        input_dir=_project_path(args.input).resolve(),
        output_dir=_project_path(args.output).resolve(),
        expected_source_fingerprint=args.expected_source_reward_fingerprint,
        suite_path=args.suite,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _add_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--mode", choices=("full", "supplement"), required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--domain", default="all")
    parser.add_argument(
        "--suite",
        default="configs/live_mcp/ten_domain_suite.yaml",
    )
    parser.add_argument("--count", type=int)
    parser.add_argument("--val-count", type=int, default=0)
    parser.add_argument("--base")
    parser.add_argument("--bucket", choices=BUCKETS)
    parser.add_argument("--net-new", type=int)
    parser.add_argument("--candidate-budget", type=int)
    parser.add_argument(
        "--chain-bin-quotas",
        help='MCP-only final selection quotas as JSON, e.g. '
        '\'{"1-2":3,"3-5":14,"6+":3}\'',
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--run-id",
        default=f"{datetime.now():%Y%m%d_%H%M}_generation",
    )
    parser.add_argument("--checkpoint-interval", type=int, default=25)
    parser.add_argument(
        "--prompt-profile",
        choices=tuple(PROMPT_PROFILES),
        default="paper_generation_baseline_v1",
    )
    parser.add_argument(
        "--semantic-gate-profile",
        choices=("diagnostic_only", "deterministic_v1"),
        default="diagnostic_only",
        help=(
            "Completed-trace semantic audit disposition; deterministic_v1 "
            "is a labeled local corpus gate, not a PROVE hard gate"
        ),
    )
    parser.add_argument(
        "--difficulty",
        choices=("complete", "missing", "minimal"),
        help="Diagnostic fixed information-completeness level",
    )
    parser.add_argument(
        "--distractor-rate",
        type=float,
        default=0.40,
        help="Teacher-time distractor injection probability",
    )
    parser.add_argument(
        "--tool-required-only",
        action="store_true",
        help="Diagnostic full run: retain only rows with tool calls",
    )
    parser.add_argument(
        "--irrelevance-ratio",
        type=float,
        help="Diagnostic full-run override; supplement buckets own this ratio",
    )
    parser.add_argument(
        "--missing-function-rate",
        type=float,
        help="Diagnostic full-run override; supplement buckets own this ratio",
    )
    parser.add_argument("--resume-candidate-dir")
    parser.add_argument(
        "--preserve-candidates",
        action="store_true",
        help="retain raw shard parquets for controlled gray-test audit",
    )
    parser.add_argument(
        "--fixed-attempt-budget",
        action="store_true",
        help=(
            "diagnostic only: treat count+val-count as candidate attempts; "
            "disable oversample, shard recovery, and merge top-up"
        ),
    )
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--dry-run", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Single public entry for LiveMCP corpus production"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser("plan", help="analyze corpus gaps")
    plan.add_argument("--base", required=True)
    plan.add_argument("--report")
    plan.add_argument("--max-net-new", type=int, default=700)
    plan.add_argument("--candidate-numerator", type=int, default=30)
    plan.add_argument("--candidate-denominator", type=int, default=7)
    plan.add_argument("--seed", type=int, default=2026072801)
    plan.add_argument("--run-id")
    plan.add_argument("--no-write", action="store_true")
    plan.set_defaults(handler=_plan_command)

    run = subparsers.add_parser("run", help="run full or gap generation")
    _add_run_arguments(run)
    run.set_defaults(handler=_run_command)

    resume = subparsers.add_parser(
        "resume", help="resume from preserved candidate/checkpoint data"
    )
    _add_run_arguments(resume)
    resume.set_defaults(handler=_run_command)

    status = subparsers.add_parser(
        "status", help="inspect and reconcile one durable generation run"
    )
    status.add_argument("--run-id", required=True)
    status.set_defaults(handler=_status_command)

    finalize = subparsers.add_parser(
        "finalize", help="merge incremental runs into a new certified corpus"
    )
    finalize.add_argument("--base", required=True)
    finalize.add_argument("--incremental", action="append", required=True)
    finalize.add_argument("--output", required=True)
    finalize.add_argument("--publish", action="store_true")
    finalize.add_argument(
        "--quarantine-invalid-incremental",
        action="store_true",
    )
    finalize.set_defaults(handler=_finalize_command)

    profile = subparsers.add_parser(
        "profile", help="build an immutable training composition view"
    )
    profile.add_argument("--input", required=True)
    profile.add_argument("--output", required=True)
    profile.add_argument(
        "--profile",
        choices=("prove_composition_proxy", "zero_tool"),
        default="prove_composition_proxy",
    )
    profile.add_argument("--zero-tool-ratio", type=float, default=0.20)
    profile.add_argument("--target-rows", type=int)
    profile.add_argument("--seed", type=int, default=20260726)
    profile.set_defaults(handler=_profile_command)

    recertify = subparsers.add_parser(
        "recertify", help="fresh-replay an immutable corpus"
    )
    recertify.add_argument("--input", "--input-dir", dest="input", required=True)
    recertify.add_argument(
        "--output", "--output-dir", dest="output", required=True
    )
    recertify.add_argument(
        "--expected-source-reward-fingerprint",
        required=True,
    )
    recertify.add_argument(
        "--suite",
        default="configs/live_mcp/ten_domain_suite.yaml",
    )
    recertify.set_defaults(handler=_recertify_command)

    build_cache = subparsers.add_parser(
        "build-cache", help="build dependency-graph caches via LLM classification"
    )
    build_cache.add_argument("--domain", default="all")
    build_cache.add_argument("--model", required=True)
    build_cache.add_argument("--teacher-model-id", default=None)
    build_cache.add_argument("--api-base", default=None)
    build_cache.add_argument("--suite", default="configs/live_mcp/ten_domain_suite.yaml")
    build_cache.add_argument("--device", type=int, default=None)
    build_cache.add_argument("--workers", type=int, default=4)
    build_cache.add_argument(
        "--prompt-profile",
        choices=tuple(PROMPT_PROFILES),
        default="paper_generation_baseline_v1",
    )
    build_cache.set_defaults(handler=_build_cache_command)

    return parser


def _build_cache_command(args: argparse.Namespace) -> int:
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from src.live_mcp.generation_runtime import TeacherGenerationRuntime
    from src.live_mcp.llm_client import LLMClient
    from src.live_mcp.orchestrator import TaskOrchestrator

    def _select_domains(runtime: TeacherGenerationRuntime, domain_arg: str) -> list[str]:
        if domain_arg == "all":
            return list(runtime.manager.server_names)
        domains = [d.strip() for d in domain_arg.split(",") if d.strip()]
        unknown = [d for d in domains if d not in runtime.manager.server_names]
        if unknown:
            raise ValueError(f"unknown domains: {unknown}")
        return domains

    runtime = TeacherGenerationRuntime.from_suite(args.suite)
    runtime.start()
    try:
        client = (
            LLMClient(
                mode="openai",
                model_path=args.model,
                contract_model_id=args.teacher_model_id,
                api_base=args.api_base,
            )
            if args.api_base
            else LLMClient(
                mode="local",
                model_path=args.model,
                contract_model_id=args.teacher_model_id,
                device=args.device,
            )
        )
        assert runtime.executor is not None
        orchestrator = TaskOrchestrator(
            runtime.suite_config, runtime.manager, runtime.executor, client,
            prompt_profile=args.prompt_profile,
        )
        domains = _select_domains(runtime, args.domain)

        def _build_one(domain: str) -> tuple[str, int, int]:
            graph = orchestrator._get_or_build_dependency_graph(domain)
            chains = orchestrator._extract_dependency_chains(domain)
            return domain, len(graph), len(chains)

        workers = min(max(1, args.workers), len(domains))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_build_one, d): d for d in domains}
            for future in as_completed(futures):
                domain, nodes, chain_count = future.result()
                logger.info(f"{domain}: graph_nodes={nodes} chains={chain_count}")
    finally:
        runtime.stop()
    return 0


def build_cache_main(argv: Sequence[str] | None = None) -> int:
    """Standalone compatibility entry for dependency-cache builders."""
    parser = argparse.ArgumentParser(description="Build dependency-graph caches.")
    parser.add_argument("--domain", default="all")
    parser.add_argument("--model", required=True)
    parser.add_argument("--teacher-model-id", default=None)
    parser.add_argument("--api-base", default=None)
    parser.add_argument("--suite", default="configs/live_mcp/ten_domain_suite.yaml")
    parser.add_argument("--device", type=int, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--prompt-profile",
        choices=tuple(PROMPT_PROFILES),
        default="paper_generation_baseline_v1",
    )
    args = parser.parse_args(argv)
    return _build_cache_command(args)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
