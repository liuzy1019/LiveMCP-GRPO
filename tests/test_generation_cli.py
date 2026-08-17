from __future__ import annotations

from argparse import Namespace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import pytest

from src.live_mcp.corpus.cli import (
    _bucket_generation_args,
    _generation_failure_summary,
    _generated_parquet_summary,
    _reserve_new_run_directory,
    reconcile_run_manifest,
    build_launcher_command,
)
from src.live_mcp.corpus import cli


def _args(**overrides) -> Namespace:
    values = {
        "command": "run",
        "mode": "full",
        "model": "models/Google/Gemma-4-31B-it",
        "domain": "all",
        "suite": "configs/live_mcp/ten_domain_suite.yaml",
        "count": 3,
        "val_count": 1,
        "base": None,
        "bucket": None,
        "net_new": None,
        "candidate_budget": None,
        "chain_bin_quotas": None,
        "seed": 7,
        "run_id": "gray",
        "checkpoint_interval": 25,
        "prompt_profile": "paper_generation_baseline_v1",
        "semantic_gate_profile": "diagnostic_only",
        "difficulty": None,
        "distractor_rate": 0.40,
        "tool_required_only": False,
        "irrelevance_ratio": None,
        "missing_function_rate": None,
        "resume_candidate_dir": None,
        "preserve_candidates": False,
        "fixed_attempt_budget": False,
        "publish": False,
        "dry_run": True,
    }
    values.update(overrides)
    return Namespace(**values)


def test_generation_failure_summary_groups_structured_dispositions(
    tmp_path: Path,
) -> None:
    failure_dir = tmp_path / "failures"
    failure_dir.mkdir()
    records = [
        {"stage": "query_generation", "reason_code": "goal_unsat"},
        {"stage": "fresh_replay", "reason_code": "replay_invalid"},
        {"stage": "fresh_replay", "reason_code": "replay_invalid"},
    ]
    (failure_dir / "shard_0.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    summary = _generation_failure_summary(failure_dir)

    assert summary["files"] == 1
    assert summary["records"] == 3
    assert summary["stages"] == {
        "fresh_replay": 2,
        "query_generation": 1,
    }
    assert summary["reasons"] == {
        "goal_unsat": 1,
        "replay_invalid": 2,
    }


def _base(tmp_path: Path) -> Path:
    base = tmp_path / "base"
    base.mkdir()
    for name in ("train.parquet", "val.parquet", "merge_report.json"):
        (base / name).write_bytes(b"test")
    return base


def _launcher_function(name: str) -> str:
    source = Path("src/live_mcp/corpus/launcher.sh").read_text()
    start = source.index(f"{name}() {{")
    end = source.index("\n}\n", start) + 3
    return source[start:end]


def test_vllm_model_probe_requires_exact_served_model() -> None:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            payload = json.dumps({"data": [{"id": "Gemma-4-31B-it"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *_args) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    listener_probe = _launcher_function("_port_is_listening")
    function = _launcher_function("_port_serves_model")
    try:
        for model, expected_code in (
            ("Gemma-4-31B-it", 0),
            ("Gemma-4-31B", 1),
        ):
            result = subprocess.run(
                ["bash", "-c", (
                    "set -euo pipefail\n"
                    f"PYTHON_BIN={sys.executable!s}\n"
                    f"{listener_probe}\n"
                    f"{function}\n"
                    f"_port_is_listening {server.server_port}\n"
                    f"_port_serves_model {server.server_port} {model}"
                )],
                check=False,
            )
            assert result.returncode == expected_code
    finally:
        server.shutdown()
        server.server_close()


def test_launcher_keeps_vllm_by_default_and_isolates_its_session() -> None:
    source = Path("src/live_mcp/corpus/launcher.sh").read_text()

    assert 'VLLM_SHUTDOWN_ON_EXIT="${VLLM_SHUTDOWN_ON_EXIT:-0}"' in source
    assert 'setsid env \\\n' in source
    assert '[ "${VLLM_OWNED[$index]:-0}" = "1" ]' in source
    assert 'VLLM_OWNED+=("0")' in source
    assert 'if _port_is_listening "${PORT}"; then' in source
    assert 'if [ -n "${PID}" ] && ! kill -0' in source


def test_launcher_probes_reusable_service_before_free_gpu_filter() -> None:
    source = Path("src/live_mcp/corpus/launcher.sh").read_text()

    assert source.index("REUSABLE_VLLM_PID") < source.index(
        "source scripts/gpu_config.sh"
    )
    assert source.index("source scripts/gpu_config.sh") < source.index(
        'if _port_is_listening "${PORT}"; then'
    )
    assert "GPU_FREE_ONLY=0" in source[
        source.index("REUSABLE_VLLM_PID"):
        source.index("source scripts/gpu_config.sh")
    ]


def test_launcher_persists_shard_failure_evidence_in_run_directory() -> None:
    source = Path("src/live_mcp/corpus/launcher.sh").read_text()

    assert source.count("--failure-records-path") == 3
    assert (
        '--failure-records-path "${RUN_DIR}/failures/'
        'shard_${inst}_${client}.jsonl"'
    ) in source
    assert (
        '--failure-records-path "${RUN_DIR}/failures/'
        'shard_${topup_prefix}.jsonl"'
    ) in source


@pytest.mark.parametrize(
    ("shutdown", "owned", "should_survive"),
    (("0", "1", True), ("1", "1", False), ("1", "0", True)),
)
def test_launcher_cleanup_respects_vllm_ownership(
    shutdown: str, owned: str, should_survive: bool,
) -> None:
    service = subprocess.Popen(["sleep", "30"], start_new_session=True)
    cleanup = _launcher_function("_cleanup")
    try:
        result = subprocess.run(
            ["bash", "-c", (
                "set -euo pipefail\n"
                f"VLLM_PIDS=({service.pid})\n"
                "VLLM_PORTS=(65530)\n"
                "VLLM_LOGS=('')\n"
                f"VLLM_OWNED=({owned})\n"
                f"VLLM_SHUTDOWN_ON_EXIT={shutdown}\n"
                "GEN_SUCCESS=0\n"
                f"{cleanup}\n"
                "_cleanup"
            )],
            check=False,
        )
        assert result.returncode == 0
        assert (service.poll() is None) is should_survive
    finally:
        if service.poll() is None:
            service.terminate()
        service.wait(timeout=5)


def test_launcher_cleanup_stops_owned_process_group_after_leader_exit() -> None:
    service = subprocess.Popen(
        ["bash", "-c", "trap '' TERM; sleep 30 & wait"],
        start_new_session=True,
    )
    cleanup = _launcher_function("_cleanup")
    try:
        result = subprocess.run(
            ["bash", "-c", (
                "set -euo pipefail\n"
                f"VLLM_PIDS=({service.pid})\n"
                "VLLM_PORTS=(65530)\n"
                "VLLM_LOGS=('')\n"
                "VLLM_OWNED=(1)\n"
                "VLLM_SHUTDOWN_ON_EXIT=1\n"
                "GEN_SUCCESS=0\n"
                f"{cleanup}\n"
                "_cleanup"
            )],
            check=False,
        )
        assert result.returncode == 0
        service.wait(timeout=5)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            try:
                os.killpg(service.pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.05)
        with pytest.raises(ProcessLookupError):
            os.killpg(service.pid, 0)
    finally:
        try:
            os.killpg(service.pid, 9)
        except ProcessLookupError:
            pass


def test_full_run_maps_to_internal_launcher_without_implicit_publish() -> None:
    command, env = build_launcher_command(_args())
    assert command[:2] == ["bash", "src/live_mcp/corpus/launcher.sh"]
    assert command[command.index("--count") + 1] == "3"
    assert command[command.index("--val-count") + 1] == "1"
    assert "--publish" not in command
    assert env == {
        "LIVEMCP_PROMPT_PROFILE": "paper_generation_baseline_v1",
        "LIVEMCP_SEMANTIC_GATE_PROFILE": "diagnostic_only",
        "PYTHON_BIN": sys.executable,
    }


def test_tool_required_full_run_disables_incompatible_irrelevance_bucket() -> None:
    command, _ = build_launcher_command(_args(tool_required_only=True))

    ratio_positions = [
        index for index, value in enumerate(command)
        if value == "--irrelevance-ratio"
    ]
    assert len(ratio_positions) == 1
    assert command[ratio_positions[0] + 1] == "0"


def test_tool_required_full_run_rejects_nonzero_irrelevance_override() -> None:
    with pytest.raises(ValueError, match="incompatible"):
        build_launcher_command(_args(
            tool_required_only=True,
            irrelevance_ratio=0.05,
        ))


def test_launcher_forwards_final_irrelevance_target_to_every_merge() -> None:
    source = Path("src/live_mcp/corpus/launcher.sh").read_text()

    assert 'FINAL_IRRELEVANCE_COUNT="$(_rounded_irrelevance_count ' in source
    assert source.count('"${MERGE_STRATUM_ARGS[@]}"') == 2
    assert 'report.get("irrelevance_deficits_by_domain", {})' in source
    assert '--irrelevance-count "${chunk_size}"' in source


def test_launcher_namespaces_initial_and_topup_seeds_by_run_and_stratum() -> None:
    source = Path("src/live_mcp/corpus/launcher.sh").read_text()

    assert source.count("-m src.live_mcp.corpus.candidate_identity") == 3
    assert source.count('--run-id "${RUN_ID}"') >= 3
    assert source.count("--stratum initial") == 2
    assert '--stratum "${topup_prefix_difficulty}"' in source
    assert "SEED + CLIENT_ID * GENERATION_CLIENT_SEED_STRIDE" not in source


def test_generated_parquet_summary_binds_rows_hash_and_provenance(
    tmp_path: Path,
) -> None:
    import pandas as pd

    path = tmp_path / "train.parquet"
    pd.DataFrame([{
        "perturbation_level": "complete",
        "scenario_type": "normal_safe_success",
        "extra_info": {
            "domain": "banking",
            "teacher_model_id": "teacher",
            "prompt_profile": "paper_generation_baseline_v1",
            "semantic_gate_profile": "deterministic_v1",
            "reward_fingerprint": "reward",
            "dependency_classifier_contract_hash": "dependency",
        },
    }]).to_parquet(path, index=False)

    summary = _generated_parquet_summary(path)

    assert summary["rows"] == 1
    assert len(summary["sha256"]) == 64
    assert summary["domains"] == {"banking": 1}
    assert summary["difficulties"] == {"complete": 1}
    assert summary["provenance"]["teacher_model_id"] == ["teacher"]


def test_run_manifest_records_launcher_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)

    def fail(*args, **kwargs):
        raise OSError("launcher unavailable")

    monkeypatch.setattr(cli, "_run_launcher", fail)
    with pytest.raises(OSError, match="launcher unavailable"):
        cli._run_command(_args(dry_run=False, run_id="exception_run"))

    manifest = json.loads(
        (tmp_path / "data/runs/exception_run/generation_manifest.json").read_text()
    )
    assert manifest["status"] == "failed"
    assert manifest["error_type"] == "OSError"
    assert manifest["finished_at"]


def test_run_manifest_records_operator_interrupt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "_run_launcher", interrupt)
    with pytest.raises(KeyboardInterrupt):
        cli._run_command(_args(dry_run=False, run_id="interrupted_run"))

    manifest = json.loads(
        (tmp_path / "data/runs/interrupted_run/generation_manifest.json").read_text()
    )
    assert manifest["status"] == "interrupted"
    assert manifest["error_type"] == "KeyboardInterrupt"
    assert manifest["finished_at"]


def test_run_manifest_records_sigterm_as_interrupted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        cli,
        "_run_launcher",
        lambda *args, **kwargs: (_ for _ in ()).throw(cli.GenerationTerminated(15)),
    )

    assert cli._run_command(
        _args(dry_run=False, run_id="terminated_run")
    ) == 143
    manifest = json.loads(
        (tmp_path / "data/runs/terminated_run/generation_manifest.json").read_text()
    )
    assert manifest["status"] == "interrupted"
    assert manifest["error_type"] == "Signal"
    assert manifest["signal"] == 15


def test_run_manifest_records_nonzero_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        cli,
        "_run_launcher",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 17),
    )

    assert cli._run_command(_args(dry_run=False, run_id="failed_run")) == 17
    manifest = json.loads(
        (tmp_path / "data/runs/failed_run/generation_manifest.json").read_text()
    )
    assert manifest["status"] == "failed"
    assert manifest["returncode"] == 17


def test_reconcile_running_manifest_from_valid_accepted_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pandas as pd

    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)
    run = tmp_path / "data/runs/recoverable"
    run.mkdir(parents=True)
    (run / "generation_manifest.json").write_text(json.dumps({
        "status": "running",
        "request": {"fixed_attempt_budget": True},
        "outputs": {},
    }))
    pd.DataFrame([{"extra_info": {"domain": "banking"}}]).to_parquet(
        run / "accepted.parquet", index=False,
    )
    monkeypatch.setattr(
        "src.live_mcp.artifact.readback.validate_parquet_readback",
        lambda _path: None,
    )

    manifest = reconcile_run_manifest(run)

    assert manifest["status"] == "completed"
    assert manifest["completion_kind"] == "fixed_attempt_diagnostic"
    assert manifest["reconciled_from_durable_evidence"] is True


def test_reconcile_failed_fixed_attempt_manifest_from_valid_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pandas as pd

    run = tmp_path / "data/runs/recoverable_failed"
    run.mkdir(parents=True)
    (run / "generation_manifest.json").write_text(json.dumps({
        "status": "failed",
        "returncode": 134,
        "request": {"fixed_attempt_budget": True},
        "outputs": {},
    }))
    pd.DataFrame().to_parquet(run / "accepted.parquet", index=False)
    monkeypatch.setattr(
        "src.live_mcp.artifact.readback.validate_parquet_readback",
        lambda _path: None,
    )

    manifest = reconcile_run_manifest(run)

    assert manifest["status"] == "completed"
    assert manifest["returncode"] == 0
    assert manifest["completion_kind"] == "fixed_attempt_diagnostic"


def test_reconcile_does_not_promote_failed_quota_run(
    tmp_path: Path,
) -> None:
    run = tmp_path / "data/runs/failed_quota"
    run.mkdir(parents=True)
    original = {
        "status": "failed",
        "returncode": 17,
        "request": {"fixed_attempt_budget": False},
    }
    (run / "generation_manifest.json").write_text(json.dumps(original))

    assert reconcile_run_manifest(run) == original


def test_reconcile_orphaned_running_manifest_as_interrupted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(cli, "_run_process_is_active", lambda _run_id: False)
    run = tmp_path / "data/runs/orphaned"
    run.mkdir(parents=True)
    (run / "generation_manifest.json").write_text(
        json.dumps({"status": "running", "outputs": {}})
    )

    manifest = reconcile_run_manifest(run)

    assert manifest["status"] == "interrupted"
    assert manifest["error_type"] == "OrphanedRun"


def test_status_command_does_not_look_like_a_run_owner() -> None:
    assert cli._command_owns_run(
        [
            "python", "-m", "src.live_mcp.corpus.cli", "status",
            "--run-id", "example",
        ],
        "example",
    ) is False
    assert cli._command_owns_run(
        [
            "bash", "src/live_mcp/corpus/launcher.sh",
            "--run-id", "example",
        ],
        "example",
    ) is True


def test_run_id_collision_is_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)
    output = tmp_path / "data/runs/collision"
    output.mkdir(parents=True)
    (output / "generation_manifest.json").write_text("{}")

    with pytest.raises(ValueError, match="already contains generation artifacts"):
        cli._run_command(_args(dry_run=False, run_id="collision"))


def test_empty_run_directory_is_already_reserved(tmp_path: Path) -> None:
    output = tmp_path / "data/runs/collision"
    output.mkdir(parents=True)

    with pytest.raises(ValueError, match="or is reserved"):
        _reserve_new_run_directory(output)


def test_run_directory_reservation_has_exactly_one_winner(
    tmp_path: Path,
) -> None:
    from concurrent.futures import ThreadPoolExecutor

    output = tmp_path / "data/runs/concurrent"

    def reserve() -> str:
        try:
            _reserve_new_run_directory(output)
        except ValueError:
            return "rejected"
        return "reserved"

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: reserve(), range(2)))

    assert sorted(results) == ["rejected", "reserved"]


def test_full_run_publish_is_explicit() -> None:
    command, _ = build_launcher_command(_args(publish=True))
    assert "--publish" in command


def test_local_trainable_profile_is_forwarded_via_environment() -> None:
    _, env = build_launcher_command(_args(
        prompt_profile="local_trainable_v1",
    ))
    assert env["LIVEMCP_PROMPT_PROFILE"] == "local_trainable_v1"


def test_build_cache_cli_accepts_the_generation_prompt_profile() -> None:
    args = cli.build_parser().parse_args([
        "build-cache",
        "--model", "Gemma-4-31B-it",
        "--prompt-profile", "local_trainable_v1",
    ])

    assert args.prompt_profile == "local_trainable_v1"


def test_semantic_gate_profile_is_orthogonal_to_prompt_profile() -> None:
    _, env = build_launcher_command(_args(
        prompt_profile="paper_generation_baseline_v1",
        semantic_gate_profile="diagnostic_only",
    ))
    assert env["LIVEMCP_PROMPT_PROFILE"] == "paper_generation_baseline_v1"
    assert env["LIVEMCP_SEMANTIC_GATE_PROFILE"] == "diagnostic_only"


def test_gray_difficulty_and_distractor_rate_are_explicitly_forwarded() -> None:
    command, _ = build_launcher_command(_args(
        difficulty="complete",
        distractor_rate=0.0,
    ))
    assert command[command.index("--difficulty") + 1] == "complete"
    assert command[command.index("--distractor-rate") + 1] == "0.0"


def test_full_tool_only_gray_overrides_are_explicitly_forwarded() -> None:
    command, _ = build_launcher_command(_args(
        tool_required_only=True,
        irrelevance_ratio=0.0,
        missing_function_rate=0.0,
    ))
    assert "--tool-required-only" in command
    assert command[command.index("--irrelevance-ratio") + 1] == "0.0"
    assert command[command.index("--missing-function-rate") + 1] == "0.0"


def test_mcp_supplement_is_provenance_isolated(tmp_path: Path) -> None:
    command, env = build_launcher_command(_args(
        mode="supplement",
        count=None,
        val_count=0,
        base=str(_base(tmp_path)),
        bucket="mcp_conversation",
        net_new=10,
        candidate_budget=20,
        domain="email",
    ))
    assert command[command.index("--domain") + 1] == "email"
    assert command[command.index("--count") + 1] == "10"
    assert command[command.index("--candidate-budget") + 1] == "20"
    assert "--tool-required-only" in command
    assert command[command.index("--missing-function-rate") + 1] == "0"
    assert command[command.index("--irrelevance-ratio") + 1] == "0"
    assert env == {
        "GEN_OVERSAMPLE_PCT": "0",
        "LIVEMCP_PROMPT_PROFILE": "paper_generation_baseline_v1",
        "LIVEMCP_SEMANTIC_GATE_PROFILE": "diagnostic_only",
        "PYTHON_BIN": sys.executable,
    }


def test_run_forwards_explicit_state_profile_suite(tmp_path: Path) -> None:
    suite = "configs/live_mcp/ten_domain_suite_payments_rare_state_v1.yaml"
    command, _ = build_launcher_command(_args(
        mode="supplement",
        count=None,
        val_count=0,
        base=str(_base(tmp_path)),
        bucket="mcp_conversation",
        net_new=5,
        candidate_budget=20,
        domain="payments",
        suite=suite,
    ))
    assert command[command.index("--suite") + 1] == suite


def test_gray_run_can_preserve_raw_candidates(tmp_path: Path) -> None:
    _, env = build_launcher_command(_args(
        mode="supplement",
        count=None,
        val_count=0,
        base=str(_base(tmp_path)),
        bucket="mcp_conversation",
        net_new=1,
        candidate_budget=8,
        domain="payments",
        preserve_candidates=True,
    ))
    assert env["GENERATION_PRESERVE_CANDIDATES"] == "1"


def test_fixed_attempt_budget_disables_all_candidate_topups() -> None:
    _, env = build_launcher_command(_args(fixed_attempt_budget=True))

    assert env["GEN_OVERSAMPLE_PCT"] == "0"
    assert env["GENERATION_MAX_RECOVERY_ROUNDS"] == "1"
    assert env["GENERATION_PRESERVE_CANDIDATES"] == "1"
    assert env["LIVEMCP_FIXED_ATTEMPT_BUDGET"] == "1"
    assert env["MERGE_TOPUP_ROUNDS"] == "0"


def test_launcher_honors_explicit_gpu_selection_before_vllm_reuse() -> None:
    launcher = Path("src/live_mcp/corpus/launcher.sh").read_text(
        encoding="utf-8"
    )
    default_block = """if [ -z "${GPU_FREE_ONLY+x}" ]; then
    if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
        GPU_FREE_ONLY=0
    else
        GPU_FREE_ONLY=1
    fi
fi
source scripts/gpu_config.sh"""
    assert default_block in launcher


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"publish": True}, "cannot publish"),
        ({"candidate_budget": 8}, "cannot be combined"),
        ({"mode": "supplement"}, "only valid for full run mode"),
    ],
)
def test_fixed_attempt_budget_rejects_production_contracts(
    overrides: dict, message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        build_launcher_command(_args(
            fixed_attempt_budget=True, **overrides,
        ))


def test_resume_forwards_existing_candidate_directory(tmp_path: Path) -> None:
    candidates = tmp_path / "candidates"
    candidates.mkdir()

    command, env = build_launcher_command(_args(
        command="resume",
        resume_candidate_dir=str(candidates),
    ))

    assert command[:2] == ["bash", "src/live_mcp/corpus/launcher.sh"]
    assert env["GENERATION_RESUME_CANDIDATE_DIR"] == str(candidates.resolve())


def test_resume_rejects_missing_candidate_directory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="resume candidate directory not found"):
        build_launcher_command(_args(
            command="resume",
            resume_candidate_dir=str(tmp_path / "missing"),
        ))


def test_mcp_supplement_forwards_chain_bin_quotas(tmp_path: Path) -> None:
    quotas = '{"1-2":3,"3-5":14,"6+":3}'
    command, _ = build_launcher_command(_args(
        mode="supplement",
        count=None,
        val_count=0,
        base=str(_base(tmp_path)),
        bucket="mcp_conversation",
        net_new=20,
        candidate_budget=86,
        domain="payments",
        chain_bin_quotas=quotas,
    ))
    assert command[command.index("--domain") + 1] == "payments"
    assert command[command.index("--chain-bin-quotas") + 1] == quotas


def test_missing_function_supplement_never_requires_tool_calls(
    tmp_path: Path,
) -> None:
    command, _ = build_launcher_command(_args(
        mode="supplement",
        count=None,
        val_count=0,
        base=str(_base(tmp_path)),
        bucket="missing_function",
        net_new=10,
    ))
    assert "--tool-required-only" not in command
    assert command[command.index("--missing-function-rate") + 1] == "1"
    assert command[command.index("--irrelevance-ratio") + 1] == "0"


def test_missing_function_rejects_chain_bin_quotas(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="only valid for mcp_conversation"):
        build_launcher_command(_args(
            mode="supplement",
            count=None,
            val_count=0,
            base=str(_base(tmp_path)),
            bucket="missing_function",
            net_new=10,
            chain_bin_quotas='{"1-2":2,"3-5":6,"6+":2}',
        ))


@pytest.mark.parametrize(
    "bucket", ["internal_abstention_proxy", "external_abstention"],
)
def test_unavailable_abstention_sources_fail_closed(bucket: str) -> None:
    with pytest.raises(ValueError):
        _bucket_generation_args(bucket)


def test_supplement_cannot_publish_directly(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="cannot publish directly"):
        build_launcher_command(_args(
            mode="supplement",
            count=None,
            base=str(_base(tmp_path)),
            bucket="mcp_conversation",
            net_new=1,
            publish=True,
        ))
