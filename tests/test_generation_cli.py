from __future__ import annotations

from argparse import Namespace
import json
from pathlib import Path
import subprocess
import sys

import pytest

from src.live_mcp.corpus.cli import (
    _bucket_generation_args,
    _generated_parquet_summary,
    _reserve_new_run_directory,
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
        "publish": False,
        "dry_run": True,
    }
    values.update(overrides)
    return Namespace(**values)


def _base(tmp_path: Path) -> Path:
    base = tmp_path / "base"
    base.mkdir()
    for name in ("train.parquet", "val.parquet", "merge_report.json"):
        (base / name).write_bytes(b"test")
    return base


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
