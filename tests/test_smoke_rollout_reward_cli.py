from __future__ import annotations

import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "smoke_rollout_reward.sh"


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(SCRIPT), *args],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_help_does_not_require_runtime_inputs() -> None:
    completed = _run("--help")

    assert completed.returncode == 0
    assert "--reward-profile NAME" in completed.stdout
    assert "--experiment-profile NAME" in completed.stdout


def test_reward_profile_is_required() -> None:
    completed = _run("--gpus", "4,5,6,7")

    assert completed.returncode == 2
    assert "--reward-profile is required" in completed.stderr


def test_unknown_reward_profile_fails_before_launch() -> None:
    completed = _run(
        "--gpus",
        "4,5,6,7",
        "--reward-profile",
        "not-a-profile",
        "--experiment-profile",
        "prove_local_v1",
    )

    assert completed.returncode == 2
    assert "--reward-profile must be oval_full or prove_baseline" in completed.stderr


def test_experiment_profile_is_required() -> None:
    completed = _run(
        "--gpus",
        "4,5,6,7",
        "--reward-profile",
        "prove_baseline",
    )

    assert completed.returncode == 2
    assert "--experiment-profile is required" in completed.stderr


def test_reward_and_experiment_profiles_must_match() -> None:
    completed = _run(
        "--gpus",
        "4,5,6,7",
        "--reward-profile",
        "oval_full",
        "--experiment-profile",
        "prove_local_v1",
    )

    assert completed.returncode == 2
    assert "PROVE experiment profiles require" in completed.stderr


def test_custom_profile_accepts_explicit_diagnostic_data_contract() -> None:
    completed = _run(
        "--gpus",
        "not-a-gpu-list",
        "--reward-profile",
        "prove_baseline",
        "--experiment-profile",
        "custom",
    )

    # The custom profile itself is accepted; validation advances to the
    # deliberately invalid GPU list without launching any process.
    assert completed.returncode == 2
    assert "unsupported --experiment-profile" not in completed.stderr
    assert "--gpus must be a comma-separated list" in completed.stderr
