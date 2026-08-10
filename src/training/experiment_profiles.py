"""Fail-closed experiment profiles for PROVE/OVAL comparisons.

Reward profiles select an objective.  Experiment profiles additionally freeze
the model, immutable data artifact, and published training-shape parameters so
that a run cannot be mislabeled as a baseline after silent overrides.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.utils import sha256_file


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass(frozen=True)
class ExperimentProfile:
    name: str
    reward_profile: str
    frozen: dict[str, Any]
    artifact_sha256: dict[str, str]
    unavailable_reason: str = ""


_LOCAL_COMMON = {
    "model_path": "models/Qwen/Qwen3-4B-Instruct-2507",
    "train_file": (
        "data/runs/"
        "20260728_gt_v1_prove_composition_proxy_r48dee_train3221_val500/"
        "train.parquet"
    ),
    "val_file": (
        "data/runs/"
        "20260728_gt_v1_prove_composition_proxy_r48dee_train3221_val500/"
        "val.parquet"
    ),
    "total_steps": 350,
    "train_batch_size": 16,
    "rollout_n": 16,
    "rollout_tp": 2,
    "max_prompt_length": 12384,
    "max_response_length": 16384,
    "max_user_turns": 5,
    "max_assistant_turns": 10,
    "lr": 1e-6,
    "kl_coef": 0.01,
    "ppo_epochs": 1,
    "agent_loop": "livemcp_oval",
}

_LOCAL_ARTIFACT_SHA256 = {
    "train_file": "e55d6e3bfe53b4478884079ce71083b64c5f23b9cdcffd10d2fe5b70345b598b",
    "val_file": "09e0f953655e5e54d8ab172d840e218a3b7de42de930a22c18e3b440f555b4ec",
    "model_config": "5beea1a4a34c62782bfb2f911c606741a3bab8f92d80a118fa053c28af12e8ba",
}

_GRAY_COMMON = {
    **_LOCAL_COMMON,
    "train_file": (
        "data/runs/20260728_reward_gray_r48dee_v1_train8/train.parquet"
    ),
}

_GRAY_ARTIFACT_SHA256 = {
    **_LOCAL_ARTIFACT_SHA256,
    "train_file": "4b12c1ee3e097321469d1962728a8017374b346626f8a0c4819335ddf94107eb",
}

PROFILES = {
    "prove_local_v1": ExperimentProfile(
        name="prove_local_v1",
        reward_profile="prove_baseline",
        frozen=dict(_LOCAL_COMMON),
        artifact_sha256=dict(_LOCAL_ARTIFACT_SHA256),
    ),
    "oval_local_v1": ExperimentProfile(
        name="oval_local_v1",
        reward_profile="oval_full",
        frozen=dict(_LOCAL_COMMON),
        artifact_sha256=dict(_LOCAL_ARTIFACT_SHA256),
    ),
    "prove_reward_gray_v1": ExperimentProfile(
        name="prove_reward_gray_v1",
        reward_profile="prove_baseline",
        frozen=dict(_GRAY_COMMON),
        artifact_sha256=dict(_GRAY_ARTIFACT_SHA256),
    ),
    "oval_reward_gray_v1": ExperimentProfile(
        name="oval_reward_gray_v1",
        reward_profile="oval_full",
        frozen=dict(_GRAY_COMMON),
        artifact_sha256=dict(_GRAY_ARTIFACT_SHA256),
    ),
    "prove_reproduction_v1": ExperimentProfile(
        name="prove_reproduction_v1",
        reward_profile="prove_baseline",
        frozen={},
        artifact_sha256={},
        unavailable_reason=(
            "strict PROVE reproduction requires the published 20-domain/"
            "343-tool environment and 13,517-row corpus with traceable "
            "When2Call and xLAM-Irrelevance sources; this checkout currently "
            "provides a 10-domain internal-proxy baseline"
        ),
    ),
}

# These are the only paper-shape fields a diagnostic run may change.  The
# resulting config records diagnostic_overrides=True and must not be reported
# as a full paper-shape run.
DIAGNOSTIC_OVERRIDE_FIELDS = frozenset({
    "total_steps",
    "train_batch_size",
})


def get_experiment_profile(name: str) -> ExperimentProfile | None:
    if name == "custom":
        return None
    if name not in PROFILES:
        raise ValueError(
            "OVAL_EXPERIMENT_PROFILE must be custom, prove_local_v1, "
            "oval_local_v1, prove_reward_gray_v1, oval_reward_gray_v1, "
            f"or prove_reproduction_v1; got {name!r}"
        )
    profile = PROFILES[name]
    if profile.unavailable_reason:
        raise ValueError(
            f"experiment profile {name!r} is unavailable: "
            f"{profile.unavailable_reason}"
        )
    return profile


def merge_profile_values(
    profile: ExperimentProfile | None,
    requested: dict[str, Any],
    *,
    diagnostic_overrides: bool,
) -> dict[str, Any]:
    if profile is None:
        return dict(requested)

    merged = dict(profile.frozen)
    for field, value in requested.items():
        if field in profile.frozen and value != profile.frozen[field]:
            if not (
                diagnostic_overrides
                and field in DIAGNOSTIC_OVERRIDE_FIELDS
            ):
                raise ValueError(
                    f"experiment profile {profile.name!r} freezes "
                    f"{field}={profile.frozen[field]!r}; got {value!r}"
                )
        merged[field] = value
    return merged


def validate_profile_artifacts(
    profile: ExperimentProfile | None,
    values: dict[str, Any],
) -> dict[str, str]:
    if profile is None:
        return {}

    targets = {
        "train_file": PROJECT_ROOT / str(values["train_file"]),
        "val_file": PROJECT_ROOT / str(values["val_file"]),
        "model_config": (
            PROJECT_ROOT / str(values["model_path"]) / "config.json"
        ),
    }
    observed: dict[str, str] = {}
    for key, path in targets.items():
        if not path.is_file():
            raise ValueError(
                f"experiment profile {profile.name!r} artifact missing: {path}"
            )
        observed[key] = sha256_file(path)
        expected = profile.artifact_sha256[key]
        if observed[key] != expected:
            raise ValueError(
                f"experiment profile {profile.name!r} {key} SHA256 mismatch: "
                f"expected={expected}, observed={observed[key]}"
            )
    return observed


__all__ = [
    "DIAGNOSTIC_OVERRIDE_FIELDS",
    "ExperimentProfile",
    "PROFILES",
    "get_experiment_profile",
    "merge_profile_values",
    "validate_profile_artifacts",
]
