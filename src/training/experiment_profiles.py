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


PROFILES = {
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
            "OVAL_EXPERIMENT_PROFILE must be custom or "
            f"prove_reproduction_v1; got {name!r}"
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
