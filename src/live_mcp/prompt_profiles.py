"""Explicit prompt contracts for causal data-generation gray tests."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PromptProfile:
    name: str
    paper_baseline: bool = False
    policy_private: bool = False
    natural_selector: bool = False
    dependency_necessary: bool = False


PAPER_GENERATION_BASELINE = PromptProfile(
    name="paper_generation_baseline_v1",
    paper_baseline=True,
    policy_private=True,
)


PROMPT_PROFILES: dict[str, PromptProfile] = {
    "paper_generation_baseline_v1": PAPER_GENERATION_BASELINE,
    "local_trainable_v1": PromptProfile(
        name="local_trainable_v1",
        policy_private=True,
        natural_selector=True,
        dependency_necessary=True,
    ),
}


def resolve_prompt_profile(value: str | PromptProfile) -> PromptProfile:
    if isinstance(value, PromptProfile):
        return value
    name = str(value or "").strip()
    try:
        return PROMPT_PROFILES[name]
    except KeyError as exc:
        raise ValueError(
            f"unknown prompt profile {name!r}; "
            f"expected one of {sorted(PROMPT_PROFILES)}"
        ) from exc


def requires_outcome_replay(value: str | PromptProfile) -> bool:
    """Return whether replayed task outcomes are a profile hard gate.

    PROVE's published replay filter counts schema and execution errors only.
    Reproducing locally-derived success criteria is a training-consumability
    extension owned by the local profile.
    """
    if not isinstance(value, PromptProfile) and not str(value or "").strip():
        # Legacy artifacts did not persist a prompt profile. Keep their prior
        # fail-closed outcome requirement; never infer paper semantics.
        return True
    return not resolve_prompt_profile(value).paper_baseline


__all__ = [
    "PAPER_GENERATION_BASELINE",
    "PROMPT_PROFILES",
    "PromptProfile",
    "requires_outcome_replay",
    "resolve_prompt_profile",
]
