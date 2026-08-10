"""Explicit prompt contracts for causal data-generation gray tests."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PromptProfile:
    name: str
    paper_baseline: bool = False
    policy_private: bool = False
    natural_selector: bool = False
    flexible_expression: bool = False
    dependency_necessary: bool = False
    task_spec: bool = False
    decision_stratified: bool = False


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
        flexible_expression=True,
        dependency_necessary=True,
        task_spec=True,
        decision_stratified=True,
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
