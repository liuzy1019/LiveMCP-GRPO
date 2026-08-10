"""Immutable task-compilation contracts for trainability-local gray tests.

The compiler intentionally contains no LLM calls and no entity values.  It
turns an already live-feasible dependency chain into a value-free contract
that Query Teacher may surface-realize and Action Teacher must realize through
real observations.  This is a local trainability contract, not a PROVE corpus
gate.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any


TASK_SPEC_VERSION = "taskspec_v1"
DECISION_STRATA = ("direct", "discovery", "dependent", "stateful")


@dataclass(frozen=True)
class ParameterBinding:
    source_capability: str
    target_capability: str
    target_argument: str
    source_output_field: str
    provenance_class: str


@dataclass(frozen=True)
class DifficultyVector:
    """Value-free, pre-Teacher decision-structure measurements."""

    selector_candidate_count: int
    viable_chain_count: int
    operational_dependency_count: int
    observation_derived_argument_count: int
    post_mutation_recheck_count: int
    distractor_count: int
    oracle_tool_count: int


@dataclass(frozen=True)
class TaskSpec:
    version: str
    domain: str
    session_seed: int
    state_profile: str
    state_fingerprint: str
    difficulty: str
    final_outcome_capability: str
    source_chain: tuple[str, ...]
    dependency_bindings: tuple[ParameterBinding, ...]
    user_decided_parameters: tuple[tuple[str, str], ...]
    natural_selector_types: tuple[str, ...]
    decision_stratum: str
    difficulty_vector: DifficultyVector
    static_strata: tuple[str, ...]
    robustness: tuple[tuple[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def fingerprint(self) -> str:
        payload = json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def to_prompt_contract(self) -> dict[str, Any]:
        """Return the value-free subset safe for Query Teacher.

        Session seeds, state fingerprints, and robustness bookkeeping are
        intentionally excluded: they do not help surface realization and can
        become accidental lexical shortcuts.
        """
        return {
            "version": self.version,
            "domain": self.domain,
            "difficulty": self.difficulty,
            "final_outcome_capability": self.final_outcome_capability,
            "source_chain": list(self.source_chain),
            "dependency_bindings": [
                asdict(binding) for binding in self.dependency_bindings
            ],
            "user_decided_parameters": [
                {"capability": capability, "argument": argument}
                for capability, argument in self.user_decided_parameters
            ],
            "natural_selector_types": list(self.natural_selector_types),
            "static_strata": list(self.static_strata),
        }


def compile_task_spec(
    *,
    domain: str,
    session_seed: int,
    state_profile: str,
    state_fingerprint: str,
    difficulty: str,
    source_chain: list[str],
    tool_schemas: list[dict[str, Any]],
    dependency_contracts: list[dict[str, str]],
    natural_selector_types: list[str] | tuple[str, ...],
    robustness: dict[str, Any],
    decision_stratum: str = "",
    difficulty_vector: DifficultyVector | None = None,
) -> TaskSpec:
    """Compile and validate one field-level dependency task.

    The input chain must already be live-feasible.  This compiler checks only
    deterministic schema and dependency facts; execution and semantic validity
    remain downstream replay gates.
    """
    if not 2 <= len(source_chain) <= 5:
        raise ValueError("TaskSpec source_chain must contain 2-5 capabilities")
    if decision_stratum and decision_stratum not in DECISION_STRATA:
        raise ValueError(
            f"unknown decision stratum {decision_stratum!r}; "
            f"expected one of {DECISION_STRATA}"
        )

    tools_by_name = {
        str(tool.get("name") or ""): tool for tool in tool_schemas
    }
    missing_tools = [name for name in source_chain if name not in tools_by_name]
    if missing_tools:
        raise ValueError(
            f"TaskSpec chain references unknown capabilities: {missing_tools}"
        )

    required_by_tool: dict[str, set[str]] = {}
    mutating_by_tool: dict[str, bool] = {}
    for name, tool in tools_by_name.items():
        input_schema = (
            tool.get("input_schema") or tool.get("inputSchema") or {}
        )
        required_by_tool[name] = {
            str(value) for value in (input_schema.get("required") or [])
        }
        mutating_by_tool[name] = bool(
            (tool.get("annotations") or {}).get("mutating") is True
        )

    bindings: list[ParameterBinding] = []
    bound_targets: set[tuple[str, str]] = set()
    for raw in dependency_contracts:
        source = str(raw.get("source_capability") or "")
        target = str(raw.get("target_capability") or "")
        argument = str(raw.get("target_argument") or "")
        output_field = str(raw.get("source_output_field") or "")
        if not all((source, target, argument, output_field)):
            continue
        if source not in source_chain or target not in source_chain:
            continue
        if source_chain.index(source) >= source_chain.index(target):
            continue
        if argument not in required_by_tool.get(target, set()):
            continue
        provenance_class = (
            "tool_derived" if mutating_by_tool.get(source, False)
            else "tool_discoverable"
        )
        binding = ParameterBinding(
            source_capability=source,
            target_capability=target,
            target_argument=argument,
            source_output_field=output_field,
            provenance_class=provenance_class,
        )
        if binding not in bindings:
            bindings.append(binding)
            bound_targets.add((target, argument))
    if not bindings:
        raise ValueError(
            "TaskSpec requires at least one operational field-level dependency"
        )

    user_decided: list[tuple[str, str]] = []
    for capability in source_chain:
        for argument in sorted(required_by_tool.get(capability, set())):
            key = (capability, argument)
            if key not in bound_targets:
                user_decided.append(key)

    strata = {
        "observation_derived_argument",
        f"dependency_depth_{len(source_chain)}",
        (
            "mutating_final_outcome"
            if mutating_by_tool.get(source_chain[-1], False)
            else "readonly_final_outcome"
        ),
    }
    provenance_classes = {binding.provenance_class for binding in bindings}
    strata.update(provenance_classes)
    if natural_selector_types:
        strata.add("natural_selector")
    if difficulty_vector is None:
        difficulty_vector = DifficultyVector(
            selector_candidate_count=0,
            viable_chain_count=1,
            operational_dependency_count=len(bindings),
            observation_derived_argument_count=len(bound_targets),
            post_mutation_recheck_count=0,
            distractor_count=int(bool(robustness.get("distractors"))),
            oracle_tool_count=len(source_chain),
        )

    normalized_robustness = tuple(sorted(
        (str(key), value) for key, value in robustness.items()
    ))
    return TaskSpec(
        version=TASK_SPEC_VERSION,
        domain=str(domain),
        session_seed=int(session_seed),
        state_profile=str(state_profile),
        state_fingerprint=str(state_fingerprint),
        difficulty=str(difficulty),
        final_outcome_capability=str(source_chain[-1]),
        source_chain=tuple(str(value) for value in source_chain),
        dependency_bindings=tuple(bindings),
        user_decided_parameters=tuple(user_decided),
        natural_selector_types=tuple(sorted({
            str(value) for value in natural_selector_types if str(value)
        })),
        decision_stratum=str(decision_stratum),
        difficulty_vector=difficulty_vector,
        static_strata=tuple(sorted(strata)),
        robustness=normalized_robustness,
    )
