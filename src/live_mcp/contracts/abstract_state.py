"""Domain-neutral symbolic state simulation for dependency chains."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.live_mcp.contracts.models import (
    ArgumentValue,
    EntityBinding,
    StatePredicate,
)


@dataclass(frozen=True)
class SimulationIssue:
    step: int
    tool_name: str
    predicate: StatePredicate
    reason: str


@dataclass
class AbstractState:
    facts: dict[tuple[str, str], Any] = field(default_factory=dict)

    @staticmethod
    def _subject_key(binding: EntityBinding, bindings: dict[str, str]) -> str:
        if binding.source == "global":
            return binding.name
        return bindings.get(
            f"{binding.source}:{binding.name}",
            bindings.get(binding.name, f"unknown:{binding.name}"),
        )

    def observe(
        self,
        predicate: StatePredicate,
        bindings: dict[str, str],
    ) -> None:
        subject = self._subject_key(predicate.subject, bindings)
        value = self._predicate_value(predicate, bindings)
        self.facts[(predicate.slot, subject)] = value

    @staticmethod
    def _predicate_value(
        predicate: StatePredicate,
        bindings: dict[str, str],
    ) -> Any:
        if not isinstance(predicate.value, ArgumentValue):
            return predicate.value
        resolved = bindings.get(
            f"argument:{predicate.value.name}",
            bindings.get(predicate.value.name),
        )
        # Name-only chain simulation binds an unconstrained call argument to a
        # ``live:...`` symbol.  That symbol means "chosen at execution time",
        # not a concrete state value.  Preserve it as unknown so a dynamic
        # transition neither proves nor contradicts a later concrete state.
        if resolved is None or str(resolved).startswith("live:"):
            return predicate.value
        return resolved

    def evaluate(
        self,
        predicate: StatePredicate,
        bindings: dict[str, str],
    ) -> bool | None:
        subject = self._subject_key(predicate.subject, bindings)
        value = self.facts.get((predicate.slot, subject))
        if value is None:
            return None
        expected = self._predicate_value(predicate, bindings)
        if isinstance(value, ArgumentValue) or isinstance(expected, ArgumentValue):
            return None
        return value == expected
