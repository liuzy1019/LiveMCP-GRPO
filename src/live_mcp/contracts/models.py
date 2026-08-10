"""Typed, domain-neutral facts used by the PROVE pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


BindingSource = Literal["argument", "output", "global"]


@dataclass(frozen=True)
class ArgumentValue:
    """Resolve a predicate value from the current tool-call arguments."""

    name: str


@dataclass(frozen=True, order=True)
class EntityBinding:
    """Bind an entity identity to a tool argument, output field, or global state."""

    entity_type: str
    name: str
    source: BindingSource = "argument"


@dataclass(frozen=True)
class StatePredicate:
    """One state fact over an entity identity.

    Predicates sharing ``slot`` and the same resolved subject are mutually
    exclusive when their values differ.  This gives the simulator explicit
    invalidation semantics without domain-specific branches.
    """

    slot: str
    subject: EntityBinding
    value: Any = True
    observed_entity_required: bool = True

    def identity(self) -> tuple[str, EntityBinding]:
        return self.slot, self.subject


@dataclass(frozen=True)
class ToolContract:
    domain: str
    name: str
    readonly: bool
    mutating: bool
    arguments: frozenset[str] = frozenset()
    required_arguments: frozenset[str] = frozenset()
    required_entity_types: frozenset[str] = frozenset()
    minimum_entity_counts: tuple[tuple[str, int], ...] = ()
    input_entities: tuple[EntityBinding, ...] = ()
    output_entities: tuple[EntityBinding, ...] = ()
    output_fields: frozenset[str] = frozenset()
    created_output_fields: frozenset[str] = frozenset()
    preconditions: tuple[StatePredicate, ...] = ()
    precondition_groups: tuple[tuple[StatePredicate, ...], ...] = ()
    postconditions: tuple[StatePredicate, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict, compare=False)

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not self.name:
            issues.append("missing tool name")
        if self.readonly and self.mutating:
            issues.append("tool cannot be both readonly and mutating")
        for binding in self.input_entities:
            if binding.source != "argument":
                issues.append(
                    f"input entity {binding.entity_type} must bind an argument"
                )
            if binding.name not in (self.arguments or self.required_arguments):
                issues.append(
                    f"input entity binding {binding.name} is not a tool argument"
                )
        for binding in self.output_entities:
            if binding.source != "output":
                issues.append(
                    f"output entity {binding.entity_type} must bind an output field"
                )
            if binding.name not in self.output_fields:
                issues.append(
                    f"output entity binding {binding.name} is not an output field"
                )
        if not self.created_output_fields <= self.output_fields:
            issues.append("created output fields must be declared output fields")
        for entity_type, count in self.minimum_entity_counts:
            if entity_type not in self.required_entity_types:
                issues.append(
                    f"minimum count declared for non-required entity {entity_type}"
                )
            if count < 1:
                issues.append("minimum entity count must be positive")
        for predicate in self.preconditions:
            if (
                predicate.subject.source == "argument"
                and predicate.subject.name not in (self.arguments or self.required_arguments)
            ):
                issues.append(
                    f"precondition binding {predicate.subject.name} is not a tool argument"
                )
        for group in self.precondition_groups:
            if not group:
                issues.append("precondition any-of group cannot be empty")
            for predicate in group:
                if (
                    predicate.subject.source == "argument"
                    and predicate.subject.name
                    not in (self.arguments or self.required_arguments)
                ):
                    issues.append(
                        f"precondition binding {predicate.subject.name} is not a tool argument"
                    )
        for predicate in self.postconditions:
            if (
                predicate.subject.source == "output"
                and predicate.subject.name not in self.output_fields
            ):
                issues.append(
                    f"postcondition binding {predicate.subject.name} is not an output field"
                )
        return tuple(issues)


@dataclass(frozen=True)
class ToolStateFacts:
    """State portion of a tool contract, audited against its handler."""

    preconditions: tuple[StatePredicate, ...] = ()
    precondition_groups: tuple[tuple[StatePredicate, ...], ...] = ()
    postconditions: tuple[StatePredicate, ...] = ()
