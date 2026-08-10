"""Shared reward-domain adapter contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from src.live_mcp.registry.tool_semantics import build_tool_semantics


class DomainAdapter(ABC):
    """Abstract base for domain adapters.

    Each MCP server domain must implement normalize_event, protected_resources,
    budget, and identity_policy. Predicate methods have sensible defaults
    shared across all domains.
    """

    domain_name: str

    def register_tool_schemas(self, tools: list[dict[str, Any]]) -> None:
        """Bind formal operations from live annotations and reviewed writes."""
        contracts = build_tool_semantics(self.domain_name, tools)
        self._schema_operations = {
            name: contract.operation for name, contract in contracts.items()
        }

    @abstractmethod
    def normalize_event(
        self,
        action_type: str,
        tool_name: str,
        tool_arguments: dict[str, Any],
        observation: dict[str, Any] | str | None,
        execution_success: bool,
        state_changed: bool,
        before_state: dict[str, Any] | None,
        after_state: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Map a raw tool_call/terminal to domain-generic event fields.

        Returns a dict with keys:
          operation, target_type, target_id,
          changed_fields, created_ids, deleted_ids,
          identity_violation, forbidden_transition, duplicate_of
        """
        ...

    # ── Predicates shared across all domains ──

    def outcome_predicates(self, task: dict[str, Any]) -> list[str]:
        return [
            "required_resource_resolved",
            "required_transition_completed",
            "required_output_fields_match",
            "task_required_fields_preserved",
            "final_response_satisfies_task",
        ]

    def safety_predicates(self, task: dict[str, Any]) -> list[str]:
        return [
            "not_forbidden_transition",
            "not_wrong_resource_mutation",
            "not_identity_or_provenance_violation",
            "not_duplicate_or_inconsistent_side_effect",
        ]

    def progress_predicates(self, task: dict[str, Any]) -> list[str]:
        return [
            "resolved_required_entity",
            "completed_required_transition",
            "verified_postcondition",
        ]

    def required_tool_names(self, task: dict[str, Any]) -> set[str]:
        calls = task.get("required_tool_calls", [])
        return {c["tool_name"] for c in calls} if calls else set()

    @property
    def entity_container_key(self) -> str:
        """Key in domain state that holds the primary entity container for recreate detection.

        Override per domain: "events" for calendar, "accounts" for banking, etc.
        """
        return "events"  # default for calendar

    @staticmethod
    def _unwrap_domain_state(state: dict[str, Any] | None, domain_name: str) -> dict[str, Any] | None:
        """Unwrap the domain-specific state from the manager's composite state.

        manager.get_state() returns {"calendar": {"events": {...}}, ...}
        This extracts the inner domain dict, or falls back to the raw state.
        """
        if state is None:
            return None
        domain_state = state.get(domain_name, None)
        if isinstance(domain_state, dict):
            return domain_state
        return state

    def tool_semantics(
        self,
        tool_name: str,
        default_target_type: str,
        state_changed: bool = False,
    ) -> tuple[str, str]:
        """Resolve one tool through the registered executable schema."""
        explicit = getattr(self, "TOOL_MAP", {}).get(tool_name)
        schema_operation = getattr(self, "_schema_operations", {}).get(tool_name)
        if schema_operation:
            target_type = explicit[1] if explicit else default_target_type
            return schema_operation, target_type
        raise ValueError(
            f"unregistered tool contract: {self.domain_name}.{tool_name}"
        )

    @staticmethod
    def generic_target_id(
        tool_arguments: dict[str, Any],
        observation: dict[str, Any] | str | None = None,
    ) -> str:
        for key, value in tool_arguments.items():
            if key.lower().endswith("_id") and isinstance(value, str):
                return value
        for key in ("path", "source", "destination", "from_account", "to_account"):
            value = tool_arguments.get(key)
            if isinstance(value, str) and value:
                return value
        if isinstance(observation, dict):
            stack = [observation]
            while stack:
                current = stack.pop()
                for key, value in current.items():
                    if key.lower().endswith("_id") and isinstance(value, str):
                        return value
                    if isinstance(value, dict):
                        stack.append(value)
        return ""

    # ── Domain-specific abstract methods ──

    @abstractmethod
    def protected_resources(self, task: dict[str, Any]) -> list[str]:
        """Return protected resource IDs for this task."""
        ...

    @abstractmethod
    def budget(self, task: dict[str, Any]) -> int:
        """Return the call budget for this task."""
        ...

    @abstractmethod
    def identity_policy(self, task: dict[str, Any]) -> str:
        """Return the identity policy: preserve | create_new | append_only | lookup_only."""
        ...

    # ── Predicate evaluation ──

    def evaluate_event(
        self,
        event: Any,
        task: dict[str, Any],
    ) -> frozenset[str]:
        """Return the set of progress predicate names satisfied by this event.

        This is the single source of truth for predicate completion used by
        R_coverage, F_gamma, and P_process.  Domain adapters SHOULD override
        this when domain-specific semantics differ from the generic mapping.

        Generic mapping (works for most domains):
          query + success → {resolved_required_entity}
          create/update/delete + success → {completed_required_transition, resolved_required_entity}
          final_answer → {verified_postcondition, produced_required_response}
          ask_clarification/report_error → {produced_required_response}
        """
        predicates: set[str] = set()

        if not getattr(event, "execution_success", False):
            return frozenset()

        op = getattr(event, "operation", "")
        action = getattr(event, "action_type", "")

        # Query / read operations
        if op == "query":
            predicates.add("resolved_required_entity")

        # State-changing operations
        if op in ("create", "update", "delete"):
            predicates.add("completed_required_transition")
            predicates.add("resolved_required_entity")  # implies entity was resolved

        # Terminal actions
        if action == "final_answer":
            predicates.add("verified_postcondition")
            predicates.add("produced_required_response")
        elif action in ("ask_clarification", "report_error"):
            predicates.add("produced_required_response")

        return frozenset(predicates)


__all__ = ["DomainAdapter"]
