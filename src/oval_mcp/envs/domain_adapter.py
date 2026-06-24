"""DomainAdapter: normalize domain-specific events to verifier predicates.

OVAL-MCP §5.2: Only DomainAdapter outputs enter reward/cost.
Algorithm does not depend on calendar/shopping-specific fields.

Phase 1 adapters: CalendarAdapter, ShoppingAdapter.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class DomainAdapter(ABC):
    """Abstract base for domain adapters.

    Each MCP server domain must implement this to map
    tool_calls + observations + state_diffs → normalized AuditEvents
    and to provide predicates for reward/cost computation.
    """

    domain_name: str

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

    @abstractmethod
    def outcome_predicates(self, task: dict[str, Any]) -> list[str]:
        """Return outcome predicate names for this task."""
        ...

    @abstractmethod
    def safety_predicates(self, task: dict[str, Any]) -> list[str]:
        """Return safety predicate names for this task."""
        ...

    @abstractmethod
    def progress_predicates(self, task: dict[str, Any]) -> list[str]:
        """Return progress predicate names for this task."""
        ...

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

    @abstractmethod
    def required_tool_names(self, task: dict[str, Any]) -> set[str]:
        """Return the set of required tool names for this task."""
        ...


class CalendarAdapter(DomainAdapter):
    """Domain adapter for the calendar MCP server.

    Calendar state:
      events: dict[event_id -> {event_id, title, start_time, end_time, attendees}]
      next_event_num: int

    target_type: "calendar_event"
    identity_policy: typically "preserve" (update, don't delete+recreate)
    """

    domain_name = "calendar"

    # Tool -> (operation, target_type)
    TOOL_MAP = {
        "list_events": ("query", "calendar_event"),
        "create_event": ("create", "calendar_event"),
        "update_event": ("update", "calendar_event"),
        "delete_event": ("delete", "calendar_event"),
    }

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
        result: dict[str, Any] = {
            "operation": "",
            "target_type": "calendar_event",
            "target_id": "",
            "changed_fields": [],
            "created_ids": [],
            "deleted_ids": [],
            "identity_violation": "",
            "forbidden_transition": "",
            "duplicate_of": None,
        }

        if action_type != "tool_call":
            result["operation"] = "terminal"
            return result

        op, target = self.TOOL_MAP.get(tool_name, ("unknown", "calendar_event"))
        result["operation"] = op
        result["target_type"] = target

        # Extract target_id from arguments
        if tool_name == "create_event":
            if execution_success and isinstance(observation, dict):
                event = observation.get("event", observation.get("observation", {}))
                if isinstance(event, dict):
                    result["target_id"] = event.get("event_id", "")
            # Detect created IDs from state diff
            if before_state and after_state:
                before_events = set(before_state.get("events", {}).keys())
                after_events = set(after_state.get("events", {}).keys())
                result["created_ids"] = list(after_events - before_events)

        elif tool_name == "update_event":
            result["target_id"] = tool_arguments.get("event_id", "")
            if execution_success and isinstance(observation, dict):
                event = observation.get("event", observation.get("observation", {}))
                if isinstance(event, dict):
                    result["target_id"] = event.get("event_id", result["target_id"])
            # Detect changed fields
            fields = tool_arguments.get("fields", {})
            if isinstance(fields, dict):
                result["changed_fields"] = list(fields.keys())

        elif tool_name == "delete_event":
            result["target_id"] = tool_arguments.get("event_id", "")
            # Detect deleted IDs from state diff
            if before_state and after_state:
                before_events = set(before_state.get("events", {}).keys())
                after_events = set(after_state.get("events", {}).keys())
                result["deleted_ids"] = list(before_events - after_events)

        elif tool_name == "list_events":
            result["target_id"] = ""

        # Forbidden transition detection:
        # delete + create with same/similar target is a forbidden pattern
        # This is detected across events by SafetyVerifier, not per-event.
        # But we can set a preliminary flag here if needed.

        return result

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

    def protected_resources(self, task: dict[str, Any]) -> list[str]:
        # Calendar: protected resources are target event IDs that must not be deleted
        return task.get("protected_event_ids", [])

    def budget(self, task: dict[str, Any]) -> int:
        return task.get("budget", 5)

    def identity_policy(self, task: dict[str, Any]) -> str:
        return task.get("identity_policy", "preserve")

    def required_tool_names(self, task: dict[str, Any]) -> set[str]:
        calls = task.get("required_tool_calls", [])
        return {c["tool_name"] for c in calls} if calls else set()


class ShoppingAdapter(DomainAdapter):
    """Domain adapter for the shopping MCP server.

    Shopping state:
      products: dict[product_id -> {name, category, price, stock, ...}]
      cart: list[{product_id, quantity, unit_price}]
      orders: dict[order_id -> {order_id, items, total}]
      next_order_num: int

    target_type: "shopping_order" / "shopping_cart" / "product"
    identity_policy: typically "create_new" (orders are new IDs)
    """

    domain_name = "shopping"

    TOOL_MAP = {
        "search_products": ("query", "product"),
        "add_to_cart": ("update", "shopping_cart"),
        "remove_from_cart": ("update", "shopping_cart"),
        "checkout": ("create", "shopping_order"),
        "get_order": ("query", "shopping_order"),
    }

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
        result: dict[str, Any] = {
            "operation": "",
            "target_type": "",
            "target_id": "",
            "changed_fields": [],
            "created_ids": [],
            "deleted_ids": [],
            "identity_violation": "",
            "forbidden_transition": "",
            "duplicate_of": None,
        }

        if action_type != "tool_call":
            result["operation"] = "terminal"
            return result

        op, target = self.TOOL_MAP.get(tool_name, ("unknown", "unknown"))
        result["operation"] = op
        result["target_type"] = target

        if tool_name == "add_to_cart":
            result["target_id"] = tool_arguments.get("product_id", "")

        elif tool_name == "remove_from_cart":
            result["target_id"] = tool_arguments.get("product_id", "")

        elif tool_name == "checkout":
            if execution_success and isinstance(observation, dict):
                order = observation.get("order", observation.get("observation", {}))
                if isinstance(order, dict):
                    result["target_id"] = order.get("order_id", "")
            if before_state and after_state:
                before_orders = set(before_state.get("orders", {}).keys())
                after_orders = set(after_state.get("orders", {}).keys())
                result["created_ids"] = list(after_orders - before_orders)

        elif tool_name == "get_order":
            result["target_id"] = tool_arguments.get("order_id", "")

        elif tool_name == "search_products":
            result["target_id"] = ""

        return result

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

    def protected_resources(self, task: dict[str, Any]) -> list[str]:
        return task.get("protected_product_ids", [])

    def budget(self, task: dict[str, Any]) -> int:
        return task.get("budget", 4)

    def identity_policy(self, task: dict[str, Any]) -> str:
        return task.get("identity_policy", "create_new")

    def required_tool_names(self, task: dict[str, Any]) -> set[str]:
        calls = task.get("required_tool_calls", [])
        return {c["tool_name"] for c in calls} if calls else set()


# Registry of known adapters
_ADAPTERS: dict[str, DomainAdapter] = {}


def get_adapter(domain_name: str) -> DomainAdapter:
    """Get or create a domain adapter by name."""
    if domain_name not in _ADAPTERS:
        if domain_name == "calendar":
            _ADAPTERS[domain_name] = CalendarAdapter()
        elif domain_name == "shopping":
            _ADAPTERS[domain_name] = ShoppingAdapter()
        else:
            raise ValueError(f"unknown domain: {domain_name}")
    return _ADAPTERS[domain_name]


__all__ = [
    "DomainAdapter",
    "CalendarAdapter",
    "ShoppingAdapter",
    "get_adapter",
]
