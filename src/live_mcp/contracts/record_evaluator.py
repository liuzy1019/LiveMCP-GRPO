"""Evaluate typed tool preconditions against readonly live entity records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.live_mcp.contracts.models import StatePredicate, ToolContract


@dataclass(frozen=True)
class RecordRule:
    fields: tuple[str, ...]
    operator: str
    operand: Any = None


_RULES: dict[tuple[str, str], RecordRule] = {
    ("banking", "account.balance_sufficient"): RecordRule(("balance",), "gt", 0),
    ("banking", "account.collateral_sufficient"): RecordRule(("balance",), "gt", 0),
    ("banking", "scheduled_transfer.cancellable"): RecordRule(
        ("status",), "not_in", frozenset({"executed", "cancelled"}),
    ),
    ("calendar", "event.recurring"): RecordRule(
        ("recurrence",), "present",
    ),
    ("crm", "lead.convertible"): RecordRule(
        ("status",), "not_in", frozenset({"converted", "lost"}),
    ),
    ("crm", "lead.deletable"): RecordRule(("deletable",), "truthy"),
    ("crm", "contact.deletable"): RecordRule(("deletable",), "truthy"),
    ("filesystem", "filesystem.protected"): RecordRule(
        ("path",), "path_protected",
    ),
    ("filesystem", "filesystem.deletable"): RecordRule(
        ("path",), "path_deletable",
    ),
    ("filesystem", "filesystem.ownership_change_allowed"): RecordRule(
        ("path", "owner"), "ownership_change_allowed",
    ),
    ("filesystem", "filesystem.archive"): RecordRule(("archive",), "truthy"),
    ("food_delivery", "order.cancellable"): RecordRule(
        ("status",), "in", frozenset({"placed", "confirmed"}),
    ),
    ("food_delivery", "order.transition_allowed"): RecordRule(
        ("status",), "in",
        frozenset({"placed", "confirmed", "preparing", "delivering"}),
    ),
    ("food_delivery", "restaurant.menu_nonempty"): RecordRule(
        ("menu", "items"), "nonempty",
    ),
    ("food_delivery", "order.tip_present"): RecordRule(("tip",), "truthy"),
    ("food_delivery", "order.rated"): RecordRule(("rating",), "present"),
    ("issue_tracker", "issue.transition_allowed"): RecordRule(
        ("state",), "not_in", frozenset({"closed", "cancelled"}),
    ),
    ("payments", "invoice.payable"): RecordRule(
        ("status",), "in", frozenset({"pending", "overdue"}),
    ),
    ("payments", "invoice.refundable"): RecordRule(
        ("status",), "in", frozenset({"paid", "partially_refunded"}),
    ),
    ("payments", "invoice.disputable"): RecordRule(
        ("status",), "in", frozenset({"paid", "pending"}),
    ),
    ("payments", "invoice.payment_linked"): RecordRule(
        ("payment_id",), "present",
    ),
    ("payments", "invoice.payment_settled"): RecordRule(
        ("payment_status",), "equals", "settled",
    ),
    ("payments", "invoice.dispute_open"): RecordRule(
        ("dispute_open",), "truthy",
    ),
    ("shopping", "order.cancellable"): RecordRule(
        ("status",), "in", frozenset({"placed", "pending"}),
    ),
    ("shopping", "product.stock_sufficient"): RecordRule(
        ("stock", "available", "in_stock"), "any_positive",
    ),
    ("shopping", "product.review_eligible"): RecordRule(
        ("review_eligible",), "truthy",
    ),
    ("shopping", "product.reviewed_by_user"): RecordRule(
        ("reviews",), "current_user_reviewed",
    ),
    ("shopping", "cart.membership"): RecordRule(("cart_member",), "truthy"),
    ("shopping", "wishlist.membership"): RecordRule(
        ("wishlist_member",), "truthy",
    ),
    ("team_chat", "message.threaded"): RecordRule(("thread_id",), "present"),
}


def _field_value(record: dict[str, Any], fields: tuple[str, ...]) -> tuple[bool, Any]:
    for field in fields:
        if field in record:
            return True, record[field]
    return False, None


def evaluate_record_predicate(
    domain: str,
    predicate: StatePredicate,
    record: dict[str, Any],
) -> bool | None:
    """Return true/false when observed, otherwise unknown."""
    if predicate.slot.endswith(".exists"):
        return bool(record) == bool(predicate.value)
    rule = _RULES.get((domain, predicate.slot))
    if rule is not None:
        known, value = _field_value(record, rule.fields)
        if not known:
            return None
        if rule.operator == "gt":
            computed = float(value) > float(rule.operand)
        elif rule.operator == "in":
            computed = value in rule.operand
        elif rule.operator == "not_in":
            computed = value not in rule.operand
        elif rule.operator == "truthy":
            computed = bool(value)
        elif rule.operator == "falsy":
            computed = not bool(value)
        elif rule.operator == "present":
            computed = value not in (None, "", [], {})
        elif rule.operator == "nonempty":
            computed = bool(value)
        elif rule.operator == "equals":
            computed = value == rule.operand
        elif rule.operator == "any_positive":
            computed = bool(value) and (
                not isinstance(value, (int, float)) or float(value) > 0
            )
        elif rule.operator == "path_protected":
            path = str(record.get("path") or value)
            computed = path == "/protected" or path.startswith("/protected/")
        elif rule.operator == "path_deletable":
            path = str(record.get("path") or value)
            computed = path != "/" and not (
                path == "/protected" or path.startswith("/protected/")
            )
        elif rule.operator == "ownership_change_allowed":
            path = str(record.get("path") or "")
            computed = (
                record.get("owner") != "root"
                and path != "/protected"
                and not path.startswith("/protected/")
            )
        elif rule.operator == "current_user_reviewed":
            computed = any(
                isinstance(review, dict)
                and str(review.get("author") or "") == "current_user"
                for review in (value or [])
            )
        else:
            raise ValueError(f"Unknown record rule operator: {rule.operator}")
        return computed == bool(predicate.value)
    field = predicate.slot.rsplit(".", 1)[-1]
    if field not in record:
        return None
    return record[field] == predicate.value


def record_satisfies_tool_contract(
    domain: str,
    contract: ToolContract,
    entity_type: str,
    record: dict[str, Any],
    *,
    unknown_is_failure: bool = True,
) -> bool:
    predicates = [
        predicate
        for predicate in contract.preconditions
        if predicate.subject.entity_type == entity_type
        and predicate.observed_entity_required
    ]
    for predicate in predicates:
        result = evaluate_record_predicate(domain, predicate, record)
        if result is False or (result is None and unknown_is_failure):
            return False
    for group in contract.precondition_groups:
        relevant = [
            predicate for predicate in group
            if predicate.subject.entity_type == entity_type
            and predicate.observed_entity_required
        ]
        if not relevant:
            continue
        results = [evaluate_record_predicate(domain, item, record) for item in relevant]
        if True not in results and (
            unknown_is_failure or all(result is False for result in results)
        ):
            return False
    return True


def record_state_is_known(
    domain: str,
    contract: ToolContract,
    entity_type: str,
    record: dict[str, Any],
) -> bool:
    relevant = [
        predicate
        for predicate in contract.preconditions
        if predicate.subject.entity_type == entity_type
        and predicate.observed_entity_required
    ]
    relevant.extend(
        predicate
        for group in contract.precondition_groups
        for predicate in group
        if predicate.subject.entity_type == entity_type
        and predicate.observed_entity_required
    )
    return all(
        evaluate_record_predicate(domain, predicate, record) is not None
        for predicate in relevant
    )
