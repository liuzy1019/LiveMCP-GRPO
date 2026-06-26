"""Oracle validation for LLM teacher replay checks."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.live_mcp.executor import LiveMCPExecutor
from src.live_mcp.manager import LiveMCPManager
from src.live_mcp.types import OracleProgram, ToolCall, ToolExecutionResult


@dataclass
class OracleValidationResult:
    valid: bool
    execution_results: list[ToolExecutionResult]
    final_state: dict[str, Any]
    failed_criteria: list[dict[str, Any]]
    error: str = ""


class OracleValidator:
    def validate(
        self,
        task: Any,
        oracle_program: OracleProgram,
        manager: LiveMCPManager,
        executor: LiveMCPExecutor,
        seed: int,
        check_state: bool = True,
    ) -> OracleValidationResult:
        session = manager.create_session(seed=seed)
        manager.discover_tools(session.session_id)
        results: list[ToolExecutionResult] = []
        try:
            for idx, call in enumerate(oracle_program.calls):
                result = executor.execute(
                    session.session_id,
                    ToolCall(call.tool_name, dict(call.arguments), call_id=f"oracle_{idx}"),
                )
                results.append(result)
                if not result.success:
                    return OracleValidationResult(
                        False, results, manager.get_state(session.session_id),
                        oracle_program.success_criteria, result.error_message,
                    )
            final_state = manager.get_state(session.session_id)
            if not check_state:
                return OracleValidationResult(True, results, final_state, [])
            failed = [
                criterion for criterion in oracle_program.success_criteria
                if not criterion_satisfied(final_state, criterion)
            ]
            return OracleValidationResult(not failed, results, final_state, failed)
        finally:
            manager.close_session(session.session_id)


def criterion_satisfied(final_state: dict[str, Any], criterion: dict[str, Any]) -> bool:
    kind = criterion.get("type")
    server = criterion.get("server")
    state = final_state.get(server, {})
    if kind == "state_equals":
        actual = _get_path(state, str(criterion["path"]))
        expected = criterion.get("value")
        op = criterion.get("op", "eq")
        if op == "gt":
            return isinstance(actual, (int, float)) and isinstance(expected, (int, float)) and actual > expected
        if op == "lt":
            return isinstance(actual, (int, float)) and isinstance(expected, (int, float)) and actual < expected
        if op == "gte":
            return isinstance(actual, (int, float)) and isinstance(expected, (int, float)) and actual >= expected
        if op == "lte":
            return isinstance(actual, (int, float)) and isinstance(expected, (int, float)) and actual <= expected
        if op == "neq":
            return actual != expected
        return actual == expected
    if kind == "cart_empty":
        return state.get("cart") == []
    if kind == "cart_not_empty":
        return len(state.get("cart", [])) > 0
    if kind == "state_exists":
        path = criterion.get("path", "")
        return _get_path(state, path) is not None
    if kind == "email_count_gte":
        count = len(state.get("emails", {}))
        expected = criterion.get("value", 0)
        return count >= expected
    if kind == "order_contains_product":
        product_id = criterion.get("product_id")
        for order in state.get("orders", {}).values():
            if any(item.get("product_id") == product_id for item in order.get("items", [])):
                return True
        return False
    if kind == "missing_function":
        return True
    if kind == "transaction_exists":
        transactions = state.get("transactions", [])
        return len(transactions) > 0
    if kind == "label_added":
        email_id = criterion.get("email_id")
        label = criterion.get("label")
        email = state.get("emails", {}).get(email_id, {})
        return label in email.get("labels", [])
    if kind == "cwd_equals":
        return state.get("cwd", "") == criterion.get("path", "")
    if kind == "file_exists":
        path = criterion.get("path", "")
        return path in state.get("fs", {})
    if kind == "deal_exists_for_lead":
        lead_id = criterion.get("lead_id")
        for deal in state.get("deals", {}).values():
            if deal.get("lead_id") == lead_id:
                return True
        return False
    if kind == "message_sent":
        channel_id = criterion.get("channel_id")
        channel = state.get("channels", {}).get(channel_id, {})
        messages = channel.get("messages", [])
        return len(messages) > 0
    if kind == "order_exists":
        status_filter = criterion.get("status")
        for order in state.get("orders", {}).values():
            if status_filter is None or order.get("status") == status_filter:
                return True
        return False
    return False


def _get_path(data: dict[str, Any], path: str) -> Any:
    value: Any = data
    for part in path.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value
