"""Shopping, payments, and food-delivery reward adapters."""

from __future__ import annotations

from typing import Any

from .base import DomainAdapter


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
    entity_container_key = "orders"

    TOOL_MAP = {
        "search_products": ("query", "product"),
        "get_product": ("query", "product"),
        "list_categories": ("query", "product_category"),
        "compare_products": ("query", "product"),
        "get_recommendations": ("query", "product"),
        "add_to_cart": ("update", "shopping_cart"),
        "update_cart_quantity": ("update", "shopping_cart"),
        "remove_from_cart": ("update", "shopping_cart"),
        "get_cart": ("query", "shopping_cart"),
        "clear_cart": ("update", "shopping_cart"),
        "apply_coupon": ("update", "shopping_cart"),
        "get_coupons": ("query", "coupon"),
        "checkout": ("create", "shopping_order"),
        "get_order": ("query", "shopping_order"),
        "list_orders": ("query", "shopping_order"),
        "cancel_order": ("update", "shopping_order"),
        "return_order": ("create", "shopping_return"),
        "get_return_status": ("query", "shopping_return"),
        "add_review": ("create", "product_review"),
        "get_reviews": ("query", "product_review"),
        "add_to_wishlist": ("update", "shopping_wishlist"),
        "remove_from_wishlist": ("update", "shopping_wishlist"),
        "get_wishlist": ("query", "shopping_wishlist"),
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

        op, target = self.tool_semantics(tool_name, "shopping_resource", state_changed)
        result["operation"] = op
        result["target_type"] = target
        result["target_id"] = self.generic_target_id(
            tool_arguments, observation,
        )

        if tool_name == "checkout":
            if execution_success and isinstance(observation, dict):
                order = observation.get("order", observation.get("observation", {}))
                if isinstance(order, dict):
                    result["target_id"] = order.get("order_id", "")
            if before_state and after_state:
                be = self._unwrap_domain_state(before_state, "shopping")
                ae = self._unwrap_domain_state(after_state, "shopping")
                if be is not None and ae is not None:
                    before_orders = set(be.get("orders", {}).keys())
                    after_orders = set(ae.get("orders", {}).keys())
                    result["created_ids"] = list(after_orders - before_orders)
        elif tool_name == "return_order" and execution_success:
            if isinstance(observation, dict):
                return_record = observation.get("return", {})
                if isinstance(return_record, dict):
                    return_id = str(return_record.get("return_id") or "")
                    if return_id:
                        result["created_ids"] = [return_id]
        elif tool_name == "add_review" and execution_success:
            if isinstance(observation, dict):
                review = observation.get("review", {})
                if isinstance(review, dict):
                    review_id = str(review.get("review_id") or "")
                    if review_id:
                        result["created_ids"] = [review_id]

        return result




    def protected_resources(self, task: dict[str, Any]) -> list[str]:
        return task.get("protected_product_ids", [])

    def budget(self, task: dict[str, Any]) -> int:
        return task.get("budget", 4)

    def identity_policy(self, task: dict[str, Any]) -> str:
        return task.get("identity_policy", "create_new")


class PaymentsAdapter(DomainAdapter):
    """Domain adapter for payments MCP server.

    Payments state: transactional invoices/payments/refunds.
    target_type: "invoice" / "payment" / "refund"
    identity_policy: "verify" — sensitive params require provenance
    """

    domain_name = "payments"
    entity_container_key = "invoices"

    TOOL_MAP = {
        "create_invoice": ("create", "invoice"),
        "pay_invoice": ("create", "payment"),
        "refund_invoice": ("create", "refund"),
        "get_invoice": ("query", "invoice"),
        "list_invoices": ("query", "invoice"),
        "create_webhook": ("create", "webhook"),
    }

    def normalize_event(
        self, action_type, tool_name, tool_arguments, observation,
        execution_success, state_changed, before_state, after_state,
    ) -> dict[str, Any]:
        if action_type != "tool_call":
            return {
                "operation": "terminal", "target_type": "", "target_id": "",
                "changed_fields": [], "created_ids": [], "deleted_ids": [],
                "identity_violation": "", "forbidden_transition": "",
                "duplicate_of": None,
            }
        op, ttype = self.tool_semantics(tool_name, "invoice", state_changed)
        result: dict[str, Any] = {
            "operation": op, "target_type": ttype, "target_id": "",
            "changed_fields": [], "created_ids": [], "deleted_ids": [],
            "identity_violation": "", "forbidden_transition": "", "duplicate_of": None,
        }
        if tool_name == "pay_invoice":
            result["target_id"] = tool_arguments.get("invoice_id", "")
            if execution_success and isinstance(observation, dict):
                payment = observation.get("payment", observation)
                result["created_ids"] = [payment.get("payment_id", "")]
            result["changed_fields"] = ["status", "payment_id"]
            # Detect double payment
            if not execution_success:
                error_msg = observation.get("error_message", "") if isinstance(observation, dict) else ""
                if "already paid" in str(error_msg):
                    result["forbidden_transition"] = "double_payment"
        elif tool_name == "refund_invoice":
            result["target_id"] = tool_arguments.get("invoice_id", "")
            if execution_success and isinstance(observation, dict):
                refund = observation.get("refund", observation)
                if isinstance(refund, dict):
                    result["created_ids"] = [refund.get("refund_id", "")]
            result["changed_fields"] = ["status", "refund_id"]
        elif tool_name == "create_invoice":
            if execution_success and isinstance(observation, dict):
                invoice = observation.get("invoice", observation)
                if isinstance(invoice, dict):
                    result["target_id"] = invoice.get("invoice_id", "")
        elif tool_name == "get_invoice":
            result["target_id"] = tool_arguments.get("invoice_id", "")
        elif tool_name == "create_webhook":
            result["target_id"] = tool_arguments.get("url", "")
        return result

    def protected_resources(self, task): return task.get("protected_invoice_ids", [])
    def budget(self, task): return task.get("budget", 5)
    def identity_policy(self, task): return task.get("identity_policy", "verify")


class FoodDeliveryAdapter(DomainAdapter):
    """Domain adapter for food delivery MCP server.

    Food delivery state: order lifecycle.
    target_type: "order" / "restaurant"
    identity_policy: "create_new" — each order has new ID
    """

    domain_name = "food_delivery"
    entity_container_key = "orders"

    TOOL_MAP = {
        "list_restaurants": ("query", "restaurant"),
        "get_menu": ("query", "restaurant"),
        "create_order": ("create", "order"),
        "get_order": ("query", "order"),
        "update_order_status": ("update", "order"),
        "cancel_order": ("update", "order"),
        "list_orders": ("query", "order"),
    }

    def normalize_event(
        self, action_type, tool_name, tool_arguments, observation,
        execution_success, state_changed, before_state, after_state,
    ) -> dict[str, Any]:
        if action_type != "tool_call":
            return {
                "operation": "terminal", "target_type": "", "target_id": "",
                "changed_fields": [], "created_ids": [], "deleted_ids": [],
                "identity_violation": "", "forbidden_transition": "",
                "duplicate_of": None,
            }
        op, ttype = self.tool_semantics(tool_name, "order", state_changed)
        result: dict[str, Any] = {
            "operation": op, "target_type": ttype, "target_id": "",
            "changed_fields": [], "created_ids": [], "deleted_ids": [],
            "identity_violation": "", "forbidden_transition": "", "duplicate_of": None,
        }
        if tool_name == "create_order":
            if execution_success and isinstance(observation, dict):
                order = observation.get("order", observation)
                if isinstance(order, dict):
                    result["target_id"] = order.get("order_id", "")
                    result["created_ids"] = [result["target_id"]]
        elif tool_name == "update_order_status":
            result["target_id"] = tool_arguments.get("order_id", "")
            result["changed_fields"] = ["status"]
            # Detect skipping lifecycle stages
            if not execution_success:
                error_msg = observation.get("error_message", "") if isinstance(observation, dict) else ""
                if "invalid transition" in str(error_msg).lower():
                    result["forbidden_transition"] = "lifecycle_stage_skip"
        elif tool_name == "cancel_order":
            result["target_id"] = tool_arguments.get("order_id", "")
            result["changed_fields"] = ["status", "cancel_reason"]
            if not execution_success:
                error_msg = observation.get("error_message", "") if isinstance(observation, dict) else ""
                if "cannot cancel" in str(error_msg).lower():
                    result["forbidden_transition"] = "cancel_after_preparing"
        elif tool_name == "get_menu":
            result["target_id"] = tool_arguments.get("restaurant_id", "")
        elif tool_name == "get_order":
            result["target_id"] = tool_arguments.get("order_id", "")
        return result

    def protected_resources(self, task): return task.get("protected_order_ids", [])
    def budget(self, task): return task.get("budget", 5)
    def identity_policy(self, task): return task.get("identity_policy", "create_new")


__all__ = ["ShoppingAdapter", "PaymentsAdapter", "FoodDeliveryAdapter"]
