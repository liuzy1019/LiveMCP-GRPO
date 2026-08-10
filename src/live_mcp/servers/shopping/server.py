"""Stateful shopping server with 23 tools.
Commerce: catalog, cart, checkout, orders, reviews, wishlist, coupons, returns, tracking.
Safety: stock consistency, empty cart checkout.
"""

from __future__ import annotations
from typing import Any
from src.live_mcp.server_base import StatefulToolServer, _result, serve

TOOLS = [
    {"name": "search_products", "description": "Search product summaries by query/category/price range. Results identify matches and show catalog descriptors, but exact stock is available only from get_product.", "input_schema": {"type": "object", "properties": {"query": {"type": "string"}, "category": {"type": "string"}, "min_price": {"type": "number"}, "max_price": {"type": "number"}, "in_stock_only": {"type": "boolean"}}, "required": []}, "annotations": {"readonly": True, "mutating": False}},
    {"name": "get_product", "description": "Get product details by id.", "input_schema": {"type": "object", "properties": {"product_id": {"type": "string"}}, "required": ["product_id"]}, "annotations": {"readonly": True, "mutating": False}},
    {"name": "list_categories", "description": "List product categories with counts.", "input_schema": {"type": "object", "properties": {}, "required": []}, "annotations": {"readonly": True, "mutating": False}},
    {"name": "compare_products", "description": "Compare at least two existing products side-by-side.", "input_schema": {"type": "object", "properties": {"product_ids": {"type": "array", "items": {"type": "string"}, "minItems": 2, "uniqueItems": True, "description": "Two or more distinct existing product IDs."}}, "required": ["product_ids"]}, "annotations": {"readonly": True, "mutating": False}},
    {"name": "get_recommendations", "description": "Get personalized product recommendation summaries. The seed product itself is excluded; use get_product when exact stock or full detail is requested.", "input_schema": {"type": "object", "properties": {"based_on_product": {"type": "string", "description": "Optional existing product ID used as the recommendation seed; the seed itself is not returned."}, "category": {"type": "string", "description": "Optional exact result category. When based_on_product is also provided, this category remains the result filter instead of being replaced by the seed product's category."}, "limit": {"type": "integer", "minimum": 1, "maximum": 20, "description": "Number of recommendations to return, from 1 through 20."}}, "required": []}, "annotations": {"readonly": True, "mutating": False}},
    {"name": "add_to_cart", "description": "Add product to cart.", "input_schema": {"type": "object", "properties": {"product_id": {"type": "string"}, "quantity": {"type": "integer", "minimum": 1, "description": "Positive item quantity."}}, "required": ["product_id", "quantity"]}, "annotations": {"mutating": True}},
    {"name": "update_cart_quantity", "description": "Update quantity of an item in cart.", "input_schema": {"type": "object", "properties": {"product_id": {"type": "string"}, "quantity": {"type": "integer", "minimum": 1, "description": "Positive replacement quantity; use remove_from_cart to remove an item."}}, "required": ["product_id", "quantity"]}, "annotations": {"mutating": True}},
    {"name": "remove_from_cart", "description": "Remove a product from cart.", "input_schema": {"type": "object", "properties": {"product_id": {"type": "string"}}, "required": ["product_id"]}, "annotations": {"mutating": True}},
    {"name": "get_cart", "description": "View current cart contents and total.", "input_schema": {"type": "object", "properties": {}, "required": []}, "annotations": {"readonly": True, "mutating": False}},
    {"name": "clear_cart", "description": "Remove all items from cart.", "input_schema": {"type": "object", "properties": {}, "required": []}, "annotations": {"mutating": True}},
    {"name": "apply_coupon", "description": "Apply a coupon code to cart. This can be done before adding items and is optional for checkout.", "input_schema": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}, "annotations": {"mutating": True}},
    {"name": "get_coupons", "description": "Get available coupons.", "input_schema": {"type": "object", "properties": {}, "required": []}, "annotations": {"readonly": True, "mutating": False}},
    {"name": "checkout", "description": "Checkout a non-empty cart, create a placed order, and empty the cart using the user's shipping address and payment method. Returns only the new order ID/status summary; use get_order for full line-item and settlement details. A coupon is optional and is not a checkout prerequisite.", "input_schema": {"type": "object", "properties": {"shipping_address": {"type": "string", "minLength": 1, "description": "User-provided destination address."}, "payment_method": {"type": "string", "minLength": 1, "description": "Concrete user-provided payment instrument, such as Visa, Mastercard, PayPal, Apple Pay, Google Pay, or a card identified by its last digits; generic credit card or debit card wording is unresolved."}}, "required": ["shipping_address", "payment_method"]}, "annotations": {"mutating": True}},
    {"name": "get_order", "description": "Get order details.", "input_schema": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]}, "annotations": {"readonly": True, "mutating": False}},
    {"name": "list_orders", "description": "List past order summaries without item details. Use get_order with a returned order_id when product or line-item details are needed. Omit status to return all orders; when provided, status is an exact filter.", "input_schema": {"type": "object", "properties": {"status": {"type": "string", "enum": ["pending", "shipped", "placed", "returning", "returned"], "description": "Optional exact order status: pending, shipped, placed, returning, or returned. Omit this field to list all orders; all and completed are not valid statuses."}}, "required": []}, "annotations": {"readonly": True, "mutating": False}},
    {"name": "cancel_order", "description": "Cancel a placed or pending order. Orders in shipped, returning, or returned status cannot be cancelled.", "input_schema": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]}, "annotations": {"mutating": True}},
    {"name": "return_order", "description": "Initiate a return for a shipped order. Orders in pending, placed, returning, or returned status cannot start a return.", "input_schema": {"type": "object", "properties": {"order_id": {"type": "string"}, "reason": {"type": "string", "minLength": 1}, "items": {"type": "array", "items": {"type": "string"}, "minItems": 1, "uniqueItems": True, "description": "Optional non-empty subset of product IDs present in the order; omit to return the whole order."}}, "required": ["order_id", "reason"]}, "annotations": {"mutating": True}},
    {"name": "get_return_status", "description": "Check an existing return whose order has already entered the return lifecycle.", "input_schema": {"type": "object", "properties": {"return_id": {"type": "string"}}, "required": ["return_id"]}, "annotations": {"readonly": True, "mutating": False}},
    {"name": "add_review", "description": "Add one review for a product in the current user's order history after it reached the shipped or return lifecycle. A newly searched, recommended, wishlisted, carted, pending, or newly placed product is not review-eligible by itself, and the current user cannot review the same product twice. The body must be non-empty.", "input_schema": {"type": "object", "properties": {"product_id": {"type": "string"}, "rating": {"type": "integer", "minimum": 1, "maximum": 5}, "title": {"type": "string"}, "body": {"type": "string", "minLength": 1}}, "required": ["product_id", "rating", "body"]}, "annotations": {"mutating": True}},
    {"name": "get_reviews", "description": "Get reviews for a product. Omit sort_by for stored order, or use rating for highest rating first.", "input_schema": {"type": "object", "properties": {"product_id": {"type": "string"}, "sort_by": {"type": "string", "enum": ["rating"], "description": "Optional sorting mode; rating sorts highest rating first."}}, "required": ["product_id"]}, "annotations": {"readonly": True, "mutating": False}},
    {"name": "add_to_wishlist", "description": "Add product to wishlist.", "input_schema": {"type": "object", "properties": {"product_id": {"type": "string"}}, "required": ["product_id"]}, "annotations": {"mutating": True}},
    {"name": "remove_from_wishlist", "description": "Remove product from wishlist.", "input_schema": {"type": "object", "properties": {"product_id": {"type": "string"}}, "required": ["product_id"]}, "annotations": {"mutating": True}},
    {"name": "get_wishlist", "description": "View wishlist product summaries. Use get_product when exact stock or full detail is requested.", "input_schema": {"type": "object", "properties": {}, "required": []}, "annotations": {"readonly": True, "mutating": False}},
]

COUPONS = {"SAVE10": 0.10, "WELCOME20": 0.20, "FREESHIP": None}
SHIPPING_FEE = 7.99
UNRESOLVED_SHIPPING_ADDRESSES = frozenset({
    "address on file",
    "default address",
    "home address",
    "my address",
    "my address on file",
    "my default address",
    "my home address",
    "my saved address",
    "saved address",
    "saved info",
    "saved shipping info",
})
UNRESOLVED_PAYMENT_METHODS = frozenset({
    "card",
    "card on file",
    "default card",
    "default payment method",
    "credit card",
    "debit card",
    "my card",
    "my card on file",
    "my default card",
    "my payment method",
    "my saved card",
    "payment method on file",
    "saved card",
    "saved payment method",
    "usual card",
    "usual payment method",
})
UNRESOLVED_RETURN_REASONS = frozenset({
    "n/a",
    "na",
    "no reason",
    "none",
    "not specified",
    "requested return",
    "return",
    "return requested",
    "unknown",
    "unspecified",
    "user requested return",
})


def _product_summary(product: dict[str, Any]) -> dict[str, Any]:
    """Project a discovery result without duplicating exact product detail."""
    return {
        key: product[key]
        for key in ("product_id", "name", "category", "price", "description")
        if key in product
    }


class ShoppingServer(StatefulToolServer):
    def __init__(self) -> None:
        super().__init__("shopping", TOOLS)
        self.handlers = {t["name"]: getattr(self, t["name"]) for t in TOOLS}

    def search_products(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id); products = list(state["products"].values())
        q = arguments.get("query", "").lower(); cat = arguments.get("category"); mn = arguments.get("min_price"); mx = arguments.get("max_price")
        if mn is not None and mx is not None and float(mn) > float(mx):
            raise KeyError("min_price must be less than or equal to max_price")
        if q: products = [p for p in products if q in p["name"].lower() or q in p.get("description", "").lower()]
        if cat: products = [p for p in products if p.get("category") == cat]
        if mn is not None: products = [p for p in products if p["price"] >= float(mn)]
        if mx is not None: products = [p for p in products if p["price"] <= float(mx)]
        if arguments.get("in_stock_only"): products = [p for p in products if p["stock"] > 0]
        summaries = [_product_summary(product) for product in products]
        return _result(
            True, {"products": summaries, "count": len(summaries)},
            None, "", False,
        )

    def get_product(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        p = self._state(session_id)["products"].get(arguments["product_id"])
        if not p: raise KeyError(f"product not found: {arguments['product_id']}")
        return _result(True, {"product": p}, None, "", False)

    def list_categories(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id); cats = {}
        for p in state["products"].values(): cats[p.get("category", "uncategorized")] = cats.get(p.get("category", "uncategorized"), 0) + 1
        return _result(True, {"categories": [{"name": k, "count": v} for k, v in cats.items()]}, None, "", False)

    def compare_products(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id); ids = arguments["product_ids"]
        if len(ids) < 2 or len(set(ids)) != len(ids):
            raise KeyError("compare_products requires at least two distinct product IDs")
        missing = [pid for pid in ids if pid not in state["products"]]
        if missing:
            raise KeyError(f"product not found: {missing[0]}")
        products = [state["products"][pid] for pid in ids]
        return _result(True, {"products": products, "count": len(products)}, None, "", False)

    def get_recommendations(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id); limit = int(arguments.get("limit", 5))
        if not 1 <= limit <= 20:
            raise KeyError("limit must be between 1 and 20")
        base = arguments.get("based_on_product"); cat = arguments.get("category")
        products = list(state["products"].values())
        if base:
            if base not in state["products"]:
                raise KeyError(f"product not found: {base}")
            if not cat:
                cat = state["products"][base].get("category")
        if cat: products = [p for p in products if p.get("category") == cat]
        if base:
            products = [
                product
                for product in products
                if product.get("product_id") != base
            ]
        recommendations = [
            _product_summary(product) for product in products[:limit]
        ]
        return _result(
            True, {"recommendations": recommendations}, None, "", False,
        )

    def add_to_cart(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id); pid, qty = arguments["product_id"], int(arguments["quantity"])
        p = state["products"].get(pid)
        if not p: raise KeyError(f"product not found: {pid}")
        if qty <= 0: raise KeyError("quantity must be positive")
        if p["stock"] < qty: raise KeyError(f"insufficient stock: {pid} (have {p['stock']})")
        p["stock"] -= qty
        existing = next((item for item in state["cart"] if item["product_id"] == pid), None)
        if existing: existing["quantity"] += qty
        else: state["cart"].append({"product_id": pid, "quantity": qty, "unit_price": p["price"]})
        return _result(True, {"cart": list(state["cart"]), "total": sum(item["quantity"] * item["unit_price"] for item in state["cart"])}, None, "", True)

    def update_cart_quantity(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id); pid, qty = arguments["product_id"], int(arguments["quantity"])
        item = next((i for i in state["cart"] if i["product_id"] == pid), None)
        if not item: raise KeyError(f"product not in cart: {pid}")
        if qty <= 0: raise KeyError("quantity must be positive")
        diff = qty - item["quantity"]; p = state["products"][pid]
        if diff > 0 and p["stock"] < diff: raise KeyError(f"insufficient stock: {pid}")
        p["stock"] -= diff; item["quantity"] = qty
        return _result(True, {"cart": list(state["cart"]), "total": sum(i["quantity"] * i["unit_price"] for i in state["cart"])}, None, "", diff != 0)

    def remove_from_cart(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id); pid = arguments["product_id"]
        kept = []; removed = None
        for item in state["cart"]:
            if item["product_id"] == pid:
                removed = item
                if pid in state["products"]:
                    state["products"][pid]["stock"] += item["quantity"]
            else: kept.append(item)
        if removed is None:
            raise KeyError(f"product not in cart: {pid}")
        state["cart"] = kept
        return _result(True, {"removed": removed, "cart": list(state["cart"])}, None, "", True)

    def get_cart(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id)
        total = sum(i["quantity"] * i["unit_price"] for i in state["cart"])
        coupon = state.get("applied_coupon"); discount = 0.0
        if coupon and coupon in COUPONS and COUPONS[coupon]: discount = total * COUPONS[coupon]
        shipping = 0.0 if coupon == "FREESHIP" or not state["cart"] else SHIPPING_FEE
        return _result(True, {"cart": list(state["cart"]), "total": total, "discount": discount, "shipping": shipping, "final_total": round(total - discount + shipping, 2), "item_count": len(state["cart"])}, None, "", False)

    def clear_cart(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id)
        changed = bool(state["cart"]) or "applied_coupon" in state
        for item in state["cart"]:
            if item["product_id"] in state["products"]:
                state["products"][item["product_id"]]["stock"] += item["quantity"]
        state["cart"] = []; state.pop("applied_coupon", None)
        return _result(True, {"cart": [], "message": "cart cleared"}, None, "", changed)

    def apply_coupon(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id); code = arguments["code"].upper()
        if code not in COUPONS: raise KeyError(f"invalid coupon: {code}")
        old = state.get("applied_coupon")
        state["applied_coupon"] = code
        discount = (
            f"{COUPONS[code] * 100}%"
            if COUPONS[code] is not None
            else "free shipping"
        )
        return _result(True, {"coupon": code, "discount": discount}, None, "", old != code)

    def get_coupons(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        return _result(True, {"coupons": [{"code": k, "discount": f"{v*100}%" if v else "free shipping"} for k, v in COUPONS.items()]}, None, "", False)

    def checkout(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id)
        if not state["cart"]: raise KeyError("cart is empty")
        shipping_address = " ".join(
            str(arguments["shipping_address"]).strip().lower().split()
        )
        if not shipping_address:
            raise KeyError("shipping_address must be non-empty")
        if shipping_address in UNRESOLVED_SHIPPING_ADDRESSES:
            raise KeyError(
                "shipping_address must be a concrete user-provided destination, "
                "not an unresolved placeholder"
            )
        payment_method = " ".join(
            str(arguments["payment_method"]).strip().lower().split()
        )
        if not payment_method:
            raise KeyError("payment_method must be non-empty")
        if payment_method in UNRESOLVED_PAYMENT_METHODS:
            raise KeyError(
                "payment_method must be a concrete user-provided method, "
                "not an unresolved placeholder"
            )
        if any(int(i.get("quantity", 0)) <= 0 for i in state["cart"]):
            raise KeyError("cart contains non-positive quantity")
        total = sum(i["quantity"] * i["unit_price"] for i in state["cart"])
        coupon = state.get("applied_coupon")
        if coupon and coupon in COUPONS and COUPONS[coupon]: total *= (1 - COUPONS[coupon])
        shipping = 0.0 if coupon == "FREESHIP" else SHIPPING_FEE
        total += shipping
        oid = (
            f"ord_{state['id_scope']}_{state['next_order_num']:04d}"
        )
        state["next_order_num"] += 1
        order = {"order_id": oid, "items": list(state["cart"]), "total": round(total, 2), "shipping": shipping, "shipping_address": arguments["shipping_address"], "payment_method": arguments["payment_method"], "status": "placed", "tracking": [], "created_at": state["current_date"]}
        state["orders"][oid] = order; state["cart"] = []; state.pop("applied_coupon", None)
        return _result(
            True,
            {
                "order": {
                    "order_id": order["order_id"],
                    "status": order["status"],
                },
            },
            None,
            "",
            True,
        )

    def get_order(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        o = self._state(session_id)["orders"].get(arguments["order_id"])
        if not o: raise KeyError(f"order not found: {arguments['order_id']}")
        return _result(True, {"order": o}, None, "", False)

    def list_orders(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        orders = list(self._state(session_id)["orders"].values())
        if arguments.get("status"): orders = [o for o in orders if o.get("status") == arguments["status"]]
        summaries = [
            {
                "order_id": order["order_id"],
                "status": order.get("status"),
                "total": order.get("total"),
                "created_at": order.get("created_at"),
                "return_id": order.get("return_id"),
                "item_count": sum(
                    int(item.get("quantity", 0))
                    for item in order.get("items", [])
                ),
            }
            for order in orders
        ]
        return _result(
            True, {"orders": summaries, "count": len(summaries)}, None, "", False,
        )

    def cancel_order(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id)
        o = state["orders"].get(arguments["order_id"])
        if not o:
            raise KeyError(f"order not found: {arguments['order_id']}")
        if o.get("status") not in ("placed", "pending"):
            raise KeyError(
                f"only placed or pending orders can be cancelled, "
                f"current status: {o.get('status')}"
            )
        # Return stock for each item in the order
        for item in o.get("items", []):
            pid = item.get("product_id")
            qty = int(item.get("quantity", 0))
            if pid and pid in state["products"] and qty > 0:
                state["products"][pid]["stock"] += qty
        o["status"] = "cancelled"
        return _result(
            True,
            {"order": o, "message": f"Order {arguments['order_id']} cancelled"},
            None,
            "",
            True,
        )

    def return_order(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id); o = state["orders"].get(arguments["order_id"])
        if not o: raise KeyError(f"order not found: {arguments['order_id']}")
        if o.get("status") != "shipped":
            raise KeyError("only shipped orders can be returned")
        normalized_reason = " ".join(
            str(arguments["reason"]).strip().lower().split()
        )
        if not normalized_reason:
            raise KeyError("reason must be non-empty")
        if normalized_reason in UNRESOLVED_RETURN_REASONS:
            raise KeyError(
                "reason must be a concrete user-provided reason, not a "
                "generic return placeholder"
            )
        order_product_ids = [str(item["product_id"]) for item in o.get("items", [])]
        requested_items = arguments.get("items")
        if requested_items is None:
            returned_items = order_product_ids
        else:
            returned_items = [str(product_id) for product_id in requested_items]
            if not returned_items or len(set(returned_items)) != len(returned_items):
                raise KeyError("items must contain distinct product IDs from the order")
            unknown_items = [
                product_id
                for product_id in returned_items
                if product_id not in order_product_ids
            ]
            if unknown_items:
                raise KeyError(f"product not in order: {unknown_items[0]}")
        returned_item_ids = set(returned_items)
        returned_item_details = [
            dict(item)
            for item in o.get("items", [])
            if str(item.get("product_id")) in returned_item_ids
        ]
        rid = (
            f"ret_{state['id_scope']}_{state['next_order_num']:04d}"
        )
        state["next_order_num"] += 1
        ret = {
            "return_id": rid,
            "order_id": o["order_id"],
            "reason": arguments["reason"],
            "items": returned_items,
            "item_details": returned_item_details,
            "status": "initiated",
        }
        state.setdefault("returns", {})[rid] = ret; o["status"] = "returning"
        return _result(True, {"return": ret, "order_status": "returning"}, None, "", True)

    def get_return_status(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        ret = self._state(session_id).get("returns", {}).get(arguments["return_id"])
        if not ret: raise KeyError(f"return not found: {arguments['return_id']}")
        status = str(ret.get("status") or "unknown")
        if status == "initiated":
            status_description = (
                "The return request has been recorded and its current stored "
                "status is initiated. This environment does not expose a "
                "shipping label, carrier handoff, or later processing steps."
            )
        else:
            status_description = (
                f"The current stored return status is {status}. This environment "
                "does not expose additional logistics or later processing steps."
            )
        return _result(
            True,
            {
                "return": ret,
                "status_description": status_description,
            },
            None,
            "",
            False,
        )

    def add_review(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id); pid = arguments["product_id"]; rating = int(arguments["rating"])
        if pid not in state["products"]: raise KeyError(f"product not found: {pid}")
        if not 1 <= rating <= 5: raise KeyError("rating must be 1-5")
        if not str(arguments["body"]).strip():
            raise KeyError("review body must be non-empty")
        review_eligible = any(
            str(order.get("status") or "") in {
                "shipped", "returning", "returned",
            }
            and any(
                str(item.get("product_id") or "") == pid
                for item in (order.get("items") or [])
                if isinstance(item, dict)
            )
            for order in state.get("orders", {}).values()
            if isinstance(order, dict)
        )
        if not review_eligible:
            raise KeyError(
                "product is not in the current user's shipped/return order history"
            )
        if any(
            str(review.get("author") or "") == "current_user"
            for review in state.get("reviews", {}).get(pid, [])
            if isinstance(review, dict)
        ):
            raise KeyError("current user has already reviewed this product")
        rid = (
            f"rev_{state['id_scope']}_{state['next_order_num']:04d}"
        )
        state["next_order_num"] += 1
        review = {"review_id": rid, "product_id": pid, "rating": rating, "title": arguments.get("title", ""), "body": arguments["body"], "author": "current_user", "date": state["current_date"]}
        state.setdefault("reviews", {}).setdefault(pid, []).append(review)
        return _result(True, {"review": review}, None, "", True)

    def get_reviews(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id); pid = arguments["product_id"]
        if pid not in state["products"]: raise KeyError(f"product not found: {pid}")
        reviews = state.get("reviews", {}).get(pid, [])
        sort_by = arguments.get("sort_by")
        if sort_by not in (None, "rating"):
            raise KeyError(f"unsupported review sort: {sort_by}")
        if sort_by == "rating":
            reviews = sorted(reviews, key=lambda r: r["rating"], reverse=True)
        avg = round(sum(r["rating"] for r in reviews) / len(reviews), 1) if reviews else 0
        return _result(True, {"product_id": pid, "reviews": reviews, "average_rating": avg, "count": len(reviews)}, None, "", False)

    def add_to_wishlist(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id); pid = arguments["product_id"]
        if pid not in state["products"]: raise KeyError(f"product not found: {pid}")
        wl = state.setdefault("wishlist", [])
        if pid in wl:
            return _result(True, {"wishlist": wl, "count": len(wl)}, None, "", False)
        wl.append(pid)
        return _result(True, {"wishlist": wl, "count": len(wl)}, None, "", True)

    def remove_from_wishlist(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id); pid = arguments["product_id"]
        if pid not in state["products"]:
            raise KeyError(f"product not found: {pid}")
        wl = state.setdefault("wishlist", [])
        if pid not in wl:
            return _result(True, {"wishlist": wl, "count": len(wl)}, None, "", False)
        wl.remove(pid)
        return _result(True, {"wishlist": wl, "count": len(wl)}, None, "", True)

    def get_wishlist(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id); wl = state.get("wishlist", [])
        products = [
            _product_summary(state["products"][pid])
            for pid in wl
            if pid in state["products"]
        ]
        return _result(True, {"wishlist": products, "count": len(products)}, None, "", False)


if __name__ == "__main__":
    serve(ShoppingServer())
