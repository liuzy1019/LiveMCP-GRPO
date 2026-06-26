"""Stateful shopping server — 23 tools (PROVE-aligned).
Commerce: catalog, cart, checkout, orders, reviews, wishlist, coupons, returns, tracking.
Safety: stock consistency, empty cart checkout.
"""

from __future__ import annotations
from typing import Any
from src.live_mcp.server_base import StatefulToolServer, _result, serve

TOOLS = [
    {"name": "search_products", "description": "Search products by query/category/price range.", "input_schema": {"type": "object", "properties": {"query": {"type": "string"}, "category": {"type": "string"}, "min_price": {"type": "number"}, "max_price": {"type": "number"}, "in_stock_only": {"type": "boolean"}}, "required": []}, "annotations": {"readonly": True, "mutating": False}},
    {"name": "get_product", "description": "Get product details by id.", "input_schema": {"type": "object", "properties": {"product_id": {"type": "string"}}, "required": ["product_id"]}, "annotations": {"readonly": True, "mutating": False}},
    {"name": "list_categories", "description": "List product categories with counts.", "input_schema": {"type": "object", "properties": {}, "required": []}, "annotations": {"readonly": True, "mutating": False}},
    {"name": "compare_products", "description": "Compare products side-by-side.", "input_schema": {"type": "object", "properties": {"product_ids": {"type": "array"}}, "required": ["product_ids"]}, "annotations": {"readonly": True, "mutating": False}},
    {"name": "get_recommendations", "description": "Get personalized product recommendations.", "input_schema": {"type": "object", "properties": {"based_on_product": {"type": "string"}, "category": {"type": "string"}, "limit": {"type": "integer"}}, "required": []}, "annotations": {"readonly": True, "mutating": False}},
    {"name": "add_to_cart", "description": "Add product to cart.", "input_schema": {"type": "object", "properties": {"product_id": {"type": "string"}, "quantity": {"type": "integer"}}, "required": ["product_id", "quantity"]}, "annotations": {"mutating": True}},
    {"name": "update_cart_quantity", "description": "Update quantity of an item in cart.", "input_schema": {"type": "object", "properties": {"product_id": {"type": "string"}, "quantity": {"type": "integer"}}, "required": ["product_id", "quantity"]}, "annotations": {"mutating": True}},
    {"name": "remove_from_cart", "description": "Remove a product from cart.", "input_schema": {"type": "object", "properties": {"product_id": {"type": "string"}}, "required": ["product_id"]}, "annotations": {"mutating": True}},
    {"name": "get_cart", "description": "View current cart contents and total.", "input_schema": {"type": "object", "properties": {}, "required": []}, "annotations": {"readonly": True, "mutating": False}},
    {"name": "clear_cart", "description": "Remove all items from cart.", "input_schema": {"type": "object", "properties": {}, "required": []}, "annotations": {"mutating": True}},
    {"name": "apply_coupon", "description": "Apply a coupon code to cart.", "input_schema": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}, "annotations": {"mutating": True}},
    {"name": "get_coupons", "description": "Get available coupons.", "input_schema": {"type": "object", "properties": {}, "required": []}, "annotations": {"readonly": True, "mutating": False}},
    {"name": "checkout", "description": "Checkout and create an order.", "input_schema": {"type": "object", "properties": {"shipping_address": {"type": "string"}, "payment_method": {"type": "string"}}, "required": []}, "annotations": {"mutating": True}},
    {"name": "get_order", "description": "Get order details.", "input_schema": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]}, "annotations": {"readonly": True, "mutating": False}},
    {"name": "list_orders", "description": "List past orders.", "input_schema": {"type": "object", "properties": {"status": {"type": "string"}}, "required": []}, "annotations": {"readonly": True, "mutating": False}},
    {"name": "track_order", "description": "Track order delivery status.", "input_schema": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]}, "annotations": {"readonly": True, "mutating": False}},
    {"name": "return_order", "description": "Initiate a return for an order.", "input_schema": {"type": "object", "properties": {"order_id": {"type": "string"}, "reason": {"type": "string"}, "items": {"type": "array"}}, "required": ["order_id", "reason"]}, "annotations": {"mutating": True}},
    {"name": "get_return_status", "description": "Check return status.", "input_schema": {"type": "object", "properties": {"return_id": {"type": "string"}}, "required": ["return_id"]}, "annotations": {"readonly": True, "mutating": False}},
    {"name": "add_review", "description": "Add a product review.", "input_schema": {"type": "object", "properties": {"product_id": {"type": "string"}, "rating": {"type": "integer"}, "title": {"type": "string"}, "body": {"type": "string"}}, "required": ["product_id", "rating", "body"]}, "annotations": {"mutating": True}},
    {"name": "get_reviews", "description": "Get reviews for a product.", "input_schema": {"type": "object", "properties": {"product_id": {"type": "string"}, "sort_by": {"type": "string"}}, "required": ["product_id"]}, "annotations": {"readonly": True, "mutating": False}},
    {"name": "add_to_wishlist", "description": "Add product to wishlist.", "input_schema": {"type": "object", "properties": {"product_id": {"type": "string"}}, "required": ["product_id"]}, "annotations": {"mutating": True}},
    {"name": "remove_from_wishlist", "description": "Remove product from wishlist.", "input_schema": {"type": "object", "properties": {"product_id": {"type": "string"}}, "required": ["product_id"]}, "annotations": {"mutating": True}},
    {"name": "get_wishlist", "description": "View wishlist.", "input_schema": {"type": "object", "properties": {}, "required": []}, "annotations": {"readonly": True, "mutating": False}},
]

COUPONS = {"SAVE10": 0.10, "WELCOME20": 0.20, "FREESHIP": None}

class ShoppingServer(StatefulToolServer):
    def __init__(self) -> None:
        super().__init__("shopping", TOOLS)
        self.handlers = {t["name"]: getattr(self, t["name"]) for t in TOOLS}

    def search_products(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id); products = list(state["products"].values())
        q = arguments.get("query", "").lower(); cat = arguments.get("category"); mn = arguments.get("min_price"); mx = arguments.get("max_price")
        if q: products = [p for p in products if q in p["name"].lower() or q in p.get("description", "").lower()]
        if cat: products = [p for p in products if p.get("category") == cat]
        if mn is not None: products = [p for p in products if p["price"] >= float(mn)]
        if mx is not None: products = [p for p in products if p["price"] <= float(mx)]
        if arguments.get("in_stock_only"): products = [p for p in products if p["stock"] > 0]
        return _result(True, {"products": products, "count": len(products)}, None, "", False)

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
        products = [state["products"][pid] for pid in ids if pid in state["products"]]
        return _result(True, {"products": products, "count": len(products)}, None, "", False)

    def get_recommendations(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id); limit = int(arguments.get("limit", 5))
        base = arguments.get("based_on_product"); cat = arguments.get("category")
        products = list(state["products"].values())
        if base and base in state["products"]: cat = state["products"][base].get("category")
        if cat: products = [p for p in products if p.get("category") == cat]
        return _result(True, {"recommendations": products[:limit]}, None, "", False)

    def add_to_cart(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id); pid, qty = arguments["product_id"], int(arguments["quantity"])
        p = state["products"].get(pid)
        if not p: raise KeyError(f"product not found: {pid}")
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
        diff = qty - item["quantity"]; p = state["products"][pid]
        if diff > 0 and p["stock"] < diff: raise KeyError(f"insufficient stock: {pid}")
        p["stock"] -= diff; item["quantity"] = qty
        return _result(True, {"cart": list(state["cart"]), "total": sum(i["quantity"] * i["unit_price"] for i in state["cart"])}, None, "", True)

    def remove_from_cart(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id); pid = arguments["product_id"]
        kept = []; removed = None
        for item in state["cart"]:
            if item["product_id"] == pid: removed = item; state["products"][pid]["stock"] += item["quantity"]
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
        return _result(True, {"cart": list(state["cart"]), "total": total, "discount": discount, "final_total": total - discount, "item_count": len(state["cart"])}, None, "", False)

    def clear_cart(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id)
        for item in state["cart"]: state["products"][item["product_id"]]["stock"] += item["quantity"]
        state["cart"] = []; state.pop("applied_coupon", None)
        return _result(True, {"cart": [], "message": "cart cleared"}, None, "", True)

    def apply_coupon(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id); code = arguments["code"].upper()
        if code not in COUPONS: raise KeyError(f"invalid coupon: {code}")
        state["applied_coupon"] = code
        return _result(True, {"coupon": code, "discount": f"{COUPONS[code]*100 if COUPONS[code] else 'free shipping'}%"}, None, "", True)

    def get_coupons(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        return _result(True, {"coupons": [{"code": k, "discount": f"{v*100}%" if v else "free shipping"} for k, v in COUPONS.items()]}, None, "", False)

    def checkout(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id)
        if not state["cart"]: raise KeyError("cart is empty")
        total = sum(i["quantity"] * i["unit_price"] for i in state["cart"])
        coupon = state.get("applied_coupon")
        if coupon and coupon in COUPONS and COUPONS[coupon]: total *= (1 - COUPONS[coupon])
        oid = f"ord_{state['next_order_num']:03d}"; state["next_order_num"] += 1
        order = {"order_id": oid, "items": list(state["cart"]), "total": round(total, 2), "shipping_address": arguments.get("shipping_address", ""), "payment_method": arguments.get("payment_method", "card"), "status": "placed", "tracking": [], "created_at": "2026-06-24"}
        state["orders"][oid] = order; state["cart"] = []; state.pop("applied_coupon", None)
        return _result(True, {"order": order}, None, "", True)

    def get_order(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        o = self._state(session_id)["orders"].get(arguments["order_id"])
        if not o: raise KeyError(f"order not found: {arguments['order_id']}")
        return _result(True, {"order": o}, None, "", False)

    def list_orders(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        orders = list(self._state(session_id)["orders"].values())
        if arguments.get("status"): orders = [o for o in orders if o.get("status") == arguments["status"]]
        return _result(True, {"orders": orders, "count": len(orders)}, None, "", False)

    def track_order(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        o = self._state(session_id)["orders"].get(arguments["order_id"])
        if not o: raise KeyError(f"order not found: {arguments['order_id']}")
        tracking = o.get("tracking", [{"status": o.get("status", "placed"), "timestamp": "2026-06-24", "location": "Warehouse"}])
        return _result(True, {"order_id": o["order_id"], "tracking": tracking, "current_status": o.get("status")}, None, "", False)

    def return_order(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id); o = state["orders"].get(arguments["order_id"])
        if not o: raise KeyError(f"order not found: {arguments['order_id']}")
        if o.get("status") == "returned": raise KeyError("order already returned")
        rid = f"ret_{state['next_order_num']:03d}"; state["next_order_num"] += 1
        ret = {"return_id": rid, "order_id": o["order_id"], "reason": arguments["reason"], "items": arguments.get("items", []), "status": "initiated"}
        state.setdefault("returns", {})[rid] = ret; o["status"] = "returning"
        return _result(True, {"return": ret, "order_status": "returning"}, None, "", True)

    def get_return_status(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        ret = self._state(session_id).get("returns", {}).get(arguments["return_id"])
        if not ret: raise KeyError(f"return not found: {arguments['return_id']}")
        return _result(True, {"return": ret}, None, "", False)

    def add_review(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id); pid = arguments["product_id"]; rating = int(arguments["rating"])
        if pid not in state["products"]: raise KeyError(f"product not found: {pid}")
        if not 1 <= rating <= 5: raise KeyError("rating must be 1-5")
        rid = f"rev_{state['next_order_num']:03d}"; state["next_order_num"] += 1
        review = {"review_id": rid, "product_id": pid, "rating": rating, "title": arguments.get("title", ""), "body": arguments["body"], "author": "current_user", "date": "2026-06-24"}
        state.setdefault("reviews", {}).setdefault(pid, []).append(review)
        return _result(True, {"review": review}, None, "", True)

    def get_reviews(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id); pid = arguments["product_id"]
        if pid not in state["products"]: raise KeyError(f"product not found: {pid}")
        reviews = state.get("reviews", {}).get(pid, [])
        if arguments.get("sort_by") == "rating": reviews = sorted(reviews, key=lambda r: r["rating"], reverse=True)
        avg = round(sum(r["rating"] for r in reviews) / len(reviews), 1) if reviews else 0
        return _result(True, {"product_id": pid, "reviews": reviews, "average_rating": avg, "count": len(reviews)}, None, "", False)

    def add_to_wishlist(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id); pid = arguments["product_id"]
        if pid not in state["products"]: raise KeyError(f"product not found: {pid}")
        wl = state.setdefault("wishlist", [])
        if pid not in wl: wl.append(pid)
        return _result(True, {"wishlist": wl, "count": len(wl)}, None, "", True)

    def remove_from_wishlist(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id); pid = arguments["product_id"]
        wl = state.setdefault("wishlist", [])
        if pid in wl: wl.remove(pid)
        return _result(True, {"wishlist": wl, "count": len(wl)}, None, "", True)

    def get_wishlist(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id); wl = state.get("wishlist", [])
        products = [state["products"][pid] for pid in wl if pid in state["products"]]
        return _result(True, {"wishlist": products, "count": len(products)}, None, "", False)


if __name__ == "__main__":
    serve(ShoppingServer())
