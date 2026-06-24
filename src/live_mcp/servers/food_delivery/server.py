"""Stateful food delivery server — 17 tools (PROVE-aligned).
Lifecycle state: placed→confirmed→preparing→delivering→delivered / cancelled.
Features: restaurants, menus, orders, tracking, rating, tip, reorder, support.
"""

from __future__ import annotations
from typing import Any
from src.live_mcp.server_base import StatefulToolServer, _result, serve

LIFECYCLE = {"placed": ["confirmed", "cancelled"], "confirmed": ["preparing", "cancelled"], "preparing": ["delivering"], "delivering": ["delivered"], "delivered": [], "cancelled": []}

TOOLS = [
    {"name": "list_restaurants", "description": "List restaurants by cuisine, rating, price level.", "input_schema": {"type": "object", "properties": {"cuisine": {"type": "string"}, "min_rating": {"type": "number"}, "max_delivery_fee": {"type": "number"}, "open_now": {"type": "boolean"}}, "required": []}, "annotations": {"readonly": True, "mutating": False}},
    {"name": "search_restaurants", "description": "Search restaurants by name or keyword.", "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}, "annotations": {"readonly": True, "mutating": False}},
    {"name": "get_restaurant", "description": "Get restaurant details with hours and rating.", "input_schema": {"type": "object", "properties": {"restaurant_id": {"type": "string"}}, "required": ["restaurant_id"]}, "annotations": {"readonly": True, "mutating": False}},
    {"name": "get_menu", "description": "Get restaurant menu with prices and dietary info.", "input_schema": {"type": "object", "properties": {"restaurant_id": {"type": "string"}}, "required": ["restaurant_id"]}, "annotations": {"readonly": True, "mutating": False}},
    {"name": "filter_by_dietary", "description": "Filter menu items by dietary restriction.", "input_schema": {"type": "object", "properties": {"restaurant_id": {"type": "string"}, "dietary": {"type": "string"}}, "required": ["restaurant_id", "dietary"]}, "annotations": {"readonly": True, "mutating": False}},
    {"name": "get_popular_items", "description": "Get most-ordered items from a restaurant.", "input_schema": {"type": "object", "properties": {"restaurant_id": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["restaurant_id"]}, "annotations": {"readonly": True, "mutating": False}},
    {"name": "create_order", "description": "Place a food delivery order.", "input_schema": {"type": "object", "properties": {"restaurant_id": {"type": "string"}, "items": {"type": "array"}, "delivery_address": {"type": "string"}, "special_instructions": {"type": "string"}}, "required": ["restaurant_id", "items", "delivery_address"]}, "annotations": {"mutating": True}},
    {"name": "get_order", "description": "Get order details and current status.", "input_schema": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]}, "annotations": {"readonly": True, "mutating": False}},
    {"name": "list_orders", "description": "List orders by status.", "input_schema": {"type": "object", "properties": {"status": {"type": "string"}}, "required": []}, "annotations": {"readonly": True, "mutating": False}},
    {"name": "update_order_status", "description": "Advance order to next lifecycle stage.", "input_schema": {"type": "object", "properties": {"order_id": {"type": "string"}, "status": {"type": "string"}}, "required": ["order_id", "status"]}, "annotations": {"mutating": True}},
    {"name": "cancel_order", "description": "Cancel an order (only before preparing).", "input_schema": {"type": "object", "properties": {"order_id": {"type": "string"}, "reason": {"type": "string"}}, "required": ["order_id"]}, "annotations": {"mutating": True}},
    {"name": "get_estimated_time", "description": "Get estimated delivery time for an order.", "input_schema": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]}, "annotations": {"readonly": True, "mutating": False}},
    {"name": "track_rider", "description": "Track delivery rider location.", "input_schema": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]}, "annotations": {"readonly": True, "mutating": False}},
    {"name": "rate_order", "description": "Rate a delivered order.", "input_schema": {"type": "object", "properties": {"order_id": {"type": "string"}, "rating": {"type": "integer"}, "review": {"type": "string"}}, "required": ["order_id", "rating"]}, "annotations": {"mutating": True}},
    {"name": "add_tip", "description": "Add a tip to an order.", "input_schema": {"type": "object", "properties": {"order_id": {"type": "string"}, "amount": {"type": "number"}}, "required": ["order_id", "amount"]}, "annotations": {"mutating": True}},
    {"name": "reorder", "description": "Re-order from a past order.", "input_schema": {"type": "object", "properties": {"order_id": {"type": "string"}, "delivery_address": {"type": "string"}}, "required": ["order_id"]}, "annotations": {"mutating": True}},
    {"name": "contact_support", "description": "Contact support about an order.", "input_schema": {"type": "object", "properties": {"order_id": {"type": "string"}, "issue_type": {"type": "string"}, "description": {"type": "string"}}, "required": ["order_id", "issue_type", "description"]}, "annotations": {"mutating": True}},
]

class FoodDeliveryServer(StatefulToolServer):
    def __init__(self) -> None:
        super().__init__("food_delivery", TOOLS)
        self.handlers = {t["name"]: getattr(self, t["name"]) for t in TOOLS}

    def _rest(self, state, rid):
        if rid not in state["restaurants"]: raise KeyError(f"restaurant not found: {rid}")
        return state["restaurants"][rid]

    def _order(self, state, oid):
        if oid not in state["orders"]: raise KeyError(f"order not found: {oid}")
        return state["orders"][oid]

    def list_restaurants(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id); rests = list(state["restaurants"].values())
        if arguments.get("cuisine"): rests = [r for r in rests if r["cuisine"] == arguments["cuisine"]]
        if arguments.get("min_rating"): rests = [r for r in rests if r["rating"] >= float(arguments["min_rating"])]
        if arguments.get("max_delivery_fee") is not None: rests = [r for r in rests if r.get("delivery_fee", 0) <= float(arguments["max_delivery_fee"])]
        if arguments.get("open_now"): rests = [r for r in rests if r.get("open", True)]
        summary = [{"restaurant_id": r["restaurant_id"], "name": r["name"], "cuisine": r["cuisine"], "rating": r["rating"], "delivery_fee": r.get("delivery_fee", 0)} for r in rests]
        return _result(True, {"restaurants": summary, "count": len(summary)}, None, "", False)

    def search_restaurants(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        q = arguments["query"].lower(); rests = [r for r in self._state(session_id)["restaurants"].values() if q in r["name"].lower() or q in r.get("cuisine", "").lower()]
        return _result(True, {"restaurants": [{"restaurant_id": r["restaurant_id"], "name": r["name"]} for r in rests], "count": len(rests)}, None, "", False)

    def get_restaurant(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        r = self._rest(self._state(session_id), arguments["restaurant_id"])
        return _result(True, {"restaurant": {k: v for k, v in r.items() if k != "menu"}}, None, "", False)

    def get_menu(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        r = self._rest(self._state(session_id), arguments["restaurant_id"])
        return _result(True, {"restaurant_id": r["restaurant_id"], "menu": r["menu"], "count": len(r["menu"])}, None, "", False)

    def filter_by_dietary(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        r = self._rest(self._state(session_id), arguments["restaurant_id"]); dietary = arguments["dietary"].lower()
        items = [item for item in r["menu"] if dietary in [d.lower() for d in item.get("dietary_tags", [])]]
        return _result(True, {"restaurant_id": r["restaurant_id"], "dietary": dietary, "items": items, "count": len(items)}, None, "", False)

    def get_popular_items(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        r = self._rest(self._state(session_id), arguments["restaurant_id"]); limit = int(arguments.get("limit", 5))
        items = sorted(r["menu"], key=lambda x: x.get("popularity", 0), reverse=True)[:limit]
        return _result(True, {"restaurant_id": r["restaurant_id"], "popular_items": items}, None, "", False)

    def create_order(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id); rid = arguments["restaurant_id"]; r = self._rest(state, rid)
        items = arguments["items"]; menu_map = {m["name"]: m for m in r["menu"]}; total = 0.0
        for item in items:
            name = item["name"]; qty = item.get("quantity", 1)
            if name not in menu_map: raise KeyError(f"item not on menu: {name}")
            if qty <= 0: raise KeyError("quantity must be positive")
            total += menu_map[name]["price"] * qty
        oid = f"ord_{state['next_order_num']:04d}"; state["next_order_num"] += 1
        delivery_fee = r.get("delivery_fee", 2.99)
        order = {"order_id": oid, "restaurant_id": rid, "restaurant_name": r["name"], "items": items, "delivery_address": arguments["delivery_address"], "special_instructions": arguments.get("special_instructions", ""), "subtotal": total, "delivery_fee": delivery_fee, "tip": 0.0, "total": round(total + delivery_fee, 2), "status": "placed", "rating": None, "created_at": "2026-06-24T21:40:00"}
        state["orders"][oid] = order
        return _result(True, {"order": order}, None, "", True)

    def get_order(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        return _result(True, {"order": self._order(self._state(session_id), arguments["order_id"])}, None, "", False)

    def list_orders(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        orders = list(self._state(session_id)["orders"].values())
        if arguments.get("status"): orders = [o for o in orders if o["status"] == arguments["status"]]
        return _result(True, {"orders": orders, "count": len(orders)}, None, "", False)

    def update_order_status(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id); oid = arguments["order_id"]; new_status = arguments["status"]
        order = self._order(state, oid); old = order["status"]; allowed = LIFECYCLE.get(old, [])
        if new_status not in allowed: raise KeyError(f"invalid transition: {old} -> {new_status}")
        order["status"] = new_status
        return _result(True, {"order": order, "previous_status": old}, None, "", True)

    def cancel_order(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id); oid = arguments["order_id"]; order = self._order(state, oid)
        if order["status"] in ("preparing", "delivering", "delivered"): raise KeyError(f"cannot cancel order in status: {order['status']}")
        if order["status"] == "cancelled": raise KeyError("order already cancelled")
        order["status"] = "cancelled"; order["cancel_reason"] = arguments.get("reason", "")
        return _result(True, {"order": order}, None, "", True)

    def get_estimated_time(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        order = self._order(self._state(session_id), arguments["order_id"]); etas = {"placed": 45, "confirmed": 35, "preparing": 25, "delivering": 10, "delivered": 0, "cancelled": 0}
        eta = etas.get(order["status"], 45)
        return _result(True, {"order_id": order["order_id"], "status": order["status"], "estimated_minutes": eta}, None, "", False)

    def track_rider(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        order = self._order(self._state(session_id), arguments["order_id"])
        if order["status"] not in ("delivering",): raise KeyError(f"rider not assigned for status: {order['status']}")
        return _result(True, {"order_id": order["order_id"], "rider_name": "Alex Driver", "rider_location": {"lat": 40.7128, "lng": -74.0060}, "distance_km": 1.2, "estimated_arrival": "2026-06-24T21:50:00"}, None, "", False)

    def rate_order(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        order = self._order(self._state(session_id), arguments["order_id"]); rating = int(arguments["rating"])
        if order["status"] != "delivered": raise KeyError("can only rate delivered orders")
        if not 1 <= rating <= 5: raise KeyError("rating must be 1-5")
        order["rating"] = rating; order["review"] = arguments.get("review", "")
        return _result(True, {"order_id": order["order_id"], "rating": rating}, None, "", True)

    def add_tip(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        order = self._order(self._state(session_id), arguments["order_id"]); amount = float(arguments["amount"])
        if amount <= 0: raise KeyError("tip amount must be positive")
        order["tip"] = (order.get("tip", 0) or 0) + amount; order["total"] = round(order["subtotal"] + order.get("delivery_fee", 0) + order["tip"], 2)
        return _result(True, {"order_id": order["order_id"], "tip": order["tip"], "total": order["total"]}, None, "", True)

    def reorder(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id); past = self._order(state, arguments["order_id"])
        return self.create_order(session_id, {"restaurant_id": past["restaurant_id"], "items": past["items"], "delivery_address": arguments.get("delivery_address", past["delivery_address"])})

    def contact_support(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id); order = self._order(state, arguments["order_id"]); issue = arguments["issue_type"]
        tid = f"ticket_{state['next_order_num']:04d}"; state["next_order_num"] += 1
        ticket = {"ticket_id": tid, "order_id": order["order_id"], "issue_type": issue, "description": arguments["description"], "status": "open", "created_at": "2026-06-24T21:40:00"}
        state.setdefault("support_tickets", []).append(ticket)
        return _result(True, {"ticket": ticket}, None, "", True)


if __name__ == "__main__":
    serve(FoodDeliveryServer())
