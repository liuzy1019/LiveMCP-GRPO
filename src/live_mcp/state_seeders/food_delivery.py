"""Deterministic state builder for food_delivery."""

from __future__ import annotations

import datetime as _datetime
import random
from typing import Any

from src.live_mcp.state_seeders.common import (
    _reference_datetime,
    _sample_entities,
    _seed_scoped_id,
)
_FOOD_DELIVERY_RESTAURANT_TEMPLATES: list[tuple[str, str, str, float, float, list[tuple[str, float, list[str]]]]] = [
    ("rest_t01", "Pizza Palace", "Italian", 4.5, 2.99,
     [("Margherita Pizza", 12.99, ["vegetarian"]),
      ("Pepperoni Pizza", 14.99, []),
      ("Caesar Salad", 8.99, ["vegetarian", "gluten-free"]),
      ("Garlic Bread", 4.99, ["vegetarian"])]),
    ("rest_t02", "Sushi Express", "Japanese", 4.8, 3.99,
     [("California Roll", 10.99, []),
      ("Salmon Nigiri", 12.99, ["gluten-free"]),
      ("Miso Soup", 3.99, ["vegetarian", "gluten-free"]),
      ("Edamame", 4.99, ["vegan", "gluten-free"])]),
    ("rest_t03", "Burger Barn", "American", 4.2, 1.99,
     [("Classic Burger", 9.99, []),
      ("Cheese Burger", 11.99, []),
      ("French Fries", 3.99, ["vegetarian", "gluten-free"]),
      ("Milkshake", 5.99, ["vegetarian"])]),
    ("rest_t04", "Taco Fiesta", "Mexican", 4.3, 2.49,
     [("Chicken Taco", 7.99, []),
      ("Beef Burrito", 10.99, []),
      ("Guacamole", 5.99, ["vegan", "gluten-free"]),
      ("Churros", 3.99, ["vegetarian"])]),
    ("rest_t05", "Curry House", "Indian", 4.6, 3.49,
     [("Chicken Tikka Masala", 13.99, ["gluten-free"]),
      ("Vegetable Biryani", 11.99, ["vegetarian", "gluten-free"]),
      ("Naan Bread", 3.49, ["vegetarian"]),
      ("Mango Lassi", 4.99, ["vegetarian", "gluten-free"])]),
    ("rest_t06", "Pho Garden", "Vietnamese", 4.4, 2.99,
     [("Beef Pho", 11.99, ["gluten-free"]),
      ("Spring Rolls", 6.99, ["vegetarian"]),
      ("Banh Mi", 8.99, []),
      ("Vietnamese Coffee", 4.49, ["vegetarian"])]),
    ("rest_t07", "Dragon Wok", "Chinese", 4.1, 2.49,
     [("Kung Pao Chicken", 13.49, []),
      ("Mapo Tofu", 10.99, ["vegetarian"]),
      ("Vegetable Fried Rice", 9.49, ["vegetarian", "vegan"]),
      ("Wonton Soup", 5.99, [])]),
    ("rest_t08", "Mediterranean Grill", "Mediterranean", 4.5, 3.49,
     [("Chicken Shawarma", 12.99, []),
      ("Falafel Plate", 10.49, ["vegetarian", "vegan"]),
      ("Hummus & Pita", 6.99, ["vegetarian", "vegan"]),
      ("Greek Salad", 8.49, ["vegetarian", "gluten-free"])]),
    ("rest_t09", "BBQ Smokehouse", "BBQ", 4.0, 3.99,
     [("Pulled Pork Sandwich", 11.99, []),
      ("Beef Brisket Plate", 16.99, ["gluten-free"]),
      ("Cornbread", 4.49, ["vegetarian"]),
      ("Coleslaw", 3.99, ["vegetarian", "gluten-free"])]),
    ("rest_t10", "Smoothie Lab", "Healthy", 4.3, 1.99,
     [("Green Detox Smoothie", 7.99, ["vegan", "gluten-free"]),
      ("Berry Blast Bowl", 9.99, ["vegetarian", "gluten-free"]),
      ("Protein Power Wrap", 8.99, []),
      ("Kale Caesar", 7.49, ["vegetarian", "gluten-free"])]),
    ("rest_t11", "Ramen House", "Japanese", 4.6, 2.99,
     [("Tonkotsu Ramen", 13.99, []),
      ("Miso Ramen", 12.49, ["vegetarian"]),
      ("Gyoza", 6.99, []),
      ("Matcha Ice Cream", 4.99, ["vegetarian"])]),
    ("rest_t12", "French Bistro", "French", 4.4, 4.49,
     [("Croque Monsieur", 10.99, []),
      ("French Onion Soup", 8.99, ["vegetarian"]),
      ("Quiche Lorraine", 9.49, []),
      ("Crème Brûlée", 7.49, ["vegetarian", "gluten-free"])]),
    ("rest_t13", "Kebab King", "Middle Eastern", 4.2, 2.99,
     [("Lamb Kebab Plate", 14.99, []),
      ("Falafel Wrap", 9.99, ["vegan"]),
      ("Baba Ganoush", 5.99, ["vegan", "gluten-free"]),
      ("Baklava", 4.99, ["vegetarian"])]),
    ("rest_t14", "Poke Bowl Co", "Hawaiian", 4.5, 3.49,
     [("Ahi Tuna Bowl", 14.99, ["gluten-free"]),
      ("Salmon Avocado Bowl", 13.99, ["gluten-free"]),
      ("Tofu Poke Bowl", 11.99, ["vegan", "gluten-free"]),
      ("Miso Soup", 3.49, ["vegetarian", "gluten-free"])]),
    ("rest_t15", "Dim Sum House", "Chinese", 4.3, 2.49,
     [("Har Gow", 7.99, []),
      ("Siu Mai", 7.49, []),
      ("Char Siu Bao", 6.99, []),
      ("Egg Tart", 4.49, ["vegetarian"])]),
    ("rest_t16", "Vegan Garden", "Vegan", 4.5, 2.99,
     [("Beyond Burger", 12.99, ["vegan"]),
      ("Jackfruit Tacos", 10.99, ["vegan", "gluten-free"]),
      ("Coconut Curry", 11.49, ["vegan", "gluten-free"]),
      ("Raw Cheesecake", 7.99, ["vegan", "gluten-free"])]),
    ("rest_t17", "Steakhouse Prime", "Steakhouse", 4.7, 5.99,
     [("Ribeye Steak 12oz", 28.99, ["gluten-free"]),
      ("Filet Mignon 8oz", 34.99, ["gluten-free"]),
      ("Loaded Baked Potato", 6.99, ["vegetarian", "gluten-free"]),
      ("Caesar Salad", 8.49, [])]),
    ("rest_t18", "Pad Thai Express", "Thai", 4.2, 2.49,
     [("Chicken Pad Thai", 11.99, []),
      ("Green Curry", 12.49, ["gluten-free"]),
      ("Tom Yum Soup", 6.99, ["gluten-free"]),
      ("Mango Sticky Rice", 5.99, ["vegetarian", "gluten-free"])]),
    ("rest_t19", "Bagel & Lox", "Deli", 4.1, 1.99,
     [("Everything Bagel with Cream Cheese", 4.99, ["vegetarian"]),
      ("Lox Bagel Sandwich", 9.99, []),
      ("Pastrami on Rye", 11.49, []),
      ("Matzo Ball Soup", 6.49, [])]),
    ("rest_t20", "Ethiopian Spice", "Ethiopian", 4.4, 2.99,
     [("Doro Wat", 13.99, ["gluten-free"]),
      ("Misir Wot", 10.99, ["vegan", "gluten-free"]),
      ("Injera Platter", 12.49, ["vegetarian"]),
      ("Sambusa", 5.99, ["vegetarian"])]),
    ("rest_t21", "Seafood Shack", "Seafood", 4.3, 3.99,
     [("Fish & Chips", 12.99, []),
      ("Garlic Butter Shrimp", 15.99, ["gluten-free"]),
      ("Clam Chowder", 7.99, []),
      ("Crab Cakes", 13.99, [])]),
    ("rest_t22", "Korean BBQ House", "Korean", 4.6, 3.99,
     [("Bulgogi Bowl", 14.99, []),
      ("Bibimbap", 12.99, ["vegetarian"]),
      ("Kimchi Jjigae", 10.99, ["gluten-free"]),
      ("Tteokbokki", 8.99, ["vegan"])]),
]

def _food_delivery_state(seed: int) -> dict[str, Any]:
    # Query generation and state seeding must share one temporal anchor.
    # Keep the import local so the generic seeder remains lightweight.
    rng = random.Random(seed)
    reference_dt = _reference_datetime(seed)
    days_since_friday = (reference_dt.weekday() - 4) % 7
    if days_since_friday == 0:
        days_since_friday = 7
    previous_friday = reference_dt - _datetime.timedelta(days=days_since_friday)
    recent_order = reference_dt - _datetime.timedelta(days=2)
    selected = _sample_entities(rng, _FOOD_DELIVERY_RESTAURANT_TEMPLATES, target_count=20, id_prefix="rest")
    restaurants = {}
    restaurant_ids: list[str] = []
    for idx_rest, (_rid, name, cuisine, rating, delivery_fee, menu_items) in enumerate(selected):
        rid = _seed_scoped_id("rest", seed, idx_rest, width=3)
        restaurant_ids.append(rid)
        menu = []
        for i, (mname, mprice, tags) in enumerate(menu_items):
            menu.append({
                "name": mname, "price": mprice + rng.randint(-1, 1),
                "dietary_tags": tags,
                "popularity": 50 + (i + rng.randint(0, 2)) * 15,
            })
        restaurants[rid] = {
            "restaurant_id": rid, "name": name, "cuisine": cuisine,
            "rating": round(rating + rng.uniform(-0.2, 0.2), 1),
            "delivery_fee": delivery_fee, "open": True,
            "hours": "11:00-22:00", "menu": menu,
        }
    order_1 = _seed_scoped_id("ord", seed, 0, width=4)
    order_2 = _seed_scoped_id("ord", seed, 1, width=4)
    order_3 = _seed_scoped_id("ord", seed, 2, width=4)
    order_4 = _seed_scoped_id("ord", seed, 3, width=4)
    order_5 = _seed_scoped_id("ord", seed, 4, width=4)
    order_6 = _seed_scoped_id("ord", seed, 5, width=4)
    order_7 = _seed_scoped_id("ord", seed, 6, width=4)
    order_8 = _seed_scoped_id("ord", seed, 7, width=4)
    first_rest = restaurants[restaurant_ids[0]]
    second_rest = restaurants[restaurant_ids[1]] if len(restaurant_ids) > 1 else first_rest
    third_rest = restaurants[restaurant_ids[2]] if len(restaurant_ids) > 2 else first_rest
    fourth_rest = restaurants[restaurant_ids[3]] if len(restaurant_ids) > 3 else first_rest
    orders = {
        order_1: {"order_id": order_1, "restaurant_id": restaurant_ids[0],
                     "restaurant_name": first_rest["name"],
                     "items": [{"name": first_rest["menu"][0]["name"], "quantity": 2}],
                     "delivery_address": "123 Main St", "subtotal": round(first_rest["menu"][0]["price"] * 2, 2),
                     "delivery_fee": first_rest["delivery_fee"], "tip": 3.00,
                     "total": round(first_rest["menu"][0]["price"] * 2 + first_rest["delivery_fee"] + 3.00, 2),
                     "status": "delivered", "rating": None,
                     "created_at": previous_friday.replace(
                         hour=18, minute=0, second=0, microsecond=0,
                     ).isoformat()},
        order_2: {"order_id": order_2, "restaurant_id": restaurant_ids[1] if len(restaurant_ids) > 1 else restaurant_ids[0],
                     "restaurant_name": second_rest["name"],
                     "items": [{"name": second_rest["menu"][0]["name"], "quantity": 1}],
                     "delivery_address": "456 Oak Ave", "subtotal": second_rest["menu"][0]["price"],
                     "delivery_fee": second_rest["delivery_fee"], "tip": 0,
                     "total": round(second_rest["menu"][0]["price"] + second_rest["delivery_fee"], 2),
                     "status": "confirmed", "rating": None,
                     "created_at": recent_order.replace(
                         hour=12, minute=30, second=0, microsecond=0,
                     ).isoformat()},
        order_3: {"order_id": order_3, "restaurant_id": restaurant_ids[0],
                     "restaurant_name": first_rest["name"],
                     "items": [{"name": first_rest["menu"][1]["name"], "quantity": 1}],
                     "delivery_address": "789 Pine Rd",
                     "subtotal": first_rest["menu"][1]["price"],
                     "delivery_fee": first_rest["delivery_fee"], "tip": 0,
                     "total": round(first_rest["menu"][1]["price"] + first_rest["delivery_fee"], 2),
                     "status": "delivering", "rating": None,
                     "created_at": reference_dt.replace(
                         hour=19, minute=15, second=0, microsecond=0,
                     ).isoformat()},
        order_4: {"order_id": order_4, "restaurant_id": restaurant_ids[2] if len(restaurant_ids) > 2 else restaurant_ids[0],
                     "restaurant_name": third_rest["name"],
                     "items": [{"name": third_rest["menu"][0]["name"], "quantity": 1}],
                     "delivery_address": "321 Elm St",
                     "subtotal": third_rest["menu"][0]["price"],
                     "delivery_fee": third_rest["delivery_fee"], "tip": 2.00,
                     "total": round(third_rest["menu"][0]["price"] + third_rest["delivery_fee"] + 2.00, 2),
                     "status": "placed", "rating": None,
                     "created_at": reference_dt.replace(
                         hour=20, minute=0, second=0, microsecond=0,
                     ).isoformat()},
        order_5: {"order_id": order_5, "restaurant_id": restaurant_ids[0],
                     "restaurant_name": first_rest["name"],
                     "items": [{"name": first_rest["menu"][2]["name"], "quantity": 2},
                               {"name": first_rest["menu"][3]["name"], "quantity": 1}],
                     "delivery_address": "123 Main St",
                     "subtotal": round(first_rest["menu"][2]["price"] * 2 + first_rest["menu"][3]["price"], 2),
                     "delivery_fee": first_rest["delivery_fee"], "tip": 4.00,
                     "total": round(first_rest["menu"][2]["price"] * 2 + first_rest["menu"][3]["price"] + first_rest["delivery_fee"] + 4.00, 2),
                     "status": "delivered", "rating": 5,
                     "created_at": (previous_friday - _datetime.timedelta(days=7)).replace(
                         hour=19, minute=30, second=0, microsecond=0,
                     ).isoformat()},
        order_6: {"order_id": order_6, "restaurant_id": restaurant_ids[3] if len(restaurant_ids) > 3 else restaurant_ids[0],
                     "restaurant_name": fourth_rest["name"],
                     "items": [{"name": fourth_rest["menu"][0]["name"], "quantity": 1}],
                     "delivery_address": "654 Maple Dr",
                     "subtotal": fourth_rest["menu"][0]["price"],
                     "delivery_fee": fourth_rest["delivery_fee"], "tip": 0,
                     "total": round(fourth_rest["menu"][0]["price"] + fourth_rest["delivery_fee"], 2),
                     "status": "cancelled", "rating": None,
                     "created_at": (reference_dt - _datetime.timedelta(days=1)).replace(
                         hour=17, minute=45, second=0, microsecond=0,
                     ).isoformat()},
        order_7: {"order_id": order_7, "restaurant_id": restaurant_ids[0],
                     "restaurant_name": first_rest["name"],
                     "items": [{"name": first_rest["menu"][3]["name"], "quantity": 1}],
                     "delivery_address": "999 Birch Ln",
                     "subtotal": first_rest["menu"][3]["price"],
                     "delivery_fee": first_rest["delivery_fee"], "tip": 1.50,
                     "total": round(first_rest["menu"][3]["price"] + first_rest["delivery_fee"] + 1.50, 2),
                     "status": "delivered", "rating": 3,
                     "created_at": (previous_friday - _datetime.timedelta(days=3)).replace(
                         hour=12, minute=0, second=0, microsecond=0,
                     ).isoformat()},
        order_8: {"order_id": order_8, "restaurant_id": restaurant_ids[1] if len(restaurant_ids) > 1 else restaurant_ids[0],
                     "restaurant_name": second_rest["name"],
                     "items": [{"name": second_rest["menu"][1]["name"], "quantity": 3}],
                     "delivery_address": "456 Oak Ave",
                     "subtotal": round(second_rest["menu"][1]["price"] * 3, 2),
                     "delivery_fee": second_rest["delivery_fee"], "tip": 5.00,
                     "total": round(second_rest["menu"][1]["price"] * 3 + second_rest["delivery_fee"] + 5.00, 2),
                     "status": "delivered", "rating": 4,
                     "created_at": (previous_friday - _datetime.timedelta(days=14)).replace(
                         hour=20, minute=30, second=0, microsecond=0,
                     ).isoformat()},
    }
    # Seed support tickets so get_support_ticket chains have material data.
    ticket_1 = _seed_scoped_id("tkt", seed, 0, width=4)
    ticket_2 = _seed_scoped_id("tkt", seed, 1, width=4)
    support_tickets = [
        {"ticket_id": ticket_1, "order_id": order_5,
         "issue": "Missing item in order", "status": "open",
         "created_at": reference_dt.replace(hour=20, minute=0, second=0, microsecond=0).isoformat()},
        {"ticket_id": ticket_2, "order_id": order_7,
         "issue": "Food arrived cold", "status": "resolved",
         "created_at": (reference_dt - _datetime.timedelta(days=2)).replace(hour=14, minute=0, second=0, microsecond=0).isoformat()},
    ]
    return {"restaurants": restaurants, "orders": orders, "support_tickets": support_tickets,
            "current_date": reference_dt.date().isoformat(),
            "next_order_num": 9, "next_ticket_num": 3}