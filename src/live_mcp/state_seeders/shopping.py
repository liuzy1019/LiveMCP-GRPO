"""Deterministic state builder for shopping."""

from __future__ import annotations

import datetime as _datetime
import random
from typing import Any

from src.live_mcp.state_seeders.common import (
    _reference_datetime,
    _sample_entities,
    _seed_scoped_id,
)
_SHOPPING_PRODUCT_TEMPLATES: list[tuple[str, str, str, int, int, str]] = [
    ("prd_001", "K3 Keyboard", "keyboard", 79, 5, "Mechanical keyboard with RGB backlight"),
    ("prd_002", "MX Mouse", "mouse", 49, 8, "Ergonomic wireless mouse"),
    ("prd_003", "USB-C Hub", "hub", 35, 4, "7-in-1 USB-C hub with HDMI"),
    ("prd_004", "Noise Canceling Headphones", "audio", 99, 3, "Wireless ANC headphones"),
    ("prd_005", "4K Monitor 27\"", "monitor", 349, 6, "27-inch 4K IPS monitor"),
    ("prd_006", "Webcam Pro", "camera", 89, 10, "1080p webcam with auto-focus"),
    ("prd_007", "Standing Desk", "furniture", 549, 2, "Electric height-adjustable desk"),
    ("prd_008", "Laptop Stand", "accessory", 39, 15, "Aluminum laptop stand"),
    ("prd_009", "Thunderbolt Cable", "cable", 29, 20, "2m Thunderbolt 4 cable"),
    ("prd_010", "External SSD 1TB", "storage", 129, 7, "1TB portable SSD, USB-C"),
    ("prd_011", "Wireless Charger", "charger", 25, 12, "Qi fast wireless charging pad"),
    ("prd_012", "Desk Lamp LED", "lighting", 59, 8, "Adjustable LED desk lamp"),
    ("prd_013", "Mouse Pad XL", "accessory", 19, 25, "Extended gaming mouse pad"),
    ("prd_014", "Monitor Arm", "mount", 79, 4, "Gas-spring monitor arm"),
    ("prd_015", "Bluetooth Speaker", "audio", 69, 9, "Portable Bluetooth speaker"),
    ("prd_016", "Mechanical Keyboard V2", "keyboard", 119, 3, "Hot-swappable mechanical keyboard"),
    ("prd_017", "Ergonomic Mouse", "mouse", 69, 6, "Vertical ergonomic mouse"),
    ("prd_018", "USB Microphone", "audio", 45, 11, "Condenser USB microphone"),
    ("prd_019", "Drawing Tablet", "tablet", 249, 2, "10-inch drawing tablet with stylus"),
    ("prd_020", "Docking Station", "hub", 199, 5, "Dual-HDMI USB-C docking station"),
    ("prd_021", "Smart Plug", "smart_home", 19, 30, "Wi-Fi smart plug with energy monitoring"),
    ("prd_022", "Portable Battery 20000mAh", "charger", 49, 14, "20000mAh power bank, USB-C PD"),
    ("prd_023", "Cable Management Kit", "accessory", 15, 40, "Velcro cable ties and clips"),
    ("prd_024", "Keyboard Wrist Rest", "accessory", 22, 18, "Memory foam wrist rest"),
    ("prd_025", "USB-A to USB-C Adapter", "cable", 9, 50, "Pack of 3 USB-A to USB-C adapters"),
    ("prd_026", "Laser Printer Pro", "printer", 199, 4, "Monochrome laser printer, wireless"),
    ("prd_027", "Smartphone X", "phone", 699, 3, "6.5-inch AMOLED, 128GB"),
    ("prd_028", "Gaming Headset", "audio", 89, 7, "7.1 surround sound, noise-cancelling mic"),
    ("prd_029", "WiFi 6 Router", "networking", 129, 8, "Dual-band AX3000 router"),
    ("prd_030", "Laptop Backpack", "bag", 59, 10, "Water-resistant, padded laptop compartment"),
    ("prd_031", "Ergonomic Chair", "furniture", 399, 3, "Mesh back, adjustable lumbar support"),
    ("prd_032", "Compact Keyboard", "keyboard", 59, 5, "Tenkeyless mechanical keyboard"),
    ("prd_033", "Trackball Mouse", "mouse", 39, 4, "Wired trackball, ambidextrous"),
    ("prd_034", "UltraWide Monitor 34\"", "monitor", 499, 3, "34-inch QHD curved monitor"),
    ("prd_035", "NAS 2-Bay", "storage", 299, 3, "2-bay NAS enclosure, diskless"),
    ("prd_036", "DSLR Camera", "camera", 599, 2, "24MP DSLR with kit lens"),
    ("prd_037", "Budget Phone", "phone", 199, 6, "6.1-inch LCD, 64GB, dual SIM"),
    ("prd_038", "Printer Ink Set", "supplies", 39, 12, "Compatible ink cartridge 4-pack"),
    ("prd_039", "SD Card 256GB", "storage", 35, 15, "UHS-I microSD with adapter"),
    ("prd_040", "HDMI Cable 3m", "cable", 15, 25, "4K@60Hz HDMI 2.0 cable"),
    ("prd_041", "USB 3.0 Hub 4-Port", "hub", 25, 8, "Powered USB 3.0 hub, 4 ports"),
    ("prd_042", "Desk Mat Leather", "accessory", 35, 10, "PU leather desk mat, 90×45cm"),
    ("prd_043", "Ring Light", "lighting", 45, 6, "10-inch LED ring light with tripod"),
    ("prd_044", "Ethernet Cable 5m", "cable", 12, 20, "Cat6 Ethernet patch cable"),
    ("prd_045", "Smart Thermostat", "smart_home", 89, 5, "Wi-Fi programmable thermostat"),
]

def _shopping_state(seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    reference_date = _reference_datetime(seed).date()
    selected = _sample_entities(rng, _SHOPPING_PRODUCT_TEMPLATES, target_count=30, id_prefix="prd")
    products = {}
    product_ids: list[str] = []
    for idx, (_pid, name, category, price, stock, desc) in enumerate(selected):
        unique_pid = _seed_scoped_id("prd", seed, idx, width=3)
        product_ids.append(unique_pid)
        products[unique_pid] = {
            "product_id": unique_pid, "name": name, "category": category,
            "price": price + rng.randint(-5, 5), "stock": stock + rng.randint(0, 3),
            "description": desc,
        }
    order_1 = _seed_scoped_id("ord", seed, 0, width=4)
    order_2 = _seed_scoped_id("ord", seed, 1, width=4)
    order_3 = _seed_scoped_id("ord", seed, 2, width=4)
    order_4 = _seed_scoped_id("ord", seed, 3, width=4)
    order_5 = _seed_scoped_id("ord", seed, 4, width=4)
    order_6 = _seed_scoped_id("ord", seed, 5, width=4)
    order_7 = _seed_scoped_id("ord", seed, 6, width=4)
    return_1 = f"ret_s{seed}_0001"
    return_2 = f"ret_s{seed}_0002"
    return_3 = f"ret_s{seed}_0003"
    return_4 = f"ret_s{seed}_0004"
    return_5 = f"ret_s{seed}_0005"
    order_1_items = [
        {"product_id": pid, "quantity": 1, "unit_price": products[pid]["price"]}
        for pid in product_ids[:2]
    ]
    order_2_items = [
        {"product_id": pid, "quantity": 1, "unit_price": products[pid]["price"]}
        for pid in product_ids[2:3]
    ]
    order_3_items = [
        {"product_id": pid, "quantity": 1, "unit_price": products[pid]["price"]}
        for pid in product_ids[5:7]
    ]
    order_4_items = [
        {"product_id": pid, "quantity": 1, "unit_price": products[pid]["price"]}
        for pid in product_ids[7:9]
    ]
    order_5_items = [
        {"product_id": pid, "quantity": 1, "unit_price": products[pid]["price"]}
        for pid in product_ids[9:10]
    ]
    order_6_items = [
        {"product_id": pid, "quantity": 1, "unit_price": products[pid]["price"]}
        for pid in product_ids[10:11]
    ]
    order_7_items = [
        {"product_id": pid, "quantity": 1, "unit_price": products[pid]["price"]}
        for pid in product_ids[11:12]
    ]
    orders = {
        order_1: {
            "order_id": order_1, "items": order_1_items,
            "product_ids": product_ids[:2],
            "total": round(sum(i["quantity"] * i["unit_price"]
                               for i in order_1_items), 2),
            "status": "shipped",
            "created_at": (reference_date - _datetime.timedelta(days=7)).isoformat(),
        },
        order_2: {
            "order_id": order_2, "items": order_2_items,
            "product_ids": product_ids[2:3],
            "total": round(sum(i["quantity"] * i["unit_price"]
                               for i in order_2_items), 2),
            "status": "pending",
            "created_at": (reference_date - _datetime.timedelta(days=1)).isoformat(),
        },
        order_3: {
            "order_id": order_3, "items": order_3_items,
            "product_ids": product_ids[5:7],
            "total": round(sum(i["quantity"] * i["unit_price"]
                               for i in order_3_items), 2),
            "status": "returning",
            "return_id": return_1,
            "created_at": (reference_date - _datetime.timedelta(days=10)).isoformat(),
        },
        order_4: {
            "order_id": order_4, "items": order_4_items,
            "product_ids": product_ids[7:9],
            "total": round(sum(i["quantity"] * i["unit_price"]
                               for i in order_4_items), 2),
            "status": "returned",
            "return_id": return_2,
            "created_at": (reference_date - _datetime.timedelta(days=14)).isoformat(),
        },
        order_5: {
            "order_id": order_5, "items": order_5_items,
            "product_ids": product_ids[9:10],
            "total": round(sum(i["quantity"] * i["unit_price"]
                               for i in order_5_items), 2),
            "status": "shipped",
            "return_id": return_3,
            "created_at": (reference_date - _datetime.timedelta(days=16)).isoformat(),
        },
        order_6: {
            "order_id": order_6, "items": order_6_items,
            "product_ids": product_ids[10:11],
            "total": round(sum(i["quantity"] * i["unit_price"]
                               for i in order_6_items), 2),
            "status": "shipped",
            "return_id": return_4,
            "created_at": (reference_date - _datetime.timedelta(days=18)).isoformat(),
        },
        order_7: {
            "order_id": order_7, "items": order_7_items,
            "product_ids": product_ids[11:12],
            "total": round(sum(i["quantity"] * i["unit_price"]
                               for i in order_7_items), 2),
            "status": "shipped",
            "return_id": return_5,
            "created_at": (reference_date - _datetime.timedelta(days=20)).isoformat(),
        },
    }
    # Seed a removable target.  Otherwise add -> remove constructs a transient
    # setup mutation whose net result is indistinguishable from the start.
    wishlist = list(product_ids[:2])
    # Seed one review on a shipped-order product so get_reviews chains have
    # material data without requiring an add_review mutation first.
    reviews = {
        product_ids[0]: [{
            "author": "current_user", "product_id": product_ids[0],
            "rating": 4, "title": "Good product",
            "body": "Works well for the price. Would recommend.",
            "date": (reference_date - _datetime.timedelta(days=2)).isoformat(),
        }],
    }
    # Seed cart with two items of different quantities so checkout,
    # update_cart_quantity, and remove_from_cart chains have a direct
    # starting point without every chain needing add_to_cart first.
    cart = [
        {"product_id": product_ids[3], "quantity": 2,
         "unit_price": products[product_ids[3]]["price"]},
        {"product_id": product_ids[4], "quantity": 1,
         "unit_price": products[product_ids[4]]["price"]},
    ]
    # Seed one return on order_3 ("returning") so get_return_status
    # chains have a target without every chain needing return_order first.
    # Also add an "initiated" return so both return statuses are covered.
    returns = {
        return_1: {
            "return_id": return_1,
            "order_id": order_3,
            "reason": "Item not as described",
            "items": [product_ids[5]],
            # Never share mutable line-item dictionaries with the order view.
            "item_details": [dict(order_3_items[0])],
            "status": "initiated",
        },
        return_2: {
            "return_id": return_2,
            "order_id": order_4,
            "reason": "Wrong size",
            "items": [product_ids[7]],
            "item_details": [dict(order_4_items[0])],
            "status": "received",
        },
        return_3: {
            "return_id": return_3,
            "order_id": order_5,
            "reason": "Changed mind",
            "items": [product_ids[9]],
            "item_details": [dict(order_5_items[0])],
            "status": "initiated",
        },
        return_4: {
            "return_id": return_4,
            "order_id": order_6,
            "reason": "Arrived damaged",
            "items": [product_ids[10]],
            "item_details": [dict(order_6_items[0])],
            "status": "received",
        },
        return_5: {
            "return_id": return_5,
            "order_id": order_7,
            "reason": "No longer needed",
            "items": [product_ids[11]],
            "item_details": [dict(order_7_items[0])],
            "status": "initiated",
        },
    }
    return {"products": products, "cart": cart, "orders": orders,
            "next_order_num": 8, "reviews": reviews, "returns": returns,
            "wishlist": wishlist,
            "id_scope": f"s{seed}",
            "current_date": reference_date.isoformat()}
