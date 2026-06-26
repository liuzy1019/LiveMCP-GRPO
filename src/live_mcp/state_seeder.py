"""Deterministic state seeders for MVP servers."""

from __future__ import annotations

import copy
import random
from typing import Any


class StateSeeder:
    def seed_state(self, server_name: str, session_id: str, seed: int) -> dict[str, Any]:
        if server_name == "calendar":
            return _calendar_state(seed)
        if server_name == "shopping":
            return _shopping_state(seed)
        if server_name == "banking":
            return _banking_state(seed)
        if server_name == "email":
            return _email_state(seed)
        if server_name == "filesystem":
            return _filesystem_state(seed)
        if server_name == "payments":
            return _payments_state(seed)
        if server_name == "crm":
            return _crm_state(seed)
        if server_name == "issue_tracker":
            return _issue_tracker_state(seed)
        if server_name == "team_chat":
            return _team_chat_state(seed)
        if server_name == "food_delivery":
            return _food_delivery_state(seed)
        raise ValueError(f"unsupported server: {server_name}")

    def reset_state(self, server_name: str, session_id: str, seed: int) -> dict[str, Any]:
        return copy.deepcopy(self.seed_state(server_name, session_id, seed))


def _calendar_state(seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    events: dict[str, dict[str, Any]] = {}
    base = [
        ("Team Sync", "alex@example.com", "Weekly team sync up"),
        ("Design Review", "sam@example.com", "Review new UI mockups"),
        ("Budget Check", "alex@example.com", "Q2 budget review"),
        ("Customer Call", "alex@example.com", "Onboarding call with new client"),
    ]
    for idx, (title, lead, desc) in enumerate(base, start=1):
        day = 22 + idx; hour = 9 + ((idx + rng.randint(0, 3)) % 6)
        events[f"evt_{idx:03d}"] = {
            "event_id": f"evt_{idx:03d}", "title": title,
            "start_time": f"2026-06-{day:02d}T{hour:02d}:00",
            "end_time": f"2026-06-{day:02d}T{hour + 1:02d}:00",
            "description": desc, "location": "Room " + str(100 + idx),
            "attendees": ["alex@example.com", "sam@example.com"][: 1 + idx % 2],
            "reminders": [], "recurrence": None,
        }
    return {"events": events, "next_event_num": len(events) + 1, "timezone": "America/New_York"}


def _shopping_state(seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    base = [
        ("prd_001", "K3 Keyboard", "keyboard", 79, 5, "Mechanical keyboard with RGB backlight"),
        ("prd_002", "MX Mouse", "mouse", 49, 8, "Ergonomic wireless mouse"),
        ("prd_003", "USB-C Hub", "hub", 35, 4, "7-in-1 USB-C hub with HDMI"),
        ("prd_004", "Noise Canceling Headphones", "audio", 99, 3, "Wireless ANC headphones"),
    ]
    products = {pid: {"product_id": pid, "name": name, "category": category, "price": price + rng.randint(0, 5), "stock": stock, "description": desc} for pid, name, category, price, stock, desc in base}
    return {"products": products, "cart": [], "orders": {}, "next_order_num": 1, "reviews": {}, "wishlist": []}


def _banking_state(seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    accounts = {
        "acc_savings": {"account_id": "acc_savings", "owner": "Alice Johnson", "balance": 25000.00 + rng.randint(-100, 100), "currency": "USD", "type": "savings", "frozen": False, "opened_date": "2018-03-15"},
        "acc_checking": {"account_id": "acc_checking", "owner": "Alice Johnson", "balance": 5000.00 + rng.randint(-50, 50), "currency": "USD", "type": "checking", "frozen": False, "opened_date": "2018-03-15"},
        "acc_business": {"account_id": "acc_business", "owner": "Bob Smith", "balance": 100000.00 + rng.randint(-500, 500), "currency": "USD", "type": "business", "frozen": False, "opened_date": "2020-01-10"},
        "acc_frozen_demo": {"account_id": "acc_frozen_demo", "owner": "Carol White", "balance": 1500.00, "currency": "USD", "type": "savings", "frozen": True, "opened_date": "2022-06-01"},
    }
    return {"accounts": accounts, "transactions": [], "freeze_log": [], "next_txn_num": 1, "scheduled_transfers": {}, "loans": {}}


def _email_state(seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    threads = {"thd_001": ["eml_0001", "eml_0002"], "thd_002": ["eml_0003"]}
    emails = {
        "eml_0001": {"email_id": "eml_0001", "to": "team@example.com", "cc": "", "sender": "boss@example.com", "subject": "Q2 Review", "body": "Please prepare Q2 review slides by Friday.", "labels": ["work", "urgent"], "thread_id": "thd_001", "status": "received", "date": "2026-06-20", "read": False, "attachments": []},
        "eml_0002": {"email_id": "eml_0002", "to": "boss@example.com", "cc": "", "sender": "current_user@example.com", "subject": "Re: Q2 Review", "body": "On it. Will send draft by Thursday.", "labels": ["work"], "thread_id": "thd_001", "status": "sent", "date": "2026-06-20", "read": True, "attachments": []},
        "eml_0003": {"email_id": "eml_0003", "to": "dev@example.com", "cc": "manager@example.com", "sender": "alice@example.com", "subject": "Sprint Planning", "body": "Let's plan the next sprint tomorrow at 10am.", "labels": [], "thread_id": "thd_002", "status": "received", "date": "2026-06-21", "read": False, "attachments": [{"name": "sprint_backlog.pdf", "size": 245000}]},
    }
    return {"emails": emails, "drafts": {}, "threads": threads, "inbox_order": ["eml_0001", "eml_0002", "eml_0003"], "next_email_num": len(emails) + 1, "next_thread_num": len(threads) + 1, "filters": {}}


def _filesystem_state(seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    fs = {
        "/": {"type": "dir", "content": "", "permissions": "755", "owner": "root"},
        "/home": {"type": "dir", "content": "", "permissions": "755", "owner": "root"},
        "/home/user": {"type": "dir", "content": "", "permissions": "700", "owner": "user"},
        "/home/user/notes.txt": {"type": "file", "content": "TODO: review design doc\nTODO: update tests\nDONE: fix login bug", "permissions": "644", "owner": "user"},
        "/home/user/script.sh": {"type": "file", "content": "#!/bin/bash\necho hello", "permissions": "755", "owner": "user"},
        "/home/user/projects": {"type": "dir", "content": "", "permissions": "700", "owner": "user"},
        "/home/user/projects/README.md": {"type": "file", "content": "# Projects\nWork in progress.", "permissions": "644", "owner": "user"},
        "/home/user/projects/config.ini": {"type": "file", "content": "[server]\nhost=localhost\nport=8080\n[database]\nname=proddb", "permissions": "644", "owner": "user"},
        "/protected": {"type": "dir", "content": "", "permissions": "700", "owner": "root"},
        "/protected/config.secret": {"type": "file", "content": "secret_key=abc123\ndb_password=xyz789", "permissions": "600", "owner": "root"},
    }
    return {"fs": fs, "cwd": "/home/user", "umask": "022"}


def _payments_state(seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    invoices = {
        "inv_0001": {"invoice_id": "inv_0001", "customer": "Acme Corp", "amount": 1500.00, "currency": "USD", "description": "Consulting Q2", "status": "pending", "payment_id": None, "refund_id": None, "due_date": "2026-07-15", "created_at": "2026-06-01"},
        "inv_0002": {"invoice_id": "inv_0002", "customer": "Globex Inc", "amount": 3200.00, "currency": "USD", "description": "Software license", "status": "pending", "payment_id": None, "refund_id": None, "due_date": "2026-07-20", "created_at": "2026-06-05"},
        "inv_0003": {"invoice_id": "inv_0003", "customer": "Acme Corp", "amount": 500.00, "currency": "EUR", "description": "Support retainer", "status": "paid", "payment_id": "pay_0001", "refund_id": None, "due_date": "2026-06-30", "created_at": "2026-06-01"},
    }
    return {"invoices": invoices, "payments": {"pay_0001": {"payment_id": "pay_0001", "invoice_id": "inv_0003", "amount": 500.00, "method": "wire", "status": "settled"}}, "refunds": {}, "webhooks": {}, "disputes": {}, "next_inv_num": len(invoices) + 1, "next_pay_num": 2, "next_ref_num": 1, "next_wh_num": 1}


def _crm_state(seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    leads = {
        "lead_0001": {"lead_id": "lead_0001", "name": "Charlie Chen", "company": "TechStars", "source": "conference", "email": "charlie@techstars.com", "phone": "555-0101", "status": "new", "contact_id": None},
        "lead_0002": {"lead_id": "lead_0002", "name": "Dana Davis", "company": "DataFlow", "source": "webinar", "email": "dana@dataflow.io", "phone": "555-0102", "status": "new", "contact_id": None},
        "lead_0003": {"lead_id": "lead_0003", "name": "Evan Ellis", "company": "CloudBase", "source": "referral", "email": "evan@cloudbase.com", "phone": "555-0100", "status": "converted", "contact_id": "contact_0001"},
    }
    contacts = {"contact_0001": {"contact_id": "contact_0001", "name": "Evan Ellis", "email": "evan@cloudbase.com", "phone": "555-0100", "company": "CloudBase", "lead_id": "lead_0003"}}
    deals = {
        "deal_0001": {"deal_id": "deal_0001", "name": "Cloud Migration", "amount": 50000.00, "stage": "prospecting", "contact_id": None, "lead_id": "lead_0001", "created_at": "2026-06-10"},
        "deal_0002": {"deal_id": "deal_0002", "name": "Data Pipeline", "amount": 75000.00, "stage": "proposal", "contact_id": "contact_0001", "lead_id": "lead_0003", "created_at": "2026-06-15"},
    }
    return {"leads": leads, "contacts": contacts, "deals": deals, "tasks": {}, "notes": {}, "next_lead_num": len(leads) + 1, "next_contact_num": len(contacts) + 1, "next_deal_num": len(deals) + 1, "next_task_num": 1, "next_note_num": 1}


def _issue_tracker_state(seed: int) -> dict[str, Any]:
    members = {
        "alice": {"name": "Alice", "role": "developer"},
        "bob": {"name": "Bob", "role": "senior developer"},
        "charlie": {"name": "Charlie", "role": "tech lead"},
        "dana": {"name": "Dana", "role": "designer"},
    }
    issues = {
        "iss_0001": {"issue_id": "iss_0001", "title": "Login timeout on mobile", "description": "Users report 30s timeout on iOS app login.", "priority": "high", "labels": ["bug", "mobile"], "state": "open", "assignee": None, "watchers": [], "sprint_id": None, "milestone": None, "comments": [], "created_at": "2026-06-20"},
        "iss_0002": {"issue_id": "iss_0002", "title": "Add dark mode support", "description": "Feature request for dark mode in settings.", "priority": "medium", "labels": ["feature"], "state": "in_progress", "assignee": "bob", "watchers": ["alice"], "sprint_id": "spr_0001", "milestone": "v2.5", "comments": [{"author": "user", "body": "Started working on this.", "timestamp": "2026-06-22"}], "created_at": "2026-06-21"},
        "iss_0003": {"issue_id": "iss_0003", "title": "Fix PDF export layout", "description": "Tables are misaligned in PDF export.", "priority": "high", "labels": ["bug", "pdf"], "state": "in_review", "assignee": "alice", "watchers": ["bob", "dana"], "sprint_id": "spr_0001", "milestone": "v2.5", "comments": [{"author": "user", "body": "Fixed the layout issues.", "timestamp": "2026-06-23"}], "created_at": "2026-06-22"},
        "iss_0004": {"issue_id": "iss_0004", "title": "Update dependencies", "description": "Security audit flagged outdated packages.", "priority": "medium", "labels": ["maintenance"], "state": "resolved", "assignee": "charlie", "watchers": [], "sprint_id": "spr_0001", "milestone": "v2.5", "comments": [], "created_at": "2026-06-18"},
    }
    sprints = {"spr_0001": {"sprint_id": "spr_0001", "name": "Sprint 24", "start_date": "2026-06-15", "end_date": "2026-06-29", "goal": "Bug fixes and dark mode", "status": "active", "issues": ["iss_0002", "iss_0003", "iss_0004"]}}
    return {"issues": issues, "members": members, "sprints": sprints, "subtasks": {}, "time_entries": [], "next_issue_num": len(issues) + 1, "next_sprint_num": len(sprints) + 1, "next_subtask_num": 1, "next_time_entry_num": 1}


def _team_chat_state(seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    channels = {
        "ch_general": {"channel_id": "ch_general", "name": "general", "description": "General discussion", "members": ["alice", "bob", "charlie"], "archived": False, "messages": [
            {"message_id": "msg_0001", "channel_id": "ch_general", "content": "Welcome everyone!", "author": "alice", "thread_id": None, "reactions": ["wave"], "timestamp": "2026-06-20T09:00:00"},
            {"message_id": "msg_0002", "channel_id": "ch_general", "content": "Thanks Alice!", "author": "bob", "thread_id": None, "reactions": [], "timestamp": "2026-06-20T09:05:00"},
        ]},
        "ch_engineering": {"channel_id": "ch_engineering", "name": "engineering", "description": "Engineering team", "members": ["bob", "charlie", "dana"], "archived": False, "messages": [
            {"message_id": "msg_0003", "channel_id": "ch_engineering", "content": "Deploy scheduled for 6pm", "author": "charlie", "thread_id": None, "reactions": ["rocket"], "timestamp": "2026-06-23T15:00:00"},
        ]},
        "ch_archived": {"channel_id": "ch_archived", "name": "old-project", "description": "Archived project channel", "members": ["alice"], "archived": True, "messages": []},
    }
    return {"channels": channels, "threads": {}, "dms": [], "next_msg_num": 4, "next_thread_num": 1}


def _food_delivery_state(seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    restaurants = {
        "rest_001": {"restaurant_id": "rest_001", "name": "Pizza Palace", "cuisine": "Italian", "rating": 4.5, "delivery_fee": 2.99, "open": True, "hours": "11:00-22:00",
            "menu": [{"name": "Margherita Pizza", "price": 12.99, "dietary_tags": ["vegetarian"], "popularity": 95}, {"name": "Pepperoni Pizza", "price": 14.99, "dietary_tags": [], "popularity": 88}, {"name": "Caesar Salad", "price": 8.99, "dietary_tags": ["vegetarian", "gluten-free"], "popularity": 60}, {"name": "Garlic Bread", "price": 4.99, "dietary_tags": ["vegetarian"], "popularity": 72}]},
        "rest_002": {"restaurant_id": "rest_002", "name": "Sushi Express", "cuisine": "Japanese", "rating": 4.8, "delivery_fee": 3.99, "open": True, "hours": "12:00-21:00",
            "menu": [{"name": "California Roll", "price": 10.99, "dietary_tags": [], "popularity": 80}, {"name": "Salmon Nigiri", "price": 12.99, "dietary_tags": ["gluten-free"], "popularity": 75}, {"name": "Miso Soup", "price": 3.99, "dietary_tags": ["vegetarian", "gluten-free"], "popularity": 50}, {"name": "Edamame", "price": 4.99, "dietary_tags": ["vegan", "gluten-free"], "popularity": 65}]},
        "rest_003": {"restaurant_id": "rest_003", "name": "Burger Barn", "cuisine": "American", "rating": 4.2, "delivery_fee": 1.99, "open": True, "hours": "10:00-23:00",
            "menu": [{"name": "Classic Burger", "price": 9.99, "dietary_tags": [], "popularity": 90}, {"name": "Cheese Burger", "price": 11.99, "dietary_tags": [], "popularity": 85}, {"name": "French Fries", "price": 3.99, "dietary_tags": ["vegetarian", "gluten-free"], "popularity": 70}, {"name": "Milkshake", "price": 5.99, "dietary_tags": ["vegetarian"], "popularity": 55}]},
    }
    orders = {
        "ord_0001": {"order_id": "ord_0001", "restaurant_id": "rest_001", "restaurant_name": "Pizza Palace", "items": [{"name": "Margherita Pizza", "quantity": 2}, {"name": "Garlic Bread", "quantity": 1}], "delivery_address": "123 Main St", "special_instructions": "", "subtotal": 30.97, "delivery_fee": 2.99, "tip": 3.00, "total": 36.96, "status": "delivered", "rating": None, "created_at": "2026-06-20T18:00:00"},
        "ord_0002": {"order_id": "ord_0002", "restaurant_id": "rest_002", "restaurant_name": "Sushi Express", "items": [{"name": "California Roll", "quantity": 1}], "delivery_address": "456 Oak Ave", "special_instructions": "", "subtotal": 10.99, "delivery_fee": 3.99, "tip": 0, "total": 14.98, "status": "preparing", "rating": None, "created_at": "2026-06-21T12:30:00"},
    }
    return {"restaurants": restaurants, "orders": orders, "support_tickets": [], "next_order_num": len(orders) + 1, "next_ticket_num": 1}
