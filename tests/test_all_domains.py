#!/usr/bin/env python3
"""Comprehensive domain tool test — all 10 domains, all tools.

Uses InProcessTransport (no subprocess overhead) to test every tool in every
domain with valid arguments. Reports pass/fail per tool.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.live_mcp.transport import InProcessTransport
from src.live_mcp.server_base import StatefulToolServer


# ── Per-domain test spec: (server_class, valid_arg_generators) ──
# Each entry: tool_name → (arguments_dict, description)


def _make_calendar_tests(server: StatefulToolServer) -> list[tuple[str, dict, str]]:
    """Calendar domain test cases."""
    state = server.seeder.seed_state("calendar", "test", 42)
    evt_ids = list(state["events"].keys())
    evt0 = evt_ids[0]
    return [
        ("get_working_hours", {}, "get working hours"),
        ("list_events", {}, "list all events"),
        ("search_events", {"query": "team"}, "search by keyword"),
        ("get_event", {"event_id": evt0}, "get event by id"),
        ("create_event", {"title": "Test Event", "start_time": "2026-06-25T10:00", "end_time": "2026-06-25T11:00", "description": "test"}, "create event"),
        ("update_event", {"event_id": evt0, "fields": {"title": "Updated Title"}}, "update event title"),
        ("add_attendee", {"event_id": evt0, "email": "test@example.com"}, "add attendee"),
        ("remove_attendee", {"event_id": evt0, "email": "test@example.com"}, "remove attendee"),
        ("get_free_busy", {"emails": ["alex@example.com"], "start_time": "2026-06-20T00:00", "end_time": "2026-06-30T00:00"}, "get free/busy"),
        ("check_conflicts", {"start_time": "2026-06-25T10:00", "end_time": "2026-06-25T11:00"}, "check conflicts"),
        ("set_reminder", {"event_id": evt0, "minutes_before": 15}, "set reminder"),
        ("create_recurring", {"title": "Weekly", "start_time": "2026-06-25T09:00", "end_time": "2026-06-25T10:00", "recurrence": "weekly"}, "create recurring"),
        ("respond_to_event", {"event_id": evt0, "email": "alex@example.com", "response": "accepted"}, "respond to event"),
        ("export_calendar", {"format": "json"}, "export calendar"),
        ("change_timezone", {"timezone": "Asia/Shanghai"}, "change timezone"),
        ("delete_event", {"event_id": evt0}, "delete event"),
    ]


def _make_shopping_tests(server: StatefulToolServer) -> list[tuple[str, dict, str]]:
    """Shopping domain test cases."""
    state = server.seeder.seed_state("shopping", "test", 42)
    prod_ids = list(state["products"].keys())
    p0 = prod_ids[0]
    return [
        ("search_products", {"query": "keyboard"}, "search products"),
        ("get_product", {"product_id": p0}, "get product"),
        ("list_categories", {}, "list categories"),
        ("compare_products", {"product_ids": [p0, prod_ids[1]]}, "compare products"),
        ("get_recommendations", {"category": "keyboard", "limit": 2}, "get recommendations"),
        ("add_to_cart", {"product_id": p0, "quantity": 1}, "add to cart"),
        ("get_cart", {}, "get cart"),
        ("update_cart_quantity", {"product_id": p0, "quantity": 2}, "update cart qty"),
        ("remove_from_cart", {"product_id": p0}, "remove from cart"),
        # Re-add for checkout test
        ("add_to_cart", {"product_id": p0, "quantity": 1}, "re-add to cart"),
        ("apply_coupon", {"code": "SAVE10"}, "apply coupon"),
        ("get_coupons", {}, "get coupons"),
        ("checkout", {"shipping_address": "123 Main St"}, "checkout"),
        ("add_to_wishlist", {"product_id": prod_ids[2]}, "add to wishlist"),
        ("get_wishlist", {}, "get wishlist"),
        ("remove_from_wishlist", {"product_id": prod_ids[2]}, "remove from wishlist"),
        ("clear_cart", {}, "clear cart"),
    ]


def _make_banking_tests(server: StatefulToolServer) -> list[tuple[str, dict, str]]:
    """Banking domain test cases."""
    state = server.seeder.seed_state("banking", "test", 42)
    acct_ids = list(state["accounts"].keys())
    a0, a1 = acct_ids[0], acct_ids[1]
    return [
        ("list_accounts", {}, "list accounts"),
        ("get_account_info", {"account_id": a0}, "get account info"),
        ("get_balance", {"account_id": a0}, "get balance"),
        ("get_history", {"account_id": a0, "limit": 5}, "get history"),
        ("get_statement", {"account_id": a0, "year": 2026, "month": 6}, "get statement"),
        ("verify_account", {"account_id": a0, "owner_name": "Alice Johnson"}, "verify account"),
        ("get_exchange_rate", {"from_currency": "USD", "to_currency": "EUR"}, "get exchange rate"),
        ("transfer", {"from_account": a0, "to_account": a1, "amount": 100.0}, "transfer"),
        ("schedule_transfer", {"from_account": a0, "to_account": a1, "amount": 50.0, "execute_date": "2026-07-01"}, "schedule transfer"),
        ("deposit", {"account_id": a0, "amount": 500.0}, "deposit"),
        ("withdraw", {"account_id": a0, "amount": 100.0}, "withdraw"),
        ("bill_pay", {"account_id": a0, "payee": "Electric Co", "amount": 85.0}, "pay bill"),
        ("apply_loan", {"account_id": a0, "amount": 10000.0, "term_months": 12, "purpose": "home"}, "apply loan"),
    ]


def _make_email_tests(server: StatefulToolServer) -> list[tuple[str, dict, str]]:
    """Email domain test cases."""
    state = server.seeder.seed_state("email", "test", 42)
    email_ids = list(state["emails"].keys())
    e0 = email_ids[0]
    return [
        ("list_inbox", {}, "list inbox"),
        ("search_emails", {"keyword": "sprint"}, "search emails"),
        ("get_email", {"email_id": e0}, "get email"),
        ("get_attachments", {"email_id": e0}, "get attachments"),
        ("mark_read", {"email_id": e0}, "mark read"),
        ("mark_unread", {"email_id": e0}, "mark unread"),
        ("add_label", {"email_id": e0, "label": "important"}, "add label"),
        ("remove_label", {"email_id": e0, "label": "important"}, "remove label"),
        ("get_thread", {"thread_id": "thd_001"}, "get thread"),
        ("create_draft", {"to": "test@example.com", "subject": "Test Draft", "body": "Hello"}, "create draft"),
        ("create_filter", {"field": "subject", "pattern": "test", "action": "label", "label": "filtered"}, "create filter"),
        ("list_filters", {}, "list filters"),
        ("send_email", {"to": "colleague@example.com", "subject": "Test", "body": "Testing"}, "send email"),
        ("reply_email", {"email_id": e0, "body": "Got it, thanks!"}, "reply email"),
        ("forward_email", {"email_id": e0, "to": "forward@example.com"}, "forward email"),
        ("archive_email", {"email_id": e0}, "archive email"),
    ]


def _make_filesystem_tests(server: StatefulToolServer) -> list[tuple[str, dict, str]]:
    """Filesystem domain test cases."""
    return [
        ("pwd", {}, "print working directory"),
        ("ls", {"path": "/home/user"}, "list directory"),
        ("ls", {"path": "/"}, "list root"),
        ("cat", {"path": "/home/user/notes.txt"}, "read file"),
        ("head", {"path": "/home/user/notes.txt", "lines": 2}, "read head"),
        ("tail", {"path": "/home/user/notes.txt", "lines": 2}, "read tail"),
        ("wc", {"path": "/home/user/notes.txt"}, "word count"),
        ("stat", {"path": "/home/user/notes.txt"}, "file stat"),
        ("find", {"path": "/home/user", "pattern": "*.txt"}, "find files"),
        ("grep", {"path": "/home/user", "pattern": "TODO"}, "grep content"),
        ("tree", {"path": "/home/user", "max_depth": 2}, "tree view"),
        ("cd", {"path": "/home/user/projects"}, "change directory"),
        ("cd", {"path": "/home/user"}, "change back"),
        ("mkdir", {"path": "/home/user/test_dir"}, "create directory"),
        ("touch", {"path": "/home/user/test_file.txt"}, "create file"),
        ("du", {"path": "/home/user"}, "disk usage"),
        ("df", {}, "disk free"),
        ("file_info", {"path": "/home/user/notes.txt"}, "file info"),
        ("md5sum", {"path": "/home/user/notes.txt"}, "md5 checksum"),
        ("sha256sum", {"path": "/home/user/notes.txt"}, "sha256 checksum"),
        ("chmod", {"path": "/home/user/test_file.txt", "mode": "755"}, "change permissions"),
        ("sort", {"path": "/home/user/notes.txt"}, "sort file"),
        ("uniq", {"path": "/home/user/notes.txt"}, "unique lines"),
        ("cut", {"path": "/home/user/notes.txt", "delimiter": ":", "fields": "1"}, "cut fields"),
        ("sed", {"path": "/home/user/notes.txt", "expression": "s/TODO/DONE/g"}, "sed replace"),
        ("awk", {"path": "/home/user/notes.txt", "script": '{print $1}'}, "awk script"),
        ("xxd", {"path": "/home/user/notes.txt", "limit": 64}, "hex dump"),
        ("truncate", {"path": "/home/user/test_file.txt", "size": 0}, "truncate file"),
        ("mv", {"source": "/home/user/test_file.txt", "target": "/home/user/moved_file.txt"}, "move file"),
        ("cp", {"source": "/home/user/notes.txt", "target": "/home/user/notes_copy.txt"}, "copy file"),
        ("symlink", {"target": "/home/user/notes.txt", "link_path": "/home/user/link_to_notes"}, "create symlink"),
        ("readlink", {"path": "/home/user/link_to_notes"}, "read symlink"),
        ("split", {"path": "/home/user/notes.txt"}, "split file"),
        ("diff", {"file1": "/home/user/notes.txt", "file2": "/home/user/notes_copy.txt"}, "diff files"),
        ("rm", {"path": "/home/user/test_dir"}, "remove directory"),
        ("rm", {"path": "/home/user/moved_file.txt"}, "remove file"),
        ("rm", {"path": "/home/user/notes_copy.txt"}, "remove copy"),
        ("rm", {"path": "/home/user/link_to_notes"}, "remove symlink"),
    ]


def _make_payments_tests(server: StatefulToolServer) -> list[tuple[str, dict, str]]:
    """Payments domain test cases."""
    state = server.seeder.seed_state("payments", "test", 42)
    inv_ids = list(state["invoices"].keys())
    i0 = inv_ids[0]
    return [
        ("list_invoices", {}, "list invoices"),
        ("get_invoice", {"invoice_id": i0}, "get invoice"),
        ("create_invoice", {"customer": "Test Corp", "amount": 999.99, "description": "Test invoice"}, "create invoice"),
        ("pay_invoice", {"invoice_id": i0, "amount": 1500.00, "method": "card"}, "pay invoice"),
        ("create_webhook", {"url": "https://example.com/hook", "events": ["payment.created"]}, "create webhook"),
        ("list_webhooks", {}, "list webhooks"),
    ]


def _make_crm_tests(server: StatefulToolServer) -> list[tuple[str, dict, str]]:
    """CRM domain test cases."""
    state = server.seeder.seed_state("crm", "test", 42)
    lead_ids = list(state["leads"].keys())
    l0 = lead_ids[0]
    return [
        ("list_leads", {}, "list leads"),
        ("list_deals", {}, "list deals"),
        ("list_tasks", {}, "list tasks"),
        ("get_deal", {"deal_id": "deal_0001"}, "get deal"),
        ("create_lead", {"name": "Test Lead", "company": "TestCo", "source": "web", "email": "test@testco.com"}, "create lead"),
        ("update_lead", {"lead_id": l0, "fields": {"phone": "555-9999"}}, "update lead"),
        ("create_contact", {"name": "Direct Contact", "email": "direct@example.com"}, "create contact"),
        ("create_deal", {"name": "New Deal", "amount": 5000.0, "stage": "prospecting"}, "create deal"),
        ("update_deal", {"deal_id": "deal_0001", "stage": "qualification"}, "update deal stage"),
        ("create_task", {"title": "Follow up", "deal_id": "deal_0001", "priority": "high"}, "create task"),
        ("add_note", {"entity_type": "deal", "entity_id": "deal_0001", "content": "Great progress"}, "add note"),
        ("complete_task", {"task_id": "task_0001"}, "complete task"),  # task created above
    ]


def _make_issue_tracker_tests(server: StatefulToolServer) -> list[tuple[str, dict, str]]:
    """Issue tracker test cases."""
    state = server.seeder.seed_state("issue_tracker", "test", 42)
    iss_ids = list(state["issues"].keys())
    i0 = iss_ids[0]
    return [
        ("list_issues", {}, "list issues"),
        ("get_issue", {"issue_id": i0}, "get issue"),
        ("list_sprints", {}, "list sprints"),
        ("get_time_report", {}, "get time report"),
        ("create_issue", {"title": "Test Bug", "priority": "high", "labels": ["bug"]}, "create issue"),
        ("update_issue", {"issue_id": i0, "fields": {"title": "Updated Title"}}, "update issue"),
        ("assign_issue", {"issue_id": i0, "assignee": "alice"}, "assign issue"),
        ("comment_issue", {"issue_id": i0, "body": "Looking into this"}, "comment issue"),
        ("add_label", {"issue_id": i0, "label": "frontend"}, "add issue label"),
        ("remove_label", {"issue_id": i0, "label": "frontend"}, "remove issue label"),
        ("add_watcher", {"issue_id": i0, "user": "bob"}, "add watcher"),
        ("remove_watcher", {"issue_id": i0, "user": "bob"}, "remove watcher"),
        ("create_sprint", {"name": "Sprint Test", "start_date": "2026-07-01", "end_date": "2026-07-14", "goal": "Testing"}, "create sprint"),
        ("add_to_sprint", {"issue_id": i0, "sprint_id": "spr_0001"}, "add to sprint"),
        ("remove_from_sprint", {"issue_id": i0}, "remove from sprint"),
        ("create_subtask", {"issue_id": i0, "title": "Investigate root cause"}, "create subtask"),
        ("list_subtasks", {"issue_id": i0}, "list subtasks"),
        ("time_track", {"issue_id": i0, "hours": 2.5, "description": "Debugging"}, "track time"),
        ("set_milestone", {"issue_id": i0, "milestone": "v3.0"}, "set milestone"),
        ("transition_issue", {"issue_id": i0, "state": "in_progress"}, "transition issue"),
    ]


def _make_team_chat_tests(server: StatefulToolServer) -> list[tuple[str, dict, str]]:
    """Team chat test cases."""
    state = server.seeder.seed_state("team_chat", "test", 42)
    return [
        ("list_channels", {}, "list channels"),
        ("get_channel", {"channel_id": "ch_general"}, "get channel"),
        ("get_user_status", {}, "get user status"),
        ("search_messages", {"query": "welcome"}, "search messages"),
        ("send_message", {"channel_id": "ch_general", "content": "Hello from test!"}, "send message"),
        ("create_channel", {"name": "test-channel", "members": ["alice", "bob"]}, "create channel"),
        ("react_message", {"message_id": "msg_0001", "channel_id": "ch_general", "reaction": "thumbsup"}, "add reaction"),
        ("create_thread", {"message_id": "msg_0001", "channel_id": "ch_general"}, "create thread"),
        ("send_dm", {"recipient": "alice", "content": "Hi Alice, testing DMs"}, "send DM"),
        ("archive_channel", {"channel_id": "ch_general"}, "archive channel"),
    ]


def _make_food_delivery_tests(server: StatefulToolServer) -> list[tuple[str, dict, str]]:
    """Food delivery test cases."""
    state = server.seeder.seed_state("food_delivery", "test", 42)
    rest_ids = list(state["restaurants"].keys())
    r0 = rest_ids[0]
    return [
        ("list_restaurants", {}, "list restaurants"),
        ("search_restaurants", {"query": "pizza"}, "search restaurants"),
        ("get_restaurant", {"restaurant_id": r0}, "get restaurant"),
        ("get_menu", {"restaurant_id": r0}, "get menu"),
        ("filter_by_dietary", {"restaurant_id": r0, "dietary": "vegetarian"}, "filter by dietary"),
        ("get_popular_items", {"restaurant_id": r0, "limit": 3}, "get popular"),
        ("create_order", {"restaurant_id": r0, "items": [{"name": "Margherita Pizza", "quantity": 1}], "delivery_address": "123 Main St"}, "create order"),
        ("get_order", {"order_id": "ord_0001"}, "get order"),
        ("list_orders", {}, "list orders"),
        ("get_estimated_time", {"order_id": "ord_0001"}, "get ETA"),
        ("add_tip", {"order_id": "ord_0001", "amount": 5.0}, "add tip"),
        ("contact_support", {"order_id": "ord_0001", "issue_type": "missing_item", "description": "Missing drink"}, "contact support"),
    ]


# ── Domain registry ──
DOMAIN_SPECS: dict[str, tuple[type, callable]] = {}


def _register_domains():
    from src.live_mcp.servers.calendar.server import CalendarServer
    from src.live_mcp.servers.shopping.server import ShoppingServer
    from src.live_mcp.servers.banking.server import BankingServer
    from src.live_mcp.servers.email.server import EmailServer
    from src.live_mcp.servers.filesystem.server import FilesystemServer
    from src.live_mcp.servers.payments.server import PaymentsServer
    from src.live_mcp.servers.crm.server import CRMServer
    from src.live_mcp.servers.issue_tracker.server import IssueTrackerServer
    from src.live_mcp.servers.team_chat.server import TeamChatServer
    from src.live_mcp.servers.food_delivery.server import FoodDeliveryServer

    DOMAIN_SPECS.update({
        "calendar":       (CalendarServer, _make_calendar_tests),
        "shopping":       (ShoppingServer, _make_shopping_tests),
        "banking":        (BankingServer, _make_banking_tests),
        "email":          (EmailServer, _make_email_tests),
        "filesystem":     (FilesystemServer, _make_filesystem_tests),
        "payments":       (PaymentsServer, _make_payments_tests),
        "crm":            (CRMServer, _make_crm_tests),
        "issue_tracker":  (IssueTrackerServer, _make_issue_tracker_tests),
        "team_chat":      (TeamChatServer, _make_team_chat_tests),
        "food_delivery":  (FoodDeliveryServer, _make_food_delivery_tests),
    })


def test_domain(domain: str) -> tuple[int, int, list[str]]:
    """Test all tools in a domain. Returns (passed, total, failures)."""
    server_cls, test_maker = DOMAIN_SPECS[domain]
    server = server_cls()
    transport = InProcessTransport(server)
    transport.start()

    session_id = f"test_{domain}_001"

    try:
        # Reset session
        transport.request("session/reset", {"session_id": session_id, "seed": 42}, timeout_s=5)

        # Discover tools
        result = transport.request("tools/list", {"session_id": session_id}, timeout_s=5)
        registered_tools = {t["name"] for t in result.get("tools", [])}

        # Build test cases
        tests = test_maker(server)

        passed = 0
        failures: list[str] = []

        for tool_name, args, desc in tests:
            if tool_name not in registered_tools:
                failures.append(f"  ✗ {tool_name} ({desc}): NOT REGISTERED")
                continue

            try:
                result = transport.request(
                    "tools/call",
                    {"session_id": session_id, "name": tool_name, "arguments": args},
                    timeout_s=5,
                )
                success = result.get("success", False)
                if success:
                    passed += 1
                else:
                    err = result.get("error_message", "unknown error")
                    failures.append(f"  ✗ {tool_name} ({desc}): {err}")
            except Exception as e:
                failures.append(f"  ✗ {tool_name} ({desc}): exception: {e}")

        total = len(tests)
        return passed, total, failures

    finally:
        transport.stop()


def main():
    _register_domains()

    print("=" * 70)
    print("Live MCP Domain Tool Test — All 10 Domains")
    print("=" * 70)

    all_passed = 0
    all_total = 0
    domain_results: list[tuple[str, int, int, list[str]]] = []

    for domain in sorted(DOMAIN_SPECS.keys()):
        print(f"\n── {domain} ", end="", flush=True)
        passed, total, failures = test_domain(domain)
        all_passed += passed
        all_total += total
        domain_results.append((domain, passed, total, failures))

        if failures:
            print(f"[{passed}/{total}] FAILED")
            for f in failures:
                print(f)
        else:
            print(f"[{passed}/{total}] PASSED")

    # Summary
    print("\n" + "=" * 70)
    print(f"SUMMARY: {all_passed}/{all_total} tools passed across {len(DOMAIN_SPECS)} domains")
    print("=" * 70)

    for domain, passed, total, failures in domain_results:
        status = "✓" if not failures else "✗"
        print(f"  {status} {domain:20s} {passed:3d}/{total:3d}")
    print()

    if all_passed < all_total:
        print(f"FAILED: {all_total - all_passed} tool(s) have errors")
        sys.exit(1)
    else:
        print("ALL TOOLS PASSED")
        sys.exit(0)


if __name__ == "__main__":
    main()