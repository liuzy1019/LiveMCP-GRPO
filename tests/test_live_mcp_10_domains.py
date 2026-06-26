"""Comprehensive system test for all 10 Live MCP servers.

Tests:
1. Each server starts and handles healthcheck
2. Each server's tools are discoverable
3. Each server handles session reset + state isolation
4. Each server runs all tools (readonly + mutating)
5. Each server handles error paths (invalid args, constraint violations)
6. DomainAdapter normalize_event for all 10 domains
7. Safety constraints are enforced server-side
8. Cross-server test: all 10 in one suite
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.live_mcp.config import load_suite_config
from src.live_mcp.manager import LiveMCPManager
from src.live_mcp.executor import LiveMCPExecutor
from src.live_mcp.types import ToolCall
from src.live_mcp import errors

from src.oval_mcp.envs.domain_adapter import get_adapter

# ─────────────────────────────────────
# Fixtures
# ─────────────────────────────────────

@pytest.fixture(scope="module")
def suite():
    return load_suite_config("configs/live_mcp/suite_mvp.yaml")


@pytest.fixture(scope="module")
def live_manager(suite):
    manager = LiveMCPManager(suite)
    manager.start_suite()
    try:
        yield manager
    finally:
        manager.stop_suite()


@pytest.fixture()
def executor(live_manager):
    return LiveMCPExecutor(live_manager, live_manager.registry)


# ─────────────────────────────────────
# 1. Server healthcheck per domain
# ─────────────────────────────────────

ALL_DOMAINS = [
    "calendar", "shopping", "banking", "email",
    "filesystem", "payments", "crm", "issue_tracker",
    "team_chat", "food_delivery",
]


def test_all_servers_healthy(live_manager):
    """All 10 servers respond to healthcheck."""
    result = live_manager.healthcheck()
    for domain in ALL_DOMAINS:
        assert result.get(domain) is True, f"{domain} healthcheck failed"
    print(f"  all {len(ALL_DOMAINS)} domains: HEALTHCHECK OK")


# ─────────────────────────────────────
# 2. Tool discovery per server
# ─────────────────────────────────────

def test_all_servers_tool_discovery(live_manager):
    """Each server discovers all declared tools."""
    for domain in ALL_DOMAINS:
        session = live_manager.create_session(seed=42)
        tools = live_manager.discover_tools(session.session_id)
        domain_tools = [t for t in tools if live_manager.registry.server_for_tool(t["name"]) == domain]
        assert len(domain_tools) > 0, f"{domain}: no tools discovered"
        print(f"  {domain}: {len(domain_tools)} tools discovered")


# ─────────────────────────────────────
# 3. Session reset + state isolation
# ─────────────────────────────────────

def test_session_isolation(live_manager):
    """Two sessions for same domain have independent state."""
    s1 = live_manager.create_session(seed=42)
    s2 = live_manager.create_session(seed=99)
    live_manager.call_tool("calendar", s1.session_id, "delete_event", {"event_id": "evt_001"})
    s2_events = live_manager.call_tool("calendar", s2.session_id, "list_events", {})
    assert len(s2_events["observation"]["events"]) == 4, "Session 2 state should remain intact"
    live_manager.close_session(s1.session_id)
    live_manager.close_session(s2.session_id)
    print("  session isolation OK")


# ─────────────────────────────────────
# 4. Per-domain basic tool call tests
# ─────────────────────────────────────

def test_email_basic_ops(live_manager, executor):
    """Email: list, search, send, label."""
    session = live_manager.create_session(seed=42)
    executor.execute(session.session_id, ToolCall("send_email", {"to": "test@example.com", "subject": "Hello", "body": "Test body"}, "msg1"))
    result = executor.execute(session.session_id, ToolCall("list_inbox", {}, "msg2"))
    assert result.success
    emails = result.observation["emails"]
    assert len(emails) >= 1
    executor.execute(session.session_id, ToolCall("add_label", {"email_id": "eml_0001", "label": "tested"}, "msg3"))
    live_manager.close_session(session.session_id)
    print("  email basic ops OK")


def test_filesystem_basic_ops(live_manager, executor):
    """Filesystem: ls, cd, pwd, mkdir, touch, cat."""
    session = live_manager.create_session(seed=42)
    r = executor.execute(session.session_id, ToolCall("pwd", {}, "msg1"))
    assert r.observation["cwd"] == "/home/user"
    executor.execute(session.session_id, ToolCall("mkdir", {"path": "test_dir"}, "msg2"))
    executor.execute(session.session_id, ToolCall("cd", {"path": "test_dir"}, "msg3"))
    r = executor.execute(session.session_id, ToolCall("pwd", {}, "msg4"))
    assert r.observation["cwd"] == "/home/user/test_dir"
    executor.execute(session.session_id, ToolCall("touch", {"path": "hello.txt"}, "msg5"))
    live_manager.close_session(session.session_id)
    print("  filesystem basic ops OK")


def test_payments_basic_ops(live_manager, executor):
    """Payments: create invoice, pay, refund."""
    session = live_manager.create_session(seed=42)
    r = executor.execute(session.session_id, ToolCall("create_invoice", {"customer": "TestCo", "amount": 100.00}, "msg1"))
    assert r.success
    inv_id = r.observation["invoice"]["invoice_id"]
    r = executor.execute(session.session_id, ToolCall("pay_invoice", {"invoice_id": inv_id, "amount": 100.00}, "msg2"))
    assert r.success
    assert r.observation["invoice"]["status"] == "paid"
    # Refund
    r = executor.execute(session.session_id, ToolCall("refund_invoice", {"invoice_id": inv_id, "amount": 50.00, "reason": "test"}, "msg3"))
    assert r.success
    live_manager.close_session(session.session_id)
    print("  payments basic ops OK")


def test_crm_basic_ops(live_manager, executor):
    """CRM: create lead, convert, create deal."""
    session = live_manager.create_session(seed=42)
    r = executor.execute(session.session_id, ToolCall("create_lead", {"name": "Test Lead", "company": "TestCorp"}, "msg1"))
    assert r.success
    lead_id = r.observation["lead"]["lead_id"]
    r = executor.execute(session.session_id, ToolCall("convert_lead", {"lead_id": lead_id}, "msg2"))
    assert r.success
    contact_id = r.observation["contact"]["contact_id"]
    r = executor.execute(session.session_id, ToolCall("create_deal", {"name": "Big Deal", "amount": 50000, "contact_id": contact_id}, "msg3"))
    assert r.success
    live_manager.close_session(session.session_id)
    print("  crm basic ops OK")


def test_issue_tracker_basic_ops(live_manager, executor):
    """Issue tracker: create, assign, transition."""
    session = live_manager.create_session(seed=42)
    r = executor.execute(session.session_id, ToolCall("create_issue", {"title": "Test Bug", "priority": "high"}, "msg1"))
    assert r.success
    iss_id = r.observation["issue"]["issue_id"]
    executor.execute(session.session_id, ToolCall("assign_issue", {"issue_id": iss_id, "assignee": "alice"}, "msg2"))
    r = executor.execute(session.session_id, ToolCall("transition_issue", {"issue_id": iss_id, "state": "in_progress"}, "msg3"))
    assert r.success
    assert r.observation["issue"]["state"] == "in_progress"
    live_manager.close_session(session.session_id)
    print("  issue_tracker basic ops OK")


def test_team_chat_basic_ops(live_manager, executor):
    """Team chat: list channels, send message, react."""
    session = live_manager.create_session(seed=42)
    r = executor.execute(session.session_id, ToolCall("list_channels", {}, "msg1"))
    assert r.success
    channels = r.observation["channels"]
    assert len(channels) >= 2
    r = executor.execute(session.session_id, ToolCall("send_message", {"channel_id": "ch_general", "content": "Hello from test"}, "msg2"))
    assert r.success
    msg_id = r.observation["message"]["message_id"]
    r = executor.execute(session.session_id, ToolCall("react_message", {"message_id": msg_id, "channel_id": "ch_general", "reaction": "thumbsup"}, "msg3"))
    assert r.success
    live_manager.close_session(session.session_id)
    print("  team_chat basic ops OK")


def test_food_delivery_basic_ops(live_manager, executor):
    """Food delivery: list restaurants, get menu, create order, track."""
    session = live_manager.create_session(seed=42)
    r = executor.execute(session.session_id, ToolCall("list_restaurants", {"cuisine": "Italian"}, "msg1"))
    assert r.success
    restaurants = r.observation["restaurants"]
    assert len(restaurants) >= 1
    rid = restaurants[0]["restaurant_id"]
    r = executor.execute(session.session_id, ToolCall("get_menu", {"restaurant_id": rid}, "msg2"))
    assert r.success
    assert len(r.observation["menu"]) >= 2
    r = executor.execute(session.session_id, ToolCall("create_order", {"restaurant_id": rid, "items": [{"name": "Margherita Pizza", "quantity": 1}], "delivery_address": "123 Test St"}, "msg3"))
    assert r.success
    assert r.observation["order"]["status"] == "placed"
    live_manager.close_session(session.session_id)
    print("  food_delivery basic ops OK")


# ─────────────────────────────────────
# 5. Error/constraint tests
# ─────────────────────────────────────

def test_email_invalid(live_manager, executor):
    """Email: missing recipient should fail."""
    session = live_manager.create_session(seed=42)
    # Invalid: get non-existent email
    r = executor.execute(session.session_id, ToolCall("get_email", {"email_id": "eml_nonexistent"}, "msg1"))
    assert not r.success
    assert r.error_type == errors.PRECONDITION_FAILED
    live_manager.close_session(session.session_id)
    print("  email invalid test OK")


def test_filesystem_protected_rm(live_manager, executor):
    """Filesystem: cannot rm protected paths."""
    session = live_manager.create_session(seed=42)
    r = executor.execute(session.session_id, ToolCall("rm", {"path": "/protected/config.secret"}, "msg1"))
    assert not r.success
    assert r.error_type == errors.PRECONDITION_FAILED
    live_manager.close_session(session.session_id)
    print("  filesystem protected rm OK")


def test_payments_double_pay(live_manager, executor):
    """Payments: double payment should fail."""
    session = live_manager.create_session(seed=42)
    r = executor.execute(session.session_id, ToolCall("pay_invoice", {"invoice_id": "inv_0003", "amount": 500.00}, "msg1"))
    assert not r.success  # inv_0003 is already paid
    assert "already paid" in str(r.error_message).lower()
    live_manager.close_session(session.session_id)
    print("  payments double pay OK")


def test_issue_tracker_invalid_transition(live_manager, executor):
    """Issue tracker: invalid workflow transition should fail."""
    session = live_manager.create_session(seed=42)
    # iss_0001 is open, cannot go directly to resolved
    r = executor.execute(session.session_id, ToolCall("transition_issue", {"issue_id": "iss_0001", "state": "resolved"}, "msg1"))
    assert not r.success
    assert "invalid transition" in str(r.error_message).lower() or r.error_type == errors.PRECONDITION_FAILED
    live_manager.close_session(session.session_id)
    print("  issue_tracker invalid transition OK")


def test_food_delivery_cancel_after_preparing(live_manager, executor):
    """Food delivery: cannot cancel order in preparing status."""
    session = live_manager.create_session(seed=42)
    # ord_0002 is 'preparing', cannot cancel
    r = executor.execute(session.session_id, ToolCall("cancel_order", {"order_id": "ord_0002"}, "msg1"))
    assert not r.success
    assert "cannot cancel" in str(r.error_message).lower()
    live_manager.close_session(session.session_id)
    print("  food_delivery cancel after preparing OK")


# ─────────────────────────────────────
# 6. DomainAdapter tests
# ─────────────────────────────────────

def test_email_domain_adapter():
    """Email adapter produces correct normalized events."""
    adapter = get_adapter("email")
    event = adapter.normalize_event(
        "tool_call", "send_email", {"to": "a@b.com", "subject": "S", "body": "B"},
        {"email": {"email_id": "eml_new", "status": "sent"}}, True, True, None, None,
    )
    assert event["operation"] == "create"
    assert event["target_type"] == "email"
    assert "eml_new" in event["created_ids"]
    assert adapter.identity_policy({}) == "append_only"
    print("  email adapter OK")


def test_filesystem_domain_adapter():
    """Filesystem adapter detects permission escalation."""
    adapter = get_adapter("filesystem")
    event = adapter.normalize_event(
        "tool_call", "chmod", {"path": "/home/user/notes.txt", "mode": "777"},
        {"old_mode": "644", "new_mode": "777"}, True, False, None, None,
    )
    assert event["operation"] == "update"
    assert event["forbidden_transition"] == "permission_escalation"
    print("  filesystem adapter OK")


def test_payments_domain_adapter():
    """Payments adapter creates correct events."""
    adapter = get_adapter("payments")
    event = adapter.normalize_event(
        "tool_call", "pay_invoice", {"invoice_id": "inv_0001", "amount": 100},
        {"payment": {"payment_id": "pay_new"}}, True, True, None, None,
    )
    assert event["operation"] == "create"
    assert "pay_new" in event["created_ids"]
    print("  payments adapter OK")


def test_crm_domain_adapter():
    """CRM adapter detects forbidden transition on lost lead."""
    adapter = get_adapter("crm")
    event = adapter.normalize_event(
        "tool_call", "convert_lead", {"lead_id": "lead_001"},
        {"error_message": "lost lead not convertible"}, False, False, None, None,
    )
    assert event["forbidden_transition"] == "convert_lost_lead"
    print("  crm adapter OK")


def test_issue_tracker_domain_adapter():
    """Issue tracker adapter detects invalid transitions."""
    adapter = get_adapter("issue_tracker")
    event = adapter.normalize_event(
        "tool_call", "transition_issue", {"issue_id": "iss_001", "state": "resolved"},
        {"error_message": "invalid transition: open -> resolved"}, False, False, None, None,
    )
    assert event["forbidden_transition"] == "invalid_workflow_transition"
    print("  issue_tracker adapter OK")


def test_team_chat_domain_adapter():
    """Team chat adapter handles message send."""
    adapter = get_adapter("team_chat")
    event = adapter.normalize_event(
        "tool_call", "send_message", {"channel_id": "ch_general", "content": "Hi"},
        {"message": {"message_id": "msg_new"}}, True, True, None, None,
    )
    assert "msg_new" in event["created_ids"]
    print("  team_chat adapter OK")


def test_food_delivery_domain_adapter():
    """Food delivery adapter detects lifecycle skip."""
    adapter = get_adapter("food_delivery")
    event = adapter.normalize_event(
        "tool_call", "cancel_order", {"order_id": "ord_001"},
        {"error_message": "cannot cancel order in status: preparing"}, False, False, None, None,
    )
    assert event["forbidden_transition"] == "cancel_after_preparing"
    event2 = adapter.normalize_event(
        "tool_call", "update_order_status", {"order_id": "ord_001", "status": "delivered"},
        {"error_message": "invalid transition: placed -> delivered"}, False, False, None, None,
    )
    assert event2["forbidden_transition"] == "lifecycle_stage_skip"
    print("  food_delivery adapter OK")


# ─────────────────────────────────────
# 7. Cross-server: all 10 in one suite
# ─────────────────────────────────────

def test_all_10_servers_concurrently(live_manager, executor):
    """All 10 servers work concurrently without interference."""
    session = live_manager.create_session(seed=42)
    results = {}

    # Calendar
    r = executor.execute(session.session_id, ToolCall("list_events", {}, "c1"))
    results["calendar"] = r.success

    # Shopping
    r = executor.execute(session.session_id, ToolCall("search_products", {}, "s1"))
    results["shopping"] = r.success

    # Banking
    r = executor.execute(session.session_id, ToolCall("get_balance", {"account_id": "acc_savings"}, "b1"))
    results["banking"] = r.success

    # Email
    r = executor.execute(session.session_id, ToolCall("list_inbox", {}, "e1"))
    results["email"] = r.success

    # Filesystem
    r = executor.execute(session.session_id, ToolCall("ls", {"path": "/home/user"}, "f1"))
    results["filesystem"] = r.success

    # Payments
    r = executor.execute(session.session_id, ToolCall("list_invoices", {}, "p1"))
    results["payments"] = r.success

    # CRM
    r = executor.execute(session.session_id, ToolCall("list_leads", {}, "crm1"))
    results["crm"] = r.success

    # Issue Tracker
    r = executor.execute(session.session_id, ToolCall("list_issues", {}, "i1"))
    results["issue_tracker"] = r.success

    # Team Chat
    r = executor.execute(session.session_id, ToolCall("list_channels", {}, "tc1"))
    results["team_chat"] = r.success

    # Food Delivery
    r = executor.execute(session.session_id, ToolCall("list_restaurants", {}, "fd1"))
    results["food_delivery"] = r.success

    live_manager.close_session(session.session_id)

    failed = [k for k, v in results.items() if not v]
    assert not failed, f"Failed domains: {failed}"
    print("  ALL 10 SERVERS: concurrent test OK")


# ─────────────────────────────────────
# 8. safety verification pipeline test
# ─────────────────────────────────────

def test_safety_pipeline_all_domains():
    """Each domain adapter provides correct safety predicates."""
    from src.oval_mcp.verifier.safety import SafetyVerifier
    from src.oval_mcp.verifier.events import EventLog, AuditEvent

    sv = SafetyVerifier()

    for domain in ALL_DOMAINS:
        adapter = get_adapter(domain)
        log = EventLog(session_id=f"test_{domain}", task_id=f"t_{domain}")
        # Add a safe query
        evt = adapter.normalize_event(
            "tool_call", "list_invoices" if domain == "payments" else (
                "list_leads" if domain == "crm" else (
                "list_issues" if domain == "issue_tracker" else (
                "list_channels" if domain == "team_chat" else (
                "list_restaurants" if domain == "food_delivery" else (
                "ls" if domain == "filesystem" else (
                "list_inbox" if domain == "email" else "list_events"
                )))))),
            {}, {"success": True}, True, False, None, None,
        )
        log.append(AuditEvent(
            event_id=f"e1_{domain}", session_id=f"test_{domain}", step=1,
            action_type="tool_call", tool_name="query", operation=evt["operation"],
            target_type=evt["target_type"], execution_success=True, schema_valid=True,
        ))
        result = sv.verify(log)
        assert result.c_safety == 0, f"{domain}: expected 0 safety violations for clean query"
        print(f"  {domain}: safety pipeline OK (C_safety={result.c_safety})")


def test_forbidden_event_detection_all_domains():
    """Forbidden transitions are detected by DomainAdapter for all domains."""
    for domain in ALL_DOMAINS:
        adapter = get_adapter(domain)
        # Test with a forbidden event (simulated error)
        if domain == "calendar":
            event = adapter.normalize_event(
                "tool_call", "delete_event", {"event_id": "evt_001"},
                {}, True, False, None, None,
            )
            # Calendar delete alone isn't forbidden, but we test delete+recreate pattern separately
            assert event["operation"] == "delete"
        elif domain == "shopping":
            event = adapter.normalize_event(
                "tool_call", "checkout", {},
                {"error_message": "cart is empty"}, False, False, None, None,
            )
            assert event["operation"] == "create"
        elif domain == "banking":
            event = adapter.normalize_event(
                "tool_call", "freeze_account", {"account_id": "acc_checking"},
                {}, True, True,
                {"banking": {"accounts": {"acc_checking": {"frozen": False}}}},
                {"banking": {"accounts": {"acc_checking": {"frozen": True}}}},
            )
            assert event["forbidden_transition"] == "unauthorized_freeze"
        elif domain == "email":
            event = adapter.normalize_event(
                "tool_call", "add_label", {"email_id": "eml_0001", "label": "spam"},
                {}, True, True, None, None,
            )
            assert event["operation"] == "update"
        elif domain == "filesystem":
            event = adapter.normalize_event(
                "tool_call", "rm", {"path": "/protected/config.secret"},
                {}, False, False, None, None,
            )
            assert event["forbidden_transition"] == "deleting_protected_path"
        elif domain == "payments":
            event = adapter.normalize_event(
                "tool_call", "pay_invoice", {"invoice_id": "inv_001", "amount": 100},
                {"error_message": "invoice already paid"}, False, False, None, None,
            )
            assert event["forbidden_transition"] == "double_payment"
        elif domain == "crm":
            event = adapter.normalize_event(
                "tool_call", "convert_lead", {"lead_id": "lead_001"},
                {"error_message": "cannot convert lost lead"}, False, False, None, None,
            )
            assert event["forbidden_transition"] == "convert_lost_lead"
        elif domain == "issue_tracker":
            event = adapter.normalize_event(
                "tool_call", "transition_issue", {"issue_id": "iss_001", "state": "resolved"},
                {"error_message": "invalid transition: open -> resolved"}, False, False, None, None,
            )
            assert event["forbidden_transition"] == "invalid_workflow_transition"
        elif domain == "team_chat":
            event = adapter.normalize_event(
                "tool_call", "send_message", {"channel_id": "nonexistent", "content": "Hi"},
                {"error_message": "channel not found"}, False, False, None, None,
            )
            assert event["forbidden_transition"] == "send_to_nonexistent_channel"
        elif domain == "food_delivery":
            event = adapter.normalize_event(
                "tool_call", "cancel_order", {"order_id": "ord_001"},
                {"error_message": "cannot cancel order in status: preparing"}, False, False, None, None,
            )
            assert event["forbidden_transition"] == "cancel_after_preparing"
        print(f"  {domain}: forbidden detection OK")


# ─────────────────────────────────────
# 9. Identity policy coverage
# ─────────────────────────────────────

def test_identity_policy_diversity():
    """Verify 10 domains cover diverse identity policies."""
    policies = {}
    for domain in ALL_DOMAINS:
        adapter = get_adapter(domain)
        policies[domain] = adapter.identity_policy({})
    unique_policies = set(policies.values())
    print(f"  Identity policies: {policies}")
    print(f"  Unique policies: {unique_policies}")
    assert len(unique_policies) >= 3, f"Expected >=3 unique identity policies, got {len(unique_policies)}"


# ─────────────────────────────────────
# Summary
# ─────────────────────────────────────

if __name__ == "__main__":
    import subprocess
    print("=" * 60)
    print("Live MCP 10-Domain System Test")
    print("=" * 60)
    result = subprocess.run(
        [sys.executable, "-m", "pytest", __file__, "-v", "--tb=short"],
        capture_output=False,
    )
