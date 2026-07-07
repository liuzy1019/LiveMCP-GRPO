"""Focused LiveMCP logic regression checks.

This script exercises invariants that are easy to miss with seed sampling:
negative-value rejection, reference integrity, initial-state consistency, and
cross-domain tool-name disambiguation.
"""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from scripts.merge_rollout_shards import merge_split
from src.live_mcp.manager import LiveMCPManager
from src.live_mcp.schema_registry import SchemaRegistry
from src.live_mcp.config import load_suite_config
from src.live_mcp.executor import LiveMCPExecutor
from src.live_mcp.servers.banking.server import BankingServer
from src.live_mcp.servers.calendar.server import CalendarServer
from src.live_mcp.servers.crm.server import CRMServer
from src.live_mcp.servers.email.server import EmailServer
from src.live_mcp.servers.filesystem.server import FilesystemServer
from src.live_mcp.servers.food_delivery.server import FoodDeliveryServer
from src.live_mcp.servers.issue_tracker.server import IssueTrackerServer
from src.live_mcp.servers.payments.server import PaymentsServer
from src.live_mcp.servers.shopping.server import ShoppingServer
from src.live_mcp.servers.team_chat.server import TeamChatServer
from src.live_mcp.transport import SubprocessStdioTransport
from src.live_mcp.types import OracleCall, ToolCall, ToolExecutionResult
from src.live_mcp.orchestrator import (
    TaskOrchestrator,
    _deterministic_schema_edges,
    _detect_missing_dependency,
    _tool_existing_entity_requirements,
)
from src.oval_mcp.envs.audit_wrapper import AuditWrapper
from src.oval_mcp.envs.domain_adapter import get_adapter
from src.oval_mcp.rewards.f_gamma import ProgressTracker

try:
    from src.live_mcp.task_planner import TaskPlanner, derive_progress_predicates
except ModuleNotFoundError as exc:
    TaskPlanner = None
    derive_progress_predicates = None
    _DERIVE_PROGRESS_IMPORT_ERROR = exc


SID = "logic_regression"
SEED = 42


def reset(server):
    server.handle_request("session/reset", {"session_id": SID, "seed": SEED})
    return server


def call(server, name: str, args: dict):
    return server.handle_request(
        "tools/call",
        {"session_id": SID, "name": name, "arguments": args},
    )["result"]


def state(server):
    return server.handle_request("debug/get_state", {"session_id": SID})["result"]["state"]


def assert_success(result):
    assert result["success"], result
    return result["observation"]


def assert_failure(result):
    assert not result["success"], result
    assert not result["state_changed"], result
    return result


def test_banking():
    server = reset(BankingServer())
    st = state(server)
    aid = next(aid for aid, acct in st["accounts"].items() if not acct.get("frozen"))
    balance = st["accounts"][aid]["balance"]
    assert_failure(call(server, "bill_pay", {"account_id": aid, "payee": "Utility", "amount": -10}))
    assert_failure(call(server, "wire_transfer", {"from_account": aid, "routing_number": "123", "recipient_name": "Bob", "amount": -10}))
    assert state(server)["accounts"][aid]["balance"] == balance
    assert_success(call(server, "deposit", {"account_id": aid, "amount": 5}))
    assert_success(call(server, "get_history", {"account_id": aid}))
    assert_success(call(server, "get_statement", {"account_id": aid, "year": 2026, "month": 6}))
    frozen = next(aid for aid, acct in state(server)["accounts"].items() if acct.get("frozen"))
    assert not call(server, "freeze_account", {"account_id": frozen})["state_changed"]
    accounts = [aid for aid, acct in state(server)["accounts"].items() if not acct.get("frozen")]
    scheduled = assert_success(call(server, "schedule_transfer", {"from_account": accounts[0], "to_account": accounts[1], "amount": 1, "execute_date": "2026-07-01"}))["scheduled_transfer"]
    assert_success(call(server, "cancel_transfer", {"scheduled_txn_id": scheduled["scheduled_txn_id"]}))
    assert not call(server, "cancel_transfer", {"scheduled_txn_id": scheduled["scheduled_txn_id"]})["state_changed"]


def test_shopping():
    server = reset(ShoppingServer())
    st = state(server)
    pid = next(iter(st["products"]))
    stock = st["products"][pid]["stock"]
    assert not call(server, "clear_cart", {})["state_changed"]
    assert_failure(call(server, "add_to_cart", {"product_id": pid, "quantity": -2}))
    assert state(server)["products"][pid]["stock"] == stock
    assert_success(call(server, "add_to_cart", {"product_id": pid, "quantity": 1}))
    assert_failure(call(server, "update_cart_quantity", {"product_id": pid, "quantity": -3}))
    assert_success(call(server, "apply_coupon", {"code": "SAVE10"}))
    assert not call(server, "apply_coupon", {"code": "SAVE10"})["state_changed"]
    assert_success(call(server, "add_to_wishlist", {"product_id": pid}))
    assert not call(server, "add_to_wishlist", {"product_id": pid})["state_changed"]
    absent_pid = next(product_id for product_id in st["products"] if product_id != pid)
    assert not call(server, "remove_from_wishlist", {"product_id": absent_pid})["state_changed"]
    order_id = next(iter(state(server)["orders"]))
    assert_success(call(server, "return_order", {"order_id": order_id, "reason": "duplicate check"}))
    returns_count = len(state(server).get("returns", {}))
    assert_failure(call(server, "return_order", {"order_id": order_id, "reason": "duplicate check"}))
    assert len(state(server).get("returns", {})) == returns_count
    for order in state(server)["orders"].values():
        assert "items" in order
        assert all(item["quantity"] > 0 for item in order["items"])


def test_payments():
    server = reset(PaymentsServer())
    next_inv = state(server)["next_inv_num"]
    assert_failure(call(server, "create_invoice", {"customer": "Acme", "amount": -1}))
    assert state(server)["next_inv_num"] == next_inv
    st = state(server)
    for inv in st["invoices"].values():
        if inv["status"] == "paid":
            payment = st["payments"].get(inv["payment_id"])
            assert payment is not None
            assert payment["amount"] == inv["amount"]
    pending = next(inv for inv in st["invoices"].values() if inv["status"] == "pending")
    assert_failure(call(server, "pay_invoice", {"invoice_id": pending["invoice_id"], "amount": -pending["amount"]}))
    next_webhook = state(server)["next_wh_num"]
    assert_failure(call(server, "create_webhook", {"url": "", "events": []}))
    assert state(server)["next_wh_num"] == next_webhook
    next_inv = state(server)["next_inv_num"]
    assert_failure(call(server, "dispute_invoice", {"invoice_id": pending["invoice_id"], "reason": ""}))
    assert state(server)["next_inv_num"] == next_inv
    webhook = assert_success(call(server, "create_webhook", {"url": "https://example.com/hook", "events": ["invoice.paid"]}))["webhook"]
    assert_success(call(server, "delete_webhook", {"webhook_id": webhook["webhook_id"]}))
    assert not call(server, "delete_webhook", {"webhook_id": webhook["webhook_id"]})["state_changed"]
    pending_payment = next(p for p in st["payments"].values() if p["status"] == "pending")
    assert_success(call(server, "cancel_payment", {"payment_id": pending_payment["payment_id"]}))
    st = state(server)
    paid = next(
        inv for inv in st["invoices"].values()
        if inv["status"] == "paid"
        and st["payments"][inv["payment_id"]]["status"] == "settled"
    )
    assert_failure(call(server, "refund_invoice", {"invoice_id": paid["invoice_id"], "amount": -1}))
    assert_failure(call(server, "cancel_payment", {"payment_id": paid["payment_id"]}))
    amount = paid["amount"]
    assert_success(call(server, "refund_invoice", {"invoice_id": paid["invoice_id"], "amount": round(amount / 2, 2)}))
    st = state(server)
    assert st["invoices"][paid["invoice_id"]]["status"] == "partially_refunded"
    assert_success(call(server, "refund_invoice", {"invoice_id": paid["invoice_id"], "amount": round(amount - round(amount / 2, 2), 2)}))
    assert state(server)["invoices"][paid["invoice_id"]]["status"] == "refunded"


def test_crm():
    server = reset(CRMServer())
    st = state(server)
    for contact in st["contacts"].values():
        assert contact["lead_id"] in st["leads"]
    existing_lead_id = next(iter(st["leads"]))
    assert not call(server, "update_lead", {"lead_id": existing_lead_id, "fields": {"unknown": "x"}})["state_changed"]
    lead = assert_success(call(server, "create_lead", {"name": "Ref Lead", "company": "RefCo"}))["lead"]
    next_deal = state(server)["next_deal_num"]
    assert_failure(call(server, "create_deal", {"name": "Bad", "amount": -1, "lead_id": lead["lead_id"]}))
    assert state(server)["next_deal_num"] == next_deal
    assert_success(call(server, "create_deal", {"name": "Good", "amount": 100, "lead_id": lead["lead_id"]}))
    assert_failure(call(server, "delete_lead", {"lead_id": lead["lead_id"]}))
    next_task = state(server)["next_task_num"]
    assert_failure(call(server, "create_task", {"title": "Ghost", "deal_id": "deal_missing"}))
    assert state(server)["next_task_num"] == next_task
    task = assert_success(call(server, "create_task", {"title": "Follow up"}))["task"]
    assert_success(call(server, "complete_task", {"task_id": task["task_id"]}))
    assert not call(server, "complete_task", {"task_id": task["task_id"]})["state_changed"]


def test_issue_tracker():
    server = reset(IssueTrackerServer())
    st = state(server)
    iid = next(iter(st["issues"]))
    next_sprint = st["next_sprint_num"]
    assert_failure(call(server, "create_sprint", {"name": "Bad", "start_date": "2026-07-02", "end_date": "2026-07-01"}))
    assert state(server)["next_sprint_num"] == next_sprint
    next_subtask = state(server)["next_subtask_num"]
    assert_failure(call(server, "create_subtask", {"issue_id": iid, "title": "Ghost", "assignee": "missing"}))
    assert state(server)["next_subtask_num"] == next_subtask
    assert_failure(call(server, "add_label", {"issue_id": iid, "label": ""}))
    assert_success(call(server, "add_label", {"issue_id": iid, "label": "x"}))
    assert not call(server, "add_label", {"issue_id": iid, "label": "x"})["state_changed"]
    assert not call(server, "remove_label", {"issue_id": iid, "label": "absent"})["state_changed"]
    member = next(iter(st["members"]))
    assert_success(call(server, "add_watcher", {"issue_id": iid, "user": member}))
    assert not call(server, "add_watcher", {"issue_id": iid, "user": member})["state_changed"]
    no_sprint_iid = next(issue_id for issue_id, issue in state(server)["issues"].items() if not issue.get("sprint_id"))
    assert not call(server, "remove_from_sprint", {"issue_id": no_sprint_iid})["state_changed"]
    assert_failure(call(server, "time_track", {"issue_id": iid, "hours": -1}))
    sprint_1 = next(iter(st["sprints"]))
    sprint_2 = assert_success(call(server, "create_sprint", {"name": "Next", "start_date": "2026-07-01", "end_date": "2026-07-14"}))["sprint"]["sprint_id"]
    assert_success(call(server, "add_to_sprint", {"issue_id": iid, "sprint_id": sprint_2}))
    assert not call(server, "add_to_sprint", {"issue_id": iid, "sprint_id": sprint_2})["state_changed"]
    st = state(server)
    assert iid not in st["sprints"][sprint_1]["issues"]
    assert iid in st["sprints"][sprint_2]["issues"]
    assert_success(call(server, "time_track", {"issue_id": iid, "hours": 2}))
    report = assert_success(call(server, "get_time_report", {"sprint_id": sprint_2}))
    assert report["total_hours"] == 2


def test_email():
    server = reset(EmailServer())
    before = assert_success(call(server, "list_inbox", {}))["total"]
    assert_success(call(server, "send_email", {"to": "a@example.com", "subject": "S", "body": "B"}))
    assert assert_success(call(server, "list_inbox", {}))["total"] == before
    received = next(eid for eid, email in state(server)["emails"].items() if email["status"] == "received")
    existing_label = state(server)["emails"][received]["labels"][0] if state(server)["emails"][received]["labels"] else "work"
    if existing_label not in state(server)["emails"][received]["labels"]:
        assert_success(call(server, "add_label", {"email_id": received, "label": existing_label}))
    assert not call(server, "add_label", {"email_id": received, "label": existing_label})["state_changed"]
    assert not call(server, "remove_label", {"email_id": received, "label": "absent_label"})["state_changed"]
    next_email = state(server)["next_email_num"]
    assert_failure(call(server, "send_email", {"to": "a@example.com", "subject": "S", "body": "B", "thread_id": "missing_thread"}))
    assert state(server)["next_email_num"] == next_email
    assert_failure(call(server, "move_to_thread", {"email_id": received, "thread_id": "missing_thread"}))
    assert_success(call(server, "mark_read", {"email_id": received}))
    assert not call(server, "mark_read", {"email_id": received})["state_changed"]
    assert_success(call(server, "reply_email", {"email_id": received, "body": "R"}))
    assert_success(call(server, "forward_email", {"email_id": received, "to": "b@example.com"}))
    assert assert_success(call(server, "list_inbox", {}))["total"] == before


def test_filesystem():
    server = reset(FilesystemServer())
    assert_failure(call(server, "head", {"path": "/home/user/notes.txt", "lines": -1}))
    assert_failure(call(server, "tail", {"path": "/home/user/notes.txt", "lines": -1}))
    assert_failure(call(server, "rm", {"path": "/protected", "recursive": True}))
    assert_failure(call(server, "mv", {"source": "/protected", "target": "/home/user/protected"}))
    assert_failure(call(server, "mv", {"source": "/home/user/projects", "target": "/home/user/projects/nested"}))
    assert_failure(call(server, "cp", {"source": "/protected/config.secret", "target": "/home/user/secret.copy"}))
    assert_failure(call(server, "chmod", {"path": "/protected", "mode": "777"}))
    assert_failure(call(server, "tar_create", {"archive": "/protected/new.tar", "paths": ["/home/user/notes.txt"]}))
    assert_failure(call(server, "tar_create", {"archive": "/home/user/new.tar", "paths": ["/protected/config.secret"]}))
    assert_failure(call(server, "symlink", {"target": "/protected/config.secret", "link_path": "/home/user/secret_link"}))
    assert_failure(call(server, "truncate", {"path": "/home/user/notes.txt", "size": -1}))
    assert_failure(call(server, "split", {"path": "/home/user/notes.txt", "lines_per_file": 0}))
    assert_failure(call(server, "touch", {"path": "/home/user/notes.txt/child"}))
    assert_failure(call(server, "mkdir", {"path": "/home/user/notes.txt/child"}))
    assert_failure(call(server, "tar_create", {"archive": "/home/user/notes.txt/archive.tar", "paths": ["/home/user/notes.txt"]}))
    assert_failure(call(server, "symlink", {"target": "/home/user/notes.txt", "link_path": "/home/user/notes.txt/link"}))
    assert_failure(call(server, "cp", {"source": "/home/user/projects", "target": "/home/user/projects_copy"}))
    assert_failure(call(server, "rm", {"path": "/", "recursive": True}))
    assert_failure(call(server, "diff", {"file1": "/home/user/logs", "file2": "/home/user/projects"}))
    assert_failure(call(server, "split", {"path": "/home/user/logs", "lines_per_file": 10}))
    assert_success(call(server, "mkdir", {"path": "/home/user/newdir"}))
    assert_success(call(server, "touch", {"path": "/home/user/newfile"}))
    assert_failure(call(server, "mv", {"source": "/home/user/newfile", "target": "/home/user/newdir"}))
    assert "/home/user/newdir" in state(server)["fs"]
    assert_failure(call(server, "cp", {"source": "/home/user/notes.txt", "target": "/home/user/newdir"}))
    assert state(server)["fs"]["/home/user/newdir"]["type"] == "dir"
    assert not call(server, "tar_extract", {"archive": "/home/user/notes.txt", "target_dir": "/home/user"})["state_changed"]
    st = state(server)
    assert "/protected" in st["fs"]
    assert "/home/user/projects" in st["fs"]


def test_calendar():
    server = reset(CalendarServer())
    next_event = state(server)["next_event_num"]
    assert_failure(call(server, "create_event", {"title": "Bad", "start_time": "2026-07-01T11:00", "end_time": "2026-07-01T10:00"}))
    assert state(server)["next_event_num"] == next_event
    evt = assert_success(call(server, "create_event", {"title": "Good", "start_time": "2026-07-01T10:00", "end_time": "2026-07-01T11:00"}))["event"]
    assert not call(server, "update_event", {"event_id": evt["event_id"], "fields": {"unknown": "x"}})["state_changed"]
    assert_failure(call(server, "add_attendee", {"event_id": evt["event_id"], "email": ""}))
    assert_success(call(server, "add_attendee", {"event_id": evt["event_id"], "email": "a@example.com"}))
    assert not call(server, "add_attendee", {"event_id": evt["event_id"], "email": "a@example.com"})["state_changed"]
    assert_failure(call(server, "respond_to_event", {"event_id": evt["event_id"], "email": "nobody@example.com", "response": "accepted"}))
    assert_failure(call(server, "set_reminder", {"event_id": evt["event_id"], "minutes_before": -5}))
    assert not call(server, "change_timezone", {"timezone": state(server)["timezone"]})["state_changed"]
    assert_success(call(server, "respond_to_event", {"event_id": evt["event_id"], "email": "a@example.com", "response": "accepted"}))
    assert not call(server, "respond_to_event", {"event_id": evt["event_id"], "email": "a@example.com", "response": "accepted"})["state_changed"]
    absent = call(server, "remove_attendee", {"event_id": evt["event_id"], "email": "nobody@example.com"})
    assert absent["success"] and not absent["state_changed"], absent


def test_team_chat():
    server = reset(TeamChatServer())
    st = state(server)
    cid = next(ch["channel_id"] for ch in st["channels"].values() if not ch.get("archived"))
    assert_failure(call(server, "create_channel", {"name": "general"}))
    assert_failure(call(server, "create_channel", {"name": "new", "members": ["ghost"]}))
    archived = next(ch["channel_id"] for ch in st["channels"].values() if ch.get("archived"))
    assert not call(server, "archive_channel", {"channel_id": archived})["state_changed"]
    assert_failure(call(server, "send_message", {"channel_id": cid, "content": ""}))
    member = next(m for ch in state(server)["channels"].values() for m in ch["members"] if m != "current_user")
    assert_failure(call(server, "send_dm", {"recipient": member, "content": "   "}))
    root = assert_success(call(server, "send_message", {"channel_id": cid, "content": "root"}))["message"]
    assert_failure(call(server, "react_message", {"channel_id": cid, "message_id": root["message_id"], "reaction": ""}))
    assert_success(call(server, "react_message", {"channel_id": cid, "message_id": root["message_id"], "reaction": "+1"}))
    assert not call(server, "react_message", {"channel_id": cid, "message_id": root["message_id"], "reaction": "+1"})["state_changed"]
    thread = assert_success(call(server, "create_thread", {"channel_id": cid, "message_id": root["message_id"]}))["thread"]
    reply = assert_success(call(server, "send_message", {"channel_id": cid, "thread_id": thread["thread_id"], "content": "reply"}))["message"]
    assert reply in state(server)["threads"][thread["thread_id"]]["messages"]


def test_food_delivery():
    server = reset(FoodDeliveryServer())
    st = state(server)
    rid = next(iter(st["restaurants"]))
    next_order = st["next_order_num"]
    assert_failure(call(server, "create_order", {"restaurant_id": rid, "items": [], "delivery_address": "123 Main St"}))
    assert state(server)["next_order_num"] == next_order
    delivered = next(o for o in state(server)["orders"].values() if o["status"] == "delivered")
    oid = delivered["order_id"]
    assert_failure(call(server, "add_tip", {"order_id": oid, "amount": 1}))
    assert_success(call(server, "rate_order", {"order_id": oid, "rating": 5}))
    assert_failure(call(server, "rate_order", {"order_id": oid, "rating": 4}))


def test_routing_helpers():
    registry = SchemaRegistry()
    from src.live_mcp.servers.email.server import TOOLS as EMAIL_TOOLS
    from src.live_mcp.servers.issue_tracker.server import TOOLS as ISSUE_TOOLS

    registry.register_tools("email", EMAIL_TOOLS)
    registry.register_tools("issue_tracker", ISSUE_TOOLS)
    email_schema = registry.get_schema("add_label", domain="email")
    issue_schema = registry.get_schema("add_label", domain="issue_tracker")
    assert email_schema is not None
    assert issue_schema is not None
    assert "email_id" in email_schema["input_schema"]["required"]
    assert "issue_id" in issue_schema["input_schema"]["required"]
    assert registry.validate_arguments("add_label", {"email_id": "e", "label": "x"}, domain="email").valid
    assert not registry.validate_arguments("add_label", {"email_id": "e", "label": "x"}, domain="issue_tracker").valid
    assert registry.get_schema("email::add_label") is not None
    assert registry.server_for_tool("email::add_label") == "email"
    assert registry.canonical_name("email::add_label") == "add_label"

    manager = LiveMCPManager.__new__(LiveMCPManager)
    assert manager is not None  # __new__ always returns instance but Pylance can't prove it
    manager._transports = {
        "dummy": SubprocessStdioTransport(argv=["true"], cwd=Path("."), env={})
    }
    assert LiveMCPManager.subprocess_stdio_used.fget(manager) is True

    suite_manager = LiveMCPManager(load_suite_config("configs/live_mcp/suite_mvp.yaml"))
    suite_manager.start_suite()
    try:
        session = suite_manager.create_session(seed=SEED)
        suite_manager.discover_tools(session.session_id)
        executor = LiveMCPExecutor(suite_manager, suite_manager.registry)
        email_id = next(iter(suite_manager.get_state(session.session_id, "email")["email"]["emails"]))
        result = executor.execute(
            session.session_id,
            ToolCall(
                name="email::add_label",
                arguments={"email_id": email_id, "label": "prefixed"},
                call_id="prefixed_email_add_label",
            ),
        )
        assert result.success, result
        assert result.canonical_tool_name == "add_label"
        assert result.metadata["server_name"] == "email"
        cross = executor.execute(
            session.session_id,
            ToolCall(name="list_orders", arguments={}, call_id="banking_cross_domain_list_orders"),
            domain="banking",
        )
        assert not cross.success, cross
        assert not cross.schema_valid, cross
        assert cross.metadata == {}, cross
    finally:
        suite_manager.stop_suite()


def test_audit_wrapper_failure_markers():
    wrapper = AuditWrapper(
        executor=None,
        manager=None,
        adapter=get_adapter("issue_tracker"),
        domain_name="issue_tracker",
    )
    event = wrapper.audit_step_with_state(
        session_id=SID,
        action_type="tool_call",
        tool_calls=[
            ToolCall(
                name="transition_issue",
                arguments={"issue_id": "iss_1", "state": "closed"},
                call_id="invalid_transition",
            )
        ],
        execution_results=[
            ToolExecutionResult(
                success=False,
                tool_name="transition_issue",
                canonical_tool_name="transition_issue",
                call_id="invalid_transition",
                session_id=SID,
                observation=None,
                error_type="precondition_failed",
                error_message="'invalid transition: open -> closed'",
                schema_valid=True,
                state_changed=False,
                latency_ms=0,
            )
        ],
        pre_state={"issue_tracker": {"issues": {"iss_1": {"issue_id": "iss_1", "state": "open"}}}},
        post_state={"issue_tracker": {"issues": {"iss_1": {"issue_id": "iss_1", "state": "open"}}}},
    )
    assert event.forbidden_transition == "invalid_workflow_transition", event


def test_progress_predicate_entity_mapping():
    if derive_progress_predicates is None:
        print(f"Skipping progress predicate mapping test: {_DERIVE_PROGRESS_IMPORT_ERROR}")
        return
    issue_calls = [
        OracleCall("get_issue", {"issue_id": "iss_1"}),
        OracleCall("add_label", {"issue_id": "iss_1", "label": "x"}),
        OracleCall("", {"text": "done"}, action="final_answer"),
    ]
    issue_progress = derive_progress_predicates(issue_calls, "issue_tracker")
    assert {
        "step": 1,
        "type": "satisfied_dependency_edge",
        "tool": "add_label",
        "from_step": 0,
        "entity": "issue",
    } in issue_progress
    assert any(
        p.get("type") == "completed_required_transition"
        and p.get("tool") == "add_label"
        and p.get("entity") == "issue"
        for p in issue_progress
    ), issue_progress

    fs_calls = [
        OracleCall("stat", {"path": "/home/user/notes.txt"}),
        OracleCall("chmod", {"path": "/home/user/notes.txt", "mode": "600"}),
        OracleCall("", {"text": "done"}, action="final_answer"),
    ]
    fs_progress = derive_progress_predicates(fs_calls, "filesystem")
    assert any(
        p.get("type") == "satisfied_dependency_edge"
        and p.get("tool") == "chmod"
        and p.get("entity") == "file"
        for p in fs_progress
    ), fs_progress

    banking_calls = [
        OracleCall("schedule_transfer", {"from_account": "a", "to_account": "b", "amount": 1}),
        OracleCall("cancel_transfer", {"scheduled_txn_id": "sched_1"}),
        OracleCall("", {"text": "done"}, action="final_answer"),
    ]
    banking_progress = derive_progress_predicates(banking_calls, "banking")
    assert any(
        p.get("type") == "satisfied_dependency_edge"
        and p.get("tool") == "cancel_transfer"
        and p.get("entity") == "scheduled_transfer"
        for p in banking_progress
    ), banking_progress

    shopping_calls = [
        OracleCall("get_product", {"product_id": "prd_1"}),
        OracleCall("add_to_cart", {"product_id": "prd_1", "quantity": 1}),
        OracleCall("", {"text": "done"}, action="final_answer"),
    ]
    shopping_progress = derive_progress_predicates(shopping_calls, "shopping")
    assert any(
        p.get("type") == "satisfied_dependency_edge"
        and p.get("tool") == "add_to_cart"
        and p.get("entity") == "product"
        for p in shopping_progress
    ), shopping_progress

    food_calls = [
        OracleCall("get_menu", {"restaurant_id": "rest_1"}),
        OracleCall("create_order", {"restaurant_id": "rest_1", "items": []}),
        OracleCall("", {"text": "done"}, action="final_answer"),
    ]
    food_progress = derive_progress_predicates(food_calls, "food_delivery")
    assert any(
        p.get("type") == "satisfied_dependency_edge"
        and p.get("tool") == "create_order"
        and p.get("entity") == "restaurant"
        for p in food_progress
    ), food_progress

    update_cart_calls = [
        OracleCall("get_cart", {}),
        OracleCall("update_cart_quantity", {"product_id": "prd_1", "quantity": 2}),
        OracleCall("", {"text": "done"}, action="final_answer"),
    ]
    update_cart_progress = derive_progress_predicates(update_cart_calls, "shopping")
    assert any(
        p.get("type") == "satisfied_dependency_edge"
        and p.get("tool") == "update_cart_quantity"
        and p.get("entity") == "order"
        for p in update_cart_progress
    ), update_cart_progress

    chat_calls = [
        OracleCall("get_channel", {"channel_id": "ch_1"}),
        OracleCall("send_message", {"channel_id": "ch_1", "content": "hello"}),
        OracleCall("", {"text": "done"}, action="final_answer"),
    ]
    chat_progress = derive_progress_predicates(chat_calls, "team_chat")
    assert any(
        p.get("type") == "satisfied_dependency_edge"
        and p.get("tool") == "send_message"
        and p.get("entity") == "channel"
        for p in chat_progress
    ), chat_progress

    thread_calls = [
        OracleCall("send_message", {"channel_id": "ch_1", "content": "root"}),
        OracleCall("create_thread", {"channel_id": "ch_1", "message_id": "msg_1"}),
        OracleCall("", {"text": "done"}, action="final_answer"),
    ]
    thread_progress = derive_progress_predicates(thread_calls, "team_chat")
    assert any(
        p.get("type") == "satisfied_dependency_edge"
        and p.get("tool") == "create_thread"
        and p.get("entity") == "message"
        for p in thread_progress
    ), thread_progress

    dm_calls = [
        OracleCall("get_user_status", {"user_ids": ["alice"]}),
        OracleCall("send_dm", {"recipient": "alice", "content": "hello"}),
        OracleCall("", {"text": "done"}, action="final_answer"),
    ]
    dm_progress = derive_progress_predicates(dm_calls, "team_chat")
    assert any(
        p.get("type") == "satisfied_dependency_edge"
        and p.get("tool") == "send_dm"
        and p.get("entity") == "user"
        for p in dm_progress
    ), dm_progress


def test_f_gamma_progress_predicate_dicts():
    tracker = ProgressTracker()
    predicates = tracker._get_task_progress_predicates({
        "progress_predicates": [
            {"type": "resolved_required_entity", "tool": "get_issue"},
            {"type": "completed_required_transition", "tool": "add_label"},
            {"type": "not_a_progress_predicate"},
        ]
    })
    assert predicates == ["resolved_required_entity", "completed_required_transition"]


def test_missing_dependency_covers_required_mutations():
    bad_cases = [
        ("issue_tracker", ["add_label"]),
        ("shopping", ["add_to_cart"]),
        ("team_chat", ["send_message"]),
        ("banking", ["schedule_transfer"]),
        ("crm", ["complete_task"]),
        ("calendar", ["add_attendee"]),
        ("food_delivery", ["add_tip"]),
    ]
    for domain, tools in bad_cases:
        calls = [OracleCall(action="tool_call", tool_name=tool, arguments={}) for tool in tools]
        assert _detect_missing_dependency(calls, domain), (domain, tools)

    good_cases = [
        ("shopping", ["get_product", "add_to_cart"]),
        ("team_chat", ["get_channel", "send_message"]),
        ("banking", ["list_accounts", "schedule_transfer"]),
        ("banking", ["list_accounts", "schedule_transfer", "cancel_transfer"]),
        ("team_chat", ["get_channel", "send_message", "create_thread"]),
    ]
    for domain, tools in good_cases:
        calls = [OracleCall(action="tool_call", tool_name=tool, arguments={}) for tool in tools]
        assert not _detect_missing_dependency(calls, domain), (domain, tools)


def test_existing_entity_requirements_distinguish_inputs_from_outputs():
    cases = {
        ("crm", "list_tasks"): set(),
        ("crm", "create_deal"): set(),
        ("email", "create_draft"): set(),
        ("issue_tracker", "create_sprint"): set(),
        ("issue_tracker", "create_subtask"): {"issue"},
        ("shopping", "add_to_cart"): {"product"},
        ("shopping", "checkout"): {"cart_item"},
        ("team_chat", "create_thread"): {"channel", "message"},
    }
    for (domain, tool_name), expected in cases.items():
        assert _tool_existing_entity_requirements(tool_name, domain) == expected, (domain, tool_name)


def test_dependency_graph_cache_requires_full_tool_coverage():
    graph = {
        "search_events": {"explicit": [], "implicit": []},
        "get_event": {"explicit": ["search_events"], "implicit": []},
        "update_event": {"explicit": ["get_event"], "implicit": []},
    }
    assert TaskOrchestrator._valid_cached_graph(
        graph,
        ["get_event", "search_events", "update_event"],
    )
    assert not TaskOrchestrator._valid_cached_graph(
        {"get_event": {"explicit": ["search_events"], "implicit": []}},
        ["get_event", "search_events", "update_event"],
    )
    assert not TaskOrchestrator._valid_cached_graph(
        graph,
        ["get_event", "get_event", "search_events", "update_event"],
    )
    assert not TaskOrchestrator._valid_cached_graph(
        {"get_event": {"explicit": ["missing_tool"], "implicit": []}},
        ["get_event"],
    )


def test_dependency_graph_cache_normalization_keeps_prove_edges_only():
    raw = {
        "get_product": {
            "explicit": ["search_products", "search_products", "missing_tool", "get_product"],
            "implicit": ["search_products", "add_to_cart"],
        },
    }
    normalized = TaskOrchestrator._normalize_cached_graph(
        raw,
        ["add_to_cart", "get_product", "search_products"],
    )
    assert set(normalized) == {"add_to_cart", "get_product", "search_products"}
    assert normalized["get_product"]["explicit"] == ["search_products"]
    assert normalized["get_product"]["implicit"] == ["add_to_cart"]
    assert normalized["add_to_cart"] == {"explicit": [], "implicit": []}
    assert TaskOrchestrator._valid_cached_graph(
        normalized,
        ["add_to_cart", "get_product", "search_products"],
    )


def test_dependency_graph_uses_domain_scoped_tools_for_duplicate_names():
    class FakeRegistry:
        def __init__(self):
            self._tools = {
                "shopping": [{"name": "list_orders"}, {"name": "get_order"}],
                "food_delivery": [{"name": "list_orders"}, {"name": "get_order"}],
            }

        def server_tools(self, server_name):
            return self._tools[server_name]

        def server_for_tool(self, tool_name):
            return "shopping"

    class FakeSession:
        session_id = "sid"

    class FakeManager:
        registry = FakeRegistry()

        def create_session(self, seed=0):
            return FakeSession()

        def discover_tools(self, session_id):
            return (
                self.registry.server_tools("shopping")
                + self.registry.server_tools("food_delivery")
            )

        def close_session(self, session_id):
            pass

    class FakeOrchestrator:
        manager = FakeManager()
        _tool_schema_hash = staticmethod(TaskOrchestrator._tool_schema_hash)
        _graph_cache_path = staticmethod(TaskOrchestrator._graph_cache_path)
        _maybe_load_cached_graph = lambda self, server_name, schema_hash, server_tools: None

        def _classify_edges_llm(self, server_tools, server_name):
            names = [tool["name"] for tool in server_tools]
            assert names == ["list_orders", "get_order"]
            return {name: {"explicit": [], "implicit": []} for name in names}

        def _save_cached_graph(self, server_name, schema_hash, server_tools, graph):
            names = [tool["name"] for tool in server_tools]
            assert names == ["list_orders", "get_order"]

    graph = TaskOrchestrator._probe_dependency_graph(FakeOrchestrator(), "shopping")
    assert set(graph) == {"list_orders", "get_order"}


def test_deterministic_schema_edges_are_prerequisite_to_dependent_order():
    tools = [
        {"name": "get_cart"},
        {"name": "checkout"},
        {"name": "get_product"},
        {"name": "add_to_cart"},
    ]
    graph = _deterministic_schema_edges(tools, "shopping")
    assert "checkout" in graph["get_cart"]["explicit"]
    assert "add_to_cart" in graph["get_product"]["explicit"]
    assert "get_cart" not in graph.get("checkout", {}).get("explicit", [])
    assert "get_product" not in graph.get("add_to_cart", {}).get("explicit", [])


def _merge_test_row(task_id: str, scenario: str, terminal: str) -> dict:
    extra = {
        "task_id": task_id,
        "domain": "crm",
        "scenario_type": scenario,
        "user_query": f"query {task_id}",
        "oracle_calls": [
            {"action": "tool_call", "tool_name": "list_leads", "arguments": {}},
            {"action": terminal, "arguments": {"text": "done"}},
        ],
        "hidden_tools": [],
        "visible_tool_names": ["list_leads"],
    }
    return {
        "prompt": '[{"role":"user","content":"query"}]',
        "extra_info": extra,
        "scenario_type": scenario,
    }


def test_merge_quality_gate_rejects_terminal_mismatch_without_writing_output():
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        shard_path = tmpdir / "shard_0_train.parquet"
        outpath = tmpdir / "train.parquet"
        pd.DataFrame([
            _merge_test_row("good", "normal_safe_success", "final_answer"),
            _merge_test_row("bad", "normal_safe_success", "ask_clarification"),
        ]).to_parquet(shard_path, index=False)

        ok, merged = merge_split(tmpdir, "shard_*_train.parquet", outpath, target=2)
        assert not ok
        assert len(merged) == 1
        assert not outpath.exists()

        ok, merged = merge_split(tmpdir, "shard_*_train.parquet", outpath, target=1)
        assert ok
        assert len(merged) == 1
        assert outpath.exists()


def test_teacher_decision_prompt_includes_oracle_chain_guidance():
    if TaskPlanner is None:
        print(f"Skipping teacher chain guidance test: {_DERIVE_PROGRESS_IMPORT_ERROR}")
        return

    class FakeClient:
        def __init__(self):
            self.messages = None

        def generate_chat(self, messages, temperature=0.0):
            self.messages = messages
            return '{"action":"tool_call","tool_name":"list_accounts","arguments":{}}'

    client = FakeClient()
    planner = TaskPlanner(client, "banking", seed=SEED)
    action = planner.decide_action(
        tool_schemas=[
            {
                "name": "list_accounts",
                "description": "List accounts",
                "input_schema": {"type": "object", "properties": {}, "required": []},
            },
            {
                "name": "transfer",
                "description": "Transfer funds",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "from_account": {"type": "string"},
                        "to_account": {"type": "string"},
                        "amount": {"type": "number"},
                    },
                    "required": ["from_account", "to_account", "amount"],
                },
            },
        ],
        user_query="Move money between two accounts.",
        execution_history=[],
        difficulty="complete",
        chain_seed=["list_accounts", "transfer"],
        chain_progress=0,
    )
    assert action.action == "tool_call"
    user_prompt = client.messages[1]["content"]
    assert "Oracle Synthesis Target" in user_prompt
    assert "Remaining chain tools in order: ['list_accounts', 'transfer']" in user_prompt


def main():
    test_banking()
    test_shopping()
    test_payments()
    test_crm()
    test_issue_tracker()
    test_email()
    test_filesystem()
    test_calendar()
    test_team_chat()
    test_food_delivery()
    test_routing_helpers()
    test_audit_wrapper_failure_markers()
    test_progress_predicate_entity_mapping()
    test_f_gamma_progress_predicate_dicts()
    test_missing_dependency_covers_required_mutations()
    test_existing_entity_requirements_distinguish_inputs_from_outputs()
    test_merge_quality_gate_rejects_terminal_mismatch_without_writing_output()
    test_teacher_decision_prompt_includes_oracle_chain_guidance()
    print("LiveMCP logic regressions passed")


if __name__ == "__main__":
    main()
