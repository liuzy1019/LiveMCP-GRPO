from __future__ import annotations

import copy

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
from src.live_mcp.state_seeder import StateSeeder


DOMAINS = (
    "banking", "calendar", "crm", "email", "filesystem",
    "food_delivery", "issue_tracker", "payments", "shopping", "team_chat",
)


def _mutable_aliases(root) -> list[list[tuple[str, ...]]]:
    paths_by_identity: dict[int, list[tuple[str, ...]]] = {}

    def visit(value, path: tuple[str, ...]) -> None:
        if not isinstance(value, (dict, list, set)):
            return
        paths_by_identity.setdefault(id(value), []).append(path)
        if isinstance(value, dict):
            for key, item in value.items():
                visit(item, (*path, str(key)))
        elif isinstance(value, list):
            for index, item in enumerate(value):
                visit(item, (*path, str(index)))

    visit(root, ())
    return [paths for paths in paths_by_identity.values() if len(paths) > 1]


def test_seeded_domain_states_do_not_share_mutable_containers() -> None:
    seeder = StateSeeder()
    for domain in DOMAINS:
        for seed in (0, 42, 999):
            state = seeder.seed_state(domain, "alias-audit", seed)
            assert _mutable_aliases(state) == [], (domain, seed)


def _reset(server):
    server.handle_request("session/reset", {"session_id": "test", "seed": 42})
    return server.sessions["test"]


def _call(server, tool_name: str, arguments: dict):
    return server._call_tool({
        "session_id": "test",
        "name": tool_name,
        "arguments": arguments,
    })


def _assert_rejected_without_mutation(server, tool_name: str, arguments: dict) -> None:
    before = copy.deepcopy(server.sessions["test"])
    result = _call(server, tool_name, arguments)
    assert result["success"] is False
    assert server.sessions["test"] == before


def test_negative_numeric_mutations_are_rejected_without_state_change() -> None:
    banking = BankingServer()
    state = _reset(banking)
    accounts = list(state["accounts"])
    _assert_rejected_without_mutation(banking, "transfer", {
        "from_account": accounts[0], "to_account": accounts[1], "amount": -1,
    })

    payments = PaymentsServer()
    _reset(payments)
    _assert_rejected_without_mutation(payments, "create_invoice", {
        "customer": "customer", "amount": -1,
    })

    shopping = ShoppingServer()
    state = _reset(shopping)
    _assert_rejected_without_mutation(shopping, "add_to_cart", {
        "product_id": next(iter(state["products"])), "quantity": -1,
    })

    crm = CRMServer()
    _reset(crm)
    _assert_rejected_without_mutation(crm, "create_deal", {
        "name": "invalid", "amount": -1, "contact_id": "missing",
    })


def test_crm_readonly_discovery_exposes_a_deletable_contact() -> None:
    crm = CRMServer()
    state = _reset(crm)

    result = _call(crm, "list_contacts", {})
    assert result["success"] is True
    candidates = [
        contact for contact in result["observation"]["contacts"]
        if contact["deletable"] is True
    ]
    assert candidates
    contact_id = candidates[0]["contact_id"]
    assert not any(
        task.get("contact_id") == contact_id
        for task in state["tasks"].values()
    )

    deleted = _call(crm, "delete_contact", {"contact_id": contact_id})
    assert deleted["success"] is True
    assert contact_id not in state["contacts"]


def test_invalid_cross_entity_transitions_are_rejected_without_state_change() -> None:
    crm = CRMServer()
    _reset(crm)
    _assert_rejected_without_mutation(crm, "create_task", {
        "title": "invalid", "contact_id": "missing",
    })

    calendar = CalendarServer()
    state = _reset(calendar)
    _assert_rejected_without_mutation(calendar, "respond_to_event", {
        "event_id": next(iter(state["events"])),
        "email": "nobody@example.com",
        "response": "accepted",
    })

    team_chat = TeamChatServer()
    state = _reset(team_chat)
    channel_id = next(iter(state["channels"]))
    state["channels"][channel_id]["archived"] = True
    _assert_rejected_without_mutation(team_chat, "send_message", {
        "channel_id": channel_id, "content": "not allowed",
    })

    email = EmailServer()
    state = _reset(email)
    _assert_rejected_without_mutation(email, "move_to_thread", {
        "email_id": next(iter(state["emails"])), "thread_id": "missing",
    })


def test_filesystem_protected_root_and_self_descendant_move_are_rejected() -> None:
    server = FilesystemServer()
    state = _reset(server)
    state["fs"]["/protected"] = {
        "type": "dir", "content": "", "permissions": "700", "owner": "root",
    }
    _assert_rejected_without_mutation(
        server, "rm", {"path": "/protected", "recursive": True},
    )
    _assert_rejected_without_mutation(server, "mv", {
        "source": "/home/user/data", "target": "/home/user/data/subdir",
    })


def test_issue_tracker_rejects_negative_time_and_deduplicates_sprint_membership() -> None:
    server = IssueTrackerServer()
    state = _reset(server)
    issue_id = next(iter(state["issues"]))
    _assert_rejected_without_mutation(server, "time_track", {
        "issue_id": issue_id, "hours": -1,
    })
    sprint = _call(server, "create_sprint", {
        "name": "audit", "start_date": "2026-01-01", "end_date": "2026-01-02",
    })["observation"]["sprint"]
    assert _call(server, "add_to_sprint", {
        "issue_id": issue_id, "sprint_id": sprint["sprint_id"],
    })["state_changed"] is True
    before = copy.deepcopy(state)
    duplicate = _call(server, "add_to_sprint", {
        "issue_id": issue_id, "sprint_id": sprint["sprint_id"],
    })
    assert duplicate["success"] is True
    assert duplicate["state_changed"] is False
    assert state == before
    assert state["sprints"][sprint["sprint_id"]]["issues"].count(issue_id) == 1


def test_food_delivery_rejects_repeat_rating_and_tip_without_mutation() -> None:
    server = FoodDeliveryServer()
    state = _reset(server)
    delivered = next(order for order in state["orders"].values() if order["status"] == "delivered")
    assert _call(server, "rate_order", {
        "order_id": delivered["order_id"], "rating": 5,
    })["success"] is True
    _assert_rejected_without_mutation(server, "rate_order", {
        "order_id": delivered["order_id"], "rating": 4,
    })

    untipped = next(order for order in state["orders"].values() if not order.get("tip"))
    assert _call(server, "add_tip", {
        "order_id": untipped["order_id"], "amount": 1,
    })["success"] is True
    _assert_rejected_without_mutation(server, "add_tip", {
        "order_id": untipped["order_id"], "amount": 1,
    })


def test_seeded_shopping_and_payment_relations_are_consistent() -> None:
    shopping = ShoppingServer()
    state = _reset(shopping)
    for order in state["orders"].values():
        assert order["total"] >= 0
        for item in order.get("items", []):
            assert {"product_id", "quantity", "unit_price"} <= set(item)
            assert item["quantity"] > 0

    payments = PaymentsServer()
    state = _reset(payments)
    for invoice_id, invoice in state["invoices"].items():
        payment_id = invoice.get("payment_id")
        if payment_id:
            assert payment_id in state["payments"]
            assert state["payments"][payment_id]["invoice_id"] == invoice_id
    for payment in state["payments"].values():
        assert payment["invoice_id"] in state["invoices"]
