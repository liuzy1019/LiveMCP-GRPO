"""Live-state grounded task samplers."""

from __future__ import annotations

import random
from typing import Any, Protocol

from src.live_mcp.dependency_graph import ToolChain
from src.live_mcp.manager import LiveMCPManager
from src.live_mcp.query_generator import StructuredTask


class LiveStateSampler(Protocol):
    def sample_task(
        self,
        session_id: str,
        manager: LiveMCPManager,
        difficulty: str,
        rng: random.Random,
    ) -> StructuredTask: ...


class CalendarSampler:
    def sample_task(
        self,
        session_id: str,
        manager: LiveMCPManager,
        difficulty: str,
        rng: random.Random,
    ) -> StructuredTask:
        response = manager.call_tool("calendar", session_id, "list_events", {})
        events = response["observation"]["events"]
        event = rng.choice(events)
        new_time = "2026-06-30T10:00"
        chain = ToolChain(
            chain_id="calendar:list_events->update_event",
            server_name="calendar",
            tools=["list_events", "update_event"],
            edges=[],
            difficulty=difficulty,
        )
        slots = {
            "event_id": event["event_id"],
            "title": event["title"],
            "old_time": event["start_time"],
            "new_time": new_time,
        }
        return StructuredTask(
            task_id="calendar_update_existing_event:template",
            server_name="calendar",
            tool_chain=chain,
            slots=slots,
            user_visible_slots=["title", "old_time", "new_time"],
            hidden_slots=["event_id"],
            success_criteria=[
                {
                    "type": "state_equals",
                    "server": "calendar",
                    "path": f"events.{event['event_id']}.start_time",
                    "value": new_time,
                }
            ],
            required_tools=["list_events", "update_event"],
            difficulty=difficulty,
        )


class ShoppingSampler:
    def sample_task(
        self,
        session_id: str,
        manager: LiveMCPManager,
        difficulty: str,
        rng: random.Random,
    ) -> StructuredTask:
        response = manager.call_tool("shopping", session_id, "search_products", {"category": "keyboard", "max_price": 100})
        products = response["observation"]["products"]
        product = rng.choice(products)
        chain = ToolChain(
            chain_id="shopping:search_products->add_to_cart->checkout",
            server_name="shopping",
            tools=["search_products", "add_to_cart", "checkout"],
            edges=[],
            difficulty=difficulty,
        )
        slots = {
            "product_id": product["product_id"],
            "category": product["category"],
            "max_price": 100,
            "quantity": 1,
        }
        return StructuredTask(
            task_id="shopping_buy_product:template",
            server_name="shopping",
            tool_chain=chain,
            slots=slots,
            user_visible_slots=["category", "max_price"],
            hidden_slots=["product_id"],
            success_criteria=[
                {"type": "order_contains_product", "server": "shopping", "product_id": product["product_id"]},
                {"type": "cart_empty", "server": "shopping"},
            ],
            required_tools=["search_products", "add_to_cart", "checkout"],
            difficulty=difficulty,
        )


class BankingSampler:
    def sample_task(
        self,
        session_id: str,
        manager: LiveMCPManager,
        difficulty: str,
        rng: random.Random,
    ) -> StructuredTask:
        response = manager.call_tool("banking", session_id, "get_account_info", {"account_id": "acc_savings"})
        response2 = manager.call_tool("banking", session_id, "get_account_info", {"account_id": "acc_checking"})
        chain = ToolChain(
            chain_id="banking:get_account_info->transfer->get_history",
            server_name="banking",
            tools=["get_account_info", "transfer", "get_history"],
            edges=[],
            difficulty=difficulty,
        )
        slots = {
            "from_account": "acc_savings",
            "to_account": "acc_checking",
            "amount": 100,
        }
        return StructuredTask(
            task_id="banking_transfer:template",
            server_name="banking",
            tool_chain=chain,
            slots=slots,
            user_visible_slots=["from_account", "to_account"],
            hidden_slots=["amount"],
            success_criteria=[
                {"type": "state_equals", "server": "banking", "path": "accounts.acc_checking.balance", "op": "gt", "value": 5000},
                {"type": "transaction_exists", "server": "banking"},
            ],
            required_tools=["get_account_info", "transfer"],
            difficulty=difficulty,
        )


class EmailSampler:
    def sample_task(
        self,
        session_id: str,
        manager: LiveMCPManager,
        difficulty: str,
        rng: random.Random,
    ) -> StructuredTask:
        response = manager.call_tool("email", session_id, "list_inbox", {})
        emails = response["observation"]["emails"]
        email = rng.choice(emails) if emails else {"email_id": "eml_0001", "thread_id": "thd_001", "sender": "boss@example.com"}
        chain = ToolChain(
            chain_id="email:list_inbox->add_label->send_email",
            server_name="email",
            tools=["list_inbox", "add_label", "send_email"],
            edges=[],
            difficulty=difficulty,
        )
        return StructuredTask(
            task_id="email_label_and_reply:template",
            server_name="email",
            tool_chain=chain,
            slots={"email_id": email["email_id"], "label": "follow-up"},
            user_visible_slots=["label"],
            hidden_slots=["email_id"],
            success_criteria=[
                {"type": "label_added", "server": "email", "email_id": email["email_id"], "label": "follow-up"},
            ],
            required_tools=["list_inbox", "add_label"],
            difficulty=difficulty,
        )


class FilesystemSampler:
    def sample_task(
        self,
        session_id: str,
        manager: LiveMCPManager,
        difficulty: str,
        rng: random.Random,
    ) -> StructuredTask:
        response = manager.call_tool("filesystem", session_id, "ls", {"path": "/home/user"})
        entries = response["observation"]["entries"]
        chain = ToolChain(
            chain_id="filesystem:ls->cd->ls",
            server_name="filesystem",
            tools=["ls", "cd", "ls"],
            edges=[],
            difficulty=difficulty,
        )
        return StructuredTask(
            task_id="filesystem_nav_and_read:template",
            server_name="filesystem",
            tool_chain=chain,
            slots={"path": "/home/user/projects"},
            user_visible_slots=["path"],
            hidden_slots=[],
            success_criteria=[
                {"type": "cwd_equals", "server": "filesystem", "path": "/home/user/projects"},
                {"type": "file_exists", "server": "filesystem", "path": "/home/user/projects/README.md"},
            ],
            required_tools=["ls", "cd", "cat"],
            difficulty=difficulty,
        )


class PaymentsSampler:
    def sample_task(
        self,
        session_id: str,
        manager: LiveMCPManager,
        difficulty: str,
        rng: random.Random,
    ) -> StructuredTask:
        response = manager.call_tool("payments", session_id, "list_invoices", {"status": "pending"})
        invoices = response["observation"]["invoices"]
        invoice = rng.choice(invoices) if invoices else {"invoice_id": "inv_0001", "customer": "Acme Corp", "amount": 1500}
        chain = ToolChain(
            chain_id="payments:list_invoices->pay_invoice",
            server_name="payments",
            tools=["list_invoices", "pay_invoice"],
            edges=[],
            difficulty=difficulty,
        )
        slots = {
            "invoice_id": invoice["invoice_id"],
            "customer": invoice["customer"],
            "amount": invoice["amount"],
        }
        return StructuredTask(
            task_id="payments_create_and_pay:template",
            server_name="payments",
            tool_chain=chain,
            slots=slots,
            user_visible_slots=["customer"],
            hidden_slots=["invoice_id", "amount"],
            success_criteria=[
                {"type": "state_equals", "server": "payments", "path": f"invoices.{invoice['invoice_id']}.status", "value": "paid"},
            ],
            required_tools=["list_invoices", "pay_invoice"],
            difficulty=difficulty,
        )


class CRMSampler:
    def sample_task(
        self,
        session_id: str,
        manager: LiveMCPManager,
        difficulty: str,
        rng: random.Random,
    ) -> StructuredTask:
        response = manager.call_tool("crm", session_id, "list_leads", {"status": "new"})
        leads = response["observation"]["leads"]
        lead = rng.choice(leads) if leads else {"lead_id": "lead_0001", "company": "TechStars"}
        chain = ToolChain(
            chain_id="crm:list_leads->convert_lead->create_deal",
            server_name="crm",
            tools=["list_leads", "convert_lead", "create_deal"],
            edges=[],
            difficulty=difficulty,
        )
        return StructuredTask(
            task_id="crm_lead_to_deal:template",
            server_name="crm",
            tool_chain=chain,
            slots={"lead_id": lead["lead_id"], "company": lead["company"], "amount": 10000},
            user_visible_slots=["company", "amount"],
            hidden_slots=["lead_id"],
            success_criteria=[
                {"type": "state_equals", "server": "crm", "path": f"leads.{lead['lead_id']}.status", "value": "converted"},
                {"type": "deal_exists_for_lead", "server": "crm", "lead_id": lead["lead_id"]},
            ],
            required_tools=["list_leads", "convert_lead", "create_deal"],
            difficulty=difficulty,
        )


class IssueTrackerSampler:
    def sample_task(
        self,
        session_id: str,
        manager: LiveMCPManager,
        difficulty: str,
        rng: random.Random,
    ) -> StructuredTask:
        response = manager.call_tool("issue_tracker", session_id, "list_issues", {"state": "open"})
        issues = response["observation"]["issues"]
        issue = rng.choice(issues) if issues else {"issue_id": "iss_0001", "title": "Login timeout", "priority": "high"}
        chain = ToolChain(
            chain_id="issue_tracker:list_issues->assign_issue->transition_issue",
            server_name="issue_tracker",
            tools=["list_issues", "assign_issue", "transition_issue"],
            edges=[],
            difficulty=difficulty,
        )
        return StructuredTask(
            task_id="issue_tracker_assign_and_start:template",
            server_name="issue_tracker",
            tool_chain=chain,
            slots={"issue_id": issue["issue_id"], "title": issue.get("title", ""), "assignee": "alice"},
            user_visible_slots=["title", "assignee"],
            hidden_slots=["issue_id"],
            success_criteria=[
                {"type": "state_equals", "server": "issue_tracker", "path": f"issues.{issue['issue_id']}.state", "value": "in_progress"},
                {"type": "state_equals", "server": "issue_tracker", "path": f"issues.{issue['issue_id']}.assignee", "value": "alice"},
            ],
            required_tools=["list_issues", "assign_issue", "transition_issue"],
            difficulty=difficulty,
        )


class TeamChatSampler:
    def sample_task(
        self,
        session_id: str,
        manager: LiveMCPManager,
        difficulty: str,
        rng: random.Random,
    ) -> StructuredTask:
        response = manager.call_tool("team_chat", session_id, "list_channels", {})
        channels = response["observation"]["channels"]
        channel = rng.choice(channels) if channels else {"channel_id": "ch_general", "name": "general"}
        chain = ToolChain(
            chain_id="team_chat:list_channels->send_message",
            server_name="team_chat",
            tools=["list_channels", "send_message"],
            edges=[],
            difficulty=difficulty,
        )
        return StructuredTask(
            task_id="team_chat_send_and_thread:template",
            server_name="team_chat",
            tool_chain=chain,
            slots={"channel_id": channel["channel_id"], "channel": channel["name"], "topic": "deploy status"},
            user_visible_slots=["channel", "topic"],
            hidden_slots=["channel_id"],
            success_criteria=[
                {"type": "message_sent", "server": "team_chat", "channel_id": channel["channel_id"]},
            ],
            required_tools=["list_channels", "send_message"],
            difficulty=difficulty,
        )


class FoodDeliverySampler:
    def sample_task(
        self,
        session_id: str,
        manager: LiveMCPManager,
        difficulty: str,
        rng: random.Random,
    ) -> StructuredTask:
        response = manager.call_tool("food_delivery", session_id, "list_restaurants", {"cuisine": "Italian"})
        restaurants = response["observation"]["restaurants"]
        restaurant = rng.choice(restaurants) if restaurants else {"restaurant_id": "rest_001", "name": "Pizza Palace"}
        chain = ToolChain(
            chain_id="food_delivery:list_restaurants->get_menu->create_order",
            server_name="food_delivery",
            tools=["list_restaurants", "get_menu", "create_order"],
            edges=[],
            difficulty=difficulty,
        )
        return StructuredTask(
            task_id="food_delivery_order:template",
            server_name="food_delivery",
            tool_chain=chain,
            slots={"restaurant_id": restaurant["restaurant_id"], "cuisine": "Italian", "address": "123 Main St"},
            user_visible_slots=["cuisine", "address"],
            hidden_slots=["restaurant_id"],
            success_criteria=[
                {"type": "order_exists", "server": "food_delivery", "status": "placed"},
            ],
            required_tools=["list_restaurants", "get_menu", "create_order"],
            difficulty=difficulty,
        )
