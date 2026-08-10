"""Prompt contracts and value objects shared by Teacher stages."""

from __future__ import annotations

import datetime as _datetime
import random
import re
from dataclasses import dataclass, field
from typing import Any


VALID_TERMINALS: tuple[str, ...] = (
    "final_answer", "report_error", "ask_clarification",
)


# ═══════════════════════════════════════════════════════════════════════
# Domain descriptions
# ═══════════════════════════════════════════════════════════════════════

DOMAIN_DESCRIPTIONS: dict[str, str] = {
    "calendar": (
        "Calendar assistant. Users ask about their schedule, need to find free slots "
        "across multiple people, book/change/cancel meetings, check who's attending, "
        "handle recurring events, or fix scheduling conflicts. Requests are often "
        "time-sensitive: 'is Thursday free?', 'move my 3pm to Friday', 'is anyone "
        "free at lunch next week?'"
    ),
    "shopping": (
        "Online store. Users browse products by category/price, compare items, manage "
        "their cart and wishlist, apply coupons, check orders, track packages, return "
        "items, leave reviews. Common requests: 'find me a good keyboard under $100', "
        "'what's in my cart?', 'has my order shipped?', 'return the blue one'. "
        "Prices, stock, and order statuses are live."
    ),
    "banking": (
        "Personal banking. Users check balances, review transactions, transfer money "
        "between their own accounts, pay bills, send wire transfers, freeze cards. "
        "Typical requests: 'how much do I have in savings?', 'did my paycheck come in?', "
        "'move $200 to checking', 'pay my rent bill', 'is my card frozen?' "
        "Account IDs and balances come from live state. Some accounts may be frozen."
    ),
    "email": (
        "Email inbox. Users read/search emails, send replies, forward threads, "
        "manage labels, archive old messages. Common requests: 'show me unread from "
        "my boss', 'reply to that thread about the budget', 'label all the Q3 reports "
        "as important', 'find the email with the contract attachment'. "
        "Emails have IDs, subjects, senders, labels, read/unread status."
    ),
    "filesystem": (
        "Remote file server. Users navigate directories, read/edit/move files, check "
        "disk space, set permissions, find files by name or content. Requests like: "
        "'what's in /home/user/projects?', 'find all .log files >10MB', 'move the "
        "config to /etc/app/', 'who owns this file?', 'make it readable by everyone'. "
        "Protected paths exist — some operations may fail."
    ),
    "payments": (
        "Business payments. Users manage invoices (create, view, pay, refund, cancel, "
        "dispute), set up webhooks for payment events. Requests: 'send invoice #42 to "
        "the client', 'has the wire for inv_005 cleared?', 'refund that overcharge', "
        "'dispute inv_099 — wrong amount'. Invoices flow through statuses: pending → "
        "paid → refunded/cancelled/disputed."
    ),
    "crm": (
        "Sales CRM. Users track leads through pipeline stages, manage contacts and "
        "deals, log tasks/notes. Requests: 'which leads are stuck in qualified?', "
        "'convert lead_023 to a deal', 'who did I call last week?', 'move the "
        "Acme deal to negotiation'. Leads: new→contacted→qualified→converted/lost."
    ),
    "issue_tracker": (
        "Project issue tracker. Users create/assign/triage bugs and tasks, update "
        "status, add labels/watchers/comments, manage sprints. Requests: 'who's on "
        "bug #432?', 'move all login bugs to in_progress', 'what's in sprint 14?', "
        "'label this as critical + frontend', 'close the ones I fixed yesterday'."
    ),
    "team_chat": (
        "Team messaging. Users join channels, send messages, reply in threads, "
        "react with emoji, search history. Requests: 'what did Sarah say in "
        "#engineering?', 'post the update to #releases', 'search for the Q2 roadmap "
        "discussion', 'join the new project channel'. Messages are append-only."
    ),
    "food_delivery": (
        "Food delivery app. Users browse restaurants/menus, filter by dietary needs, "
        "place/cancel/track orders, rate meals, tip drivers. Requests: 'order sushi "
        "from that place on 14th st', 'where's my pad thai?', 'cancel the pizza before "
        "they start making it', 'reorder what I had last Friday'. Orders flow: "
        "placed→confirmed→preparing→delivering→delivered (with cancellation only before preparing)."
    ),
}

DIFFICULTY_DESCRIPTIONS: dict[str, str] = {
    "complete": (
        "The user knows exactly what they want and provides all user-known "
        "information needed for the desired outcome. Use a specific existing ID "
        "only when that entity is naturally known before execution; an ID produced "
        "by an earlier chain capability does not belong in the user message. "
        "No follow-up needed. "
        "Example tone: 'move $500 from acc_01 to checking', "
        "'cancel evt_042', 'cat /home/user/report.txt'."
    ),
    "missing": (
        "The user forgets ONE critical detail — like scheduling a meeting but "
        "not saying when, requesting a transfer without the destination, or "
        "asking to label an email without saying which one. It reads like a "
        "real person forgetting, not a puzzle."
    ),
    "minimal": (
        "The user sends a terse message — just intent, no specifics. Like a "
        "quick text: 'check my schedule', 'find the invoice', 'pay rent'. "
        "No entity IDs, no parameters — just what they want done."
    ),
}


@dataclass(frozen=True)
class GeneratedQuery:
    """Auditable result of chain-seeded query synthesis."""

    user_query: str
    target_capability: str
    chain_supported: bool
    attempts: int
    mutation_evidence: list[dict[str, Any]] = None
    dependency_evidence: list[str] = None
    initial_goal: str = ""
    initial_goal_grounding_basis: dict[str, Any] = None
    initial_goal_causal_steps: list[str] = None
    initial_goal_planning_attempts: int = 0


def _chain_goal_phrase(final_tool: str) -> str:
    """Natural-language outcome hint for chain-seeded query generation.

    The hint guides the user goal without exposing tool names or forcing the
    prompt to list internal workflow steps.
    """
    name = final_tool.lower()
    explicit: dict[str, str] = {
        "checkout": "place or complete an order from the cart",
        "return_order": "return an order",
        "track_order": "check an order's shipping or delivery status",
        "refund_invoice": "refund an invoice",
        "pay_invoice": "pay an invoice",
        "dispute_invoice": "dispute an invoice",
        "cancel_payment": "cancel a pending payment",
        "complete_task": "mark a CRM task complete",
        "create_thread": "start a thread from a message",
        "react_message": "add a reaction to a message",
        "rate_order": "rate a delivered food order",
        "add_tip": "add a tip to a food order",
        "cancel_order": "cancel a food order",
        "update_order_status": "update a food order's status",
        "remove_from_wishlist": "remove an item from the wishlist",
        "remove_from_cart": "remove an item from the cart",
        "update_cart_quantity": "change the quantity of an item in the cart",
        "clear_cart": "clear the cart",
        "get_balance": "check an account balance",
        "get_history": "check an account transaction history",
        "get_statement": "get an account statement",
        "list_orders": "list existing orders",
        "get_order": "get order details",
        "search_events": "search calendar events",
        "list_events": "list calendar events",
    }
    if name in explicit:
        return explicit[name]
    verb_map = {
        "create_": "create",
        "update_": "update",
        "delete_": "delete",
        "remove_": "remove",
        "add_": "add",
        "send_": "send",
        "reply_": "reply to",
        "forward_": "forward",
        "archive_": "archive",
        "mark_": "mark",
        "set_": "set",
        "assign_": "assign",
        "transition_": "change the status of",
        "convert_": "convert",
        "freeze_": "freeze",
        "unfreeze_": "unfreeze",
        "get_": "get",
        "list_": "list",
        "search_": "search for",
        "find_": "find",
    }
    for prefix, verb in verb_map.items():
        if name.startswith(prefix):
            entity = name[len(prefix):].replace("_", " ")
            return f"{verb} {entity}".strip()
    return name.replace("_", " ")


def _target_tool_requirement(
    tool_schemas: list[dict[str, Any]],
    tool_name: str,
) -> str:
    """Describe an internal chain target without leaking it into user text."""
    schema = next(
        (tool for tool in tool_schemas if tool.get("name") == tool_name),
        {},
    )
    description = str(schema.get("description") or "").strip()
    annotations = schema.get("annotations") or {}
    readonly_note = ""
    if annotations.get("readonly") is True and annotations.get("mutating") is False:
        readonly_note = (
            " This capability is read-only and does not modify server state. "
            "Any transformed content is display-only output and is never persisted."
        )
    required = list(schema.get("input_schema", {}).get("required", []) or [])
    required_text = ", ".join(str(name) for name in required) or "none"
    return (
        f"Internal target capability: {tool_name}\n"
        f"Capability description: {description or _chain_goal_phrase(tool_name)}{readonly_note}\n"
        f"Required information fields: {required_text}"
    )


# ═══════════════════════════════════════════════════════════════════════
# Persona templates and reference dates.
# ═══════════════════════════════════════════════════════════════════════

_PERSONA_TEMPLATES: list[str] = [
    # ── Professional roles ──
    "a busy team lead with back-to-back meetings (direct, short requests)",
    "a freelancer juggling multiple client projects (casual but specific)",
    "a graduate student deep in research (asks for lookups and re-formatting)",
    "a small business owner handling everything themselves (practical, to-the-point)",
    "a project manager coordinating across timezones (schedule-aware, mentions dates)",
    "an executive assistant clearing a backlog (batch-style: 'handle all the X')",
    "a software engineer debugging in prod (technical, knows exact file paths and IDs)",
    "a marketing manager launching a campaign (deadline-driven, 'need this done by EOD')",
    "a data analyst pulling reports (asks for aggregation: 'how many X since Y?')",
    "a customer support agent triaging tickets (urgency cues: 'this one is on fire')",
    # ── Casual / everyday users ──
    "someone in a hurry on their phone (typos, fragments, no punctuation)",
    "a non-technical user who doesn't know exact names of things (descriptive: 'that blue thing')",
    "a frustrated customer whose order is wrong (emotion: 'this is the third time')",
    "an older relative learning the system (polite, over-explains, 'can you help me with...')",
    "a user who sends one-line texts like a chat (minimal words, no greeting)",
]

_REFERENCE_DATES: list[str] = [
    "Thursday, January 15, 2026",
    "Sunday, March 8, 2026",
    "Saturday, June 20, 2026",
    "Wednesday, September 16, 2026",
    "Thursday, November 12, 2026",
    "Tuesday, February 3, 2026",
    "Sunday, April 26, 2026",
    "Thursday, July 30, 2026",
    "Tuesday, October 13, 2026",
    "Saturday, December 5, 2026",
]


def reference_date_for_seed(seed: int) -> str:
    """Return the shared temporal anchor for generation and seeded state."""
    return _REFERENCE_DATES[
        (seed // len(_PERSONA_TEMPLATES)) % len(_REFERENCE_DATES)
    ]


def reference_datetime_for_seed(seed: int) -> _datetime.datetime:
    return _datetime.datetime.strptime(
        reference_date_for_seed(seed), "%A, %B %d, %Y",
    )

# ═══════════════════════════════════════════════════════════════════════
# Turn-decay schedule:
#   min_turns=2, max_turns=3 for CONVERSATION ROUNDS — i.e. user turns)
#
# NOTE ON TERMINOLOGY (avoids confusion with _run_turn_loop's max_turns=8):
#   * MIN/MAX_CONVERSATION_ROUNDS (this file)  = # of user turns per task
#     Conversation-round bounds: 2..3.
#   * max_turns argument in orchestrator._run_turn_loop / generate_one
#     = # of teacher decision STEPS inside a single conversation round
#       (tool-call attempts + retries + terminal),
#       Step 3.5.  It only bounds the state-machine's inner loop so it
#       terminates on stuck LLMs.  RL rollout uses a verified per-row action
#       budget for generated data; the configured max is only a runtime default
#       independent of conversation-round count.
# ═══════════════════════════════════════════════════════════════════════

class ContinuationPolicy:
    """Turn-decay schedule for deciding when to end a conversation.

    Normal tasks use two to three conversation rounds. A dependency chain seeds
    one complete user goal; follow-up turns continue that goal instead of
    exposing individual internal chain nodes as separate user requests.

    Perturbations (intermittent errors, pagination) may add 1-2 extra turns.
    """

    # min_turns=2 means at least two conversation rounds for normal success.
    # must span at least two user turns with at least one follow-up).
    MIN_CONVERSATION_ROUNDS = 2
    MAX_CONVERSATION_ROUNDS = 3
    # P1-3: Per-turn turn-decay probabilities for middle rounds.
    # Local continuation probabilities:
    # the exact turn-decay schedule.  These are local defaults configurable via
    # experiment metadata.
    CLARIFICATION_PROB: float = 0.10   # probability of clarification at middle rounds
    END_PROB_BASE: float = 0.30        # base end probability, increases with round

    @staticmethod
    def sample_continuation_decision(
        rounds_done: int,
        rng: random.Random,
    ) -> str:
        """Sample the continuation decision for one conversation round.

        Returns one of {"end", "follow_up", "clarification"}.

        Rules:
        - Before min_turns: cannot end. Sample follow_up vs clarification.
        - At or after max_turns: must end.
        - Middle rounds: turn-decay schedule (P(end) increases with round).

        The probabilities are configurable via CLARIFICATION_PROB and
        END_PROB_BASE class attributes.
        """
        min_r = ContinuationPolicy.MIN_CONVERSATION_ROUNDS
        max_r = ContinuationPolicy.MAX_CONVERSATION_ROUNDS

        if rounds_done >= max_r:
            return "end"
        if rounds_done < min_r:
            # Cannot end — must continue.  Mostly follow_up, occasionally clarify.
            return (
                "clarification"
                if rng.random() < ContinuationPolicy.CLARIFICATION_PROB
                else "follow_up"
            )
        # Middle rounds: turn-decay end probability
        t = (rounds_done - min_r) / max(1, max_r - min_r)
        end_prob = min(0.70, ContinuationPolicy.END_PROB_BASE + 0.2 * t)
        r = rng.random()
        if r < end_prob:
            return "end"
        remaining = 1.0 - end_prob
        if r < end_prob + ContinuationPolicy.CLARIFICATION_PROB:
            return "clarification"
        return "follow_up"

# ═══════════════════════════════════════════════════════════════════════
# TaskPlanner — LLM-in-the-loop state machine
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class ActionPlan:
    """A single action decided by the LLM."""
    action: str          # "tool_call" | "final_answer" | "report_error" | "ask_clarification"
    tool_name: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)
    text: str = ""       # terminal text / error reason / clarification question


_DIRECT_USER_QUESTION_RE = re.compile(
    r"(?:^|[.!;:]\s+|—\s*|\n\s*)"
    r"(?:"
    r"(?:would|could|can|will|may|should|do|did|are)\s+you\b"
    r"|(?:what|which|who|when|where|why|how)\b[^?？\n]{0,160}"
    r"\b(?:you|should\s+i|can\s+i|do\s+i)\b"
    r")"
    r"[^?？\n]{0,240}[?？]",
    re.IGNORECASE,
)
_DIRECT_USER_IMPERATIVE_RE = re.compile(
    r"(?:^|[.!;:]\s+|—\s*|\n\s*)"
    r"(?:to\s+[^.!?\n]{1,80}[,;:]\s*)?"
    r"(?:please\s+)?"
    r"(?:provide|specify|confirm|choose|identify|tell\s+me|let\s+me\s+know)\b",
    re.IGNORECASE,
)
_COURTESY_OFFER_RE = re.compile(
    r"(?:please\s+)?let\s+me\s+know\s+if\s+you\s+"
    r"(?:need|want|would\s+like)\b[^.!?\n]*[.!]?",
    re.IGNORECASE,
)


def _final_answer_requests_user_input(text: str) -> bool:
    """Return whether a final answer directly asks the user for new input.

    This is deliberately a narrow terminal-format check, not a semantic judge.
    It recognizes direct interrogatives and imperatives while allowing quoted
    question-bearing titles or history inside an otherwise completed answer.
    """
    normalized = re.sub(r"[ \t]+", " ", text).strip()
    if not normalized:
        return False
    # A completed answer may end with a conventional offer of further help.
    # It does not request information needed to complete the current task.
    normalized = _COURTESY_OFFER_RE.sub("", normalized)
    return bool(
        _DIRECT_USER_QUESTION_RE.search(normalized)
        or _DIRECT_USER_IMPERATIVE_RE.search(normalized)
    )
