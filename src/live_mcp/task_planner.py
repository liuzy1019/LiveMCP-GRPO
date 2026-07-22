"""State-machine task generation for live MCP environments.

LLM-in-the-loop at every turn: the LLM sees domain + tool schemas + live state
+ full execution history, and decides the next action (tool_call with arguments,
or terminal).  Oracle trace is the recorded interaction — no heuristic parameter
inference needed.

Pipeline per task:
  1. create_session(seed) — fresh isolated state
  2. LLM generates user_query
  3. Per-round action loop (tool/terminal action budget):
     a. LLM decides next action: tool_call(name, args) | final_answer | report_error
     b. Execute tool_call against live MCP → record observation
     c. Apply execution perturbations (intermittent errors, pagination, …)
     d. Append to history
  4. Derive success criteria from state delta
  5. Replay validation against fresh session
  6. Robustness plan is fixed before Teacher processing and reused by Replay
"""

from __future__ import annotations

import datetime as _datetime
import fcntl
import hashlib
import json as _json
import os
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TYPE_CHECKING

from loguru import logger

from src.live_mcp.observation import (
    DEFAULT_TEACHER_OBSERVATION_CHARS,
    project_observation,
)
from src.live_mcp.tool_semantics import is_mutating_tool
from src.live_mcp.types import OracleCall
from src.utils import extract_json as _extract_json

if TYPE_CHECKING:
    from src.live_mcp.llm_client import LLMClient
    from src.live_mcp.manager import LiveMCPManager
    from src.live_mcp.executor import LiveMCPExecutor


# Mutations whose effects are confined to the current MCP execution context,
# rather than a user-visible business/resource outcome. They remain mutating
# for execution/state tracking, but do not require a separate authorization
# span in the synthesized user request.
_SESSION_INTERNAL_MUTATIONS: frozenset[str] = frozenset({"cd"})


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
    r"(?:please\s+)?"
    r"(?:provide|specify|confirm|choose|identify|tell\s+me|let\s+me\s+know)\b",
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
    return bool(
        _DIRECT_USER_QUESTION_RE.search(normalized)
        or _DIRECT_USER_IMPERATIVE_RE.search(normalized)
    )


class TaskPlanner:
    """State-machine teacher for query and action generation.

    The LLM is called at EVERY turn with full context (domain, tools, live state,
    execution history) and decides the next action.  Parameters come from the LLM's
    understanding of real state values, not from heuristic inference.
    """

    # These stages all return one small JSON object.  The generic client
    # default (1024 tokens) reserves far more decode KV than these contracts
    # need and reduces vLLM batching headroom.  Keep conservative margins over
    # the largest responses observed in full ten-domain traces.
    STAGE_MAX_TOKENS: dict[str, int] = {
        "query_generation": 256,
        "action_decision": 384,
        "continuation_generation": 256,
        "clarification_generation": 192,
        "recovery_decision": 384,
    }

    def __init__(
        self,
        client: "LLMClient",
        domain: str,
        seed: int = 0,
        max_observation_chars: int = DEFAULT_TEACHER_OBSERVATION_CHARS,
    ):
        self.client = client
        self.domain = domain
        self.seed = int(seed)
        self.max_observation_chars = max(256, int(max_observation_chars))
        self.domain_desc = DOMAIN_DESCRIPTIONS.get(domain, "")
        trace_setting = os.environ.get("LIVEMCP_TEACHER_TRACE_PATH", "").strip()
        self._trace_path: Path | None = None
        if trace_setting:
            project_root = Path(__file__).resolve().parents[2]
            candidate = Path(trace_setting)
            candidate = candidate if candidate.is_absolute() else project_root / candidate
            candidate = candidate.resolve()
            logs_root = (project_root / "logs").resolve()
            try:
                candidate.relative_to(logs_root)
            except ValueError:
                logger.warning(
                    "Ignoring LIVEMCP_TEACHER_TRACE_PATH outside project logs/: {}",
                    candidate,
                )
            else:
                candidate.parent.mkdir(parents=True, exist_ok=True)
                self._trace_path = candidate
        self.trace_includes_state = (
            self._trace_path is not None
            and os.environ.get(
                "LIVEMCP_TEACHER_TRACE_INCLUDE_STATE", "0",
            ).strip().lower() in {"1", "true", "yes", "on"}
        )

    def _generate_chat(
        self,
        stage: str,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        max_tokens: int | None = None,
        json_mode: bool = True,
    ) -> str:
        """Call the Teacher and optionally persist the exact inference boundary."""
        effective_max_tokens = (
            max_tokens
            if max_tokens is not None
            else self.STAGE_MAX_TOKENS.get(stage)
        )
        raw = ""
        error = ""
        try:
            kwargs: dict[str, Any] = {"temperature": temperature}
            kwargs["json_mode"] = json_mode
            if effective_max_tokens is not None:
                kwargs["max_tokens"] = effective_max_tokens
            raw = self.client.generate_chat(messages, **kwargs)
            return raw
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            if self._trace_path is not None:
                serialized_messages = _json.dumps(
                    messages, ensure_ascii=False, default=str,
                )
                self._append_trace_event({
                    "timestamp": _datetime.datetime.now(
                        _datetime.timezone.utc
                    ).isoformat(),
                    "domain": self.domain,
                    "seed": self.seed,
                    "stage": stage,
                    "temperature": temperature,
                    "max_tokens": effective_max_tokens,
                    "messages": messages,
                    "prompt_chars": len(serialized_messages),
                    "prompt_sha256": hashlib.sha256(
                        serialized_messages.encode("utf-8")
                    ).hexdigest(),
                    "raw_response": raw,
                    "error": error,
                })

    def record_environment_event(self, stage: str, **payload: Any) -> None:
        """Append parsed actions and real MCP feedback to the same audit trace."""
        if self._trace_path is None:
            return
        self._append_trace_event({
            "timestamp": _datetime.datetime.now(
                _datetime.timezone.utc
            ).isoformat(),
            "domain": self.domain,
            "seed": self.seed,
            "stage": stage,
            **payload,
        })

    def _append_trace_event(self, event: dict[str, Any]) -> None:
        if self._trace_path is None:
            return
        with self._trace_path.open("a", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            handle.write(_json.dumps(event, ensure_ascii=False, default=str) + "\n")
            handle.flush()
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    # ── Step 1: generate user query ──

    def generate_query(
        self,
        tool_schemas: list[dict[str, Any]],
        grounded_state: dict[str, Any],
        difficulty: str,
        rng: random.Random,
        dep_hints: str = "",
        persona: str = "",
        reference_date: str = "",
        chain_seed: list[str] | None = None,
        chain_context: dict[str, Any] | None = None,
    ) -> GeneratedQuery:
        """LLM generates a natural-language user query grounded in live state.

        Persona and reference_date increase query diversity. chain_seed guides
        the query toward a realistic dependency chain without making that
        chain the only valid tool trajectory.

        chain_context is a compact subset of live-state entity IDs extracted by
        _extract_chain_context(). The grounding constraint prevents ID invention.
        """
        difficulty_desc = DIFFICULTY_DESCRIPTIONS.get(
            difficulty, DIFFICULTY_DESCRIPTIONS["complete"]
        )
        state_text = _format_state_compact(grounded_state, max_entities=20)

        # Date context
        date_block = ""
        if reference_date:
            date_block = (
                f"\n## Reference Date\nToday is {reference_date}. Use relative "
                "dates when appropriate, but keep every relative date, stated "
                "weekday, and recurrence rule calendar-consistent.\n"
            )

        system = (
            "You are role-playing as a real person messaging their AI assistant. "
            "Write ONE short message — the way a real human would actually type it. "
            "Real people state what they WANT, not HOW to do it. "
            "They don't list steps, don't mention tool names, don't describe workflows. "
            "They just say their goal in 1-2 sentences max.\n\n"
            "BAD (AI-like): 'I need to search for events, then create a new one, then add attendees.'\n"
            "GOOD (human-like): 'set up my project sync next Tuesday at 2pm'\n\n"
            "BAD: 'First verify the account, then check the balance, then transfer funds.'\n"
            "GOOD: 'move $200 from savings to checking'"
        )
        if difficulty == "minimal":
            grounding_line = (
                "Do NOT include entity IDs — just express your intent naturally."
            )
        elif difficulty == "complete":
            grounding_line = (
                "Include all user-known information needed to complete the goal. "
                "If you naturally reference an entity that already exists, copy its exact ID "
                "from Current State. Do not substitute an existing ID for an entity that an "
                "earlier capability in the dependency chain will create. If an earlier discovery "
                "capability intentionally hides IDs, use enough exact shown selector facts to "
                "identify one candidate uniquely. A label such as 'savings', 'the invoice', or "
                "'that email' is not complete when multiple shown candidates match; choose a "
                "unique shown selector or return UNSAT."
            )
        else:
            grounding_line = (
                "You forgot one key detail. Use IDs from Current State where you remember them, "
                "but leave out the missing piece naturally — don't signal that you're omitting it."
            )

        # Priority rule: when persona style conflicts with difficulty constraints,
        # difficulty wins for structure (what info to include/omit), persona wins
        # for voice (how it's phrased).
        priority_note = (
            "\nIMPORTANT: If your persona style conflicts with the difficulty level, "
            "follow the difficulty for WHAT to include, but keep the persona's VOICE and TONE."
        )

        # ── The complete dependency chain seeds one task. ──
        chain_goal_block = ""
        if chain_seed:
            final_tool = chain_seed[-1]
            goal_phrase = _chain_goal_phrase(final_tool)
            if goal_phrase:
                chain_requirements = "\n".join(
                    f"- {_target_tool_requirement(tool_schemas, tool_name)}"
                    for tool_name in chain_seed
                )
                chain_fact_note = ""
                if (
                    self.domain == "team_chat"
                    and "create_thread" in chain_seed
                    and "react_message" in chain_seed
                    and chain_seed.index("create_thread") < chain_seed.index("react_message")
                ):
                    chain_fact_note = (
                        " Handler fact: a newly created thread starts with no replies. "
                        "A requested reaction must therefore target the existing root "
                        "message; do not request a reaction to a new thread reply."
                    )
                chain_goal_block = (
                    "\n## Complete Task Goal (internal synthesis guide)\n"
                    f"The grounded dependency chain is: {chain_seed}.\n"
                    f"Capabilities in execution order:\n{chain_requirements}\n"
                    f"The message MUST clearly request the final outcome: {goal_phrase}. "
                    f"Return target_capability exactly as {final_tool!r}. If you cannot "
                    "write a natural request for this target, return user_query as "
                    "'UNSAT' instead of switching to another task. "
                    "The request must preserve the dependency: do not give the user "
                    "an ID or value that an earlier capability is supposed to discover "
                    "or produce. Each earlier capability must remain a plausible internal "
                    "prerequisite for the final outcome. If that cannot be expressed as "
                    "one natural user goal, return UNSAT. Set chain_supported=true only "
                    "when these conditions hold. "
                    "Treat each entity label as a typed resource: an invoice ID is not "
                    "a payment ID, order ID, or any other ID type. Choose only entities "
                    "whose shown status and numeric facts satisfy the target capability. "
                    "For a state-dependent amount, use the exact shown amount or remaining "
                    "allowance; do not invent a convenient value. "
                    "Read-only earlier items may be internal prerequisites. A dependency "
                    "edge does not itself grant permission for an unrelated side effect: "
                    "state the final user-visible outcome naturally, and mention an earlier "
                    "state change only when it is genuinely part of that same requested "
                    "outcome. If the final outcome cannot be requested without an unrelated "
                    "mutation, return UNSAT instead of listing internal workflow steps. "
                    "Do not mention tool names or split chain nodes into future requests. "
                    "Session-internal navigation such as changing the current working "
                    "directory may remain implicit only when it is an earlier execution "
                    "detail rather than the final requested outcome. "
                    "Every required field that controls a state change MUST have a "
                    "concrete user-authorized value in the message or an unambiguous "
                    "value in Current State; never leave such a field for the assistant "
                    "to invent (for example an amount, recipient, status, path, date, "
                    "quantity, or message body).\n"
                    f"{chain_fact_note}\n"
                )

        # ── Grounded-entity constraint ──
        # Chain-aligned entity context: only these IDs are known to exist
        # in the live state.  The teacher MUST reference only these, never
        # invent IDs.  Without this constraint, the LLM hallucinates entity
        # IDs that don't exist in any seed, causing 100% replay failure.
        anti_halluc_block = ""
        if chain_context and chain_context.get("query_grounding_summaries"):
            summaries_text = "\n".join(
                chain_context["query_grounding_summaries"][:30]
            )
            hidden_types = chain_context.get("opaque_id_hidden_types", [])
            if hidden_types:
                anti_halluc_block = (
                    f"\n## Chain-Aligned Grounded Candidates\n"
                    f"{summaries_text}\n\n"
                    f"Opaque IDs are intentionally hidden for these entity types: "
                    f"{hidden_types}. Refer to them only through the shown natural "
                    f"selectors (name, customer, status, category, amount, etc.). "
                    f"When using a name or other selector, copy an exact shown value; "
                    f"do not invent a person, customer, or resource that is absent. "
                    f"Do NOT invent an ID or copy an ID for another entity. The "
                    f"assistant must discover the exact ID with the earlier read/list/search "
                    f"capability in the sampled chain.\n"
                )
            else:
                anti_halluc_block = (
                    f"\n## Chain-Aligned Entities (ONLY these IDs exist)\n"
                    f"{summaries_text}\n\n"
                    f"⚠️ CRITICAL: You MUST ONLY reference entity IDs from this list "
                    f"or from Current State above. Do NOT invent, modify, or guess IDs. "
                    f"If an ID is 'evt_001', write 'evt_001' — not 'evt_005' or 'evt_0001'.\n"
                )
        elif difficulty == "complete":
            # Even without chain_context, enforce anti-hallucination on
            # the full state for complete-difficulty tasks.
            anti_halluc_block = (
                "\n## Anti-Hallucination Rule\n"
                "⚠️ CRITICAL: Reference ONLY entity IDs that appear in Current State "
                "above. Do NOT invent or modify IDs. Copy them exactly as shown.\n"
            )

        user = f"""## Persona
{persona if persona else 'A normal user messaging their AI assistant.'}
{date_block}
## What this assistant can help with
{self.domain_desc}
{dep_hints}
## Current State (real IDs and values)
{state_text}
{chain_goal_block}
{anti_halluc_block}
## Your task
Type ONE message to your assistant. Difficulty: {difficulty}. {difficulty_desc}

{grounding_line}{priority_note}
Remember: state your GOAL, not the steps. One message, 1-2 sentences max.

Return only:
{{"user_query": "<the message or UNSAT>", "target_capability": "<sampled chain final capability>", "chain_supported": <true or false>, "mutation_evidence": [{{"capability": "<final state-changing target capability>", "query_span": "<exact words from user_query that authorize it>"}}]}}

If the final requested capability changes user-visible state, mutation_evidence
must contain an item for that final capability. Internal dependency-chain
mutations are not separate user goals and do not require one item per node.
If cd is the final requested capability, it still needs evidence. query_span must
be copied verbatim from user_query and directly authorize the stated goal. Do not
invent an evidence span for an operation the user did not request. Read-only
capabilities must not appear in mutation_evidence.
"""
        tools_by_name = {
            str(tool.get("name") or ""): tool for tool in tool_schemas
        }
        target_tool = chain_seed[-1] if chain_seed else ""
        target_is_mutating = (
            (tools_by_name.get(target_tool, {}).get("annotations") or {}).get(
                "mutating"
            )
            is True
        )
        expected_mutations = {target_tool} if target_tool and target_is_mutating else set()
        for attempt in range(3):
            try:
                raw = self._generate_chat(
                    "query_generation",
                    [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
                    temperature=0.4 + 0.1 * attempt,
                )
                data = _extract_json(raw)
                if not isinstance(data, dict):
                    continue
                query = data.get("user_query", "")
                target_capability = str(data.get("target_capability", "")).strip()
                chain_supported = data.get("chain_supported") is True
                mutation_evidence = data.get("mutation_evidence", [])
                expected_target = chain_seed[-1] if chain_seed else ""
                if str(query).strip().upper() == "UNSAT":
                    logger.debug(
                        f"generate_query attempt {attempt + 1}/3 reported UNSAT "
                        f"for {self.domain} chain={chain_seed}"
                    )
                    # UNSAT is a structured judgment about this fixed
                    # chain/state pair. Retrying the identical prompt with a
                    # higher temperature cannot make the sampled task
                    # satisfiable; let pool-level oversampling choose a fresh
                    # seed and dependency chain instead.
                    break
                if expected_target and target_capability != expected_target:
                    logger.debug(
                        f"generate_query attempt {attempt + 1}/3 target mismatch "
                        f"for {self.domain}: expected={expected_target!r}, "
                        f"got={target_capability!r}"
                    )
                    continue
                if expected_target and not chain_supported:
                    logger.debug(
                        f"generate_query attempt {attempt + 1}/3 did not support "
                        f"the sampled chain for {self.domain}: chain={chain_seed}"
                    )
                    continue
                if expected_mutations:
                    if not isinstance(mutation_evidence, list):
                        continue
                    evidence_by_capability: dict[str, str] = {}
                    for item in mutation_evidence:
                        if not isinstance(item, dict):
                            continue
                        capability = str(item.get("capability") or "").strip()
                        query_span = str(item.get("query_span") or "").strip()
                        if capability and query_span:
                            evidence_by_capability[capability] = query_span
                    if not expected_mutations.issubset(evidence_by_capability):
                        logger.debug(
                            f"generate_query attempt {attempt + 1}/3 mutation "
                            f"target coverage mismatch for {self.domain}: expected="
                            f"{sorted(expected_mutations)}, got="
                            f"{sorted(evidence_by_capability)}"
                        )
                        continue
                    normalized_query = " ".join(str(query).lower().split())
                    if any(
                        " ".join(span.lower().split()) not in normalized_query
                        for span in evidence_by_capability.values()
                    ):
                        logger.debug(
                            f"generate_query attempt {attempt + 1}/3 used mutation "
                            f"evidence absent from query for {self.domain}"
                        )
                        continue
                if query:
                    return GeneratedQuery(
                        user_query=str(query),
                        target_capability=target_capability,
                        chain_supported=chain_supported,
                        attempts=attempt + 1,
                    )
            except Exception as e:
                logger.debug(
                    f"generate_query attempt {attempt + 1}/3 failed for "
                    f"{self.domain}: {type(e).__name__}: {e}"
                )
        raise RuntimeError(f"Failed to generate query for {self.domain}")

    # ── Generate a follow-up user message ──

    def generate_followup(
        self,
        tool_schemas: list[dict[str, Any]],
        grounded_state: dict[str, Any],
        previous_query: str,
        difficulty: str,
        rng: random.Random,
        persona: str = "",
        reference_date: str = "",
        chain_seed: list[str] | None = None,
        chain_progress: int = 0,
        previous_response: str = "",
        conversation_context: list[dict[str, Any]] | None = None,
    ) -> str:
        """Generate a follow-up user message from the user's perspective.

        The user continues without knowing which tools the assistant called.
        The follow-up is grounded in refreshed live state rather than hidden
        oracle details. The initial request owns the complete dependency-chain
        goal; a follow-up continues that intent instead of exposing an internal
        chain node as a new request.

        Passing execution_history to the follow-up generator was wrong: it
        caused the LLM to adopt the assistant's confirmation tone ("Got it,
        the transfer is scheduled...") instead of a genuine user perspective.

        chain_seed + chain_progress is retained only when the initial chain is
        still incomplete. Once it is complete, the refreshed live state and
        previous query ground a related next request without synthesizing a new
        dependency graph or exposing internal chain nodes.
        """
        state_text = _format_state_compact(grounded_state, max_entities=20)
        tools_text = _format_tools(tool_schemas)
        conversation_text = _format_conversation_context(conversation_context)

        date_block = ""
        if reference_date:
            date_block = f"\n## Reference Date\nToday is {reference_date}. Use relative dates when appropriate.\n"

        system = (
            "You are role-playing as a real user who sent a request to an AI assistant "
            "and is now sending a follow-up message.\n\n"
            "IMPORTANT: You do NOT know what tools the assistant used internally. "
            "You only know what you originally asked for. "
            "You also know the assistant's visible reply from the preceding round. "
            "Write the follow-up purely from your own perspective — what you want next.\n\n"
            "DO NOT write confirmation phrases like 'Got it', 'Great', 'Thanks', "
            "'That looks right', or any acknowledgment of the assistant's actions. "
            "Just state your next request directly.\n"
            "DO NOT mention tool names or internal steps.\n"
            "Keep it to 1-2 sentences. Be natural and direct.\n"
            "Do not introduce an unrelated new goal; continue the same task. "
            "Staying in the same domain is NOT enough: the next request must reuse "
            "the same transaction, an exact entity, or a concrete result from the "
            "immediately preceding round. For example, after clearing log files, "
            "switching to finding shell scripts is unrelated and forbidden. If the "
            "previous goal is complete, ask for a detail, adjustment, reversal, or "
            "next action on its exact entity/result instead of starting a new task.\n"
            "Do not ask for an outcome that Current State already shows as satisfied. "
            "For example, do not request the same reminder, attendee, label, or field "
            "value twice.\n"
            "Choose a request that is feasible for the entity's current status. "
            "For example, do not ask to cancel an already delivered order.\n"
            "Use the Available Tools descriptions as the authority for allowed "
            "statuses, resource types, exact amounts, and other preconditions. "
            "Never use one resource type's ID where another is required.\n"
            "Across every difficulty level, if you refer to an existing resource "
            "by ID, name, email, username, path, or another selector, copy an exact "
            "value shown in Current State. Never invent a person or resource. "
            "Missing or minimal difficulty may omit a required task detail, but "
            "that omission never permits a fabricated entity.\n"
            "If you request a state-changing action, include user-decided required "
            "values when the difficulty is complete. For missing or minimal difficulty, "
            "it is acceptable to omit such a value so the assistant can clarify; never "
            "copy a business value from an unrelated entity."
        )

        response_block = (
            f'\n## Assistant reply you just received\n"{previous_response}"\n'
            if previous_response.strip() else ""
        )

        next_goal_block = ""
        if chain_seed and chain_progress < len(chain_seed):
            next_goal = _chain_goal_phrase(chain_seed[chain_progress])
            if next_goal:
                next_goal_block = (
                    "\n## Next Conversation Goal\n"
                    f"{_target_tool_requirement(tool_schemas, chain_seed[chain_progress])}\n"
                    f"The next user message should naturally ask to {next_goal}. "
                    "It MUST directly request this outcome. For a write/update, state "
                    "the concrete field or value to change. Do not ask whether the "
                    "assistant needs more information, do not merely request status, "
                    "and do not mention the internal tool name or workflow steps.\n"
                )

        user = f"""## Persona
{persona if persona else 'A normal user messaging their AI assistant.'}
{date_block}
## What this assistant can help with
{self.domain_desc}

## Available Tools and Preconditions
{tools_text}

## Immediately preceding user request
"{previous_query}"
{response_block}

## All Completed Conversation Rounds
{conversation_text}

## Current State (real IDs and values you can reference)
{state_text}
{next_goal_block}

## Your task
Write ONE short follow-up message as the user. Difficulty: {difficulty}.
Ask for the next thing you need. Do NOT acknowledge or confirm what the assistant did.
The follow-up MUST stay within the assistant capabilities listed above and
continue this same domain conversation. Do not request an unavailable
cross-domain capability.
For every difficulty, any existing entity selector you mention (including an
ID, name, email, username, or path) MUST be copied exactly from Current State.
For missing or minimal difficulty, omit a task detail instead of inventing an
entity or selector that is absent from Current State.
For complete difficulty, if the target capability requires an existing entity,
copy its exact ID from Current State into the message and include every concrete
detail needed for the requested change. Select only an entity whose shown type,
status, amount, and linked IDs satisfy the tool description. If no such entity
exists, ask for a read-only status/detail check rather than requesting an
impossible mutation.

Return only:
{{"user_query": "<the follow-up message>"}}
"""
        for attempt in range(3):
            try:
                raw = self._generate_chat(
                    "continuation_generation",
                    [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
                    temperature=0.4 + 0.1 * attempt,
                )
                data = _extract_json(raw)
                if not isinstance(data, dict):
                    continue
                query = data.get("user_query", "")
                if query:
                    return query
            except Exception as e:
                logger.debug(
                    f"generate_followup attempt {attempt + 1}/3 failed for "
                    f"{self.domain}: {type(e).__name__}: {e}"
                )
        raise RuntimeError(f"Failed to generate followup for {self.domain}")

    def generate_clarification(
        self,
        tool_schemas: list[dict[str, Any]],
        grounded_state: dict[str, Any],
        previous_query: str,
        difficulty: str,
        rng: random.Random,
        persona: str = "",
        reference_date: str = "",
        previous_response: str = "",
        conversation_context: list[dict[str, Any]] | None = None,
    ) -> str:
        """Generate a user clarification question.

        Unlike generate_followup (which drives the task forward), this
        generates a natural user question seeking clarification or more detail
        — simulating a user who needs additional information before proceeding.
        The clarification enters conversation_queries and round_contracts,
        producing a genuine multi-round training sample.
        """
        state_text = _format_state_compact(grounded_state, max_entities=20)
        tools_text = _format_tools(tool_schemas)
        conversation_text = _format_conversation_context(conversation_context)

        date_block = ""
        if reference_date:
            date_block = f"\n## Reference Date\nToday is {reference_date}.\n"

        system = (
            "You are role-playing as a real user who asked an AI assistant for help "
            "and the assistant needs more information to proceed.\n\n"
            "Write a brief, natural clarification question the user might ask. "
            "This should sound like a real person asking for more details, NOT "
            "like a system prompt or a tool description.\n\n"
            "Do not introduce a new goal. Stay in the same domain and clarify only "
            "the immediately preceding request or response. Do not request any "
            "capability that is absent from the Available Tools.\n\n"
            "DO NOT mention tool names, API calls, or technical implementation.\n"
            "Keep it to 1-2 sentences. Be natural.\n"
        )

        user = f"""## Persona
{persona if persona else 'A normal user messaging their AI assistant.'}
{date_block}
## Original request
"{previous_query}"

## Assistant reply you just received
"{previous_response}"

## All Completed Conversation Rounds
{conversation_text}

## What this assistant can help with
{self.domain_desc}

## Available Tools and Preconditions
{tools_text}

## Current State
{state_text}

## Task
Write ONE short user clarification question. The user realized they need to provide
more information or ask a follow-up detail question. The question must remain
within the current domain and visible tools. Do not request an unavailable
cross-domain capability or start a separate task.

Return only:
{{"user_query": "<the clarification question>"}}
"""
        for attempt in range(3):
            try:
                raw = self._generate_chat(
                    "clarification_generation",
                    [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
                    temperature=0.5 + 0.1 * attempt,
                )
                data = _extract_json(raw)
                if not isinstance(data, dict):
                    continue
                query = data.get("user_query", "")
                if query:
                    return query
            except Exception as e:
                logger.debug(
                    f"generate_clarification attempt {attempt + 1}/3 failed "
                    f"for {self.domain}: {type(e).__name__}: {e}"
                )
        raise RuntimeError(f"Failed to generate clarification for {self.domain}")

    # ── Step 2-N: decide next action (LLM-in-the-loop) ──

    def decide_action(
        self,
        tool_schemas: list[dict[str, Any]],
        user_query: str,
        execution_history: list[dict[str, Any]],
        attempt: int = 0,
        difficulty: str = "complete",
        reference_date: str = "",
        chain_context: dict[str, Any] | None = None,
        conversation_context: list[dict[str, Any]] | None = None,
        blocked_tools: set[str] | None = None,
        missing_function: bool = False,
        allow_direct_answer: bool = False,
    ) -> ActionPlan:
        """LLM decides the next action given full context.

        Called at every turn.  The LLM sees the complete execution history
        and current state, so its decisions are grounded in real values.

        For 'missing' difficulty tasks, ask_clarification is expected on the
        first turn (the query deliberately omits a parameter), so the
        first-turn enforcement is relaxed.

        chain_context: live-probed entities used to ground ID arguments.
        """
        tools_text = _format_tools(tool_schemas)
        history_text = _format_history(
            execution_history,
            max_chars=self.max_observation_chars,
        )
        conversation_text = _format_conversation_context(conversation_context)
        tool_names_set = {t["name"] for t in tool_schemas}  # for action auto-correction

        # Guard: tool names must not collide with reserved terminal actions.
        # If a domain tool were named "final_answer", "ask_clarification", or
        # "report_error", the auto-correction below would misclassify it.
        # This is checked once per decide_action call so it fires early.
        _terminal_collision = tool_names_set & set(_VALID_TERMINALS)
        if _terminal_collision:
            logger.warning(
                f"decide_action: tool name(s) {_terminal_collision} in domain "
                f"'{self.domain}' collide with reserved terminal action names. "
                f"Terminal interpretation takes precedence — these tools cannot "
                f"be reached via auto-correction."
            )

        # First-turn guidance: prevent LLM from answering without tools.
        # Exception: 'missing' difficulty tasks omit a parameter on purpose,
        # so ask_clarification is the correct first action.
        # 'minimal' tasks are deliberately terse — allow ask_clarification as a
        # valid first action; forcing a tool call on vague queries causes spurious
        # execution errors. 'complete' tasks have specific entity IDs and should
        # start with a real tool call.
        if attempt == 0:
            if missing_function:
                first_turn_hint = (
                    "\nThe user's complete request requires a capability that is not "
                    "present in Available Tools. You may use visible tools when they make "
                    "useful progress or establish that the capability is missing. Never "
                    "invent or call an unavailable tool. End with a concise clarification "
                    "or report that the request cannot be completed.\n"
                )
                default_action = "ask_clarification"
            elif allow_direct_answer:
                first_turn_hint = (
                    "\nThis is the first assistant action for a new user clarification "
                    "round. You may answer directly when no live lookup or state change "
                    "is needed. Otherwise, call the appropriate tool. A terminal answer "
                    "must contain a concrete, non-empty response.\n"
                )
                default_action = "final_answer"
            elif difficulty == "missing":
                first_turn_hint = (
                    "\nNote: This task has a MISSING parameter. "
                    "ask_clarification may be needed before calling a tool.\n"
                )
                default_action = "ask_clarification"
            elif difficulty == "minimal":
                first_turn_hint = (
                    "\nThis is your FIRST turn. The user's request is terse. "
                    "If you can identify a concrete action, call a tool. "
                    "If the request is genuinely ambiguous, ask_clarification "
                    "is acceptable. Do NOT produce final_answer.\n"
                )
                default_action = "tool_call"
            else:
                first_turn_hint = (
                    "\n⚠️  This is your FIRST turn. You MUST call a tool to "
                    "make progress — do NOT produce final_answer. "
                    "Use a read/list/search tool if you need information "
                    "before acting. Do NOT ask the user questions — the "
                    "request has enough detail to start.\n"
                )
                default_action = "tool_call"
        else:
            # After the first turn, the LLM decides the next action from the
            # query, schemas and execution history.
            first_turn_hint = ""
            default_action = "final_answer"

        date_guide = (
            f"## Reference Date\nToday is {reference_date}. Do not invent dates from an earlier year.\n"
            if reference_date else ""
        )

        # ── Live-state grounding constraint ──
        # The initial round receives a chain-aligned subset. Continuation rounds
        # receive a refreshed live-state snapshot, so the planner must not treat
        # the initial subset as an exhaustive list of server entities.
        anti_halluc_block = ""
        if chain_context and chain_context.get("entity_summaries"):
            summaries_text = "\n".join(chain_context["entity_summaries"][:15])
            anti_halluc_block = (
                f"\n## Current Grounded Entities (Observable Context)\n"
                f"{summaries_text}\n\n"
                f"⚠️ CRITICAL — ID Provenance Rule:\n"
                f"- Entity IDs (event_id, account_id, invoice_id, order_id, etc.) "
                f"MUST come from one of these sources:\n"
                f"  1. The User Task itself\n"
                f"  2. Current State (shown in the Execution History)\n"
                f"  3. Prior tool observations in the Execution History\n"
                f"- If you don't know the correct ID, call a read/search/list tool first "
                f"to discover it — NEVER guess or invent an ID.\n"
                f"- Copy IDs exactly as they appear. Do NOT modify, renumber, or "
                f"create IDs that look similar (e.g., if you see 'evt_aa3_001', "
                f"don't write 'evt_aa3_002' unless you observed it).\n"
            )
        elif difficulty == "complete":
            # Even without chain_context, enforce anti-hallucination for
            # complete-difficulty tasks that use entity IDs.
            anti_halluc_block = (
                "\n## Anti-Hallucination Rule\n"
                "⚠️ Entity IDs (event_id, account_id, invoice_id, etc.) MUST come "
                "from the User Task, Execution History, or Current State. If unsure, call a "
                "read/search/list tool to find the correct ID. NEVER guess.\n"
            )

        system = (
            "You are an agent acting in a live MCP environment. At each step, infer "
            "the unresolved user outcome from the current conversation, inspect only "
            "observable state and real tool feedback, then take exactly one next action. "
            "Never treat a successful call as proof of completion when its observation "
            "or state_changed field shows that the requested outcome is still unresolved.\n"
            "\n"
            "Output ONE JSON object per turn:\n"
            '- {"action": "tool_call", "tool_name": "<tool>", "arguments": {"<param>": <value>}}\n'
            '    → to interact with a tool (read, write, search, execute).\n'
            '- {"action": "final_answer", "text": "<answer>"}\n'
            '    → when the task is done. Give the user their result.\n'
            '- {"action": "report_error", "reason": "<why>"}\n'
            '    → when the task cannot be completed with available tools/state.\n'
            '- {"action": "ask_clarification", "question": "<what you need>"}\n'
            '    → only when genuinely ambiguous and no tool can resolve it.\n'
            "\n"
            "Completion and recovery rules:\n"
            "- Use final_answer only after the current user request is actually complete. "
            "A successful read or partial side effect is not completion when a requested "
            "outcome remains undone.\n"
            "- If the response asks the user to choose, identify, confirm, or provide "
            "anything, use ask_clarification. final_answer must not request new user "
            "input anywhere in its text.\n"
            "- Resolve words such as 'that', 'it', and 'that email' from the complete "
            "Prior Conversation Rounds and execution observations. If the latest "
            "history identifies exactly one referent, use it instead of asking the "
            "user to repeat information.\n"
            "- For a request with multiple independent outcomes, complete every still-"
            "feasible outcome before reporting that another outcome is blocked. One "
            "failed subtask does not justify abandoning an independent available one.\n"
            "- When a required capability is absent from Available Tools, ask for user "
            "input only if that information can actually unblock an available tool. "
            "If no user answer can restore the missing capability, use report_error.\n"
            "- For a mutating call, state_changed=true proves the call changed the "
            "recorded state. Do not later claim that the changed condition was already "
            "true before that call unless an earlier observation explicitly proves it.\n"
            "- On failure, retry with corrected parameters or an alternative tool only "
            "when it preserves the same user-requested outcome. Do not substitute a "
            "different business action (for example, disputing an invoice when the user "
            "asked to cancel it).\n"
            "- If no available tool can complete the requested outcome, stop making "
            "state changes and use report_error with a concise explanation.\n"
            "- Separate the requested outcome from incidental context. A phrase such as "
            "'for tomorrow's meeting' does not require calendar or messaging access when "
            "the requested outcome itself can be completed with the available tools.\n"
            "- Never invent a user-decided required value for a mutating call (for example "
            "an amount, destination, address, message body, or replacement value), and do "
            "not copy it from another entity merely to satisfy the schema. If the user did "
            "not provide it and no prior tool output determines it, ask_clarification.\n"
            "- Treat grounded IDs as typed resources. Match invoice_id only to an invoice, "
            "payment_id only to a payment, and likewise for every other schema field.\n"
            "- Before every mutating call, compare its arguments with the grounded entity "
            "and latest tool observations. Respect shown status, exact amount, remaining "
            "allowance, and linked IDs. If a prerequisite is unknown, discover it with a "
            "read/list/search tool rather than guessing.\n"
            "\n"
            "⚠ FORMAT RULES (follow exactly):\n"
            "- When calling a tool, \"action\" MUST be \"tool_call\". Put the tool name in \"tool_name\".\n"
            "- NEVER put the tool name directly in \"action\" (e.g., WRONG: {\"action\": \"search_events\", ...}).\n"
            "\n"
            "Examples:\n"
            '✓ CORRECT tool call: {"action": "tool_call", "tool_name": "search_events", "arguments": {"keyword": "team meeting"}}\n'
            '✓ CORRECT final answer: {"action": "final_answer", "text": "You have 3 meetings this week."}\n'
            '✓ CORRECT ask user:     {"action": "ask_clarification", "question": "Which account would you like to check?"}\n'
            '✗ WRONG tool call:      {"action": "search_events", "arguments": {"keyword": "team meeting"}}\n'
            '✗ WRONG tool call:      {"action": "search_events", "tool_name": "search_events", "arguments": {...}}'
        )
        user = f"""## Domain
{self.domain_desc}

## Available Tools
{tools_text}

{anti_halluc_block}
{date_guide}
## User Task
{user_query}

## Prior Conversation Rounds
{conversation_text}

## Real MCP Execution Events
{history_text}
{first_turn_hint}
## Your Turn
Output one JSON object:
"""
        for _retry in range(3):
            try:
                raw = self._generate_chat(
                    "action_decision",
                    [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
                    temperature=0.3 + 0.05 * _retry,
                )
                data = _extract_json(raw)
                if not isinstance(data, dict):
                    continue
                action = data.get("action", default_action)

                # Validate: tool_call MUST have a non-empty tool_name
                if action == "tool_call":
                    tool_name = data.get("tool_name", "").strip()
                    if not tool_name:
                        logger.debug(
                            f"decide_action got tool_call with empty tool_name for {self.domain}, "
                            f"retrying (attempt {_retry + 1}/3). LLM raw: {raw[:120]}..."
                        )
                        continue
                    # P0: reject calls to blocked (hidden) tools so Teacher
                    # cannot produce an oracle that references a missing function.
                    if blocked_tools and tool_name in blocked_tools:
                        logger.debug(
                            f"decide_action rejected blocked tool '{tool_name}' for "
                            f"{self.domain}, retrying (attempt {_retry + 1}/3)."
                        )
                        continue
                    if tool_name not in tool_names_set:
                        logger.debug(
                            f"decide_action rejected unknown candidate tool '{tool_name}' "
                            f"for {self.domain}."
                        )
                        continue
                elif action in _VALID_TERMINALS:
                    tool_name = ""
                    terminal_text = data.get(
                        "text", data.get("reason", data.get("question", ""))
                    )
                    if not isinstance(terminal_text, str) or not terminal_text.strip():
                        logger.debug(
                            f"decide_action rejected empty terminal '{action}' for "
                            f"{self.domain}, retrying (attempt {_retry + 1}/3)."
                        )
                        continue
                    if (
                        action == "final_answer"
                        and _final_answer_requests_user_input(terminal_text)
                    ):
                        logger.debug(
                            f"decide_action rejected question-shaped final_answer "
                            f"for {self.domain}; retrying as a terminal format error."
                        )
                        continue
                elif action in tool_names_set:
                    # Model used a tool name as the action type (e.g.,
                    # {"action": "cd", "arguments": {...}} instead of
                    # {"action": "tool_call", "tool_name": "cd", ...}).
                    # Auto-correct: treat as tool_call with this tool_name.
                    logger.debug(
                        f"decide_action auto-corrected action '{action}' → tool_call "
                        f"for {self.domain} (tool name used as action type). "
                        f"LLM raw: {raw[:120]}..."
                    )
                    tool_name = action
                    action = "tool_call"
                else:
                    # Unknown action type — retry
                    logger.debug(
                        f"decide_action unknown action '{action}' for {self.domain}, "
                        f"retrying (attempt {_retry + 1}/3). LLM raw: {raw[:120]}..."
                    )
                    continue

                # P2-1: tool_call arguments MUST be a JSON object (dict).
                # Teacher may return list, string, or other types — reject and retry.
                if action == "tool_call":
                    raw_args = data.get("arguments")
                    if not isinstance(raw_args, dict):
                        logger.debug(
                            f"decide_action rejected non-dict arguments "
                            f"({type(raw_args).__name__}) for {self.domain}, "
                            f"retrying (attempt {_retry + 1}/3). "
                            f"LLM raw: {raw[:120]}..."
                        )
                        continue

                plan = ActionPlan(
                    action=action,
                    tool_name=tool_name,
                    arguments=data.get("arguments", {}),
                    text=data.get("text", data.get("reason", data.get("question", ""))),
                )
                self.record_environment_event(
                    "parsed_action",
                    action=plan.action,
                    tool_name=plan.tool_name,
                    arguments=plan.arguments,
                    text=plan.text,
                )
                return plan
            except Exception as e:
                logger.debug(
                    f"decide_action attempt {_retry + 1}/3 failed for "
                    f"{self.domain}: {type(e).__name__}: {e}"
                )
        raise RuntimeError(
            f"decide_action failed after 3 attempts for {self.domain} — "
            f"LLM could not produce a valid decision"
        )


    # ── Recovery module ──

    def decide_recovery(
        self,
        last_tool_name: str,
        last_arguments: dict[str, Any],
        error_observation: dict[str, Any],
        tool_schemas: list[dict[str, Any]],
        execution_history: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Choose a recovery action after a failed tool call.

        Returns one of:
          - {"action": "retry_same", "corrected_args": {...}}  — retry with tweaked params
          - {"action": "retry_alt", "tool_name": "..."}        — use alternative tool
          - {"action": "retry", "arguments": {...}}            — plain retry (intermittent)
          - {"action": "give_up", "reason": "..."}             — unrecoverable
        """
        tools_text = _format_tools(tool_schemas)
        history_text = _format_history(
            execution_history[-4:],
            max_chars=self.max_observation_chars,
        )  # last 4 steps

        # Intermittent errors → plain retry, don't ask LLM
        if isinstance(error_observation, dict) and error_observation.get("retry"):
            return {"action": "retry", "arguments": last_arguments}

        system = (
            "You are recovering from a failed tool call. "
            "Decide the best recovery strategy. Output EXACTLY one JSON object."
        )
        user = f"""## Failed Call
Tool: {last_tool_name}
Arguments: {_json.dumps(last_arguments, ensure_ascii=False)}
Error: {_json.dumps(error_observation, ensure_ascii=False, default=str)}

## Available Tools
{tools_text}

## Recent History
{history_text}

## Recovery Options
Choose ONE:

- Retry with corrected parameters:
  {{"action": "retry_same", "corrected_args": {{"<param>": <new_value>}}}}

- Try an alternative tool:
  {{"action": "retry_alt", "tool_name": "<alternative_tool>", "arguments": {{"<param>": <value>}}}}

- Give up (task impossible with current tools/state):
  {{"action": "give_up", "reason": "<why>"}}

Output ONLY the JSON, nothing else:
"""
        try:
            raw = self._generate_chat(
                "recovery_decision",
                [{"role": "system", "content": system},
                 {"role": "user", "content": user}],
                temperature=0.3,
            )
            data = _extract_json(raw)
            action = data.get("action", "give_up")
            if action == "retry_same":
                return {"action": "retry_same", "corrected_args": data.get("corrected_args", last_arguments)}
            elif action == "retry_alt":
                return {"action": "retry_alt", "tool_name": data.get("tool_name", ""), "arguments": data.get("arguments", {})}
            else:
                return {"action": "give_up", "reason": data.get("reason", "recovery failed")}
        except Exception as e:
            logger.debug(f"decide_recovery failed for {self.domain}: {e}")
            return {"action": "give_up", "reason": str(e)}




# ═══════════════════════════════════════════════════════════════════════
# Success criteria derivation (from state delta)
# ═══════════════════════════════════════════════════════════════════════

# Tools that perform writes without observable state changes in the
# tracked state machine (e.g. network side-effects: send email, send DM).
# These legitimately produce empty success_criteria even though they
# are mutating the outside world.
_SELF_CONTAINED_WRITE_TOOLS: frozenset[str] = frozenset({
    "send_email", "send_message", "send_dm", "reply_email",
    "forward_email",
})


def derive_success_criteria(
    initial_state: dict[str, Any],
    final_state: dict[str, Any],
    oracle_calls: list[OracleCall],
    domain: str,
) -> list[dict[str, Any]]:
    """Derive verifiable success criteria from the delta between initial and final state.

    Since the oracle trace was just executed, final_state is the ground truth.
    Criteria verify key state changes that the model must produce.
    """
    criteria: list[dict[str, Any]] = []

    def _append_nested_delta(
        initial_value: Any,
        final_value: Any,
        path_parts: list[str],
    ) -> None:
        """Record verifiable leaf deltas, including newly-added nested maps."""
        path = ".".join(path_parts)
        if isinstance(initial_value, dict) and isinstance(final_value, dict):
            for child_key in final_value.keys() - initial_value.keys():
                child_path = [*path_parts, str(child_key)]
                if isinstance(final_value[child_key], dict):
                    criteria.append({
                        "type": "state_exists", "server": domain,
                        "path": ".".join(child_path),
                        "path_parts": child_path,
                    })
                _append_nested_delta(
                    None, final_value[child_key], child_path,
                )
            for child_key in initial_value.keys() - final_value.keys():
                child_path = [*path_parts, str(child_key)]
                criteria.append({
                    "type": "state_absent", "server": domain,
                    "path": ".".join(child_path),
                    "path_parts": child_path,
                })
            for child_key in initial_value.keys() & final_value.keys():
                if initial_value[child_key] != final_value[child_key]:
                    _append_nested_delta(
                        initial_value[child_key], final_value[child_key],
                        [*path_parts, str(child_key)],
                    )
            return
        if final_value is None:
            criteria.append({
                "type": "state_equals", "server": domain,
                "path": path, "path_parts": path_parts, "value": None,
            })
            return
        if isinstance(final_value, dict):
            for child_key, child_value in final_value.items():
                _append_nested_delta(
                    None, child_value, [*path_parts, str(child_key)],
                )
            return
        if isinstance(final_value, list) or isinstance(
            final_value, (str, int, float, bool)
        ):
            criteria.append({
                "type": "state_equals", "server": domain,
                "path": path,
                "path_parts": path_parts,
                "value": final_value,
            })

    # One recursive tree walk handles top-level scalars, containers, additions,
    # removals and explicit None values with the same semantics.
    _append_nested_delta(initial_state, final_state, [])

    # Domain-specific semantic criteria
    tool_names = [c.tool_name for c in oracle_calls if c.action == "tool_call"]
    criteria.extend(_domain_criteria(tool_names, initial_state, final_state, domain))

    # Generic state deltas and domain-specific helpers can describe the same
    # postcondition (for example lead.status after convert_lead).  Keep the
    # first, richer criterion and avoid rewarding one state change twice.
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for criterion in criteria:
        value_key = _json.dumps(
            criterion.get("value", None), sort_keys=True,
            ensure_ascii=False, default=str,
        )
        key = (
            str(criterion.get("type", "")),
            str(criterion.get("server", "")),
            str(criterion.get("path", "")),
            value_key,
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(criterion)

    # When no criteria can be derived, keep the list empty.
    # r_coverage uses max(outcome_count + criteria_count, 1) as denominator,
    # so an empty criteria list degrades gracefully to outcome-only coverage.

    return deduped


def derive_progress_predicates(
    oracle_calls: list[OracleCall],
    domain: str,
    *,
    entity_resolver,
    requirements_resolver,
) -> list[dict[str, Any]]:
    """Derive step-level progress predicates from the oracle trace.

    OVAL-MCP §5.4 verifier automaton defines five progress predicates:
      - resolved_required_entity:   a required entity/resource is uniquely resolved
      - satisfied_dependency_edge:  a dependency-ordered predecessor is completed
      - completed_required_transition: expected state transition is completed
      - verified_postcondition:    required postcondition is checked or observed
      - produced_required_response: terminal action satisfies task predicate

    Each predicate is tagged with the oracle_call index so the reward function
    can compute F_gamma = gamma * Phi(m_{t+1}) - Phi(m_t) per step.

    Returns a list of dicts, one per progress event:
        {"step": int, "type": str, "tool": str, "entity": str}
    """
    predicates: list[dict[str, Any]] = []
    real_calls = [c for c in oracle_calls if c.action == "tool_call"]

    if not real_calls:
        return predicates

    # Step 1: resolved_required_entity for every read/lookup call
    read_prefixes = ("list_", "search_", "get_", "find_", "check_", "lookup_",
                     "view_", "browse_", "ls", "cat", "stat", "head", "tail")
    resolved_entities: set[str] = set()

    for i, call in enumerate(real_calls):
        is_read = any(call.tool_name.lower().startswith(p) for p in read_prefixes)
        entity = entity_resolver(call.tool_name, domain)
        if is_read and entity not in resolved_entities:
            resolved_entities.add(entity)
            predicates.append({
                "step": i,
                "type": "resolved_required_entity",
                "tool": call.tool_name,
                "entity": entity,
            })

    # Step 2: completed_required_transition for mutation calls
    for i, call in enumerate(real_calls):
        if is_mutating_tool(call.tool_name, domain):
            entity = entity_resolver(call.tool_name, domain)
            # Extract target entity ID from arguments
            target = ""
            for key, val in (call.arguments or {}).items():
                if isinstance(val, str) and ("_id" in key.lower() or key.lower() in ("path", "event_id")):
                    target = val
                    break
            predicates.append({
                "step": i,
                "type": "completed_required_transition",
                "tool": call.tool_name,
                "entity": entity,
                "target_id": target,
            })

    # Step 3: satisfied_dependency_edge for consecutive calls where a read or
    # creator step resolves/produces an entity consumed by a later mutation.
    # The producer's entity can differ from the mutation's primary output
    # entity, e.g. get_product -> add_to_cart or send_message -> create_thread.
    read_prefixes_dep = ("list_", "search_", "get_", "find_", "check_", "lookup_",
                         "view_", "browse_", "ls", "cat", "stat", "head", "tail")
    creator_prefixes_dep = ("create_", "add_", "send_", "schedule_", "mkdir", "touch")
    for i in range(1, len(real_calls)):
        prev = real_calls[i - 1]
        curr = real_calls[i]
        prev_is_read = any(prev.tool_name.lower().startswith(p) for p in read_prefixes_dep)
        prev_is_creator = any(prev.tool_name.lower().startswith(p) for p in creator_prefixes_dep)
        curr_is_mutate = is_mutating_tool(curr.tool_name, domain)
        if not ((prev_is_read or prev_is_creator) and curr_is_mutate):
            continue
        prev_entity = entity_resolver(prev.tool_name, domain)
        curr_entity = entity_resolver(curr.tool_name, domain)
        curr_required_entities = set(
            requirements_resolver(curr.tool_name, domain)
        )
        acceptable_entities = set(curr_required_entities) | {curr_entity}
        # A dependency edge is satisfied when a read resolves an entity that
        # the subsequent mutation consumes.  The mutation's primary output
        # entity can differ from the required input entity, e.g.
        # get_product -> add_to_cart or get_channel -> send_message.
        if prev_entity in acceptable_entities:
            predicates.append({
                "step": i,
                "type": "satisfied_dependency_edge",
                "tool": curr.tool_name,
                "from_step": i - 1,
                "entity": prev_entity,
            })

    # Step 4: verified_postcondition for terminal or verification calls
    terminal_actions = [c for c in oracle_calls if c.action != "tool_call"]
    if terminal_actions:
        last_terminal = terminal_actions[-1]
        last_step = len(real_calls)  # virtual step after all tool calls
        predicates.append({
            "step": last_step,
            "type": "verified_postcondition",
            "action": last_terminal.action,
        })

    # Step 5: produced_required_response (terminal action itself)
    if terminal_actions:
        last_terminal = terminal_actions[-1]
        predicates.append({
            "step": len(real_calls),
            "type": "produced_required_response",
            "action": last_terminal.action,
        })

    return predicates


def _domain_criteria(
    tool_names: list[str],
    initial_state: dict[str, Any],
    final_state: dict[str, Any],
    domain: str,
) -> list[dict[str, Any]]:
    """Domain-specific success criteria from tool semantics.

    Only emits state_equals for entities whose value differs from
    initial_state. Untouched entities are not outcomes of this trajectory and
    therefore do not belong in replay postconditions.
    """
    criteria: list[dict[str, Any]] = []

    def _changed(container_key: str, entity_id: str, field: str) -> bool:
        init_container = initial_state.get(container_key) or {}
        final_container = final_state.get(container_key) or {}
        if not isinstance(init_container, dict) or not isinstance(final_container, dict):
            return True  # be permissive when shape is unexpected
        init_entity = init_container.get(entity_id)
        final_entity = final_container.get(entity_id)
        if init_entity is None and final_entity is not None:
            return True  # newly created
        if not isinstance(init_entity, dict) or not isinstance(final_entity, dict):
            return final_entity != init_entity
        return init_entity.get(field) != final_entity.get(field)

    if "transfer" in tool_names:
        for acc_id, acc in final_state.get("accounts", {}).items():
            if not _changed("accounts", acc_id, "balance"):
                continue
            criteria.append({
                "type": "state_equals", "server": domain,
                "path": f"accounts.{acc_id}.balance",
                "value": acc.get("balance", 0),
            })
    if "create_order" in tool_names:
        for oid, order in final_state.get("orders", {}).items():
            if not _changed("orders", oid, "status"):
                continue
            criteria.append({
                "type": "state_equals", "server": domain,
                "path": f"orders.{oid}.status",
                "value": order.get("status", "confirmed"),
            })
    if any(t in tool_names for t in ("create_invoice", "pay_invoice")):
        for inv_id, inv in final_state.get("invoices", {}).items():
            if "status" not in inv:
                continue
            if not _changed("invoices", inv_id, "status"):
                continue
            criteria.append({
                "type": "state_equals", "server": domain,
                "path": f"invoices.{inv_id}.status", "value": inv["status"],
            })
    if any(t in tool_names for t in ("update_lead", "convert_lead", "create_deal")):
        for lead_id, lead in final_state.get("leads", {}).items():
            if not _changed("leads", lead_id, "status"):
                continue
            criteria.append({
                "type": "state_equals", "server": domain,
                "path": f"leads.{lead_id}.status",
                "value": lead.get("status", "new"),
            })
    if any(t in tool_names for t in ("create_issue", "update_issue", "transition_issue")):
        for iss_id, issue in final_state.get("issues", {}).items():
            if not _changed("issues", iss_id, "state"):
                continue
            criteria.append({
                "type": "state_equals", "server": domain,
                "path": f"issues.{iss_id}.state",
                "value": issue.get("state", "open"),
            })
    # Generic state deltas already emit exact path/value criteria for cart,
    # newly-created email/file entities, and channel message lists.  Do not add
    # pathless/count-only aliases here: they cannot be attributed to a concrete
    # mutation field and would reward the same state transition twice.
    return criteria


# ═══════════════════════════════════════════════════════════════════════
# Replay validation
# ═══════════════════════════════════════════════════════════════════════

def replay_validate(
    oracle_calls: list[OracleCall],
    manager: "LiveMCPManager",
    executor: "LiveMCPExecutor",
    seed: int,
    domain: str,
    success_criteria: list[dict[str, Any]] | None = None,
    max_error_rate: float = 0.30,
    blocked_tools: set[str] | None = None,
    trace_recorder: Any = None,
    trace_include_state: bool = False,
) -> tuple[bool, float, int, int, bool, int]:
    """Replay oracle trace against a fresh session to verify it's reproducible.

    Counts only schema-level and execution errors, not empty-result responses.
    The default threshold is 30%. Training export separately validates terminal
    shape and tool-call budget.

    Returns:
        (passed, error_rate, num_errors, num_calls, criteria_ok, criteria_failed)
        - passed: True if schema/execution error_rate <= max_error_rate
        - error_rate: fraction of tool calls that failed
        - num_errors: count of schema/execution errors only
        - num_calls: total tool calls replayed
        - criteria_ok: True if replay session satisfies all success_criteria
          (or criteria list is empty). NOT merged into error_rate.
        - criteria_failed: actual count of failed success_criteria
    """
    session = manager.create_session(seed=seed)
    num_errors = 0
    num_calls = 0
    criteria_ok = True
    criteria_failed = 0
    replay_consistent = True
    try:
        manager.discover_tools(session.session_id)
        if callable(trace_recorder):
            initial_state = manager.get_state(session.session_id)
            initial_serialized = _json.dumps(
                initial_state, sort_keys=True, ensure_ascii=True, default=str,
            )
            trace_recorder(
                "replay_start",
                session_id=session.session_id,
                replay_seed=seed,
                replay_domain=domain,
                blocked_tools=sorted(blocked_tools or set()),
                initial_state_hash=hashlib.sha256(
                    initial_serialized.encode("utf-8")
                ).hexdigest(),
                initial_state=initial_state if trace_include_state else None,
            )
        for idx, call in enumerate(oracle_calls):
            # Terminal actions are oracle contract metadata, not MCP calls.
            if call.action != "tool_call":
                continue
            from src.live_mcp.types import ToolCall
            result = executor.execute(
                session.session_id,
                ToolCall(call.tool_name, dict(call.arguments), call_id=f"replay_{idx}"),
                blocked_tools=blocked_tools,
                domain=getattr(call, "server_name", "") or domain,
            )
            num_calls += 1
            if callable(trace_recorder):
                trace_recorder(
                    "replay_call",
                    call_index=idx,
                    tool_name=call.tool_name,
                    arguments=dict(call.arguments),
                    expected_success=getattr(call, "expected_success", None),
                    success=bool(getattr(result, "success", False)),
                    schema_valid=bool(getattr(result, "schema_valid", False)),
                    execution_status=str(
                        getattr(result, "execution_status", "FAILURE")
                    ),
                    state_changed=bool(getattr(result, "state_changed", False)),
                    observation=getattr(result, "observation", None),
                    error_type=str(getattr(result, "error_type", "") or ""),
                    error_message=str(
                        getattr(result, "error_message", "") or ""
                    ),
                )
            expected_success = getattr(call, "expected_success", None)
            if expected_success is False and result.success:
                # Replay must preserve the Teacher attempt outcome. A call that
                # failed during synthesis but succeeds after reset can mutate
                # state and no longer represents the completed conversation.
                num_errors += 1
                replay_consistent = False
                if callable(trace_recorder):
                    trace_recorder(
                        "replay_outcome_mismatch",
                        call_index=idx,
                        tool_name=call.tool_name,
                        expected_success=False,
                        actual_success=True,
                        state_changed=bool(getattr(result, "state_changed", False)),
                    )
                # Later calls would observe a state that the Teacher trajectory
                # never produced, so replay stops at this mismatch.
                break
            if not result.success or not result.schema_valid:
                # Count schema/execution errors, not successful empty results.
                # Schema validation failures (schema_valid=False) are ALWAYS
                # counted as errors — the observation dict may lack an "error"
                # key, containing only validation details.
                # A failed execution is an execution error regardless of its
                # message text. Precondition failures such as "entity not found"
                # remain execution errors.
                num_errors += 1

        # ── Criteria check (independent of tool-error-rate) ──
        # The 30% threshold applies only to schema/execution errors.
        # Criteria validation is separate from num_errors.
        if success_criteria:
            from src.live_mcp.oracle import criterion_satisfied

            replay_state = manager.get_state(session.session_id)
            criteria_failed = sum(
                1 for criterion in success_criteria
                if not criterion_satisfied(replay_state, criterion)
            )
            criteria_ok = (criteria_failed == 0)
            if not criteria_ok:
                logger.debug(
                    f"Replay criteria check: {criteria_failed}/{len(success_criteria)} "
                    f"failed in fresh session — criteria_ok=False"
                )
        else:
            # Outcome-criteria diagnostics:
            # Mutating tool calls with empty success_criteria are suspicious
            # (e.g. teacher called update_* but state was already at target).
            # Empty outcome criteria remain valid; coverage falls back to
            # tool-call matching.  We log a diagnostic warning but accept
            # the task so the teacher retry budget isn't wasted.
            tool_call_count = sum(1 for c in oracle_calls if c.action == "tool_call")
            if tool_call_count > 0:
                has_mutating = any(
                    is_mutating_tool(c.tool_name, domain)
                    for c in oracle_calls if c.action == "tool_call"
                )
                if has_mutating:
                    logger.warning(
                        "Empty success_criteria for trace with {} tool call(s) "
                        "including mutating tool(s). Accepting — R_coverage "
                        "will use pure tool-call matching.",
                        tool_call_count,
                    )

        error_rate = num_errors / num_calls if num_calls > 0 else float(num_errors > 0)
        passed = replay_consistent and error_rate <= max_error_rate

        if callable(trace_recorder):
            replay_state = manager.get_state(session.session_id)
            replay_serialized = _json.dumps(
                replay_state, sort_keys=True, ensure_ascii=True, default=str,
            )
            trace_recorder(
                "replay_result",
                passed=passed,
                error_rate=error_rate,
                num_errors=num_errors,
                num_calls=num_calls,
                max_error_rate=max_error_rate,
                criteria_ok=criteria_ok,
                criteria_failed=criteria_failed,
                replay_consistent=replay_consistent,
                final_state_hash=hashlib.sha256(
                    replay_serialized.encode("utf-8")
                ).hexdigest(),
                final_state=replay_state if trace_include_state else None,
            )

        return passed, error_rate, num_errors, num_calls, criteria_ok, criteria_failed
    finally:
        manager.close_session(session.session_id)


# ═══════════════════════════════════════════════════════════════════════
# Sensitive-parameter provenance check.
# ═══════════════════════════════════════════════════════════════════════

# Parameter names indicative of sensitive data.
# Recognised terminal action types (must match the prompt format).
_VALID_TERMINALS: tuple[str, ...] = ("final_answer", "report_error", "ask_clarification")

_SENSITIVE_PARAM_PATTERNS: tuple[str, ...] = (
    "password", "passwd", "token", "api_key", "apikey", "secret",
    "access_key", "private_key", "credential", "auth_token",
    "session_token", "refresh_token", "otp",
)

# Parameter names that carry security-relevant values but are NOT inherently
# suspicious (e.g., account numbers used in transfers). These are checked but
# with lower severity — they should be traceable but don't fail the provenance
# check on their own unless they appear with a sensitive param.
_SECURITY_RELEVANT_PARAMS: tuple[str, ...] = (
    "account_number", "account_id", "routing_number",
)


def provenance_check(
    oracle_calls: list[OracleCall],
    user_query: str,
    aligned_observations: list[dict[str, Any]],
    tool_schemas: list[dict[str, Any]],
    domain: str,
    user_queries: list[str] | None = None,
    call_round_indices: list[int] | None = None,
) -> tuple[bool, list[dict[str, Any]]]:
    """Check that sensitive parameters are traceable.

    Sensitive parameters (passwords, tokens, API keys, etc.) must appear ONLY
    when traceable to prior user turns or tool outputs. Parameters that appear
    "from nowhere" indicate the teacher LLM hallucinated them, which is a
    security risk in training data.

    aligned_observations is 1:1 aligned with oracle_calls by index:
    aligned_observations[i] is the observation produced by oracle_calls[i]
    (empty dict for terminal actions like final_answer/ask_clarification).

    Returns:
        (passed, violations)
        - passed: True if all sensitive parameters are traceable
        - violations: list of dicts describing each violation
          [{"param": str, "value": str, "tool": str, "reason": str}, ...]
    """
    violations: list[dict[str, Any]] = []
    from src.live_mcp.tool_semantics import build_tool_semantics

    contracts = build_tool_semantics(domain, tool_schemas)

    def _traceable(value: Any, sources: list[str]) -> bool:
        import re

        currency_symbols = {
            "USD": "$",
            "EUR": "€",
            "GBP": "£",
            "JPY": "¥",
        }
        values = value if isinstance(value, list) else [value]
        for item in values:
            if item is None or item == "":
                continue
            item_text = str(item).strip()
            if not item_text:
                continue
            if any(item_text.casefold() in source.casefold() for source in sources):
                continue
            if (
                item_text.upper() in currency_symbols
                and any(currency_symbols[item_text.upper()] in source for source in sources)
            ):
                continue
            if isinstance(item, (int, float)) and not isinstance(item, bool):
                numeric_match = False
                for source in sources:
                    # User-facing amounts commonly contain thousands separators
                    # (for example "$1,200").  Compare their numeric value
                    # rather than requiring the serialized tool argument to be
                    # a literal substring of the prompt.
                    for token in re.findall(
                        r"(?<![\w.])-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?",
                        source,
                    ):
                        try:
                            if float(token.replace(",", "")) == float(item):
                                numeric_match = True
                                break
                        except ValueError:
                            continue
                    if numeric_match:
                        break
                if numeric_match:
                    continue
            return False
        return True

    queries = list(user_queries or [user_query])
    if not queries:
        queries = [user_query]
    round_indices = list(call_round_indices or [0] * len(oracle_calls))
    if len(round_indices) != len(oracle_calls):
        raise ValueError(
            "call_round_indices must align 1:1 with oracle_calls"
        )

    # Check each oracle call's arguments for sensitive params
    traceable_values: list[str] = []
    latest_query_round = -1
    for idx, call in enumerate(oracle_calls):
        call_round = round_indices[idx]
        if call_round < 0 or call_round >= len(queries):
            raise ValueError(
                f"call_round_indices[{idx}]={call_round} outside "
                f"user_queries range 0..{len(queries) - 1}"
            )
        while latest_query_round < call_round:
            latest_query_round += 1
            traceable_values.append(queries[latest_query_round])
        if call.action != "tool_call":
            continue
        contract = contracts.get(call.tool_name)
        if contract is None:
            violations.append({
                "param": "", "value": "", "tool": call.tool_name,
                "call_index": idx,
                "reason": "Tool schema unavailable for provenance validation",
            })
            continue
        sensitive_fields = set(contract.sensitive_params)
        for param_name, param_value in call.arguments.items():
            param_lower = param_name.lower()

            # Check if this parameter looks sensitive
            is_sensitive = (
                param_name in sensitive_fields
                or any(p in param_lower for p in _SENSITIVE_PARAM_PATTERNS)
            )
            is_security = any(p in param_lower for p in _SECURITY_RELEVANT_PARAMS)

            if not is_sensitive and not is_security:
                continue

            # Skip empty/None values
            if param_value is None or param_value == "":
                continue

            # For sensitive params: value MUST be traceable
            # For security-relevant params: warn but don't fail on their own
            param_str = str(param_value)
            if len(param_str) < 3:
                continue  # too short to meaningfully check

            # Check if this value appears in any traceable source observed
            # STRICTLY BEFORE this call (no future leak).
            traceable = _traceable(param_value, traceable_values)

            if not traceable:
                if is_sensitive:
                    violations.append({
                        "param": param_name,
                        "value": param_str[:80],
                        "tool": call.tool_name,
                        "call_index": idx,
                        "reason": (
                            f"Sensitive parameter '{param_name}' value not traceable "
                            f"to user query or prior tool outputs"
                        ),
                    })
                else:
                    # Security-relevant but not sensitive: log-only
                    logger.debug(
                        f"provenance_check: security-relevant param '{param_name}' "
                        f"in {call.tool_name} call {idx} not traced — "
                        f"non-blocking (security_relevant category)"
                    )

        # AFTER checking call idx, fold its observation into traceable_values
        # so that subsequent calls (idx+1, idx+2, …) can reference it.
        # aligned_observations is 1:1 with oracle_calls — no index mismatch.
        # aligned_observations[idx] is the raw observation (dict from tool result)
        # or {} for terminal actions (falsy → skip).
        if idx < len(aligned_observations):
            step_obs = aligned_observations[idx] if aligned_observations[idx] else None
            if isinstance(step_obs, dict):
                import json as _json
                traceable_values.append(_json.dumps(step_obs, ensure_ascii=False, default=str))
            elif isinstance(step_obs, str):
                traceable_values.append(step_obs)

    passed = len(violations) == 0
    return passed, violations


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def _format_tools(tool_schemas: list[dict[str, Any]], strip_enums: bool = False) -> str:
    """Format tool schemas as human-readable text, optionally hiding enum values."""
    lines: list[str] = []
    for tool in tool_schemas:
        name = tool["name"]
        desc = tool.get("description", "")
        annotations = tool.get("annotations") or {}
        if annotations.get("readonly") is True and annotations.get("mutating") is False:
            desc = (
                f"{desc.rstrip()} Read-only: this tool does not modify server state. "
                "Any transformed content is display-only output and is never persisted."
            ).strip()
        props = tool.get("input_schema", {}).get("properties", {})
        required = tool.get("input_schema", {}).get("required", [])
        args_parts = []
        for k, info in props.items():
            if strip_enums and "enum" in info:
                info = {kk: vv for kk, vv in info.items() if kk != "enum"}
            req = "*" if k in required else ""
            ptype = _schema_type_hint(info)
            enum_str = f": {', '.join(info['enum'])}" if "enum" in info else ""
            desc_part = f" ({ptype}{enum_str})" if ptype else ""
            param_desc = str(info.get("description") or "").strip()
            desc_suffix = f" — {param_desc}" if param_desc else ""
            args_parts.append(f"{k}{req}{desc_part}{desc_suffix}")
        args_str = ", ".join(args_parts)
        lines.append(f"  - {name}({args_str}): {desc}")
    return "\n".join(lines)


def _schema_type_hint(schema: dict[str, Any]) -> str:
    """Render the argument structure the Teacher must actually produce."""
    ptype = str(schema.get("type") or "")
    if ptype == "array" and isinstance(schema.get("items"), dict):
        return f"array<{_schema_type_hint(schema['items'])}>"
    if ptype == "object":
        properties = schema.get("properties") or {}
        required = set(schema.get("required") or [])
        fields = []
        for name, child in properties.items():
            marker = "*" if name in required else ""
            child_hint = _schema_type_hint(child) if isinstance(child, dict) else ""
            child_desc = str(child.get("description") or "").strip() if isinstance(child, dict) else ""
            fields.append(
                f"{name}{marker}: {child_hint or 'any'}"
                + (f" [{child_desc}]" if child_desc else "")
            )
        return "object{" + ", ".join(fields) + "}"
    constraints = []
    if "enum" in schema:
        constraints.append("enum=" + "|".join(str(item) for item in schema["enum"]))
    if "minimum" in schema:
        constraints.append(f"minimum={schema['minimum']}")
    if "exclusiveMinimum" in schema:
        constraints.append(f"exclusiveMinimum={schema['exclusiveMinimum']}")
    if "maximum" in schema:
        constraints.append(f"maximum={schema['maximum']}")
    if constraints:
        return f"{ptype}({', '.join(constraints)})"
    return ptype


def _format_state_compact(state: dict[str, Any], max_entities: int = 20) -> str:
    """Format grounded state as compact entity summaries.

    Instead of dumping full JSON (which can exceed teacher attention window),
    output one line per entity with key fields only.
    """
    if not isinstance(state, dict) or not state:
        return "(empty state)"

    lines: list[str] = []
    groups: list[tuple[str, list[tuple[str, Any]]]] = []
    for entity_type, entities in sorted(state.items()):
        if isinstance(entities, dict) and entities:
            groups.append((entity_type, sorted(entities.items())))

    # Round-robin over resource types.  A global first-N truncation made later
    # types disappear completely (for example payments after invoices).
    selected: list[tuple[str, str, Any]] = []
    index = 0
    while len(selected) < max_entities:
        added = False
        for entity_type, entities in groups:
            if index < len(entities):
                entity_id, entity_data = entities[index]
                selected.append((entity_type, entity_id, entity_data))
                added = True
                if len(selected) >= max_entities:
                    break
        if not added:
            break
        index += 1

    for entity_type, entity_id, entity_data in selected:
        if isinstance(entity_data, dict):
                # Extract key identity fields (expanded for all domains)
            id_fields: list[str] = []
            for fk in (
                    "name", "title", "subject", "status", "type",
                    "balance", "amount", "price", "quantity",
                    "date", "start_time", "end_time", "due_date",
                    "priority", "stage", "label", "category",
                    "sender", "recipient",
            ):
                if fk in entity_data:
                    val = entity_data[fk]
                    if isinstance(val, str) and len(val) > 60:
                        val = val[:57] + "..."
                    id_fields.append(f"{fk}={val}")
                # Also capture id-like fields
            for fk, fv in entity_data.items():
                if fk.endswith("_id") or fk.endswith("_name"):
                    id_fields.append(f"{fk}={fv}")
            if entity_data.get("summary"):
                id_fields.append(f"facts={entity_data['summary']}")
            summary = ", ".join(id_fields[:5])
            lines.append(f"  {entity_type}/{entity_id}: {summary}" if summary else f"  {entity_type}/{entity_id}")
        else:
            lines.append(f"  {entity_type}/{entity_id}: {entity_data}")
    total_entities = sum(len(entities) for _, entities in groups)
    if total_entities > len(selected):
        shown_by_type: dict[str, int] = {}
        for entity_type, _, _ in selected:
            shown_by_type[entity_type] = shown_by_type.get(entity_type, 0) + 1
        distribution = ", ".join(
            f"{entity_type}={shown_by_type.get(entity_type, 0)}/{len(entities)}"
            for entity_type, entities in groups
        )
        lines.append(
            f"... ({total_entities} total entities; stratified view: {distribution})"
        )
    if not lines:
        return str(state)[:2000]
    return "\n".join(lines)


def _format_conversation_context(context: list[dict[str, Any]] | None) -> str:
    if not context:
        return "(this is the first conversation round)"
    lines: list[str] = []
    for item in context:
        round_idx = item.get("round_idx", "?")
        query = str(item.get("user_query") or "").strip()
        response = str(item.get("assistant_response") or "").strip()
        terminal = str(item.get("terminal_action") or "").strip()
        lines.append(f"Round {round_idx} user: {query or '(missing)'}")
        lines.append(
            f"Round {round_idx} assistant ({terminal or 'response'}): "
            f"{response or '(no visible text)'}"
        )
    return "\n".join(lines)


def _format_history(
    history: list[dict[str, Any]],
    *,
    max_chars: int = DEFAULT_TEACHER_OBSERVATION_CHARS,
) -> str:
    """Format loss-aware MCP execution events for the next Agent decision."""
    if not history:
        return "(no actions yet — this is the first turn)"
    lines = []
    for i, entry in enumerate(history, 1):
        tool = entry.get("tool_name", "?")
        args = _json.dumps(entry.get("arguments", {}), ensure_ascii=False)
        success = entry.get("success", True)
        outcome = str(entry.get("execution_status") or ("SUCCESS" if success else "FAILURE"))
        state_changed = entry.get("state_changed")
        envelope = {
            "success": bool(success),
            "execution_status": outcome,
            "error_type": entry.get("error_type"),
            "error_message": str(entry.get("error_message") or ""),
            "state_changed": bool(state_changed),
            "schema_valid": bool(entry.get("schema_valid", False)),
            "observation": entry.get("observation"),
        }
        lines.append(
            f"Step {i}: {tool}({args}) → "
            f"{outcome}"
            + (f"; state_changed={bool(state_changed)}" if state_changed is not None else "")
        )
        error_type = str(envelope["error_type"] or "").strip()
        error_message = envelope["error_message"].strip()
        if error_type or error_message:
            lines.append(
                f"  Error: {error_type or 'execution_error'}"
                + (f": {error_message}" if error_message else "")
            )
        loop_warning = str(entry.get("no_progress_warning") or "").strip()
        if loop_warning:
            lines.append(f"  No-progress warning: {loop_warning}")
        lines.append(
            "  Result envelope: "
            f"{project_observation(envelope, max_chars=max_chars)}"
        )
    return "\n".join(lines)
