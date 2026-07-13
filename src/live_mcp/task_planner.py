"""PROVE-style state-machine task generation.

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

import copy
import json as _json
import random
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

from loguru import logger

from src.live_mcp.types import OracleCall
from src.utils import extract_json as _extract_json

if TYPE_CHECKING:
    from src.live_mcp.llm_client import LLMClient
    from src.live_mcp.manager import LiveMCPManager
    from src.live_mcp.executor import LiveMCPExecutor


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
        "confirmed→preparing→in_transit→delivered."
    ),
}

DIFFICULTY_DESCRIPTIONS: dict[str, str] = {
    "complete": (
        "The user knows exactly what they want and says it clearly — includes "
        "specific IDs and the desired outcome. No follow-up needed. "
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


def _chain_goal_phrase(final_tool: str) -> str:
    """Natural-language outcome hint for chain-seeded query generation.

    This keeps PROVE's chain-seeded query idea without exposing tool names or
    forcing the user prompt to list internal workflow steps.
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
    required = list(schema.get("input_schema", {}).get("required", []) or [])
    required_text = ", ".join(str(name) for name in required) or "none"
    return (
        f"Internal target capability: {tool_name}\n"
        f"Capability description: {description or _chain_goal_phrase(tool_name)}\n"
        f"Required information fields: {required_text}"
    )


# ═══════════════════════════════════════════════════════════════════════
# Persona templates & reference dates (PROVE §4 diversity injection)
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

# ═══════════════════════════════════════════════════════════════════════
# Turn-decay schedule (PROVE §3.2 Step 3.5:
#   min_turns=2, max_turns=3 for CONVERSATION ROUNDS — i.e. user turns)
#
# NOTE ON TERMINOLOGY (avoids confusion with _run_turn_loop's max_turns=8):
#   * MIN/MAX_CONVERSATION_ROUNDS (this file)  = # of user turns per task
#     PROVE-aligned bounds: 2..3.
#   * max_turns argument in orchestrator._run_turn_loop / generate_one
#     = # of teacher decision STEPS inside a single conversation round
#       (tool_call attempts + retries + terminal), unrelated to PROVE §3.2
#       Step 3.5.  It only bounds the state-machine's inner loop so it
#       terminates on stuck LLMs.  RL rollout uses a verified per-row action
#       budget for generated data; the configured max is only a legacy/default
#       fallback and is unrelated to PROVE §3.2 conversation rounds.
# ═══════════════════════════════════════════════════════════════════════

class ContinuationPolicy:
    """PROVE-style turn-decay schedule for deciding when to end a conversation.

    PROVE §3.2 Step 3.5: min_turns=2, max_turns=3 for conversation rounds
    (user follow-up turns).  A dependency chain seeds one complete user goal;
    follow-up turns continue the live conversation after that goal instead of
    exposing individual internal chain nodes as separate user requests.

    Perturbations (intermittent errors, pagination) may add 1-2 extra turns.
    """

    # PROVE §3.2 Step 3.5 explicitly requires min_turns=2 (i.e., conversations
    # must span at least two user turns with at least one follow-up).
    MIN_CONVERSATION_ROUNDS = 2
    MAX_CONVERSATION_ROUNDS = 3
    # P1-3: Per-turn turn-decay probabilities for middle rounds.
    # These are NOT published PROVE parameters — the paper does not disclose
    # the exact turn-decay schedule.  These are local defaults configurable via
    # experiment metadata.
    CLARIFICATION_PROB: float = 0.10   # probability of clarification at middle rounds
    END_PROB_BASE: float = 0.30        # base end probability, increases with round

    @staticmethod
    def sample_continuation_decision(
        rounds_done: int,
        rng: random.Random,
    ) -> str:
        """PROVE §3.2 Step 3.5: per-turn continuation decision.

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


class TaskPlanner:
    """PROVE-style state-machine teacher.

    The LLM is called at EVERY turn with full context (domain, tools, live state,
    execution history) and decides the next action.  Parameters come from the LLM's
    understanding of real state values, not from heuristic inference.
    """

    def __init__(self, client: "LLMClient", domain: str, seed: int = 0):
        self.client = client
        self.domain = domain
        self.domain_desc = DOMAIN_DESCRIPTIONS.get(domain, "")

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
    ) -> str:
        """LLM generates a natural-language user query grounded in live state.

        PROVE §4: injects persona (character role) and reference_date (temporal anchor)
        to increase query diversity. chain_seed guides the query toward a
        realistic dependency chain without making that chain the only valid
        tool trajectory.

        chain_context (PROVE §3.2 Step 2): a compact, chain-aligned subset of
        live-state entity IDs extracted by _extract_chain_context().  The
        anti-hallucination constraint uses this to prevent ID invention.
        """
        difficulty_desc = DIFFICULTY_DESCRIPTIONS.get(
            difficulty, DIFFICULTY_DESCRIPTIONS["complete"]
        )
        state_text = _format_state_compact(grounded_state, max_entities=20)

        # Date context
        date_block = ""
        if reference_date:
            date_block = f"\n## Reference Date\nToday is {reference_date}. Use relative dates when appropriate.\n"

        system = (
            "You are role-playing as a real person messaging their AI assistant. "
            "Write ONE short message — the way a real human would actually type it. "
            "Real people state what they WANT, not HOW to do it. "
            "They don't list steps, don't mention tool names, don't describe workflows. "
            "They just say their goal in 1-2 sentences max.\n\n"
            "BAD (AI-like): 'I need to search for events, then create a new one, then add attendees.'\n"
            "GOOD (human-like): 'set up a meeting with Sarah next Tuesday at 2pm'\n\n"
            "BAD: 'First verify the account, then check the balance, then transfer funds.'\n"
            "GOOD: 'move $200 from savings to checking'"
        )
        if difficulty == "minimal":
            grounding_line = (
                "Do NOT include entity IDs — just express your intent naturally."
            )
        elif difficulty == "complete":
            grounding_line = (
                "Reference the exact entity IDs from Current State — weave them in naturally."
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

        # ── PROVE §3.2: the complete dependency chain seeds one task. ──
        chain_goal_block = ""
        if chain_seed:
            final_tool = chain_seed[-1]
            goal_phrase = _chain_goal_phrase(final_tool)
            if goal_phrase:
                chain_requirements = "\n".join(
                    f"- {_target_tool_requirement(tool_schemas, tool_name)}"
                    for tool_name in chain_seed
                )
                chain_goal_block = (
                    "\n## Complete Task Goal (internal synthesis guide)\n"
                    f"The grounded dependency chain is: {chain_seed}.\n"
                    f"Capabilities in execution order:\n{chain_requirements}\n"
                    f"The message MUST clearly request the final outcome: {goal_phrase}. "
                    "Read-only earlier items may be internal prerequisites. Every "
                    "write/update/delete/send/pay or other state-changing item MUST "
                    "be explicitly requested in this same message; a dependency edge "
                    "does not grant permission for its side effect. Do not mention tool "
                    "names, a workflow, or split chain nodes into future requests. For "
                    "each state change, state the concrete change that authorizes it. "
                    "Every required field that controls a state change MUST have a "
                    "concrete user-authorized value in the message or an unambiguous "
                    "value in Current State; never leave such a field for the assistant "
                    "to invent (for example an amount, recipient, status, path, date, "
                    "quantity, or message body).\n"
                )

        # ── Anti-hallucination constraint (PROVE §3.2 Step 2) ──
        # Chain-aligned entity context: only these IDs are known to exist
        # in the live state.  The teacher MUST reference only these, never
        # invent IDs.  Without this constraint, the LLM hallucinates entity
        # IDs that don't exist in any seed, causing 100% replay failure.
        anti_halluc_block = ""
        if chain_context and chain_context.get("entity_summaries"):
            summaries_text = "\n".join(chain_context["entity_summaries"][:15])
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
{{"user_query": "<the message>"}}
"""
        for attempt in range(3):
            try:
                raw = self.client.generate_chat(
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
                    f"generate_query attempt {attempt + 1}/3 failed for "
                    f"{self.domain}: {type(e).__name__}: {e}"
                )
        raise RuntimeError(f"Failed to generate query for {self.domain}")

    # ── Step 1b: generate follow-up user message (PROVE CONTINUATION) ──

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
    ) -> str:
        """Generate a follow-up user message from the user's perspective.

        PROVE §3.2 Step 3.5: the user continues the conversation without
        knowing what tools the assistant called internally. The follow-up
        is grounded in the refreshed live server state (real entity IDs), not
        in hidden oracle execution details.  The initial request already owns
        the complete dependency-chain goal; a follow-up continues that user
        intent instead of exposing an internal chain node as a new request.

        Passing execution_history to the follow-up generator was wrong: it
        caused the LLM to adopt the assistant's confirmation tone ("Got it,
        the transfer is scheduled...") instead of a genuine user perspective.

        chain_seed + chain_progress is retained only when the initial chain is
        still incomplete. Once it is complete, the refreshed live state and
        previous query ground a related next request without synthesizing a new
        dependency graph or exposing internal chain nodes.
        """
        state_text = _format_state_compact(grounded_state, max_entities=20)

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
            "Do not introduce an unrelated new goal; continue the same task.\n"
            "Choose a request that is feasible for the entity's current status. "
            "For example, do not ask to cancel an already delivered order.\n"
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

## Your original request
"{previous_query}"
{response_block}

## Current State (real IDs and values you can reference)
{state_text}
{next_goal_block}

## Your task
Write ONE short follow-up message as the user. Difficulty: {difficulty}.
Ask for the next thing you need. Do NOT acknowledge or confirm what the assistant did.
The follow-up MUST stay within the assistant capabilities listed above and
continue this same domain conversation. Do not request an unavailable
cross-domain capability.
For complete difficulty, if the target capability requires an existing entity,
copy its exact ID from Current State into the message and include every concrete
detail needed for the requested change.

Return only:
{{"user_query": "<the follow-up message>"}}
"""
        for attempt in range(3):
            try:
                raw = self.client.generate_chat(
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
    ) -> str:
        """Generate a user clarification question (PROVE §3.2 Step 3.5).

        Unlike generate_followup (which drives the task forward), this
        generates a natural user question seeking clarification or more detail
        — simulating a user who needs additional information before proceeding.
        The clarification enters conversation_queries and round_contracts,
        producing a genuine multi-round training sample.
        """
        state_text = _format_state_compact(grounded_state, max_entities=20)

        date_block = ""
        if reference_date:
            date_block = f"\n## Reference Date\nToday is {reference_date}.\n"

        system = (
            "You are role-playing as a real user who asked an AI assistant for help "
            "and the assistant needs more information to proceed.\n\n"
            "Write a brief, natural clarification question the user might ask. "
            "This should sound like a real person asking for more details, NOT "
            "like a system prompt or a tool description.\n\n"
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

## Current State
{state_text}

## Task
Write ONE short user clarification question. The user realized they need to provide
more information or ask a follow-up detail question.

Return only:
{{"user_query": "<the clarification question>"}}
"""
        for attempt in range(3):
            try:
                raw = self.client.generate_chat(
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
        history_text = _format_history(execution_history)
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

        # ── Live-state grounding constraint (PROVE §3.2 Step 2) ──
        # The initial round receives a chain-aligned subset. Continuation rounds
        # receive a refreshed live-state snapshot, so the planner must not treat
        # the initial subset as an exhaustive list of server entities.
        anti_halluc_block = ""
        if chain_context and chain_context.get("entity_summaries"):
            summaries_text = "\n".join(chain_context["entity_summaries"][:15])
            anti_halluc_block = (
                f"\n## Current Grounded Entities (use these exact IDs)\n"
                f"{summaries_text}\n\n"
                f"⚠️ CRITICAL — ID Provenance Rule:\n"
                f"- Entity IDs (event_id, account_id, invoice_id, order_id, etc.) "
                f"MUST come from one of these sources:\n"
                f"  1. The Chain-Aligned Entities list above\n"
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
                "from the Execution History or Current State. If unsure, call a "
                "read/search/list tool to find the correct ID. NEVER guess.\n"
            )

        system = (
            "You are an AI assistant helping a user complete a task via tool calls. "
            "Think about what the user needs, then take the best next step.\n"
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

## Execution History
{history_text}
{first_turn_hint}
## Your Turn
Output one JSON object:
"""
        for _retry in range(3):
            try:
                raw = self.client.generate_chat(
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

                return ActionPlan(
                    action=action,
                    tool_name=tool_name,
                    arguments=data.get("arguments", {}),
                    text=data.get("text", data.get("reason", data.get("question", ""))),
                )
            except Exception as e:
                logger.debug(
                    f"decide_action attempt {_retry + 1}/3 failed for "
                    f"{self.domain}: {type(e).__name__}: {e}"
                )
        raise RuntimeError(
            f"decide_action failed after 3 attempts for {self.domain} — "
            f"LLM could not produce a valid decision"
        )


    # ── Recovery module (PROVE §6 step 5a: explicit retry states) ──

    def decide_recovery(
        self,
        last_tool_name: str,
        last_arguments: dict[str, Any],
        error_observation: dict[str, Any],
        tool_schemas: list[dict[str, Any]],
        execution_history: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """PROVE-style recovery decision after a failed tool call.

        Returns one of:
          - {"action": "retry_same", "corrected_args": {...}}  — retry with tweaked params
          - {"action": "retry_alt", "tool_name": "..."}        — use alternative tool
          - {"action": "retry", "arguments": {...}}            — plain retry (intermittent)
          - {"action": "give_up", "reason": "..."}             — unrecoverable
        """
        tools_text = _format_tools(tool_schemas)
        history_text = _format_history(execution_history[-4:])  # last 4 steps

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
            raw = self.client.generate_chat(
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

# ── Centralized mutating-tool definitions (PROVE §3.2) ──
# Single source of truth for "does this tool change live server state?"
# Replaces three independent hand-maintained prefix lists that had drifted
# apart across derive_progress_predicates, replay_validate, and
# _validate_task_training_contract.

_MUTATING_PREFIXES: tuple[str, ...] = (
    "create_", "update_", "delete_", "remove_", "add_",
    "send_", "transfer", "pay_", "checkout", "transition_",
    "convert_", "archive_", "mkdir", "touch", "mv", "cp",
    "chmod", "set_", "apply_",
    "cancel_", "refund_", "return_", "deposit", "withdraw",
    "bill_pay", "rm", "sed", "unzip", "zip", "tar_",
    "freeze_", "unfreeze_", "dispute_", "verify_",
    "rate_order", "clear_cart", "reorder",
    "complete_task", "schedule_",
    "mark_read", "mark_unread", "change_",
    "forward_", "chown", "wire_", "assign_",
    "comment_", "time_track", "respond_", "react_",
    "contact_",
    # Domain-specific compound names that don't start with a prefix:
    "move_to_thread", "cd", "umask", "symlink", "split", "truncate",
)

# Tools that perform writes without observable state changes in the
# tracked state machine (e.g. network side-effects: send email, send DM).
# These legitimately produce empty success_criteria even though they
# are mutating the outside world.
_SELF_CONTAINED_WRITE_TOOLS: frozenset[str] = frozenset({
    "send_email", "send_message", "send_dm", "reply_email",
    "forward_email",
})


def _is_mutating_tool(tool_name: str) -> bool:
    """Return True if *tool_name* mutates live server state.

    Uses a unified prefix set + domain-specific exceptions, keeping
    derive_progress_predicates, replay_validate and the training
    contract in sync.
    """
    name_lower = tool_name.lower()
    if name_lower in _SELF_CONTAINED_WRITE_TOOLS:
        return True
    for prefix in _MUTATING_PREFIXES:
        if name_lower.startswith(prefix):
            return True
    return False


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

    # Entity count changes — verify new/removed entities
    for key in final_state:
        init_val = initial_state.get(key)
        final_val = final_state.get(key)
        if isinstance(init_val, dict) and isinstance(final_val, dict):
            init_keys = set(init_val.keys())
            final_keys = set(final_val.keys())
            for nk in (final_keys - init_keys):
                criteria.append({
                    "type": "state_exists", "server": domain,
                    "path": f"{key}.{nk}",
                    "path_parts": [key, nk],
                })
                entity = final_val[nk]
                if isinstance(entity, dict):
                    # PROVE: verify all scalar fields of newly created entities,
                    # not just a hardcoded 4-field whitelist that misses
                    # domain-specific fields like "phase", "priority", "amount".
                    for ek, ev in entity.items():
                        if ev is None:
                            continue
                        if isinstance(ev, (dict, list)):
                            continue
                        if not isinstance(ev, (str, int, float, bool)):
                            continue
                        criteria.append({
                            "type": "state_equals", "server": domain,
                            "path": f"{key}.{nk}.{ek}",
                            "path_parts": [key, nk, ek],
                            "value": ev,
                        })
            for rk in (init_keys - final_keys):
                criteria.append({
                    "type": "state_absent", "server": domain,
                    "path": f"{key}.{rk}",
                    "path_parts": [key, rk],
                })

    # Value changes on existing entities
    for key in final_state:
        init_val = initial_state.get(key)
        final_val = final_state.get(key)
        if isinstance(init_val, dict) and isinstance(final_val, dict):
            common = set(init_val.keys()) & set(final_val.keys())
            for ck in common:
                ie = init_val[ck]
                fe = final_val[ck]
                if isinstance(ie, dict) and isinstance(fe, dict):
                    for fk in fe:
                        if fk in ie and ie[fk] != fe[fk] and fe[fk] is not None:
                            if isinstance(fe[fk], (dict, list)):
                                continue
                            criteria.append({
                                "type": "state_equals", "server": domain,
                                "path": f"{key}.{ck}.{fk}",
                                "path_parts": [key, ck, fk],
                                "value": fe[fk],
                            })
                    # ── P3c: list-field changes (e.g. wishlist.append, watchers, labels) ──
                    # Entity dicts may have list-type fields tracking ordered
                    # collections.  Changes here are real state mutations that
                    # must produce success_criteria to avoid reward-coverage gaps.
                    for fk in fe:
                        if fk not in ie:
                            continue
                        iv = ie[fk]
                        fv = fe[fk]
                        if not isinstance(iv, list) or not isinstance(fv, list):
                            continue
                        if iv == fv:
                            continue
                        # List content changed: verify final list equals expected
                        criteria.append({
                            "type": "state_equals", "server": domain,
                            "path": f"{key}.{ck}.{fk}",
                            "path_parts": [key, ck, fk],
                            "value": fv,
                        })

        # ── Top-level list changes (e.g. cart, wishlist in shopping) ──
        # derive_success_criteria previously only handled dict-type state fields,
        # missing list-type fields like shopping.cart and shopping.wishlist.
        # For lists, a simple equality check covers the full diff.
        elif isinstance(final_val, list):
            init_is_list = isinstance(init_val, list)
            if not init_is_list or init_val != final_val:
                criteria.append({
                    "type": "state_equals", "server": domain,
                    "path": key,
                    "path_parts": [key],
                    "value": final_val,
                })

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


def _tool_entity(name: str, domain: str = "") -> str:
    """Extract entity type from tool name.

    Mirrors orchestrator._tool_entity.  Defined locally to avoid circular import.
    Keep the override map in sync with orchestrator._TOOL_ENTITY_OVERRIDE.
    """
    _ov: dict[str, str] = {
        "checkout": "order", "get_cart": "order", "clear_cart": "order",
        "add_to_cart": "order", "remove_from_cart": "order",
        "update_cart_quantity": "order",
        "rate_order": "order", "return_order": "order", "reorder": "order",
        "apply_coupon": "order",
        "get_balance": "account", "transfer": "account",
        "wire_transfer": "account", "deposit": "account", "withdraw": "account",
        "apply_loan": "account", "bill_pay": "account",
        "get_history": "account", "get_statement": "account",
        "pay_invoice": "invoice", "get_invoice": "invoice",
        "refund_invoice": "invoice", "cancel_payment": "payment",
        "get_payment": "invoice",
        "add_to_wishlist": "wishlist",
        "move_to_thread": "email", "list_inbox": "email",
        "get_thread": "email", "get_attachments": "email",
        "mark_read": "email", "mark_unread": "email",
        "add_label": "email", "remove_label": "email",
        "chmod": "file", "chown": "file", "cp": "file", "rm": "file", "mv": "file",
        "mkdir": "file", "touch": "file",
    }
    _domain_ov: dict[str, dict[str, str]] = {
        "banking": {
            "schedule_transfer": "scheduled_transfer",
            "cancel_transfer": "scheduled_transfer",
        },
        "issue_tracker": {
            "add_label": "issue",
            "remove_label": "issue",
            "add_watcher": "issue",
            "remove_watcher": "issue",
            "set_milestone": "issue",
            "time_track": "issue",
            "add_to_sprint": "issue",
            "remove_from_sprint": "issue",
            "create_subtask": "issue",
        },
        "team_chat": {
            "get_thread": "thread",
            "get_user_status": "user",
            "send_dm": "user",
        },
        "calendar": {
            "add_attendee": "event",
            "remove_attendee": "event",
            "set_reminder": "event",
            "create_recurring": "event",
        },
        "food_delivery": {
            "get_menu": "restaurant",
            "filter_by_dietary": "restaurant",
            "get_popular_items": "restaurant",
            "add_tip": "order",
            "contact_support": "order",
            "rate_order": "order",
        },
        "filesystem": {
            "ls": "file",
            "cat": "file",
            "stat": "file",
            "head": "file",
            "tail": "file",
            "find": "file",
            "grep": "file",
            "tree": "file",
            "pwd": "file",
            "du": "file",
            "df": "file",
        },
    }
    tool = name.lower()
    server = domain.lower()
    if server and tool in _domain_ov.get(server, {}):
        return _domain_ov[server][tool]
    if tool in _ov:
        return _ov[tool]
    for et in ("event", "order", "account", "email", "invoice",
                "issue", "lead", "deal", "product", "restaurant",
                "channel", "message", "file", "contact", "payment",
                "menu", "cart", "transfer", "transaction"):
        if et in name:
            return et
    return name.split("_")[-1] if "_" in name else name


def _tool_required_entities(name: str, domain: str = "") -> set[str]:
    try:
        from src.live_mcp.orchestrator import _tool_existing_entity_requirements
        return set(_tool_existing_entity_requirements(name, domain))
    except Exception:
        return set()


def derive_progress_predicates(
    oracle_calls: list[OracleCall],
    domain: str,
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
        entity = _tool_entity(call.tool_name, domain)
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
        if _is_mutating_tool(call.tool_name):
            entity = _tool_entity(call.tool_name, domain)
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
        curr_is_mutate = _is_mutating_tool(curr.tool_name)
        if not ((prev_is_read or prev_is_creator) and curr_is_mutate):
            continue
        prev_entity = _tool_entity(prev.tool_name, domain)
        curr_entity = _tool_entity(curr.tool_name, domain)
        curr_required_entities = _tool_required_entities(curr.tool_name, domain)
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
    initial_state — never for untouched entities (which would otherwise
    flood the criteria list and systematically deflate r_coverage when
    the verifier cannot reconstruct the full final state from the
    last observation).
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
    if "add_to_cart" in tool_names and "cart" in final_state:
        criteria.append({"type": "cart_not_empty", "server": domain})
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
    if "send_email" in tool_names:
        criteria.append({
            "type": "email_count_gte", "server": domain,
            "value": len(final_state.get("emails", {})),
        })
    if any(t in tool_names for t in ("write_file", "create_file", "mkdir")):
        init_fs = (initial_state.get("fs") or {}) if isinstance(initial_state.get("fs"), dict) else {}
        for path in final_state.get("fs", {}):
            if path in init_fs:
                continue  # only assert newly created paths
            criteria.append({"type": "file_exists", "server": domain, "path": path})
    if "send_message" in tool_names:
        for ch_id, ch in final_state.get("channels", {}).items():
            init_ch = (initial_state.get("channels") or {}).get(ch_id) or {}
            init_count = len(init_ch.get("messages", [])) if isinstance(init_ch, dict) else 0
            final_count = len(ch.get("messages", []))
            if final_count == init_count:
                continue
            criteria.append({
                "type": "state_equals", "server": domain,
                "path": f"channels.{ch_id}.messages_count",
                "value": final_count,
            })
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
) -> tuple[bool, float, int, int, bool, int]:
    """Replay oracle trace against a fresh session to verify it's reproducible.

    Counts only schema-level and execution errors (not empty-result responses).
    PROVE's corpus filter permits up to 30% replay errors; use the same
    threshold here. Training export still applies a separate contract filter
    for terminal shape and tool-call budget.

    Returns:
        (passed, error_rate, num_errors, num_calls, criteria_ok, criteria_failed)
        - passed: True if error_rate <= max_error_rate (default 0.30)
          ONLY counts schema/execution errors — paper-definition aligned.
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
    try:
        manager.discover_tools(session.session_id)
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
            expected_success = getattr(call, "expected_success", None)
            if expected_success is False and result.success:
                # Replay must preserve the Teacher attempt outcome. A call that
                # failed during synthesis but succeeds after reset can mutate
                # state and no longer represents the completed conversation.
                num_errors += 1
                continue
            if not result.success or not result.schema_valid:
                # Count only schema/execution errors, not empty-result responses.
                # PROVE: "We count only schema-level and execution errors
                # (not empty-result responses)."
                #
                # Schema validation failures (schema_valid=False) are ALWAYS
                # counted as errors — the observation dict may lack an "error"
                # key, containing only validation details.
                # A failed execution is an execution error regardless of its
                # message text. PROVE exempts successful empty-result responses,
                # not precondition/schema failures such as "entity not found".
                num_errors += 1

        # ── Criteria check (independent of tool-error-rate) ──
        # PROVE's 30% threshold applies ONLY to schema/execution errors.
        # Criteria validation is a separate quality signal — NOT merged
        # into num_errors so it does not pollute the paper's error_rate.
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
            # Project-specific quality gate (NOT part of PROVE / OVAL-MCP §5.0):
            # Mutating tool calls with empty success_criteria are suspicious
            # (e.g. teacher called update_* but state was already at target).
            # PROVE does NOT reject these — R_coverage falls back to pure
            # tool-call matching.  We log a diagnostic warning but accept
            # the task so the teacher retry budget isn't wasted.
            tool_call_count = sum(1 for c in oracle_calls if c.action == "tool_call")
            if tool_call_count > 0:
                has_mutating = any(
                    _is_mutating_tool(c.tool_name)
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
        passed = error_rate <= max_error_rate

        return passed, error_rate, num_errors, num_calls, criteria_ok, criteria_failed
    finally:
        manager.close_session(session.session_id)


# ═══════════════════════════════════════════════════════════════════════
# Sensitive parameter provenance check (PROVE §3.2 Step 5)
# ═══════════════════════════════════════════════════════════════════════

# Parameter names indicative of sensitive data (PROVE: passwords, tokens, etc.)
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
    user_queries: list[str] | None = None,
    call_round_indices: list[int] | None = None,
) -> tuple[bool, list[dict[str, Any]]]:
    """PROVE §3.2 Step 5: check that sensitive parameters are traceable.

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
        for param_name, param_value in call.arguments.items():
            param_lower = param_name.lower()

            # Check if this parameter looks sensitive
            is_sensitive = any(p in param_lower for p in _SENSITIVE_PARAM_PATTERNS)
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
            traceable = any(param_str in src for src in traceable_values)

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
            args_parts.append(f"{k}{req}{desc_part}")
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
            fields.append(f"{name}{marker}: {child_hint or 'any'}")
        return "object{" + ", ".join(fields) + "}"
    constraints = []
    if "minimum" in schema:
        constraints.append(f"minimum={schema['minimum']}")
    if constraints:
        return f"{ptype}({', '.join(constraints)})"
    return ptype


def _format_state_compact(state: dict[str, Any], max_entities: int = 20) -> str:
    """Format grounded state as compact entity summaries (PROVE §4 sampling context).

    Instead of dumping full JSON (which can exceed teacher attention window),
    output one line per entity with key fields only.
    """
    if not isinstance(state, dict) or not state:
        return "(empty state)"

    lines: list[str] = []
    count = 0
    for entity_type, entities in sorted(state.items()):
        if not isinstance(entities, dict):
            continue
        for entity_id, entity_data in sorted(entities.items()):
            if count >= max_entities:
                lines.append(f"... ({sum(len(v) if isinstance(v, dict) else 0 for v in state.values())} total entities, showing first {max_entities})")
                return "\n".join(lines)
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
                summary = ", ".join(id_fields[:5])
                lines.append(f"  {entity_type}/{entity_id}: {summary}" if summary else f"  {entity_type}/{entity_id}")
            else:
                lines.append(f"  {entity_type}/{entity_id}: {entity_data}")
            count += 1
    if not lines:
        return str(state)[:2000]
    return "\n".join(lines)


def _format_history(history: list[dict[str, Any]]) -> str:
    """Format execution history for the LLM prompt."""
    if not history:
        return "(no actions yet — this is the first turn)"
    lines = []
    for i, entry in enumerate(history, 1):
        tool = entry.get("tool_name", "?")
        args = _json.dumps(entry.get("arguments", {}), ensure_ascii=False)
        obs = entry.get("observation")
        success = entry.get("success", True)
        lines.append(
            f"Step {i}: {tool}({args}) → "
            f"{'OK' if success else 'FAILED'}"
        )
        if isinstance(obs, dict):
            obs_str = _json.dumps(obs, ensure_ascii=False, default=str)
            lines.append(f"  Result: {obs_str[:500]}")
        elif obs:
            lines.append(f"  Result: {str(obs)[:500]}")
    return "\n".join(lines)
