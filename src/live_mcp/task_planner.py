"""PROVE-style state-machine task generation.

LLM-in-the-loop at every turn: the LLM sees domain + tool schemas + live state
+ full execution history, and decides the next action (tool_call with arguments,
or terminal).  Oracle trace is the recorded interaction — no heuristic parameter
inference needed.

Pipeline per task:
  1. create_session(seed) — fresh isolated state
  2. LLM generates user_query
  3. Loop (max_turns):
     a. LLM decides next action: tool_call(name, args) | final_answer | report_error
     b. Execute tool_call against live MCP → record observation
     c. Apply execution perturbations (intermittent errors, pagination, …)
     d. Append to history
  4. Derive success criteria from state delta
  5. Replay validation against fresh session
  6. Robustness knobs applied post-generation
"""

from __future__ import annotations

import copy
import json as _json
import random
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from src.live_mcp.types import OracleCall
from src.utils import extract_json as _extract_json


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
# Turn-decay schedule (PROVE §6: min_turns=2, max_turns=3 per chain)
# ═══════════════════════════════════════════════════════════════════════

class ContinuationPolicy:
    """PROVE-style turn-decay schedule for deciding when to end a conversation.

    The target turn count depends on chain length:
      chain_len=2 → 2-3 turns
      chain_len=3 → 3-4 turns
      chain_len=4 → 4-5 turns
      chain_len=5 → 5-6 turns

    Perturbations (intermittent errors, pagination) may add 1-2 extra turns.
    """

    @staticmethod
    def target_turns(chain_length: int, rng: random.Random) -> int:
        """Return target turn count for a given chain length.

        PROVE §6: min_turns=2, max_turns=3 per chain round.
        Conservative: exact = chain_length + 1 (query + N tools + terminal),
        no jitter.  Perturbations (intermittent errors) may cause additional
        retry turns beyond this budget, handled by recovery.
        """
        return max(2, chain_length + 1)

    @staticmethod
    def should_continue(turn: int, target: int, last_action_success: bool, tool_calls_done: int) -> bool:
        """Decide whether the conversation should continue.

        Conservative: stop after target turns.  Allow 1 extra turn only if
        the last action was unsuccessful (recovery budget).
        """
        if turn >= target:
            return False
        return True

    @staticmethod
    def conversation_rounds(rng: random.Random) -> int:
        """PROVE §3.2 Step 3.5: turn-decay schedule.

        PROVE keeps generated conversations between 2 and 3 user turns.  The
        training rollout injects these follow-up user turns after intermediate
        terminal actions, so the oracle remains a single live conversation
        instead of hidden teacher history in the initial prompt.
        """
        return 2 if rng.random() < 0.7 else 3


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

    def __init__(self, client: object, domain: str, seed: int = 0):
        self.client = client
        self.domain = domain
        self.domain_desc = DOMAIN_DESCRIPTIONS.get(domain, "")
        self._strip_enums = random.Random(seed).random() < 0.30  # per-task seed, aligns with PROVE

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
    ) -> str:
        """LLM generates a natural-language user query grounded in live state.

        PROVE §4: injects persona (character role) and reference_date (temporal anchor)
        to increase query diversity. chain_seed guides the query toward a
        realistic dependency chain without making that chain the only valid
        tool trajectory.
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

        # Chain hint: describe the multi-step flow in natural language
        # so the generated query implies a sequence of operations
        chain_hint = ""
        if chain_seed and len(chain_seed) >= 2:
            tool_desc_map = {t["name"]: t.get("description", "") for t in tool_schemas}
            step_descs = []
            for tn in chain_seed:
                desc = tool_desc_map.get(tn, "")
                # Use first sentence of description as the natural hint
                first_sent = desc.split(".")[0].strip() if desc else tn
                step_descs.append(first_sent)
            flow_text = " → ".join(step_descs)
            chain_hint = (
                f"\n## Underlying Task Flow\n"
                f"The user's task requires this sequence: {flow_text}\n"
                f"Express this as a SINGLE natural goal — do NOT list steps or mention tool names.\n"
            )
        user = f"""## Persona
{persona if persona else 'A normal user messaging their AI assistant.'}
{date_block}
## What this assistant can help with
{self.domain_desc}
{chain_hint}
## Current State (real IDs and values)
{state_text}

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
        execution_history: list[dict[str, Any]],
        difficulty: str,
        rng: random.Random,
        persona: str = "",
        reference_date: str = "",
        chain_seed: list[str] | None = None,
        chain_progress: int = 0,
    ) -> str:
        """Generate a follow-up user message grounded in previous results.

        PROVE §3.2 Step 3.5: the user continues the conversation based on
        what the assistant just did. The follow-up should reference previous
        outputs naturally (e.g., "that looks right, now also...") without
        mentioning tool names.

        chain_seed + chain_progress: guides the follow-up to ask for the
        remaining dependency chain steps naturally.
        """
        state_text = _format_state_compact(grounded_state, max_entities=20)
        # Summarize last few tool results for the follow-up generator
        recent = execution_history[-5:]
        history_lines = []
        for step in recent:
            tool = step.get("tool_name", "?")
            obs = step.get("observation", {})
            if isinstance(obs, dict):
                summary = _json.dumps(obs, ensure_ascii=False, default=str)[:300]
            else:
                summary = str(obs)[:300]
            history_lines.append(f"  called {tool} → {summary}")
        history_text = "\n".join(history_lines) if history_lines else "(no history)"

        date_block = ""
        if reference_date:
            date_block = f"\n## Reference Date\nToday is {reference_date}. Use relative dates when appropriate.\n"

        system = (
            "You are role-playing as a real person who just received a response "
            "from their AI assistant. Write ONE short follow-up message — the way "
            "a real human would actually type it.\n\n"
            "Real people react to what they just heard and ask for the NEXT thing. "
            "They might say 'great, now also...' or 'actually, can you change...' "
            "or 'oh wait, I also need...'. They don't re-explain the original task.\n\n"
            "DO NOT mention tool names or steps. Just state what you want next.\n"
            "Keep it to 1-2 sentences. Be natural and casual."
        )

        # Chain context: guide the follow-up to ask for remaining dependency steps
        chain_guide = ""
        if chain_seed and chain_progress < len(chain_seed):
            remaining = chain_seed[chain_progress:]
            tool_desc_map = {t["name"]: t.get("description", "").split(".")[0].strip() or t["name"] for t in tool_schemas}
            remaining_descs = [tool_desc_map.get(tn, tn) for tn in remaining]
            chain_guide = (
                f"\n## Remaining task flow\n"
                f"The task still needs these operations (in order): {' → '.join(remaining_descs)}.\n"
                f"Your follow-up should naturally steer toward the next step without mentioning tool names.\n"
            )

        user = f"""## Persona
{persona if persona else 'A normal user messaging their AI assistant.'}
{date_block}
## Original request
"{previous_query}"

## Recent results (what the assistant just did)
{history_text}
{chain_guide}
## Current State (real IDs and values)
{state_text}

## Your task
Write ONE short follow-up message. Difficulty: {difficulty}.
React to what just happened and ask for the next thing naturally.

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

    # ── Step 2-N: decide next action (LLM-in-the-loop) ──

    def decide_action(
        self,
        tool_schemas: list[dict[str, Any]],
        user_query: str,
        execution_history: list[dict[str, Any]],
        attempt: int = 0,
        dep_hints: str = "",
        difficulty: str = "complete",
        chain_seed: list[str] | None = None,
        chain_progress: int = 0,
        reference_date: str = "",
    ) -> ActionPlan:
        """LLM decides the next action given full context.

        Called at every turn.  The LLM sees the complete execution history
        and current state, so its decisions are grounded in real values.

        For 'missing' difficulty tasks, ask_clarification is expected on the
        first turn (the query deliberately omits a parameter), so the
        first-turn enforcement is relaxed.

        chain_seed + chain_progress: guides the LLM toward multi-step tasks,
        showing which tools have been called and which remain.
        """
        tools_text = _format_tools(tool_schemas, strip_enums=self._strip_enums)
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
        # For complete/minimal difficulty, the prompt guides the model to
        # use tools first, but ask_clarification is allowed if genuinely needed.
        if not execution_history:
            if difficulty == "missing":
                first_turn_hint = (
                    "\nNote: This task has a MISSING parameter. "
                    "ask_clarification may be needed before calling a tool.\n"
                )
                default_action = "ask_clarification"
                blocked_first = ("final_answer", "report_error")
            else:
                first_turn_hint = (
                    "\nThis is your FIRST turn. Call a tool to make progress on the task. "
                    "Resolve ambiguities via tool calls before asking the user.\n"
                )
                default_action = "tool_call"
                blocked_first = ("final_answer", "report_error")
        else:
            first_turn_hint = ""
            default_action = "final_answer"
            blocked_first = ()

        # Chain progress guide: show the LLM which tools have been called
        # and which remain, preventing premature final_answer.
        chain_guide = ""
        if chain_seed and len(chain_seed) >= 2:
            tool_desc_map = {t["name"]: t.get("description", "").split(".")[0].strip() or t["name"] for t in tool_schemas}
            lines = ["## Task Progress"]
            for i, tn in enumerate(chain_seed):
                if i < chain_progress:
                    marker = "✓ done"
                elif i == chain_progress:
                    marker = "← NEXT"
                else:
                    marker = ""
                desc = tool_desc_map.get(tn, tn)
                lines.append(f"  {i+1}. {tn} ({desc}) {marker}")
            lines.append(
                "Use this dependency chain as a planning prior. Prefer the NEXT "
                "tool when it fits the user request and live state, but choose "
                "another valid action if the observation indicates a better path."
            )
            chain_guide = "\n".join(lines) + "\n"

        # Chain-seed guidance: PROVE uses dependency chains to seed grounded
        # multi-step queries.  The chain is a planning prior, not a hidden
        # script that the teacher must match exactly.
        next_tool_hint = ""
        tool_desc = ""
        # Missing-difficulty tasks may need clarification instead of the next
        # chain tool, so keep this hint out of that path.
        if (chain_seed and chain_progress < len(chain_seed)
                and difficulty != "missing"):
            next_tool = chain_seed[chain_progress]
            for t in tool_schemas:
                if t["name"] == next_tool:
                    desc = t.get("description", "").split(".")[0].strip()
                    tool_desc = f" — {desc}" if desc else ""
                    break
            next_tool_hint = (
                f"\n## Planning Hint\n"
                f"The sampled dependency chain suggests \"{next_tool}\"{tool_desc} "
                f"as the next useful tool. Prefer it when it fits the user request "
                f"and current state; otherwise choose the best valid tool or terminal action.\n"
            )

        date_guide = (
            f"## Reference Date\nToday is {reference_date}. Do not invent dates from an earlier year.\n"
            if reference_date else ""
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
        if next_tool_hint:
            system += (
                "\n\nUse the planning hint as guidance, but keep the trajectory "
                "consistent with the user request and live execution state."
            )
        user = f"""## Domain
{self.domain_desc}

## Available Tools
{tools_text}

{dep_hints}
{chain_guide}
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

                # On first turn, reject blocked action types (only final_answer/report_error)
                if not execution_history and action in blocked_first:
                    logger.debug(
                        f"decide_action first turn rejected '{action}' for {self.domain} "
                        f"(difficulty={difficulty}), retrying (attempt {_retry + 1}/3). LLM raw: {raw[:120]}..."
                    )
                    continue

                # Validate: tool_call MUST have a non-empty tool_name
                if action == "tool_call":
                    tool_name = data.get("tool_name", "").strip()
                    if not tool_name:
                        logger.debug(
                            f"decide_action got tool_call with empty tool_name for {self.domain}, "
                            f"retrying (attempt {_retry + 1}/3). LLM raw: {raw[:120]}..."
                        )
                        continue
                elif action in _VALID_TERMINALS:
                    tool_name = ""
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
        tools_text = _format_tools(tool_schemas, strip_enums=self._strip_enums)
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

# Domain → perturbation group mapping, per PROVE §6 step 5a
_DOMAIN_PERTURBATION_GROUP = {
    # PROVE mappings
    "filesystem":    "filesystem_terminal",
    "calendar":      "calendar_crm",
    "crm":           "calendar_crm",
    "email":         "email_teamchat",
    "team_chat":     "email_teamchat",
    "shopping":      "search_shopping",
    # Extended mappings (closest PROVE category)
    "banking":       "transactional",
    "payments":      "transactional",
    "food_delivery": "lifecycle",
    "issue_tracker": "workflow",
}

_PERTURBATION_SPEC = {
    "filesystem_terminal": ["intermittent_api_error", "partial_batch_failure"],
    "calendar_crm":        ["intermittent_api_error", "partial_batch_failure"],
    "email_teamchat":      ["paginated_response", "partial_batch_failure"],
    "search_shopping":     ["paginated_response", "incomplete_intermediate"],
    "transactional":       ["intermittent_api_error", "partial_batch_failure"],
    "lifecycle":           ["intermittent_api_error", "partial_batch_failure"],
    "workflow":            ["intermittent_api_error", "partial_batch_failure"],
}

# Per-type probability (~0.10 each, total ~0.20 within PROVE 0.15–0.30)
_PERTURBATION_PROB = {
    "intermittent_api_error":   0.10,
    "paginated_response":       0.10,
    "incomplete_intermediate":  0.10,
    "partial_batch_failure":    0.10,
}


def _perturb_intermittent_api_error(
    observation: dict[str, Any] | str | None,
    rng: random.Random,
) -> dict[str, Any] | str | None:
    """Return internal server error → oracle retries the same call."""
    return {"error": "Internal Server Error", "retry": True}


def _perturb_paginated_response(
    observation: dict[str, Any] | str | None,
    rng: random.Random,
) -> dict[str, Any] | str | None:
    """Wrap partial results with a cursor → oracle must paginate."""
    if not isinstance(observation, dict):
        return None
    items = observation.get("items", observation.get("results", []))
    if isinstance(items, list) and len(items) > 1:
        mid = max(1, len(items) // 2)
        return {**observation, "items": items[:mid], "next_cursor": "page_2"}
    return None


def _perturb_incomplete_intermediate(
    observation: dict[str, Any] | str | None,
    rng: random.Random,
) -> dict[str, Any] | str | None:
    """Return snippets instead of full details → oracle must extract/get detail.

    PROVE: intermediate search results only show summaries, forcing
    subsequent extract/detail calls to retrieve complete information.
    """
    if not isinstance(observation, dict):
        return None
    result_keys = ("items", "results", "matches", "entries", "records")
    if not any(k in observation for k in result_keys):
        return None
    total = 0
    for k in result_keys:
        v = observation.get(k)
        if isinstance(v, list):
            total = len(v)
            break
    if total < 2:
        return None
    summary = {
        "summary": "Partial results returned. Use get_detail / extract / get_item "
                   "to retrieve complete information.",
        "snippet_count": min(total, rng.randint(1, 3)),
        "requires_detail_fetch": True,
    }
    for k in ("total", "count", "next_cursor", "cursor"):
        if k in observation:
            summary[k] = observation[k]
    return summary


def _perturb_partial_batch_failure(
    observation: dict[str, Any] | str | None,
    rng: random.Random,
) -> dict[str, Any] | str | None:
    """Mark a subset of batch results as failed → oracle must retry individually.

    PROVE: bulk updates where some items fail. The model must inspect results
    and re-process failed items one by one.
    """
    if not isinstance(observation, dict):
        return None
    batch_keys = ("results", "updated", "processed", "created")
    batch_key = None
    items = []
    for k in batch_keys:
        v = observation.get(k)
        if isinstance(v, list) and len(v) > 1:
            batch_key = k
            items = v
            break
    if not items:
        return None
    fail_count = max(1, len(items) // 3)
    fail_indices = rng.sample(range(len(items)), fail_count)
    new_items = list(items)
    for idx in fail_indices:
        if isinstance(new_items[idx], dict):
            new_items[idx] = {
                **new_items[idx],
                "status": "failed",
                "error": "Transient processing failure — retry required",
            }
    return {**observation, batch_key: new_items, "partial_failure": True, "failed_count": fail_count}


_PERTURBATION_HANDLERS = {
    "intermittent_api_error":   _perturb_intermittent_api_error,
    "paginated_response":       _perturb_paginated_response,
    "incomplete_intermediate":  _perturb_incomplete_intermediate,
    "partial_batch_failure":    _perturb_partial_batch_failure,
}


def apply_perturbation(
    observation: dict[str, Any] | str | None,
    domain: str,
    rng: random.Random,
) -> dict[str, Any] | str | None:
    """Apply domain-specific execution perturbation (PROVE-style).

    Total perturbation probability: ~0.20 per tool call (PROVE: 0.15–0.30).

    Perturbation types (per domain group):

    ======================  ==============================================  ============================
    Domain group             Domains                                         Perturbation types
    ======================  ==============================================  ============================
    filesystem_terminal      filesystem                                      intermittent, partial_batch
    calendar_crm             calendar, crm                                   intermittent, partial_batch
    email_teamchat           email, team_chat                                paginated, partial_batch
    search_shopping          shopping                                        paginated, incomplete
    transactional            banking, payments                               intermittent, partial_batch
    lifecycle                food_delivery                                   intermittent, partial_batch
    workflow                 issue_tracker                                   intermittent, partial_batch
    ======================  ==============================================  ============================

    Each applicable type is rolled independently at ~0.10 probability.
    Types that don't match the observation structure silently skip
    (e.g., paginated_response on a non-list observation does nothing).
    """
    group = _DOMAIN_PERTURBATION_GROUP.get(domain, "transactional")
    pert_types = _PERTURBATION_SPEC.get(group, ["intermittent_api_error"])

    for ptype in pert_types:
        prob = _PERTURBATION_PROB.get(ptype, 0.10)
        if rng.random() < prob:
            handler = _PERTURBATION_HANDLERS.get(ptype)
            if handler:
                result = handler(observation, rng)
                if result is not None:
                    return result

    return observation


# ═══════════════════════════════════════════════════════════════════════
# Success criteria derivation (from state delta)
# ═══════════════════════════════════════════════════════════════════════

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
                    for ek in ("status", "stage", "type", "state"):
                        if ek in entity and entity[ek] is not None:
                            criteria.append({
                                "type": "state_equals", "server": domain,
                                "path": f"{key}.{nk}.{ek}",
                                "path_parts": [key, nk, ek],
                                "value": entity[ek],
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
                            criteria.append({
                                "type": "state_equals", "server": domain,
                                "path": f"{key}.{ck}.{fk}",
                                "path_parts": [key, ck, fk],
                                "value": fe[fk],
                            })

    # Domain-specific semantic criteria
    tool_names = [c.tool_name for c in oracle_calls if c.action == "tool_call"]
    criteria.extend(_domain_criteria(tool_names, initial_state, final_state, domain))

    # P3b FIX: Removed the empty-path fallback below.
    # Previously: if not criteria: criteria.append({"type": "state_exists",
    # "server": domain, "path": ""})
    # This inserted a criterion that task_reward.py skips (if not path: continue),
    # making it a no-op that consumed a criteria slot and silently lowered
    # r_coverage.  When no criteria can be derived, keep the list empty —
    # r_coverage denominator uses max(outcome_count + criteria_count, 1), so
    # an empty criteria list degrades gracefully to outcome-only coverage.

    return criteria


def _tool_entity(name: str) -> str:
    """Extract entity type from tool name.

    Mirrors orchestrator._tool_entity.  Defined locally to avoid circular import.
    """
    for et in ("event", "order", "account", "email", "invoice",
                "issue", "lead", "deal", "product", "restaurant",
                "channel", "message", "file", "contact", "payment",
                "menu", "cart", "transfer", "transaction"):
        if et in name:
            return et
    return name.split("_")[-1] if "_" in name else name


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
        entity = _tool_entity(call.tool_name)
        if is_read and entity not in resolved_entities:
            resolved_entities.add(entity)
            predicates.append({
                "step": i,
                "type": "resolved_required_entity",
                "tool": call.tool_name,
                "entity": entity,
            })

    # Step 2: completed_required_transition for mutation calls
    mutate_prefixes = ("create_", "update_", "delete_", "remove_", "add_",
                       "send_", "transfer", "pay_", "checkout", "transition_",
                       "convert_", "archive_", "mkdir", "touch", "mv", "cp",
                       "chmod", "write_", "set_", "apply_")
    for i, call in enumerate(real_calls):
        is_mutate = any(call.tool_name.lower().startswith(p) for p in mutate_prefixes)
        if is_mutate:
            entity = _tool_entity(call.tool_name)
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

    # Step 3: satisfied_dependency_edge for consecutive calls where predecessor's
    # entity type is used by successor
    for i in range(1, len(real_calls)):
        prev = real_calls[i - 1]
        curr = real_calls[i]
        # A dependency edge is satisfied when prev resolved/produced an entity
        # type that curr consumes or mutates
        prev_entity = _tool_entity(prev.tool_name)
        curr_entity = _tool_entity(curr.tool_name)
        if prev_entity == curr_entity:
            predicates.append({
                "step": i,
                "type": "satisfied_dependency_edge",
                "tool": curr.tool_name,
                "from_step": i - 1,
                "entity": curr_entity,
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
    manager: object,
    executor: object,
    seed: int,
    domain: str,
    success_criteria: list[dict[str, Any]] | None = None,
    max_error_rate: float = 0.30,
) -> tuple[bool, float, int, int]:
    """Replay oracle trace against a fresh session to verify it's reproducible.

    Counts only schema-level and execution errors (not empty-result responses).
    PROVE's corpus filter permits up to 30% replay errors; use the same
    threshold here. Training export still applies a separate contract filter
    for terminal shape and tool-call budget.

    Returns:
        (passed, error_rate, num_errors, num_calls)
        - passed: True if no schema/execution error occurred
        - error_rate: fraction of calls that failed
        - num_errors: count of schema/execution errors
        - num_calls: total tool calls replayed
    """
    session = manager.create_session(seed=seed)
    num_errors = 0
    num_calls = 0
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
                domain=domain,
            )
            num_calls += 1
            if not result.success or not result.schema_valid:
                # Count only schema/execution errors, not empty-result responses.
                # PROVE: "We count only schema-level and execution errors
                # (not empty-result responses)."
                #
                # Schema validation failures (schema_valid=False) are ALWAYS
                # counted as errors — the observation dict may lack an "error"
                # key, containing only validation details.
                if not result.schema_valid:
                    num_errors += 1
                    continue

                obs = result.observation
                if isinstance(obs, dict):
                    err_msg = obs.get("error", "")
                    # Empty results (e.g., search returned 0 items) are NOT errors
                    empty_indicators = (
                        "not found", "no results", "empty", "no items",
                        "0 results", "no matches",
                    )
                    if err_msg and not any(ind in str(err_msg).lower() for ind in empty_indicators):
                        num_errors += 1
                elif isinstance(obs, str):
                    empty_indicators = (
                        "not found", "no results", "empty", "no items",
                        "0 results", "no matches",
                    )
                    if not any(ind in obs.lower() for ind in empty_indicators):
                        num_errors += 1
                else:
                    # Unknown error type — count as error
                    num_errors += 1

        if success_criteria:
            from src.live_mcp.oracle import criterion_satisfied

            replay_state = manager.get_state(session.session_id)
            criteria_errors = sum(
                1 for criterion in success_criteria
                if not criterion_satisfied(replay_state, criterion)
            )
            # PROVE allows up to 30% error rate on tool calls.  For state
            # criteria we use a floor-based 25% threshold: int(N*0.25).
            # With 1-3 criteria this is 0 (all must pass); with 4+ criteria
            # this is 1+ (one failure tolerated).  Unlike max(1, ...) this
            # does not silently accept the only failing criterion when N=1.
            threshold = int(len(success_criteria) * 0.25)
            num_errors += criteria_errors if criteria_errors > threshold else 0

        error_rate = num_errors / num_calls if num_calls > 0 else float(num_errors > 0)
        passed = error_rate <= max_error_rate

        return passed, error_rate, num_errors, num_calls
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
    execution_history: list[dict[str, Any]],
) -> tuple[bool, list[dict[str, Any]]]:
    """PROVE §3.2 Step 5: check that sensitive parameters are traceable.

    Sensitive parameters (passwords, tokens, API keys, etc.) must appear ONLY
    when traceable to prior user turns or tool outputs. Parameters that appear
    "from nowhere" indicate the teacher LLM hallucinated them, which is a
    security risk in training data.

    Returns:
        (passed, violations)
        - passed: True if all sensitive parameters are traceable
        - violations: list of dicts describing each violation
          [{"param": str, "value": str, "tool": str, "reason": str}, ...]
    """
    violations: list[dict[str, Any]] = []

    # Build corpus of traceable values from user query and prior tool outputs.
    # P1-8 fix: previously this loaded the ENTIRE execution_history before
    # checking the first call — the first call could "validate" against a
    # value that only appeared in a later step's observation. Now we walk
    # the timeline strictly: only step i-1's observation is visible when
    # checking call i's arguments.
    initial_traceable: list[str] = [user_query]

    # Check each oracle call's arguments for sensitive params
    traceable_values: list[str] = list(initial_traceable)
    for idx, call in enumerate(oracle_calls):
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
        if idx < len(execution_history):
            step_obs = execution_history[idx].get("observation")
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
            ptype = info.get("type", "")
            enum_str = f": {', '.join(info['enum'])}" if "enum" in info else ""
            desc_part = f" ({ptype}{enum_str})" if ptype else ""
            args_parts.append(f"{k}{req}{desc_part}")
        args_str = ", ".join(args_parts)
        lines.append(f"  - {name}({args_str}): {desc}")
    return "\n".join(lines)


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
