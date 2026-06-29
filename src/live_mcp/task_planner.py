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
        "Calendar management system. Users can list events, search by date/keyword, "
        "create/update/delete events, manage recurring events, add/remove attendees, "
        "check free/busy slots, set reminders, change timezone, and export calendars. "
        "Events have start_time, end_time, title, description, attendees, location, "
        "and recurrence rules."
    ),
    "shopping": (
        "E-commerce shopping system. Users can search products by category and price, "
        "view product details, compare products, get recommendations, manage cart "
        "(add/update/remove items), apply coupons, checkout, view orders, track "
        "shipments, return items, write reviews, and manage wishlists."
    ),
    "banking": (
        "Banking system. Users can list accounts, view balances, get transaction "
        "history, transfer funds between accounts, wire transfer externally, "
        "deposit, withdraw, pay bills, schedule/cancel transfers, freeze/unfreeze "
        "accounts, verify account ownership, check exchange rates, and apply for loans."
    ),
    "email": (
        "Email system. Users can list inbox, search emails, read individual emails, "
        "send emails, create drafts, forward/reply, add/remove labels, manage threads, "
        "archive, mark read/unread, create filters, and view attachments. "
        "Emails are append-only (no delete)."
    ),
    "filesystem": (
        "Unix-like filesystem. Users can navigate (ls, cd, pwd), read files (cat, head, "
        "tail, wc), manage files (mkdir, touch, mv, cp, rm), set permissions (chmod, "
        "chown), check disk usage (du, df), create symlinks, archive (tar, zip), "
        "diff files, sort, compute checksums, and more. Protected paths exist "
        "(e.g., /protected/). Root ownership cannot be transferred."
    ),
    "payments": (
        "Payment processing system. Users can create invoices, view invoices, pay "
        "invoices, issue refunds, cancel payments, dispute invoices, create webhooks, "
        "and manage webhook subscriptions. Invoices have status: pending, paid, "
        "refunded, cancelled, disputed."
    ),
    "crm": (
        "CRM system. Users can create/update/convert/delete leads, manage contacts, "
        "create/update deals, track tasks, add notes to leads/contacts/deals. "
        "Leads flow through status: new → contacted → qualified → converted/lost. "
        "Deals have stages: prospecting → proposal → negotiation → closed_won/closed_lost."
    ),
    "issue_tracker": (
        "Issue tracking system. Users can create/get/list/update issues, assign to "
        "team members, transition workflow states (open→in_progress→in_review→resolved→closed), "
        "comment, add/remove labels and watchers, manage sprints, subtasks, time tracking, "
        "and milestones. State transitions are strictly enforced."
    ),
    "team_chat": (
        "Team chat system. Users can list/join/create/archive channels, send messages "
        "to channels, send direct messages, create message threads, add reactions, "
        "search messages, and view user status. Messages are append-only."
    ),
    "food_delivery": (
        "Food delivery system. Users can list/search restaurants, view menus, filter "
        "by dietary restrictions, view popular items, create/cancel orders, track "
        "delivery status, rate orders, add tips, reorder past orders, and contact support. "
        "Order lifecycle: confirmed→preparing→in_transit→delivered (can only cancel before preparing)."
    ),
}

DIFFICULTY_DESCRIPTIONS: dict[str, str] = {
    "complete": (
        "User query contains ALL information needed — tool, arguments, and "
        "explicit goal clearly stated. Model should execute without asking."
    ),
    "missing": (
        "User query OMITS ONE critical parameter (e.g., date, recipient, "
        "amount). Model must ask_clarification before proceeding."
    ),
    "minimal": (
        "User query is very BRIEF — just an intent, no specifics. "
        "Model must infer details from context or ask clarification."
    ),
}

# ═══════════════════════════════════════════════════════════════════════
# Persona templates & reference dates (PROVE §4 diversity injection)
# ═══════════════════════════════════════════════════════════════════════

_PERSONA_TEMPLATES: list[str] = [
    "a busy team lead with back-to-back meetings",
    "a freelancer managing multiple clients",
    "a graduate student organizing a research project",
    "a small business owner handling daily operations",
    "a project manager coordinating a remote team",
    "an executive assistant preparing for a board meeting",
    "a software engineer debugging a production issue",
    "a marketing manager running a campaign",
    "a data analyst preparing a weekly report",
    "a customer support agent resolving tickets",
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
        """Return target turn count for a given chain length (PROVE: 2-3 range)."""
        base = chain_length + 1  # query_turn + N tool_calls + final_answer
        jitter = rng.randint(-1, 1)
        return max(2, min(base + jitter, chain_length + 2))

    @staticmethod
    def should_continue(turn: int, target: int, last_action_success: bool, tool_calls_done: int) -> bool:
        """Decide whether the conversation should continue.

        Returns False if turn limit reached."""
        if turn >= target:
            return False
        return True


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
        to increase query diversity. chain_seed constrains tools to a dependency chain.
        """
        difficulty_desc = DIFFICULTY_DESCRIPTIONS.get(
            difficulty, DIFFICULTY_DESCRIPTIONS["complete"]
        )
        tools_text = _format_tools(tool_schemas, strip_enums=self._strip_enums)
        state_text = _format_state_compact(grounded_state, max_entities=20)

        # Persona & date context
        persona_block = ""
        if persona:
            persona_block = f"\n## Persona\nYou are generating a query from the perspective of: {persona}.\n"
        date_block = ""
        if reference_date:
            date_block = f"\n## Reference Date\nToday is {reference_date}. Use relative dates when appropriate.\n"

        # Chain constraint
        chain_block = ""
        if chain_seed and len(chain_seed) >= 2:
            chain_block = (
                f"\n## Constraint\nYour query must require these tools IN ORDER:\n"
                f"{' → '.join(chain_seed)}\n"
                f"The task should naturally flow through this tool chain.\n"
            )

        system = (
            "You are generating training data for AI tool-use agents. "
            "Your job is to write a realistic user query that references SPECIFIC entity IDs."
        )
        user = f"""## Domain
{self.domain_desc}

## Available Tools
{tools_text}
{persona_block}{date_block}{dep_hints}{chain_block}
## Current State (Real Entities — use these exact IDs and values)
{state_text}

## Task
Write ONE natural language user query ({difficulty} difficulty). {difficulty_desc}

CRITICAL: Your query MUST include the EXACT entity IDs from the Current State
(e.g., "update event evt_003", "cancel order ord_042", "transfer $500 from acc_01 to acc_02").
Without entity IDs, the agent cannot act. Extract the IDs from the state above.

The query should sound like a real person asking for help.

## Output Format
{{"user_query": "<the query>"}}

Output ONLY the JSON, nothing else:
"""
        for attempt in range(3):
            try:
                raw = self.client.generate_chat(
                    [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
                    temperature=0.7 + 0.1 * attempt,
                )
                data = _extract_json(raw)
                query = data.get("user_query", "")
                if query:
                    return query
            except Exception as e:
                logger.debug(
                    f"generate_query attempt {attempt + 1}/3 failed for "
                    f"{self.domain}: {type(e).__name__}: {e}"
                )
        raise RuntimeError(f"Failed to generate query for {self.domain}")

    # ── Step 2-N: decide next action (LLM-in-the-loop) ──

    def decide_action(
        self,
        tool_schemas: list[dict[str, Any]],
        user_query: str,
        execution_history: list[dict[str, Any]],
        attempt: int = 0,
        dep_hints: str = "",
        difficulty: str = "complete",
    ) -> ActionPlan:
        """LLM decides the next action given full context.

        Called at every turn.  The LLM sees the complete execution history
        and current state, so its decisions are grounded in real values.

        For 'missing' difficulty tasks, ask_clarification is expected on the
        first turn (the query deliberately omits a parameter), so the
        first-turn enforcement is relaxed.
        """
        tools_text = _format_tools(tool_schemas, strip_enums=self._strip_enums)
        history_text = _format_history(execution_history)

        # First-turn guidance: prevent LLM from answering without tools.
        # Exception: 'missing' difficulty tasks omit a parameter on purpose,
        # so ask_clarification is the correct first action.
        # For complete/minimal difficulty, also block ask_clarification —
        # the query already contains enough information (or can be resolved
        # via list/search/get lookups). LLMs over-aligned for safety (esp.
        # on financial / sensitive domains like banking, payments) tend to
        # ask clarifications unnecessarily, producing zero oracle calls.
        if not execution_history:
            if difficulty == "missing":
                first_turn_hint = (
                    "\nIMPORTANT: This task has a MISSING parameter. "
                    "You may need to ask_clarification before calling a tool.\n"
                )
                default_action = "ask_clarification"
                blocked_first = ("final_answer", "report_error")
            else:
                first_turn_hint = (
                    "\nIMPORTANT: This is your FIRST turn. You MUST use a tool to complete the task.\n"
                    "Do NOT use final_answer, report_error, or ask_clarification before calling any tools.\n"
                    "The query already contains the entity IDs you need (or you can look them up with list/search/get tools).\n"
                    "If you are unsure which tool to call, default to a read-only lookup (list_*, get_*, search_*).\n"
                )
                default_action = "tool_call"
                blocked_first = ("final_answer", "report_error", "ask_clarification")
        else:
            first_turn_hint = ""
            default_action = "final_answer"
            blocked_first = ()

        system = (
            "You are controlling tools to complete a user task. "
            "For each turn, decide ONE action. Output EXACTLY one JSON object."
        )
        user = f"""## Domain
{self.domain_desc}

## Available Tools
{tools_text}

{dep_hints}

## User Task
{user_query}

## Execution History (what has happened so far)
{history_text}

## Your Turn
Decide the NEXT action. Output ONE JSON object:

- To call a tool:
  {{"action": "tool_call", "tool_name": "<tool>", "arguments": {{"<param>": <value>}}}}

- To give the final answer:
  {{"action": "final_answer", "text": "<answer>"}}

- To report an error (task impossible with current tools):
  {{"action": "report_error", "reason": "<why>"}}

- To ask the user for missing information:
  {{"action": "ask_clarification", "question": "<what you need>"}}
{first_turn_hint}
Output ONLY the JSON, nothing else:
"""
        for _retry in range(3):
            try:
                raw = self.client.generate_chat(
                    [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
                    temperature=0.7 + 0.1 * attempt,
                )
                data = _extract_json(raw)
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
                else:
                    tool_name = ""

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
                })
                entity = final_val[nk]
                if isinstance(entity, dict):
                    for ek in ("status", "stage", "type", "state"):
                        if ek in entity:
                            criteria.append({
                                "type": "state_equals", "server": domain,
                                "path": f"{key}.{nk}.{ek}", "value": entity[ek],
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
                        if fk in ie and ie[fk] != fe[fk]:
                            criteria.append({
                                "type": "state_equals", "server": domain,
                                "path": f"{key}.{ck}.{fk}", "value": fe[fk],
                            })

    # Domain-specific semantic criteria
    tool_names = [c.tool_name for c in oracle_calls]
    criteria.extend(_domain_criteria(tool_names, initial_state, final_state, domain))

    # Fallback: at minimum verify the domain state exists
    if not criteria:
        criteria.append({"type": "state_exists", "server": domain, "path": ""})

    return criteria


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
) -> tuple[bool, float, int, int]:
    """Replay oracle trace against a fresh session to verify it's reproducible.

    PROVE §3.2 Step 5: counts only schema-level and execution errors (not
    empty-result responses). Discards conversations with error rate > 30%.

    Returns:
        (passed, error_rate, num_errors, num_calls)
        - passed: True if error_rate <= 0.30
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
            # Skip clarification calls — they are not real tool executions
            if call.action == "clarification":
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

        error_rate = num_errors / num_calls if num_calls > 0 else 0.0
        passed = error_rate <= 0.30

        return passed, error_rate, num_errors, num_calls
    finally:
        manager.close_session(session.session_id)


# ═══════════════════════════════════════════════════════════════════════
# Sensitive parameter provenance check (PROVE §3.2 Step 5)
# ═══════════════════════════════════════════════════════════════════════

# Parameter names indicative of sensitive data (PROVE: passwords, tokens, etc.)
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
                # Extract key identity fields
                id_fields: list[str] = []
                for fk in ("name", "title", "subject", "status", "type", "balance", "amount"):
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
