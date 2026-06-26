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

    def __init__(self, client: object, domain: str):
        self.client = client
        self.domain = domain
        self.domain_desc = DOMAIN_DESCRIPTIONS.get(domain, "")
        self._strip_enums = random.Random().random() < 0.30  # per-task, aligns with PROVE

    # ── Step 1: generate user query ──

    def generate_query(
        self,
        tool_schemas: list[dict[str, Any]],
        grounded_state: dict[str, Any],
        difficulty: str,
        rng: random.Random,
        dep_hints: str = "",
    ) -> str:
        """LLM generates a natural-language user query grounded in live state."""
        difficulty_desc = DIFFICULTY_DESCRIPTIONS.get(
            difficulty, DIFFICULTY_DESCRIPTIONS["complete"]
        )
        tools_text = _format_tools(tool_schemas, strip_enums=self._strip_enums)
        state_text = _format_state(grounded_state)

        system = (
            "You are generating training data for AI tool-use agents. "
            "Your job is to write a realistic user query."
        )
        user = f"""## Domain
{self.domain_desc}

## Available Tools
{tools_text}

{dep_hints}

## Current State (Real Entities — use these exact IDs and values)
{state_text}

## Task
Write ONE natural language user query ({difficulty} difficulty). {difficulty_desc}

The query should sound like a real person asking for help.
Use EXACT IDs and values from the Current State above.

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
            except Exception:
                pass
        raise RuntimeError(f"Failed to generate query for {self.domain}")

    # ── Step 2-N: decide next action (LLM-in-the-loop) ──

    def decide_action(
        self,
        tool_schemas: list[dict[str, Any]],
        user_query: str,
        execution_history: list[dict[str, Any]],
        attempt: int = 0,
        dep_hints: str = "",
    ) -> ActionPlan:
        """LLM decides the next action given full context.

        Called at every turn.  The LLM sees the complete execution history
        and current state, so its decisions are grounded in real values.
        """
        tools_text = _format_tools(tool_schemas, strip_enums=self._strip_enums)
        history_text = _format_history(execution_history)

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
                return ActionPlan(
                    action=data.get("action", "final_answer"),
                    tool_name=data.get("tool_name", ""),
                    arguments=data.get("arguments", {}),
                    text=data.get("text", data.get("reason", data.get("question", ""))),
                )
            except Exception:
                pass
        raise RuntimeError(
            f"decide_action failed after 3 attempts for {self.domain} — "
            f"LLM could not produce a valid decision"
        )


# ═══════════════════════════════════════════════════════════════════════
# Execution perturbations (PROVE-style robustness)
# ═══════════════════════════════════════════════════════════════════════

def apply_perturbation(
    observation: dict[str, Any] | str | None,
    domain: str,
    rng: random.Random,
) -> dict[str, Any] | str | None:
    """Optionally perturb an execution result to simulate real-world noise.

    PROVE-style perturbations (15-30% probability per call):
      - intermittent_api_error:   return error message → oracle retries
      - paginated_response:       wrap partial results with cursor
      - incomplete_intermediate:  return summary instead of full details
    """
    roll = rng.random()
    if roll < 0.08:
        return {"error": "Internal Server Error", "retry": True}
    if roll < 0.16 and isinstance(observation, dict):
        items = observation.get("items", observation.get("results", []))
        if isinstance(items, list) and len(items) > 1:
            mid = max(1, len(items) // 2)
            return {**observation, "items": items[:mid], "next_cursor": "page_2"}
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
    criteria.extend(_domain_criteria(tool_names, final_state, domain))

    # Fallback: at minimum verify the domain state exists
    if not criteria:
        criteria.append({"type": "state_exists", "server": domain, "path": ""})

    return criteria


def _domain_criteria(
    tool_names: list[str],
    final_state: dict[str, Any],
    domain: str,
) -> list[dict[str, Any]]:
    """Domain-specific success criteria from tool semantics."""
    criteria: list[dict[str, Any]] = []

    if "transfer" in tool_names:
        for acc_id, acc in final_state.get("accounts", {}).items():
            criteria.append({
                "type": "state_equals", "server": domain,
                "path": f"accounts.{acc_id}.balance",
                "value": acc.get("balance", 0),
            })
    if "add_to_cart" in tool_names and "cart" in final_state:
        criteria.append({"type": "cart_not_empty", "server": domain})
    if "create_order" in tool_names:
        for oid, order in final_state.get("orders", {}).items():
            criteria.append({
                "type": "state_equals", "server": domain,
                "path": f"orders.{oid}.status",
                "value": order.get("status", "confirmed"),
            })
    if any(t in tool_names for t in ("create_invoice", "pay_invoice")):
        for inv_id, inv in final_state.get("invoices", {}).items():
            if "status" in inv:
                criteria.append({
                    "type": "state_equals", "server": domain,
                    "path": f"invoices.{inv_id}.status", "value": inv["status"],
                })
    if any(t in tool_names for t in ("update_lead", "convert_lead", "create_deal")):
        for lead_id, lead in final_state.get("leads", {}).items():
            criteria.append({
                "type": "state_equals", "server": domain,
                "path": f"leads.{lead_id}.status",
                "value": lead.get("status", "new"),
            })
    if any(t in tool_names for t in ("create_issue", "update_issue", "transition_issue")):
        for iss_id, issue in final_state.get("issues", {}).items():
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
        for path in final_state.get("fs", {}):
            criteria.append({"type": "file_exists", "server": domain, "path": path})
    if "send_message" in tool_names:
        for ch_id, ch in final_state.get("channels", {}).items():
            criteria.append({
                "type": "state_equals", "server": domain,
                "path": f"channels.{ch_id}.messages_count",
                "value": len(ch.get("messages", [])),
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
) -> bool:
    """Replay oracle trace against a fresh session to verify it's reproducible.

    Returns True if ALL calls execute successfully in the replay session.
    """
    session = manager.create_session(seed=seed)
    try:
        manager.discover_tools(session.session_id)
        for idx, call in enumerate(oracle_calls):
            from src.live_mcp.types import ToolCall
            result = executor.execute(
                session.session_id,
                ToolCall(call.tool_name, dict(call.arguments), call_id=f"replay_{idx}"),
            )
            if not result.success:
                return False
        return True
    finally:
        manager.close_session(session.session_id)


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def _extract_json(text: str) -> dict[str, Any]:
    """Extract JSON object from LLM output, handling markdown fences and think tags."""
    import re
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    if "<think>" in text:
        text = re.sub(r"<think>[\s\S]+", "", text)
    text = text.strip()
    try:
        return _json.loads(text)
    except _json.JSONDecodeError:
        pass
    m = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if m:
        try:
            return _json.loads(m.group(1).strip())
        except _json.JSONDecodeError:
            pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return _json.loads(m.group(0))
        except _json.JSONDecodeError:
            pass
    raise ValueError(f"Could not extract JSON from: {text[:200]}")


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


def _format_state(state: dict[str, Any]) -> str:
    """Format grounded state compactly."""
    try:
        return _json.dumps(state, indent=2, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(state)[:3000]


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
