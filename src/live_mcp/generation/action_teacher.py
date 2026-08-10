"""Action Teacher decisions and recovery policy."""

from __future__ import annotations

import json as _json
from typing import Any

from loguru import logger

from src.live_mcp.generation.teacher_contracts import (
    ActionPlan,
    VALID_TERMINALS,
    _final_answer_requests_user_input,
)
from src.live_mcp.planner_format import (
    format_conversation_context as _format_conversation_context,
    format_history as _format_history,
    format_tools as _format_tools,
)
from src.utils import extract_json as _extract_json


class ActionTeacherMixin:
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
        allowed_missing_mutations: set[str] | None = None,
        allow_direct_answer: bool = False,
        dependency_plan: dict[str, Any] | None = None,
        round_idx: int = 0,
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
        _terminal_collision = tool_names_set & set(VALID_TERMINALS)
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
                    "When you terminate with report_error, the message MUST attribute the "
                    "failure to a missing capability/tool (e.g. 'no tool is available to "
                    "modify cart quantities'). Do NOT substitute a fabricated state reason "
                    "(e.g. 'the item is not in your cart') when the true cause is that the "
                    "required tool is absent from Available Tools.\n"
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
                    "make progress — do NOT produce final_answer or "
                    "ask_clarification. The user request includes concrete "
                    "entity IDs and values where needed. "
                    "Use a read/list/search tool only when you genuinely need "
                    "to discover which entities match the user's description. "
                    "When the user already provides an entity ID or the exact "
                    "selector, use it directly — do not insert a redundant "
                    "detail/get lookup before the first mutating call. "
                    "A search result that returns entity IDs is already "
                    "sufficient grounding for the next tool.\n"
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
            "allowance, and linked IDs. Search and list results already provide valid "
            "entity IDs — use them directly for the next mutation without an intermediate "
            "detail/get call. Add a get/detail call only when the mutation requires a "
            "specific field (status, amount, linked ID, etc.) that the preceding search "
            "or list result does not show. If even that field is unknown after discovery, "
            "ask_clarification rather than guessing.\n"
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
                elif action in VALID_TERMINALS:
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
        **_kwargs: Any,
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
