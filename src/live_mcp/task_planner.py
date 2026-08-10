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
from pathlib import Path
from typing import Any, TYPE_CHECKING

from loguru import logger

from src.live_mcp.protocol.observation import (
    DEFAULT_TEACHER_OBSERVATION_CHARS,
)
from src.live_mcp.planner_format import (
    format_conversation_context as _format_conversation_context,
    format_state_compact as _format_state_compact,
    format_tools as _format_tools,
)
from src.live_mcp.generation.action_teacher import ActionTeacherMixin
from src.live_mcp.generation.query_teacher import QueryTeacherMixin
from src.live_mcp.generation.teacher_contracts import (
    ContinuationPolicy,
    DOMAIN_DESCRIPTIONS,
    GeneratedQuery,
    _PERSONA_TEMPLATES,
    _chain_goal_phrase,
    _target_tool_requirement,
    reference_date_for_seed,
    reference_datetime_for_seed,
)


from src.live_mcp.replay.criteria import (
    derive_progress_predicates,
    derive_success_criteria,
)
from src.utils import extract_json as _extract_json

if TYPE_CHECKING:
    from src.live_mcp.llm_client import LLMClient


class TaskPlanner(QueryTeacherMixin, ActionTeacherMixin):
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
        prompt_profile: str = "paper_generation_baseline_v1",
    ):
        self.client = client
        self.domain = domain
        self.seed = int(seed)
        self.max_observation_chars = max(256, int(max_observation_chars))
        self.prompt_profile = prompt_profile
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
        **_kwargs: Any,
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
        **_kwargs: Any,
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
