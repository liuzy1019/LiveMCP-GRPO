"""Query-Teacher contract for one fixed dependency chain and live state."""

from __future__ import annotations

import random
import re
from typing import Any

from loguru import logger

from src.live_mcp.planner_format import format_state_compact as _format_state_compact
from src.live_mcp.generation.teacher_contracts import (
    DIFFICULTY_DESCRIPTIONS,
    GeneratedQuery,
    _chain_goal_phrase,
    _target_tool_requirement,
)
from src.utils import extract_json as _extract_json


class QueryGenerationError(RuntimeError):
    """Structured failure of one fixed chain/state query contract."""

    def __init__(self, message: str, *, reason: str) -> None:
        super().__init__(message)
        self.reason = reason


class QueryTeacherMixin:
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
        **_kwargs: Any,
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
                    "state every user-visible outcome naturally, including each earlier "
                    "state change when it is genuinely part of that same requested "
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
                "above. Do NOT invent or modify existing IDs; copy them exactly as "
                "shown. A new ID or path is allowed only when an earlier capability "
                "in the sampled chain creates it, and its parent/namespace must come "
                "from Current State.\n"
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
{{"user_query": "<the message or UNSAT>", "target_capability": "<sampled chain final capability>", "chain_supported": <true or false>, "mutation_evidence": [{{"capability": "<state-changing chain capability>", "query_span": "<exact words from user_query that authorize it>"}}]}}

mutation_evidence must contain one item for every state-changing capability in
the sampled chain. A dependency edge never authorizes an unstated side effect.
If cd is the final requested capability, it still needs evidence. query_span must
be copied verbatim from user_query and directly authorize the stated goal. Do not
invent an evidence span for an operation the user did not request. Read-only
capabilities must not appear in mutation_evidence.
"""
        tools_by_name = {
            str(tool.get("name") or ""): tool for tool in tool_schemas
        }
        expected_mutations = {
            tool_name
            for tool_name in (chain_seed or [])
            if (tools_by_name.get(tool_name, {}).get("annotations") or {}).get(
                "mutating"
            ) is True
        }
        failure_reason = "invalid_query_contract"
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
                    failure_reason = "goal_unsat"
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
                hidden_types = {
                    str(value)
                    for value in chain_context.get(
                        "opaque_id_hidden_types", []
                    )
                } if chain_context else set()
                hidden_ids = {
                    str(item.get("id") or "")
                    for item in (chain_context or {}).get("entity_ids", [])
                    if isinstance(item, dict)
                    and str(item.get("type") or "") in hidden_types
                    and str(item.get("id") or "")
                }
                leaked_ids = sorted(
                    entity_id
                    for entity_id in hidden_ids
                    if re.search(
                        rf"(?<![A-Za-z0-9_]){re.escape(entity_id)}"
                        rf"(?![A-Za-z0-9_])",
                        str(query),
                    )
                )
                if leaked_ids:
                    logger.debug(
                        f"generate_query attempt {attempt + 1}/3 exposed "
                        f"sampler-private IDs for {self.domain}: {leaked_ids}"
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
                        mutation_evidence=[
                            dict(item)
                            for item in mutation_evidence
                            if isinstance(item, dict)
                        ],
                        dependency_evidence=[],
                        initial_goal="",
                        initial_goal_grounding_basis={},
                        initial_goal_causal_steps=[],
                        initial_goal_planning_attempts=0,
                    )
            except Exception as e:
                logger.debug(
                    f"generate_query attempt {attempt + 1}/3 failed for "
                    f"{self.domain}: {type(e).__name__}: {e}"
                )
        raise QueryGenerationError(
            f"Failed to generate query for {self.domain}",
            reason=failure_reason,
        )


__all__ = ["QueryGenerationError", "QueryTeacherMixin"]
