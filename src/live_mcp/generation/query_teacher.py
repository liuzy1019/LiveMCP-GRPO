"""Query-Teacher contract for one fixed dependency chain and live state."""

from __future__ import annotations

import random
import re
from typing import Any

from loguru import logger

from src.live_mcp.errors import CandidateGenerationError
from src.live_mcp.planner_format import format_state_compact as _format_state_compact
from src.live_mcp.generation.teacher_contracts import (
    DIFFICULTY_DESCRIPTIONS,
    GeneratedQuery,
    _chain_goal_phrase,
    _target_tool_requirement,
)
from src.utils import extract_json as _extract_json


class QueryGenerationError(CandidateGenerationError):
    """Structured failure of one fixed chain/state query contract."""

    def __init__(
        self,
        message: str,
        *,
        reason: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message,
            stage="query_generation",
            reason=reason,
            details=details,
        )


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
    ) -> GeneratedQuery:
        """LLM generates a natural-language user query grounded in live state.

        Persona and reference_date increase query diversity. chain_seed guides
        the query toward a realistic dependency chain without making that
        chain the only valid tool trajectory.

        chain_context is a compact chain-aligned view extracted from live
        state. PROVE baseline includes real entity IDs; the local trainable
        profile can replace opaque IDs with natural selectors.
        """
        if difficulty not in DIFFICULTY_DESCRIPTIONS:
            raise ValueError(
                f"unknown query difficulty {difficulty!r}; expected one of "
                f"{sorted(DIFFICULTY_DESCRIPTIONS)}"
            )
        difficulty_desc = DIFFICULTY_DESCRIPTIONS[difficulty]
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
                "If you reference an existing entity, copy an exact identifier or natural "
                "selector shown in Grounded Live State. Never invent a value that is not "
                "shown. Do not substitute an existing ID for an entity that an "
                "earlier capability in the dependency chain will create. If an earlier discovery "
                "capability intentionally hides IDs, use enough exact shown selector facts to "
                "identify one candidate uniquely. A label such as 'savings', 'the invoice', or "
                "'that email' is not complete when multiple shown candidates match; choose a "
                "unique shown selector or return UNSAT."
            )
        else:
            grounding_line = (
                "You forgot one key detail. Use only values shown in Grounded Live State, "
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
        causal_relation_block = ""
        if chain_seed:
            final_tool = chain_seed[-1]
            goal_phrase = _chain_goal_phrase(tool_schemas, final_tool)
            if goal_phrase:
                chain_requirements = "\n".join(
                    f"- {_target_tool_requirement(tool_schemas, tool_name)}"
                    for tool_name in chain_seed
                )
                if difficulty == "complete":
                    state_change_parameter_contract = (
                        "Every required field that controls a state change MUST "
                        "have a concrete user-authorized value in the message or "
                        "an unambiguous value in Grounded Live State; never leave "
                        "such a field for the assistant to invent (for example an "
                        "amount, recipient, status, path, date, quantity, or "
                        "message body)."
                    )
                elif difficulty == "missing":
                    state_change_parameter_contract = (
                        "Exactly ONE critical user-supplied field may be absent. "
                        "Every other user-known field that controls a state change "
                        "must have a concrete value in the message or an unambiguous "
                        "value in Grounded Live State. The missing field must be "
                        "resolved by clarification, never invented by the assistant."
                    )
                else:
                    state_change_parameter_contract = (
                        "Do NOT require concrete entity selectors or parameters in "
                        "this minimal message. It must still authorize every requested "
                        "user-visible state change, while missing values are left for "
                        "clarification and are never invented by the assistant."
                    )
                chain_goal_block = (
                    "\n## Complete Task Goal (internal synthesis guide)\n"
                    f"The grounded dependency chain is: {chain_seed}.\n"
                    f"Capabilities in execution order:\n{chain_requirements}\n"
                    f"The message MUST clearly request the final outcome: {goal_phrase}. "
                    "If you cannot "
                    "write a natural request for this target, return user_query as "
                    "'UNSAT' instead of switching to another task. "
                    "The request must preserve the dependency: do not give the user "
                    "an ID or value that an earlier capability is supposed to discover "
                    "or produce. Each earlier capability must remain a plausible internal "
                    "prerequisite for the final outcome. If that cannot be expressed as "
                    "one natural user goal, return UNSAT. "
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
                    f"{state_change_parameter_contract}\n"
                )
            relation_lines: list[str] = []
            for relation in (chain_context or {}).get(
                "dependency_relations", []
            ):
                if not isinstance(relation, dict):
                    continue
                source = str(relation.get("source_capability") or "")
                target = str(relation.get("target_capability") or "")
                kind = str(relation.get("relation") or "")
                for binding in relation.get("value_bindings", []):
                    if not isinstance(binding, dict):
                        continue
                    relation_lines.append(
                        f"- {source} output field "
                        f"{binding.get('source_output_field')} supplies "
                        f"{target} argument {binding.get('target_argument')}."
                    )
                for binding in relation.get("state_bindings", []):
                    if not isinstance(binding, dict):
                        continue
                    relation_lines.append(
                        f"- {source} field {binding.get('source_field')} must "
                        f"establish {binding.get('state_slot')} for the same "
                        f"entity used as {target} argument "
                        f"{binding.get('target_argument')}."
                    )
                if kind == "implicit" and not relation.get("state_bindings"):
                    relation_lines.append(
                        f"- {source} must establish the live state required "
                        f"by {target} on the same resource."
                    )
            if relation_lines:
                causal_relation_block = (
                    "\n## Verified Causal Relations (internal synthesis guide)\n"
                    "Use these relations to keep one concrete entity lineage. "
                    "Do not repeat them as workflow steps in the user message.\n"
                    + "\n".join(relation_lines)
                    + "\n"
                )

        # ── Grounded-entity constraint ──
        # Chain-aligned entity context: only these IDs are known to exist
        # in the live state.  The teacher MUST reference only these, never
        # invent IDs.  Without this constraint, the LLM hallucinates entity
        # IDs that don't exist in any seed, causing 100% replay failure.
        anti_halluc_block = ""
        if chain_context and chain_context.get("query_grounding_summaries"):
            hidden_types = chain_context.get("opaque_id_hidden_types", [])
            if hidden_types:
                anti_halluc_block = (
                    f"\n## Chain-Aligned Grounding Rule\n"
                    "Use only candidates already shown once in Grounded Live "
                    "State above. "
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
                    f"\n## Chain-Aligned Grounding Rule\n"
                    "Only the entities already shown once in Grounded Live State "
                    "above exist for this task. "
                    f"⚠️ CRITICAL: You MUST ONLY reference entity IDs from this list "
                    f"or from Grounded Live State above. Do NOT invent, modify, or guess IDs. "
                    f"If an ID is 'evt_001', write 'evt_001' — not 'evt_005' or 'evt_0001'.\n"
                )
        elif difficulty == "complete":
            # Even without chain_context, enforce anti-hallucination on
            # the full state for complete-difficulty tasks.
            anti_halluc_block = (
                "\n## Anti-Hallucination Rule\n"
                "Reference only identifiers or selectors that appear in Grounded Live "
                "State above. Do not invent or infer IDs. A new value is allowed only when "
                "an earlier capability in the sampled chain creates it.\n"
            )

        user = f"""## Persona
{persona if persona else 'A normal user messaging their AI assistant.'}
{date_block}
## What this assistant can help with
{self.domain_desc}
{dep_hints}
## Grounded Live State
{state_text}
{chain_goal_block}
{causal_relation_block}
{anti_halluc_block}
## Your task
Type ONE message to your assistant. Difficulty: {difficulty}. {difficulty_desc}

{grounding_line}{priority_note}
Remember: state your GOAL, not the steps. One message, 1-2 sentences max.

Return only:
{{"user_query": "<the message or UNSAT>"}}
"""
        failure_reason = "invalid_query_contract"
        attempt_diagnostics: list[dict[str, Any]] = []
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
                    attempt_diagnostics.append({
                        "attempt": attempt + 1,
                        "parsed_type": type(data).__name__,
                    })
                    continue
                query = data.get("user_query", "")
                target_capability = chain_seed[-1] if chain_seed else ""
                attempt_diagnostics.append({
                    "attempt": attempt + 1,
                    "user_query": str(query),
                    "target_capability": target_capability,
                })
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
                natural_selector_profile = bool(
                    getattr(self.prompt_profile, "natural_selector", False)
                    or self.prompt_profile == "local_trainable_v1"
                )
                hidden_types = ({
                    str(value)
                    for value in chain_context.get(
                        "opaque_id_hidden_types", []
                    )
                } if chain_context and natural_selector_profile else set())
                explicit_hidden_ids = {
                    str(value)
                    for value in (chain_context or {}).get(
                        "opaque_id_hidden_ids", []
                    )
                    if str(value)
                }
                hidden_ids = explicit_hidden_ids or {
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
                if query:
                    return GeneratedQuery(
                        user_query=str(query),
                        target_capability=target_capability,
                        attempts=attempt + 1,
                    )
            except Exception as e:
                attempt_diagnostics.append({
                    "attempt": attempt + 1,
                    "exception": f"{type(e).__name__}: {e}",
                })
                logger.debug(
                    f"generate_query attempt {attempt + 1}/3 failed for "
                    f"{self.domain}: {type(e).__name__}: {e}"
                )
        raise QueryGenerationError(
            f"Failed to generate query for {self.domain}",
            reason=failure_reason,
            details={
                "source_chain_seed": list(chain_seed or []),
                "expected_target_capability": (
                    chain_seed[-1] if chain_seed else ""
                ),
                "attempt_diagnostics": attempt_diagnostics,
            },
        )


__all__ = ["QueryGenerationError", "QueryTeacherMixin"]
