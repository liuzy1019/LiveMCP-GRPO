"""Prompt contracts and value objects shared by Teacher stages."""

from __future__ import annotations

import datetime as _datetime
import random
import re
from dataclasses import dataclass, field
from typing import Any

from src.live_mcp.contracts.factory import build_contract_registry
from src.live_mcp.dependency_value_flow import (
    _dependency_value_key,
    _field_values,
)
from src.live_mcp.domain_contracts.reference_visibility import (
    DOMAIN_OPAQUE_ENTITY_TYPES,
    is_public_entity_reference,
    is_sampler_private_handle,
)


VALID_TERMINALS: tuple[str, ...] = (
    "final_answer", "report_error", "ask_clarification",
)

_SAMPLER_PRIVATE_ID_RE = re.compile(
    r"(?<![A-Za-z0-9_])[A-Za-z][A-Za-z0-9]*_s\d+_\d+"
    r"(?![A-Za-z0-9_])"
)


@dataclass(frozen=True)
class UserVisiblePrivateIdExposure:
    round_idx: int
    surface: str
    text: str
    leaked_ids: tuple[str, ...]


@dataclass(frozen=True)
class UserVisibleToolNameExposure:
    round_idx: int
    text: str
    exposed_tool_names: tuple[str, ...]


def sampler_private_ids(text: str) -> tuple[str, ...]:
    """Return deterministic seeded backend IDs exposed in public text."""
    return tuple(sorted(set(_SAMPLER_PRIVATE_ID_RE.findall(str(text or "")))))


def user_visible_private_id_exposure(
    conversation_queries: list[str],
    oracle_calls_per_round: list[list[Any]],
    *,
    private_entity_ids: set[str] | None = None,
    public_entity_ids: set[str] | None = None,
) -> UserVisiblePrivateIdExposure | None:
    """Find the first private entity ID on a user-visible surface.

    Exact IDs proven by typed trace facts cover runtime-created entities; the
    seeded-ID pattern remains a fail-safe for a leaked sampler reference that
    did not reach a successful call. Tool arguments and observations are
    deliberately excluded because the runtime needs opaque IDs internally.
    """
    known_private_ids = {
        str(value) for value in (private_entity_ids or set()) if str(value)
    }
    known_public_ids = {
        str(value) for value in (public_entity_ids or set()) if str(value)
    }

    def exposed_ids(text: str) -> tuple[str, ...]:
        exact = {
            entity_id
            for entity_id in known_private_ids
            if re.search(
                rf"(?<![A-Za-z0-9_]){re.escape(entity_id)}"
                rf"(?![A-Za-z0-9_])",
                text,
            )
        }
        # Sampler provenance is an unconditional private-reference fact.
        # A typed public declaration may expose a separate business identifier,
        # but can never relabel a seeded backend handle as public.
        seeded_backend_ids = set(sampler_private_ids(text))
        derived_private_aliases: set[str] = set()
        for entity_id in known_private_ids:
            if not _SAMPLER_PRIVATE_ID_RE.fullmatch(entity_id):
                continue
            suffix = entity_id.rsplit("_", 1)[-1]
            for marker in ("_", "..."):
                alias = f"{marker}{suffix}"
                if re.search(
                    rf"(?<![A-Za-z0-9_]){re.escape(alias)}"
                    rf"(?![A-Za-z0-9_])",
                    text,
                ):
                    derived_private_aliases.add(alias)
        return tuple(sorted(
            exact | seeded_backend_ids | derived_private_aliases,
        ))

    round_count = max(len(conversation_queries), len(oracle_calls_per_round))
    for round_idx in range(round_count):
        if round_idx < len(conversation_queries):
            query = str(conversation_queries[round_idx] or "")
            leaked = exposed_ids(query)
            if leaked:
                return UserVisiblePrivateIdExposure(
                    round_idx, "user_query", query, leaked,
                )
        if round_idx >= len(oracle_calls_per_round):
            continue
        for call in oracle_calls_per_round[round_idx]:
            action = str(
                call.get("action", "tool_call")
                if isinstance(call, dict)
                else getattr(call, "action", "tool_call")
            )
            if action == "tool_call":
                continue
            arguments = (
                call.get("arguments", {})
                if isinstance(call, dict)
                else getattr(call, "arguments", {})
            ) or {}
            terminal_text = str(
                arguments.get("text")
                or arguments.get("question")
                or arguments.get("reason")
                or ""
            )
            leaked = exposed_ids(terminal_text)
            if leaked:
                return UserVisiblePrivateIdExposure(
                    round_idx, "assistant_terminal", terminal_text, leaked,
                )
    return None


def user_visible_terminal_tool_name_exposure(
    oracle_calls_per_round: list[list[Any]],
    *,
    tool_names: set[str],
    hidden_tool_names: set[str] | None = None,
) -> UserVisibleToolNameExposure | None:
    """Find an internal capability identifier in assistant terminal text."""
    hidden_tool_names = hidden_tool_names or set()
    for round_idx, calls in enumerate(oracle_calls_per_round):
        for call in calls:
            action = str(
                call.get("action", "tool_call")
                if isinstance(call, dict)
                else getattr(call, "action", "tool_call")
            )
            if action == "tool_call":
                continue
            arguments = (
                call.get("arguments", {})
                if isinstance(call, dict)
                else getattr(call, "arguments", {})
            ) or {}
            terminal_text = str(
                arguments.get("text")
                or arguments.get("question")
                or arguments.get("reason")
                or ""
            )
            exposed: list[str] = []
            for tool_name in sorted(tool_names):
                bounded = (
                    rf"(?<![A-Za-z0-9_]){re.escape(tool_name)}"
                    rf"(?![A-Za-z0-9_])"
                )
                if "_" in tool_name and re.search(
                    bounded, terminal_text, re.IGNORECASE,
                ):
                    exposed.append(tool_name)
                    continue
                if tool_name not in hidden_tool_names:
                    continue
                code_like = (
                    rf"(?:`{re.escape(tool_name)}`|"
                    rf"\({re.escape(tool_name)}\)|"
                    rf"\b{re.escape(tool_name)}\s+"
                    rf"(?:tool|command|function)\b)"
                )
                if re.search(code_like, terminal_text, re.IGNORECASE):
                    exposed.append(tool_name)
            if exposed:
                return UserVisibleToolNameExposure(
                    round_idx=round_idx,
                    text=terminal_text,
                    exposed_tool_names=tuple(exposed),
                )
    return None


def _entity_values(value: Any) -> list[Any]:
    if isinstance(value, list):
        return [item for child in value for item in _entity_values(child)]
    if isinstance(value, (dict, tuple, set)):
        return []
    return [] if value is None else [value]


def round_entity_occurrences(
    *,
    domain: str,
    calls: list[Any],
    observations: list[Any],
    server_tools: list[dict[str, Any]],
) -> tuple[
    dict[tuple[str, str], dict[str, Any]],
    dict[tuple[str, str], dict[str, Any]],
]:
    """Return typed entity facts and typed input targets for one round."""
    registry = build_contract_registry({domain: server_tools})
    facts: dict[tuple[str, str], dict[str, Any]] = {}
    inputs: dict[tuple[str, str], dict[str, Any]] = {}
    for call_index, call in enumerate(calls):
        action = (
            call.get("action", "tool_call")
            if isinstance(call, dict)
            else getattr(call, "action", "tool_call")
        )
        if action != "tool_call":
            continue
        tool_name = str(
            call.get("tool_name")
            if isinstance(call, dict)
            else getattr(call, "tool_name", "")
        )
        arguments = (
            call.get("arguments", {})
            if isinstance(call, dict)
            else getattr(call, "arguments", {})
        ) or {}
        try:
            contract = registry.get(domain, tool_name)
        except KeyError:
            continue
        for binding in contract.input_entities:
            if binding.name not in arguments:
                continue
            for value in _entity_values(arguments[binding.name]):
                key = (binding.entity_type, _dependency_value_key(value))
                occurrence = {
                    "entity_type": binding.entity_type,
                    "value": value,
                    "capability": tool_name,
                    "field": binding.name,
                    "surface": "argument",
                    "call_index": call_index,
                }
                facts.setdefault(key, occurrence)
                inputs.setdefault(key, occurrence)
        # Domain-neutral tools may accept a polymorphic entity pair.
        dynamic_entity_type = arguments.get("entity_type")
        dynamic_entity_id = arguments.get("entity_id")
        if (
            isinstance(dynamic_entity_type, str)
            and dynamic_entity_type.strip()
            and dynamic_entity_id is not None
        ):
            for value in _entity_values(dynamic_entity_id):
                entity_type = dynamic_entity_type.strip()
                key = (entity_type, _dependency_value_key(value))
                occurrence = {
                    "entity_type": entity_type,
                    "value": value,
                    "capability": tool_name,
                    "field": "entity_id",
                    "surface": "argument",
                    "call_index": call_index,
                }
                facts.setdefault(key, occurrence)
                inputs.setdefault(key, occurrence)
        observation = (
            observations[call_index] if call_index < len(observations) else {}
        )
        for binding in contract.output_entities:
            for raw_value in _field_values(observation, binding.name):
                for value in _entity_values(raw_value):
                    key = (binding.entity_type, _dependency_value_key(value))
                    facts.setdefault(key, {
                        "entity_type": binding.entity_type,
                        "value": value,
                        "capability": tool_name,
                        "field": binding.name,
                        "surface": "observation",
                        "call_index": call_index,
                    })
    return facts, inputs


def contract_entity_types(
    domain: str,
    server_tools: list[dict[str, Any]],
) -> set[str]:
    """Return every canonical entity type covered by the tool contracts."""
    if not server_tools:
        return set()
    try:
        registry = build_contract_registry({domain: server_tools})
    except (KeyError, TypeError, ValueError):
        # Persisted diagnostic fixtures and legacy artifacts may carry only a
        # schema projection. The sampler-pattern fail-safe remains independent
        # of typed extraction, so degraded metadata must not disable it.
        return set()
    return {
        binding.entity_type
        for contract in registry.domain(domain)
        for binding in (*contract.input_entities, *contract.output_entities)
        if binding.entity_type
    }


def reference_entity_types(
    domain: str,
    server_tools: list[dict[str, Any]],
) -> set[str]:
    """Return the canonical entity types covered by local visibility policy."""
    entity_types = contract_entity_types(domain, server_tools)
    entity_types.update(DOMAIN_OPAQUE_ENTITY_TYPES.get(domain, frozenset()))
    return entity_types


def reference_visibility_from_execution_history(
    *,
    domain: str,
    execution_history: list[dict[str, Any]],
    server_tools: list[dict[str, Any]],
) -> tuple[set[str], set[str]]:
    """Derive exact private/public references from successful typed events.

    A Teacher-visible schema set may include cross-domain distractors.  Typed
    contracts and reference policy must therefore follow each execution
    event's recorded owner, rather than the candidate's primary domain.
    """
    successful = [
        event for event in execution_history
        if isinstance(event, dict)
        and event.get("success") is True
        and str(event.get("tool_name") or "")
    ]
    schemas_by_owner: dict[str, list[dict[str, Any]]] = {}
    for schema in server_tools:
        owner = str(schema.get("_server_name") or domain)
        schemas_by_owner.setdefault(owner, []).append(schema)
    events_by_owner: dict[str, list[dict[str, Any]]] = {}
    for event in successful:
        owner = str(event.get("server_name") or domain)
        events_by_owner.setdefault(owner, []).append(event)

    private_ids: set[str] = set()
    public_ids: set[str] = set()
    for owner, events in sorted(events_by_owner.items()):
        owned_schemas = schemas_by_owner.get(owner, [])
        if not owned_schemas:
            raise ValueError(
                f"Missing visible tool schemas for execution owner {owner}"
            )
        calls = [{
            "action": "tool_call",
            "tool_name": str(event.get("tool_name") or ""),
            "arguments": dict(event.get("arguments") or {}),
        } for event in events]
        observations = [event.get("observation") or {} for event in events]
        owner_private, owner_public = typed_entity_reference_visibility_from_rounds(
            domain=owner,
            calls_per_round=[calls],
            observations_per_round=[observations],
            server_tools=owned_schemas,
            entity_types=reference_entity_types(owner, owned_schemas),
        )
        private_ids.update(owner_private)
        public_ids.update(owner_public)
    return private_ids - public_ids, public_ids


def typed_entity_reference_visibility_from_rounds(
    *,
    domain: str,
    calls_per_round: list[list[Any]],
    observations_per_round: list[list[Any]],
    server_tools: list[dict[str, Any]],
    entity_types: set[str],
) -> tuple[set[str], set[str]]:
    """Return private backend handles and public business references.

    Visibility is decided from the canonical typed binding field, rather than
    treating every identifier belonging to an entity type as equivalent.
    """
    private_ids: set[str] = set()
    public_ids: set[str] = set()
    if not server_tools or not entity_types:
        return private_ids, public_ids
    for round_idx, calls in enumerate(calls_per_round):
        facts, _ = round_entity_occurrences(
            domain=domain,
            calls=calls,
            observations=(
                observations_per_round[round_idx]
                if round_idx < len(observations_per_round)
                else []
            ),
            server_tools=server_tools,
        )
        for item in facts.values():
            value = item.get("value")
            entity_type = str(item.get("entity_type") or "")
            field = str(item.get("field") or "")
            if entity_type not in entity_types or not isinstance(value, str) or not value:
                continue
            if is_sampler_private_handle(value):
                private_ids.add(value)
            elif is_public_entity_reference(
                domain, entity_type, field, value,
            ):
                public_ids.add(value)
            elif field == "id" or field.endswith("_id"):
                private_ids.add(value)
    return private_ids - public_ids, public_ids


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
    attempts: int


def _chain_goal_phrase(
    tool_schemas: list[dict[str, Any]], final_tool: str,
) -> str:
    """Derive a domain-correct outcome hint from the exact visible schema."""
    schema = next(
        (tool for tool in tool_schemas if tool.get("name") == final_tool),
        {},
    )
    description = " ".join(str(schema.get("description") or "").split())
    if description:
        sentence = description.split(". ", 1)[0].rstrip(".")
        return sentence[:1].lower() + sentence[1:]

    name = final_tool.lower()
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
        f"Capability description: "
        f"{description or _chain_goal_phrase(tool_schemas, tool_name)}"
        f"{readonly_note}\n"
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


def reference_date_for_candidate_state(
    generation_seed: int,
    state_seed: int | None,
) -> str:
    """Return the temporal anchor owned by the candidate's Live-State.

    ``generation_seed`` still controls persona and sampling diversity.  When a
    distinct state seed is supplied, however, the state seeder and both Teacher
    prompts must describe the same current date.
    """
    effective_state_seed = generation_seed if state_seed is None else state_seed
    return reference_date_for_seed(int(effective_state_seed))

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
