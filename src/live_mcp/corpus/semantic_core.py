"""Shared deterministic semantic-quarantine evidence helpers."""

from __future__ import annotations

import json

import re

from dataclasses import asdict, dataclass

from typing import Any

from src.live_mcp.generation.teacher_contracts import (
    reference_entity_types,
    _final_answer_requests_user_input,
    typed_entity_reference_visibility_from_rounds,
    user_visible_private_id_exposure,
    user_visible_terminal_tool_name_exposure,
)
from src.live_mcp.generation.robustness import (
    missing_function_has_mutation,
    missing_function_unresolved_failures,
)

_ASSISTANT_ROLE_USER_QUERY_RE = re.compile(
    r"^\s*(?:would\s+you\s+like\s+(?:me\s+)?to|shall\s+i)\b",
    re.IGNORECASE,
)

_NEGATIVE_RECOMMENDATION_RE = re.compile(
    r"\b(?:not\s+(?:a|an)|isn['’]?t\s+(?:a|an)|cannot|can['’]?t|"
    r"does\s+not\s+match|no\s+matching|only\s+returned|"
    r"should\s+not\s+recommend|no\s+other|"
    r"(?:could|can|did)\s*n['’]?t\s+find|"
    r"only\s+(?:one|ones?)\s+(?:is|are\s+)?available)\b",
    re.I,
)

_ORDER_ITEM_COMPARE_RE = re.compile(r"\bcompar(?:e|ison)\b", re.I)

_ORDER_ITEM_DETAIL_RE = re.compile(
    r"\b(?:detail|details|information|info|tell\s+me\s+about)\b",
    re.I,
)

_ORDER_ITEM_REVIEW_RE = re.compile(r"\b(?:review|reviews|rating|ratings)\b", re.I)

_POSITIVE_RECOMMENDATION_RE = re.compile(
    r"\b(?:i\s+recommend|recommend(?:ed|ation)?|suggest(?:ed|ion)?|"
    r"you\s+might\s+like|good\s+(?:choice|option)|alternative)\b",
    re.I,
)

_SHOPPING_EXACT_SUBTYPE_PATTERNS: dict[str, re.Pattern[str]] = {
    "headphones": re.compile(r"\b(?:headphones?|earphones?)\b", re.I),
    "microphone": re.compile(r"\b(?:microphones?|mics?)\b", re.I),
    "speaker": re.compile(r"\bspeakers?\b", re.I),
    "keyboard": re.compile(r"\bkeyboards?\b", re.I),
    "mouse": re.compile(r"\b(?:mouse|mice)\b", re.I),
    "webcam": re.compile(r"\bwebcams?\b", re.I),
}

def resolve_semantic_gate_profile(extra: dict[str, Any]) -> str:
    """Resolve the orthogonal semantic-gate profile for one row.

    Single source of truth shared by merge, merge_validation and the
    consumption-side validator.  Legacy rows (no explicit profile) infer
    ``diagnostic_only`` for the strict generation profile and
    ``deterministic_v1`` otherwise.
    """
    profile = str(extra.get("semantic_gate_profile") or "")
    if not profile:
        profile = (
            "diagnostic_only"
            if str(extra.get("prompt_profile") or "")
            == "paper_generation_baseline_v1"
            else "deterministic_v1"
        )
    if profile not in {"diagnostic_only", "deterministic_v1"}:
        raise ValueError(f"invalid semantic_gate_profile: {profile!r}")
    return profile


def expected_artifact_purpose(extra: dict[str, Any]) -> str:
    """Derive the only valid artifact purpose for a profile pair."""
    if extra.get("fixed_attempt_budget") is True:
        return "experiment"
    prompt_profile = str(extra.get("prompt_profile") or "")
    semantic_profile = resolve_semantic_gate_profile(extra)
    if (
        prompt_profile == "paper_generation_baseline_v1"
        and semantic_profile == "diagnostic_only"
    ):
        return "paper_audit"
    if (
        prompt_profile == "local_trainable_v1"
        and semantic_profile == "deterministic_v1"
    ):
        return "training_candidate"
    return "experiment"


def validate_artifact_purpose(
    extra: dict[str, Any], *, require_training: bool = False,
) -> str:
    """Validate persisted purpose and optionally enforce training eligibility."""
    expected = expected_artifact_purpose(extra)
    persisted = str(extra.get("artifact_purpose") or "")
    if not persisted:
        raise ValueError("missing artifact_purpose")
    if persisted != expected:
        raise ValueError(
            "artifact_purpose/profile mismatch: "
            f"persisted={persisted!r}, expected={expected!r}"
        )
    if require_training and persisted != "training_candidate":
        raise ValueError(
            f"artifact purpose {persisted!r} is not training-consumable"
        )
    return persisted

@dataclass(frozen=True)
class SemanticQuarantineIssue:
    """One fact-backed reason why a Teacher candidate cannot be training GT.

    Semantic quarantine is a LOCAL trainability contract
    and must be reported separately from the three PROVE hard gates
    (fresh replay ≤30%, sensitive provenance, Jaccard 0.70).
    """

    reason_code: str
    round_idx: int
    user_evidence: str
    trace_evidence: dict[str, Any]
    hard_gate: bool = True
    gate_level: str = "LOCAL"  # "LOCAL" (semantic quarantine) | "PROVE" (paper gate)

    @property
    def quality_issue(self) -> str:
        return f"semantic_quarantine:{self.reason_code}"

    def to_dict(self, *, task_id: str = "", domain: str = "") -> dict[str, Any]:
        payload = asdict(self)
        payload["quality_issue"] = self.quality_issue
        payload["task_id"] = task_id
        payload["domain"] = domain
        return payload

def _json_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    return list(value) if isinstance(value, (list, tuple)) else []

def _json_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return {}
    return dict(value) if isinstance(value, dict) else {}

def _normalize(value: Any) -> str:
    return " ".join(str(value or "").casefold().split())


def _rounds(extra: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        _json_dict(value)
        for value in _json_list(extra.get("teacher_round_trace"))
        if _json_dict(value)
    ]


def _user_visible_backend_id_issue(
    extra: dict[str, Any], rounds: list[dict[str, Any]],
) -> SemanticQuarantineIssue | None:
    """Enforce the local natural-selector contract, not PROVE Step 2."""
    prompt_profile = str(extra.get("prompt_profile") or "")
    if prompt_profile and prompt_profile != "local_trainable_v1":
        return None
    domain = str(extra.get("domain") or "")
    queries = [
        str(item.get("user_query") or item.get("query") or "")
        for item in rounds
    ]
    calls = [_json_list(item.get("oracle_calls")) for item in rounds]
    fact_calls: list[list[dict[str, Any]]] = []
    observations: list[list[Any]] = []
    for round_trace in rounds:
        successful_events = _successful_history(round_trace)
        fact_calls.append([
            {
                "action": "tool_call",
                "tool_name": str(event.get("tool_name") or ""),
                "arguments": _json_dict(event.get("arguments")),
            }
            for event in successful_events
            if str(event.get("tool_name") or "")
        ])
        observations.append([
            event.get("observation") or {}
            for event in successful_events
            if str(event.get("tool_name") or "")
        ])
    entity_types = reference_entity_types(domain, [
        _json_dict(value)
        for value in _json_list(extra.get("clean_visible_tools"))
        if _json_dict(value)
    ])
    private_entity_ids, public_entity_ids = (
        typed_entity_reference_visibility_from_rounds(
        domain=domain,
        calls_per_round=fact_calls,
        observations_per_round=observations,
        server_tools=[
            _json_dict(value)
            for value in _json_list(extra.get("clean_visible_tools"))
            if _json_dict(value)
        ],
        entity_types=entity_types,
        )
    )
    exposure = user_visible_private_id_exposure(
        queries,
        calls,
        private_entity_ids=private_entity_ids,
        public_entity_ids=public_entity_ids,
    )
    if exposure is None:
        return None
    if exposure.surface == "user_query" and exposure.round_idx == 0:
        reason_code = "initial_query_exposes_private_entity_id"
    elif exposure.surface == "user_query":
        reason_code = "continuation_query_exposes_private_entity_id"
    else:
        reason_code = "terminal_exposes_private_entity_id"
    return SemanticQuarantineIssue(
        reason_code=reason_code,
        round_idx=exposure.round_idx,
        user_evidence=queries[exposure.round_idx],
        trace_evidence={
            "surface": exposure.surface,
            "surface_text": exposure.text,
            "leaked_ids": list(exposure.leaked_ids),
        },
    )


def _irrelevance_capability_issue(
    extra: dict[str, Any], rounds: list[dict[str, Any]],
) -> SemanticQuarantineIssue | None:
    """Recompute the deterministic no-tool capability proof."""
    if not _json_dict(extra.get("irrelevance_capability_proof")):
        return None
    from src.live_mcp.generation.irrelevance import (
        validate_irrelevance_capability_proof,
    )

    issue = validate_irrelevance_capability_proof(
        query=str(extra.get("user_query") or ""),
        proof=_json_dict(extra.get("irrelevance_capability_proof")),
        tool_schemas=[
            _json_dict(item)
            for item in _json_list(extra.get("clean_visible_tools"))
        ],
    )
    if issue is None:
        return None
    return SemanticQuarantineIssue(
        reason_code=issue,
        round_idx=0,
        user_evidence=str(extra.get("user_query") or ""),
        trace_evidence={
            "proof": _json_dict(extra.get("irrelevance_capability_proof")),
            "rounds": len(rounds),
        },
    )

def _successful_history(round_trace: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        event
        for raw in _json_list(round_trace.get("execution_history"))
        if (event := _json_dict(raw)) and event.get("success") is True
    ]

def _terminal(round_trace: dict[str, Any]) -> tuple[str, str]:
    for raw in reversed(_json_list(round_trace.get("oracle_calls"))):
        call = _json_dict(raw)
        action = str(call.get("action") or "tool_call")
        if action == "tool_call":
            continue
        arguments = _json_dict(call.get("arguments"))
        text = str(
            arguments.get("text")
            or arguments.get("question")
            or arguments.get("reason")
            or ""
        )
        return action, text
    return "", ""


def _terminal_final_answer_requests_user_input_issue(
    round_trace: dict[str, Any],
    *,
    round_idx: int,
) -> SemanticQuarantineIssue | None:
    """Cross-domain: a final_answer must not request new user input.

    A terminal final_answer that asks the user to provide/confirm/choose
    something violates the terminal contract and is provable from the
    terminal text alone under the local semantic gate contract.
    """
    action, terminal_text = _terminal(round_trace)
    if action == "final_answer" and _final_answer_requests_user_input(terminal_text):
        return SemanticQuarantineIssue(
            reason_code="terminal_final_answer_requests_user_input",
            round_idx=round_idx,
            user_evidence=str(round_trace.get("user_query") or ""),
            trace_evidence={"terminal_text": terminal_text},
        )
    return None


def _terminal_exposes_private_tool_name_issue(
    round_trace: dict[str, Any],
    *,
    round_idx: int,
    tool_names: set[str],
    hidden_tool_names: set[str] | None = None,
) -> SemanticQuarantineIssue | None:
    """Cross-domain: terminal text must not leak internal tool names.

    Tool names are sampler-internal identifiers; surfacing them verbatim in
    a terminal answer is provable from the terminal text plus the visible
    schema under the local reference-visibility contract.
    """
    exposure = user_visible_terminal_tool_name_exposure(
        [_json_list(round_trace.get("oracle_calls"))],
        tool_names=tool_names,
        hidden_tool_names=hidden_tool_names,
    )
    if exposure is not None:
        return SemanticQuarantineIssue(
            reason_code="terminal_exposes_private_tool_name",
            round_idx=round_idx,
            user_evidence=str(round_trace.get("user_query") or ""),
            trace_evidence={
                "terminal_text": exposure.text,
                "exposed_tool_names": list(exposure.exposed_tool_names),
            },
        )
    return None


def _missing_function_contract_issue(
    extra: dict[str, Any],
    rounds: list[dict[str, Any]],
) -> SemanticQuarantineIssue | None:
    """Reject a persisted missing-function contract contradiction."""
    scenario_type = str(extra.get("scenario_type") or "")
    explicit_missing_function = bool(
        extra.get("has_missing_function")
    ) or scenario_type == "missing_function"
    if not explicit_missing_function and scenario_type not in {
        "missing_function", "clarification_required",
    }:
        return None
    hidden_tools = [
        str(value) for value in _json_list(extra.get("hidden_tools"))
        if str(value or "")
    ]
    if not explicit_missing_function and not hidden_tools:
        return None
    if len(hidden_tools) != 1:
        return SemanticQuarantineIssue(
            reason_code="missing_function_hidden_target_mismatch",
            round_idx=0,
            user_evidence="\n".join(
                str(item.get("user_query") or item.get("query") or "")
                for item in rounds
            ),
            trace_evidence={
                "hidden_tools": hidden_tools,
                "source_chain_seed": _json_list(
                    extra.get("source_chain_seed")
                ),
            },
        )
    source_chain = [
        str(value) for value in _json_list(extra.get("source_chain_seed"))
        if str(value or "")
    ]
    if not source_chain or hidden_tools[0] != source_chain[-1]:
        return SemanticQuarantineIssue(
            reason_code="missing_function_hidden_target_mismatch",
            round_idx=0,
            user_evidence="\n".join(
                str(item.get("user_query") or item.get("query") or "")
                for item in rounds
            ),
            trace_evidence={
                "hidden_tool": hidden_tools[0],
                "source_chain_seed": source_chain,
            },
        )
    evidence = [
        str(value) for value in _json_list(
            extra.get("missing_function_evidence")
        ) if str(value or "")
    ]
    prompt_profile = str(extra.get("prompt_profile") or "")
    if prompt_profile != "paper_generation_baseline_v1" and not evidence:
        return SemanticQuarantineIssue(
            reason_code="missing_function_capability_evidence_missing",
            round_idx=0,
            user_evidence="\n".join(
                str(item.get("user_query") or item.get("query") or "")
                for item in rounds
            ),
            trace_evidence={"hidden_tool": hidden_tools[0]},
        )
    oracle_calls = [
        _json_dict(value) for value in _json_list(extra.get("oracle_calls"))
        if _json_dict(value)
    ]
    tool_schemas = [
        _json_dict(value) for value in _json_list(extra.get("clean_visible_tools"))
        if _json_dict(value)
    ]
    execution_history = [
        event
        for round_trace in rounds
        for event in _json_list(round_trace.get("execution_history"))
        if isinstance(event, dict)
    ]
    unresolved_failures = missing_function_unresolved_failures(
        True, execution_history,
    )
    if unresolved_failures:
        return SemanticQuarantineIssue(
            reason_code="missing_function_unresolved_execution_failure",
            round_idx=0,
            user_evidence="\n".join(
                str(item.get("user_query") or item.get("query") or "")
                for item in rounds
            ),
            trace_evidence={
                "hidden_tool": hidden_tools[0],
                "source_chain_seed": source_chain,
                "unresolved_failed_tools": sorted(unresolved_failures),
            },
        )
    if not missing_function_has_mutation(
        True,
        oracle_calls,
        tool_schemas,
    ):
        return None
    schema_by_name = {
        str(item.get("name") or ""): item for item in tool_schemas
    }
    mutating_calls = [
        str(call.get("tool_name") or "")
        for call in oracle_calls
        if call.get("action", "tool_call") == "tool_call"
        and (
            schema_by_name.get(str(call.get("tool_name") or ""), {}).get(
                "annotations"
            ) or {}
        ).get("mutating") is True
    ]
    return SemanticQuarantineIssue(
        reason_code="missing_function_mutation",
        round_idx=0,
        user_evidence="\n".join(
            str(item.get("user_query") or item.get("query") or "")
            for item in rounds
        ),
        trace_evidence={
            "hidden_tool": hidden_tools[0],
            "source_chain_seed": source_chain,
            "unauthorized_calls": mutating_calls,
        },
    )

def _assistant_role_user_query_issue(
    rounds: list[dict[str, Any]],
) -> SemanticQuarantineIssue | None:
    for position, round_trace in enumerate(rounds):
        query = str(
            round_trace.get("user_query") or round_trace.get("query") or ""
        )
        if not _ASSISTANT_ROLE_USER_QUERY_RE.search(query):
            continue
        return SemanticQuarantineIssue(
            reason_code="assistant_role_offer_recorded_as_user_query",
            round_idx=int(round_trace.get("round_idx", position)),
            user_evidence=query,
            trace_evidence={"user_query": query},
        )
    return None

def _requested_recommendation_subtype(query: str) -> str:
    """Return a subtype requested as the recommendation outcome.

    A subtype mentioned only as the ``based_on`` source is not an output type
    constraint.  This narrower parser avoids rejecting honest cross-category
    recommendation results when the user asked for accessories *based on*
    headphones.
    """
    goal_clause = re.split(
        r"\bbased\s+on\b",
        query,
        maxsplit=1,
        flags=re.I,
    )[0]
    mentioned = {
        subtype
        for subtype, pattern in _SHOPPING_EXACT_SUBTYPE_PATTERNS.items()
        if pattern.search(query)
    }
    if len(mentioned) != 1:
        return ""
    requested: set[str] = set()
    for subtype, subtype_pattern in _SHOPPING_EXACT_SUBTYPE_PATTERNS.items():
        subtype_source = subtype_pattern.pattern
        after_verb = re.search(
            rf"\b(?:recommend|suggest|find|show)\b"
            rf"(?P<modifier>[^.!?\n]{{0,60}}?){subtype_source}",
            goal_clause,
            re.I,
        )
        if not after_verb:
            continue
        modifier = _normalize(after_verb.group("modifier"))
        # Generic nouns/connectors make the subtype a similarity source rather
        # than the requested output class: "other products similar to the
        # headphones" does not require another headphone.
        if re.search(
            r"\b(?:products?|things?|items?|accessor(?:y|ies)|"
            r"similar\s+to|based\s+on|like)\b",
            modifier,
            re.I,
        ):
            continue
        if len(modifier.split()) <= 5:
            requested.add(subtype)
    return next(iter(requested)) if len(requested) == 1 else ""

def _record_matches_subtype(record: dict[str, Any], subtype: str) -> bool:
    name = _normalize(record.get("name") or record.get("title"))
    description = _normalize(record.get("description"))
    category = _normalize(record.get("category"))
    visible = f"{name} {description}"
    if subtype == "keyboard":
        return bool(
            category == "keyboard"
            or (
                re.search(r"\bkeyboard\b", name)
                and not re.search(r"\bwrist\s+rest\b", name)
            )
        )
    if subtype == "mouse":
        return bool(
            category == "mouse"
            or (
                re.search(r"\bmouse\b", name)
                and not re.search(r"\bmouse\s+pad\b", name)
            )
        )
    if subtype == "webcam":
        return bool(re.search(r"\bwebcam\b", visible)) or category == "camera"
    return bool(_SHOPPING_EXACT_SUBTYPE_PATTERNS[subtype].search(visible))

def _terminal_positively_presents_record(
    terminal_text: str,
    record: dict[str, Any],
) -> bool:
    normalized = _normalize(terminal_text)
    selectors = [
        _normalize(record.get("name")),
        _normalize(record.get("product_id")),
    ]
    positions = [
        normalized.find(selector)
        for selector in selectors
        if selector and selector in normalized
    ]
    if not positions:
        return False
    position = min(positions)
    window = normalized[max(0, position - 100):position + 180]
    return bool(
        _POSITIVE_RECOMMENDATION_RE.search(window)
        and not _NEGATIVE_RECOMMENDATION_RE.search(window)
    )

def _terminal_presents_record_without_rejection(
    terminal_text: str,
    record: dict[str, Any],
) -> bool:
    normalized = _normalize(terminal_text)
    selectors = [
        _normalize(record.get("name")),
        _normalize(record.get("product_id")),
    ]
    positions = [
        normalized.find(selector)
        for selector in selectors
        if selector and selector in normalized
    ]
    if not positions:
        return False
    position = min(positions)
    window = normalized[max(0, position - 100):position + 180]
    return not _NEGATIVE_RECOMMENDATION_RE.search(window)

def _order_memberships(
    history: list[dict[str, Any]],
) -> list[tuple[str, set[str]]]:
    memberships: list[tuple[str, set[str]]] = []
    for event in history:
        if event.get("tool_name") != "get_order":
            continue
        order = _json_dict(_json_dict(event.get("observation")).get("order"))
        order_id = str(order.get("order_id") or "")
        product_ids = {
            str(value)
            for value in _json_list(order.get("product_ids"))
            if str(value or "")
        }
        for raw_item in _json_list(order.get("items")):
            product_id = str(_json_dict(raw_item).get("product_id") or "")
            if product_id:
                product_ids.add(product_id)
        if order_id and len(product_ids) > 1:
            memberships.append((order_id, product_ids))
    return memberships

def _observed_product_names(history: list[dict[str, Any]]) -> set[str]:
    names: set[str] = set()
    for event in history:
        observation = _json_dict(event.get("observation"))
        product = _json_dict(observation.get("product"))
        name = _normalize(product.get("name"))
        if name:
            names.add(name)
    return names

def _relevant_item_tools(query: str) -> set[str]:
    relevant: set[str] = set()
    if _ORDER_ITEM_REVIEW_RE.search(query):
        relevant.update({"get_reviews", "add_review"})
    if _ORDER_ITEM_DETAIL_RE.search(query):
        relevant.add("get_product")
    if _ORDER_ITEM_COMPARE_RE.search(query):
        relevant.add("compare_products")
    return relevant

def _event_product_ids(event: dict[str, Any]) -> set[str]:
    arguments = _json_dict(event.get("arguments"))
    values: list[Any] = []
    if arguments.get("product_id") is not None:
        values.append(arguments["product_id"])
    values.extend(_json_list(arguments.get("product_ids")))
    return {str(value) for value in values if str(value or "")}

def _call_signature(event: dict[str, Any]) -> str:
    return json.dumps(
        {
            "tool_name": str(event.get("tool_name") or ""),
            "arguments": _json_dict(event.get("arguments")),
        },
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    )

def _product_names_by_id(history: list[dict[str, Any]]) -> dict[str, set[str]]:
    names_by_id: dict[str, set[str]] = {}

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            product_id = _normalize(value.get("product_id"))
            if product_id:
                for field in ("name", "title"):
                    name = _normalize(value.get(field))
                    if name:
                        names_by_id.setdefault(product_id, set()).add(name)
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    for event in history:
        visit(_json_dict(event.get("observation")))
    return names_by_id

def _product_stock_by_id(history: list[dict[str, Any]]) -> dict[str, float]:
    stock_by_id: dict[str, float] = {}

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            product_id = _normalize(value.get("product_id"))
            stock = value.get("stock")
            if product_id and isinstance(stock, (int, float)):
                stock_by_id[product_id] = float(stock)
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    for event in history:
        visit(_json_dict(event.get("observation")))
    return stock_by_id

def _product_records_by_id(
    history: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            product_id = _normalize(value.get("product_id"))
            if product_id and any(
                field in value
                for field in ("name", "title", "category", "description")
            ):
                records.setdefault(product_id, {}).update(value)
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    for event in history:
        visit(_json_dict(event.get("observation")))
    return records

def _review_selector_candidate_ids(
    *,
    query: str,
    terminal_sentence: str,
    candidate_records: dict[str, dict[str, Any]],
) -> set[str]:
    normalized_sentence = _normalize(terminal_sentence)
    named_ids = {
        product_id
        for product_id, record in candidate_records.items()
        if any(
            selector and selector in normalized_sentence
            for selector in (
                _normalize(record.get("name")),
                _normalize(product_id),
            )
        )
    }
    if named_ids:
        return named_ids
    selector_text = f"{query} {terminal_sentence}"
    priced = [
        (product_id, record.get("price"))
        for product_id, record in candidate_records.items()
        if isinstance(record.get("price"), (int, float))
    ]
    if priced and re.search(
        r"\b(?:highest[-\s]?priced|highest\s+price|most\s+expensive)\b",
        selector_text,
        re.IGNORECASE,
    ):
        highest = max(price for _, price in priced)
        return {product_id for product_id, price in priced if price == highest}
    if priced and re.search(
        r"\b(?:lowest[-\s]?priced|lowest\s+price|cheapest)\b",
        selector_text,
        re.IGNORECASE,
    ):
        lowest = min(price for _, price in priced)
        return {product_id for product_id, price in priced if price == lowest}
    if candidate_records and re.search(
        r"\bfirst\s+(?:one|item|product|result|option)\b",
        selector_text,
        re.IGNORECASE,
    ):
        return {next(iter(candidate_records))}
    return set(candidate_records)
