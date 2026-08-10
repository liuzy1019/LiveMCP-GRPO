"""Shared deterministic semantic-quarantine evidence helpers."""

from __future__ import annotations

import json

import re

from dataclasses import asdict, dataclass

from typing import Any

from src.live_mcp.generation.teacher_contracts import (
    _final_answer_requests_user_input,
)
from src.live_mcp.generation.robustness import (
    missing_function_has_nonprefix_mutation,
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

_SYNTHETIC_BACKEND_ID_RE = re.compile(
    r"(?<![A-Za-z0-9_])[A-Za-z][A-Za-z0-9]*_s\d+_\d+"
    r"(?![A-Za-z0-9_])"
)

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

    Per OVAL-MCP.md §5, semantic quarantine is a LOCAL trainability contract
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

def _initial_query_backend_id_issue(
    rounds: list[dict[str, Any]],
) -> SemanticQuarantineIssue | None:
    """Reject sampler-private seeded IDs leaked into the initial user turn."""
    if not rounds:
        return None
    query = str(rounds[0].get("user_query") or rounds[0].get("query") or "")
    leaked = sorted(set(_SYNTHETIC_BACKEND_ID_RE.findall(query)))
    if not leaked:
        return None
    return SemanticQuarantineIssue(
        reason_code="initial_query_exposes_sampler_private_id",
        round_idx=int(rounds[0].get("round_idx", 0)),
        user_evidence=query,
        trace_evidence={"leaked_ids": leaked},
    )

def _successful_history(round_trace: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        event
        for raw in _json_list(round_trace.get("execution_history"))
        if (event := _json_dict(raw)) and event.get("success") is True
    ]

def _successful_trace_source_target_issue(
    extra: dict[str, Any],
    rounds: list[dict[str, Any]],
) -> SemanticQuarantineIssue | None:
    """Mirror the online source-target contract on persisted traces."""
    scenario_type = str(extra.get("scenario_type") or "")
    if scenario_type not in {"normal_safe_success", "tool_error_recovery"}:
        return None
    source_chain = [
        str(value or "")
        for value in _json_list(extra.get("source_chain_seed"))
        if str(value or "")
    ]
    if not source_chain:
        return None
    target = source_chain[-1]
    primary_domain = str(extra.get("domain") or "")
    realized_tools: list[str] = []
    for round_trace in rounds:
        for raw_call in _json_list(round_trace.get("oracle_calls")):
            call = _json_dict(raw_call)
            if str(call.get("action") or "tool_call") != "tool_call":
                continue
            server_name = str(call.get("server_name") or primary_domain)
            if server_name != primary_domain:
                continue
            realized_tools.append(str(call.get("tool_name") or ""))
    if target in realized_tools:
        return None
    return SemanticQuarantineIssue(
        reason_code="successful_trace_missing_source_target",
        round_idx=0,
        user_evidence="\n".join(
            str(round_trace.get("user_query") or round_trace.get("query") or "")
            for round_trace in rounds
        ),
        trace_evidence={
            "source_chain_seed": source_chain,
            "missing_target": target,
            "realized_tools": realized_tools,
            "scenario_type": scenario_type,
        },
    )

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
    terminal text alone (OVAL-MCP.md §5).
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
    schema (OVAL-MCP.md §5).
    """
    action, terminal_text = _terminal(round_trace)
    if not action:
        return None
    hidden_tool_names = hidden_tool_names or set()
    exposed: list[str] = []
    for tool_name in sorted(tool_names):
        bounded = rf"(?<![A-Za-z0-9_]){re.escape(tool_name)}(?![A-Za-z0-9_])"
        if "_" in tool_name and re.search(
            bounded, terminal_text, re.IGNORECASE,
        ):
            exposed.append(tool_name)
            continue
        if tool_name not in hidden_tool_names:
            continue
        code_like = (
            rf"(?:`{re.escape(tool_name)}`|\({re.escape(tool_name)}\)|"
            rf"\b{re.escape(tool_name)}\s+(?:tool|command|function)\b)"
        )
        if re.search(code_like, terminal_text, re.IGNORECASE):
            exposed.append(tool_name)
    if exposed:
        return SemanticQuarantineIssue(
            reason_code="terminal_exposes_private_tool_name",
            round_idx=round_idx,
            user_evidence=str(round_trace.get("user_query") or ""),
            trace_evidence={
                "terminal_text": terminal_text,
                "exposed_tool_names": exposed,
            },
        )
    return None


def _missing_function_nonprefix_mutation_issue(
    extra: dict[str, Any],
    rounds: list[dict[str, Any]],
) -> SemanticQuarantineIssue | None:
    """Reject a persisted missing-function trace with an unauthorized write."""
    if str(extra.get("scenario_type") or "") != "missing_function":
        return None
    hidden_tools = [
        str(value) for value in _json_list(extra.get("hidden_tools"))
        if str(value or "")
    ]
    if len(hidden_tools) != 1:
        return None
    source_chain = [
        str(value) for value in _json_list(extra.get("source_chain_seed"))
        if str(value or "")
    ]
    if hidden_tools[0] not in source_chain:
        return None
    oracle_calls = [
        _json_dict(value) for value in _json_list(extra.get("oracle_calls"))
        if _json_dict(value)
    ]
    tool_schemas = [
        _json_dict(value) for value in _json_list(extra.get("clean_visible_tools"))
        if _json_dict(value)
    ]
    mutation_evidence = [
        _json_dict(value)
        for value in _json_list(extra.get("query_mutation_evidence"))
        if _json_dict(value)
    ]
    authorized = {
        str(item.get("capability") or "")
        for item in mutation_evidence
        if str(item.get("capability") or "") != hidden_tools[0]
    }
    hidden_position = source_chain.index(hidden_tools[0])
    authorized.update(source_chain[:hidden_position])
    if not missing_function_has_nonprefix_mutation(
        True,
        oracle_calls,
        source_chain,
        hidden_tools[0],
        tool_schemas,
        authorized,
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
        and str(call.get("tool_name") or "") not in authorized
    ]
    return SemanticQuarantineIssue(
        reason_code="missing_function_nonprefix_mutation",
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
