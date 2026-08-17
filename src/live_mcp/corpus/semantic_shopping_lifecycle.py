"""Shopping lifecycle, terminal, and mutation semantic gates."""

from __future__ import annotations


import re


from typing import Any

from src.live_mcp.corpus.semantic_core import (

    SemanticQuarantineIssue,

    _call_signature,

    _json_dict,

    _json_list,

    _normalize,

    _order_memberships,

    _product_names_by_id,

    _product_stock_by_id,

    _record_matches_subtype,

    _review_selector_candidate_ids,

    _successful_history,

    _terminal,

)

_CART_PURCHASE_REQUEST_RE = re.compile(
    r"add\s+(?:\w[\w\s]*?)\s+to\s+(?:my\s+)?cart"
    r"|add\s+to\s+(?:my\s+)?cart"
    r"|put\s+(?:\w[\w\s]*?)\s+(?:in|into)\s+(?:my\s+)?(?:cart|basket)"
    r"|\b(?:buy|purchas\w*|check\s*out|checkout|place\s+(?:an\s+)?order)\b"
    r"|\b(?:i(?:'ll|\s+will|\s+would)?\s+(?:take|get|grab|go\s+with)|"
    r"let\s+me\s+(?:get|have|grab)|i(?:'d|\s+would)\s+like\s+to\s+(?:get|buy|order|purchase))"
    r"|\b(?:i\s+want\s+(?:to\s+(?:get|buy|order|purchase)\s+)?(?:the|a|an|this|that|those|these|one|some))"
    r"|\b(?:order|get\s+me|give\s+me|i\s+need)\s+(?:the|a|an|this|that)",
    re.IGNORECASE,
)

_CONCRETE_PRODUCT_MUTATION_PATTERNS: dict[str, re.Pattern[str]] = {
    "add_to_cart": re.compile(
        r"\b(?:add|put)\b\s+(?:the\s+|a\s+|an\s+|one\s+)?"
        r"(?P<selector>[^.!?\n]{2,80}?)\s+"
        r"(?:to|in|into|on)\s+(?:my\s+|the\s+)?(?:cart|basket)\b",
        re.IGNORECASE,
    ),
    "add_to_wishlist": re.compile(
        r"\b(?:add|put|save)\b\s+(?:the\s+|a\s+|an\s+|one\s+)?"
        r"(?P<selector>[^.!?\n]{2,80}?)\s+"
        r"(?:to|on|into)\s+(?:my\s+|the\s+)?(?:wish\s*list|wishlist)\b",
        re.IGNORECASE,
    ),
    "remove_from_wishlist": re.compile(
        r"\b(?:remove|delete|take)\b\s+(?:the\s+|a\s+|an\s+|one\s+)?"
        r"(?P<selector>[^.!?\n]{2,80}?)\s+"
        r"(?:from|off)\s+(?:my\s+|the\s+)?(?:wish\s*list|wishlist)\b",
        re.IGNORECASE,
    ),
}

_CONFIRMATION_ONLY_RE = re.compile(
    r"\b(?:are\s+you\s+(?:absolutely\s+|really\s+)?sure|"
    r"can\s+you\s+(?:double[- ]?check|confirm|verify)|"
    r"(?:double[- ]?check|check)\s+(?:it\s+)?again|"
    r"please\s+(?:confirm|verify))\b",
    re.IGNORECASE,
)

_FRESH_VERIFICATION_CLAIM_RE = re.compile(
    r"\b(?:i\s+have\s+double[- ]?checked|i\s+double[- ]?checked|"
    r"i\s+have\s+(?:checked|verified)|i\s+(?:rechecked|verified)|"
    r"checked\s+(?:the\s+)?system\s+again)\b",
    re.IGNORECASE,
)

_META_TERMINAL_RE = re.compile(
    r"^\s*(?:the\s+)?user(?:"
    r"\s+(?:is\s+asking|asks?|wants?|requested)\b"
    r"|['’]s\s+(?:request|question|query)\b)",
    re.IGNORECASE,
)

_OPAQUE_ORDER_ID_RE = re.compile(
    r"(?<![A-Za-z0-9_])ord_[A-Za-z0-9_-]+(?![A-Za-z0-9_])",
    re.IGNORECASE,
)

_OPAQUE_PRODUCT_ID_RE = re.compile(
    r"(?<![A-Za-z0-9_])prd_[A-Za-z0-9_-]+(?![A-Za-z0-9_])",
    re.IGNORECASE,
)

_ORDER_ITEMS_OUTCOME_RE = re.compile(
    r"\b(?:check|find|show|list|identify|what\s+(?:are|were))\b"
    r"[^.!?\n]{0,64}(?:"
    r"\bitems?\b\s+(?:in|from|of)\s+[^.!?\n]{0,32}\border\b"
    r"|\border(?:'s)?\s+(?:line\s+)?items?\b)",
    re.IGNORECASE,
)

_RECOMMENDATION_REQUEST_RE = re.compile(
    r"\b(?:recommend|recommendation|suggest|suggestion)\w*\b",
    re.IGNORECASE,
)

_REVIEW_AVAILABILITY_QUERY_RE = re.compile(
    r"\b(?:with|have|has|any|show|see|read)\b[^.!?\n]{0,48}"
    r"\b(?:reviews?|ratings?)\b|\b(?:reviews?|ratings?)\b",
    re.IGNORECASE,
)

_SHOPPING_EXACT_SUBTYPE_PATTERNS: dict[str, re.Pattern[str]] = {
    "headphones": re.compile(r"\b(?:headphones?|earphones?)\b", re.I),
    "microphone": re.compile(r"\b(?:microphones?|mics?)\b", re.I),
    "speaker": re.compile(r"\bspeakers?\b", re.I),
    "keyboard": re.compile(r"\bkeyboards?\b", re.I),
    "mouse": re.compile(r"\b(?:mouse|mice)\b", re.I),
    "webcam": re.compile(r"\bwebcams?\b", re.I),
}

_SHOPPING_LIFECYCLE_STATUS_TOOLS = frozenset({
    "get_return_status",
})

_SOCIAL_ONLY_RE = re.compile(
    r"^\s*(?:(?:actually|okay|ok|well|awesome|great|perfect)[\s,!.:-]*)?"
    r"(?:(?:i(?:'m|\s+am)\s+all\s+set|that(?:'s|\s+is)\s+all|"
    r"nothing\s+else|no(?:thing)?\s+more|all\s+done)[\s,!.:-]*)?"
    r"(?:thank\s+you|thanks)(?:\s+(?:again|so\s+much|"
    r"for\s+(?:doing\s+)?that|for\s+your\s+help))?[\s!.]*$",
    re.IGNORECASE,
)

_UNSOLICITED_MUTATIONS: frozenset[str] = frozenset({
    "add_to_cart", "checkout",
})

def _shopping_repeated_lifecycle_read_issue(
    rounds: list[dict[str, Any]],
) -> SemanticQuarantineIssue | None:
    """Reject a cross-round status read when deterministic state did not move."""
    state_version = 0
    seen: dict[str, tuple[int, int, str, dict[str, Any]]] = {}
    for position, round_trace in enumerate(rounds):
        round_idx = int(round_trace.get("round_idx", position))
        query = str(
            round_trace.get("user_query")
            or round_trace.get("query")
            or ""
        )
        for event in _successful_history(round_trace):
            tool_name = str(event.get("tool_name") or "")
            if tool_name in _SHOPPING_LIFECYCLE_STATUS_TOOLS:
                signature = _call_signature(event)
                previous = seen.get(signature)
                if (
                    previous is not None
                    and previous[0] != round_idx
                    and previous[1] == state_version
                ):
                    return SemanticQuarantineIssue(
                        reason_code=(
                            "shopping_repeated_lifecycle_read_without_state_change"
                        ),
                        round_idx=round_idx,
                        user_evidence=query,
                        trace_evidence={
                            "tool_name": tool_name,
                            "arguments": _json_dict(event.get("arguments")),
                            "previous_round_idx": previous[0],
                            "previous_query": previous[2],
                            "previous_observation": previous[3],
                            "current_observation": _json_dict(
                                event.get("observation")
                            ),
                            "state_version": state_version,
                        },
                        hard_gate=False,
                    )
                seen[signature] = (
                    round_idx,
                    state_version,
                    query,
                    _json_dict(event.get("observation")),
                )
            if event.get("state_changed") is True:
                state_version += 1
    return None

def _shopping_social_only_continuation_issue(
    rounds: list[dict[str, Any]],
) -> SemanticQuarantineIssue | None:
    for position, round_trace in enumerate(rounds[1:], start=1):
        query = str(round_trace.get("user_query") or round_trace.get("query") or "")
        if not _SOCIAL_ONLY_RE.match(query) or _successful_history(round_trace):
            continue
        action, text = _terminal(round_trace)
        if action == "final_answer":
            return SemanticQuarantineIssue(
                reason_code="shopping_non_actionable_social_continuation",
                round_idx=int(round_trace.get("round_idx", position)),
                user_evidence=query,
                trace_evidence={"terminal_text": text, "successful_tool_calls": 0},
                hard_gate=False,
            )
    return None

def _confirmation_claims_fresh_verification_issue(
    rounds: list[dict[str, Any]],
) -> SemanticQuarantineIssue | None:
    """Reject a fresh-verification claim made with no current-round evidence."""
    for position, round_trace in enumerate(rounds[1:], start=1):
        query = str(round_trace.get("user_query") or round_trace.get("query") or "")
        if not _CONFIRMATION_ONLY_RE.search(query):
            continue
        if _successful_history(round_trace):
            continue
        action, terminal_text = _terminal(round_trace)
        if action not in {"final_answer", "report_error"}:
            continue
        if not _FRESH_VERIFICATION_CLAIM_RE.search(terminal_text):
            continue
        return SemanticQuarantineIssue(
            reason_code=(
                "confirmation_claims_fresh_verification_without_execution"
            ),
            round_idx=int(round_trace.get("round_idx", position)),
            user_evidence=query,
            trace_evidence={
                "terminal_text": terminal_text,
                "successful_current_round_events": 0,
            },
        )
    return None

def _shopping_order_status_filter_issue(
    rounds: list[dict[str, Any]],
) -> SemanticQuarantineIssue | None:
    """Flag order-status filters that are not grounded in user intent.

    PROVE §5: semantic quarantine is a LOCAL trainability contract, not a
    PROVE hard gate.  The original implementation required the status value
    to appear in the user query in a constraint pattern — this produced
    excessive false positives (30 removals in one run) because exploratory
    ``list_orders`` filtering with a status the user never mentioned is
    legitimate Teacher behavior.

    Relaxed rule (three cases):
      1. User constrains order status (e.g., "show my shipped orders"):
         quarantine only if the Teacher's filter **contradicts** the
         user's stated status.
      2. Teacher filters by a status word that appears in the user text
         OUTSIDE a constraint pattern (e.g., "my order placed on January
         11th" → ``status="placed"``): likely a verb misread → quarantine.
      3. Teacher filters by a status that appears NOWHERE in the user text
         (e.g., user says "find my orders" → ``status="shipped"``): pure
         exploration → allow.
    """
    visible_user_parts: list[str] = []
    for position, round_trace in enumerate(rounds):
        query = str(round_trace.get("user_query") or round_trace.get("query") or "")
        visible_user_parts.append(query)
        visible_user_text = "\n".join(visible_user_parts)
        for event in _successful_history(round_trace):
            if event.get("tool_name") != "list_orders":
                continue
            status = _normalize(_json_dict(event.get("arguments")).get("status"))
            if not status:
                continue

            # Case 1: user's explicitly constrained statuses.
            user_statuses: set[str] = set()
            for pattern in (
                # "shipped orders" (status before noun)
                r"\b(pending|processing|placed|shipped|delivered|cancelled|refunded|returned|returning)\s+orders?\b",
                # "orders that are shipped" / "orders currently shipped"
                # / "orders with status shipped"
                r"\borders?\s+(?:that\s+(?:are|is|were|was|have\s+been|has\s+been)\s+|currently\s+|with\s+(?:the\s+)?status\s+)(pending|processing|placed|shipped|delivered|cancelled|refunded|returned|returning)\b",
                # "order status is shipped" / "status of shipped"
                r"\b(?:order\s+)?status\s+(?:is\s+|of\s+)?(pending|processing|placed|shipped|delivered|cancelled|refunded|returned|returning)\b",
                # "the item I'm currently returning" / "products we returned"
                r"\b(?:items?|products?|purchases?|things?)\s+"
                r"(?:(?:that|which)\s+)?"
                r"(?:i(?:'m|\s+am|\s+was|\s+have)?|"
                r"we(?:'re|\s+are|\s+were|\s+have)?)\s+"
                r"(?:currently\s+)?"
                r"(pending|processing|placed|shipped|delivered|cancelled|refunded|returned|returning)\b",
            ):
                for match in re.finditer(pattern, visible_user_text, re.IGNORECASE):
                    user_statuses.add(_normalize(match.group(1)))
            if user_statuses:
                if status in user_statuses:
                    continue  # grounded in a user constraint — fine
                return SemanticQuarantineIssue(
                    reason_code="shopping_unauthorized_order_status_filter",
                    round_idx=int(round_trace.get("round_idx", position)),
                    user_evidence=query,
                    trace_evidence={
                        "status": status,
                        "user_statuses": sorted(user_statuses),
                        "arguments": _json_dict(event.get("arguments")),
                        "observation": _json_dict(event.get("observation")),
                    },
                    hard_gate=True,  # contradiction with stated constraint
                )

            # Case 2: no user constraint, but the status word appears in the
            # user text anyway (non-constraint usage) — likely a verb misread.
            if re.search(rf"\b{re.escape(status)}\b", visible_user_text, re.IGNORECASE):
                return SemanticQuarantineIssue(
                    reason_code="shopping_unauthorized_order_status_filter",
                    round_idx=int(round_trace.get("round_idx", position)),
                    user_evidence=query,
                    trace_evidence={
                        "status": status,
                        "non_constraint_mention": True,
                        "arguments": _json_dict(event.get("arguments")),
                        "observation": _json_dict(event.get("observation")),
                    },
                    hard_gate=True,  # verb misread is a semantic error
                )

            # Case 3: status appears nowhere in user text — exploratory
            # filtering is legitimate; do not quarantine.
    return None

def _shopping_skipped_explicit_product_mutation_issue(
    round_trace: dict[str, Any],
    *,
    round_idx: int,
    visible_tool_names: set[str],
) -> SemanticQuarantineIssue | None:
    action, terminal_text = _terminal(round_trace)
    if action not in {"ask_clarification", "report_error"}:
        return None
    query = str(round_trace.get("user_query") or round_trace.get("query") or "")
    history = _successful_history(round_trace)
    names_by_id = _product_names_by_id(history)
    stock_by_id = _product_stock_by_id(history)
    discovery_attempted = any(
        event.get("tool_name") == "search_products"
        for event in history
    )
    for tool_name, pattern in _CONCRETE_PRODUCT_MUTATION_PATTERNS.items():
        if tool_name not in visible_tool_names:
            continue
        completed_ids = {
            _normalize(_json_dict(event.get("arguments")).get("product_id"))
            for event in history
            if event.get("tool_name") == tool_name
        }
        for match in pattern.finditer(query):
            selector = _normalize(match.group("selector"))
            if selector in {
                "", "anything", "item", "one", "product", "something",
                "that", "that item", "that one", "this", "this item", "this one",
            } or re.search(
                r"^(?:one|item|product|thing)\s+"
                r"(?:i|that|which|you|from|in|on)\b",
                selector,
                re.IGNORECASE,
            ):
                continue
            matching_ids = {
                product_id
                for product_id, names in names_by_id.items()
                if selector in names or any(name in selector for name in names)
            }
            blocked_by_stock = bool(
                tool_name == "add_to_cart"
                and len(matching_ids) == 1
                and stock_by_id.get(next(iter(matching_ids)), 1.0) <= 0
            )
            if (
                len(matching_ids) == 1
                and matching_ids.isdisjoint(completed_ids)
                and not blocked_by_stock
            ):
                return SemanticQuarantineIssue(
                    reason_code="shopping_terminal_skips_explicit_product_mutation",
                    round_idx=round_idx,
                    user_evidence=query,
                    trace_evidence={
                        "tool_name": tool_name,
                        "selector": selector,
                        "resolved_product_id": next(iter(matching_ids)),
                        "terminal_action": action,
                        "terminal_text": terminal_text,
                    },
                )
            if (
                not matching_ids
                and not discovery_attempted
                and "search_products" in visible_tool_names
            ):
                return SemanticQuarantineIssue(
                    reason_code="shopping_terminal_skips_explicit_product_mutation",
                    round_idx=round_idx,
                    user_evidence=query,
                    trace_evidence={
                        "tool_name": "search_products",
                        "selector": selector,
                        "resolved_product_id": "",
                        "terminal_action": action,
                        "terminal_text": terminal_text,
                        "missing_product_discovery": True,
                    },
                )
    return None

def _shopping_order_items_name_coverage_issue(
    round_trace: dict[str, Any],
    *,
    round_idx: int,
) -> SemanticQuarantineIssue | None:
    query = str(round_trace.get("user_query") or round_trace.get("query") or "")
    action, terminal_text = _terminal(round_trace)
    if action != "final_answer" or not _ORDER_ITEMS_OUTCOME_RE.search(query):
        return None
    history = _successful_history(round_trace)
    order: dict[str, Any] = {}
    for event in reversed(history):
        if event.get("tool_name") == "get_order":
            order = _json_dict(_json_dict(event.get("observation")).get("order"))
            if order:
                break
    items = _json_list(order.get("items"))
    if not items:
        return SemanticQuarantineIssue(
            reason_code="shopping_order_items_left_as_opaque_ids",
            round_idx=round_idx,
            user_evidence=query,
            trace_evidence={
                "order_id": str(order.get("order_id") or ""),
                "missing_product_ids": [],
                "known_names_by_id": {},
                "terminal_text": terminal_text,
                "missing_order_item_observation": True,
            },
            hard_gate=False,
        )
    names_by_id = _product_names_by_id(history)
    normalized_terminal = _normalize(terminal_text)
    missing_ids: list[str] = []
    for raw_item in items:
        product_id = _normalize(_json_dict(raw_item).get("product_id"))
        names = names_by_id.get(product_id, set())
        if not names or not any(name in normalized_terminal for name in names):
            missing_ids.append(product_id)
    if not missing_ids:
        return None
    return SemanticQuarantineIssue(
        reason_code="shopping_order_items_left_as_opaque_ids",
        round_idx=round_idx,
        user_evidence=query,
        trace_evidence={
            "order_id": str(order.get("order_id") or ""),
            "missing_product_ids": missing_ids,
            "known_names_by_id": {
                product_id: sorted(names)
                for product_id, names in names_by_id.items()
            },
            "terminal_text": terminal_text,
        },
        hard_gate=False,
    )

def _shopping_terminal_surface_issue(
    round_trace: dict[str, Any],
    *,
    round_idx: int,
    tool_names: set[str],
) -> SemanticQuarantineIssue | None:
    """Soft terminal-naturalness diagnostic (shopping layer).

    Cross-domain terminal *contract* violations (final_answer requesting user
    input, terminal leaking an internal tool name) run in semantic_core's
    Layer 0 for every domain.  This shopping-only rule keeps the subjective
    meta-analysis check as a soft diagnostic: it never drops a row, per
    local semantic gate contract (naturalness is grayscale audit, not a hard gate).
    """
    action, terminal_text = _terminal(round_trace)
    if action and _META_TERMINAL_RE.search(terminal_text):
        return SemanticQuarantineIssue(
            reason_code="terminal_meta_user_analysis",
            round_idx=round_idx,
            user_evidence=str(round_trace.get("user_query") or ""),
            trace_evidence={"terminal_text": terminal_text},
            hard_gate=False,
        )
    return None

def _shopping_terminal_product_id_issue(
    round_trace: dict[str, Any],
    *,
    round_idx: int,
) -> SemanticQuarantineIssue | None:
    _, terminal_text = _terminal(round_trace)
    opaque_product_ids = sorted(set(
        match.group(0) for match in _OPAQUE_PRODUCT_ID_RE.finditer(terminal_text)
    ))
    if not opaque_product_ids:
        return None
    return SemanticQuarantineIssue(
        reason_code="terminal_exposes_opaque_product_id",
        round_idx=round_idx,
        user_evidence=str(round_trace.get("user_query") or ""),
        trace_evidence={
            "terminal_text": terminal_text,
            "opaque_product_ids": opaque_product_ids,
        },
        hard_gate=False,
    )

def _shopping_order_product_resolution_issue(
    round_trace: dict[str, Any],
    *,
    round_idx: int,
    visible_tool_names: set[str],
) -> SemanticQuarantineIssue | None:
    query = str(round_trace.get("user_query") or round_trace.get("query") or "")
    if "get_product" not in visible_tool_names or not re.search(
        r"\border\b", query, re.IGNORECASE,
    ):
        return None
    mentioned_subtypes = {
        subtype
        for subtype, pattern in _SHOPPING_EXACT_SUBTYPE_PATTERNS.items()
        if pattern.search(query)
    }
    if len(mentioned_subtypes) != 1:
        return None
    subtype = next(iter(mentioned_subtypes))
    history = _successful_history(round_trace)
    names_by_id = _product_names_by_id(history)
    normalized_query = _normalize(query)
    if any(
        _record_matches_subtype({"product_id": product_id, "name": name}, subtype)
        and name in normalized_query
        for product_id, names in names_by_id.items()
        for name in names
    ):
        return None
    for _, product_ids in _order_memberships(history):
        unresolved = product_ids - set(names_by_id)
        if unresolved:
            return SemanticQuarantineIssue(
                reason_code="shopping_terminal_skips_order_product_resolution",
                round_idx=round_idx,
                user_evidence=query,
                trace_evidence={
                    "requested_subtype": subtype,
                    "unresolved_product_ids": sorted(unresolved),
                },
                hard_gate=False,
            )
    return None

def _shopping_review_evidence_issue(
    rounds: list[dict[str, Any]],
) -> SemanticQuarantineIssue | None:
    reviewed_product_ids: set[str] = set()
    for position, round_trace in enumerate(rounds):
        history = _successful_history(round_trace)
        for event in history:
            if event.get("tool_name") == "get_reviews":
                product_id = str(
                    _json_dict(event.get("observation")).get("product_id")
                    or _json_dict(event.get("arguments")).get("product_id")
                    or ""
                )
                if product_id:
                    reviewed_product_ids.add(product_id)
        query = str(round_trace.get("user_query") or round_trace.get("query") or "")
        action, terminal_text = _terminal(round_trace)
        if action != "final_answer" or not _REVIEW_AVAILABILITY_QUERY_RE.search(query):
            continue
        candidate_records: dict[str, dict[str, Any]] = {}
        for event in history:
            if event.get("tool_name") != "search_products":
                continue
            for raw in _json_list(_json_dict(event.get("observation")).get("products")):
                record = _json_dict(raw)
                product_id = str(record.get("product_id") or "")
                if product_id:
                    candidate_records[product_id] = record
        review_sentences = [
            sentence
            for sentence in re.split(r"(?<=[.!?])\s+|\n+", terminal_text)
            if re.search(r"\b(?:reviews?|ratings?)\b", sentence, re.IGNORECASE)
        ]
        claimed_ids: set[str] = set()
        for sentence in review_sentences:
            claimed_ids.update(_review_selector_candidate_ids(
                query=query,
                terminal_sentence=sentence,
                candidate_records=candidate_records,
            ))
        uncovered = claimed_ids - reviewed_product_ids
        if uncovered:
            return SemanticQuarantineIssue(
                reason_code="shopping_review_claim_without_product_evidence",
                round_idx=int(round_trace.get("round_idx", position)),
                user_evidence=query,
                trace_evidence={
                    "unreviewed_candidate_ids": sorted(uncovered),
                    "reviewed_product_ids": sorted(reviewed_product_ids),
                    "terminal_text": terminal_text,
                },
            )
    return None

def _shopping_resolved_order_focus_clarification_issue(
    rounds: list[dict[str, Any]],
) -> SemanticQuarantineIssue | None:
    """Reject re-asking which order after the prior reply fixed one focus."""
    for position in range(1, len(rounds)):
        current = rounds[position]
        query = str(current.get("user_query") or current.get("query") or "")
        if not re.search(
            r"\b(?:this|that|the|same)\s+order\b|\btrack\s+(?:it|this|that)\b",
            query,
            re.IGNORECASE,
        ):
            continue
        action, terminal_text = _terminal(current)
        if action != "ask_clarification":
            continue
        previous = rounds[position - 1]
        previous_query = str(
            previous.get("user_query") or previous.get("query") or ""
        )
        _, previous_terminal = _terminal(previous)
        prior_order_ids = {
            match.group(0).casefold()
            for text in (previous_query, previous_terminal)
            for match in _OPAQUE_ORDER_ID_RE.finditer(text)
        }
        if len(prior_order_ids) != 1:
            continue
        return SemanticQuarantineIssue(
            reason_code="shopping_reasks_resolved_order_focus",
            round_idx=int(current.get("round_idx", position)),
            user_evidence=query,
            trace_evidence={
                "focused_order_id": next(iter(prior_order_ids)),
                "previous_terminal": previous_terminal,
                "clarification": terminal_text,
            },
        )
    return None

def _shopping_unsolicited_outcome_issue(
    rounds: list[dict[str, Any]],
) -> SemanticQuarantineIssue | None:
    """Reject oracle tool calls for outcomes the user never requested.

    A successful conversation round that includes add_to_cart, checkout,
    get_recommendations, or get_reviews must have a corresponding request
    in the user query.  This is a local corpus-quality gate, not a PROVE
    paper hard gate.
    """
    for position, round_trace in enumerate(rounds):
        query = str(round_trace.get("user_query") or round_trace.get("query") or "")
        history = _successful_history(round_trace)
        for event in history:
            tool_name = str(event.get("tool_name") or "")
            # Mutating calls: query must express a cart/purchase intent
            if tool_name in _UNSOLICITED_MUTATIONS:
                if not _CART_PURCHASE_REQUEST_RE.search(query):
                    return SemanticQuarantineIssue(
                        reason_code="shopping_unsolicited_outcome",
                        round_idx=int(round_trace.get("round_idx", position)),
                        user_evidence=query,
                        trace_evidence={
                            "unsolicited_tool": tool_name,
                            "arguments": _json_dict(event.get("arguments")),
                        },
                        hard_gate=False,
                    )
            # Read calls: query must express a recommendation/review intent
            if tool_name == "get_recommendations":
                if not _RECOMMENDATION_REQUEST_RE.search(query):
                    return SemanticQuarantineIssue(
                        reason_code="shopping_unsolicited_outcome",
                        round_idx=int(round_trace.get("round_idx", position)),
                        user_evidence=query,
                        trace_evidence={
                            "unsolicited_tool": tool_name,
                            "arguments": _json_dict(event.get("arguments")),
                        },
                    )
            if tool_name == "get_reviews":
                if not _REVIEW_AVAILABILITY_QUERY_RE.search(query):
                    return SemanticQuarantineIssue(
                        reason_code="shopping_unsolicited_outcome",
                        round_idx=int(round_trace.get("round_idx", position)),
                        user_evidence=query,
                        trace_evidence={
                            "unsolicited_tool": tool_name,
                            "arguments": _json_dict(event.get("arguments")),
                        },
                    )
    return None
