"""Shopping selection and recommendation semantic gates."""

from __future__ import annotations


import re


from typing import Any

from src.live_mcp.corpus.semantic_core import (

    SemanticQuarantineIssue,

    _event_product_ids,

    _json_dict,

    _json_list,

    _normalize,

    _observed_product_names,

    _order_memberships,

    _product_records_by_id,

    _record_matches_subtype,

    _relevant_item_tools,

    _requested_recommendation_subtype,

    _successful_history,

    _terminal,

    _terminal_positively_presents_record,

    _terminal_presents_record_without_rejection,

)

_ACCESSORY_GOAL_RE = re.compile(
    r"\baccessor(?:y|ies)\b[^.!?\n]{0,56}"
    r"\b(?:for|with|to\s+go\s+with|based\s+on)\b"
    r"|\b(?:for|with|to\s+go\s+with|based\s+on)\b"
    r"[^.!?\n]{0,56}\baccessor(?:y|ies)\b",
    re.IGNORECASE,
)

_AMBIGUOUS_SELECTION_RE = re.compile(
    r"\b(?:i(?:'ll|\s+will|\s+would)\s+(?:just\s+)?(?:go|stick)\s+with|"
    r"keep\b[^.!?\n]{0,80}\b(?:in|on)\s+(?:my|the)\s+"
    r"(?:cart|wishlist|wish\s+list))\b",
    re.IGNORECASE,
)

_CART_PURCHASE_REQUEST_RE = re.compile(
    r"add\s+(?:\w[\w\s]*?)\s+to\s+(?:my\s+)?cart"
    r"|add\s+to\s+(?:my\s+)?cart"
    r"|\b(?:buy|purchas\w*|check\s*out|checkout|place\s+(?:an\s+)?order)\b"
    r"|\b(?:i(?:'ll|\s+will|\s+would)?\s+(?:take|get|grab|go\s+with)|"
    r"let\s+me\s+(?:get|have|grab)|i(?:'d|\s+would)\s+like\s+to\s+(?:get|buy|order|purchase))"
    r"|\b(?:i\s+want\s+(?:to\s+(?:get|buy|order|purchase)\s+)?(?:the|a|an|this|that|those|these|one|some))"
    r"|\b(?:order|get\s+me|give\s+me|i\s+need)\s+(?:the|a|an|this|that)",
    re.IGNORECASE,
)

_DELEGATED_PRODUCT_CHOICE_RE = re.compile(
    r"(?:"
    r"\b(?:one|any)\s+of\s+(?:them|those|the\s+(?:options|recommendations))\b"
    r"|\b(?:whichever|whatever|your\s+choice)\b"
    r"|\b(?:pick|choose|select)\b[^.!?\n]{0,48}"
    r"\b(?:one|any|best|you\s+(?:like|prefer|recommend))\b"
    r"|\b(?:add|buy|purchase|order)\b[^.!?\n]{0,48}"
    r"\b(?:the\s+)?best\s+(?:one|option|item|product)\b"
    r"|\bif\b[^.!?\n]{0,64}\b(?:looks?|seems?)\s+"
    r"(?:good|great|best|nice|suitable)\b"
    r"|\bif\b[^.!?\n]{0,64}\bfind\b[^.!?\n]{0,32}"
    r"\b(?:good|great|best|nice|suitable)\b"
    r"|\bif\b[^.!?\n]{0,64}\b(?:you(?:'ll|\s+will)\s+|i(?:'ll|\s+will|\s+might)\s+)?"
    r"(?:like|love|enjoy)\b"
    r")",
    re.IGNORECASE,
)

_DETERMINISTIC_RECOMMENDATION_SELECTOR_RE = re.compile(
    r"\b(?:first|top)\s+(?:one|item|product|option|recommendation)\b"
    r"|\b(?:cheapest|lowest[-\s]?priced|most\s+expensive|highest[-\s]?priced)\b"
    r"|\b(?:best|highest)[-\s]?rated(?:\s+(?:one|item|product|option))?\b",
    re.IGNORECASE,
)

_EMAIL_ADDRESS_CLARIFICATION_RE = re.compile(
    r"\b(?:email\s+address|which\s+email|what\s+email|"
    r"where[^.!?\n]{0,30}\bemail)\b",
    re.I,
)

_EMAIL_SEND_INTENT_RE = re.compile(
    r"(?:"
    r"\b(?:send|email|mail)\b[^.!?\n]{0,80}\b(?:detail|details|"
    r"information|info|receipt|summary)\b"
    r"|\b(?:detail|details|information|info|receipt|summary)\b"
    r"[^.!?\n]{0,80}\b(?:to|via)\b[^.!?\n]{0,30}\b(?:email|mail)\b"
    r")",
    re.I,
)

_EXPLICIT_ACTION_RE = re.compile(
    r"\b(?:add|put|buy|purchase|checkout|check\s+out|remove|delete|"
    r"return|refund|review|rate|compare|show|find|search|recommend|suggest|"
    r"track|update|change|clear|apply)\b"
    r"|\border\s+(?:it|one|this|that|the\s+(?:item|product))\b",
    re.IGNORECASE,
)

_GENERIC_PAYMENT_VALUES = frozenset({"card", "credit card", "debit card"})

_GENERIC_SINGULAR_ORDER_ITEM_RE = re.compile(
    r"\b(?:the|that|this)\s+(?:product|item)\s+"
    r"(?:in|from|of)\s+(?:it|my|the|that|this)?"
    r"(?:\s+(?:shipped|pending|delivered|placed|returned))?"
    r"(?:\s+(?:order|purchase))?\b",
    re.I,
)

_HISTORICAL_ORDER_TARGET_RE = re.compile(
    r"(?:"
    r"\b(?:previous|past|existing|already\s+placed|placed|shipped|"
    r"delivered|completed)\s+order\b"
    r"|\bmessed\s+up\s+(?:my|the|this|that)?\s*order\b"
    r"|\border\b[^.!?\n]{0,45}\b(?:messed\s+up|already\s+placed|"
    r"shipped|delivered|completed)\b"
    r")",
    re.I,
)

_IDENTIFIER_ONLY_QUESTION_RE = re.compile(
    r"^\s*(?:(?:can|could|would)\s+you\s+)?(?:please\s+)?"
    r"(?:(?:tell|give|show)\s+me\s+|what\s+(?:is|was)\s+|which\s+is\s+)?"
    r"(?:the\s+)?(?P<kind>order|return)\s+(?:id|number)\b[^.!]*[?.!]?\s*$",
    re.IGNORECASE,
)

_ORDER_ID_RE = re.compile(r"\bord_[A-Za-z0-9_-]+\b", re.IGNORECASE)

_PRODUCT_DISCOVERY_REQUEST_RE = re.compile(
    r"\b(?:find|look(?:ing)?\s+for|search(?:ing)?\s+for)\b",
    re.IGNORECASE,
)

_PROSPECTIVE_CART_TARGET_RE = re.compile(
    r"\b(?:cart|basket|new\s+(?:cart|purchase|order)|next\s+purchase|"
    r"future\s+(?:purchase|order)|upcoming\s+(?:purchase|order)|"
    r"checkout|before\s+i\s+(?:buy|order|checkout))\b",
    re.I,
)

_RECOMMENDATION_REQUEST_RE = re.compile(
    r"\b(?:recommend|recommendation|suggest|suggestion)\w*\b",
    re.IGNORECASE,
)

_RELATION_RECOMMENDATION_RE = re.compile(
    r"\b(?:similar\s+to|based\s+on|go(?:es)?\s+well\s+with|"
    r"to\s+go\s+with)\b",
    re.IGNORECASE,
)

_RETURN_ID_RE = re.compile(r"\bret_[A-Za-z0-9_-]+\b", re.IGNORECASE)

_SHOPPING_EXACT_SUBTYPE_PATTERNS: dict[str, re.Pattern[str]] = {
    "headphones": re.compile(r"\b(?:headphones?|earphones?)\b", re.I),
    "microphone": re.compile(r"\b(?:microphones?|mics?)\b", re.I),
    "speaker": re.compile(r"\bspeakers?\b", re.I),
    "keyboard": re.compile(r"\bkeyboards?\b", re.I),
    "mouse": re.compile(r"\b(?:mouse|mice)\b", re.I),
    "webcam": re.compile(r"\bwebcams?\b", re.I),
}

def _shopping_subtype_terminal_issue(
    round_trace: dict[str, Any],
    *,
    round_idx: int,
) -> SemanticQuarantineIssue | None:
    query = str(
        round_trace.get("user_query")
        or round_trace.get("query")
        or ""
    )
    subtype = _requested_recommendation_subtype(query)
    terminal_action, terminal_text = _terminal(round_trace)
    if not subtype or terminal_action != "final_answer":
        return None
    for event in _successful_history(round_trace):
        if event.get("tool_name") != "get_recommendations":
            continue
        observation = _json_dict(event.get("observation"))
        for raw_record in _json_list(observation.get("recommendations")):
            record = _json_dict(raw_record)
            if (
                record
                and not _record_matches_subtype(record, subtype)
                and _terminal_positively_presents_record(terminal_text, record)
            ):
                return SemanticQuarantineIssue(
                    reason_code="shopping_entity_subtype_terminal_mismatch",
                    round_idx=round_idx,
                    user_evidence=query,
                    trace_evidence={
                        "requested_subtype": subtype,
                        "tool_name": "get_recommendations",
                        "observed_product": record,
                        "terminal_text": terminal_text,
                    },
                )
    return None

def _shopping_accessory_role_issue(
    round_trace: dict[str, Any],
    *,
    round_idx: int,
) -> SemanticQuarantineIssue | None:
    """Reject presenting another source-type product as its accessory."""
    query = str(round_trace.get("user_query") or round_trace.get("query") or "")
    terminal_action, terminal_text = _terminal(round_trace)
    if terminal_action != "final_answer" or not _ACCESSORY_GOAL_RE.search(query):
        return None
    history = _successful_history(round_trace)
    records = _product_records_by_id(history)
    query_subtypes = {
        subtype
        for subtype, pattern in _SHOPPING_EXACT_SUBTYPE_PATTERNS.items()
        if pattern.search(query)
    }
    for event in history:
        if event.get("tool_name") != "get_recommendations":
            continue
        seed_id = _normalize(
            _json_dict(event.get("arguments")).get("based_on_product")
        )
        seed_record = records.get(seed_id, {})
        source_subtypes = {
            subtype
            for subtype in _SHOPPING_EXACT_SUBTYPE_PATTERNS
            if seed_record and _record_matches_subtype(seed_record, subtype)
        }
        # A multi-product order lookup may expose only product IDs before the
        # recommendation calls.  In that case, retain the user-visible source
        # types as factual fallback rather than silently skipping the role
        # check.  A mismatch is reported only when a recommendation itself has
        # one of those same observable types.
        candidate_source_subtypes = source_subtypes or query_subtypes
        if not candidate_source_subtypes:
            continue
        observation = _json_dict(event.get("observation"))
        for raw_record in _json_list(observation.get("recommendations")):
            record = _json_dict(raw_record)
            matching_source_subtypes = {
                subtype
                for subtype in candidate_source_subtypes
                if record and _record_matches_subtype(record, subtype)
            }
            if matching_source_subtypes and _terminal_presents_record_without_rejection(
                terminal_text, record
            ):
                return SemanticQuarantineIssue(
                    reason_code="shopping_accessory_entity_role_mismatch",
                    round_idx=round_idx,
                    user_evidence=query,
                    trace_evidence={
                        "source_subtype": sorted(matching_source_subtypes)[0],
                        "observed_product": record,
                        "terminal_text": terminal_text,
                    },
                )
    return None

def _shopping_coupon_resource_issue(
    round_trace: dict[str, Any],
    *,
    round_idx: int,
) -> SemanticQuarantineIssue | None:
    query = str(
        round_trace.get("user_query")
        or round_trace.get("query")
        or ""
    )
    if (
        not _HISTORICAL_ORDER_TARGET_RE.search(query)
        or _PROSPECTIVE_CART_TARGET_RE.search(query)
    ):
        return None
    for event in _successful_history(round_trace):
        if (
            event.get("tool_name") == "apply_coupon"
            and event.get("state_changed") is True
        ):
            return SemanticQuarantineIssue(
                reason_code="shopping_coupon_bound_to_historical_order",
                round_idx=round_idx,
                user_evidence=query,
                trace_evidence={
                    "tool_name": "apply_coupon",
                    "arguments": _json_dict(event.get("arguments")),
                    "observation": _json_dict(event.get("observation")),
                    "state_delta_paths": _json_list(
                        event.get("state_delta_paths")
                    ),
                    "actual_resource": "shopping_cart",
                    "requested_resource": "historical_order",
                },
            )
    return None

def _shopping_singular_order_item_issue(
    round_trace: dict[str, Any],
    *,
    round_idx: int,
) -> SemanticQuarantineIssue | None:
    # Initial-turn generic singularity is directly decidable.  A later pronoun
    # may be grounded by prior conversation, so it is intentionally left for a
    # future conversation-reference check rather than over-filtered here.
    if round_idx != 0:
        return None
    query = str(
        round_trace.get("user_query")
        or round_trace.get("query")
        or ""
    )
    selector = _GENERIC_SINGULAR_ORDER_ITEM_RE.search(query)
    relevant_tools = _relevant_item_tools(query)
    terminal_action, terminal_text = _terminal(round_trace)
    if (
        not selector
        or not re.search(r"\b(?:order|purchase)\b", query, re.I)
        or not relevant_tools
        or terminal_action != "final_answer"
    ):
        return None
    history = _successful_history(round_trace)
    memberships = _order_memberships(history)
    if not memberships:
        return None

    normalized_query = _normalize(query)
    if any(
        name and name in normalized_query
        for name in _observed_product_names(history)
    ):
        return None
    query_ids = set(re.findall(r"\bprd_[a-z0-9_]+\b", query, re.I))
    if query_ids:
        return None

    for order_id, member_ids in memberships:
        relevant_events = [
            event
            for event in history
            if event.get("tool_name") in relevant_tools
            and _event_product_ids(event) & member_ids
        ]
        if relevant_events:
            return SemanticQuarantineIssue(
                reason_code="shopping_ambiguous_singular_order_item",
                round_idx=round_idx,
                user_evidence=selector.group(0),
                trace_evidence={
                    "query": query,
                    "order_id": order_id,
                    "order_product_ids": sorted(member_ids),
                    "item_level_calls": [
                        {
                            "tool_name": str(event.get("tool_name") or ""),
                            "arguments": _json_dict(event.get("arguments")),
                        }
                        for event in relevant_events
                    ],
                    "terminal_text": terminal_text,
                },
                hard_gate=False,
            )
    return None

def _shopping_unresolvable_email_clarification_issue(
    round_trace: dict[str, Any],
    *,
    round_idx: int,
    visible_tool_names: set[str],
) -> SemanticQuarantineIssue | None:
    query = str(
        round_trace.get("user_query")
        or round_trace.get("query")
        or ""
    )
    terminal_action, terminal_text = _terminal(round_trace)
    if (
        terminal_action != "ask_clarification"
        or not _EMAIL_SEND_INTENT_RE.search(query)
        or not _EMAIL_ADDRESS_CLARIFICATION_RE.search(terminal_text)
        or "send_email" in visible_tool_names
    ):
        return None
    return SemanticQuarantineIssue(
        reason_code="shopping_clarification_cannot_enable_email_send",
        round_idx=round_idx,
        user_evidence=query,
        trace_evidence={
            "terminal_text": terminal_text,
            "required_capability": "send_email",
            "visible_tool_names": sorted(visible_tool_names),
        },
        hard_gate=False,
    )

def _shopping_ambiguous_selection_issue(
    rounds: list[dict[str, Any]],
) -> SemanticQuarantineIssue | None:
    for position, round_trace in enumerate(rounds[1:], start=1):
        query = str(round_trace.get("user_query") or round_trace.get("query") or "")
        if not (
            _AMBIGUOUS_SELECTION_RE.search(query)
            and not _EXPLICIT_ACTION_RE.search(query)
        ):
            continue
        action, terminal_text = _terminal(round_trace)
        return SemanticQuarantineIssue(
            reason_code="shopping_ambiguous_selection_without_action",
            round_idx=int(round_trace.get("round_idx", position)),
            user_evidence=query,
            trace_evidence={
                "tool_calls": [
                    str(event.get("tool_name") or "")
                    for event in _successful_history(round_trace)
                ],
                "terminal_action": action,
                "terminal_text": terminal_text,
            },
            hard_gate=False,
        )
    return None

def _shopping_delegated_recommendation_purchase_issue(
    rounds: list[dict[str, Any]],
) -> SemanticQuarantineIssue | None:
    """Reject a consequential purchase chosen by the Teacher subjectively.

    This check requires a completed factual pattern: a user recommendation
    request with non-deterministic delegated-choice language, a real
    recommendation observation, and a later successful add of one of those
    recommendation IDs.  Merely asking for recommendations is never enough to
    trigger quarantine.
    """
    for position, round_trace in enumerate(rounds):
        query = str(
            round_trace.get("user_query") or round_trace.get("query") or ""
        )
        delegated_choice = bool(_DELEGATED_PRODUCT_CHOICE_RE.search(query))
        deterministic_selector = _DETERMINISTIC_RECOMMENDATION_SELECTOR_RE.search(
            query
        )
        if not (
            (
                _RECOMMENDATION_REQUEST_RE.search(query)
                or _PRODUCT_DISCOVERY_REQUEST_RE.search(query)
            )
            and (delegated_choice or deterministic_selector)
        ):
            continue
        # When the user explicitly authorizes a cart/checkout action
        # alongside a delegated recommendation choice (e.g. "If you
        # find a good pair, please add them to my cart"), the purchase
        # is user-authorized and delegation is not a quality issue.
        if delegated_choice and _CART_PURCHASE_REQUEST_RE.search(query):
            continue
        recommendation_records: dict[str, dict[str, Any]] = {}
        observed_ratings: dict[str, float] = {}
        selected_ids_by_action: dict[str, set[str]] = {
            "add_to_cart": set(),
            "add_to_wishlist": set(),
        }
        for event in _successful_history(round_trace):
            tool_name = str(event.get("tool_name") or "")
            if tool_name == "get_recommendations":
                observation = _json_dict(event.get("observation"))
                for raw in _json_list(observation.get("recommendations")):
                    record = _json_dict(raw)
                    product_id = _normalize(record.get("product_id"))
                    if product_id:
                        recommendation_records[product_id] = record
            elif tool_name == "search_products":
                observation = _json_dict(event.get("observation"))
                for raw in _json_list(observation.get("products")):
                    record = _json_dict(raw)
                    product_id = _normalize(record.get("product_id"))
                    if product_id:
                        recommendation_records[product_id] = record
            elif tool_name == "get_reviews":
                observation = _json_dict(event.get("observation"))
                product_id = _normalize(observation.get("product_id"))
                count = observation.get("count")
                average_rating = observation.get("average_rating")
                if (
                    product_id
                    and isinstance(count, int)
                    and count > 0
                    and isinstance(average_rating, (int, float))
                ):
                    observed_ratings[product_id] = float(average_rating)
            elif tool_name in selected_ids_by_action:
                product_id = _normalize(
                    _json_dict(event.get("arguments")).get("product_id")
                )
                if product_id:
                    selected_ids_by_action[tool_name].add(product_id)
        selected_ids = set().union(*selected_ids_by_action.values()) & set(
            recommendation_records
        )
        if not selected_ids:
            continue
        normalized_query = _normalize(query)
        named_selection = {
            product_id
            for product_id in selected_ids
            if (
                (name := _normalize(
                    recommendation_records[product_id].get("name")
                ))
                and name in normalized_query
            )
        }
        # A catalog name can also be identical to the requested subtype (for
        # example "Noise Canceling Headphones").  Its incidental presence in
        # a discovery query does not resolve an explicit subjective condition
        # such as "if you find a good pair".  Only use name grounding to retain
        # rows when no delegated-choice condition is present.
        if selected_ids <= named_selection and not delegated_choice:
            continue
        expected_ids: set[str] = set()
        if deterministic_selector:
            ordered_ids = list(recommendation_records)
            selector_text = deterministic_selector.group(0).casefold()
            if re.search(r"\b(?:first|top)\b", selector_text):
                expected_ids = set(ordered_ids[:1])
            elif re.search(r"\b(?:best|highest)[-\s]?rated\b", selector_text):
                rated = {
                    product_id: rating
                    for product_id, rating in observed_ratings.items()
                    if product_id in recommendation_records
                }
                if rated:
                    target_rating = max(rated.values())
                    expected_ids = {
                        product_id
                        for product_id, rating in rated.items()
                        if rating == target_rating
                    }
            else:
                priced = [
                    (product_id, record.get("price"))
                    for product_id, record in recommendation_records.items()
                    if isinstance(record.get("price"), (int, float))
                ]
                if len(priced) == len(recommendation_records) and priced:
                    choose_max = bool(re.search(
                        r"most\s+expensive|highest", selector_text
                    ))
                    target_price = (
                        max(price for _, price in priced)
                        if choose_max
                        else min(price for _, price in priced)
                    )
                    expected_ids = {
                        product_id
                        for product_id, price in priced
                        if price == target_price
                    }
            if len(expected_ids) == 1 and selected_ids <= expected_ids:
                continue
            return SemanticQuarantineIssue(
                reason_code="shopping_recommendation_selector_mismatch",
                round_idx=int(round_trace.get("round_idx", position)),
                user_evidence=query,
                trace_evidence={
                    "selector": deterministic_selector.group(0),
                    "expected_product_ids": sorted(expected_ids),
                    "purchased_recommendation_ids": sorted(selected_ids),
                },
                hard_gate=False,
            )
        if not delegated_choice:
            continue
        selected_actions = sorted(
            action
            for action, action_ids in selected_ids_by_action.items()
            if action_ids & selected_ids
        )
        return SemanticQuarantineIssue(
            reason_code=(
                "shopping_delegated_recommendation_purchase"
                if "add_to_cart" in selected_actions
                else "shopping_delegated_recommendation_selection"
            ),
            round_idx=int(round_trace.get("round_idx", position)),
            user_evidence=query,
            trace_evidence={
                "recommended_product_ids": sorted(recommendation_records),
                "purchased_recommendation_ids": sorted(selected_ids),
                "purchased_product_names": sorted(
                    str(recommendation_records[item].get("name") or "")
                    for item in selected_ids
                ),
                "selection_actions": selected_actions,
            },
            hard_gate=False,
        )
    return None

def _shopping_redundant_visible_identifier_issue(
    rounds: list[dict[str, Any]],
) -> SemanticQuarantineIssue | None:
    visible_ids: dict[str, set[str]] = {"order": set(), "return": set()}
    for position, round_trace in enumerate(rounds):
        query = str(round_trace.get("user_query") or round_trace.get("query") or "")
        match = _IDENTIFIER_ONLY_QUESTION_RE.match(query)
        kind = match.group("kind").casefold() if match else ""
        if position > 0 and kind and len(visible_ids[kind]) == 1:
            return SemanticQuarantineIssue(
                reason_code=f"shopping_reasks_already_visible_{kind}_id",
                round_idx=int(round_trace.get("round_idx", position)),
                user_evidence=query,
                trace_evidence={
                    f"already_visible_{kind}_id": next(iter(visible_ids[kind])),
                    "tool_calls": [
                        str(event.get("tool_name") or "")
                        for event in _successful_history(round_trace)
                    ],
                },
                hard_gate=False,
            )
        _, terminal_text = _terminal(round_trace)
        visible_ids["order"].update(
            match.group(0).casefold()
            for match in _ORDER_ID_RE.finditer(terminal_text)
        )
        visible_ids["return"].update(
            match.group(0).casefold()
            for match in _RETURN_ID_RE.finditer(terminal_text)
        )
    return None

def _shopping_generic_payment_issue(
    round_trace: dict[str, Any],
    *,
    round_idx: int,
) -> SemanticQuarantineIssue | None:
    for event in _successful_history(round_trace):
        if event.get("tool_name") != "checkout":
            continue
        payment = _normalize(_json_dict(event.get("arguments")).get("payment_method"))
        if payment in _GENERIC_PAYMENT_VALUES:
            return SemanticQuarantineIssue(
                reason_code="shopping_checkout_generic_payment_method",
                round_idx=round_idx,
                user_evidence=str(round_trace.get("user_query") or round_trace.get("query") or ""),
                trace_evidence={
                    "payment_method": payment,
                    "observation": _json_dict(event.get("observation")),
                },
                hard_gate=False,
            )
    return None

def _shopping_relational_recommendation_issue(
    rounds: list[dict[str, Any]],
) -> SemanticQuarantineIssue | None:
    observed_product_ids: set[str] = set()
    prior_relation = False

    def observe(event: dict[str, Any]) -> None:
        def visit(value: Any) -> None:
            if isinstance(value, dict):
                product_id = _normalize(value.get("product_id"))
                if product_id:
                    observed_product_ids.add(product_id)
                for child in value.values():
                    visit(child)
            elif isinstance(value, list):
                for child in value:
                    visit(child)

        visit(_json_dict(event.get("observation")))

    for position, round_trace in enumerate(rounds):
        query = str(round_trace.get("user_query") or round_trace.get("query") or "")
        current_relation = bool(_RELATION_RECOMMENDATION_RE.search(query))
        anaphoric_followup = bool(re.search(
            r"\b(?:any\s+)?(?:other|more|another|else)\b",
            query,
            re.IGNORECASE,
        ))
        relational = current_relation or (prior_relation and anaphoric_followup)
        grounded_relational_call_seen = False
        for event in _successful_history(round_trace):
            if event.get("tool_name") == "get_recommendations" and relational:
                seed_id = _normalize(
                    _json_dict(event.get("arguments")).get("based_on_product")
                )
                user_exposes_seed = bool(
                    seed_id
                    and re.search(
                        rf"(?<![A-Za-z0-9_-]){re.escape(seed_id)}"
                        rf"(?![A-Za-z0-9_-])",
                        query,
                        re.IGNORECASE,
                    )
                )
                seed_is_grounded = bool(
                    seed_id
                    and (
                        seed_id in observed_product_ids
                        or user_exposes_seed
                    )
                )
                if not seed_is_grounded and not grounded_relational_call_seen:
                    return SemanticQuarantineIssue(
                        reason_code=(
                            "shopping_relational_recommendation_without_grounded_seed"
                        ),
                        round_idx=int(round_trace.get("round_idx", position)),
                        user_evidence=query,
                        trace_evidence={
                            "arguments": _json_dict(event.get("arguments")),
                            "observed_product_ids_before_call": sorted(
                                observed_product_ids
                            ),
                            "observation": _json_dict(event.get("observation")),
                        },
                        hard_gate=False,
                    )
                grounded_relational_call_seen = (
                    grounded_relational_call_seen or seed_is_grounded
                )
            observe(event)
        prior_relation = prior_relation or current_relation
    return None
