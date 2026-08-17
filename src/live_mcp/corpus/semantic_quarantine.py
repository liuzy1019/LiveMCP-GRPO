"""Deterministic post-generation semantic quarantine dispatcher."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from src.live_mcp.corpus.semantic_core import (
    SemanticQuarantineIssue,
    _json_list,
    _rounds,
    _user_visible_backend_id_issue,
    _irrelevance_capability_issue,
    _assistant_role_user_query_issue,
    _terminal_final_answer_requests_user_input_issue,
    _terminal_exposes_private_tool_name_issue,
    _missing_function_contract_issue,
)
from src.live_mcp.corpus.semantic_shopping_selection import (
    _shopping_subtype_terminal_issue,
    _shopping_accessory_role_issue,
    _shopping_coupon_resource_issue,
    _shopping_singular_order_item_issue,
    _shopping_unresolvable_email_clarification_issue,
    _shopping_ambiguous_selection_issue,
    _shopping_delegated_recommendation_purchase_issue,
    _shopping_redundant_visible_identifier_issue,
    _shopping_generic_payment_issue,
    _shopping_relational_recommendation_issue,
)
from src.live_mcp.corpus.semantic_shopping_lifecycle import (
    _shopping_repeated_lifecycle_read_issue,
    _shopping_social_only_continuation_issue,
    _shopping_order_status_filter_issue,
    _shopping_skipped_explicit_product_mutation_issue,
    _shopping_order_items_name_coverage_issue,
    _shopping_terminal_surface_issue,
    _shopping_terminal_product_id_issue,
    _shopping_order_product_resolution_issue,
    _shopping_review_evidence_issue,
    _shopping_resolved_order_focus_clarification_issue,
    _shopping_unsolicited_outcome_issue,
)
from src.live_mcp.corpus.semantic_calendar import (
    calendar_external_side_effect_claim_issue,
)
from src.live_mcp.corpus.semantic_issue_tracker import (
    issue_tracker_bounded_set_terminal_issue,
)

def evaluate_semantic_quarantine(
    extra: dict[str, Any],
) -> SemanticQuarantineIssue | None:
    """Return the first deterministic semantic contradiction in one row.

    Under the local semantic gate contract, quarantine only rejects provable
    contradictions; subjective naturalness is left to grayscale audit:

      Layer 0 — cross-domain provable contradictions, all domains run:
                backend-id leak, assistant-role-as-query, missing source
                target, fresh-verification-without-evidence, final_answer
                requesting user input, terminal leaking an internal tool name.
      Layer 1 — domain-specific rules (shopping today).  Rules that judge
                subjective naturalness carry hard_gate=False and are recorded
                as diagnostics; they never drop a row.
    """
    rounds = _rounds(extra)

    def diagnostic(
        issue: SemanticQuarantineIssue | None,
    ) -> SemanticQuarantineIssue | None:
        return replace(issue, hard_gate=False) if issue is not None else None

    # ── Layer 0: cross-domain provable contradictions ──
    irrelevance_issue = _irrelevance_capability_issue(extra, rounds)
    if irrelevance_issue is not None:
        return irrelevance_issue
    backend_id_issue = _user_visible_backend_id_issue(extra, rounds)
    if backend_id_issue is not None:
        return backend_id_issue
    role_issue = _assistant_role_user_query_issue(rounds)
    if role_issue is not None:
        return role_issue
    missing_function_mutation_issue = _missing_function_contract_issue(
        extra, rounds,
    )
    if missing_function_mutation_issue is not None:
        return missing_function_mutation_issue

    visible_tool_names = {
        str(value)
        for value in _json_list(extra.get("visible_tool_names"))
        if str(value or "")
    }
    hidden_tool_names = {
        str(value)
        for value in _json_list(extra.get("hidden_tools"))
        if str(value or "")
    }
    all_tool_names = visible_tool_names | hidden_tool_names
    for position, round_trace in enumerate(rounds):
        round_idx = int(round_trace.get("round_idx", position))
        input_issue = _terminal_final_answer_requests_user_input_issue(
            round_trace, round_idx=round_idx,
        )
        if input_issue is not None:
            return input_issue
        leak_issue = _terminal_exposes_private_tool_name_issue(
            round_trace,
            round_idx=round_idx,
            tool_names=all_tool_names,
            hidden_tool_names=hidden_tool_names,
        )
        if leak_issue is not None:
            return leak_issue

    # ── Layer 1: domain-specific rules ──
    domain = str(extra.get("domain") or "")
    if domain == "issue_tracker":
        return diagnostic(issue_tracker_bounded_set_terminal_issue(rounds))
    if domain == "calendar":
        for position, round_trace in enumerate(rounds):
            issue = calendar_external_side_effect_claim_issue(
                round_trace,
                round_idx=int(round_trace.get("round_idx", position)),
                evidence_rounds=rounds[:position + 1],
            )
            if issue is not None:
                return diagnostic(issue)
        return None
    if domain != "shopping":
        return None
    for conversation_check in (
        _shopping_repeated_lifecycle_read_issue,
        _shopping_social_only_continuation_issue,
        _shopping_resolved_order_focus_clarification_issue,
        _shopping_delegated_recommendation_purchase_issue,
        _shopping_ambiguous_selection_issue,
        _shopping_redundant_visible_identifier_issue,
        _shopping_order_status_filter_issue,
        _shopping_relational_recommendation_issue,
        _shopping_review_evidence_issue,
    ):
        issue = conversation_check(rounds)
        if issue is not None:
            return diagnostic(issue)
    checks = (
        _shopping_coupon_resource_issue,
        _shopping_singular_order_item_issue,
        _shopping_subtype_terminal_issue,
        _shopping_accessory_role_issue,
        _shopping_generic_payment_issue,
    )
    for position, round_trace in enumerate(rounds):
        round_idx = int(round_trace.get("round_idx", position))
        surface_issue = _shopping_terminal_surface_issue(
            round_trace,
            round_idx=round_idx,
            tool_names=visible_tool_names | hidden_tool_names,
        )
        if surface_issue is not None:
            return diagnostic(surface_issue)
        product_resolution_issue = _shopping_order_product_resolution_issue(
            round_trace,
            round_idx=round_idx,
            visible_tool_names=visible_tool_names,
        )
        if product_resolution_issue is not None:
            return diagnostic(product_resolution_issue)
        unavailable_issue = _shopping_unresolvable_email_clarification_issue(
            round_trace,
            round_idx=round_idx,
            visible_tool_names=visible_tool_names,
        )
        if unavailable_issue is not None:
            return diagnostic(unavailable_issue)
        skipped_add_issue = _shopping_skipped_explicit_product_mutation_issue(
            round_trace,
            round_idx=round_idx,
            visible_tool_names=visible_tool_names,
        )
        if skipped_add_issue is not None:
            return diagnostic(skipped_add_issue)
        for check in checks:
            issue = check(round_trace, round_idx=round_idx)
            if issue is not None:
                return diagnostic(issue)
        order_items_issue = _shopping_order_items_name_coverage_issue(
            round_trace,
            round_idx=round_idx,
        )
        if order_items_issue is not None:
            return diagnostic(order_items_issue)
        product_id_issue = _shopping_terminal_product_id_issue(
            round_trace,
            round_idx=round_idx,
        )
        if product_id_issue is not None:
            return diagnostic(product_id_issue)
    # Last-resort catch: reject tool calls for outcomes the user never requested.
    # Runs after all domain-specific checks so that targeted rules (accessory,
    # relational, delegated purchase, etc.) take priority.
    unsolicited_issue = _shopping_unsolicited_outcome_issue(rounds)
    if unsolicited_issue is not None:
        return diagnostic(unsolicited_issue)
    return None
