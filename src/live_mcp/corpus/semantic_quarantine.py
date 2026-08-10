"""Deterministic post-generation semantic quarantine dispatcher."""

from __future__ import annotations

from typing import Any

from src.live_mcp.corpus.semantic_core import (
    SemanticQuarantineIssue,
    _json_list,
    _json_dict,
    _normalize,
    _rounds,
    _initial_query_backend_id_issue,
    _successful_history,
    _successful_trace_source_target_issue,
    _terminal,
    _assistant_role_user_query_issue,
    _requested_recommendation_subtype,
    _record_matches_subtype,
    _terminal_positively_presents_record,
    _terminal_presents_record_without_rejection,
    _order_memberships,
    _observed_product_names,
    _relevant_item_tools,
    _event_product_ids,
    _call_signature,
    _product_names_by_id,
    _product_stock_by_id,
    _product_records_by_id,
    _review_selector_candidate_ids,
    _terminal_final_answer_requests_user_input_issue,
    _terminal_exposes_private_tool_name_issue,
    _missing_function_nonprefix_mutation_issue,
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
    _confirmation_claims_fresh_verification_issue,
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

def evaluate_semantic_quarantine(
    extra: dict[str, Any],
) -> SemanticQuarantineIssue | None:
    """Return the first deterministic semantic contradiction in one row.

    Layered per OVAL-MCP.md §5 — semantic quarantine only rejects provable
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

    # ── Layer 0: cross-domain provable contradictions ──
    backend_id_issue = _initial_query_backend_id_issue(rounds)
    if backend_id_issue is not None:
        return backend_id_issue
    role_issue = _assistant_role_user_query_issue(rounds)
    if role_issue is not None:
        return role_issue
    source_target_issue = _successful_trace_source_target_issue(extra, rounds)
    if source_target_issue is not None:
        return source_target_issue
    confirmation_issue = _confirmation_claims_fresh_verification_issue(rounds)
    if confirmation_issue is not None:
        return confirmation_issue
    missing_function_mutation_issue = _missing_function_nonprefix_mutation_issue(
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
    if str(extra.get("domain") or "") != "shopping":
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
            return issue
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
            return surface_issue
        product_resolution_issue = _shopping_order_product_resolution_issue(
            round_trace,
            round_idx=round_idx,
            visible_tool_names=visible_tool_names,
        )
        if product_resolution_issue is not None:
            return product_resolution_issue
        unavailable_issue = _shopping_unresolvable_email_clarification_issue(
            round_trace,
            round_idx=round_idx,
            visible_tool_names=visible_tool_names,
        )
        if unavailable_issue is not None:
            return unavailable_issue
        skipped_add_issue = _shopping_skipped_explicit_product_mutation_issue(
            round_trace,
            round_idx=round_idx,
            visible_tool_names=visible_tool_names,
        )
        if skipped_add_issue is not None:
            return skipped_add_issue
        for check in checks:
            issue = check(round_trace, round_idx=round_idx)
            if issue is not None:
                return issue
        order_items_issue = _shopping_order_items_name_coverage_issue(
            round_trace,
            round_idx=round_idx,
        )
        if order_items_issue is not None:
            return order_items_issue
        product_id_issue = _shopping_terminal_product_id_issue(
            round_trace,
            round_idx=round_idx,
        )
        if product_id_issue is not None:
            return product_id_issue
    # Last-resort catch: reject tool calls for outcomes the user never requested.
    # Runs after all domain-specific checks so that targeted rules (accessory,
    # relational, delegated purchase, etc.) take priority.
    unsolicited_issue = _shopping_unsolicited_outcome_issue(rounds)
    if unsolicited_issue is not None:
        return unsolicited_issue
    return None
