from __future__ import annotations

import math

from src.live_mcp.corpus.planning import (
    DOMAINS,
    _increment_quotas,
    _select_bucket,
    _select_domain,
)


def _evidence() -> dict[str, dict[str, object]]:
    tool_counts = {
        "banking": 18,
        "calendar": 17,
        "crm": 16,
        "email": 17,
        "filesystem": 40,
        "food_delivery": 17,
        "issue_tracker": 21,
        "payments": 10,
        "shopping": 23,
        "team_chat": 11,
    }
    unique_chains = {
        "banking": 26,
        "calendar": 36,
        "crm": 29,
        "email": 48,
        "filesystem": 26,
        "food_delivery": 32,
        "issue_tracker": 67,
        "payments": 13,
        "shopping": 90,
        "team_chat": 20,
    }
    missing_weights = {
        "banking": 9,
        "calendar": 7,
        "crm": 7,
        "email": 9,
        "filesystem": 20,
        "food_delivery": 8,
        "issue_tracker": 10,
        "payments": 4,
        "shopping": 11,
        "team_chat": 6,
    }
    return {
        domain: {
            "tool_count": tool_counts[domain],
            "mcp_structural_weight": math.sqrt(
                tool_counts[domain] * unique_chains[domain]
            ),
            "missing_function_weight": missing_weights[domain],
        }
        for domain in DOMAINS
    }


def test_increment_quotas_allocate_only_the_remaining_global_gap() -> None:
    quotas = _increment_quotas(
        gaps={
            "mcp_conversation": 2837,
            "missing_function": 294,
            "internal_abstention_proxy": 283,
        },
        evidence=_evidence(),
    )

    assert sum(
        value["mcp_conversation"] for value in quotas.values()
    ) == 2837
    assert sum(
        value["missing_function"] for value in quotas.values()
    ) == 294
    assert all(
        value["internal_abstention_proxy"] == 0
        for value in quotas.values()
    )
    assert quotas["shopping"]["mcp_conversation"] == 494
    assert quotas["payments"]["mcp_conversation"] == 124
    assert quotas["filesystem"]["missing_function"] == 65


def test_planner_finishes_mcp_before_missing_function() -> None:
    assert _select_bucket({
        "mcp_conversation": 1,
        "missing_function": 100,
    }) == "mcp_conversation"


def test_domain_selection_uses_allocated_bucket_gap() -> None:
    plans = {
        domain: {
            "gaps": {
                "mcp_conversation": index,
                "missing_function": 0,
            },
            "capacity_evidence": {"mcp_structural_weight": index},
        }
        for index, domain in enumerate(DOMAINS, start=1)
    }
    assert _select_domain(plans, "mcp_conversation") == "team_chat"
