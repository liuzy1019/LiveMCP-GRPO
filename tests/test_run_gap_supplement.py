from src.live_mcp.corpus.planning import (
    _batch_chain_quotas,
    _select_bucket,
    _select_domain,
)


def test_select_bucket_prioritizes_mcp_before_missing_function() -> None:
    assert _select_bucket({
        "mcp_conversation": 12,
        "missing_function": 31,
        "internal_abstention_proxy": 999,
    }) == "mcp_conversation"


def test_select_bucket_uses_missing_function_after_mcp_closes() -> None:
    assert _select_bucket({
        "mcp_conversation": 0,
        "missing_function": 31,
        "internal_abstention_proxy": 999,
    }) == "missing_function"


def test_select_bucket_returns_none_without_native_gap() -> None:
    assert _select_bucket({
        "mcp_conversation": 0,
        "missing_function": 0,
        "internal_abstention_proxy": 100,
    }) is None


def test_select_domain_uses_largest_gap_in_selected_bucket() -> None:
    assert _select_domain({
        "payments": {
            "gaps": {"mcp_conversation": 478},
            "capacity_evidence": {"mcp_structural_weight": 10},
        },
        "food_delivery": {
            "gaps": {"mcp_conversation": 408},
            "capacity_evidence": {"mcp_structural_weight": 20},
        },
        "shopping": {
            "gaps": {"mcp_conversation": 61},
            "capacity_evidence": {"mcp_structural_weight": 30},
        },
    }, "mcp_conversation") == "payments"


def test_batch_chain_quotas_follow_remaining_chain_gaps() -> None:
    quotas = _batch_chain_quotas(
        {"1-2": 63, "3-5": 325, "6+": 72},
        20,
    )
    assert quotas == {"1-2": 3, "3-5": 14, "6+": 3}
