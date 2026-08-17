from src.live_mcp.corpus.profile import (
    PROVE_PUBLISHED_COUNTS,
    _largest_remainder_quotas,
    _largest_remainder_ratio_quotas,
    _max_capacity_ratio_quotas,
)
from src.live_mcp.corpus.merge_allocation import (
    _capacity_weighted_domain_quotas,
    _proportional_stratum_order,
    _suggest_topup_count,
    _stratified_head,
)
from src.live_mcp.corpus.merge_dedup import _dedup_jaccard
from src.live_mcp.domain_allocation import jaccard_unique_sequence_count
from src.live_mcp.corpus.shard_recovery import _accepted_generation_deficits
from src.live_mcp.generation.batch import (
    _difficulty_attempt_schedule,
    largest_remainder_mix_quotas,
)

import pandas as pd
from collections import Counter
from types import SimpleNamespace


def test_merge_capacity_wrapper_uses_shared_allocator() -> None:
    assert _capacity_weighted_domain_quotas(
        6,
        ["calendar", "email"],
        {"calendar": 1, "email": 3},
        minimum_per_domain=1,
    ) == {"calendar": 2, "email": 4}


def test_largest_remainder_quotas_are_exact_and_capacity_bounded() -> None:
    counts = {
        ("calendar", "clarification_required"): 7,
        ("crm", "missing_function"): 2,
        ("email", "no_tool_or_abstention"): 1,
    }
    quotas = _largest_remainder_quotas(counts, target=6)
    assert sum(quotas.values()) == 6
    assert all(0 <= quotas[key] <= counts[key] for key in counts)
    assert quotas == {
        ("calendar", "clarification_required"): 4,
        ("crm", "missing_function"): 1,
        ("email", "no_tool_or_abstention"): 1,
    }


def test_topup_candidate_budget_uses_observed_low_retention() -> None:
    assert _suggest_topup_count(
        missing=2,
        available=305,
        candidates=1000,
    ) == 9


def test_prove_ratio_quotas_match_published_composition() -> None:
    quotas = _largest_remainder_ratio_quotas(
        PROVE_PUBLISHED_COUNTS,
        target=2507,
    )
    assert quotas == {
        "mcp_conversation": 2021,
        "missing_function": 278,
        "internal_abstention_proxy": 208,
    }


def test_prove_ratio_capacity_uses_largest_feasible_selection() -> None:
    capacities = {
        "mcp_conversation": 2021,
        "missing_function": 443,
        "internal_abstention_proxy": 278,
    }
    quotas = _max_capacity_ratio_quotas(
        capacities,
        PROVE_PUBLISHED_COUNTS,
    )
    assert quotas == {
        "mcp_conversation": 2021,
        "missing_function": 278,
        "internal_abstention_proxy": 208,
    }
    next_quotas = _largest_remainder_ratio_quotas(
        PROVE_PUBLISHED_COUNTS,
        target=sum(quotas.values()) + 1,
    )
    assert next_quotas["mcp_conversation"] > capacities["mcp_conversation"]


def test_accepted_difficulty_quotas_are_exact() -> None:
    quotas = largest_remainder_mix_quotas(
        20,
        {"complete": 0.6, "missing": 0.2, "minimal": 0.2},
    )
    assert quotas == {"complete": 12, "missing": 4, "minimal": 4}
    schedule = _difficulty_attempt_schedule(quotas)
    assert set(schedule) == set(quotas)
    assert all(schedule.count(key) > value for key, value in quotas.items())

    fixed_schedule = _difficulty_attempt_schedule(
        quotas, fill_shortfalls=False,
    )
    assert Counter(fixed_schedule) == quotas


def test_fixed_attempt_budget_does_not_replace_failed_candidates(
    monkeypatch,
) -> None:
    class Harness:
        from src.live_mcp.generation.batch import BatchGenerationMixin

    class FakeGenerator(Harness.BatchGenerationMixin):
        CHAIN_SAMPLING_JACCARD_THRESHOLD = 0.70
        SAMPLING_CONTEXT_REFRESH_K = 10

        def __init__(self) -> None:
            self.manager = SimpleNamespace(server_names=["banking"])
            self.attempts = 0

        def _generate_task_with_postprocess(
            self, server_name, seed, state_seed, difficulty,
            distractor_rate, missing_function_rate,
        ):
            self.attempts += 1
            if self.attempts <= 2:
                return None
            return SimpleNamespace(metadata={"difficulty": difficulty})

        def _generate_irrelevant_tasks(
            self, count, seed, servers, *, failure_callback=None,
            max_candidate_attempts=None,
        ):
            assert count == 1
            assert max_candidate_attempts == 1
            return []

    monkeypatch.setenv("LIVEMCP_FIXED_ATTEMPT_BUDGET", "1")
    generator = FakeGenerator()
    tasks = generator.generate_many(
        "banking",
        count=6,
        seed=42,
        difficulty_mix={"complete": 0.6, "missing": 0.2, "minimal": 0.2},
        irrelevance_count=1,
        failure_callback=lambda _record: None,
    )

    assert generator.attempts == 5
    assert len(tasks) == 3
    assert all(task.metadata["fixed_attempt_budget"] is True for task in tasks)


def test_fixed_attempt_budget_limits_irrelevance_without_failure_callback(
    monkeypatch,
) -> None:
    class Harness:
        from src.live_mcp.generation.batch import BatchGenerationMixin

    class FakeGenerator(Harness.BatchGenerationMixin):
        CHAIN_SAMPLING_JACCARD_THRESHOLD = 0.70
        SAMPLING_CONTEXT_REFRESH_K = 10

        def __init__(self) -> None:
            self.manager = SimpleNamespace(server_names=["banking"])

        def _generate_task_with_postprocess(self, *args, **kwargs):
            return SimpleNamespace(metadata={"difficulty": args[3]})

        def _generate_irrelevant_tasks(
            self, count, seed, servers, *, max_candidate_attempts=None,
        ):
            assert count == 1
            assert max_candidate_attempts == 1
            return []

    monkeypatch.setenv("LIVEMCP_FIXED_ATTEMPT_BUDGET", "1")
    tasks = FakeGenerator().generate_many(
        "banking", count=4, seed=42, irrelevance_count=1,
    )
    assert len(tasks) == 3
    assert all(task.metadata["fixed_attempt_budget"] is True for task in tasks)


def test_explicit_attempt_budget_contract_overrides_process_environment(
    monkeypatch,
) -> None:
    class Harness:
        from src.live_mcp.generation.batch import BatchGenerationMixin

    class FakeGenerator(Harness.BatchGenerationMixin):
        CHAIN_SAMPLING_JACCARD_THRESHOLD = 0.70
        SAMPLING_CONTEXT_REFRESH_K = 10

        def __init__(self) -> None:
            self.manager = SimpleNamespace(server_names=["banking"])

        def _generate_task_with_postprocess(self, *args, **kwargs):
            return SimpleNamespace(metadata={"difficulty": args[3]})

        def _generate_irrelevant_tasks(self, count, seed, servers):
            assert count == 0
            return []

    monkeypatch.setenv("LIVEMCP_FIXED_ATTEMPT_BUDGET", "1")
    tasks = FakeGenerator().generate_many(
        "banking",
        count=3,
        seed=42,
        irrelevance_ratio=0.0,
        fixed_attempt_budget=False,
    )
    assert len(tasks) == 3
    assert all(task.metadata["fixed_attempt_budget"] is False for task in tasks)


def test_merge_selection_preserves_population_ratio_not_equal_scenarios() -> None:
    frame = pd.DataFrame({
        "scenario_type": ["normal_safe_success"] * 8
        + ["clarification_required"] * 2,
        "extra_info": [
            {"domain": "banking"} for _ in range(10)
        ],
        "uid": [f"row-{index}" for index in range(10)],
    })
    selected = _stratified_head(frame, 5)
    assert selected["scenario_type"].value_counts().to_dict() == {
        "normal_safe_success": 4,
        "clarification_required": 1,
    }


def test_jaccard_input_order_interleaves_joint_generation_strata() -> None:
    rows = []
    for difficulty, count in (("complete", 6), ("missing", 2), ("minimal", 2)):
        for index in range(count):
            rows.append({
                "difficulty": difficulty,
                "scenario_type": "normal_safe_success",
                "extra_info": {"domain": "banking", "oracle_calls": [
                    {"action": "tool_call", "tool_name": f"tool_{index}"},
                ]},
            })
    ordered = _proportional_stratum_order(pd.DataFrame(rows))
    assert set(ordered.iloc[:3]["difficulty"]) == {
        "complete", "missing", "minimal",
    }

    # All three strata offer the same first sequence. Proportional ordering
    # prevents six complete rows from claiming every conflict before another
    # stratum is even considered; the hard threshold itself remains intact.
    unique, removed = _dedup_jaccard(ordered)
    assert len(unique) == 6
    assert removed == 4


def test_prove_jaccard_ignores_hidden_source_chain_length() -> None:
    def row(task_id: str, source_chain: list[str]) -> dict:
        return {
            "uid": task_id,
            "extra_info": {
                "task_id": task_id,
                "domain": "email",
                "source_chain_seed": source_chain,
                "oracle_calls": [
                    {
                        "action": "tool_call",
                        "tool_name": "search_emails",
                        "server_name": "email",
                    }
                ],
            },
        }

    frame = pd.DataFrame([
        row("short", ["search_emails", "forward_email"]),
        row(
            "long",
            ["search_emails", "forward_email", "get_thread", "reply_email"],
        ),
    ])
    retained, removed = _dedup_jaccard(frame)
    assert retained["uid"].tolist() == ["short"]
    assert removed == 1

    local_retained, local_removed = _dedup_jaccard(frame, mode="local")
    assert len(local_retained) == 2
    assert local_removed == 0


def test_prove_jaccard_is_plain_tool_name_set_not_position_aware() -> None:
    frame = pd.DataFrame([
        {"uid": "forward", "extra_info": {"domain": "email", "oracle_calls": [
            {"action": "tool_call", "tool_name": "search_emails"},
            {"action": "tool_call", "tool_name": "get_email"},
        ]}},
        {"uid": "reverse", "extra_info": {"domain": "email", "oracle_calls": [
            {"action": "tool_call", "tool_name": "get_email"},
            {"action": "tool_call", "tool_name": "search_emails"},
        ]}},
    ])
    retained, removed = _dedup_jaccard(frame, mode="prove")
    assert retained["uid"].tolist() == ["forward"]
    assert removed == 1

    local_retained, local_removed = _dedup_jaccard(frame, mode="local")
    assert local_retained["uid"].tolist() == ["forward", "reverse"]
    assert local_removed == 0


def test_capacity_estimate_uses_same_plain_jaccard_as_merge() -> None:
    assert jaccard_unique_sequence_count([
        ["search_emails", "get_email"],
        ["get_email", "search_emails"],
    ]) == 1


def test_recovery_targets_accepted_difficulty_and_irrelevance_deficits() -> None:
    tasks = [
        SimpleNamespace(task_type="task_planner", difficulty="complete")
        for _ in range(10)
    ] + [
        SimpleNamespace(task_type="task_planner", difficulty="missing")
        for _ in range(4)
    ] + [
        SimpleNamespace(task_type="task_planner", difficulty="minimal")
        for _ in range(5)
    ]
    irrelevance, mix = _accepted_generation_deficits(
        tasks,
        pool_target=23,
        configured_irrelevance_count=1,
        configured_irrelevance_ratio=0.05,
        difficulty_mix={"complete": 0.6, "missing": 0.2, "minimal": 0.2},
    )
    assert irrelevance == 1
    assert mix == {"complete": 3.0, "missing": 0.0, "minimal": 0.0}


def test_generate_many_preserves_accepted_mix_across_failures() -> None:
    class Harness:
        from src.live_mcp.generation.batch import BatchGenerationMixin

    class FakeGenerator(Harness.BatchGenerationMixin):
        CHAIN_SAMPLING_JACCARD_THRESHOLD = 0.70
        SAMPLING_CONTEXT_REFRESH_K = 10

        def __init__(self) -> None:
            self.manager = SimpleNamespace(server_names=["banking"])
            self.failures = Counter()

        def _generate_task_with_postprocess(
            self, server_name, seed, state_seed, difficulty,
            distractor_rate, missing_function_rate,
        ):
            if difficulty == "missing" and self.failures[difficulty] < 2:
                self.failures[difficulty] += 1
                return None
            return SimpleNamespace(metadata={"difficulty": difficulty})

        def _generate_irrelevant_tasks(self, count, seed, servers):
            assert count == 0
            return []

    generator = FakeGenerator()
    tasks = generator.generate_many(
        "banking",
        count=20,
        seed=42,
        difficulty_mix={"complete": 0.6, "missing": 0.2, "minimal": 0.2},
        irrelevance_ratio=0.0,
    )
    assert Counter(task.metadata["difficulty"] for task in tasks) == {
        "complete": 12,
        "missing": 4,
        "minimal": 4,
    }


def test_generate_many_returns_hard_gate_prefix_for_outer_recovery() -> None:
    class Harness:
        from src.live_mcp.generation.batch import BatchGenerationMixin

    class FakeGenerator(Harness.BatchGenerationMixin):
        CHAIN_SAMPLING_JACCARD_THRESHOLD = 0.70
        SAMPLING_CONTEXT_REFRESH_K = 10

        def __init__(self) -> None:
            self.manager = SimpleNamespace(server_names=["banking"])

        def _generate_task_with_postprocess(
            self, server_name, seed, state_seed, difficulty,
            distractor_rate, missing_function_rate,
        ):
            if difficulty == "complete":
                return None
            return SimpleNamespace(metadata={"difficulty": difficulty})

        def _generate_irrelevant_tasks(self, count, seed, servers):
            assert count == 0
            return []

    tasks = FakeGenerator().generate_many(
        "banking",
        count=10,
        seed=42,
        difficulty_mix={"complete": 0.6, "missing": 0.2, "minimal": 0.2},
        irrelevance_ratio=0.0,
    )

    # Successful candidates remain available to the shard-level recovery
    # loop; hard-gate failures are never converted into accepted rows.
    assert Counter(task.metadata["difficulty"] for task in tasks) == {
        "missing": 2,
        "minimal": 2,
    }
