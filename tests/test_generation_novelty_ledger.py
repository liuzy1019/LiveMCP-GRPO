import random
import threading

from src.live_mcp.generation.chain_scheduler import ChainSchedulerMixin
from src.live_mcp.generation_runtime import TeacherGenerationRuntime


class _Manager:
    server_names = ["payments"]


def test_retained_sequences_seed_scheduler_history_before_start() -> None:
    runtime = TeacherGenerationRuntime.__new__(TeacherGenerationRuntime)
    runtime.manager = _Manager()
    runtime._started = False
    runtime._chain_sampling_stats = {}
    runtime._chain_sampling_sequences = {}

    runtime.preload_retained_sequences({
        "payments": [["list_invoices"], ["get_invoice", "refund_invoice"]],
    })

    assert len(runtime._chain_sampling_stats["payments"]) == 2
    assert all(
        counters["attempted"] == 1 and counters["accepted"] == 1
        for counters in runtime._chain_sampling_stats["payments"].values()
    )
    assert set(runtime._chain_sampling_sequences["payments"].values()) == {
        ("list_invoices",),
        ("get_invoice", "refund_invoice"),
    }


def test_retained_sequence_is_not_selected_when_fresh_exact_chain_exists() -> None:
    subject = ChainSchedulerMixin()
    subject.prompt_profile = type("Profile", (), {"paper_baseline": True})()
    subject._dependency_graph_lock = threading.RLock()
    subject._chain_sampling_lock = threading.RLock()
    subject._chain_sampling_stats = {}
    subject._chain_sampling_sequences = {}
    retained = ["list_invoices"]
    fingerprint = subject._chain_fingerprint("payments", retained)
    subject._chain_sampling_stats["payments"] = {
        fingerprint: {"attempted": 1, "accepted": 1, "rejected_goal": 0},
    }
    subject._chain_sampling_sequences["payments"] = {
        fingerprint: tuple(retained),
    }

    selected, _, _, _ = subject._select_feasible_chain(
        "payments",
        [retained, ["get_invoice", "refund_invoice"]],
        random.Random(7),
    )

    assert selected == ["get_invoice", "refund_invoice"]
