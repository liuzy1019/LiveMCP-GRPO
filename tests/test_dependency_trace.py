from src.live_mcp.dependency_trace import (
    align_sampled_chain,
    auxiliary_tool_call_indices,
    dependency_edges_from_alignment,
    verify_implicit_edges_counterfactually,
)

from types import SimpleNamespace


def test_sampled_chain_alignment_keeps_auxiliary_calls_out_of_edges() -> None:
    calls = [
        {"action": "tool_call", "tool_name": "search"},
        {"action": "tool_call", "tool_name": "detail"},
        {"action": "tool_call", "tool_name": "mutate"},
        {"action": "final_answer", "tool_name": "final_answer"},
    ]

    aligned = align_sampled_chain(calls, ["search", "mutate"])

    assert aligned == [0, 2]
    assert dependency_edges_from_alignment(aligned) == [[0, 2]]
    assert auxiliary_tool_call_indices(calls, aligned) == [1]


def test_sampled_chain_alignment_fails_closed_on_missing_or_reordered_step() -> None:
    calls = [
        {"action": "tool_call", "tool_name": "mutate"},
        {"action": "tool_call", "tool_name": "search"},
    ]

    assert align_sampled_chain(calls, ["search", "mutate"]) is None


def test_empty_sampled_chain_has_no_edges_or_auxiliary_dependency_partition() -> None:
    calls = [{"action": "tool_call", "tool_name": "search"}]

    aligned = align_sampled_chain(calls, [])

    assert aligned == []
    assert dependency_edges_from_alignment(aligned) == []
    assert auxiliary_tool_call_indices(calls, aligned) == [0]


def test_verified_evidence_selects_the_value_producing_repeated_source() -> None:
    calls = [
        {"action": "tool_call", "tool_name": "search"},
        {"action": "tool_call", "tool_name": "search"},
        {"action": "tool_call", "tool_name": "mutate"},
    ]
    evidence = [{
        "source_capability": "search",
        "target_capability": "mutate",
        "source_call_index": 1,
        "target_call_index": 2,
        "evidence_type": "explicit_value_binding",
    }]

    assert align_sampled_chain(
        calls,
        ["search", "mutate"],
        verified_dependency_evidence=evidence,
    ) == [1, 2]


def test_implicit_edge_requires_target_failure_without_source_and_success_with_it() -> None:
    class Manager:
        def __init__(self):
            self.states = {}
            self.next_id = 0

        def create_session(self, *, seed, server_names):
            self.next_id += 1
            session_id = f"s{self.next_id}"
            self.states[session_id] = False
            return SimpleNamespace(session_id=session_id)

        def close_session(self, session_id):
            self.states.pop(session_id, None)

    class Executor:
        def __init__(self, manager):
            self.manager = manager

        def execute(self, session_id, call, *, domain):
            if call.name == "source":
                self.manager.states[session_id] = True
                return SimpleNamespace(
                    success=True, state_changed=True, error_type=None,
                )
            success = self.manager.states[session_id]
            return SimpleNamespace(
                success=success,
                state_changed=False,
                error_type=None if success else "precondition_failed",
            )

    manager = Manager()
    evidence, issues = verify_implicit_edges_counterfactually(
        manager=manager,
        executor=Executor(manager),
        server_name="domain",
        seed=42,
        oracle_calls=[
            {"tool_name": "source", "arguments": {}},
            {"tool_name": "target", "arguments": {}},
        ],
        sampled_chain=["source", "target"],
        explicitly_verified_edges=set(),
    )

    assert issues == []
    assert evidence[0]["evidence_type"] == "implicit_counterfactual_replay"
    assert evidence[0]["target_without_source_error_type"] == "precondition_failed"
