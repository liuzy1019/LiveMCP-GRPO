from src.live_mcp.dependency_trace import (
    align_sampled_chain,
    auxiliary_tool_call_indices,
    dependency_edges_from_alignment,
    select_realized_dependency_chain,
    verify_implicit_edges_counterfactually,
)
from src.live_mcp.contracts.outcome import (
    successful_state_transition_noop_indices,
)
from src.live_mcp.dependency_value_flow import _verify_dependency_evidence
from src.live_mcp.types import OracleCall

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


def test_sampled_chain_can_select_latest_repeated_state_transition() -> None:
    calls = [
        {"action": "tool_call", "tool_name": "create"},
        {"action": "tool_call", "tool_name": "update"},
        {"action": "tool_call", "tool_name": "update"},
        {"action": "tool_call", "tool_name": "update"},
        {"action": "tool_call", "tool_name": "target"},
    ]

    assert align_sampled_chain(
        calls, ["create", "update", "target"], prefer_latest=True,
    ) == [0, 3, 4]


def test_empty_sampled_chain_has_no_edges_or_auxiliary_dependency_partition() -> None:
    calls = [{"action": "tool_call", "tool_name": "search"}]

    aligned = align_sampled_chain(calls, [])

    assert aligned == []
    assert dependency_edges_from_alignment(aligned) == []
    assert auxiliary_tool_call_indices(calls, aligned) == [0]


def test_state_transition_noop_indices_exclude_readonly_success() -> None:
    calls = [
        {
            "action": "tool_call",
            "tool_name": "search_emails",
            "arguments": {"keyword": "migration"},
        },
        {
            "action": "tool_call",
            "tool_name": "move_to_thread",
            "arguments": {"email_id": "e1", "thread_id": "t1"},
        },
    ]
    history = [
        {
            "success": True,
            "state_changed": False,
            "tool_name": "search_emails",
            "arguments": {"keyword": "migration"},
        },
        {
            "success": True,
            "state_changed": False,
            "tool_name": "move_to_thread",
            "arguments": {"email_id": "e1", "thread_id": "t1"},
        },
    ]

    assert successful_state_transition_noop_indices(
        oracle_calls=calls,
        execution_history=history,
        domain="email",
    ) == {1}


def test_realized_chain_selects_longest_actual_graph_path() -> None:
    calls = [
        {"action": "tool_call", "tool_name": "search"},
        {"action": "tool_call", "tool_name": "read"},
        {"action": "tool_call", "tool_name": "update"},
        {"action": "tool_call", "tool_name": "audit"},
    ]
    graph = {
        "search": {"explicit": ["read"], "implicit": []},
        "read": {"explicit": ["update"], "implicit": []},
        "update": {"explicit": [], "implicit": ["audit"]},
    }

    assert select_realized_dependency_chain(calls, graph) == (
        ["search", "read", "update", "audit"],
        [0, 1, 2, 3],
    )


def test_realized_chain_ignores_ambiguous_or_absent_graph_edges() -> None:
    calls = [
        {"action": "tool_call", "tool_name": "source"},
        {"action": "tool_call", "tool_name": "ambiguous"},
        {"action": "tool_call", "tool_name": "isolated"},
    ]
    graph = {
        "source": {
            "explicit": ["ambiguous"],
            "implicit": ["ambiguous"],
        },
    }

    assert select_realized_dependency_chain(calls, graph) == ([], [])


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


def test_implicit_counterfactual_replays_auxiliary_calls_before_target() -> None:
    """Removing the source must not also remove unrelated setup calls."""

    class Manager:
        def __init__(self):
            self.states = {}
            self.next_id = 0

        def create_session(self, *, seed, server_names):
            self.next_id += 1
            session_id = f"s{self.next_id}"
            self.states[session_id] = {"source": False, "auxiliary": False}
            return SimpleNamespace(session_id=session_id)

        def close_session(self, session_id):
            self.states.pop(session_id, None)

    class Executor:
        def __init__(self, manager):
            self.manager = manager

        def execute(self, session_id, call, *, domain):
            state = self.manager.states[session_id]
            if call.name == "source":
                state["source"] = True
                return SimpleNamespace(
                    success=True, state_changed=True, error_type=None,
                )
            if call.name == "auxiliary":
                state["auxiliary"] = True
                return SimpleNamespace(
                    success=True, state_changed=True, error_type=None,
                )
            success = state["source"] and state["auxiliary"]
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
            {"action": "tool_call", "tool_name": "source", "arguments": {}},
            {"action": "tool_call", "tool_name": "auxiliary", "arguments": {}},
            {"action": "tool_call", "tool_name": "target", "arguments": {}},
        ],
        sampled_chain=["source", "target"],
        explicitly_verified_edges=set(),
    )

    assert issues == []
    assert evidence == [{
        "source_capability": "source",
        "target_capability": "target",
        "evidence_type": "implicit_counterfactual_replay",
        "source_call_index": 0,
        "target_call_index": 2,
        "target_without_source_error_type": "precondition_failed",
        "source_state_changed": True,
        "target_with_source_succeeded": True,
    }]


def test_implicit_counterfactual_uses_last_repeated_transition() -> None:
    class Manager:
        def __init__(self):
            self.states = {}
            self.next_id = 0

        def create_session(self, *, seed, server_names):
            self.next_id += 1
            session_id = f"s{self.next_id}"
            self.states[session_id] = "placed"
            return SimpleNamespace(session_id=session_id)

        def close_session(self, session_id):
            self.states.pop(session_id, None)

    class Executor:
        _next = {
            "placed": "confirmed",
            "confirmed": "preparing",
            "preparing": "delivering",
        }

        def __init__(self, manager):
            self.manager = manager

        def execute(self, session_id, call, *, domain):
            current = self.manager.states[session_id]
            if call.name == "update":
                requested = call.arguments["status"]
                success = self._next.get(current) == requested
                if success:
                    self.manager.states[session_id] = requested
                return SimpleNamespace(
                    success=success,
                    state_changed=success,
                    error_type=None if success else "precondition_failed",
                )
            success = current == "delivering"
            return SimpleNamespace(
                success=success,
                state_changed=False,
                error_type=None if success else "precondition_failed",
            )

    manager = Manager()
    calls = [
        {"tool_name": "update", "arguments": {"status": "confirmed"}},
        {"tool_name": "update", "arguments": {"status": "preparing"}},
        {"tool_name": "update", "arguments": {"status": "delivering"}},
        {"tool_name": "target", "arguments": {}},
    ]
    evidence, issues = verify_implicit_edges_counterfactually(
        manager=manager,
        executor=Executor(manager),
        server_name="domain",
        seed=42,
        oracle_calls=calls,
        sampled_chain=["update", "target"],
        explicitly_verified_edges=set(),
    )

    assert issues == []
    assert evidence[0]["source_call_index"] == 2
    assert evidence[0]["target_call_index"] == 3


def test_explicit_evidence_keeps_repeated_targets_for_canonical_path() -> None:
    calls = [
        OracleCall("create", {}),
        OracleCall("update", {"resource_id": "r1", "status": "confirmed"}),
        OracleCall("update", {"resource_id": "r1", "status": "preparing"}),
        OracleCall("update", {"resource_id": "r1", "status": "delivering"}),
        OracleCall("target", {"resource_id": "r1"}),
    ]
    observations = [
        {"resource_id": "r1"}, {}, {}, {}, {},
    ]
    evidence = _verify_dependency_evidence(
        [{
            "source_capability": "create",
            "target_capability": "update",
            "target_argument": "resource_id",
            "source_output_field": "resource_id",
        }],
        calls,
        observations,
        "Create a resource and advance it until it can be tracked.",
        [{
            "name": "update",
            "input_schema": {
                "type": "object",
                "required": ["resource_id", "status"],
            },
        }],
    )
    evidence.append({
        "source_capability": "update",
        "target_capability": "target",
        "source_call_index": 3,
        "target_call_index": 4,
        "evidence_type": "implicit_counterfactual_replay",
    })

    assert [item["target_call_index"] for item in evidence[:-1]] == [1, 2, 3]
    assert align_sampled_chain(
        calls,
        ["create", "update", "target"],
        verified_dependency_evidence=evidence,
    ) == [0, 3, 4]
