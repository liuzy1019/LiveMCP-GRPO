from __future__ import annotations

import json
import multiprocessing as mp
import os
import time
from pathlib import Path
from types import SimpleNamespace

import src.live_mcp.orchestrator as orchestrator_module
from src.live_mcp.orchestrator import TaskOrchestrator


TOOLS = [
    {
        "name": "tool_a",
        "description": "Read A.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
        "annotations": {"readonly": True},
    },
    {
        "name": "tool_b",
        "description": "Read B.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
        "annotations": {"readonly": True},
    },
]


class _NoneRelationClient:
    def __init__(self, counter=None, delay: float = 0.0):
        self.counter = counter
        self.delay = delay
        self.calls = 0

    def generate_chat(self, *_args, **_kwargs) -> str:
        self.calls += 1
        if self.counter is not None:
            with self.counter.get_lock():
                self.counter.value += 1
        if self.delay:
            time.sleep(self.delay)
        return json.dumps({
            "classifications": [
                {
                    "pair": "tool_a → tool_b",
                    "source": "tool_a",
                    "target": "tool_b",
                    "relation": "none",
                },
                {
                    "pair": "tool_b → tool_a",
                    "source": "tool_b",
                    "target": "tool_a",
                    "relation": "none",
                },
            ],
        })


class _IncompleteClient:
    def __init__(self):
        self.calls = 0

    def generate_chat(self, *_args, **_kwargs) -> str:
        self.calls += 1
        return '{"classifications": []}'


class _BidirectionalClient:
    def generate_chat(self, *_args, **_kwargs) -> str:
        return json.dumps({
            "classifications": [
                {
                    "pair": "tool_a → tool_b", "source": "tool_a",
                    "target": "tool_b", "relation": "explicit",
                },
                {
                    "pair": "tool_b → tool_a", "source": "tool_b",
                    "target": "tool_a", "relation": "implicit",
                },
            ],
        })


class _Registry:
    def server_tools(self, _server_name: str) -> list[dict]:
        return TOOLS


class _Manager:
    def __init__(self):
        self.registry = _Registry()

    def create_session(self, seed: int):
        return SimpleNamespace(session_id=f"session-{seed}")

    def discover_tools(self, _session_id: str) -> None:
        return None

    def close_session(self, _session_id: str) -> None:
        return None


def _probe_cache_worker(root: str, counter, start_event, result_queue) -> None:
    os.chdir(root)
    start_event.wait(timeout=5)
    orchestrator = TaskOrchestrator.__new__(TaskOrchestrator)
    orchestrator.manager = _Manager()
    orchestrator.client = _NoneRelationClient(counter=counter, delay=0.25)
    try:
        graph = orchestrator._probe_dependency_graph("probe")
        result_queue.put(("ok", graph))
    except Exception as exc:  # pragma: no cover - surfaced in parent assertion
        result_queue.put(("error", repr(exc)))


def test_none_relation_uses_pair_when_model_returns_none_endpoints() -> None:
    orchestrator = TaskOrchestrator.__new__(TaskOrchestrator)
    orchestrator.client = _NoneRelationClient()

    graph = orchestrator._classify_edges_llm(TOOLS, "probe")

    assert orchestrator.client.calls == 1
    assert graph == {
        "tool_a": {"explicit": [], "implicit": []},
        "tool_b": {"explicit": [], "implicit": []},
    }


def test_ordered_pairs_can_retain_both_dependency_directions() -> None:
    orchestrator = TaskOrchestrator.__new__(TaskOrchestrator)
    orchestrator.client = _BidirectionalClient()

    graph = orchestrator._classify_edges_llm(TOOLS, "probe")

    assert graph["tool_a"]["explicit"] == ["tool_b"]
    assert graph["tool_b"]["implicit"] == ["tool_a"]


def test_atomic_cache_save_publishes_only_complete_json(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    orchestrator = TaskOrchestrator.__new__(TaskOrchestrator)
    schema_hash = orchestrator._tool_schema_hash(TOOLS)
    cache_path = orchestrator._graph_cache_path("probe", schema_hash)
    cache_path.parent.mkdir(parents=True)
    cache_path.write_text('{"sentinel": "old"}', encoding="utf-8")
    graph = {
        "tool_a": {"explicit": [], "implicit": []},
        "tool_b": {"explicit": [], "implicit": []},
    }

    real_replace = os.replace
    replace_observations = []

    def _observed_replace(source, destination) -> None:
        replace_observations.append({
            "old": json.loads(Path(destination).read_text(encoding="utf-8")),
            "new": json.loads(Path(source).read_text(encoding="utf-8")),
        })
        real_replace(source, destination)

    monkeypatch.setattr(orchestrator_module.os, "replace", _observed_replace)
    orchestrator._save_cached_graph("probe", schema_hash, TOOLS, graph)

    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    assert replace_observations[0]["old"] == {"sentinel": "old"}
    assert replace_observations[0]["new"]["classification_complete"] is True
    assert payload["classified_pair_count"] == 2
    assert payload["dependency_semantics_version"] == TaskOrchestrator.DEPENDENCY_SEMANTICS_VERSION
    assert not list(cache_path.parent.glob(f".{cache_path.name}.*.tmp"))


def test_recent_classification_failure_is_not_immediately_retried(
    tmp_path, monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    orchestrator = TaskOrchestrator.__new__(TaskOrchestrator)
    orchestrator.manager = _Manager()
    orchestrator.client = _IncompleteClient()
    orchestrator._dependency_graph_failures = {}

    for _ in range(2):
        try:
            orchestrator._probe_dependency_graph("probe")
        except RuntimeError:
            pass
        else:  # pragma: no cover - defensive assertion
            raise AssertionError("incomplete classification unexpectedly succeeded")

    assert orchestrator.client.calls == 3


def test_cold_cache_is_classified_once_across_processes(tmp_path) -> None:
    ctx = mp.get_context("fork")
    counter = ctx.Value("i", 0)
    start_event = ctx.Event()
    result_queue = ctx.Queue()
    processes = [
        ctx.Process(
            target=_probe_cache_worker,
            args=(str(tmp_path), counter, start_event, result_queue),
        )
        for _ in range(2)
    ]
    for process in processes:
        process.start()
    start_event.set()
    for process in processes:
        process.join(timeout=10)

    assert all(not process.is_alive() for process in processes)
    assert all(process.exitcode == 0 for process in processes)
    results = [result_queue.get(timeout=2) for _ in processes]
    assert all(status == "ok" for status, _ in results), results
    assert counter.value == 1

    cache_files = list((tmp_path / "data" / "dependency_graphs").glob("probe_*.json"))
    assert len(cache_files) == 1
    payload = json.loads(cache_files[0].read_text(encoding="utf-8"))
    assert payload["classification_complete"] is True
    assert payload["classified_pair_count"] == 2


def test_production_cache_load_preserves_llm_classification(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    tools = [
        {
            "name": "pay_invoice",
            "input_schema": {"required": ["invoice_id"], "properties": {}},
            "annotations": {"mutating": True},
        },
        {
            "name": "cancel_payment",
            "input_schema": {"required": ["payment_id"], "properties": {}},
            "annotations": {"mutating": True},
        },
    ]
    orchestrator = TaskOrchestrator.__new__(TaskOrchestrator)
    schema_hash = orchestrator._tool_schema_hash(tools, "payments")
    cache_path = orchestrator._graph_cache_path("payments", schema_hash)
    cache_path.parent.mkdir(parents=True)
    graph = {
        "pay_invoice": {"explicit": [], "implicit": ["cancel_payment"]},
        "cancel_payment": {"explicit": [], "implicit": []},
    }
    cache_path.write_text(json.dumps({
        "cache_version": TaskOrchestrator.DEPENDENCY_CACHE_VERSION,
        "dependency_semantics_version": TaskOrchestrator.DEPENDENCY_SEMANTICS_VERSION,
        "server_name": "payments",
        "schema_hash": schema_hash,
        "tool_names": sorted(t["name"] for t in tools),
        "graph": graph,
        "expected_pair_count": 2,
        "classified_pair_count": 2,
        "classification_complete": True,
    }), encoding="utf-8")

    loaded = orchestrator._maybe_load_cached_graph(
        "payments", schema_hash, tools,
    )
    assert loaded is not None
    assert loaded["pay_invoice"]["implicit"] == ["cancel_payment"]


def test_dependency_cache_hash_is_not_bound_to_handler_domain() -> None:
    tools = [
        {
            "name": "tool_a",
            "description": "Read a value.",
            "input_schema": {"type": "object", "properties": {}},
            "annotations": {"readonly": True},
        }
    ]
    assert TaskOrchestrator._tool_schema_hash(
        tools, "calendar",
    ) == TaskOrchestrator._tool_schema_hash(tools, "payments")
