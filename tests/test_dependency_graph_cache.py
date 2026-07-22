from __future__ import annotations

import json
import multiprocessing as mp
import os
import time
from pathlib import Path
from types import SimpleNamespace

import src.live_mcp.orchestrator as orchestrator_module
from src.live_mcp.config import project_root
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

DEPENDENT_TOOLS = [
    TOOLS[0],
    {
        "name": "tool_b",
        "description": "Mutate B using a required value.",
        "input_schema": {
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
        },
        "annotations": {"mutating": True},
    },
]


class _NoneRelationClient:
    model_path = "test-teacher"

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
            ],
        })


class _IncompleteClient:
    model_path = "test-teacher"

    def __init__(self):
        self.calls = 0

    def generate_chat(self, *_args, **_kwargs) -> str:
        self.calls += 1
        return '{"classifications": []}'


class _DirectedClient:
    model_path = "test-teacher"

    def generate_chat(self, *_args, **_kwargs) -> str:
        return json.dumps({
            "classifications": [
                {
                    "pair": "tool_a → tool_b", "source": "tool_a",
                    "target": "tool_b", "relation": "explicit",
                },
            ],
        })


class _ReverseDirectedClient:
    model_path = "test-teacher"

    def generate_chat(self, *_args, **_kwargs) -> str:
        return json.dumps({
            "classifications": [{
                "pair": "tool_a → tool_b",
                "source": "tool_b",
                "target": "tool_a",
                "relation": "implicit",
            }],
        })


class _CaptureBankingClient:
    model_path = "test-teacher"

    def __init__(self):
        self.messages = []

    def generate_chat(self, messages, **_kwargs) -> str:
        self.messages.append(messages)
        relation = "none" if len(self.messages) == 1 else "explicit"
        return json.dumps({
            "classifications": [{
                "pair": "get_balance → list_accounts",
                "source": "" if relation == "none" else "list_accounts",
                "target": "" if relation == "none" else "get_balance",
                "relation": relation,
            }],
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
    TaskOrchestrator.DEPENDENCY_CACHE_ROOT = Path(root) / "data" / "dependency_graphs"
    start_event.wait(timeout=5)
    orchestrator = TaskOrchestrator.__new__(TaskOrchestrator)
    orchestrator.manager = _Manager()
    orchestrator.client = _NoneRelationClient(counter=counter, delay=0.25)
    try:
        graph = orchestrator._get_or_build_dependency_graph("probe")
        result_queue.put(("ok", graph))
    except Exception as exc:  # pragma: no cover - surfaced in parent assertion
        result_queue.put(("error", repr(exc)))


def test_none_relation_uses_pair_when_model_returns_none_endpoints() -> None:
    orchestrator = TaskOrchestrator.__new__(TaskOrchestrator)
    orchestrator.client = _NoneRelationClient()

    classification = orchestrator._classify_edges_llm(TOOLS, "probe")

    assert orchestrator.client.calls == 1
    assert classification is not None
    graph, pairs = classification
    assert graph == {
        "tool_a": {"explicit": [], "implicit": []},
        "tool_b": {"explicit": [], "implicit": []},
    }
    assert pairs == [{
        "pair": ["tool_a", "tool_b"],
        "source": "",
        "target": "",
        "relation": "none",
    }]


def test_unordered_pair_records_teacher_selected_direction() -> None:
    orchestrator = TaskOrchestrator.__new__(TaskOrchestrator)
    orchestrator.client = _DirectedClient()

    classification = orchestrator._classify_edges_llm(DEPENDENT_TOOLS, "probe")

    assert classification is not None
    graph, pairs = classification
    assert graph["tool_a"]["explicit"] == ["tool_b"]
    assert graph["tool_b"]["implicit"] == []
    assert pairs[0]["source"] == "tool_a"
    assert pairs[0]["target"] == "tool_b"


def test_unordered_pair_accepts_reverse_teacher_direction() -> None:
    orchestrator = TaskOrchestrator.__new__(TaskOrchestrator)
    orchestrator.client = _ReverseDirectedClient()

    classification = orchestrator._classify_edges_llm(DEPENDENT_TOOLS, "probe")

    assert classification is not None
    graph, pairs = classification
    assert graph["tool_a"]["explicit"] == []
    assert graph["tool_b"]["implicit"] == ["tool_a"]
    assert pairs[0]["source"] == "tool_b"
    assert pairs[0]["target"] == "tool_a"


def test_relation_contract_violation_retries_only_the_invalid_pair() -> None:
    class _InvalidThenNoneClient:
        model_path = "test-teacher"

        def __init__(self) -> None:
            self.calls = 0

        def generate_chat(self, *_args, **_kwargs) -> str:
            self.calls += 1
            relation = "explicit" if self.calls == 1 else "none"
            return json.dumps({"classifications": [{
                "pair": "tool_a → tool_b",
                "source": "tool_a",
                "target": "tool_b",
                "relation": relation,
            }]})

    orchestrator = TaskOrchestrator.__new__(TaskOrchestrator)
    orchestrator.client = _InvalidThenNoneClient()

    classification = orchestrator._classify_edges_llm(TOOLS, "probe")

    assert classification is not None
    assert orchestrator.client.calls == 2
    assert classification[1][0]["relation"] == "none"


def test_relation_contract_rejects_readonly_implicit_source() -> None:
    issue = TaskOrchestrator._pair_classification_contract_issue(
        {
            "pair": ["tool_a", "tool_b"],
            "source": "tool_a",
            "target": "tool_b",
            "relation": "implicit",
        },
        {tool["name"]: tool for tool in TOOLS},
    )

    assert issue == "implicit source is explicitly readonly/non-mutating"


def test_relation_contract_rejects_explicit_entity_type_mismatch() -> None:
    tools = [
        {
            "name": "create_draft",
            "input_schema": {"type": "object", "required": ["to", "subject"]},
            "annotations": {"mutating": True},
        },
        {
            "name": "move_to_thread",
            "input_schema": {
                "type": "object",
                "required": ["email_id", "thread_id"],
            },
            "annotations": {"mutating": True},
        },
    ]
    issue = TaskOrchestrator._pair_classification_contract_issue(
        {
            "pair": ["create_draft", "move_to_thread"],
            "source": "create_draft",
            "target": "move_to_thread",
            "relation": "explicit",
        },
        {tool["name"]: tool for tool in tools},
        "email",
    )

    assert issue is not None
    assert "draft" in issue
    assert "email" in issue

    # A created entity may return foreign-key IDs that legitimately satisfy a
    # later required input; do not confuse creation type with output contract.
    issue_tools = [
        {"name": "create_subtask", "input_schema": {
            "type": "object", "required": ["issue_id", "title"],
        }},
        {"name": "get_issue", "input_schema": {
            "type": "object", "required": ["issue_id"],
        }},
    ]
    assert TaskOrchestrator._pair_classification_contract_issue(
        {
            "pair": ["create_subtask", "get_issue"],
            "source": "create_subtask", "target": "get_issue",
            "relation": "explicit",
        },
        {tool["name"]: tool for tool in issue_tools},
        "issue_tracker",
    ) is None


def test_strict_cache_validator_uses_domain_entity_contract() -> None:
    from scripts.validate_generation_pipeline import _strict_cache_issue

    tools = [
        {
            "name": "create_draft",
            "input_schema": {"type": "object", "required": ["to"]},
            "annotations": {"mutating": True},
        },
        {
            "name": "move_to_thread",
            "input_schema": {
                "type": "object", "required": ["email_id", "thread_id"],
            },
            "annotations": {"mutating": True},
        },
    ]
    ledger = [{
        "pair": ["create_draft", "move_to_thread"],
        "source": "create_draft",
        "target": "move_to_thread",
        "relation": "explicit",
    }]
    names = sorted(tool["name"] for tool in tools)
    data = {
        "cache_version": TaskOrchestrator.DEPENDENCY_CACHE_VERSION,
        "dependency_semantics_version": TaskOrchestrator.DEPENDENCY_SEMANTICS_VERSION,
        "tool_names": names,
        "tool_count": 2,
        "pair_classifications": ledger,
        "expected_pair_count": 1,
        "classified_pair_count": 1,
        "classification_complete": True,
        "graph": TaskOrchestrator._graph_from_pair_classifications(ledger, names),
        "classifier_contract_hash": "contract",
        "teacher_model_id": "teacher",
        "classifier_prompt_sha256": "prompt",
    }

    assert "关系定义冲突" in _strict_cache_issue(
        "email", data, names, tools,
    )


def test_complete_initial_ledger_sends_no_teacher_request() -> None:
    orchestrator = TaskOrchestrator.__new__(TaskOrchestrator)
    orchestrator.client = _IncompleteClient()
    initial = [{
        "pair": ["tool_a", "tool_b"],
        "source": "",
        "target": "",
        "relation": "none",
    }]

    classification = orchestrator._classify_edges_llm(
        TOOLS, "probe", initial,
    )

    assert classification is not None
    assert orchestrator.client.calls == 0


def test_atomic_cache_save_publishes_only_complete_json(tmp_path, monkeypatch) -> None:
    cache_root = tmp_path / "data" / "dependency_graphs"
    monkeypatch.setattr(TaskOrchestrator, "DEPENDENCY_CACHE_ROOT", cache_root)
    orchestrator = TaskOrchestrator.__new__(TaskOrchestrator)
    orchestrator.client = _NoneRelationClient()
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
    pairs = orchestrator._pair_classifications_from_graph(graph, ["tool_a", "tool_b"])
    orchestrator._save_dependency_cache("probe", schema_hash, TOOLS, graph, pairs)

    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    assert replace_observations[0]["old"] == {"sentinel": "old"}
    assert replace_observations[0]["new"]["classification_complete"] is True
    assert payload["classified_pair_count"] == 1
    assert len(payload["pair_classifications"]) == 1
    assert payload["classifier_contract_hash"] == orchestrator._classifier_contract_hash()
    assert payload["dependency_semantics_version"] == TaskOrchestrator.DEPENDENCY_SEMANTICS_VERSION
    assert not list(cache_path.parent.glob(f".{cache_path.name}.*.tmp"))


def test_recent_classification_failure_is_not_immediately_retried(
    tmp_path, monkeypatch,
) -> None:
    monkeypatch.setattr(
        TaskOrchestrator,
        "DEPENDENCY_CACHE_ROOT",
        tmp_path / "data" / "dependency_graphs",
    )
    orchestrator = TaskOrchestrator.__new__(TaskOrchestrator)
    orchestrator.manager = _Manager()
    orchestrator.client = _IncompleteClient()
    orchestrator._dependency_graph_failures = {}

    for _ in range(2):
        try:
            orchestrator._get_or_build_dependency_graph("probe")
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
    assert payload["classified_pair_count"] == 1
    assert len(payload["pair_classifications"]) == 1


def test_production_cache_load_preserves_llm_classification(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        TaskOrchestrator,
        "DEPENDENCY_CACHE_ROOT",
        tmp_path / "data" / "dependency_graphs",
    )
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
    orchestrator.client = _NoneRelationClient()
    schema_hash = orchestrator._tool_schema_hash(tools, "payments")
    cache_path = orchestrator._graph_cache_path("payments", schema_hash)
    cache_path.parent.mkdir(parents=True)
    graph = {
        "pay_invoice": {"explicit": [], "implicit": ["cancel_payment"]},
        "cancel_payment": {"explicit": [], "implicit": []},
    }
    pairs = orchestrator._pair_classifications_from_graph(
        graph, sorted(t["name"] for t in tools),
    )
    orchestrator._save_dependency_cache(
        "payments", schema_hash, tools, graph, pairs,
    )

    loaded = orchestrator._load_dependency_cache(
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


def test_classifier_prompt_does_not_contradict_preexisting_state_rule() -> None:
    prompt = TaskOrchestrator._dependency_classifier_system_prompt()
    assert "schedule_transfer → cancel_transfer" not in prompt
    assert "reversal" in prompt
    assert "pre-existing server state" in prompt


def test_classifier_receives_factual_discovery_output_fields() -> None:
    tools = [
        {
            "name": "list_accounts",
            "description": "List accounts.",
            "input_schema": {"type": "object", "properties": {}, "required": []},
            "annotations": {"readonly": True, "mutating": False},
        },
        {
            "name": "get_balance",
            "description": "Get an account balance.",
            "input_schema": {
                "type": "object",
                "properties": {"account_id": {"type": "string"}},
                "required": ["account_id"],
            },
            "annotations": {"readonly": True, "mutating": False},
        },
    ]
    orchestrator = TaskOrchestrator.__new__(TaskOrchestrator)
    orchestrator.client = _CaptureBankingClient()

    classification = orchestrator._classify_edges_llm(tools, "banking")

    assert classification is not None
    user_prompt = orchestrator.client.messages[0][1]["content"]
    assert "Tool: list_accounts" in user_prompt
    assert "Known output fields: account_id" in user_prompt
    assert "list_accounts → get_balance" in user_prompt
    assert len(orchestrator.client.messages) == 2
    assert "Contract Feedback" in orchestrator.client.messages[1][1]["content"]


def test_none_relation_rejects_known_output_to_required_input() -> None:
    tools = {
        "list_accounts": {
            "name": "list_accounts",
            "input_schema": {"type": "object", "required": []},
            "annotations": {"readonly": True, "mutating": False},
        },
        "get_balance": {
            "name": "get_balance",
            "input_schema": {
                "type": "object", "required": ["account_id"],
            },
            "annotations": {"readonly": True, "mutating": False},
        },
    }
    issue = TaskOrchestrator._pair_classification_contract_issue(
        {
            "pair": ["get_balance", "list_accounts"],
            "source": "",
            "target": "",
            "relation": "none",
        },
        tools,
        "banking",
    )

    assert issue is not None
    assert "account_id" in issue
    assert "list_accounts as source" in issue
    assert "get_balance as target" in issue
    assert "relation as explicit" in issue


def test_output_field_contract_invalidates_only_classifier_cache(monkeypatch) -> None:
    orchestrator = TaskOrchestrator.__new__(TaskOrchestrator)
    orchestrator.client = _NoneRelationClient()
    schema_hash = orchestrator._tool_schema_hash(TOOLS, "probe")
    classifier_hash = orchestrator._classifier_contract_hash("probe")
    expanded = {
        **orchestrator_module._DEPENDENCY_TOOL_OUTPUT_FIELDS,
        "probe": {"tool_a": ("value_id",)},
    }

    monkeypatch.setattr(
        orchestrator_module, "_DEPENDENCY_TOOL_OUTPUT_FIELDS", expanded,
    )

    assert orchestrator._tool_schema_hash(TOOLS, "probe") == schema_hash
    assert orchestrator._classifier_contract_hash("probe") != classifier_hash


def test_legacy_cache_migration_preserves_unaffected_pair_labels(
    tmp_path, monkeypatch,
) -> None:
    monkeypatch.setattr(
        TaskOrchestrator,
        "DEPENDENCY_CACHE_ROOT",
        tmp_path / "data" / "dependency_graphs",
    )
    tools = [
        DEPENDENT_TOOLS[0],
        DEPENDENT_TOOLS[1],
        {
            "name": "tool_c",
            "description": "Read C.",
            "input_schema": {"type": "object", "properties": {}, "required": []},
            "annotations": {"readonly": True},
        },
    ]
    orchestrator = TaskOrchestrator.__new__(TaskOrchestrator)
    orchestrator.client = _NoneRelationClient()
    schema_hash = orchestrator._tool_schema_hash(tools, "probe")
    graph = {
        tool["name"]: {"explicit": [], "implicit": []} for tool in tools
    }
    pairs = orchestrator._pair_classifications_from_graph(
        graph, [tool["name"] for tool in tools],
    )
    orchestrator._save_dependency_cache(
        "probe", schema_hash, tools, graph, pairs,
    )

    expanded = {
        **orchestrator_module._DEPENDENCY_TOOL_OUTPUT_FIELDS,
        "probe": {"tool_a": ("value",)},
    }
    monkeypatch.setattr(
        orchestrator_module, "_DEPENDENCY_TOOL_OUTPUT_FIELDS", expanded,
    )

    assert orchestrator._load_dependency_cache(
        "probe", schema_hash, tools,
    ) is None
    preserved = orchestrator._dependency_graph_repairs[("probe", schema_hash)]
    assert {tuple(entry["pair"]) for entry in preserved} == {
        ("tool_a", "tool_c"),
        ("tool_b", "tool_c"),
    }


def test_nonempty_output_contract_change_does_not_use_legacy_migration(
    tmp_path, monkeypatch,
) -> None:
    monkeypatch.setattr(
        TaskOrchestrator,
        "DEPENDENCY_CACHE_ROOT",
        tmp_path / "data" / "dependency_graphs",
    )
    first_contract = {
        **orchestrator_module._DEPENDENCY_TOOL_OUTPUT_FIELDS,
        "probe": {"tool_a": ("unrelated",)},
    }
    monkeypatch.setattr(
        orchestrator_module, "_DEPENDENCY_TOOL_OUTPUT_FIELDS", first_contract,
    )
    orchestrator = TaskOrchestrator.__new__(TaskOrchestrator)
    orchestrator.client = _NoneRelationClient()
    schema_hash = orchestrator._tool_schema_hash(DEPENDENT_TOOLS, "probe")
    graph = {
        tool["name"]: {"explicit": [], "implicit": []}
        for tool in DEPENDENT_TOOLS
    }
    pairs = orchestrator._pair_classifications_from_graph(
        graph, [tool["name"] for tool in DEPENDENT_TOOLS],
    )
    orchestrator._save_dependency_cache(
        "probe", schema_hash, DEPENDENT_TOOLS, graph, pairs,
    )

    changed_contract = {
        **first_contract,
        "probe": {"tool_a": ("value",)},
    }
    monkeypatch.setattr(
        orchestrator_module, "_DEPENDENCY_TOOL_OUTPUT_FIELDS", changed_contract,
    )

    assert orchestrator._load_dependency_cache(
        "probe", schema_hash, DEPENDENT_TOOLS,
    ) is None
    assert ("probe", schema_hash) not in getattr(
        orchestrator, "_dependency_graph_repairs", {},
    )


def test_production_cache_path_is_project_root_anchored(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(TaskOrchestrator, "DEPENDENCY_CACHE_ROOT", None)
    monkeypatch.chdir(tmp_path)
    path = TaskOrchestrator._graph_cache_path("probe", "abc")
    assert path == project_root() / "data" / "dependency_graphs" / "probe_abc.json"


def test_classifier_model_change_invalidates_cache(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        TaskOrchestrator,
        "DEPENDENCY_CACHE_ROOT",
        tmp_path / "data" / "dependency_graphs",
    )
    orchestrator = TaskOrchestrator.__new__(TaskOrchestrator)
    orchestrator.client = _NoneRelationClient()
    schema_hash = orchestrator._tool_schema_hash(TOOLS)
    graph = {
        "tool_a": {"explicit": [], "implicit": []},
        "tool_b": {"explicit": [], "implicit": []},
    }
    pairs = orchestrator._pair_classifications_from_graph(graph, ["tool_a", "tool_b"])
    orchestrator._save_dependency_cache("probe", schema_hash, TOOLS, graph, pairs)
    assert orchestrator._load_dependency_cache("probe", schema_hash, TOOLS) is not None

    orchestrator.client = SimpleNamespace(model_path="different-teacher")
    assert orchestrator._load_dependency_cache("probe", schema_hash, TOOLS) is None


def test_incomplete_pair_ledger_is_rejected_even_with_complete_counters(
    tmp_path, monkeypatch,
) -> None:
    monkeypatch.setattr(
        TaskOrchestrator,
        "DEPENDENCY_CACHE_ROOT",
        tmp_path / "data" / "dependency_graphs",
    )
    orchestrator = TaskOrchestrator.__new__(TaskOrchestrator)
    orchestrator.client = _NoneRelationClient()
    schema_hash = orchestrator._tool_schema_hash(TOOLS)
    cache_path = orchestrator._graph_cache_path("probe", schema_hash)
    cache_path.parent.mkdir(parents=True)
    contract = orchestrator._classifier_contract_payload()
    cache_path.write_text(json.dumps({
        "cache_version": TaskOrchestrator.DEPENDENCY_CACHE_VERSION,
        "dependency_semantics_version": TaskOrchestrator.DEPENDENCY_SEMANTICS_VERSION,
        "server_name": "probe",
        "schema_hash": schema_hash,
        "tool_names": ["tool_a", "tool_b"],
        "graph": {
            "tool_a": {"explicit": [], "implicit": []},
            "tool_b": {"explicit": [], "implicit": []},
        },
        "pair_classifications": [],
        "classifier_contract_hash": orchestrator._classifier_contract_hash(),
        **contract,
        "expected_pair_count": 1,
        "classified_pair_count": 1,
        "classification_complete": True,
    }), encoding="utf-8")
    assert orchestrator._load_dependency_cache("probe", schema_hash, TOOLS) is None


def test_in_memory_cache_key_changes_with_schema_and_teacher() -> None:
    orchestrator = TaskOrchestrator.__new__(TaskOrchestrator)
    orchestrator.client = _NoneRelationClient()
    orchestrator.manager = _Manager()
    first = orchestrator._dependency_cache_key("probe", TOOLS)
    changed_tools = [*TOOLS, {
        "name": "tool_c",
        "description": "Read C.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
        "annotations": {"readonly": True},
    }]
    second = orchestrator._dependency_cache_key("probe", changed_tools)
    orchestrator.client = SimpleNamespace(model_path="different-teacher")
    third = orchestrator._dependency_cache_key("probe", changed_tools)
    assert first != second
    assert second != third
