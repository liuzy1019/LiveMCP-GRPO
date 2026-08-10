from __future__ import annotations

import json
import multiprocessing as mp
import os
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import src.live_mcp.orchestrator as orchestrator_module
from src.live_mcp.domain_contracts import dependency as dependency_contracts
from src.live_mcp.domain_contracts import outputs as output_contracts
from src.live_mcp.domain_contracts import states as state_contracts
from src.live_mcp.domain_contracts import value_bindings
from src.live_mcp.contracts.chain_simulator import simulate_symbolic_chain
from src.live_mcp.contracts.factory import build_contract_registry
from src.live_mcp.config import project_root
from src.live_mcp.live_state_feasibility import chain_is_feasible
from src.live_mcp.orchestrator import TaskOrchestrator
from src.live_mcp.prompt_profiles import resolve_prompt_profile
from src.live_mcp.servers.payments.server import TOOLS as PAYMENT_TOOLS
from src.live_mcp.servers.calendar.server import TOOLS as CALENDAR_TOOLS
from src.live_mcp.servers.shopping.server import TOOLS as SHOPPING_TOOLS
from src.live_mcp.types import OracleProgram


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

    def generate_chat(self, messages, **_kwargs) -> str:
        self.calls += 1
        if self.counter is not None:
            with self.counter.get_lock():
                self.counter.value += 1
        if self.delay:
            time.sleep(self.delay)
        is_pair_audit = "independent adversarial reviewer" in messages[0]["content"]
        return json.dumps({
            "classifications": [
                {
                    "pair": "tool_a → tool_b",
                    "source": "" if is_pair_audit else "tool_a",
                    "target": "" if is_pair_audit else "tool_b",
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

    def create_session(self, seed: int, **_kwargs):
        return SimpleNamespace(session_id=f"session-{seed}")

    def discover_tools(self, _session_id: str) -> None:
        return None

    def close_session(self, _session_id: str) -> None:
        return None


def _self_agreeing_pair_audits(
    pairs: list[dict],
    tools: list[dict],
    server_name: str,
) -> list[dict]:
    tools_by_name = {tool["name"]: tool for tool in tools}
    audits = []
    for entry in pairs:
        pair = tuple(entry["pair"])
        bindings = TaskOrchestrator._dependency_pair_binding_candidates(
            pair, tools_by_name, server_name,
        )
        decision = {
            "source": entry["source"],
            "target": entry["target"],
            "relation": entry["relation"],
        }
        audits.append({
            "pair": list(pair),
            "binding_candidates": bindings,
            "initial": decision,
            "review": decision,
            "tie_break": None,
            "final": decision,
            "disagrees_with_raw": False,
        })
    return audits


def _install_complete_probe_contract(monkeypatch, tools: list[dict]) -> None:
    names = {tool["name"] for tool in tools}
    monkeypatch.setitem(
        output_contracts.DOMAIN_VALUE_OUTPUT_FIELDS,
        "probe",
        {name: () for name in names},
    )
    monkeypatch.setitem(
        dependency_contracts._DEPENDENCY_TOOL_OUTPUT_FIELDS,
        "probe",
        {name: () for name in names},
    )
    monkeypatch.setitem(
        dependency_contracts._DEPENDENCY_TOOL_STATE_PRECONDITIONS,
        "probe",
        {name: frozenset() for name in names},
    )
    monkeypatch.setitem(
        dependency_contracts._DEPENDENCY_TOOL_STATE_POSTCONDITIONS,
        "probe",
        {name: frozenset() for name in names},
    )


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

    assert orchestrator.client.calls == 2
    assert classification is not None
    graph, pairs, audits = classification
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
    assert len(audits) == 1
    assert audits[0]["final"]["relation"] == "none"


def test_unordered_pair_records_teacher_selected_direction() -> None:
    orchestrator = TaskOrchestrator.__new__(TaskOrchestrator)
    orchestrator.client = _DirectedClient()

    classification = orchestrator._classify_edges_llm(DEPENDENT_TOOLS, "probe")

    assert classification is not None
    graph, pairs, audits = classification
    # The raw direction is preserved, but an uncovered probe domain cannot
    # promote it into the eligible graph.
    assert graph["tool_a"]["explicit"] == []
    assert graph["tool_b"]["implicit"] == []
    assert pairs[0]["source"] == "tool_a"
    assert pairs[0]["target"] == "tool_b"
    assert audits[0]["initial"] == audits[0]["review"]
    assert audits[0]["binding_candidates"] == []
    relation_audit = TaskOrchestrator._build_local_relation_audits(
        pairs, DEPENDENT_TOOLS, "probe",
    )[0]
    assert relation_audit["verdict"] == "insufficient_evidence"


def test_unordered_pair_accepts_reverse_teacher_direction() -> None:
    orchestrator = TaskOrchestrator.__new__(TaskOrchestrator)
    orchestrator.client = _ReverseDirectedClient()

    classification = orchestrator._classify_edges_llm(DEPENDENT_TOOLS, "probe")

    assert classification is not None
    graph, pairs, audits = classification
    assert graph["tool_a"]["explicit"] == []
    assert graph["tool_b"]["implicit"] == []
    assert pairs[0]["source"] == "tool_b"
    assert pairs[0]["target"] == "tool_a"
    assert audits[0]["final"]["relation"] == "implicit"


def test_relation_contract_does_not_rewrite_first_structurally_valid_raw_pair() -> None:
    class _InvalidThenNoneClient:
        model_path = "test-teacher"

        def __init__(self) -> None:
            self.calls = 0

        def generate_chat(self, *_args, **_kwargs) -> str:
            self.calls += 1
            relation = "explicit" if self.calls == 1 else "none"
            return json.dumps({"classifications": [{
                "pair": "tool_a → tool_b",
                "source": "tool_a" if relation == "explicit" else "",
                "target": "tool_b" if relation == "explicit" else "",
                "relation": relation,
            }]})

    orchestrator = TaskOrchestrator.__new__(TaskOrchestrator)
    orchestrator.client = _InvalidThenNoneClient()

    classification = orchestrator._classify_edges_llm(TOOLS, "probe")

    assert classification is not None
    assert orchestrator.client.calls == 2
    assert classification[1][0]["relation"] == "explicit"
    assert classification[0]["tool_a"]["explicit"] == []


def test_readonly_pair_is_not_rewritten_by_contract_feedback() -> None:
    class _FeedbackAwareClient:
        model_path = "test-teacher"

        def __init__(self) -> None:
            self.messages = []

        def generate_chat(self, messages, **_kwargs) -> str:
            self.messages.append(messages)
            feedback = messages[1]["content"]
            relation = (
                "none"
                if (
                    'only valid answer is relation="none"' in feedback
                    or "independent adversarial reviewer"
                    in messages[0]["content"]
                )
                else "explicit"
            )
            return json.dumps({"classifications": [{
                "pair": "tool_a → tool_b",
                "source": "" if relation == "none" else "tool_a",
                "target": "" if relation == "none" else "tool_b",
                "relation": relation,
            }]})

    orchestrator = TaskOrchestrator.__new__(TaskOrchestrator)
    orchestrator.client = _FeedbackAwareClient()

    classification = orchestrator._classify_edges_llm(TOOLS, "probe")

    assert classification is not None
    assert len(orchestrator.client.messages) == 2
    assert classification[1][0]["relation"] == "explicit"
    assert "Contract Feedback" not in orchestrator.client.messages[0][1]["content"]


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

    # A foreign-key ID copied from the source call's required arguments is an
    # echo, not a newly discovered value.  Incomplete output ledgers used to
    # let this pair bypass the explicit binding check.
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
    ) is not None

    # checkout now returns only the created order ID/status summary.  Product
    # IDs live in get_order detail, so checkout -> get_product must fail closed.
    shopping_tools = {tool["name"]: tool for tool in SHOPPING_TOOLS}
    assert TaskOrchestrator._pair_classification_contract_issue(
        {
            "pair": ["checkout", "get_product"],
            "source": "checkout",
            "target": "get_product",
            "relation": "explicit",
        },
        shopping_tools,
        "shopping",
    ) is not None

    # product and cart_item share product_id identity.  Whether the observed
    # product is currently in the cart is a live Step-2 membership check, not a
    # Step-1 entity-type contradiction.
    assert TaskOrchestrator._pair_classification_contract_issue(
        {
            "pair": ["get_order", "remove_from_cart"],
            "source": "get_order",
            "target": "remove_from_cart",
            "relation": "explicit",
        },
        shopping_tools,
        "shopping",
    ) is None


def test_strict_cache_validator_keeps_raw_relation_separate_from_eligibility() -> None:
    from scripts.validate_generation_pipeline import _strict_cache_issue

    tools = [
        tool for tool in SHOPPING_TOOLS
        if tool["name"] in {"get_product", "update_cart_quantity"}
    ]
    ledger = [{
        "pair": ["get_product", "update_cart_quantity"],
        "source": "get_product",
        "target": "update_cart_quantity",
        "relation": "explicit",
    }]
    names = sorted(tool["name"] for tool in tools)
    relation_audits = TaskOrchestrator._build_local_relation_audits(
        ledger, tools, "shopping",
    )
    graph = TaskOrchestrator._eligible_graph_from_relation_audits(
        ledger, relation_audits, names,
    )
    pair_audits = _self_agreeing_pair_audits(ledger, tools, "shopping")
    data = {
        "cache_version": TaskOrchestrator.DEPENDENCY_CACHE_VERSION,
        "dependency_semantics_version": TaskOrchestrator.DEPENDENCY_SEMANTICS_VERSION,
        "tool_names": names,
        "tool_count": 2,
        "pair_classifications": ledger,
        "expected_pair_count": 1,
        "classified_pair_count": 1,
        "classification_complete": True,
        "raw_graph": TaskOrchestrator._graph_from_pair_classifications(
            ledger, names,
        ),
        "graph": graph,
        "pair_audits": pair_audits,
        "audited_pair_count": 1,
        "audit_complete": True,
        "relation_audits": relation_audits,
        "relation_audited_pair_count": 1,
        "relation_audit_complete": True,
        "relation_audit_counts": dict(
                Counter(
                audit["verdict"] for audit in relation_audits
            )
        ),
        "classifier_contract_hash": "contract",
        "teacher_model_id": "teacher",
        "classifier_prompt_sha256": "prompt",
        "output_field_contract_sha256": "output-fields",
    }

    assert _strict_cache_issue("shopping", data, names, tools) == ""
    assert data["raw_graph"] != data["graph"]

    paper_data = {
        **data,
        "graph_source": "local_relation_audit_supported_subset",
        "review_policy": "not_required_for_paper_baseline",
        "pair_audits": [],
        "audited_pair_count": 0,
        "audit_complete": False,
    }
    assert _strict_cache_issue(
        "shopping",
        paper_data,
        names,
        tools,
        paper_baseline=True,
    ) == ""


def test_fresh_classification_always_runs_raw_pass_and_independent_review() -> None:
    orchestrator = TaskOrchestrator.__new__(TaskOrchestrator)
    orchestrator.client = _NoneRelationClient()

    classification = orchestrator._classify_edges_llm(TOOLS, "probe")

    assert classification is not None
    assert orchestrator.client.calls == 2


def test_paper_baseline_classifies_each_pair_once_without_local_review_gate() -> None:
    orchestrator = TaskOrchestrator.__new__(TaskOrchestrator)
    orchestrator.client = _NoneRelationClient()
    orchestrator.prompt_profile = resolve_prompt_profile("paper_generation_baseline_v1")

    classification = orchestrator._classify_edges_llm(TOOLS, "probe")

    assert classification is not None
    graph, pairs, audits = classification
    assert orchestrator.client.calls == 1
    assert audits == []
    assert graph == TaskOrchestrator._graph_from_pair_classifications(
        pairs, ["tool_a", "tool_b"],
    )


def test_paper_baseline_cache_loads_complete_raw_ledger_without_audits(
    tmp_path, monkeypatch,
) -> None:
    _install_complete_probe_contract(monkeypatch, TOOLS)
    monkeypatch.setattr(
        TaskOrchestrator,
        "DEPENDENCY_CACHE_ROOT",
        tmp_path / "data" / "dependency_graphs",
    )
    orchestrator = TaskOrchestrator.__new__(TaskOrchestrator)
    orchestrator.client = _NoneRelationClient()
    orchestrator.prompt_profile = resolve_prompt_profile("paper_generation_baseline_v1")
    schema_hash = orchestrator._tool_schema_hash(TOOLS, "probe")
    graph, pairs, audits = orchestrator._classify_edges_llm(TOOLS, "probe")

    assert orchestrator._save_dependency_cache(
        "probe", schema_hash, TOOLS, graph, pairs, audits,
    )
    eligible = orchestrator._eligible_graph_from_relation_audits(
        pairs,
        orchestrator._build_local_relation_audits(pairs, TOOLS, "probe"),
        sorted(tool["name"] for tool in TOOLS),
    )
    assert orchestrator._load_dependency_cache(
        "probe", schema_hash, TOOLS,
    ) == eligible
    payload = json.loads(
        orchestrator._graph_cache_path(
            "probe", schema_hash,
            orchestrator._classifier_contract_hash("probe"),
        ).read_text()
    )
    assert payload["classification_complete"] is True
    assert payload["audit_complete"] is False
    assert payload["review_policy"] == "not_required_for_paper_baseline"


def test_paper_baseline_reuses_raw_ledger_across_local_audit_change(
    tmp_path, monkeypatch,
) -> None:
    _install_complete_probe_contract(monkeypatch, TOOLS)
    monkeypatch.setattr(
        TaskOrchestrator,
        "DEPENDENCY_CACHE_ROOT",
        tmp_path / "data" / "dependency_graphs",
    )
    orchestrator = TaskOrchestrator.__new__(TaskOrchestrator)
    orchestrator.client = _NoneRelationClient()
    orchestrator.prompt_profile = resolve_prompt_profile("paper_generation_baseline_v1")
    schema_hash = orchestrator._tool_schema_hash(TOOLS, "probe")
    graph, pairs, audits = orchestrator._classify_edges_llm(TOOLS, "probe")
    assert orchestrator._save_dependency_cache(
        "probe", schema_hash, TOOLS, graph, pairs, audits,
    )
    cache_path = orchestrator._graph_cache_path(
        "probe", schema_hash,
        orchestrator._classifier_contract_hash("probe"),
    )
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    payload["graph_source"] = "obsolete_local_audit_contract"
    cache_path.write_text(json.dumps(payload), encoding="utf-8")

    assert orchestrator._load_dependency_cache(
        "probe", schema_hash, TOOLS,
    ) is None
    reused = orchestrator._load_raw_dependency_provenance(
        "probe", schema_hash, TOOLS,
    )
    assert reused is not None
    raw_graph, reused_pairs = reused
    assert raw_graph == graph
    assert reused_pairs == pairs


def test_paper_baseline_migrates_legacy_semantics_mixed_schema_hash(
    tmp_path, monkeypatch,
) -> None:
    _install_complete_probe_contract(monkeypatch, TOOLS)
    monkeypatch.setattr(
        TaskOrchestrator,
        "DEPENDENCY_CACHE_ROOT",
        tmp_path / "data" / "dependency_graphs",
    )
    orchestrator = TaskOrchestrator.__new__(TaskOrchestrator)
    orchestrator.client = _NoneRelationClient()
    orchestrator.prompt_profile = resolve_prompt_profile(
        "paper_generation_baseline_v1"
    )
    schema_hash = orchestrator._tool_schema_hash(TOOLS, "probe")
    graph, pairs, audits = orchestrator._classify_edges_llm(TOOLS, "probe")
    assert orchestrator._save_dependency_cache(
        "probe", schema_hash, TOOLS, graph, pairs, audits,
    )
    current_path = orchestrator._graph_cache_path(
        "probe", schema_hash,
        orchestrator._classifier_contract_hash("probe"),
    )
    payload = json.loads(current_path.read_text(encoding="utf-8"))
    legacy_hash = orchestrator._legacy_tool_schema_hash(TOOLS, 29)
    payload["schema_hash"] = legacy_hash
    payload["dependency_semantics_version"] = 29
    legacy_path = current_path.with_name(f"probe_{legacy_hash}_legacy.json")
    legacy_path.write_text(json.dumps(payload), encoding="utf-8")
    current_path.unlink()

    reused = orchestrator._load_raw_dependency_provenance(
        "probe", schema_hash, TOOLS,
    )
    assert reused is not None
    assert reused[0] == graph
    assert reused[1] == pairs

    changed_tools = [dict(tool) for tool in TOOLS]
    changed_tools[0] = {**changed_tools[0], "description": "changed schema"}
    changed_hash = orchestrator._tool_schema_hash(changed_tools, "probe")
    assert orchestrator._load_raw_dependency_provenance(
        "probe", changed_hash, changed_tools,
    ) is None


def test_raw_ledger_reuse_rejects_teacher_identity_change(
    tmp_path, monkeypatch,
) -> None:
    _install_complete_probe_contract(monkeypatch, TOOLS)
    monkeypatch.setattr(
        TaskOrchestrator,
        "DEPENDENCY_CACHE_ROOT",
        tmp_path / "data" / "dependency_graphs",
    )
    orchestrator = TaskOrchestrator.__new__(TaskOrchestrator)
    orchestrator.client = _NoneRelationClient()
    orchestrator.prompt_profile = resolve_prompt_profile("paper_generation_baseline_v1")
    schema_hash = orchestrator._tool_schema_hash(TOOLS, "probe")
    graph, pairs, audits = orchestrator._classify_edges_llm(TOOLS, "probe")
    assert orchestrator._save_dependency_cache(
        "probe", schema_hash, TOOLS, graph, pairs, audits,
    )

    orchestrator.client = SimpleNamespace(model_path="different-teacher")
    assert orchestrator._load_raw_dependency_provenance(
        "probe", schema_hash, TOOLS,
    ) is None


def test_paper_baseline_cache_keeps_uncertain_pairs_out_without_rebuild(
    tmp_path, monkeypatch,
) -> None:
    monkeypatch.setattr(
        TaskOrchestrator,
        "DEPENDENCY_CACHE_ROOT",
        tmp_path / "data" / "dependency_graphs",
    )
    orchestrator = TaskOrchestrator.__new__(TaskOrchestrator)
    orchestrator.client = _NoneRelationClient()
    orchestrator.prompt_profile = resolve_prompt_profile("paper_generation_baseline_v1")
    schema_hash = orchestrator._tool_schema_hash(TOOLS, "probe")
    graph, pairs, audits = orchestrator._classify_edges_llm(TOOLS, "probe")

    assert orchestrator._save_dependency_cache(
        "probe", schema_hash, TOOLS, graph, pairs, audits,
    )
    payload = json.loads(
        orchestrator._graph_cache_path(
            "probe", schema_hash,
            orchestrator._classifier_contract_hash("probe"),
        ).read_text()
    )
    assert payload["relation_audit_counts"] == {"insufficient_evidence": 1}
    assert orchestrator._load_dependency_cache(
        "probe", schema_hash, TOOLS,
    ) == {
        "tool_a": {"explicit": [], "implicit": []},
        "tool_b": {"explicit": [], "implicit": []},
    }


def test_dependency_cache_env_root_is_project_anchored_and_isolated(
    monkeypatch,
) -> None:
    monkeypatch.setattr(TaskOrchestrator, "DEPENDENCY_CACHE_ROOT", None)
    monkeypatch.setenv(
        "LIVEMCP_DEPENDENCY_CACHE_ROOT",
        "data/runs/gray/cache",
    )

    assert TaskOrchestrator._graph_cache_path(
        "shopping", "schema",
    ) == project_root() / "data/runs/gray/cache/shopping_schema.json"


def test_dependency_cache_path_namespaces_classifier_contracts() -> None:
    first = TaskOrchestrator._graph_cache_path(
        "shopping", "schema", "contract_a",
    )
    second = TaskOrchestrator._graph_cache_path(
        "shopping", "schema", "contract_b",
    )
    unnamespaced = TaskOrchestrator._graph_cache_path("shopping", "schema")

    assert first.name == "shopping_schema_contract_a.json"
    assert second.name == "shopping_schema_contract_b.json"
    assert len({first, second, unnamespaced}) == 3


def test_atomic_cache_save_publishes_only_complete_json(tmp_path, monkeypatch) -> None:
    _install_complete_probe_contract(monkeypatch, TOOLS)
    cache_root = tmp_path / "data" / "dependency_graphs"
    monkeypatch.setattr(TaskOrchestrator, "DEPENDENCY_CACHE_ROOT", cache_root)
    orchestrator = TaskOrchestrator.__new__(TaskOrchestrator)
    orchestrator.client = _NoneRelationClient()
    schema_hash = orchestrator._tool_schema_hash(TOOLS)
    cache_path = orchestrator._graph_cache_path(
        "probe", schema_hash,
        orchestrator._classifier_contract_hash("probe"),
    )
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
    orchestrator._save_dependency_cache(
        "probe", schema_hash, TOOLS, graph, pairs,
        _self_agreeing_pair_audits(pairs, TOOLS, "probe"),
    )

    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    assert replace_observations[0]["old"] == {"sentinel": "old"}
    assert replace_observations[0]["new"]["classification_complete"] is True
    assert payload["classified_pair_count"] == 1
    assert len(payload["pair_classifications"]) == 1
    assert payload["classifier_contract_hash"] == orchestrator._classifier_contract_hash(
        "probe"
    )
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


def test_cold_cache_is_classified_once_across_processes(tmp_path, monkeypatch) -> None:
    _install_complete_probe_contract(monkeypatch, TOOLS)
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
    assert counter.value == 2

    cache_files = list((tmp_path / "data" / "dependency_graphs").glob("probe_*.json"))
    assert len(cache_files) == 1
    payload = json.loads(cache_files[0].read_text(encoding="utf-8"))
    assert payload["classification_complete"] is True
    assert payload["classified_pair_count"] == 1
    assert len(payload["pair_classifications"]) == 1


def test_production_cache_separates_raw_classification_from_eligible_graph(tmp_path, monkeypatch) -> None:
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
    _install_complete_probe_contract(monkeypatch, tools)
    orchestrator = TaskOrchestrator.__new__(TaskOrchestrator)
    orchestrator.client = _NoneRelationClient()
    schema_hash = orchestrator._tool_schema_hash(tools, "probe")
    cache_path = orchestrator._graph_cache_path(
        "probe", schema_hash,
        orchestrator._classifier_contract_hash("probe"),
    )
    cache_path.parent.mkdir(parents=True)
    raw_graph = {
        "pay_invoice": {"explicit": [], "implicit": ["cancel_payment"]},
        "cancel_payment": {"explicit": [], "implicit": []},
    }
    pairs = orchestrator._pair_classifications_from_graph(
        raw_graph, sorted(t["name"] for t in tools),
    )
    graph = {
        "pay_invoice": {"explicit": [], "implicit": []},
        "cancel_payment": {"explicit": [], "implicit": []},
    }
    orchestrator._save_dependency_cache(
        "probe", schema_hash, tools, graph, pairs,
        _self_agreeing_pair_audits(pairs, tools, "probe"),
    )

    loaded = orchestrator._load_dependency_cache(
        "probe", schema_hash, tools,
    )
    assert loaded is not None
    assert loaded["pay_invoice"]["implicit"] == []
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    assert payload["raw_graph"] == raw_graph
    assert payload["graph"] == graph


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


def test_classifier_prompt_separates_explicit_data_flow_from_implicit_state() -> None:
    prompt = TaskOrchestrator._dependency_classifier_system_prompt()
    assert "schedule_transfer → cancel_transfer" not in prompt
    assert "explicit data-flow edge remains explicit" in prompt
    assert "Use the pre-existing-state test only for implicit edges" in prompt
    assert "Apply explicit bindings exhaustively and consistently" in prompt


def test_classifier_preserves_raw_none_and_relation_audit_records_missed_binding() -> None:
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
    assert "Novel/discovered output fields: account_id" in user_prompt
    assert "get_balance → list_accounts" in user_prompt
    assert (
        "list_accounts.account_id -> get_balance.account_id"
        in user_prompt
    )
    assert len(orchestrator.client.messages) == 2
    assert classification[1][0]["relation"] == "none"
    assert classification[2][0]["initial"]["relation"] == "none"
    assert classification[2][0]["review"]["relation"] == "explicit"
    assert classification[2][0]["tie_break"] is None
    relation_audit = TaskOrchestrator._build_local_relation_audits(
        classification[1], tools, "banking",
    )[0]
    assert relation_audit["verdict"] == "contradicted"
    assert relation_audit["eligible"] is False
    audit_prompt = orchestrator.client.messages[1][1]["content"]
    assert "list_accounts.account_id -> get_balance.account_id" in audit_prompt
    assert "Conflicting Independent Decisions" not in audit_prompt


def test_pair_review_is_diagnostic_and_not_rewritten_by_contract_feedback() -> None:
    class _StableFalseExplicitUntilFeedbackClient:
        model_path = "test-teacher"

        def __init__(self) -> None:
            self.messages = []

        def generate_chat(self, messages, **_kwargs) -> str:
            self.messages.append(messages)
            user = messages[1]["content"]
            if len(self.messages) == 1 or 'only valid answer is relation="none"' in user:
                relation = "none"
            else:
                relation = "explicit"
            return json.dumps({"classifications": [{
                "pair": "get_cart ↔ get_recommendations",
                "source": "get_cart",
                "target": "get_recommendations",
                "relation": relation,
            }]})

    tools = [
        tool for tool in SHOPPING_TOOLS
        if tool["name"] in {"get_cart", "get_recommendations"}
    ]
    orchestrator = TaskOrchestrator.__new__(TaskOrchestrator)
    orchestrator.client = _StableFalseExplicitUntilFeedbackClient()

    classification = orchestrator._classify_edges_llm(tools, "shopping")

    assert classification is not None
    _, pairs, audits = classification
    assert len(orchestrator.client.messages) == 2
    assert pairs[0] == {
        "pair": ["get_cart", "get_recommendations"],
        "source": "",
        "target": "",
        "relation": "none",
    }
    assert audits[0]["initial"]["relation"] == "none"
    assert audits[0]["review"]["relation"] == "explicit"
    assert audits[0]["disagrees_with_raw"] is True
    assert audits[0]["tie_break"] is None


def test_adversarial_review_cannot_overwrite_raw_batch_classification() -> None:
    class _ExplicitThenNoneClient:
        model_path = "test-teacher"

        def __init__(self) -> None:
            self.calls = 0

        def generate_chat(self, *_args, **_kwargs) -> str:
            self.calls += 1
            relation = "explicit" if self.calls == 1 else "none"
            return json.dumps({"classifications": [{
                "pair": "tool_a ↔ tool_b",
                "source": "tool_a" if relation == "explicit" else "",
                "target": "tool_b" if relation == "explicit" else "",
                "relation": relation,
            }]})

    orchestrator = TaskOrchestrator.__new__(TaskOrchestrator)
    orchestrator.client = _ExplicitThenNoneClient()

    classification = orchestrator._classify_edges_llm(
        DEPENDENT_TOOLS, "probe",
    )

    assert classification is not None
    graph, pairs, audits = classification
    assert orchestrator.client.calls == 2
    assert pairs[0]["relation"] == "explicit"
    assert graph["tool_a"]["explicit"] == []
    assert audits[0]["initial"]["relation"] == "explicit"
    assert audits[0]["review"]["relation"] == "none"
    assert audits[0]["tie_break"] is None
    assert audits[0]["final"]["relation"] == "explicit"
    assert audits[0]["disagrees_with_raw"] is True


def test_pair_audit_disagreement_does_not_request_a_third_decision() -> None:
    class _ThreeWayDisagreementClient:
        model_path = "test-teacher"

        def __init__(self) -> None:
            self.calls = 0

        def generate_chat(self, *_args, **_kwargs) -> str:
            self.calls += 1
            if self.calls == 1:
                source, target, relation = "tool_a", "tool_b", "explicit"
            elif self.calls == 2:
                source = target = ""
                relation = "none"
            else:
                source, target, relation = "tool_b", "tool_a", "implicit"
            return json.dumps({"classifications": [{
                "pair": "tool_a ↔ tool_b",
                "source": source,
                "target": target,
                "relation": relation,
            }]})

    orchestrator = TaskOrchestrator.__new__(TaskOrchestrator)
    orchestrator.client = _ThreeWayDisagreementClient()

    classification = orchestrator._classify_edges_llm(
        DEPENDENT_TOOLS, "probe",
    )
    assert classification is not None
    graph, pairs, audits = classification
    assert orchestrator.client.calls == 2
    assert pairs[0]["relation"] == "explicit"
    assert graph["tool_a"]["explicit"] == []
    assert audits[0]["review"]["relation"] == "none"
    assert audits[0]["final"] == audits[0]["initial"]
    assert audits[0]["disagrees_with_raw"] is True


def test_none_relation_is_rejected_by_certified_output_binding() -> None:
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
    assert "list_accounts -> get_balance" in issue


def test_fixed_state_contradiction_blocks_syntactic_explicit_binding() -> None:
    tools = {tool["name"]: tool for tool in SHOPPING_TOOLS}

    explicit_issue = TaskOrchestrator._pair_classification_contract_issue(
        {
            "pair": ["checkout", "return_order"],
            "source": "checkout",
            "target": "return_order",
            "relation": "explicit",
        },
        tools,
        "shopping",
    )
    assert explicit_issue is not None
    assert "placed" in explicit_issue
    assert "shipped" in explicit_issue

    none_issue = TaskOrchestrator._pair_classification_contract_issue(
        {
            "pair": ["checkout", "return_order"],
            "source": "",
            "target": "",
            "relation": "none",
        },
        tools,
        "shopping",
    )
    assert none_issue is None
    assert (
        TaskOrchestrator._dependency_pair_certified_explicit_directions(
            ("checkout", "return_order"), tools, "shopping",
        )
        == []
    )

    # add_to_wishlist is idempotent for an existing member.  This is not a
    # hard execution contradiction; live state-change/task suitability decides
    # whether the resulting no-op is useful.
    wishlist_issue = TaskOrchestrator._pair_classification_contract_issue(
        {
            "pair": ["add_to_wishlist", "get_wishlist"],
            "source": "get_wishlist",
            "target": "add_to_wishlist",
            "relation": "explicit",
        },
        tools,
        "shopping",
    )
    assert wishlist_issue is None
    assert (
        "get_wishlist", "add_to_wishlist",
    ) in TaskOrchestrator._dependency_pair_certified_explicit_directions(
        ("add_to_wishlist", "get_wishlist"), tools, "shopping",
    )
    implicit_issue = TaskOrchestrator._pair_classification_contract_issue(
        {
            "pair": ["add_to_wishlist", "get_wishlist"],
            "source": "add_to_wishlist",
            "target": "get_wishlist",
            "relation": "implicit",
        },
        tools,
        "shopping",
    )
    # This helper belongs to local eligibility auditing, not the raw parser.
    # The raw label remains preserved elsewhere, while the factual reverse
    # explicit binding prevents this implicit direction entering the graph.
    assert implicit_issue is not None
    assert "explicit-precedence" in implicit_issue
    none_issue = TaskOrchestrator._pair_classification_contract_issue(
        {
            "pair": ["add_to_wishlist", "get_wishlist"],
            "source": "",
            "target": "",
            "relation": "none",
        },
        tools,
        "shopping",
    )
    assert none_issue is not None
    assert "get_wishlist -> add_to_wishlist" in none_issue


def test_shopping_list_orders_does_not_bypass_get_order_for_products() -> None:
    tools = {tool["name"]: tool for tool in SHOPPING_TOOLS}

    assert (
        output_contracts.DOMAIN_VALUE_OUTPUT_FIELDS["shopping"][
            "list_orders"
        ]
        == ("order_id",)
    )
    assert (
        TaskOrchestrator._dependency_pair_certified_explicit_directions(
            ("list_orders", "compare_products"), tools, "shopping",
        )
        == []
    )
    assert (
        TaskOrchestrator._dependency_pair_certified_explicit_directions(
            ("get_order", "list_orders"), tools, "shopping",
        )
        == [("list_orders", "get_order")]
    )


def test_certified_binding_takes_precedence_over_reverse_implicit() -> None:
    tools = {tool["name"]: tool for tool in SHOPPING_TOOLS}
    issue = TaskOrchestrator._pair_classification_contract_issue(
        {
            "pair": ["add_to_cart", "get_cart"],
            "source": "add_to_cart",
            "target": "get_cart",
            "relation": "implicit",
        },
        tools,
        "shopping",
    )

    assert issue is not None
    assert "get_cart -> add_to_cart" in issue


def test_shopping_v19_manual_counterexamples_are_fail_closed() -> None:
    tools = {tool["name"]: tool for tool in SHOPPING_TOOLS}
    rejected = [
        {
            "pair": ["add_to_cart", "get_cart"],
            "source": "add_to_cart",
            "target": "get_cart",
            "relation": "implicit",
        },
        {
            # return_order exposes product_id values that can ground the
            # required product_id of add_to_cart in the reverse direction.
            "pair": ["add_to_cart", "return_order"],
            "source": "",
            "target": "",
            "relation": "none",
        },
        {
            "pair": ["add_to_wishlist", "checkout"],
            "source": "checkout",
            "target": "add_to_wishlist",
            "relation": "explicit",
        },
        {
            "pair": ["checkout", "return_order"],
            "source": "checkout",
            "target": "return_order",
            "relation": "explicit",
        },
    ]
    for entry in rejected:
        assert TaskOrchestrator._pair_classification_contract_issue(
            entry, tools, "shopping",
        ) is not None

    accepted = [
        {
            "pair": ["add_to_cart", "get_cart"],
            "source": "get_cart",
            "target": "add_to_cart",
            "relation": "explicit",
        },
        {
            "pair": ["add_to_cart", "return_order"],
            "source": "return_order",
            "target": "add_to_cart",
            "relation": "explicit",
        },
        {
            "pair": ["add_to_wishlist", "checkout"],
            "source": "",
            "target": "",
            "relation": "none",
        },
        {
            "pair": ["checkout", "return_order"],
            "source": "",
            "target": "",
            "relation": "none",
        },
    ]
    for entry in accepted:
        assert TaskOrchestrator._pair_classification_contract_issue(
            entry, tools, "shopping",
        ) is None


def test_explicit_contract_supports_factual_argument_aliases() -> None:
    from src.live_mcp.servers.banking.server import TOOLS as banking_tools

    tools = {tool["name"]: tool for tool in banking_tools}
    issue = TaskOrchestrator._pair_classification_contract_issue(
        {
            "pair": ["list_accounts", "transfer"],
            "source": "list_accounts",
            "target": "transfer",
            "relation": "explicit",
        },
        tools,
        "banking",
    )

    assert issue is None

    shopping_tools = {tool["name"]: tool for tool in SHOPPING_TOOLS}
    shopping_issue = TaskOrchestrator._pair_classification_contract_issue(
        {
            "pair": ["compare_products", "search_products"],
            "source": "search_products",
            "target": "compare_products",
            "relation": "explicit",
        },
        shopping_tools,
        "shopping",
    )
    assert shopping_issue is None


def test_shopping_echoed_required_identifiers_do_not_create_explicit_edges() -> None:
    tools = {tool["name"]: tool for tool in SHOPPING_TOOLS}

    for source_name, target_name in (
        ("get_product", "update_cart_quantity"),
        ("get_reviews", "add_to_cart"),
        ("cancel_order", "get_order"),
        ("add_review", "add_to_cart"),
        ("compare_products", "get_product"),
    ):
        explicit = {
            "pair": sorted([source_name, target_name]),
            "source": source_name,
            "target": target_name,
            "relation": "explicit",
        }
        issue = TaskOrchestrator._pair_classification_contract_issue(
            explicit, tools, "shopping",
        )
        assert issue is not None
        assert "no novel output field" in issue

        none = {
            "pair": sorted([source_name, target_name]),
            "source": "",
            "target": "",
            "relation": "none",
        }
        assert TaskOrchestrator._pair_classification_contract_issue(
            none, tools, "shopping",
        ) is None


def test_shopping_local_relation_audit_separates_raw_and_eligible_edges() -> None:
    tools_by_name = {tool["name"]: tool for tool in SHOPPING_TOOLS}
    cases = [
        ({
            "pair": ["get_product", "update_cart_quantity"],
            "source": "get_product",
            "target": "update_cart_quantity",
            "relation": "explicit",
        }, "contradicted", False),
        ({
            "pair": ["get_product", "search_products"],
            "source": "search_products",
            "target": "get_product",
            "relation": "explicit",
        }, "supported", True),
        ({
            "pair": ["add_review", "checkout"],
            "source": "checkout",
            "target": "add_review",
            "relation": "implicit",
        }, "contradicted", False),
        ({
            "pair": ["add_to_cart", "checkout"],
            "source": "add_to_cart",
            "target": "checkout",
            "relation": "implicit",
        }, "supported", True),
        ({
            "pair": ["apply_coupon", "get_coupons"],
            "source": "",
            "target": "",
            "relation": "none",
        }, "contradicted", False),
        ({
            "pair": ["get_coupons", "list_categories"],
            "source": "",
            "target": "",
            "relation": "none",
        }, "supported", False),
    ]

    for raw, expected_verdict, expected_eligible in cases:
        audit = TaskOrchestrator._local_dependency_relation_audit(
            raw, tools_by_name, "shopping",
        )
        assert audit["raw"] == {
            "source": raw["source"],
            "target": raw["target"],
            "relation": raw["relation"],
        }
        assert audit["verdict"] == expected_verdict
        assert audit["eligible"] is expected_eligible


def test_shopping_relation_contract_coverage_matches_all_public_tools() -> None:
    names = {tool["name"] for tool in SHOPPING_TOOLS}
    assert set(output_contracts.DOMAIN_VALUE_OUTPUT_FIELDS["shopping"]) == names
    assert set(state_contracts.DOMAIN_STATE_FACTS["shopping"]) == names


def test_calendar_relation_contract_coverage_matches_all_public_tools() -> None:
    names = {tool["name"] for tool in CALENDAR_TOOLS}
    assert set(output_contracts.DOMAIN_VALUE_OUTPUT_FIELDS["calendar"]) == names
    assert set(state_contracts.DOMAIN_STATE_FACTS["calendar"]) == names


def test_calendar_single_event_cannot_feed_recurring_info() -> None:
    tools_by_name = {tool["name"]: tool for tool in CALENDAR_TOOLS}
    bad = {
        "pair": ["create_event", "get_recurring_info"],
        "source": "create_event",
        "target": "get_recurring_info",
        "relation": "explicit",
    }
    good = {
        "pair": ["create_recurring", "get_recurring_info"],
        "source": "create_recurring",
        "target": "get_recurring_info",
        "relation": "explicit",
    }

    assert TaskOrchestrator._local_dependency_relation_audit(
        bad, tools_by_name, "calendar",
    )["verdict"] == "contradicted"
    assert TaskOrchestrator._local_dependency_relation_audit(
        good, tools_by_name, "calendar",
    )["verdict"] == "supported"
    registry = build_contract_registry({"calendar": CALENDAR_TOOLS})
    assert simulate_symbolic_chain(
        registry, "calendar", ["create_event", "get_recurring_info"],
    )[1]
    assert not simulate_symbolic_chain(
        registry, "calendar", ["create_recurring", "get_recurring_info"],
    )[1]


def test_calendar_recurring_info_requires_live_recurring_event() -> None:
    empty_context = {
        "entity_ids": [],
        "entity_records": [],
        "probe_results": [],
    }
    assert chain_is_feasible(
        ["get_recurring_info", "get_event"], "calendar", empty_context,
        build_contract_registry({"calendar": CALENDAR_TOOLS}),
    ) == (False, "get_recurring_info requires missing entity types ['event(0/1)']")

    recurring_context = {
        "entity_ids": [{"type": "event", "id": "evt-recurring"}],
        "entity_records": [{
            "type": "event",
            "id": "evt-recurring",
            "data": {"recurrence": "FREQ=WEEKLY;BYDAY=MO"},
        }],
        "probe_results": [],
    }
    assert chain_is_feasible(
        ["get_recurring_info", "get_event"], "calendar", recurring_context,
        build_contract_registry({"calendar": CALENDAR_TOOLS}),
    )[0] is True


def test_shopping_cache_round_trip_keeps_bad_raw_edges_out_of_eligible_graph(
    tmp_path, monkeypatch,
) -> None:
    monkeypatch.setattr(
        TaskOrchestrator,
        "DEPENDENCY_CACHE_ROOT",
        tmp_path / "dependency_graphs",
    )
    names = sorted(tool["name"] for tool in SHOPPING_TOOLS)
    ledger = [
        {
            "pair": [names[i], names[j]],
            "source": "",
            "target": "",
            "relation": "none",
        }
        for i in range(len(names))
        for j in range(i + 1, len(names))
    ]
    by_pair = {tuple(entry["pair"]): entry for entry in ledger}

    def set_raw(source: str, target: str, relation: str) -> None:
        pair = tuple(sorted((source, target)))
        by_pair[pair].update({
            "source": source,
            "target": target,
            "relation": relation,
        })

    set_raw("get_product", "update_cart_quantity", "explicit")
    set_raw("checkout", "add_review", "implicit")
    set_raw("search_products", "get_product", "explicit")
    set_raw("add_to_cart", "checkout", "implicit")

    orchestrator = TaskOrchestrator.__new__(TaskOrchestrator)
    orchestrator.client = _NoneRelationClient()
    relation_audits = orchestrator._build_local_relation_audits(
        ledger, SHOPPING_TOOLS, "shopping",
    )
    graph = orchestrator._eligible_graph_from_relation_audits(
        ledger, relation_audits, names,
    )
    raw_graph = orchestrator._graph_from_pair_classifications(ledger, names)

    assert raw_graph["get_product"]["explicit"] == ["update_cart_quantity"]
    assert raw_graph["checkout"]["implicit"] == ["add_review"]
    assert graph["get_product"]["explicit"] == []
    assert graph["checkout"]["implicit"] == []
    assert graph["search_products"]["explicit"] == ["get_product"]
    assert graph["add_to_cart"]["implicit"] == ["checkout"]

    schema_hash = orchestrator._tool_schema_hash(SHOPPING_TOOLS, "shopping")
    assert orchestrator._save_dependency_cache(
        "shopping",
        schema_hash,
        SHOPPING_TOOLS,
        graph,
        ledger,
        _self_agreeing_pair_audits(ledger, SHOPPING_TOOLS, "shopping"),
    )
    assert orchestrator._load_dependency_cache(
        "shopping", schema_hash, SHOPPING_TOOLS,
    ) == graph

    payload = json.loads(
        orchestrator._graph_cache_path(
            "shopping", schema_hash,
            orchestrator._classifier_contract_hash("shopping"),
        ).read_text()
    )
    assert payload["raw_graph"] == raw_graph
    assert payload["graph"] == graph
    assert payload["relation_audit_complete"] is True
    from scripts.validate_generation_pipeline import _strict_cache_issue
    assert _strict_cache_issue(
        "shopping", payload, names, SHOPPING_TOOLS,
    ) == ""


def test_payments_get_invoice_uses_only_novel_output_bindings() -> None:
    tools = {
        tool["name"]: tool
        for tool in PAYMENT_TOOLS
    }
    for target_name in ("pay_invoice", "refund_invoice", "cancel_payment"):
        issue = TaskOrchestrator._pair_classification_contract_issue(
            {
                "pair": sorted(["get_invoice", target_name]),
                "source": "",
                "target": "",
                "relation": "none",
            },
            tools,
            "payments",
        )
        assert issue is not None
        assert "certified novel-output" in issue

    # invoice_id is merely echoed and dispute_invoice does not consume any
    # other novel get_invoice output, so none remains valid for this pair.
    dispute_issue = TaskOrchestrator._pair_classification_contract_issue(
        {
            "pair": sorted(["get_invoice", "dispute_invoice"]),
            "source": "",
            "target": "",
            "relation": "none",
        },
        tools,
        "payments",
    )
    assert dispute_issue is None

    create_issue = TaskOrchestrator._pair_classification_contract_issue(
        {
            "pair": ["create_invoice", "get_invoice"],
            "source": "",
            "target": "",
            "relation": "none",
        },
        tools,
        "payments",
    )
    assert create_issue is not None
    assert "create_invoice -> get_invoice" in create_issue


def test_explicit_relation_rejects_echoed_source_required_field() -> None:
    tools = {tool["name"]: tool for tool in PAYMENT_TOOLS}
    issue = TaskOrchestrator._pair_classification_contract_issue(
        {
            "pair": sorted(["get_invoice", "dispute_invoice"]),
            "source": "get_invoice",
            "target": "dispute_invoice",
            "relation": "explicit",
        },
        tools,
        "payments",
    )
    assert issue is not None
    assert "echoed source-required fields" in issue


def test_output_field_contract_invalidates_only_classifier_cache(monkeypatch) -> None:
    orchestrator = TaskOrchestrator.__new__(TaskOrchestrator)
    orchestrator.client = _NoneRelationClient()
    schema_hash = orchestrator._tool_schema_hash(TOOLS, "probe")
    classifier_hash = orchestrator._classifier_contract_hash("probe")
    expanded = {
        **output_contracts.DOMAIN_VALUE_OUTPUT_FIELDS,
        "probe": {"tool_a": ("value_id",)},
    }

    monkeypatch.setattr(
        output_contracts, "DOMAIN_VALUE_OUTPUT_FIELDS", expanded,
    )

    assert orchestrator._tool_schema_hash(TOOLS, "probe") == schema_hash
    assert orchestrator._classifier_contract_hash("probe") != classifier_hash


def test_argument_alias_contract_is_part_of_classifier_provenance(
    monkeypatch,
) -> None:
    before = TaskOrchestrator._dependency_output_field_contract_hash("shopping")
    expanded = {
        **value_bindings.OUTPUT_ARGUMENT_ALIASES,
        "shopping": {
            **value_bindings.OUTPUT_ARGUMENT_ALIASES[
                "shopping"
            ],
            "return_id": ("order_id",),
        },
    }
    monkeypatch.setattr(
        value_bindings, "OUTPUT_ARGUMENT_ALIASES", expanded,
    )

    after = TaskOrchestrator._dependency_output_field_contract_hash("shopping")

    assert after != before


def test_typed_state_contract_is_part_of_classifier_provenance(
    monkeypatch,
) -> None:
    before = TaskOrchestrator._dependency_output_field_contract_hash("shopping")
    checkout = state_contracts.DOMAIN_STATE_FACTS["shopping"]["checkout"]
    changed_predicate = replace(
        checkout.preconditions[0], observed_entity_required=True,
    )
    expanded = {
        **state_contracts.DOMAIN_STATE_FACTS,
        "shopping": {
            **state_contracts.DOMAIN_STATE_FACTS["shopping"],
            "checkout": replace(
                checkout, preconditions=(changed_predicate,),
            ),
        },
    }
    monkeypatch.setattr(
        state_contracts, "DOMAIN_STATE_FACTS", expanded,
    )

    assert (
        TaskOrchestrator._dependency_output_field_contract_hash("shopping")
        != before
    )


def test_stale_cache_contract_forces_full_domain_rebuild(
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
        _self_agreeing_pair_audits(pairs, tools, "probe"),
    )

    expanded = {
        **output_contracts.DOMAIN_VALUE_OUTPUT_FIELDS,
        "probe": {"tool_a": ("value",)},
    }
    monkeypatch.setattr(
        output_contracts, "DOMAIN_VALUE_OUTPUT_FIELDS", expanded,
    )

    assert orchestrator._load_dependency_cache(
        "probe", schema_hash, tools,
    ) is None
    assert ("probe", schema_hash) not in getattr(
        orchestrator, "_dependency_graph_repairs", {},
    )


def test_nonempty_output_contract_change_does_not_use_legacy_migration(
    tmp_path, monkeypatch,
) -> None:
    monkeypatch.setattr(
        TaskOrchestrator,
        "DEPENDENCY_CACHE_ROOT",
        tmp_path / "data" / "dependency_graphs",
    )
    first_contract = {
        **output_contracts.DOMAIN_VALUE_OUTPUT_FIELDS,
        "probe": {"tool_a": ("unrelated",)},
    }
    monkeypatch.setattr(
        output_contracts, "DOMAIN_VALUE_OUTPUT_FIELDS", first_contract,
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
        _self_agreeing_pair_audits(pairs, DEPENDENT_TOOLS, "probe"),
    )

    changed_contract = {
        **first_contract,
        "probe": {"tool_a": ("value",)},
    }
    monkeypatch.setattr(
        output_contracts, "DOMAIN_VALUE_OUTPUT_FIELDS", changed_contract,
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
    _install_complete_probe_contract(monkeypatch, TOOLS)
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
    orchestrator._save_dependency_cache(
        "probe", schema_hash, TOOLS, graph, pairs,
        _self_agreeing_pair_audits(pairs, TOOLS, "probe"),
    )
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


def test_classifier_contract_uses_stable_model_identity_over_serving_alias() -> None:
    orchestrator = TaskOrchestrator.__new__(TaskOrchestrator)
    orchestrator.client = SimpleNamespace(
        model_path="Gemma-4-31B-it",
        contract_model_id="models/Google/Gemma-4-31B-it",
    )

    contract = orchestrator._classifier_contract_payload("shopping")

    assert contract["teacher_model_id"] == "models/Google/Gemma-4-31B-it"


def test_live_task_persists_stable_teacher_model_identity() -> None:
    orchestrator = TaskOrchestrator.__new__(TaskOrchestrator)
    orchestrator.client = SimpleNamespace(
        model_path="http://127.0.0.1:8000/v1",
        contract_model_id="models/Google/Gemma-4-31B-it",
    )
    orchestrator.suite_config = SimpleNamespace(
        suite_name="probe", rollout={"max_turns": 8},
    )
    orchestrator.manager = SimpleNamespace(
        registry=SimpleNamespace(server_tools=lambda _server: TOOLS),
    )
    oracle = OracleProgram(task_id="probe-1", calls=[], success_criteria=[])

    task = orchestrator._to_live_task(
        "probe", "List the items.", "session-1", 7, TOOLS, oracle, [],
        "complete", "probe-1",
    )

    assert task.metadata["teacher_model_id"] == (
        "models/Google/Gemma-4-31B-it"
    )
