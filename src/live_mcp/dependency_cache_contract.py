"""Dependency classifier identity, schema, and cache validation contract."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from src.live_mcp.config import project_root
from src.live_mcp.dependency_value_flow import _dependency_argument_bindings
from src.live_mcp.domain_contracts import value_bindings
from src.live_mcp.domain_contracts.states import domain_state_fact_payload
from src.live_mcp.domain_contracts import outputs as output_contracts
from src.live_mcp.domain_contracts.probes import (
    _DOMAIN_ENTITY_ID_FIELD_TYPES,
    _ENTITY_ID_FIELD_TYPES,
)


class DependencyCacheContractMixin:
    DEPENDENCY_CACHE_VERSION: int = 8
    # v30 keeps observation novelty separate from state identity, binds each
    # chain input to the nearest compatible output, and preserves filesystem
    # type facts across copy/move.
    DEPENDENCY_SEMANTICS_VERSION: int = 30
    DEPENDENCY_PAIR_BATCH_SIZE: int = 2
    DEPENDENCY_CLASSIFICATION_MAX_TOKENS: int = 512
    DEPENDENCY_CACHE_ROOT: Path | None = None

    @staticmethod
    def _tool_schema_payload(server_tools: list[dict]) -> list[dict[str, Any]]:
        return [
            {
                "name": tool.get("name"),
                "description": tool.get("description", ""),
                "input_schema": tool.get("input_schema", {}),
                "annotations": tool.get("annotations", {}),
            }
            for tool in sorted(
                server_tools, key=lambda item: str(item.get("name", ""))
            )
        ]

    @classmethod
    def _tool_schema_hash(
        cls, server_tools: list[dict], server_name: str | None = None,
    ) -> str:
        """Hash the public tool schema independently of local semantics.

        Handler implementation changes affect live feasibility and execution,
        not the cached classifier output.
        """
        raw = json.dumps(
            {"schema": cls._tool_schema_payload(server_tools)},
            sort_keys=True,
            ensure_ascii=True,
            default=str,
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    @classmethod
    def _legacy_tool_schema_hash(
        cls, server_tools: list[dict], dependency_semantics_version: int,
    ) -> str:
        """Recompute the pre-v30 mixed hash for exact cache migration."""
        payload: dict[str, Any] = {
            "schema": cls._tool_schema_payload(server_tools),
            "dependency_semantics_version": dependency_semantics_version,
        }
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _dependency_classifier_system_prompt() -> str:
        """Return the exact Step-1 classifier contract included in cache provenance."""
        return (
            "You are analyzing tool dependencies for an MCP server. "
            "For each unordered tool pair {A, B}, decide whether a dependency "
            "exists and, if so, choose its source and target:\n"
            '- "explicit": source produces output that is a REQUIRED INPUT of target '
            "(e.g., source returns an entity ID that target needs as a parameter).\n"
            '- "implicit": source must execute BEFORE target to establish state, '
            "but source's output is not a direct input to target.\n"
            '- "none": neither listed direction is a required dependency.\n\n'
            "Classification rules:\n"
            "- If A creates/returns something that B's required parameters "
            "reference, mark explicit.\n"
            "- Mark implicit only when source establishes live server state that target "
            "cannot succeed without, such as create_draft → send_draft, "
            "or add_to_cart → checkout.\n"
            "- An explicit data-flow edge remains explicit when target could also be "
            "called with a user-provided or pre-existing value. Step 1 records that "
            "source output CAN supply a required target input; Step 2 separately checks "
            "whether a live source-produced entity satisfies target state preconditions.\n"
            "- A verified output-to-required-input binding is strong evidence for an "
            "explicit edge, but not proof when the source-produced entity can never "
            "satisfy a fixed target precondition stated by the tool contracts.\n"
            "- Apply explicit bindings exhaustively and consistently. If the same "
            "source output entity (for example product_id) can populate required "
            "inputs of several analogous target tools, do not arbitrarily label only "
            "some of those pairs explicit. Use none only when the field has a different "
            "business meaning or a stated fixed precondition makes the source-produced "
            "entity unusable.\n"
            "- Use the pre-existing-state test only for implicit edges: if target does "
            "not require source's state mutation, do not call their common workflow "
            "order implicit. Mere topical relevance is none.\n"
            "- A read-only A is not an implicit dependency merely because its result is "
            "helpful. It is explicit only when B requires a value produced by A.\n"
            "- A field that A merely echoes from A's own REQUIRED INPUT is not a new "
            "value produced by A and does not establish A → B. For example, "
            "get_invoice(invoice_id) returning the same invoice_id does not make it a "
            "required predecessor of another invoice_id tool. A newly discovered "
            "payment_id from that response may still establish a dependency.\n"
            "- Prefer explicit over implicit when both could apply.\n"
            "- Only mark implicit if there is a genuine required state dependency."
        )

    @staticmethod
    def _dependency_pair_audit_system_prompt() -> str:
        """Return the independent per-pair review contract."""
        return (
            DependencyCacheContractMixin._dependency_classifier_system_prompt()
            + "\n\n"
            "You are now the independent adversarial reviewer for exactly one "
            "unordered pair. Re-evaluate the pair from the tool contracts rather "
            "than copying a previous batch decision. A displayed verified binding "
            "has passed the factual output ledger, required-input, alias, and typed "
            "entity checks. Treat it as explicit unless it is visibly marked as "
            "blocked by a fixed target-state contradiction. Reject workflow "
            "convention as implicit unless "
            "the target truly cannot succeed before the source mutation. Return "
            "exactly one classification in the requested JSON format."
        )

    @staticmethod
    def _dependency_output_field_contract_hash(server_name: str = "") -> str:
        """Hash factual handler-output fields exposed only to Step-1 classification."""
        output_fields = output_contracts.DOMAIN_VALUE_OUTPUT_FIELDS.get(
            server_name, {},
        )
        aliases = value_bindings.OUTPUT_ARGUMENT_ALIASES.get(server_name, {})
        typed_state_contract = domain_state_fact_payload(server_name)
        if (
            not output_fields
            and not aliases
            and not typed_state_contract
        ):
            return ""
        raw = json.dumps(
            {
                "output_fields": output_fields,
                "argument_aliases": aliases,
                "entity_id_field_types": _ENTITY_ID_FIELD_TYPES,
                "domain_entity_id_field_types": (
                    _DOMAIN_ENTITY_ID_FIELD_TYPES.get(server_name, {})
                ),
                "typed_state_contract": typed_state_contract,
            },
            sort_keys=True,
            ensure_ascii=True,
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _classifier_contract_payload(
        self, server_name: str = "",
    ) -> dict[str, Any]:
        model_id = str(
            getattr(
                self.client,
                "contract_model_id",
                getattr(self.client, "model_path", "unknown"),
            )
        )
        paper_baseline = self._uses_paper_baseline()
        prompt = (
            self._dependency_classifier_system_prompt()
            if paper_baseline
            else "\n\n--- independent pair audit ---\n\n".join((
                self._dependency_classifier_system_prompt(),
                self._dependency_pair_audit_system_prompt(),
            ))
        )
        payload = {
            "teacher_model_id": model_id,
            "classifier_prompt_sha256": hashlib.sha256(
                prompt.encode("utf-8")
            ).hexdigest(),
            "dependency_semantics_version": self.DEPENDENCY_SEMANTICS_VERSION,
            "raw_graph_source": "first_structurally_valid_classification",
            # Runtime scheduling always consumes the locally executable graph.
            # The paper-faithful classifier output remains immutable in
            # ``raw_graph``; describing the filtered ``graph`` as raw would make
            # cache provenance contradict the persisted payload.
            "graph_source": "local_relation_audit_supported_subset",
            "review_policy": (
                "not_required_for_paper_baseline"
                if paper_baseline
                else "diagnostic_only"
            ),
            "tie_break_policy": "disabled",
        }
        output_contract_hash = self._dependency_output_field_contract_hash(
            server_name
        )
        if output_contract_hash:
            payload["output_field_contract_sha256"] = output_contract_hash
        return payload

    def _classifier_contract_hash(self, server_name: str = "") -> str:
        raw = json.dumps(
            self._classifier_contract_payload(server_name),
            sort_keys=True,
            ensure_ascii=True,
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    @classmethod
    def _graph_cache_path(
        cls,
        server_name: str,
        schema_hash: str,
        classifier_contract_hash: str = "",
    ) -> Path:
        # Anchor project artifacts to the checkout, independent of process cwd.
        # A gray run may isolate a new classifier contract without overwriting
        # the currently certified cache evidence.
        configured_root = os.environ.get("LIVEMCP_DEPENDENCY_CACHE_ROOT", "")
        if cls.DEPENDENCY_CACHE_ROOT is not None:
            root = cls.DEPENDENCY_CACHE_ROOT
        elif configured_root:
            root = Path(configured_root)
            if not root.is_absolute():
                root = project_root() / root
        else:
            root = project_root() / "data" / "dependency_graphs"
        contract_suffix = (
            f"_{classifier_contract_hash}"
            if classifier_contract_hash else ""
        )
        return root / f"{server_name}_{schema_hash}{contract_suffix}.json"

    def _dependency_cache_key(
        self,
        server_name: str,
        server_tools: list[dict] | None = None,
    ) -> tuple[str, str, str]:
        tools = (
            server_tools
            if server_tools is not None
            else self.manager.registry.server_tools(server_name)
        )
        return (
            server_name,
            self._tool_schema_hash(tools, server_name),
            self._classifier_contract_hash(server_name),
        )

    @staticmethod
    @contextmanager
    def _graph_cache_file_lock(cache_path: Path):
        """Serialize one domain/schema cache build across Python processes."""
        lock_path = cache_path.parent / ".locks" / f"{cache_path.name}.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a+", encoding="utf-8") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def _valid_cached_graph(
        graph: Any,
        expected_tool_names: list[str] | None = None,
    ) -> bool:
        if not isinstance(graph, dict) or not graph:
            return False
        expected_tool_set: set[str] | None = None
        if expected_tool_names is not None:
            if len(expected_tool_names) != len(set(expected_tool_names)):
                return False
            expected_tool_set = set(expected_tool_names)
            if set(graph.keys()) != expected_tool_set:
                return False
        for source_name, edges in graph.items():
            if not isinstance(edges, dict):
                return False
            relation_targets: dict[str, list[str]] = {}
            for relation in ("explicit", "implicit"):
                targets = edges.get(relation)
                if not isinstance(targets, list):
                    return False
                if len(targets) != len(set(targets)):
                    return False
                if expected_tool_set is not None and any(t not in expected_tool_set for t in targets):
                    return False
                if source_name in targets:
                    return False
                relation_targets[relation] = targets
            if set(relation_targets["explicit"]) & set(relation_targets["implicit"]):
                return False
        return True

    @staticmethod
    def _normalize_cached_graph(
        graph: dict,
        expected_tool_names: list[str],
    ) -> dict:
        expected_tool_set = set(expected_tool_names)
        normalized: dict[str, dict[str, list[str]]] = {}
        for tool_name in expected_tool_names:
            edge_groups = graph.get(tool_name, {}) if isinstance(graph, dict) else {}
            explicit: list[str] = []
            implicit: list[str] = []
            for relation, target_list in (("explicit", explicit), ("implicit", implicit)):
                raw_targets = edge_groups.get(relation, []) if isinstance(edge_groups, dict) else []
                if not isinstance(raw_targets, list):
                    continue
                for target in raw_targets:
                    if (
                        isinstance(target, str)
                        and target in expected_tool_set
                        and target != tool_name
                        and target not in target_list
                    ):
                        target_list.append(target)
            explicit_set = set(explicit)
            normalized[tool_name] = {
                "explicit": sorted(explicit),
                "implicit": sorted(
                    target for target in implicit if target not in explicit_set
                ),
            }
        return normalized

    @staticmethod
    def _expected_dependency_pairs(
        expected_tool_names: list[str],
    ) -> set[tuple[str, str]]:
        names = sorted(expected_tool_names)
        return {
            (names[i], names[j])
            for i in range(len(names))
            for j in range(i + 1, len(names))
        }

    @classmethod
    def _validate_pair_classifications(
        cls,
        pair_classifications: Any,
        expected_tool_names: list[str],
    ) -> list[dict[str, Any]] | None:
        """Validate and canonicalize the complete C(n,2) classification ledger."""
        if not isinstance(pair_classifications, list):
            return None
        expected_pairs = cls._expected_dependency_pairs(expected_tool_names)
        seen: dict[tuple[str, str], dict[str, Any]] = {}
        for entry in pair_classifications:
            if not isinstance(entry, dict):
                return None
            raw_pair = entry.get("pair")
            if (
                not isinstance(raw_pair, list)
                or len(raw_pair) != 2
                or not all(isinstance(name, str) for name in raw_pair)
            ):
                return None
            pair = tuple(sorted(raw_pair))
            if pair not in expected_pairs or pair in seen:
                return None
            relation = str(entry.get("relation", "")).strip().lower()
            if relation not in ("explicit", "implicit", "none"):
                return None
            source = str(entry.get("source") or "").strip()
            target = str(entry.get("target") or "").strip()
            if relation == "none":
                if source or target:
                    return None
            elif (
                source == target
                or {source, target} != set(pair)
            ):
                return None
            seen[pair] = {
                "pair": list(pair),
                "source": source,
                "target": target,
                "relation": relation,
            }
        if set(seen) != expected_pairs:
            return None
        return [seen[pair] for pair in sorted(seen)]
    @classmethod
    def _validate_dependency_pair_audits(
        cls,
        pair_audits: Any,
        pair_classifications: list[dict[str, Any]],
        server_tools: list[dict],
        server_name: str = "",
    ) -> list[dict[str, Any]] | None:
        """Validate independent-review provenance for every unordered pair."""
        if not isinstance(pair_audits, list):
            return None
        tools_by_name = {
            str(tool.get("name") or ""): tool for tool in server_tools
        }
        raw_by_pair = {
            tuple(entry["pair"]): entry for entry in pair_classifications
        }

        def _normalize_decision(
            raw: Any, pair: tuple[str, str],
        ) -> dict[str, str] | None:
            if not isinstance(raw, dict):
                return None
            relation = str(raw.get("relation") or "").strip().lower()
            source = str(raw.get("source") or "").strip()
            target = str(raw.get("target") or "").strip()
            if relation == "none":
                if source or target:
                    return None
            elif (
                relation not in ("explicit", "implicit")
                or source == target
                or {source, target} != set(pair)
            ):
                return None
            return {
                "source": source,
                "target": target,
                "relation": relation,
            }

        seen: dict[tuple[str, str], dict[str, Any]] = {}
        for raw_audit in pair_audits:
            if not isinstance(raw_audit, dict):
                return None
            raw_pair = raw_audit.get("pair")
            if (
                not isinstance(raw_pair, list)
                or len(raw_pair) != 2
                or not all(isinstance(name, str) for name in raw_pair)
            ):
                return None
            pair = tuple(sorted(raw_pair))
            if pair not in raw_by_pair or pair in seen:
                return None
            bindings = cls._dependency_pair_binding_candidates(
                pair, tools_by_name, server_name,
            )
            if raw_audit.get("binding_candidates") != bindings:
                return None
            initial = _normalize_decision(raw_audit.get("initial"), pair)
            review = _normalize_decision(raw_audit.get("review"), pair)
            final = _normalize_decision(raw_audit.get("final"), pair)
            if initial is None or review is None or final is None:
                return None
            tie_break_raw = raw_audit.get("tie_break")
            tie_break = (
                None
                if tie_break_raw is None
                else _normalize_decision(tie_break_raw, pair)
            )
            if tie_break_raw is not None and tie_break is None:
                return None
            expected_raw = raw_by_pair[pair]
            raw_decision = {
                "source": expected_raw["source"],
                "target": expected_raw["target"],
                "relation": expected_raw["relation"],
            }
            # Review is diagnostic provenance. It cannot select or overwrite
            # the raw classification used to build the dependency graph.
            if initial != raw_decision or final != raw_decision or tie_break is not None:
                return None
            disagrees_with_raw = review != raw_decision
            recorded_disagreement = raw_audit.get(
                "disagrees_with_raw", disagrees_with_raw,
            )
            if recorded_disagreement is not disagrees_with_raw:
                return None
            seen[pair] = {
                "pair": list(pair),
                "binding_candidates": bindings,
                "initial": initial,
                "review": review,
                "tie_break": None,
                "final": raw_decision,
                "disagrees_with_raw": disagrees_with_raw,
            }

        if set(seen) != set(raw_by_pair):
            return None
        return [seen[pair] for pair in sorted(seen)]
