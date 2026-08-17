"""Atomic persistence and strict reload of PROVE dependency caches."""

from __future__ import annotations

import json
import os
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any

from loguru import logger


class DependencyCacheStoreMixin:
    def _load_raw_dependency_provenance(
        self,
        server_name: str,
        schema_hash: str,
        server_tools: list[dict],
    ) -> tuple[dict, list[dict[str, Any]]] | None:
        """Reuse an immutable raw ledger across local audit-contract changes.

        A relation-audit or runtime-consumer change must invalidate the
        executable cache, but it does not invalidate an earlier complete LLM
        classification when the tool schema, Teacher identity, and classifier
        prompt are byte-for-byte unchanged. Generation prompt profiles do not
        participate in this corpus-level identity.
        """
        current = self._classifier_contract_payload(server_name)
        expected_tools = sorted(tool.get("name", "") for tool in server_tools)
        expected_pairs = len(expected_tools) * (len(expected_tools) - 1) // 2
        cache_dir = self._graph_cache_path(server_name, schema_hash).parent
        candidates = sorted(
            cache_dir.glob(f"{server_name}_*.json"),
            key=lambda path: path.name,
        )
        valid: list[tuple[str, Path, dict, list[dict[str, Any]]]] = []
        for path in candidates:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(payload, dict):
                    continue
                ledger = self._validate_pair_classifications(
                    payload.get("pair_classifications"), expected_tools,
                )
                if ledger is None or len(ledger) != expected_pairs:
                    continue
                raw_graph = self._graph_from_pair_classifications(
                    ledger, expected_tools,
                )
                recorded_semantics = payload.get("dependency_semantics_version")
                legacy_schema_hash = (
                    self._legacy_tool_schema_hash(
                        server_tools, int(recorded_semantics),
                    )
                    if isinstance(recorded_semantics, int)
                    else ""
                )
                schema_identity_matches = payload.get("schema_hash") in {
                    schema_hash,
                    legacy_schema_hash,
                }
                if not (
                    payload.get("server_name") == server_name
                    and schema_identity_matches
                    and payload.get("tool_names") == expected_tools
                    and payload.get("teacher_model_id")
                        == current["teacher_model_id"]
                    and payload.get("classifier_prompt_sha256")
                        == current["classifier_prompt_sha256"]
                    and payload.get("raw_graph_source")
                        == current["raw_graph_source"]
                    and payload.get("output_field_contract_sha256")
                        == current.get("output_field_contract_sha256")
                    and payload.get("classification_complete") is True
                    and payload.get("expected_pair_count") == expected_pairs
                    and payload.get("classified_pair_count") == expected_pairs
                    and payload.get("raw_graph") == raw_graph
                ):
                    continue
                valid.append((
                    str(payload.get("created_at_utc") or ""),
                    path,
                    raw_graph,
                    ledger,
                ))
            except Exception as exc:
                logger.debug(
                    f"Ignoring invalid raw dependency provenance {path}: {exc}"
                )
        if not valid:
            return None
        # The raw contract is explicitly first-valid classification. Empty
        # timestamps sort last so dated provenance wins deterministically.
        valid.sort(key=lambda item: (item[0] == "", item[0], item[1].name))
        _, source_path, raw_graph, ledger = valid[0]
        logger.info(
            f"Reusing immutable raw dependency provenance: {source_path}"
        )
        return raw_graph, ledger

    def _load_dependency_cache(
        self,
        server_name: str,
        schema_hash: str,
        server_tools: list[dict],
    ) -> dict | None:
        """Load a dependency graph cache matching the current schema contract."""
        contract_hash = self._classifier_contract_hash(server_name)
        cache_path = self._graph_cache_path(
            server_name, schema_hash, contract_hash,
        )
        if not cache_path.exists():
            return None
        try:
            payload = json.loads(cache_path.read_text())
            graph = payload.get("graph") if isinstance(payload, dict) else None
            expected_tool_names = sorted(t.get("name", "") for t in server_tools)
            cached_tool_names = payload.get("tool_names") if isinstance(payload, dict) else None
            expected_pair_count = len(expected_tool_names) * (len(expected_tool_names) - 1) // 2
            pair_classifications = self._validate_pair_classifications(
                payload.get("pair_classifications") if isinstance(payload, dict) else None,
                expected_tool_names,
            )
            raw_pair_audits = (
                payload.get("pair_audits") if isinstance(payload, dict) else None
            )
            pair_audits = (
                self._validate_dependency_pair_audits(
                    raw_pair_audits,
                    pair_classifications,
                    server_tools,
                    server_name,
                )
                if isinstance(payload, dict) and pair_classifications is not None
                else None
            )
            raw_relation_audits = (
                payload.get("relation_audits")
                if isinstance(payload, dict) else None
            )
            relation_audits = (
                self._validate_local_relation_audits(
                    raw_relation_audits,
                    pair_classifications,
                    server_tools,
                    server_name,
                )
                if isinstance(payload, dict) and pair_classifications is not None
                else None
            )
            derived_raw_graph = (
                self._graph_from_pair_classifications(
                    pair_classifications, expected_tool_names,
                )
                if pair_classifications is not None
                else None
            )
            derived_graph = (
                self._eligible_graph_from_relation_audits(
                    pair_classifications,
                    relation_audits,
                    expected_tool_names,
                )
                if pair_classifications is not None and relation_audits is not None
                else None
            )
            relation_audit_counts = (
                dict(Counter(
                    audit["verdict"] for audit in relation_audits
                ))
                if relation_audits is not None else None
            )
            classifier_contract = self._classifier_contract_payload(server_name)
            classification_complete = bool(
                isinstance(payload, dict)
                and payload.get("cache_version") == self.DEPENDENCY_CACHE_VERSION
                and payload.get("dependency_semantics_version")
                    == self.DEPENDENCY_SEMANTICS_VERSION
                and payload.get("classification_complete") is True
                and payload.get("expected_pair_count") == expected_pair_count
                and payload.get("classified_pair_count") == expected_pair_count
                and pair_classifications is not None
                and len(pair_classifications) == expected_pair_count
            )
            relation_complete = bool(
                classification_complete
                and isinstance(raw_relation_audits, list)
                and payload.get("relation_audited_pair_count")
                    == expected_pair_count
                and payload.get("relation_audit_complete") is True
                and relation_audits is not None
                and relation_audit_counts is not None
                and payload.get("relation_audit_counts")
                    == relation_audit_counts
                and self._valid_cached_graph(graph, expected_tool_names)
                and graph == derived_graph
            )
            raw_cache_contract_matches = bool(
                isinstance(payload, dict)
                and payload.get("schema_hash") == schema_hash
                and payload.get("server_name") == server_name
                and payload.get("teacher_model_id")
                    == classifier_contract["teacher_model_id"]
                and payload.get("classifier_prompt_sha256")
                    == classifier_contract["classifier_prompt_sha256"]
                and payload.get("graph_source")
                    == classifier_contract["graph_source"]
                and payload.get("raw_graph_source")
                    == classifier_contract["raw_graph_source"]
                and payload.get("review_policy")
                    == classifier_contract["review_policy"]
                and payload.get("tie_break_policy")
                    == classifier_contract["tie_break_policy"]
                and cached_tool_names == expected_tool_names
                and payload.get("tool_count") == len(expected_tool_names)
                and classification_complete
                and payload.get("raw_graph") == derived_raw_graph
            )
            cache_contract_matches = bool(
                raw_cache_contract_matches
                and relation_complete
                and payload.get("classifier_contract_hash")
                    == self._classifier_contract_hash(server_name)
                and payload.get("output_field_contract_sha256")
                    == classifier_contract.get("output_field_contract_sha256")
            )
            if cache_contract_matches:
                logger.info(f"Loaded dependency graph cache: {cache_path}")
                # raw_graph remains immutable classifier provenance. Runtime
                # scheduling always consumes the locally executable subset.
                return graph
            logger.warning(
                f"Ignoring stale or incomplete dependency graph cache: {cache_path}; "
                "a full fresh pairwise LLM classification is required"
            )
        except Exception as e:
            logger.warning(f"Failed to load dependency graph cache {cache_path}: {e}")
        return None

    def _save_dependency_cache(
        self,
        server_name: str,
        schema_hash: str,
        server_tools: list[dict],
        graph: dict,
        pair_classifications: list[dict[str, Any]],
        pair_audits: list[dict[str, Any]] | None = None,
    ) -> bool:
        """Persist a per-environment graph cache keyed by tool schema."""
        expected_tool_names = sorted(t.get("name", "") for t in server_tools)
        pair_classifications = self._validate_pair_classifications(
            pair_classifications, expected_tool_names,
        )
        if pair_classifications is None:
            logger.warning(f"Skipping incomplete dependency pair ledger for {server_name}")
            return False
        supplied_pair_audits = pair_audits if pair_audits is not None else []
        validated_pair_audits = (
            []
            if not supplied_pair_audits
            else self._validate_dependency_pair_audits(
                supplied_pair_audits,
                pair_classifications,
                server_tools,
                server_name,
            )
        )
        if validated_pair_audits is None:
            logger.warning(
                f"Skipping dependency graph cache for {server_name}: "
                "invalid optional independent pair-audit provenance"
            )
            return False
        pair_audits = validated_pair_audits
        relation_audits = self._build_local_relation_audits(
            pair_classifications, server_tools, server_name,
        )
        relation_audit_counts = dict(Counter(
            audit["verdict"] for audit in relation_audits
        ))
        raw_graph = self._graph_from_pair_classifications(
            pair_classifications, expected_tool_names,
        )
        derived_graph = self._eligible_graph_from_relation_audits(
            pair_classifications, relation_audits, expected_tool_names,
        )
        normalized_input_graph = self._normalize_cached_graph(
            graph, expected_tool_names,
        )
        # The ledger is authoritative. Refuse a graph that does not exactly
        # match it instead of silently normalizing corruption during load.
        accepted_input_graphs = (raw_graph, derived_graph)
        if (
            not self._valid_cached_graph(graph, expected_tool_names)
            or normalized_input_graph not in accepted_input_graphs
        ):
            logger.warning(f"Skipping invalid dependency graph cache for {server_name}")
            return False
        graph = derived_graph
        cache_path = self._graph_cache_path(
            server_name,
            schema_hash,
            self._classifier_contract_hash(server_name),
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        classifier_contract = self._classifier_contract_payload(server_name)
        expected_pair_count = len(pair_classifications)
        review_disagreement_count = sum(
            1 for audit in pair_audits if audit["disagrees_with_raw"]
        )
        payload = {
            "cache_version": self.DEPENDENCY_CACHE_VERSION,
            "dependency_semantics_version": self.DEPENDENCY_SEMANTICS_VERSION,
            "server_name": server_name,
            "schema_hash": schema_hash,
            "tool_names": expected_tool_names,
            "graph": graph,
            "raw_graph": raw_graph,
            "pair_classifications": pair_classifications,
            "pair_audits": pair_audits,
            "relation_audits": relation_audits,
            "classifier_contract_hash": self._classifier_contract_hash(server_name),
            **classifier_contract,
            "tool_count": len(expected_tool_names),
            "expected_pair_count": expected_pair_count,
            "classified_pair_count": expected_pair_count,
            "audited_pair_count": len(pair_audits),
            "relation_audited_pair_count": len(relation_audits),
            "review_disagreement_count": review_disagreement_count,
            "relation_audit_counts": relation_audit_counts,
            "classification_complete": True,
            "audit_complete": len(pair_audits) == expected_pair_count,
            "relation_audit_complete": True,
            "created_at_utc": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime(),
            ),
        }
        serialized = json.dumps(payload, indent=2, ensure_ascii=False)
        temp_fd, temp_name = tempfile.mkstemp(
            dir=cache_path.parent,
            prefix=f".{cache_path.name}.",
            suffix=".tmp",
        )
        temp_path = Path(temp_name)
        try:
            with os.fdopen(temp_fd, "w", encoding="utf-8") as temp_file:
                temp_file.write(serialized)
                temp_file.flush()
                os.fsync(temp_file.fileno())
            os.replace(temp_path, cache_path)
            directory_fd = os.open(cache_path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            if temp_path.exists():
                temp_path.unlink()
        logger.info(f"Saved dependency graph cache: {cache_path}")
        return True
