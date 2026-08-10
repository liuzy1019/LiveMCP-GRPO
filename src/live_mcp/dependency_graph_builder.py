"""Discovery, cache coordination, and fail-closed graph construction."""

from __future__ import annotations

import time

from loguru import logger


class DependencyGraphBuilderMixin:
    def _get_or_build_dependency_graph(self, server_name: str) -> dict:
        """Load or build the dependency graph for one MCP server.

        Probe the live MCP server for tool schemas, load the schema-hash graph
        cache when available, and otherwise ask the teacher LLM to classify
        every unordered C(n,2) tool pair as explicit, implicit, or none.
        """
        session = self.manager.create_session(
            seed=0,
            server_names=[server_name],
        )
        try:
            self.manager.discover_tools(session.session_id)
            server_tools = self.manager.registry.server_tools(server_name)
        except Exception as e:
            logger.debug(f"_get_or_build_dependency_graph: tool discovery failed for {server_name}: {e}")
            self.manager.close_session(session.session_id)
            return {}

        if len(server_tools) < 2:
            self.manager.close_session(session.session_id)
            return {}

        schema_hash = self._tool_schema_hash(server_tools, server_name)
        try:
            cached = self._load_dependency_cache(server_name, schema_hash, server_tools)
            if cached is not None:
                return cached

            classifier_contract_hash = self._classifier_contract_hash(server_name)
            failure_key = (server_name, schema_hash, classifier_contract_hash)
            failures = getattr(self, "_dependency_graph_failures", None)
            if failures is None:
                failures = {}
                self._dependency_graph_failures = failures
            previous_failure = failures.get(failure_key)
            if previous_failure is not None:
                failed_at, failure_message = previous_failure
                retry_after = self.DEPENDENCY_FAILURE_RETRY_SECONDS - (
                    time.monotonic() - failed_at
                )
                if retry_after > 0:
                    raise RuntimeError(
                        f"Recent dependency classification failure for {server_name}; "
                        f"retry suppressed for {retry_after:.1f}s: {failure_message}"
                    )
                failures.pop(failure_key, None)

            cache_path = self._graph_cache_path(
                server_name, schema_hash, classifier_contract_hash,
            )
            with self._graph_cache_file_lock(cache_path):
                # Another process may have completed the graph while this
                # process waited for the lock. Never classify before reloading.
                cached = self._load_dependency_cache(
                    server_name, schema_hash, server_tools,
                )
                if cached is not None:
                    return cached

                raw_provenance = self._load_raw_dependency_provenance(
                    server_name, schema_hash, server_tools,
                )
                if raw_provenance is None:
                    classification = self._classify_edges_llm(
                        server_tools, server_name,
                    )
                else:
                    raw_graph, pair_classifications = raw_provenance
                    classification = (raw_graph, pair_classifications, [])
                if classification is None:
                    failure_message = (
                        f"Pairwise dependency classification incomplete for {server_name}; "
                        f"refusing incomplete graph"
                    )
                    failures[failure_key] = (time.monotonic(), failure_message)
                    raise RuntimeError(failure_message)

                graph, pair_classifications, pair_audits = classification
                cache_saved = self._save_dependency_cache(
                    server_name, schema_hash, server_tools, graph,
                    pair_classifications, pair_audits,
                )
                if not cache_saved:
                    failure_message = (
                        f"Dependency classification for {server_name} did not "
                        "satisfy cache provenance; refusing the in-memory graph"
                    )
                    failures[failure_key] = (time.monotonic(), failure_message)
                    raise RuntimeError(failure_message)
                # Reload the persisted consumer artifact so a fresh build and
                # every later process use the same eligible graph.
                persisted = self._load_dependency_cache(
                    server_name, schema_hash, server_tools,
                )
                if persisted is None:
                    raise RuntimeError(
                        f"Fresh dependency cache for {server_name} could not be "
                        "reloaded under the runtime consumer contract"
                    )
                return persisted
        finally:
            self.manager.close_session(session.session_id)
