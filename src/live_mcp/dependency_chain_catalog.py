"""Dependency-path extraction and in-process chain catalog."""

from __future__ import annotations

from loguru import logger

from src.live_mcp.contracts.chain_simulator import simulate_symbolic_chain
from src.live_mcp.contracts.factory import build_contract_registry
from src.live_mcp.dependency_chain_policy import (
    chain_contract_issue,
    goal_coherence_issue,
)


class DependencyChainCatalogMixin:
    def _extract_dependency_chains(self, server_name: str) -> list[list[str]]:
        """Extract length-2 to length-5 tool chains from the dependency graph.

        Depth-first search through the dependency graph to find all valid tool chains.
        """
        cache_key = self._dependency_cache_key(server_name)
        graph = self._domain_graphs.get(cache_key) or self._get_or_build_dependency_graph(server_name)
        self._domain_graphs[cache_key] = graph
        if not graph:
            return []

        chains: list[list[str]] = []

        def _dfs(current: str, path: list[str], visited: set[str]):
            if len(path) >= 5:
                return
            neighbors = (
                list(graph.get(current, {}).get("explicit", []))
                + list(graph.get(current, {}).get("implicit", []))
            )
            for neighbor in neighbors:
                if neighbor in visited:
                    continue
                new_path = path + [neighbor]
                if len(new_path) >= 2:
                    chains.append(new_path)
                _dfs(neighbor, new_path, visited | {neighbor})

        for start_node in graph:
            _dfs(start_node, [start_node], {start_node})

        # Exact chain deduplication after extracting length-2 to length-5 paths.
        # discovered graph; do not add local caps or relation-preference ranking
        # that would bias the sampled chain distribution.
        deduped: list[list[str]] = []
        seen: set[tuple] = set()
        task_seed_drop_reasons: dict[str, int] = {}
        contract_registry = build_contract_registry({
            server_name: self.manager.registry.server_tools(server_name),
        })
        for c in chains:
            key = tuple(c)
            task_seed_issue = chain_contract_issue(server_name, c)
            if task_seed_issue is None and not self._uses_paper_baseline():
                task_seed_issue = goal_coherence_issue(
                    contract_registry, server_name, c,
                )
                if task_seed_issue is None:
                    _, simulation_issues = simulate_symbolic_chain(
                        contract_registry, server_name, c,
                    )
                    task_seed_issue = (
                        f"{simulation_issues[0].tool_name}:"
                        f"{simulation_issues[0].predicate.slot}:"
                        f"{simulation_issues[0].reason}"
                        if simulation_issues else None
                    )
            if task_seed_issue:
                task_seed_drop_reasons[task_seed_issue] = (
                    task_seed_drop_reasons.get(task_seed_issue, 0) + 1
                )
                continue
            if key not in seen:
                seen.add(key)
                deduped.append(c)

        logger.debug(
            f"_extract_dependency_chains: {server_name} → {len(deduped)} "
            f"chains discovered"
        )
        if task_seed_drop_reasons:
            logger.debug(
                "_extract_dependency_chains: {} task-seed filters={}",
                server_name,
                task_seed_drop_reasons,
            )
        self._chain_filter_stats[cache_key] = dict(task_seed_drop_reasons)
        return deduped

    def _get_chains(self, server_name: str) -> list[list[str]]:
        """Return cached dependency chains for *server_name*, extracting if needed."""
        with self._dependency_graph_lock:
            cache_key = self._dependency_cache_key(server_name)
            if cache_key not in self._domain_chains:
                self._domain_chains[cache_key] = self._extract_dependency_chains(server_name)
            return self._domain_chains[cache_key]
