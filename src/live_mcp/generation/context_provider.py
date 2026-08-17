"""Dependency-chain and readonly context provider for generation."""

from __future__ import annotations

import copy
from typing import Any

from loguru import logger

from src.live_mcp.contracts.chain_simulator import simulate_symbolic_chain
from src.live_mcp.contracts.factory import build_contract_registry
from src.live_mcp.dependency_chain_policy import chain_contract_issue
from src.live_mcp.domain_contracts.entities import _format_graph_hints
from src.live_mcp.live_state_discovery import probe_live_sampling_context
from src.live_mcp.live_state_feasibility import chain_is_feasible


class GenerationContextMixin:
    def _probe_live_sampling_context(
        self,
        session_id: str,
        server_name: str,
        server_tools: list[dict],
    ) -> dict[str, Any]:
        """Run the canonical readonly discovery pipeline."""
        return probe_live_sampling_context(
            executor=self.executor,
            session_id=session_id,
            server_name=server_name,
            server_tools=server_tools,
        )

    def _filter_feasible_chains(
        self,
        chains: list[list[str]],
        server_name: str,
        live_context: dict[str, Any],
    ) -> list[list[str]]:
        """Keep only chains executable in the current live state.

        The chain seed is not just a hint for prompt text. It determines which
        IDs the teacher can ground tool arguments on, so an infeasible chain must
        be removed before query/action generation. Returning the original chains
        when all checks fail reintroduces hallucination pressure, so this method
        deliberately returns [] in that case; the candidate preparation layer
        then fails the chain-required candidate closed and lets the batch
        coordinator retry with a fresh candidate identity/state seed.
        """
        if not chains:
            return chains

        entity_ids = live_context.get("entity_ids")
        if not isinstance(entity_ids, list):
            raise RuntimeError("live context missing entity_ids")
        chain_context = live_context

        feasible: list[list[str]] = []
        drop_reasons: dict[str, int] = {}
        drop_examples: dict[str, list[str]] = {}
        contract_registry = build_contract_registry({
            server_name: self.manager.registry.server_tools(server_name),
        })
        for chain in chains:
            domain_contract_issue = chain_contract_issue(server_name, chain)
            if domain_contract_issue is not None:
                ok = False
                reason = f"domain chain contract: {domain_contract_issue}"
            else:
                _, simulation_issues = simulate_symbolic_chain(
                    contract_registry, server_name, chain,
                )
            if domain_contract_issue is None and simulation_issues:
                issue = simulation_issues[0]
                ok = False
                reason = (
                    f"contract state contradiction: {issue.tool_name} "
                    f"requires {issue.predicate.slot}"
                )
            elif domain_contract_issue is None:
                ok, reason = chain_is_feasible(
                    chain, server_name, chain_context, contract_registry,
                )
            if ok:
                feasible.append(chain)
            else:
                reason_key = reason or "unknown"
                drop_reasons[reason_key] = drop_reasons.get(reason_key, 0) + 1
                drop_examples.setdefault(reason_key, chain)

        observed_count = live_context.get("observed_entity_count", 0)
        logger.info(
            f"_filter_feasible_chains [{server_name}]: "
            f"observed_entities={observed_count} "
            f"feasible_before={len(chains)} feasible_after={len(feasible)}"
        )

        if not feasible:
            logger.warning(
                f"_filter_feasible_chains: all {len(chains)} chains infeasible "
                f"for {server_name}; returning no chain_seed instead of forcing "
                f"an impossible dependency chain"
            )
            return []

        logger.debug(
            f"_filter_feasible_chains: {server_name} {len(feasible)}/{len(chains)} "
            f"chains pass live-state feasibility check"
        )
        if drop_reasons:
            top_reasons = sorted(
                drop_reasons.items(),
                key=lambda item: (-item[1], item[0]),
            )[:5]
            summary = "; ".join(
                f"{count}x {reason} e.g. {drop_examples[reason]}"
                for reason, count in top_reasons
            )
            logger.debug(
                f"_filter_feasible_chains: {server_name} dropped "
                f"{len(chains) - len(feasible)} infeasible chains: {summary}"
            )
        return feasible

    def _get_graph_hints(self, server_name: str) -> str:
        """Return cached dependency hints for *server_name*, probing if needed."""
        with self._dependency_graph_lock:
            cache_key = self._dependency_cache_key(server_name)
            if cache_key not in self._domain_graphs:
                graph = self._get_or_build_dependency_graph(server_name)
                self._domain_graphs[cache_key] = graph
            return _format_graph_hints(self._domain_graphs[cache_key])

    def _get_live_sampling_context(
        self,
        session_id: str,
        server_name: str,
        server_tools: list[dict],
        force_refresh: bool = False,
        sampling_epoch_key: str | None = None,
    ) -> dict[str, Any]:
        """Return a compact context for one state-consistent sampling epoch.

        Baseline callers give each group of ``k`` conversations the same
        deterministic initial-state seed while retaining separate MCP sessions.
        Their readonly probe result can therefore be reused safely.  A forced
        refresh is session-local after mutation and never overwrites the epoch
        baseline.  Compatibility callers without an epoch key remain scoped to
        their exact session.
        """
        if force_refresh:
            return self._probe_live_sampling_context(
                session_id=session_id,
                server_name=server_name,
                server_tools=server_tools,
            )

        schema_hash = self._tool_schema_hash(server_tools, server_name)
        scope = sampling_epoch_key or f"session:{session_id}"
        cache_key = (server_name, schema_hash, scope)
        with self._sampling_context_lock:
            cached = self._sampling_context_cache.get(cache_key)
            if cached is not None:
                logger.debug(
                    "_get_live_sampling_context: {} reused scope={} ({} entities)",
                    server_name,
                    scope,
                    len(cached.get("entity_ids", [])),
                )
                return copy.deepcopy(cached)
            fresh = self._probe_live_sampling_context(
                session_id=session_id,
                server_name=server_name,
                server_tools=server_tools,
            )
            self._sampling_context_cache[cache_key] = copy.deepcopy(fresh)
            logger.debug(
                "_get_live_sampling_context: {} refreshed scope={} ({} entities)",
                server_name,
                scope,
                len(fresh.get("entity_ids", [])),
            )
            return fresh
