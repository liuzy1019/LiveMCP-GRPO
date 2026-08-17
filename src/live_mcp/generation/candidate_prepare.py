"""Live-state probing and dependency-chain preparation for one candidate."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Callable

from loguru import logger

from src.live_mcp.contracts.factory import build_contract_registry
from src.live_mcp.contracts.state_relations import implicit_transition_bindings
from src.live_mcp.dependency_chain_policy import scenario_chain_issue
from src.live_mcp.errors import CandidateGenerationError
from src.live_mcp.dependency_value_flow import (
    _filter_relation_verifiable_chains,
    _operational_dependency_contracts,
    _sampled_chain_edges,
)
from src.live_mcp.generation.robustness import bind_missing_function_contract
from src.live_mcp.live_state_query_view import (
    compact_sampling_context as _compact_sampling_context,
    extract_chain_context as _extract_chain_context,
    generation_query_prompt_state as _generation_query_prompt_state,
)
from src.live_mcp.replay.task_outcome import stable_state_hash as _stable_state_hash


@dataclass
class PreparedCandidate:
    retry_candidate: bool = False
    rejection_reason: str = ""
    initial_state_snapshot: Any = None
    initial_state_hashes: Any = None
    live_sampling_context: Any = None
    dep_hints: Any = None
    chain_seed: Any = None
    source_chain_seed: Any = None
    source_chain_edges: Any = None
    source_chain_fingerprint: Any = None
    chain_sampling_attempt_number: Any = None
    chain_sampling_jaccard_novel: Any = None
    query_chain_context: Any = None
    query_grounding_state: Any = None


def prepare_candidate(
    *, orchestrator: Any, session: Any, session_id: str,
    server_name: str, server_tools: list[dict[str, Any]],
    sampling_state_seed: int, local_seed: int, local_rng: Any,
    retry_attempt: int, max_task_attempts: int, teacher: Any,
    difficulty: str, robustness_plan: Any,
    trace_generation: Callable[..., None],
) -> PreparedCandidate:
    grounded_state = orchestrator.manager.get_state(session_id)
    domain_state = grounded_state.get(server_name, {})
    initial_state_snapshot = copy.deepcopy(domain_state)
    initial_state_hashes = {
        owner: _stable_state_hash(owner_state)
        for owner, owner_state in grounded_state.items()
        if isinstance(owner_state, dict)
    }
    live_sampling_context = orchestrator._get_live_sampling_context(
        session_id=session_id,
        server_name=server_name,
        server_tools=server_tools,
        sampling_epoch_key=(
            f"{server_name}:"
            f"{(session.metadata.get('state_profiles') or {}).get(server_name, 'baseline')}:"
            f"{sampling_state_seed}"
            if orchestrator._uses_paper_baseline()
            else None
        ),
    )
    dep_hints = orchestrator._get_graph_hints(server_name)

    # ── Chain selection after live state is available ──
    # Deferred from above: we need the real state to filter
    # infeasible chains whose first step has no entity.
    all_chains = orchestrator._get_chains(server_name)
    feasible_chains = orchestrator._filter_feasible_chains(
        all_chains, server_name, live_sampling_context,
    ) if all_chains else []
    if orchestrator.prompt_profile.dependency_necessary and feasible_chains:
        graph = orchestrator._domain_graphs.get(
            orchestrator._dependency_cache_key(server_name), {}
        )
        relation_verifiable_chains, precheck_issues = (
            _filter_relation_verifiable_chains(
                feasible_chains, graph, server_name, server_tools,
            )
        )
        logger.info(
            "Dependency relation precheck [{}]: live_feasible={} "
            "relation_verifiable={} issues={}",
            server_name,
            len(feasible_chains),
            len(relation_verifiable_chains),
            precheck_issues,
        )
        feasible_chains = relation_verifiable_chains
    scenario_issues: dict[str, int] = {}
    if (
        feasible_chains
        and not orchestrator._uses_paper_baseline()
    ):
        registry = build_contract_registry({server_name: server_tools})
        compatible_chains: list[list[str]] = []
        for candidate_chain in feasible_chains:
            issue = scenario_chain_issue(
                registry,
                server_name,
                candidate_chain,
                difficulty=difficulty,
                missing_function=bool(robustness_plan.missing_function),
            )
            if issue is None:
                compatible_chains.append(candidate_chain)
            else:
                scenario_issues[issue] = scenario_issues.get(issue, 0) + 1
        feasible_chains = compatible_chains
    source_chain_edges: list[dict[str, str]] = []
    source_chain_fingerprint = ""
    chain_sampling_attempt_number = 0
    chain_sampling_jaccard_novel = False
    chain_seed: list[str] | None = None
    if feasible_chains:
        (
            chain_seed,
            source_chain_fingerprint,
            chain_sampling_attempt_number,
            chain_sampling_jaccard_novel,
        ) = orchestrator._select_feasible_chain(
            server_name, feasible_chains, local_rng,
        )
    elif all_chains:
        # Feasible chains exist but none passed live-state filter.
        # Optional candidate regeneration retries with a fresh
        # session; the default is one attempt.
        if retry_attempt + 1 < max_task_attempts:
            logger.debug(
                f"No feasible chain for {server_name} "
                f"(attempt {retry_attempt + 1}/{max_task_attempts}), re-sampling session"
            )
            orchestrator.manager.close_session(session_id)
            return PreparedCandidate(
                retry_candidate=True,
                rejection_reason="no_live_feasible_dependency_chain",
            )
        reason = (
            "no_scenario_compatible_dependency_chain"
            if scenario_issues else "no_live_feasible_dependency_chain"
        )
        raise CandidateGenerationError(
            f"No feasible chain for {server_name} after "
            f"{max_task_attempts} attempt(s); "
            f"rejecting task so generate_many retries with a fresh seed. "
            f"Unseeded fallback is NOT allowed in baseline.",
            stage="dependency_chain_selection",
            reason=reason,
            details={"scenario_chain_issues": scenario_issues},
        )
    else:
        # No chains at all for this domain — also not allowed in baseline.
        # This can happen for single-tool domains or domains without
        # dependency graph edges. generate_many will try another domain.
        raise CandidateGenerationError(
            f"No dependency chains for {server_name}; "
            f"rejecting task — chain-seeded generation is required "
            f"for baseline. generate_many will retry with a fresh domain/seed.",
            stage="dependency_chain_selection",
            reason="no_dependency_chain",
        )

    source_chain_seed = list(chain_seed) if chain_seed else None
    if robustness_plan.missing_function:
        bound_plan, binding_reason = bind_missing_function_contract(
            domain=server_name,
            source_chain_seed=source_chain_seed,
            tool_schemas=server_tools,
            plan=robustness_plan,
            require_capability_evidence=not orchestrator._uses_paper_baseline(),
        )
        if bound_plan is None:
            raise CandidateGenerationError(
                f"Missing-function contract could not be bound for "
                f"{server_name}: {binding_reason}",
                stage="dependency_chain_selection",
                reason=binding_reason,
                details={"source_chain_seed": source_chain_seed or []},
            )
    if source_chain_seed:
        graph = orchestrator._domain_graphs.get(
            orchestrator._dependency_cache_key(server_name), {}
        )
        source_chain_edges = _sampled_chain_edges(
            source_chain_seed, graph,
        )
    query_chain_context = (
        _extract_chain_context(
            chain=source_chain_seed,
            domain=server_name,
            live_context=live_sampling_context,
            server_tools=server_tools,
        )
        if source_chain_seed else {}
    )
    if source_chain_seed:
        dependency_contracts = _operational_dependency_contracts(
            source_chain_seed, server_name, server_tools,
        )
        query_chain_context["dependency_contracts"] = dependency_contracts
        registry = build_contract_registry({server_name: server_tools})
        query_chain_context["dependency_relations"] = [
            {
                **edge,
                "value_bindings": [
                    {
                        "source_output_field": item["source_output_field"],
                        "target_argument": item["target_argument"],
                    }
                    for item in dependency_contracts
                    if item["source_capability"] == edge["source_capability"]
                    and item["target_capability"] == edge["target_capability"]
                ],
                "state_bindings": [
                    {
                        "source_field": source_field,
                        "target_argument": target_argument,
                        "state_slot": state_slot,
                    }
                    for source_field, target_argument, state_slot in (
                        implicit_transition_bindings(
                            registry.get(
                                server_name, edge["source_capability"],
                            ),
                            registry.get(
                                server_name, edge["target_capability"],
                            ),
                        )
                        if edge["relation"] == "implicit"
                        else ()
                    )
                ],
            }
            for edge in source_chain_edges
        ]
    natural_selector = orchestrator.prompt_profile.natural_selector
    query_grounding_state = _generation_query_prompt_state(
        query_chain_context,
        server_name,
        natural_selector=natural_selector,
    )
    if not query_grounding_state or (
        natural_selector
        and not query_grounding_state.get("public_entity_summaries")
    ):
        # A creator-led chain may have no pre-existing entity that satisfies
        # every step. Still expose the readonly-observed namespace so the
        # query can choose valid parents/sources instead of inventing them.
        query_grounding_state = _generation_query_prompt_state(
            live_sampling_context,
            server_name,
            natural_selector=natural_selector,
        )
    trace_generation(
        "dependency_chain_selection",
        all_chain_count=len(all_chains),
        feasible_chain_count=len(feasible_chains),
        selected_chain=source_chain_seed or [],
        selected_chain_fingerprint=source_chain_fingerprint,
        selected_chain_attempt_number=chain_sampling_attempt_number,
        selected_chain_jaccard_novel=chain_sampling_jaccard_novel,
        dependency_hints=dep_hints,
        task_seed_filter_counts=dict(
            orchestrator._chain_filter_stats.get(
                orchestrator._dependency_cache_key(server_name), {}
            )
        ),
        scenario_chain_issues=scenario_issues,
    )
    trace_generation(
        "live_state_sampling",
        session_id=session_id,
        initial_state_hash=_stable_state_hash(initial_state_snapshot),
        initial_state=(
            initial_state_snapshot
            if bool(getattr(teacher, "trace_includes_state", False))
            else None
        ),
        live_sampling_context=_compact_sampling_context(
            live_sampling_context
        ),
        query_chain_context=query_chain_context,
        query_grounding_state=query_grounding_state,
    )
    return PreparedCandidate(
        initial_state_snapshot=initial_state_snapshot,
        initial_state_hashes=initial_state_hashes,
        live_sampling_context=live_sampling_context,
        dep_hints=dep_hints,
        chain_seed=chain_seed,
        source_chain_seed=source_chain_seed,
        source_chain_edges=source_chain_edges,
        source_chain_fingerprint=source_chain_fingerprint,
        chain_sampling_attempt_number=chain_sampling_attempt_number,
        chain_sampling_jaccard_novel=chain_sampling_jaccard_novel,
        query_chain_context=query_chain_context,
        query_grounding_state=query_grounding_state,
    )
