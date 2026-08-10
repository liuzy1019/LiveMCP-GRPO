"""Live-state probing and dependency-chain preparation for one candidate."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Callable

from loguru import logger

from src.live_mcp.dependency_value_flow import (
    _decision_stratum, _difficulty_vector_for_chain,
    _operational_dependency_contracts, _sampled_chain_edges,
)
from src.live_mcp.live_state_query_view import (
    compact_sampling_context as _compact_sampling_context,
    extract_chain_context as _extract_chain_context,
    live_context_to_prompt_state as _live_context_to_prompt_state,
)
from src.live_mcp.replay.task_outcome import stable_state_hash as _stable_state_hash
from src.live_mcp.task_spec import (
    DECISION_STRATA, DifficultyVector, TaskSpec, compile_task_spec,
)


@dataclass
class PreparedCandidate:
    retry_candidate: bool = False
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
    difficulty_vector: Any = None
    requested_decision_stratum: Any = None
    selected_decision_stratum: Any = None
    task_spec: Any = None


def prepare_candidate(
    *, orchestrator: Any, session: Any, session_id: str,
    server_name: str, server_tools: list[dict[str, Any]],
    sampling_state_seed: int, local_seed: int, local_rng: Any,
    retry_attempt: int, max_task_attempts: int, plan: Any,
    difficulty: str, teacher: Any,
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
        operational_chains = [
            chain for chain in feasible_chains
            if _operational_dependency_contracts(
                chain, server_name, server_tools,
            )
        ]
        logger.info(
            "Dependency contract precheck [{}]: live_feasible={} "
            "operational={}",
            server_name,
            len(feasible_chains),
            len(operational_chains),
        )
        feasible_chains = operational_chains
    decision_chain_contexts: dict[
        tuple[str, ...], dict[str, Any]
    ] = {}
    decision_vectors: dict[
        tuple[str, ...], DifficultyVector
    ] = {}
    requested_decision_stratum = ""
    selected_decision_stratum = ""
    difficulty_vector: DifficultyVector | None = None
    task_spec: TaskSpec | None = None
    source_chain_edges: list[dict[str, str]] = []
    source_chain_fingerprint = ""
    chain_sampling_attempt_number = 0
    chain_sampling_jaccard_novel = False
    if (
        orchestrator.prompt_profile.decision_stratified
        and feasible_chains
    ):
        pre_stratum_chains = list(feasible_chains)
        requested_decision_stratum = DECISION_STRATA[
            local_seed % len(DECISION_STRATA)
        ]
        stratum_counts = {
            stratum: 0 for stratum in DECISION_STRATA
        }
        for candidate_chain in pre_stratum_chains:
            candidate_context = _extract_chain_context(
                chain=candidate_chain,
                domain=server_name,
                live_context=live_sampling_context,
                server_tools=server_tools,
            )
            candidate_vector = _difficulty_vector_for_chain(
                chain=candidate_chain,
                server_name=server_name,
                server_tools=server_tools,
                chain_context=candidate_context,
                feasible_chains=pre_stratum_chains,
                distractor_count=(
                    len(plan.distractor_tools)
                    if plan.inject_distractors else 0
                ),
            )
            key = tuple(candidate_chain)
            decision_chain_contexts[key] = candidate_context
            decision_vectors[key] = candidate_vector
            stratum_counts[
                _decision_stratum(candidate_vector)
            ] += 1
        feasible_chains = [
            candidate_chain
            for candidate_chain in pre_stratum_chains
            if _decision_stratum(
                decision_vectors[tuple(candidate_chain)]
            ) == requested_decision_stratum
        ]
        logger.info(
            "Decision-stratified precheck [{}]: requested={} "
            "counts={} eligible={}",
            server_name,
            requested_decision_stratum,
            stratum_counts,
            len(feasible_chains),
        )
        if not feasible_chains:
            raise RuntimeError(
                "No live-feasible operational chain for decision "
                f"stratum {requested_decision_stratum!r} in "
                f"{server_name}; counts={stratum_counts}"
            )
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
            return PreparedCandidate(retry_candidate=True)
        raise RuntimeError(
            f"No feasible chain for {server_name} after "
            f"{max_task_attempts} attempt(s); "
            f"rejecting task so generate_many retries with a fresh seed. "
            f"Unseeded fallback is NOT allowed in baseline."
        )
    else:
        # No chains at all for this domain — also not allowed in baseline.
        # This can happen for single-tool domains or domains without
        # dependency graph edges. generate_many will try another domain.
        raise RuntimeError(
            f"No dependency chains for {server_name}; "
            f"rejecting task — chain-seeded generation is required "
            f"for baseline. generate_many will retry with a fresh domain/seed."
        )

    source_chain_seed = list(chain_seed) if chain_seed else None
    if source_chain_seed:
        graph = orchestrator._domain_graphs.get(
            orchestrator._dependency_cache_key(server_name), {}
        )
        source_chain_edges = _sampled_chain_edges(
            source_chain_seed, graph,
        )
    query_chain_context = (
        decision_chain_contexts.get(tuple(source_chain_seed), {})
        if (
            source_chain_seed
            and orchestrator.prompt_profile.decision_stratified
        )
        else (
            _extract_chain_context(
                chain=source_chain_seed,
                domain=server_name,
                live_context=live_sampling_context,
                server_tools=server_tools,
            )
            if source_chain_seed else {}
        )
    )
    if source_chain_seed:
        query_chain_context["dependency_contracts"] = (
            _operational_dependency_contracts(
                source_chain_seed, server_name, server_tools,
            )
        )
        if orchestrator.prompt_profile.decision_stratified:
            difficulty_vector = decision_vectors[
                tuple(source_chain_seed)
            ]
            selected_decision_stratum = _decision_stratum(
                difficulty_vector
            )
    # The query generator uses the chain-specific, handler-feasible
    # state view. Passing the full live state here could select
    # an entity that _extract_chain_context deliberately excluded
    # (for example an overdue invoice for refund_invoice).
    query_visible_context = {
        **query_chain_context,
        "entity_ids": query_chain_context.get(
            "query_visible_entity_ids",
            query_chain_context.get("entity_ids", []),
        ),
        "entity_summaries": query_chain_context.get(
            "query_visible_entity_summaries",
            query_chain_context.get("entity_summaries", []),
        ),
    }
    query_grounding_state = _live_context_to_prompt_state(
        query_visible_context
    )
    if not query_grounding_state:
        # A creator-led chain may have no pre-existing entity that satisfies
        # every step. Still expose the readonly-observed namespace so the
        # query can choose valid parents/sources instead of inventing them.
        query_grounding_state = _live_context_to_prompt_state(
            live_sampling_context
        )
    if orchestrator.prompt_profile.task_spec:
        state_profiles = dict(
            session.metadata.get("state_profiles") or {}
        )
        task_spec = compile_task_spec(
            domain=server_name,
            session_seed=sampling_state_seed,
            state_profile=str(
                state_profiles.get(server_name, "baseline")
            ),
            state_fingerprint=_stable_state_hash(
                initial_state_snapshot
            ),
            difficulty=difficulty,
            source_chain=list(source_chain_seed or []),
            tool_schemas=server_tools,
            dependency_contracts=list(
                query_chain_context.get(
                    "dependency_contracts", []
                )
            ),
            natural_selector_types=list(
                query_chain_context.get(
                    "opaque_id_hidden_types", []
                )
            ),
            robustness={
                "distractors": bool(plan.inject_distractors),
                "enum_stripped": bool(plan.strip_enums),
                "missing_function": bool(plan.missing_function),
                "irrelevance": bool(plan.irrelevance),
            },
            decision_stratum=selected_decision_stratum,
            difficulty_vector=difficulty_vector,
        )
    trace_generation(
        "dependency_chain_selection",
        all_chain_count=len(all_chains),
        feasible_chain_count=len(feasible_chains),
        selected_chain=source_chain_seed or [],
        selected_chain_fingerprint=source_chain_fingerprint,
        selected_chain_attempt_number=chain_sampling_attempt_number,
        selected_chain_jaccard_novel=chain_sampling_jaccard_novel,
        requested_decision_stratum=requested_decision_stratum,
        selected_decision_stratum=selected_decision_stratum,
        difficulty_vector=(
            difficulty_vector.__dict__
            if difficulty_vector is not None else {}
        ),
        dependency_hints=dep_hints,
        task_seed_filter_counts=dict(
            orchestrator._chain_filter_stats.get(
                orchestrator._dependency_cache_key(server_name), {}
            )
        ),
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
        difficulty_vector=difficulty_vector,
        requested_decision_stratum=requested_decision_stratum,
        selected_decision_stratum=selected_decision_stratum,
        task_spec=task_spec,
    )
