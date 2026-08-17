"""State-machine task generation over live MCP servers.

Per environment:
  1. Auto-discover tool dependency graph via live MCP probing
  2. State machine alternating LLM decisions and tool execution
     against a live MCP server
  3. Robustness knobs applied before Teacher processing
  4. Replay-validate each perturbed conversation before conversion
"""

from __future__ import annotations

import json
import os
import random
from typing import Any

from loguru import logger

from src.live_mcp.contracts.catalog import domain_contract_registry
from src.live_mcp.contracts.outcome import (
    mutation_outcome_issue,
    successful_state_transition_noop_indices,
)
from src.live_mcp.domain_contracts.entities import _tool_entity
from src.live_mcp.domain_contracts.reference_visibility import (
    public_entity_reference_ids_from_records,
)
from src.live_mcp.dependency_trace import (
    align_sampled_chain,
    auxiliary_tool_call_indices,
    verify_implicit_edges_counterfactually,
    select_realized_dependency_chain,
)
from src.live_mcp.dependency_value_flow import (
    _operational_dependency_contracts,
    _sampled_chain_edges,
    _verify_dependency_evidence,
    _verify_realized_chain_dependencies,
)
from src.live_mcp.fsm import ConversationFSM, RobustnessPlan
from src.live_mcp.generation.candidate_finalize import (
    FinalizationContext,
    finalize_generated_task,
)
from src.live_mcp.generation.candidate_prepare import prepare_candidate
from src.live_mcp.generation.candidate_query import generate_query_contract
from src.live_mcp.generation.candidate_setup import setup_candidate_attempt
from src.live_mcp.generation.candidate_trace_validation import (
    validate_early_candidate_trace,
)
from src.live_mcp.generation.conversation_runner import run_candidate_conversation
from src.live_mcp.generation.robustness import (
    profile_scenario_is_valid as _profile_scenario_is_valid,
)
from src.live_mcp.replay.task_outcome import (
    attribute_success_criteria as _attribute_success_criteria,
)
from src.live_mcp.generation.query_teacher import QueryGenerationError
from src.live_mcp.generation.teacher_contracts import (
    reference_date_for_candidate_state,
)
from src.live_mcp.types import LiveTask, OracleCall, to_plain
from src.live_mcp.errors import CandidateGenerationError


def _candidate_conversation_reason(exc: Exception) -> str:
    """Map expected fail-closed conversation outcomes to stable taxonomy."""
    structured_reason = str(getattr(exc, "reason", "") or "")
    if structured_reason:
        return structured_reason
    message = str(exc)
    if message.startswith("Failed to generate followup"):
        return "continuation_generation_failed"
    if "per-round action budget without a terminal response" in message:
        return "teacher_action_budget_exhausted"
    if "successful no-progress tool call" in message:
        return "teacher_no_progress_exhausted"
    return "candidate_conversation_exception"


class GenerationCandidateMixin:
    def generate_one(
            self,
            server_name: str,
            seed: int,
            difficulty: str,
            max_turns: int = 8,
            robustness_plan: RobustnessPlan | None = None,
            state_seed: int | None = None,
        ) -> LiveTask:
            """Generate one task with an LLM-driven state machine.

            1. Sample a dependency-chain seed
            2. Apply the robustness plan to Teacher-visible schemas
            3. Generate a user query with persona and reference date
            4. State-machine loop: LLM decides next action → execute → recovery →
               record.  The inner step budget is ``max_turns`` (default 8).
            5. Derive success criteria from state delta
            6. Replay validate the perturbed conversation against fresh session

            robustness_plan: if None, defaults to clean (no perturbations).
            When provided, distractor/enum-stripping are applied to the
            Teacher/Policy-visible candidate schemas while the live handler keeps
            its authoritative execution contract. Missing-function is sampled
            and bound to an audited hidden capability before query synthesis,
            then blocked in generation replay and Policy rollout.

            Retries with different seed if oracle_calls is empty or replay fails.
            """
            from src.live_mcp.generation.teacher_contracts import _PERSONA_TEMPLATES
            from src.live_mcp.replay.criteria import (
                derive_progress_predicates,
                derive_success_criteria,
            )
            from src.live_mcp.replay.gates import provenance_check, replay_validate
            from src.live_mcp.types import ToolCall

            rng = random.Random(seed)

            # ── Sample diversity injectors ──
            persona = _PERSONA_TEMPLATES[seed % len(_PERSONA_TEMPLATES)]
            reference_date = reference_date_for_candidate_state(seed, state_seed)

            # ── Sample dependency-chain seed ──
            # Defer chain selection until live state is available so chains with
            # unsatisfied first-step entity requirements can be removed.

            # ── Conversation-level continuation ──
            # The dependency chain is one atomic initial user goal and is not split
            # into later turns. Continuation samples a related follow-up from
            # refreshed live state after that round; it does not bind the new user
            # request to another hidden dependency-chain node.

            # ── Defensive initialisation (all variables reused after the retry loop).
            # Candidate-level regeneration is intentionally one-shot by default.
            # Recovery runs inside the state machine. Replay and provenance filter
            # the completed candidate; pool-level oversampling replaces rejected
            # candidates.
            try:
                max_task_attempts = max(
                    1, int(os.environ.get("LIVEMCP_TASK_GENERATION_ATTEMPTS", "1")),
                )
            except ValueError:
                max_task_attempts = 1

            # Python guarantees range(max_task_attempts) iterates at least once, but Pylance cannot
            # prove that and flags "possibly unbound".  Initialising here silences
            # the linter and protects against edge cases.
            all_oracle_calls: list = []
            all_execution_history: list = []
            all_required_tools: set = set()
            conversation_queries: list = []
            oracle_calls_per_round: list = []
            oracle_observations_per_round: list = []
            execution_history_per_round: list = []
            task_id = ""
            user_query = ""
            session_id = ""
            local_seed = seed
            all_tools: list = []
            chain_context: dict = {}
            live_sampling_context: dict = {}
            initial_state_snapshot: dict = {}
            success_criteria: list = []
            success_criteria_provenance: list[dict[str, Any]] = []
            verified_dependency_evidence: list[dict[str, Any]] = []
            progress_predicates: list = []
            teacher = None
            scenario_type = ""
            identity_policy = ""
            final_teacher_visible_tools: list[dict] = []
            chain_seed: list[str] | None = None
            source_chain_seed: list[str] | None = None
            source_chain_edges: list[dict[str, str]] = []
            source_chain_fingerprint = ""
            chain_sampling_attempt_number = 0
            chain_sampling_jaccard_novel = False
            all_attempt_calls: list[OracleCall] = []
            all_attempt_observations: list[Any] = []
            all_attempt_round_indices: list[int] = []
            initial_state_hashes: dict[str, str] = {}
            initial_action_entity_summaries: list[Any] = []
            continuation_goal_specs: list[Any] = []
            reward_dependency_chain: list[str] = []
            realized_tool_sequence: list[str] = []
            dependency_call_indices: list[int] = []
            auxiliary_call_indices: list[int] = []
            generated_query: Any = None
            prov_ok = False
            prov_violations: list[dict[str, Any]] = []
            # replay / provenance variables (set inside retry loop, used after)
            valid: bool = False
            criteria_ok: bool = True
            error_rate: float = 0.0
            num_errors: int = 0
            num_calls: int = 0
            criteria_failed: int = 0
            plan: RobustnessPlan = RobustnessPlan()
            server_tools: list[dict] = []
            generation_succeeded = False
            conversation_fsm = ConversationFSM()
            sampling_state_seed = int(seed if state_seed is None else state_seed)
            rejection_history: list[dict[str, Any]] = []

            def _reject(
                stage: str, reason: str, **details: Any,
            ) -> None:
                rejection_history.append({
                    "stage": stage,
                    "reason_code": reason,
                    "attempt": len(rejection_history) + 1,
                    **details,
                })

            def _conversation_failure_evidence(
                **details: Any,
            ) -> dict[str, Any]:
                """Build self-contained evidence after a conversation exists."""
                return {
                    "source_chain_seed": list(source_chain_seed or []),
                    "source_chain_edges": list(source_chain_edges),
                    "user_query": user_query,
                    "conversation_queries": list(conversation_queries),
                    "oracle_calls_per_round": [
                        [to_plain(call) for call in calls]
                        for calls in oracle_calls_per_round
                    ],
                    "oracle_observations_per_round": [
                        list(observations)
                        for observations in oracle_observations_per_round
                    ],
                    "realized_tool_sequence": list(realized_tool_sequence),
                    **details,
                }

            # ── Retry with different seed if LLM refuses to call tools ──
            for retry_attempt in range(max_task_attempts):
                local_seed = seed + retry_attempt * 1000
                local_rng = random.Random(local_seed)

                setup = setup_candidate_attempt(
                    orchestrator=self,
                    server_name=server_name,
                    local_seed=local_seed,
                    sampling_state_seed=sampling_state_seed,
                    robustness_plan=robustness_plan,
                    retry_attempt=retry_attempt,
                    difficulty=difficulty,
                )
                teacher = setup.teacher
                conversation_fsm = setup.conversation_fsm
                session = setup.session
                session_id = setup.session_id
                all_tools = setup.all_tools
                server_tools = setup.server_tools
                plan = setup.plan
                teacher_visible_tools = setup.teacher_visible_tools
                query_teacher_visible_tools = setup.query_teacher_visible_tools
                _trace_generation = setup.trace_generation

                try:
                    prepared = prepare_candidate(
                        orchestrator=self, session=session, session_id=session_id,
                        server_name=server_name, server_tools=server_tools,
                        sampling_state_seed=sampling_state_seed,
                        local_seed=local_seed, local_rng=local_rng,
                        retry_attempt=retry_attempt,
                        max_task_attempts=max_task_attempts, teacher=teacher,
                        difficulty=difficulty, robustness_plan=plan,
                        trace_generation=_trace_generation,
                    )
                    if prepared.retry_candidate:
                        _reject(
                            "dependency_chain_selection",
                            prepared.rejection_reason
                            or "dependency_chain_retry",
                        )
                        continue
                    initial_state_snapshot = prepared.initial_state_snapshot
                    initial_state_hashes = prepared.initial_state_hashes
                    live_sampling_context = prepared.live_sampling_context
                    dep_hints = prepared.dep_hints
                    chain_seed = prepared.chain_seed
                    source_chain_seed = prepared.source_chain_seed
                    source_chain_edges = prepared.source_chain_edges
                    source_chain_fingerprint = prepared.source_chain_fingerprint
                    chain_sampling_attempt_number = prepared.chain_sampling_attempt_number
                    chain_sampling_jaccard_novel = prepared.chain_sampling_jaccard_novel
                    query_chain_context = prepared.query_chain_context
                    query_grounding_state = prepared.query_grounding_state
                    try:
                        query_contract = generate_query_contract(
                            teacher=teacher,
                            conversation_fsm=conversation_fsm,
                            teacher_visible_tools=teacher_visible_tools,
                            query_teacher_visible_tools=query_teacher_visible_tools,
                            query_grounding_state=query_grounding_state,
                            difficulty=difficulty,
                            local_rng=local_rng,
                            dep_hints=dep_hints,
                            persona=persona,
                            reference_date=reference_date,
                            source_chain_seed=source_chain_seed,
                            query_chain_context=query_chain_context,
                            plan=plan,
                            server_tools=server_tools,
                            trace_generation=_trace_generation,
                        )
                    except QueryGenerationError as exc:
                        self._record_chain_rejected(
                            server_name,
                            source_chain_fingerprint,
                            exc.reason,
                        )
                        raise
                    generated_query = query_contract.generated_query
                    user_query = query_contract.user_query
                    blocked_tools_set = query_contract.blocked_tools
                    chain_seed = query_contract.chain_seed
                    chain_context = query_contract.chain_context
                    teacher_visible_tools = query_contract.teacher_visible_tools
                    plan = query_contract.plan

                    try:
                        conversation = run_candidate_conversation(
                            orchestrator=self,
                            teacher=teacher,
                            session_id=session_id,
                            server_name=server_name,
                            server_tools=server_tools,
                            teacher_visible_tools=teacher_visible_tools,
                            difficulty=difficulty,
                            local_seed=local_seed,
                            local_rng=local_rng,
                            retry_attempt=retry_attempt,
                            max_task_attempts=max_task_attempts,
                            user_query=user_query,
                            query_chain_context=query_chain_context,
                            conversation_fsm=conversation_fsm,
                            generated_query=generated_query,
                            source_chain_seed=source_chain_seed,
                            blocked_tools_set=blocked_tools_set,
                            plan=plan,
                            max_turns=max_turns,
                            reference_date=reference_date,
                            persona=persona,
                            trace_generation=_trace_generation,
                        )
                    except CandidateGenerationError:
                        raise
                    except Exception as exc:
                        conversation_reason = _candidate_conversation_reason(exc)
                        raise CandidateGenerationError(
                            f"Candidate conversation failed for {server_name}: "
                            f"{type(exc).__name__}: {exc}",
                            stage="candidate_conversation",
                            reason=conversation_reason,
                            details={
                                "source_chain_seed": list(
                                    source_chain_seed or []
                                ),
                                "source_chain_edges": list(source_chain_edges),
                                "user_query": user_query,
                                "exception_type": type(exc).__name__,
                                "exception_message": str(exc),
                                "partial_trajectory": dict(
                                    getattr(exc, "details", {}) or {}
                                ),
                            },
                            rejection_history=rejection_history,
                        ) from exc
                    all_oracle_calls = conversation.oracle_calls
                    all_execution_history = conversation.execution_history
                    all_aligned_observations = conversation.aligned_observations
                    all_attempt_calls = conversation.attempt_calls
                    all_attempt_observations = conversation.attempt_observations
                    all_attempt_round_indices = conversation.attempt_round_indices
                    all_required_tools = conversation.required_tools
                    conversation_queries = conversation.conversation_queries
                    oracle_calls_per_round = conversation.oracle_calls_per_round
                    oracle_observations_per_round = (
                        conversation.oracle_observations_per_round
                    )
                    execution_history_per_round = (
                        conversation.execution_history_per_round
                    )
                    task_id = conversation.task_id
                    initial_action_entity_summaries = (
                        conversation.initial_action_entity_summaries
                    )

                    early_validation = validate_early_candidate_trace(
                        domain=server_name,
                        difficulty=difficulty,
                        plan=plan,
                        oracle_calls=all_oracle_calls,
                        execution_history=all_execution_history,
                        conversation_queries=conversation_queries,
                        oracle_calls_per_round=oracle_calls_per_round,
                        oracle_observations_per_round=(
                            oracle_observations_per_round
                        ),
                        source_chain_seed=source_chain_seed,
                        server_tools=server_tools,
                        teacher_visible_tools=teacher_visible_tools,
                        paper_baseline=self._uses_paper_baseline(),
                        oracle_observations=all_aligned_observations,
                        dependency_contracts=[
                            dict(item)
                            for item in query_chain_context.get(
                                "dependency_contracts", []
                            )
                            if isinstance(item, dict)
                        ],
                        live_state_public_entity_ids=(
                            public_entity_reference_ids_from_records(
                                server_name,
                                live_sampling_context.get(
                                    "entity_records", [],
                                ),
                            )
                        ),
                    )
                    if not early_validation.accepted:
                        logger.debug(
                            "Rejecting {} task {} during early trace validation: {}",
                            server_name,
                            task_id,
                            early_validation.reason,
                        )
                        _reject(
                            "early_trace_validation",
                            early_validation.reason,
                            task_id=task_id,
                        )
                        continue
                    continuation_goal_specs = list(
                        early_validation.continuation_link_evidence
                    )
                    _real_now = early_validation.real_calls
                    _terminal_now = early_validation.terminal_action
                    scenario_type = early_validation.scenario_type
                    initial_round_oracle_calls = (
                        early_validation.initial_round_calls
                    )
                    initial_round_history = (
                        execution_history_per_round[0]
                        if execution_history_per_round else []
                    )
                    initial_round_state_transition_noops = (
                        successful_state_transition_noop_indices(
                            oracle_calls=initial_round_oracle_calls,
                            execution_history=initial_round_history,
                            domain=server_name,
                        )
                    )

                    realized_tool_sequence: list[str] = []
                    dependency_call_indices: list[int] = []
                    auxiliary_call_indices: list[int] = []
                    reward_dependency_chain: list[str] = []
                    realized_chain_edges: list[dict[str, str]] = []
                    success_scenario = scenario_type in {
                        "normal_safe_success", "tool_error_recovery",
                    }
                    if chain_seed and not plan.missing_function and success_scenario:
                        realized_tool_sequence = [
                            call.tool_name
                            for call in initial_round_oracle_calls
                            if getattr(call, "action", "tool_call") == "tool_call"
                        ]
                        source_alignment = align_sampled_chain(
                            initial_round_oracle_calls, chain_seed,
                        )
                        if source_alignment is not None:
                            reward_dependency_chain = list(chain_seed)
                            dependency_call_indices = source_alignment
                        else:
                            graph = self._domain_graphs.get(
                                self._dependency_cache_key(server_name), {}
                            )
                            (
                                reward_dependency_chain,
                                dependency_call_indices,
                            ) = select_realized_dependency_chain(
                                initial_round_oracle_calls, graph,
                            )
                        if len(reward_dependency_chain) < 2:
                            _reject(
                                "dependency_evidence",
                                "realized_dependency_path_missing",
                                task_id=task_id,
                            )
                            continue
                        noop_dependency_indices = sorted(
                            set(dependency_call_indices)
                            & initial_round_state_transition_noops
                        )
                        if noop_dependency_indices:
                            _reject(
                                "outcome_evidence",
                                "dependency_state_transition_noop",
                                task_id=task_id,
                                call_indices=noop_dependency_indices,
                                chain_seed=list(reward_dependency_chain),
                            )
                            continue
                        graph = self._domain_graphs.get(
                            self._dependency_cache_key(server_name), {}
                        )
                        realized_chain_edges = _sampled_chain_edges(
                            reward_dependency_chain, graph,
                        )
                        auxiliary_call_indices = auxiliary_tool_call_indices(
                            initial_round_oracle_calls, dependency_call_indices,
                        )
                    initial_round_calls = (
                        oracle_calls_per_round[0]
                        if oracle_calls_per_round else []
                    )
                    initial_round_observations = (
                        all_aligned_observations[:len(initial_round_calls)]
                        if initial_round_calls else []
                    )
                    deterministic_contracts = _operational_dependency_contracts(
                        reward_dependency_chain,
                        server_name,
                        server_tools,
                    )
                    verified_dependency_evidence = _verify_dependency_evidence(
                        deterministic_contracts,
                        initial_round_calls,
                        initial_round_observations,
                        user_query,
                        server_tools,
                    )
                    if (
                        success_scenario
                        and reward_dependency_chain
                        and len(reward_dependency_chain) > 1
                    ):
                        classifier_explicit_edges = {
                            (
                                str(item.get("source_capability") or ""),
                                str(item.get("target_capability") or ""),
                            )
                            for item in realized_chain_edges
                            if item.get("relation") == "explicit"
                        }
                        (
                            counterfactual_evidence,
                            counterfactual_issues,
                        ) = verify_implicit_edges_counterfactually(
                            manager=self.manager,
                            executor=self.executor,
                            server_name=server_name,
                            seed=sampling_state_seed,
                            oracle_calls=initial_round_calls,
                            sampled_chain=reward_dependency_chain,
                            explicitly_verified_edges=classifier_explicit_edges,
                        )
                        (
                            verified_dependency_evidence,
                            dependency_issues,
                        ) = _verify_realized_chain_dependencies(
                            reward_dependency_chain,
                            deterministic_contracts,
                            initial_round_calls,
                            initial_round_observations,
                            user_query,
                            server_tools,
                            server_name,
                            realized_chain_edges,
                            counterfactual_evidence,
                        )
                        dependency_issues.extend(counterfactual_issues)
                        if dependency_issues:
                            logger.debug(
                                "Rejecting {} task {}: realized dependency issues={}",
                                server_name,
                                task_id,
                                dependency_issues,
                            )
                            _reject(
                                "dependency_evidence",
                                "realized_dependency_invalid",
                                issues=list(dependency_issues),
                                task_id=task_id,
                            )
                            continue
                        evidence_alignment = align_sampled_chain(
                            initial_round_calls,
                            reward_dependency_chain,
                            verified_dependency_evidence=(
                                verified_dependency_evidence
                            ),
                        )
                        if evidence_alignment is None:
                            logger.debug(
                                "Rejecting {} task {}: verified dependency "
                                "evidence does not form one canonical call path",
                                server_name,
                                task_id,
                            )
                            _reject(
                                "dependency_evidence",
                                "dependency_evidence_not_canonical_path",
                                task_id=task_id,
                            )
                            continue
                        dependency_call_indices = evidence_alignment
                        auxiliary_call_indices = auxiliary_tool_call_indices(
                            initial_round_calls, evidence_alignment,
                        )
                    if (
                        success_scenario
                        and reward_dependency_chain
                        and len(reward_dependency_chain) > 1
                        and not verified_dependency_evidence
                    ):
                        logger.debug(
                            "Dependency-necessary profile rejected {} task {}: "
                            "Teacher evidence was not realized by the executed trace",
                            server_name,
                            task_id,
                        )
                        _reject(
                            "dependency_evidence",
                            "dependency_evidence_not_realized",
                            task_id=task_id,
                        )
                        continue

                    from collections import Counter

                    def _trace_key(call: OracleCall) -> tuple[str, str]:
                        return (
                            call.tool_name,
                            json.dumps(
                                call.arguments or {}, sort_keys=True,
                                ensure_ascii=False, default=str,
                            ),
                        )

                    successful_attempts = Counter(
                        _trace_key(call) for call in all_attempt_calls
                        if call.expected_success is True
                    )
                    oracle_successes = Counter(
                        _trace_key(call) for call in _real_now
                    )
                    if successful_attempts - oracle_successes:
                        _reject(
                            "oracle_projection",
                            "successful_attempt_omitted_from_oracle",
                            omitted_count=sum(
                                (successful_attempts - oracle_successes).values()
                            ),
                            task_id=task_id,
                        )
                        continue

                    # ── Derive success criteria from state delta ──
                    final_state_full = self.manager.get_state(session_id)
                    final_state = final_state_full.get(server_name, {})
                    success_criteria = derive_success_criteria(
                        initial_state=initial_state_snapshot,
                        final_state=final_state,
                        oracle_calls=all_oracle_calls,
                        domain=server_name,
                    )
                    success_criteria_provenance = _attribute_success_criteria(
                        success_criteria, all_execution_history, server_name,
                    )
                    if not self._uses_paper_baseline():
                        schema_by_name = {
                            str(tool.get("name") or ""): tool
                            for tool in server_tools
                        }
                        outcome_issue = mutation_outcome_issue(
                            tool_names=[call.tool_name for call in _real_now],
                            success_criteria=success_criteria,
                            criterion_provenance=success_criteria_provenance,
                            is_mutating=lambda name: bool(
                                (schema_by_name.get(name, {}).get("annotations") or {})
                                .get("mutating")
                            ),
                        )
                        if outcome_issue:
                            _reject(
                                "outcome_evidence", outcome_issue,
                                task_id=task_id,
                            )
                            continue
                    progress_predicates = derive_progress_predicates(
                        oracle_calls=all_oracle_calls,
                        domain=server_name,
                        entity_resolver=_tool_entity,
                        requirements_resolver=lambda tool, domain: (
                            domain_contract_registry(domain)
                            .get(domain, tool)
                            .required_entity_types
                        ),
                    )

                    # ── Replay validation: schema/execution error rate ≤30% ──
                    valid, error_rate, num_errors, num_calls, criteria_ok, criteria_failed = replay_validate(
                        oracle_calls=all_attempt_calls,
                        manager=self.manager,
                        executor=self.executor,
                        seed=sampling_state_seed,
                        domain=server_name,
                        success_criteria=success_criteria,
                        blocked_tools=blocked_tools_set,
                        trace_recorder=(
                            getattr(teacher, "record_environment_event", None)
                            if getattr(teacher, "_trace_path", None) is not None
                            else None
                        ),
                        trace_include_state=bool(
                            getattr(teacher, "trace_includes_state", False)
                        ),
                    )
                    if not valid:
                        replay_details = _conversation_failure_evidence(
                            num_errors=num_errors,
                            num_calls=num_calls,
                            error_rate=error_rate,
                            task_id=task_id,
                        )
                        if retry_attempt + 1 < max_task_attempts:
                            logger.debug(
                                f"Replay validation failed for {server_name}: "
                                f"{num_errors}/{num_calls} errors ({error_rate:.0%}), "
                                f"retrying (attempt {retry_attempt + 1}/3)"
                            )
                            _reject(
                                "fresh_replay", "replay_invalid",
                                **replay_details,
                            )
                            continue
                        raise CandidateGenerationError(
                            f"Replay validation failed for {server_name} task {task_id}: "
                            f"{num_errors}/{num_calls} errors ({error_rate:.0%})",
                            stage="fresh_replay",
                            reason="replay_invalid",
                            details=replay_details,
                            rejection_history=rejection_history,
                        )
                    # Outcome replay is a local trainability gate. PROVE's
                    # published replay filter counts schema/execution errors,
                    # so paper-audit rows retain this result as a diagnostic.
                    if not criteria_ok and not self._uses_paper_baseline():
                        criteria_details = _conversation_failure_evidence(
                            criteria_failed=criteria_failed,
                            task_id=task_id,
                        )
                        if retry_attempt + 1 < max_task_attempts:
                            logger.debug(
                                f"Replay outcome criteria failed for "
                                f"{server_name} task {task_id}; retrying"
                            )
                            _reject(
                                "fresh_replay", "outcome_criteria_not_reproduced",
                                **criteria_details,
                            )
                            continue
                        raise CandidateGenerationError(
                            f"Replay outcome criteria failed for {server_name} "
                            f"task {task_id}: {criteria_failed} criterion/criteria "
                            f"not reproduced",
                            stage="fresh_replay",
                            reason="outcome_criteria_not_reproduced",
                            details=criteria_details,
                            rejection_history=rejection_history,
                        )

                    # ── Sensitive-parameter provenance check ──
                    prov_ok, prov_violations = provenance_check(
                        oracle_calls=all_attempt_calls,
                        user_query=conversation_queries[0],
                        aligned_observations=all_attempt_observations,
                        tool_schemas=teacher_visible_tools,
                        domain=server_name,
                        user_queries=conversation_queries,
                        call_round_indices=all_attempt_round_indices,
                    )
                    _trace_generation(
                        "provenance_result",
                        passed=prov_ok,
                        violation_count=len(prov_violations),
                        violations=prov_violations,
                    )
                    if not prov_ok:
                        provenance_details = _conversation_failure_evidence(
                            violation_count=len(prov_violations),
                            violations=prov_violations,
                            task_id=task_id,
                        )
                        if retry_attempt + 1 < max_task_attempts:
                            logger.debug(
                                f"Provenance check failed for {server_name}: "
                                f"{len(prov_violations)} untraceable sensitive params "
                                f"(e.g., {prov_violations[0]['param']} in {prov_violations[0]['tool']}), "
                                f"retrying (attempt {retry_attempt + 1}/3)"
                            )
                            _reject(
                                "sensitive_provenance", "provenance_invalid",
                                **provenance_details,
                            )
                            continue
                        raise CandidateGenerationError(
                            f"Provenance check failed for {server_name} task {task_id}: "
                            f"{len(prov_violations)} untraceable sensitive params",
                            stage="sensitive_provenance",
                            reason="provenance_invalid",
                            details=provenance_details,
                            rejection_history=rejection_history,
                        )

                    # The scenario was classified before the source-target gate;
                    # reuse that exact result for metadata and profile checks.
                    _real = _real_now
                    _terminal = _terminal_now
                    if not _profile_scenario_is_valid(
                        profile=self.prompt_profile,
                        difficulty=difficulty,
                        scenario_type=scenario_type,
                        missing_function=plan.missing_function,
                        irrelevance=plan.irrelevance,
                    ):
                        logger.debug(
                            "Prompt profile {} rejected {} task {}: difficulty={} "
                            "is incompatible with scenario_type={}",
                            self.prompt_profile.name,
                            server_name,
                            task_id,
                            difficulty,
                            scenario_type,
                        )
                        _reject(
                            "prompt_profile",
                            "scenario_profile_mismatch",
                            difficulty=difficulty,
                            scenario_type=scenario_type,
                            task_id=task_id,
                        )
                        continue
                    # Readonly success may have no state delta. State-transition
                    # mutations were already required to carry attributed criteria.
                    if (
                        scenario_type in frozenset({
                            "normal_safe_success", "tool_error_recovery",
                        })
                        and not success_criteria
                    ):
                        logger.warning(
                            f"Empty success_criteria for {scenario_type} task {task_id} "
                            f"(oracle has {len(_real)} readonly/action tool call(s))."
                        )

                    # ── Success ──
                    final_teacher_visible_tools = teacher_visible_tools
                    _trace_generation(
                        "task_acceptance",
                        task_id=task_id,
                        accepted=True,
                        scenario_type=scenario_type,
                        source_chain_seed=source_chain_seed or [],
                        realized_tool_sequence=realized_tool_sequence,
                        dependency_call_indices=dependency_call_indices,
                        auxiliary_call_indices=auxiliary_call_indices,
                        oracle_calls=[to_plain(call) for call in all_oracle_calls],
                        success_criteria=success_criteria,
                        replay={
                            "passed": valid,
                            "error_rate": error_rate,
                            "num_errors": num_errors,
                            "num_calls": num_calls,
                            "criteria_ok": criteria_ok,
                            "criteria_failed": criteria_failed,
                        },
                        provenance_passed=prov_ok,
                    )
                    generation_succeeded = True
                    break

                finally:
                    self.manager.close_session(session_id)

            if not generation_succeeded:
                last = rejection_history[-1] if rejection_history else {
                    "stage": "candidate_generation",
                    "reason_code": "candidate_exhausted_without_disposition",
                }
                raise CandidateGenerationError(
                    f"Teacher generation exhausted {max_task_attempts} attempt(s) "
                    f"for {server_name}: {last['reason_code']}",
                    stage=str(last["stage"]),
                    reason=str(last["reason_code"]),
                    details=_conversation_failure_evidence(
                        initial_round_oracle_calls=[
                            to_plain(call)
                            for call in (
                                oracle_calls_per_round[0]
                                if oracle_calls_per_round else []
                            )
                        ],
                        initial_round_oracle_observations=list(
                            oracle_observations_per_round[0]
                            if oracle_observations_per_round else []
                        ),
                        realized_tool_sequence=list(realized_tool_sequence),
                        last_rejection={
                            key: value for key, value in last.items()
                            if key not in {"stage", "reason_code"}
                        },
                    ),
                    rejection_history=rejection_history,
                )

            return finalize_generated_task(
                self,
                FinalizationContext(
                    max_task_attempts=max_task_attempts,
                    server_name=server_name,
                    all_oracle_calls=all_oracle_calls,
                    difficulty=difficulty,
                    plan=plan,
                    task_id=task_id,
                    success_criteria=success_criteria,
                    progress_predicates=progress_predicates,
                    user_query=user_query,
                    session_id=session_id,
                    sampling_state_seed=sampling_state_seed,
                    all_tools=all_tools,
                    all_required_tools=all_required_tools,
                    conversation_queries=conversation_queries,
                    oracle_calls_per_round=oracle_calls_per_round,
                    execution_history_per_round=execution_history_per_round,
                    chain_context=chain_context,
                    live_sampling_context=live_sampling_context,
                    initial_action_entity_summaries=initial_action_entity_summaries,
                    final_teacher_visible_tools=final_teacher_visible_tools,
                    server_tools=server_tools,
                    initial_state_snapshot=initial_state_snapshot,
                    source_chain_seed=source_chain_seed,
                    source_chain_edges=source_chain_edges,
                    realized_chain_edges=realized_chain_edges,
                    initial_state_hashes=initial_state_hashes,
                    local_seed=local_seed,
                    reference_date=reference_date,
                    reward_dependency_chain=reward_dependency_chain,
                    realized_tool_sequence=realized_tool_sequence,
                    dependency_call_indices=dependency_call_indices,
                    auxiliary_call_indices=auxiliary_call_indices,
                    source_chain_fingerprint=source_chain_fingerprint,
                    chain_sampling_attempt_number=chain_sampling_attempt_number,
                    chain_sampling_jaccard_novel=chain_sampling_jaccard_novel,
                    generated_query=generated_query,
                    continuation_goal_specs=continuation_goal_specs,
                    verified_dependency_evidence=verified_dependency_evidence,
                    valid=valid,
                    criteria_ok=criteria_ok,
                    error_rate=error_rate,
                    num_errors=num_errors,
                    num_calls=num_calls,
                    all_attempt_calls=all_attempt_calls,
                    all_attempt_observations=all_attempt_observations,
                    all_attempt_round_indices=all_attempt_round_indices,
                    criteria_failed=criteria_failed,
                    prov_ok=prov_ok,
                    prov_violations=prov_violations,
                    success_criteria_provenance=success_criteria_provenance,
                    conversation_fsm=conversation_fsm,
                    scenario_type=scenario_type,
                ),
            )
