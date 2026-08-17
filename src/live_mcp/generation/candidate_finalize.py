"""Canonical LiveTask construction after a candidate passes generation gates."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from src.live_mcp.generation.teacher_contracts import ContinuationPolicy
from src.live_mcp.generation.robustness import zero_tool_terminal_is_valid as _zero_tool_terminal_is_valid
from src.live_mcp.live_state_query_view import (
    compact_sampling_context as _compact_sampling_context,
)
from src.live_mcp.registry.environment_metadata import build_environment_metadata
from src.live_mcp.replay.task_outcome import (
    identity_policy_for_domain as _identity_policy_for_domain,
    oracle_deleted_target_ids as _oracle_deleted_target_ids,
    oracle_target_ids as _oracle_target_ids,
    protected_fields_by_resource as _protected_fields_by_resource,
    stable_state_hash as _stable_state_hash,
)
from src.live_mcp.types import OracleProgram, to_plain


@dataclass
class FinalizationContext:
    max_task_attempts: Any
    server_name: Any
    all_oracle_calls: Any
    difficulty: Any
    plan: Any
    task_id: Any
    success_criteria: Any
    progress_predicates: Any
    user_query: Any
    session_id: Any
    sampling_state_seed: Any
    all_tools: Any
    all_required_tools: Any
    conversation_queries: Any
    oracle_calls_per_round: Any
    execution_history_per_round: Any
    chain_context: Any
    live_sampling_context: Any
    initial_action_entity_summaries: Any
    final_teacher_visible_tools: Any
    server_tools: Any
    initial_state_snapshot: Any
    source_chain_seed: Any
    source_chain_edges: Any
    realized_chain_edges: Any
    initial_state_hashes: Any
    local_seed: Any
    reference_date: Any
    reward_dependency_chain: Any
    realized_tool_sequence: Any
    dependency_call_indices: Any
    auxiliary_call_indices: Any
    source_chain_fingerprint: Any
    chain_sampling_attempt_number: Any
    chain_sampling_jaccard_novel: Any
    generated_query: Any
    continuation_goal_specs: Any
    verified_dependency_evidence: Any
    valid: Any
    criteria_ok: Any
    error_rate: Any
    num_errors: Any
    num_calls: Any
    all_attempt_calls: Any
    all_attempt_observations: Any
    all_attempt_round_indices: Any
    criteria_failed: Any
    prov_ok: Any
    prov_violations: Any
    success_criteria_provenance: Any
    conversation_fsm: Any
    scenario_type: Any


def finalize_generated_task(
    orchestrator: Any, ctx: FinalizationContext,
):
    max_task_attempts = ctx.max_task_attempts
    server_name = ctx.server_name
    all_oracle_calls = ctx.all_oracle_calls
    difficulty = ctx.difficulty
    plan = ctx.plan
    task_id = ctx.task_id
    success_criteria = ctx.success_criteria
    progress_predicates = ctx.progress_predicates
    user_query = ctx.user_query
    session_id = ctx.session_id
    sampling_state_seed = ctx.sampling_state_seed
    all_tools = ctx.all_tools
    all_required_tools = ctx.all_required_tools
    conversation_queries = ctx.conversation_queries
    oracle_calls_per_round = ctx.oracle_calls_per_round
    execution_history_per_round = ctx.execution_history_per_round
    chain_context = ctx.chain_context
    live_sampling_context = ctx.live_sampling_context
    initial_action_entity_summaries = ctx.initial_action_entity_summaries
    final_teacher_visible_tools = ctx.final_teacher_visible_tools
    server_tools = ctx.server_tools
    initial_state_snapshot = ctx.initial_state_snapshot
    source_chain_seed = ctx.source_chain_seed
    source_chain_edges = ctx.source_chain_edges
    realized_chain_edges = ctx.realized_chain_edges
    initial_state_hashes = ctx.initial_state_hashes
    local_seed = ctx.local_seed
    reference_date = ctx.reference_date
    reward_dependency_chain = ctx.reward_dependency_chain
    realized_tool_sequence = ctx.realized_tool_sequence
    dependency_call_indices = ctx.dependency_call_indices
    auxiliary_call_indices = ctx.auxiliary_call_indices
    source_chain_fingerprint = ctx.source_chain_fingerprint
    chain_sampling_attempt_number = ctx.chain_sampling_attempt_number
    chain_sampling_jaccard_novel = ctx.chain_sampling_jaccard_novel
    generated_query = ctx.generated_query
    continuation_goal_specs = ctx.continuation_goal_specs
    verified_dependency_evidence = ctx.verified_dependency_evidence
    valid = ctx.valid
    criteria_ok = ctx.criteria_ok
    error_rate = ctx.error_rate
    num_errors = ctx.num_errors
    num_calls = ctx.num_calls
    all_attempt_calls = ctx.all_attempt_calls
    all_attempt_observations = ctx.all_attempt_observations
    all_attempt_round_indices = ctx.all_attempt_round_indices
    criteria_failed = ctx.criteria_failed
    prov_ok = ctx.prov_ok
    prov_violations = ctx.prov_violations
    success_criteria_provenance = ctx.success_criteria_provenance
    conversation_fsm = ctx.conversation_fsm
    scenario_type = ctx.scenario_type

    # ── Final guard: ensure the oracle matches the task type ──
    # Exception: difficulty="missing" expects clarification-only behavior
    # (missing-required information level). If the oracle has at
    # least one ask_clarification, that's a valid task — don't raise.
    real_calls = [c for c in all_oracle_calls
                  if getattr(c, "action", "tool_call") == "tool_call"]
    abstention_calls = [
        c for c in all_oracle_calls
        if getattr(c, "action", "tool_call") in ("ask_clarification", "report_error")
    ]
    if plan.missing_function:
        if not abstention_calls:
            raise RuntimeError(
                f"Invalid missing-function oracle for {server_name} task {task_id}: "
                f"real_calls={len(real_calls)} terminals={len(abstention_calls)}"
            )
    elif not real_calls and not _zero_tool_terminal_is_valid(
        difficulty, all_oracle_calls,
    ):
        raise RuntimeError(
            f"No real tool_call recorded for {server_name} task {task_id} "
            f"after {max_task_attempts} attempt(s) "
            f"(LLM only produced clarifications/refusals)"
        )
    # ── Build final task ──
    oracle_program = OracleProgram(
        task_id=task_id,
        calls=all_oracle_calls,
        success_criteria=success_criteria,
        progress_predicates=progress_predicates,
    )

    live_task = orchestrator._to_live_task(
        server_name=server_name, query=user_query,
        session_id=session_id, seed=sampling_state_seed,
        all_tools=all_tools, oracle_program=oracle_program,
        required_tools=sorted(all_required_tools),
        difficulty=difficulty, task_id=task_id,
        conversation_queries=conversation_queries,
        oracle_calls_per_round=oracle_calls_per_round,
        execution_history_per_round=execution_history_per_round,
        sampling_context={
            "source": "live_readonly_probe",
            "chain_context": chain_context,
            "live_sampling_context": _compact_sampling_context(live_sampling_context),
            "initial_action_context": {
                "entity_summaries": initial_action_entity_summaries,
            },
        },
    )
    # Preserve the same candidate contract used by the Teacher. Distractors
    # were already added before generation; missing tools remain hidden.
    live_task.visible_tools = list(
        final_teacher_visible_tools or live_task.visible_tools
    )
    # P0: hidden_tools must be set on the LiveTask object so the corpus shard
    # serialises it into Parquet and livemcp_oval_loop.py can build blocked_tools.
    if plan.hidden_tool:
        live_task.hidden_tools = [plan.hidden_tool]
        live_task.task_type = "missing_function"
    target_ids = _oracle_target_ids(real_calls)
    identity_policy = _identity_policy_for_domain(server_name)
    deleted_targets = _oracle_deleted_target_ids(
        real_calls, server_name, server_tools,
    )
    protected_fields_by_resource = _protected_fields_by_resource(
        initial_state_snapshot, target_ids, success_criteria
    )
    terminal_action = next(
        (c.action for c in reversed(all_oracle_calls) if c.action != "tool_call"),
        "final_answer",
    )
    # scenario_type already computed inside the retry loop (with
    # missing_dependency gate applied there).  Re-use the value from
    # the successful iteration that broke out of the loop.
    owner_server_tools = {server_name: server_tools}
    for tool in live_task.visible_tools:
        owner = str(tool.get("_server_name") or server_name)
        if owner not in owner_server_tools:
            owner_server_tools[owner] = orchestrator.manager.registry.server_tools(owner)
    if not source_chain_seed:
        raise RuntimeError(
            "successful baseline generation is missing its dependency-chain seed"
        )
    live_task.metadata.update({
        "teacher_model_id": str(
            getattr(
                orchestrator.client,
                "contract_model_id",
                getattr(orchestrator.client, "model_path", "unknown"),
            )
        ),
        "initial_state_hash": _stable_state_hash(initial_state_snapshot),
        "generation_seed": local_seed,
        "sampling_state_seed": sampling_state_seed,
        "semantic_gate_profile": os.environ.get(
            "LIVEMCP_SEMANTIC_GATE_PROFILE", "diagnostic_only"
        ),
        **build_environment_metadata(
            orchestrator.suite_config,
            server_tools,
            primary_server_name=server_name,
            owner_server_tools=owner_server_tools,
            initial_state_hashes=initial_state_hashes,
        ),
        "identity_policy": identity_policy,
        "target_resource_ids": target_ids,
        "protected_resources": (
            sorted(set(target_ids) - set(deleted_targets))
            if identity_policy == "preserve" else []
        ),
        "protected_fields_by_resource": protected_fields_by_resource,
        "scenario_type": scenario_type,
        "terminal_action": terminal_action,
        "reference_date": reference_date,
        "chain_seed": list(reward_dependency_chain),
        "source_chain_seed": list(source_chain_seed),
        "source_chain_edges": list(source_chain_edges),
        "realized_chain_edges": list(realized_chain_edges),
        "initial_round_realized_tool_sequence": realized_tool_sequence,
        "initial_round_dependency_call_indices": dependency_call_indices,
        "initial_round_auxiliary_call_indices": auxiliary_call_indices,
        "source_chain_fingerprint": source_chain_fingerprint,
        "dependency_semantics_version": orchestrator.DEPENDENCY_SEMANTICS_VERSION,
        "dependency_classifier_contract_hash": (
            orchestrator._classifier_contract_hash(server_name)
        ),
        "chain_sampling_attempt_number": chain_sampling_attempt_number,
        "chain_sampling_jaccard_novel": chain_sampling_jaccard_novel,
        "query_generation_attempts": generated_query.attempts,
        "query_target_capability": generated_query.target_capability,
        "continuation_goal_specs": list(continuation_goal_specs),
        "verified_dependency_evidence": verified_dependency_evidence,
        "prompt_profile": orchestrator.prompt_profile.name,
        "generation_mode": "chain_seeded",
        # Replay and final-state validation signals.
        # Replay acceptance requires schema/execution error rate <= 30%.
        # project_outcome_valid = all success_criteria satisfied on fresh session.
        "paper_replay_valid": valid,
        "project_outcome_valid": criteria_ok,
        "replay_error_rate": error_rate,
        "replay_num_errors": num_errors,
        "replay_num_calls": num_calls,
        "teacher_attempt_count": len(all_attempt_calls),
        "teacher_failed_attempt_count": sum(
            1 for call in all_attempt_calls if call.expected_success is False
        ),
        # Persist the factual Teacher execution evidence needed
        # to distinguish model decisions from environment/input failures
        # after transient JSONL logs have been cleaned.
        "teacher_attempt_trace": [
            {
                "round_idx": int(round_idx),
                "call": to_plain(call),
                # Raw MCP observation; the exact loss-aware text shown to
                # Teacher is preserved in the optional inference JSONL and
                # is reproducible with the recorded observation budget.
                "observation": to_plain(observation),
            }
            for call, observation, round_idx in zip(
                all_attempt_calls,
                all_attempt_observations,
                all_attempt_round_indices,
                strict=True,
            )
        ],
        "teacher_round_trace": [
            {
                "round_idx": round_idx,
                "user_query": conversation_queries[round_idx],
                "oracle_calls": to_plain(round_calls),
                "execution_history": to_plain(
                    execution_history_per_round[round_idx]
                ),
            }
            for round_idx, round_calls in enumerate(oracle_calls_per_round)
        ],
        "criteria_failed": criteria_failed,
        # Persist accepted sensitive-parameter provenance evidence.
        # result so Parquet/readback audits can verify it directly instead
        # of inferring it from task acceptance.
        "provenance_valid": prov_ok,
        "provenance_violation_count": len(prov_violations),
        "success_criteria_provenance": success_criteria_provenance,
        "unattributed_success_criteria": sum(
            1 for item in success_criteria_provenance
            if not item.get("source_calls")
        ),
        "robustness_applied_before_replay": True,
        "distractor_injection_stage": "pre_teacher",
        "has_distractors": plan.inject_distractors,
        "distractor_count": len(plan.distractor_tools),
        "strip_enums": plan.strip_enums,
        "has_missing_function": plan.missing_function,
        "missing_function_requested": plan.missing_function_requested,
        "missing_function_evidence": list(plan.missing_function_evidence),
        "missing_function_binding_failure": (
            plan.missing_function_binding_failure
        ),
        # Continuation-decision schedule.
        "continuation_min_rounds": ContinuationPolicy.MIN_CONVERSATION_ROUNDS,
        "continuation_max_rounds": ContinuationPolicy.MAX_CONVERSATION_ROUNDS,
        "continuation_clarification_prob": ContinuationPolicy.CLARIFICATION_PROB,
        "continuation_end_prob_base": ContinuationPolicy.END_PROB_BASE,
        "conversation_rounds_actual": len(conversation_queries),
        "hidden_tool": plan.hidden_tool,
        # clean_visible_tools: the unperturbed domain tools (diagnostic only)
        "clean_visible_tools": server_tools,
        "fsm_final_state": conversation_fsm.state.value,
        "fsm_transitions": list(conversation_fsm.transitions),
    })
    orchestrator._record_chain_accepted(server_name, source_chain_fingerprint)
    return live_task
