"""Conversation-round execution for one prepared generation candidate."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from loguru import logger

from src.live_mcp.fsm import FSMStateGroup, teacher_tool_call_budget
from src.live_mcp.generation.robustness import (
    missing_function_original_round_should_abort as _missing_function_original_round_should_abort,
    zero_tool_terminal_is_valid as _zero_tool_terminal_is_valid,
)
from src.live_mcp.live_state_query_view import live_context_to_prompt_state as _live_context_to_prompt_state
from src.live_mcp.task_planner import ContinuationPolicy


@dataclass
class ConversationResult:
    oracle_calls: list[Any]
    execution_history: list[Any]
    aligned_observations: list[Any]
    attempt_calls: list[Any]
    attempt_observations: list[Any]
    attempt_round_indices: list[int]
    required_tools: set[str]
    conversation_queries: list[str]
    oracle_calls_per_round: list[list[Any]]
    execution_history_per_round: list[list[Any]]
    continuation_goal_specs: list[dict[str, Any]]
    task_id: str
    initial_action_entity_summaries: list[Any]


def run_candidate_conversation(
    *, orchestrator: Any, teacher: Any, session_id: str, server_name: str,
    server_tools: list[dict[str, Any]],
    teacher_visible_tools: list[dict[str, Any]], difficulty: str,
    local_seed: int, local_rng: Any, retry_attempt: int,
    max_task_attempts: int, user_query: str,
    query_chain_context: dict[str, Any], conversation_fsm: Any,
    generated_query: Any, source_chain_seed: list[str] | None,
    blocked_tools_set: set[str] | None, plan: Any, max_turns: int,
    reference_date: str, persona: str,
    trace_generation: Callable[..., None],
) -> ConversationResult:
    # Accumulators across conversation rounds.
    # (re-assigned each retry; types declared before the loop)
    all_oracle_calls = []
    all_execution_history = []
    all_aligned_observations: list[Any] = []
    all_attempt_calls = []
    all_attempt_observations = []
    all_attempt_round_indices = []
    all_required_tools = set()
    conversation_queries = [user_query]  # track all user messages
    oracle_calls_per_round = []  # per-round for prompt construction
    execution_history_per_round = []
    continuation_goal_specs: list[dict[str, Any]] = []
    task_id = f"{server_name}_{local_seed}_{local_rng.randint(0, 99999)}"
    retry_label = f" (retry {retry_attempt})" if retry_attempt > 0 else ""

    current_query = user_query
    previous_assistant_response = ""
    completed_conversation_context: list[dict[str, Any]] = []
    # Preserve exactly the compact public entity view consumed by
    # the first Action Teacher turn.  Training serialization uses
    # this same view so the Policy is not asked to reproduce an
    # oracle from less information than the Teacher had.
    initial_action_entity_summaries = (
        []
        if orchestrator.prompt_profile.policy_private
        else list(
            query_chain_context.get(
                "query_grounding_summaries", []
            )
        )[:15]
    )

    logger.debug(
        f"CONTINUATION: {server_name} task {task_id} "
        f"starting state-machine continuation "
        f"(max={ContinuationPolicy.MAX_CONVERSATION_ROUNDS})"
    )

    round_idx = 0
    decision = "follow_up"  # dummy, overwritten on round_idx==0 path below
    continuation_grounding_state: dict[str, Any] = {}
    while True:
        # The Action Teacher receives the same public facts used to
        # formulate the current query, but never sampler-private IDs.
        # The sampled chain itself remains hidden.
        round_action_context: dict[str, Any] = (
            {}
            if orchestrator.prompt_profile.policy_private
            else {
                "entity_summaries": list(
                    query_chain_context.get(
                        "query_grounding_summaries", []
                    )
                )
            }
        )
        if round_idx > 0:
            if decision == "clarification":
                current_query = teacher.generate_clarification(
                    tool_schemas=teacher_visible_tools,
                    grounded_state=continuation_grounding_state,
                    previous_query=current_query,
                    difficulty=difficulty,
                    rng=local_rng,
                    persona=persona,
                    reference_date=reference_date,
                    previous_response=previous_assistant_response,
                    conversation_context=completed_conversation_context,
                )
            else:
                current_query = teacher.generate_followup(
                    tool_schemas=teacher_visible_tools,
                    grounded_state=continuation_grounding_state,
                    previous_query=current_query,
                    difficulty=difficulty,
                    rng=local_rng,
                    persona=persona,
                    reference_date=reference_date,
                    chain_seed=None,
                    chain_progress=0,
                    previous_response=previous_assistant_response,
                    conversation_context=completed_conversation_context,
                    goal_spec=None,
                )
            conversation_queries.append(current_query)
            conversation_fsm.transition(
                FSMStateGroup.TURN,
                "continuation_query_generated",
                decision=decision,
                round_idx=round_idx,
            )
            round_action_context = {}
        else:
            # round_idx == 0: first round, no decision yet
            decision = "follow_up"  # dummy for the first iteration

        max_calls_r = 0

        orchestrator._record_round_trace(
            teacher=teacher,
            session_id=session_id,
            server_name=server_name,
            round_idx=round_idx,
            phase="input",
            current_query=current_query,
            visible_tools=teacher_visible_tools,
            public_context=round_action_context,
        )

        (
            round_ocs,
            round_hist,
            round_obs,
            round_reqs,
            round_attempts,
            round_attempt_obs,
        ) = orchestrator._run_turn_loop(
            teacher=teacher,
            current_query=current_query,
            server_tools=teacher_visible_tools,  # P0: Teacher sees perturbed schemas
            server_name=server_name,
            session_id=session_id,
            difficulty=difficulty,
            round_idx=round_idx,
    turn_budget=teacher_tool_call_budget(
                max_turns,
                source_chain_seed if round_idx == 0 else None,
            ),
            reference_date=reference_date,
            max_calls_this_round=max_calls_r,
            chain_context=round_action_context,
            blocked_tools=blocked_tools_set,
            missing_function_contract=plan.missing_function,
            allowed_missing_mutations={
                str(item.get("capability") or "")
                for item in generated_query.mutation_evidence
                if str(item.get("capability") or "")
                != plan.hidden_tool
            },
            prior_execution_history=all_execution_history,
            conversation_context=completed_conversation_context,
            allow_direct_answer=(
                round_idx > 0 and decision == "clarification"
            ),
            dependency_plan=(
                list(generated_query.dependency_evidence)
                if (
                    round_idx == 0
                    and orchestrator.prompt_profile.dependency_necessary
                )
                else []
            ),
            fsm=conversation_fsm,
        )
        orchestrator._record_round_trace(
            teacher=teacher,
            session_id=session_id,
            server_name=server_name,
            round_idx=round_idx,
            phase="output",
            current_query=current_query,
            visible_tools=teacher_visible_tools,
            public_context=round_action_context,
            oracle_calls=round_ocs,
        )

        if round_idx == 0:
            _real_round = [c for c in round_ocs if getattr(c, "action", "tool_call") == "tool_call"]
            _abstain_round = [
                c for c in round_ocs
                if getattr(c, "action", "tool_call") in ("ask_clarification", "report_error")
            ]
            allow_zero_tool = bool(
                plan.missing_function and _abstain_round
            ) or _zero_tool_terminal_is_valid(difficulty, round_ocs)
            if not _real_round and not allow_zero_tool:
                if retry_attempt + 1 < max_task_attempts:
                    logger.debug(
                        f"No tool calls recorded for {server_name}{retry_label}, "
                        f"retrying with new seed ({retry_attempt + 1}/3)"
                    )
                    break  # break conversation loop → continue retry loop
                raise RuntimeError(
                    f"No tool calls recorded for {server_name} task {task_id} "
                    f"(LLM answered without using tools)"
                )

        # Preserve the successful oracle emitted by the state
        # machine. Conversation-level Jaccard does not remove
        # repeated calls inside recovery or later user rounds.
        filtered_round_ocs = list(round_ocs)
        filtered_round_obs = list(round_obs)

        all_oracle_calls.extend(filtered_round_ocs)
        all_aligned_observations.extend(filtered_round_obs)
        all_execution_history.extend(round_hist)
        all_attempt_calls.extend(round_attempts)
        all_attempt_observations.extend(round_attempt_obs)
        all_attempt_round_indices.extend(
            [round_idx] * len(round_attempts)
        )
        all_required_tools |= round_reqs
        oracle_calls_per_round.append(list(filtered_round_ocs))
        execution_history_per_round.append(list(round_hist))

        round_terminals = [
            oc for oc in filtered_round_ocs
            if getattr(oc, "action", "tool_call") != "tool_call"
        ]
        if round_terminals:
            terminal_args = getattr(round_terminals[-1], "arguments", {}) or {}
            previous_assistant_response = str(
                terminal_args.get("text")
                or terminal_args.get("question")
                or ""
            )
            completed_conversation_context.append({
                "round_idx": round_idx,
                "user_query": current_query,
                "assistant_response": previous_assistant_response,
                "terminal_action": getattr(
                    round_terminals[-1], "action", "final_answer"
                ),
            })

        # The hidden capability is selected from the initial
        # sampled chain.  If the original request already reaches
        # final_answer, this candidate did not actually exercise
        # the missing-function variant.  Fail the candidate here;
        # do not let an unrelated later continuation manufacture
        # the required abstention terminal.
        if _missing_function_original_round_should_abort(
            plan.missing_function, round_idx, filtered_round_ocs,
        ):
            logger.debug(
                f"Missing-function original round completed without "
                f"clarification/abstention for {server_name} task "
                f"{task_id}; rejecting candidate before continuation."
            )
            break

        # ask_clarification / report_error break the conversation
        # immediately — they indicate the Teacher cannot proceed further
        # in this round, and generating follow-up rounds after them
        # creates rollout-unreachable contracts (rollout terminates
        # on report_error unconditionally).
        if any(
            getattr(oc, "action", "tool_call") in ("ask_clarification", "report_error")
            for oc in filtered_round_ocs
        ):
            break

        # Continuation is sampled after completing a conversation round,
        # not a mechanism for splitting dependency-chain nodes. Once
        # the initial chain goal is complete, the refreshed live state
        # and prior conversation still ground a natural follow-up.
        # Missing-function/clarification/report_error paths terminate
        # above and are not forced to meet the normal 2--3-turn bound.
        round_idx += 1
        conversation_fsm.transition(
            FSMStateGroup.CONTINUATION,
            "continuation_decision_requested",
            rounds_done=round_idx,
        )
        decision = ContinuationPolicy.sample_continuation_decision(
            round_idx, local_rng,
        )
        if decision == "end":
            conversation_fsm.transition(
                FSMStateGroup.CONTINUATION,
                "continuation_selected",
                decision="end",
                round_idx=round_idx,
            )
            break
        # Only a selected new Query state needs a fresh sample.
        # Ground it against the current session, not the epoch
        # baseline sampled before round 0. The probe is readonly
        # and session-local, so completed mutations are visible
        # without contaminating other conversations.
        refreshed_continuation_context = (
            orchestrator._get_live_sampling_context(
                session_id=session_id,
                server_name=server_name,
                server_tools=server_tools,
                force_refresh=True,
            )
        )
        continuation_grounding_state = _live_context_to_prompt_state(
            refreshed_continuation_context
        )
        trace_generation(
            "continuation_live_state_refresh",
            completed_round_idx=round_idx - 1,
            refreshed_entity_count=len(
                refreshed_continuation_context.get(
                    "entity_ids", []
                )
            ),
        )
        conversation_fsm.transition(
            FSMStateGroup.QUERY,
            "continuation_selected",
            decision=decision,
            round_idx=round_idx,
        )
        # A grounded follow-up continues the loop.
    return ConversationResult(
        oracle_calls=all_oracle_calls,
        execution_history=all_execution_history,
        aligned_observations=all_aligned_observations,
        attempt_calls=all_attempt_calls,
        attempt_observations=all_attempt_observations,
        attempt_round_indices=all_attempt_round_indices,
        required_tools=all_required_tools,
        conversation_queries=conversation_queries,
        oracle_calls_per_round=oracle_calls_per_round,
        execution_history_per_round=execution_history_per_round,
        continuation_goal_specs=continuation_goal_specs,
        task_id=task_id,
        initial_action_entity_summaries=initial_action_entity_summaries,
    )
