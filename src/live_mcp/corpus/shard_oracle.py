"""Shard Oracle Contract."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from loguru import logger

from src.live_mcp.task_planner import DOMAIN_DESCRIPTIONS
from src.live_mcp.planner_format import format_tools
from src.live_mcp.prompt_profiles import PROMPT_PROFILES, resolve_prompt_profile
from src.live_mcp.registry.tool_semantics import (
    SELF_CONTAINED_WRITE_TOOLS,
    is_mutating_tool,
    resolve_tool_execution_semantics,
)
from src.live_mcp.dedup import dedup_tasks
from src.live_mcp.dependency_trace import (
    align_sampled_chain, auxiliary_tool_call_indices,
    dependency_edges_from_alignment,
)

def _task_scenario(task) -> str:
    explicit = task.metadata.get("scenario_type") if task.metadata else None
    if explicit:
        return str(explicit)
    # Missing-function variants produce clarification trajectories. Abstention
    # is reserved for the `irrelevant` scenario and imported irrelevance rows.
    if task.task_type == "missing_function":
        return "clarification_required"
    if task.task_type == "irrelevant":
        return "no_tool_or_abstention"
    return "normal_safe_success"

def _identity_policy(domain: str) -> str:
    return {
        "calendar": "preserve",
        "banking": "preserve",
        "filesystem": "domain_defined",
        "payments": "preserve",
        "crm": "preserve",
        "issue_tracker": "preserve",
        "email": "append_only",
        "team_chat": "append_only",
        "shopping": "create_new",
        "food_delivery": "create_new",
    }.get(domain, "domain_defined")

def _required_round_oracle_projection(
    task, round_idx: int, round_calls: list,
) -> tuple[list, list[dict[str, Any]]]:
    """Project Teacher actions onto required workflow steps for RL labels.

    The full live trace remains untouched on ``task.oracle_program`` and in the
    audit log.  Only calls explicitly tagged by the state machine as an exact
    no-progress repeat are omitted from the required workflow view.
    """
    histories = getattr(task, "execution_history_per_round", None) or []
    history = histories[round_idx] if round_idx < len(histories) else []
    history_cursor = 0
    required = []
    decisions: list[dict[str, Any]] = []
    for call in round_calls:
        if getattr(call, "action", "tool_call") != "tool_call":
            required.append(call)
            continue
        matched = None
        for index in range(history_cursor, len(history)):
            event = history[index]
            if not isinstance(event, dict) or not bool(event.get("success")):
                continue
            if (
                str(event.get("tool_name") or "") == str(call.tool_name)
                and dict(event.get("arguments") or {}) == dict(call.arguments or {})
            ):
                matched = event
                history_cursor = index + 1
                break
        if matched is not None:
            if matched.get("no_progress_warning"):
                decisions.append({
                    "round_idx": round_idx,
                    "tool_name": str(call.tool_name),
                    "decision": "drop",
                    "reason": "exact_no_progress_repeat",
                })
                continue
            domain = str(
                getattr(call, "server_name", "")
                or matched.get("server_name")
                or (
                    task.target_servers[0]
                    if getattr(task, "target_servers", None)
                    else ""
                )
            )
            if (
                domain
                and matched.get("state_changed") is False
                and resolve_tool_execution_semantics(
                    call.tool_name, domain,
                ) == "state_transition"
            ):
                # The factual attempt remains in teacher_attempt_trace.  A
                # successful state-transition no-op did not produce a required
                # outcome and must not be rewarded as ground truth.  Successful
                # action-execution tools (for example unzip) remain required
                # even when their net state delta is empty.
                decisions.append({
                    "round_idx": round_idx,
                    "tool_name": str(call.tool_name),
                    "decision": "drop",
                    "reason": "state_transition_noop",
                })
                continue
            if (
                domain
                and matched.get("state_changed") is False
                and resolve_tool_execution_semantics(
                    call.tool_name, domain,
                ) == "action_execution"
            ):
                decisions.append({
                    "round_idx": round_idx,
                    "tool_name": str(call.tool_name),
                    "decision": "keep",
                    "reason": "action_execution_no_net_change",
                })
        required.append(call)
    return required, decisions

def _required_round_oracle_calls(task, round_idx: int, round_calls: list) -> list:
    required, _ = _required_round_oracle_projection(
        task, round_idx, round_calls,
    )
    return required

def _required_workflow_projection_summary(task) -> dict[str, Any]:
    decisions: list[dict[str, Any]] = []
    for round_idx, round_calls in enumerate(
        getattr(task, "oracle_calls_per_round", None) or []
    ):
        _, round_decisions = _required_round_oracle_projection(
            task, round_idx, list(round_calls),
        )
        decisions.extend(round_decisions)
    counts: dict[str, int] = {}
    for item in decisions:
        reason = str(item["reason"])
        counts[reason] = counts.get(reason, 0) + 1
    return {"counts": counts, "events": decisions}

def _serialize_training_oracle(task) -> list[dict]:
    """Return tool calls plus exactly one terminal action for training.

    All task types that reach this function must have a non-empty oracle
    program.  The orchestrator always pre-fills oracle calls for
    missing_function and irrelevant tasks. A future auditable importer for
    external abstention rows (When2Call, xLAM-Irrelevance) must write those
    directly to Parquet; the current repository has no such importer, so they
    do not flow through this path.
    """
    scenario_type = task.metadata.get("scenario_type") if task.metadata else None

    # ── Assert invariants ──────────────────────────────────────────
    # missing_function / clarification_required tasks MUST have a
    # pre-filled oracle.  If this fires, the orchestrator was changed
    # to skip oracle population — fix the caller, not this function.
    if task.task_type == "missing_function" or scenario_type == "clarification_required":
        if not (task.oracle_program and task.oracle_program.calls):
            raise ValueError(
                f"Task {task.task_id}: missing_function/clarification_required "
                f"task has no oracle calls — orchestrator should have pre-filled "
                f"ask_clarification terminal"
            )

    # irrelevance / abstention tasks MUST also have a pre-filled oracle.
    if scenario_type in ("no_tool_or_abstention", "irrelevant"):
        if not (task.oracle_program and task.oracle_program.calls):
            raise ValueError(
                f"Task {task.task_id}: {scenario_type} task has no oracle "
                f"calls — orchestrator should have pre-filled report_error"
            )

    raw_calls = []
    if task.oracle_program and task.oracle_program.calls:
        source_calls = list(task.oracle_program.calls)
        if getattr(task, "oracle_calls_per_round", None):
            source_calls = []
            for round_idx, round_calls in enumerate(task.oracle_calls_per_round):
                source_calls.extend(
                    _required_round_oracle_calls(task, round_idx, list(round_calls))
                )
        for oc in source_calls:
            raw_calls.append({
                "tool_name": oc.tool_name,
                "arguments": dict(oc.arguments) if oc.arguments else {},
                "action": getattr(oc, "action", "tool_call"),
                "server_name": str(getattr(oc, "server_name", "") or ""),
            })

    terminals = [
        call for call in raw_calls
        if call.get("action") in ("final_answer", "ask_clarification", "report_error")
    ]
    if not terminals:
        raise ValueError(f"Task {task.task_id} has no explicit terminal oracle action")

    tool_calls = [
        call for call in raw_calls
        if call.get("action", "tool_call") == "tool_call"
    ]
    return tool_calls + [terminals[-1]]

def _build_round_contracts(task) -> list[dict]:
    """P0-2: Build per-round contracts from oracle_calls_per_round.

    Each contract defines the expected tools and allowed terminal action
    for one conversation round.  The rollout loop enforces these contracts
    to prevent illegal terminal-advancing (e.g. report_error → follow-up).

    Returns:
        list[dict] with keys: round_idx, required_tools, allowed_terminal_actions
    """
    if not task.oracle_calls_per_round:
        raise ValueError(
            f"Task {task.task_id} has no canonical oracle_calls_per_round"
        )

    contracts = []
    for round_idx, round_calls in enumerate(task.oracle_calls_per_round):
        if not round_calls:
            raise ValueError(
                f"Task {task.task_id}: oracle round {round_idx} is empty; "
                "a conversation round must contain a Teacher-emitted action"
            )
        required_round_calls = _required_round_oracle_calls(
            task, round_idx, list(round_calls),
        )
        tools: list[str] = []
        terminal = "final_answer"
        for oc in required_round_calls:
            action = getattr(oc, "action", "tool_call")
            if action == "tool_call":
                tools.append(getattr(oc, "tool_name", ""))
            elif action in ("final_answer", "ask_clarification", "report_error"):
                terminal = action
        contracts.append({
            "round_idx": round_idx,
            "required_tools": [t for t in tools if t],
            "allowed_terminal_actions": [terminal],
        })
    return contracts

def _minimum_action_budget(
    oracle_calls_serialized: list[dict],
    round_contracts: list[dict],
) -> int:
    """Minimum model actions needed to reproduce a multi-round reference.

    The rollout loop spends one iteration on every tool call and on the
    terminal that closes each conversation round.
    """
    n_tool_calls = sum(
        1 for call in oracle_calls_serialized
        if call.get("action", "tool_call") == "tool_call"
    )
    return n_tool_calls + max(1, len(round_contracts))

def _task_success_criteria(task) -> list:
    if task.oracle_program and task.oracle_program.success_criteria:
        return list(task.oracle_program.success_criteria)
    if hasattr(task, "success_criteria") and task.success_criteria:
        return list(task.success_criteria)
    return []

def _validate_task_training_contract(task) -> None:
    oracle_calls_serialized = _serialize_training_oracle(task)
    terminal_actions = [
        call["action"] for call in oracle_calls_serialized
        if call.get("action") in ("final_answer", "ask_clarification", "report_error")
    ]
    if len(terminal_actions) != 1:
        raise ValueError(
            f"Task {task.task_id} has {len(terminal_actions)} terminal oracle actions"
        )
    terminal_action = terminal_actions[0]

    real_required_tools = [
        call["tool_name"] for call in oracle_calls_serialized
        if call.get("action", "tool_call") == "tool_call"
    ]
    scenario_type = _task_scenario(task)
    # Export contract:
    #   missing_function variant (Step 3)   → ask_clarification (1,500 traj.)
    #   local irrelevance queries           → report_error
    #   external abstention                 → not imported by this generator
    #   normal / recovery / dependency      → final_answer (main slice)
    #
    is_no_tool = scenario_type in ("no_tool_or_abstention", "irrelevant")
    is_optional_tool = scenario_type in (
        "clarification_required", "missing_function",
    )
    if is_no_tool and real_required_tools:
        raise ValueError(
            f"No-tool task {task.task_id} unexpectedly has "
            f"{len(real_required_tools)} tool calls"
        )
    if not is_no_tool and not is_optional_tool and not real_required_tools:
        raise ValueError(
            f"Tool task {task.task_id} has oracle length "
            f"{len(real_required_tools)}, expected at least one call"
        )

    # ── P1-2(now P0): tool tasks must be chain-seeded ──
    # Normal MCP conversations require a dependency-graph seed.
    # chain-seed query (§3.2 Step 2).  Unseeded fallback data pollutes the
    # training distribution — reject before Parquet.
    if not is_no_tool and not is_optional_tool:
        generation_mode = (
            task.metadata.get("generation_mode", "")
            if task.metadata
            else ""
        )
        if generation_mode != "chain_seeded":
            raise ValueError(
                f"Task {task.task_id}: generation_mode='{generation_mode}', "
                f"expected 'chain_seeded' for tool-task baseline. "
                f"Unseeded fallback is NOT allowed in baseline training data."
            )

    # ── P3c: Detect final_answer tasks whose oracle did not produce state
    # criteria despite the user query requesting a write/mutate action.
    # These tasks teach models to call a few tools then final_answer without
    # actually completing the user's request.  We only WARN (not reject)
    # because some legitimate operations (e.g. send_email) are not tracked
    # in the state machine and naturally have empty criteria.
    if terminal_action == "final_answer" and real_required_tools:
        criteria = _task_success_criteria(task)
        if not criteria:
            state_changing = [t for t in real_required_tools
                             if is_mutating_tool(t, task.target_servers[0])
                             and t not in SELF_CONTAINED_WRITE_TOOLS]
            if state_changing:
                # Empty criteria remain valid; R_coverage
                # operates on tool-call sequences, not state diffs (§3.3).
                # Rejecting here conflicts with the oracle length [1,8] gate
                # above (which already accepted the task) and causes ~50% yield
                # loss.  The task still has a valid oracle trace; empty criteria
                # just means R_state will not reward this specific dimension.
                logger.warning(
                    f"Task {task.task_id}: final_answer with {state_changing} "
                    f"but empty success_criteria.  Accepting — R_coverage "
                    f"will use pure tool-call matching."
                )

    # ── P3d: tool_error_recovery with empty criteria is semantically broken ──
    # tool_error_recovery indicates the oracle encountered execution failures
    # and performed recovery steps.  If the oracle trace uses only readonly
    # tools (e.g. get_order + list_orders in food_delivery), P3c won't catch
    # it because none of the tools are mutating.  But a recovery scenario
    # without any state change means the "recovery" was just reading data —
    # the task teaches nothing useful about error handling.
    if scenario_type == "tool_error_recovery":
        criteria = _task_success_criteria(task)
        if not criteria:
            # Empty criteria remain valid.
            # tool_error_recovery classification is based on execution
            # history heuristics, not ground truth.  An empty-criteria
            # recovery task still has a valid oracle trace; R_coverage
            # will use pure tool-call matching.
            logger.warning(
                f"Task {task.task_id} scenario=tool_error_recovery with "
                f"empty success_criteria — oracle tools {real_required_tools} "
                f"are all readonly.  Accepting — R_coverage will use pure "
                f"tool-call matching."
            )

    # ── P0-2: validate round contract integrity before Parquet export ──
    contracts = _build_round_contracts(task)
    queries = task.conversation_queries or [task.user_prompt]
    n_contracts = len(contracts)
    n_queries = len(queries)
    if n_contracts != n_queries:
        raise ValueError(
            f"Task {task.task_id}: {n_contracts} round_contracts vs "
            f"{n_queries} conversation queries — counts must match."
        )
    for i, c in enumerate(contracts):
        if c.get("round_idx", -1) != i:
            raise ValueError(
                f"Task {task.task_id}: round_contracts[{i}] "
                f"round_idx={c.get('round_idx')}, expected {i}"
            )
        required = c.get("required_tools", [])
        if not isinstance(required, list) or not all(isinstance(t, str) for t in required):
            raise ValueError(
                f"Task {task.task_id}: round_contracts[{i}].required_tools "
                f"must be list[str], got {type(required)}"
            )
        allowed = c.get("allowed_terminal_actions", [])
        if not isinstance(allowed, list) or not allowed:
            raise ValueError(
                f"Task {task.task_id}: round_contracts[{i}]."
                f"allowed_terminal_actions must be non-empty list[str], "
                f"got {allowed}"
            )
        for a in allowed:
            if a not in ("final_answer", "ask_clarification", "report_error"):
                raise ValueError(
                    f"Task {task.task_id}: round_contracts[{i}]."
                    f"allowed_terminal_actions contains unknown action '{a}'"
                )

    generation_method = str((task.metadata or {}).get("generation_method", ""))
    if generation_method in {"task_planner", "irrelevant_teacher_fsm"}:
        attempt_trace = (task.metadata or {}).get("teacher_attempt_trace")
        round_trace = (task.metadata or {}).get("teacher_round_trace")
        if not isinstance(attempt_trace, list):
            raise ValueError(
                f"Task {task.task_id}: missing canonical teacher_attempt_trace"
            )
        if len(attempt_trace) != int(
            (task.metadata or {}).get("teacher_attempt_count", -1)
        ):
            raise ValueError(
                f"Task {task.task_id}: teacher_attempt_trace/count mismatch"
            )
        if not isinstance(round_trace, list) or len(round_trace) != n_queries:
            raise ValueError(
                f"Task {task.task_id}: teacher_round_trace/query mismatch"
            )
        for round_idx, trace in enumerate(round_trace):
            if not isinstance(trace, dict) or trace.get("round_idx") != round_idx:
                raise ValueError(
                    f"Task {task.task_id}: invalid teacher round trace "
                    f"at index {round_idx}"
                )
            if str(trace.get("user_query", "")) != str(queries[round_idx]):
                raise ValueError(
                    f"Task {task.task_id}: teacher round query mismatch "
                    f"at index {round_idx}"
                )
    # ── P0-3: dependency edge integrity ──
    # Chain-seeded tasks MUST produce exactly len(chain_seed)-1 valid edges.
    # Incomplete or invalid edges are data integrity errors — reject before split.
    chain_seed = (
        task.metadata.get("chain_seed", [])
        if task.metadata
        else []
    )
    if chain_seed:
        aligned_indices = align_sampled_chain(
            oracle_calls_serialized,
            chain_seed,
            verified_dependency_evidence=(
                task.metadata.get("verified_dependency_evidence", [])
                if len(chain_seed) > 1 else None
            ),
        )
        if aligned_indices is None:
            raise ValueError(
                f"Task {task.task_id}: sampled dependency chain does not "
                f"align to canonical oracle: chain_seed={chain_seed}"
            )
        dependency_edges = dependency_edges_from_alignment(aligned_indices)
        expected_edge_count = len(chain_seed) - 1
        if len(dependency_edges) != expected_edge_count:
            raise ValueError(
                f"Task {task.task_id}: incomplete dependency graph: "
                f"got {len(dependency_edges)} edges, "
                f"expected {expected_edge_count}; "
                f"chain_seed={chain_seed}"
            )
        # Validate every edge: src < dst, indices in range of real_required_tools
        for edge in dependency_edges:
            if (
                not isinstance(edge, list)
                or len(edge) != 2
                or not all(isinstance(i, int) for i in edge)
                or edge[0] < 0
                or edge[1] >= len(real_required_tools)
                or edge[0] >= edge[1]
            ):
                raise ValueError(
                    f"Task {task.task_id}: invalid dependency edge "
                    f"{edge}; chain_seed={chain_seed}; "
                    f"oracle_tool_count={len(real_required_tools)}"
                )
        auxiliary_indices = auxiliary_tool_call_indices(
            oracle_calls_serialized, aligned_indices,
        )
        all_tool_indices = {
            index
            for index, call in enumerate(oracle_calls_serialized)
            if call.get("action", "tool_call") == "tool_call"
        }
        if set(aligned_indices) & set(auxiliary_indices) or (
            set(aligned_indices) | set(auxiliary_indices)
        ) != all_tool_indices:
            raise ValueError(
                f"Task {task.task_id}: dependency/auxiliary partition does "
                "not cover canonical oracle tool calls exactly once"
            )

    # ── P0: missing-function contract integrity ──
    # Enforce that missing-function samples are internally consistent before
    # they reach Parquet / rollout / reward.
    has_missing_func = bool(
        (task.metadata or {}).get("has_missing_function")
    )
    if has_missing_func:
        hidden_tools_list = list(task.hidden_tools) if task.hidden_tools else []
        hidden_tool = (task.metadata or {}).get("hidden_tool", "")
        visible_names = {t.get("name", "") for t in (task.visible_tools or [])}

        # 1. hidden_tools must be non-empty and consistent with metadata
        if not hidden_tools_list:
            raise ValueError(
                f"Task {task.task_id}: has_missing_function=True but "
                f"hidden_tools is empty — missing-function contract broken."
            )
        if hidden_tool and hidden_tool not in hidden_tools_list:
            raise ValueError(
                f"Task {task.task_id}: metadata.hidden_tool='{hidden_tool}' "
                f"not in hidden_tools={hidden_tools_list}."
            )

        # 2. hidden tool must NOT appear in visible_tool_names (schema leak)
        leaked = set(hidden_tools_list) & visible_names
        if leaked:
            raise ValueError(
                f"Task {task.task_id}: hidden tool(s) {leaked} still present "
                f"in visible_tools schema — schema leak."
            )

        # 3. hidden tool must NOT appear in oracle tool calls
        oracle_tool_names = {
            call["tool_name"] for call in oracle_calls_serialized
            if call.get("action", "tool_call") == "tool_call"
        }
        oracle_blocked = set(hidden_tools_list) & oracle_tool_names
        if oracle_blocked:
            raise ValueError(
                f"Task {task.task_id}: hidden tool(s) {oracle_blocked} "
                f"appear in oracle tool calls — execution block failed."
            )

        # 4. terminal must be ask_clarification or report_error
        if terminal_action not in ("ask_clarification", "report_error"):
            raise ValueError(
                f"Task {task.task_id}: missing_function terminal is "
                f"'{terminal_action}', expected ask_clarification or report_error."
            )

def _filter_training_eligible_tasks(tasks: list) -> list:
    eligible = []
    dropped = 0
    for task in tasks:
        if task.metadata.get("paper_replay_valid") is not True:
            dropped += 1
            logger.warning(
                "Dropping generated task before split: {} has no positive "
                "PROVE replay evidence",
                task.task_id,
            )
            continue
        if task.metadata.get("provenance_valid") is not True:
            dropped += 1
            logger.warning(
                "Dropping generated task before split: {} has no positive "
                "PROVE provenance evidence",
                task.task_id,
            )
            continue
        try:
            _validate_task_training_contract(task)
        except ValueError as exc:
            dropped += 1
            logger.warning("Dropping generated task before split: {}", exc)
            continue
        eligible.append(task)
    if dropped:
        logger.warning(
            "Dropped {} generated task(s) that violate the training contract",
            dropped,
        )
    return eligible

def _filter_required_tool_tasks(tasks: list) -> list:
    """Keep candidates with at least one required executable tool call."""
    kept = [
        task for task in tasks
        if task.oracle_program
        and any(
            getattr(call, "action", "tool_call") == "tool_call"
            for call in task.oracle_program.calls
        )
    ]
    dropped = len(tasks) - len(kept)
    if dropped:
        logger.info(
            "Tool-required generation excluded {} no-tool candidate(s)",
            dropped,
        )
    return kept
