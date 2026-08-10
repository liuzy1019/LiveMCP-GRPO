"""Shard Row Projection."""

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

from src.live_mcp.corpus.shard_oracle import (
    _build_round_contracts,
    _identity_policy,
    _minimum_action_budget,
    _required_workflow_projection_summary,
    _serialize_training_oracle,
    _task_scenario,
    _task_success_criteria,
    _validate_task_training_contract,
)
from src.live_mcp.corpus.semantic_core import expected_artifact_purpose

def _compute_dependency_edges(
    oracle_calls: list[dict],
    chain_seed: list[str],
    verified_dependency_evidence: object | None = None,
) -> list[list[int]]:
    """Compute dependency edges E by aligning chain_seed to oracle_calls.

    Algorithm:
      1. Map every tool_call in oracle_calls to its index, grouped by tool_name.
      2. Walk chain_seed left-to-right, consuming the next occurrence after the
         previous cursor.  A chain step can be both a dst (of the previous edge)
         and a src (of the next edge) — intermediate nodes are NOT consumed.
      3. If any chain step cannot be aligned (no occurrence after cursor), return
         an empty list.  The caller (_validate_task_training_contract) rejects
         chain-seeded tasks with incomplete edges.

    Returns:
        list[list[int]] — [[src_idx, dst_idx], ...] where src_idx < dst_idx,
        or [] if the chain cannot be aligned to the oracle sequence.
    """
    aligned = align_sampled_chain(
        oracle_calls,
        chain_seed,
        verified_dependency_evidence=verified_dependency_evidence,
    )
    if aligned is None:
        cursor = -1
        for tool_name in chain_seed:
            partial = align_sampled_chain(
                oracle_calls[cursor + 1:], [tool_name],
            )
            if partial is None:
                break
            cursor += partial[0] + 1
        else:
            cursor = len(oracle_calls) - 1
        if chain_seed:
            logger.warning(
                "_compute_dependency_edges: cannot align chain step "
                "after oracle index {}. chain_seed={}", cursor, chain_seed,
            )
        return []
    return dependency_edges_from_alignment(aligned)


def _tool_owner_domains(
    visible_tools: list[dict], primary_domain: str,
) -> dict[str, str]:
    owners: dict[str, str] = {}
    for tool in visible_tools:
        name = str(tool.get("name") or "")
        if not name:
            continue
        if name in owners:
            raise RuntimeError(
                f"policy-visible tool name is ambiguous across owners: {name}"
            )
        owner = str(tool.get("_server_name") or primary_domain)
        if not owner:
            raise RuntimeError(f"policy-visible tool has no owner: {name}")
        owners[name] = owner
    return owners

def _tasks_to_rows(tasks: list, _base_seed: int) -> list[dict]:
    """Convert LiveTask list to verl-compatible data rows."""
    rows = []
    skipped_no_tools = 0
    for task in tasks:
        _validate_task_training_contract(task)
        teacher_model_id = str(
            task.metadata.get("teacher_model_id") or ""
        ).strip()
        if not teacher_model_id:
            raise RuntimeError(
                "generated task is missing teacher_model_id provenance: "
                f"task={task.task_id!r}"
            )

        query_chain_supported = task.metadata.get("query_chain_supported")
        if query_chain_supported is not None and not isinstance(
            query_chain_supported, bool
        ):
            raise RuntimeError(
                "query_chain_supported must be true, false, or null: "
                f"task={task.task_id!r}, value={query_chain_supported!r}"
            )
        expected_query_chain_status = (
            "verified_by_query_contract"
            if query_chain_supported is True
            else (
                "rejected_by_query_contract"
                if query_chain_supported is False
                else "unverified_paper_baseline"
            )
        )
        query_chain_support_status = str(
            task.metadata.get("query_chain_support_status")
            or expected_query_chain_status
        )
        if query_chain_support_status != expected_query_chain_status:
            raise RuntimeError(
                "query-chain support value/status mismatch: "
                f"task={task.task_id!r}, value={query_chain_supported!r}, "
                f"status={query_chain_support_status!r}, "
                f"expected={expected_query_chain_status!r}"
            )

        # Determine visible tools — use task-provided tools, fall back to required
        visible_tools = task.visible_tools if task.visible_tools else []
        if not visible_tools:
            skipped_no_tools += 1
            logger.warning(
                f"Skipping task {task.task_id}: no visible_tools "
                f"(required_tools={task.required_tools}, "
                f"oracle_calls={len(task.oracle_program.calls) if task.oracle_program else 0})"
            )
            continue  # Skip tasks without tool schemas

        visible_tool_names = [t.get("name", "") for t in visible_tools if t.get("name")]

        domain = task.target_servers[0] if task.target_servers else "unknown"

        domain_desc = DOMAIN_DESCRIPTIONS.get(domain, "")
        reference_date = task.metadata.get("reference_date", "")
        # Robustness knobs (enum stripping, distractors, missing_function) are
        # applied inside generate_one BEFORE Teacher processing and Replay.
        # task.visible_tools already contains the Teacher-visible candidate set.
        tools_text = format_tools(visible_tools)
        date_line = f"\nToday's date: {reference_date}." if reference_date else ""
        prompt_profile = str(
            task.metadata.get("prompt_profile")
            or "paper_generation_baseline_v1"
        )
        semantic_gate_profile = str(
            task.metadata.get("semantic_gate_profile", "diagnostic_only")
        )
        artifact_purpose = expected_artifact_purpose({
            "prompt_profile": prompt_profile,
            "semantic_gate_profile": semantic_gate_profile,
        })
        initial_action_context = (
            task.sampling_context.get("initial_action_context", {})
            if isinstance(task.sampling_context, dict) else {}
        )
        initial_entity_summaries = (
            initial_action_context.get("entity_summaries", [])
            if isinstance(initial_action_context, dict) else []
        )
        initial_entity_summaries = [
            str(summary) for summary in initial_entity_summaries[:15]
            if str(summary).strip()
        ]
        observable_context = ""
        if (
            initial_entity_summaries
            and not resolve_prompt_profile(prompt_profile).policy_private
        ):
            observable_context = (
                "\n\n## Current Grounded Entities (Observable Context)\n"
                + "\n".join(initial_entity_summaries)
            )

        system_prompt = (
            f"You are an AI assistant for the following domain:\n{domain_desc}\n\n"
            f"## Available Tools\n{tools_text}{observable_context}\n\n"
            f"## Response Format\n"
            f"Output exactly ONE action per turn using XML tags:\n\n"
            f"- <tool_call>{{\"name\": \"<tool_name>\", \"arguments\": {{...}}}}</tool_call>\n"
            f"  Call a tool with its required parameters.\n\n"
            f"- <final_answer>your answer</final_answer>\n"
            f"  When the task is fully completed.\n\n"
            f"- <report_error>brief reason</report_error>\n"
            f"  When the task cannot be completed with available tools.\n\n"
            f"- <ask_clarification>what you need to know</ask_clarification>\n"
            f"  When genuinely missing critical information and no tool can resolve it.\n\n"
            f"## Rules\n"
            f"- Call ONE tool at a time. Wait for the result before the next action.\n"
            f"- Do not output hidden reasoning, chain-of-thought, or <think> tags.\n"
            f"- Use entity IDs only when they appear in the user request or tool results. "
            f"Never invent or guess IDs.\n"
            f"- If a required opaque ID is absent from the user request and prior "
            f"tool results, and an available list/search/find/get tool can discover "
            f"it, call that discovery tool before the downstream action. A natural "
            f"selector is not an ID. Never copy an identifier from descriptive "
            f"prompt text.{date_line}"
        )

        # One row always starts from reset(session_seed).  Teacher tool calls
        # are never exposed in the initial prompt. For continuation
        # data, the rollout loop injects conversation_queries[1:] after
        # intermediate terminal actions in the same live MCP session.
        prompt = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task.user_prompt},
        ]
        n_conversation_rounds = len(task.conversation_queries) or 1
        conversation_queries = list(task.conversation_queries) if task.conversation_queries else [task.user_prompt]

        has_distractors = task.metadata.get("has_distractors", False)
        has_missing_func = task.metadata.get("has_missing_function", False)

        # perturbation_level encodes query information completeness, not the
        # robustness knob. Keep difficulty intact; expose knob status via the
        # separate scenario_type/has_* fields.
        perturbation_level = task.difficulty
        scenario_type = _task_scenario(task)

        # 每个 task 独立一组：verl repeat(N) 后同一 prompt 的 N 个 rollout
        # 自然形成一个 group，回归标准 GRPO per-prompt 对比语义
        group_id = task.task_id

        # The prompt contains no teacher trajectory, so the complete oracle
        # (tool calls plus one explicit terminal action) is the unresolved
        # ground truth from reset(session_seed).  Multi-round teacher internals
        # can include per-round terminal actions; training rows keep only the
        # final terminal so the reward contract remains single-terminal.
        oracle_calls_serialized = _serialize_training_oracle(task)
        projection_summary = _required_workflow_projection_summary(task)
        round_contracts = _build_round_contracts(task)
        minimum_action_budget = _minimum_action_budget(
            oracle_calls_serialized, round_contracts,
        )
        action_budget = max(int(task.max_turns), minimum_action_budget)

        success_criteria = _task_success_criteria(task)

        # success_criteria is a list[dict] whose 'value' field can hold mixed
        # types. Serialize to JSON for a stable Parquet round-trip; reward side
        # parses it back via json.loads.
        success_criteria_json = json.dumps(
            success_criteria, ensure_ascii=False, default=str
        )

        # terminal_actions and real_required_tools were checked above by
        # _validate_task_training_contract; keep the local assert as an internal
        # consistency guard for this serialization block.
        terminal_actions = [
            c["action"] for c in oracle_calls_serialized
            if c.get("action") in ("final_answer", "ask_clarification", "report_error")
        ]
        assert terminal_actions, f"Bug: {task.task_id} serialized oracle has no terminal"
        allowed_terminal_actions = [terminal_actions[-1]]
        real_required_tools = [
            c["tool_name"] for c in oracle_calls_serialized
            if c.get("action", "tool_call") == "tool_call"
        ]

        # ── Dependency edges used by coverage scoring ──
        chain_seed = task.metadata.get("chain_seed", []) if task.metadata else []
        verified_dependency_evidence = task.metadata.get(
            "verified_dependency_evidence", []
        ) if task.metadata else []
        dependency_edges = _compute_dependency_edges(
            oracle_calls_serialized,
            chain_seed,
            verified_dependency_evidence=(
                verified_dependency_evidence if len(chain_seed) > 1 else None
            ),
        )
        canonical_alignment = align_sampled_chain(
            oracle_calls_serialized,
            chain_seed,
            verified_dependency_evidence=(
                verified_dependency_evidence if len(chain_seed) > 1 else None
            ),
        )
        if canonical_alignment is None:
            raise ValueError(
                f"Task {task.task_id}: sampled dependency chain is not fully "
                "aligned to the canonical oracle"
            )
        canonical_auxiliary_indices = auxiliary_tool_call_indices(
            oracle_calls_serialized, canonical_alignment,
        )
        canonical_tool_sequence = [
            str(call.get("tool_name") or "")
            for call in oracle_calls_serialized
            if call.get("action", "tool_call") == "tool_call"
        ]
        dependency_edges_json = json.dumps(dependency_edges, ensure_ascii=False)
        # P0 quality flag: chain_seeded tasks MUST produce exactly
        # len(chain_seed)-1 edges, enforced by _validate_task_training_contract
        # before reaching this point.  This field is diagnostic only.
        expected_edges = len(chain_seed) - 1 if chain_seed else 0
        dependency_graph_complete = (
            len(dependency_edges) == expected_edges and expected_edges > 0
        ) or not chain_seed
        generation_mode = task.metadata.get("generation_mode", "chain_seeded") if task.metadata else "chain_seeded"

        extra_info = {
            "task_id": task.task_id,
            "teacher_model_id": teacher_model_id,
            "domain": domain,
            "target_servers": task.target_servers,
            "required_tools": real_required_tools,
            "session_seed": task.session_seed,
            "initial_state_hash": task.metadata.get("initial_state_hash", ""),
            "server_schema_hash": task.metadata.get("server_schema_hash", ""),
            "server_schema_hashes": json.dumps(
                task.metadata.get("server_schema_hashes", {}),
                ensure_ascii=False,
                sort_keys=True,
            ),
            "transition_fingerprints": json.dumps(
                task.metadata.get("transition_fingerprints", {}),
                ensure_ascii=False,
                sort_keys=True,
            ),
            "initial_state_hashes": json.dumps(
                task.metadata.get("initial_state_hashes", {}),
                ensure_ascii=False,
                sort_keys=True,
            ),
            "state_profiles": json.dumps(
                task.metadata.get("state_profiles", {}),
                ensure_ascii=False,
                sort_keys=True,
            ),
            "reward_fingerprint": task.metadata.get(
                "reward_fingerprint", ""
            ),
            "reward_profile_fingerprints": json.dumps(
                task.metadata.get("reward_profile_fingerprints", {}),
                ensure_ascii=False,
                sort_keys=True,
            ),
            "observation_schema_version": task.metadata.get(
                "observation_schema_version", ""
            ),
            "observation_projection_version": task.metadata.get("observation_projection_version", ""),
            "trajectory_schema_version": task.metadata.get(
                "trajectory_schema_version", ""
            ),
            "max_observation_chars": int(
                task.metadata.get("max_observation_chars", 4096)
            ),
            "reward_profile_compatibility": list(
                task.metadata.get(
                    "reward_profile_compatibility",
                    ["prove_baseline", "oval_full"],
                )
            ),
            "user_query": task.user_prompt,
            "budget": action_budget,
            "minimum_action_budget": minimum_action_budget,
            "perturbation_level": perturbation_level,
            "scenario_type": scenario_type,
            "group_id": group_id,
            "uid": task.task_id,
            "has_distractors": has_distractors,
            "has_missing_function": has_missing_func,
            "enum_stripped": task.metadata.get("strip_enums", False),
            "identity_policy": task.metadata.get("identity_policy", _identity_policy(domain)),
            "target_resource_ids": task.metadata.get("target_resource_ids", []),
            "protected_resources": task.metadata.get("protected_resources", []),
            "protected_fields": task.metadata.get("protected_fields", []),
            # JSON string avoids Arrow's unsupported empty struct type when
            # a split happens to contain no protected-field mappings.
            "protected_fields_by_resource": json.dumps(
                task.metadata.get("protected_fields_by_resource", {}),
                ensure_ascii=False,
                default=str,
            ),
            "allowed_terminal_actions": allowed_terminal_actions,
            "semantic_fingerprint": task.metadata.get("semantic_fingerprint", ""),
            "generation_method": task.metadata.get("generation_method", "task_planner"),
            "chain_seed": list(chain_seed),
            "source_chain_seed": list(
                task.metadata.get("source_chain_seed", []) if task.metadata else []
            ),
            "source_chain_edges": json.dumps(
                task.metadata.get("source_chain_edges", []) if task.metadata else [],
                ensure_ascii=False,
                default=str,
            ),
            "initial_round_realized_tool_sequence": list(
                task.metadata.get("initial_round_realized_tool_sequence", [])
                if task.metadata else []
            ),
            "initial_round_dependency_call_indices": list(
                task.metadata.get("initial_round_dependency_call_indices", [])
                if task.metadata else []
            ),
            "initial_round_auxiliary_call_indices": list(
                task.metadata.get("initial_round_auxiliary_call_indices", [])
                if task.metadata else []
            ),
            "realized_tool_sequence": canonical_tool_sequence,
            "dependency_call_indices": canonical_alignment,
            "auxiliary_call_indices": canonical_auxiliary_indices,
            "source_chain_fingerprint": str(
                task.metadata.get("source_chain_fingerprint", "")
            ),
            "dependency_semantics_version": int(
                task.metadata.get("dependency_semantics_version", 0)
            ),
            "dependency_classifier_contract_hash": str(
                task.metadata.get(
                    "dependency_classifier_contract_hash",
                    "",
                )
            ),
            "chain_sampling_attempt_number": int(
                task.metadata.get("chain_sampling_attempt_number", 0)
            ),
            "chain_sampling_jaccard_novel": bool(
                task.metadata.get("chain_sampling_jaccard_novel", False)
            ),
            "query_generation_attempts": int(
                task.metadata.get("query_generation_attempts", 1)
            ),
            "query_target_capability": str(
                task.metadata.get("query_target_capability", "")
            ),
            # This is deliberately tri-state.  PROVE's published baseline does
            # not define the local query-chain contract, so ``None`` means
            # "not evaluated", not "evaluated and rejected".
            "query_chain_supported": query_chain_supported,
            "query_chain_support_status": query_chain_support_status,
            "query_dependency_evidence": json.dumps(
                task.metadata.get("query_dependency_evidence", []),
                ensure_ascii=False,
                default=str,
            ),
            "query_mutation_evidence": json.dumps(
                task.metadata.get("query_mutation_evidence", []),
                ensure_ascii=False,
                default=str,
            ),
            "verified_dependency_evidence": json.dumps(
                task.metadata.get("verified_dependency_evidence", []),
                ensure_ascii=False,
                default=str,
            ),
            "prompt_profile": prompt_profile,
            "semantic_gate_profile": semantic_gate_profile,
            "artifact_purpose": artifact_purpose,
            "generation_seed": int(
                task.metadata.get("generation_seed", task.session_seed)
            ),
            "sampling_state_seed": int(
                task.metadata.get("sampling_state_seed", task.session_seed)
            ),
            "task_spec": json.dumps(
                task.metadata.get("task_spec", {}),
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            ),
            "task_spec_fingerprint": str(
                task.metadata.get("task_spec_fingerprint", "")
            ),
            "difficulty": str(task.difficulty),
            # Preserve the completed Teacher conversation sequence for rollout.
            # Jaccard dedup even when the required RL oracle omits an execution-
            # tagged no-progress repeat.
            "teacher_trace_tool_sequence": [
                str(call.tool_name)
                for call in (task.oracle_program.calls if task.oracle_program else [])
                if getattr(call, "action", "tool_call") == "tool_call"
            ],
            # Serialize oracle_calls as JSON so sparse heterogeneous argument
            # dicts round-trip through Parquet without struct unification.
            "oracle_calls": json.dumps(oracle_calls_serialized, ensure_ascii=False, default=str),
            "success_criteria": success_criteria_json,
            "has_state_outcome_oracle": bool(success_criteria),
            "hidden_tools": list(task.hidden_tools) if task.hidden_tools else [],
            "visible_tool_names": visible_tool_names,
            "tool_owner_domains": json.dumps(
                _tool_owner_domains(visible_tools, domain),
                ensure_ascii=False,
                sort_keys=True,
            ),
            "conversation_rounds": n_conversation_rounds,
            # Unperturbed schemas retained for robustness auditing.
            "clean_visible_tools": json.dumps(
                task.metadata.get("clean_visible_tools", visible_tools),
                ensure_ascii=False,
                default=str,
            ),
            "domain_desc": domain_desc,
            "reference_date": task.metadata.get("reference_date", ""),
            # JSON string avoids pyarrow nested-list surprises and lets the
            # live rollout inject follow-up user turns deterministically.
            "conversation_queries": json.dumps(
                conversation_queries, ensure_ascii=False, default=str
            ),
            # P0-2: per-round contracts for rollout enforcement.
            # Each contract specifies required_tools and allowed_terminal_actions
            # for one conversation round.  The rollout loop MUST validate the
            # model's terminal against the contract before injecting follow-up.
            "round_contracts": json.dumps(
                round_contracts, ensure_ascii=False, default=str
            ),
            "dependency_edges": dependency_edges_json,
            "dependency_graph_complete": dependency_graph_complete,
            "generation_mode": generation_mode,
            # P0-3: data quality signals from replay validation.
            "paper_replay_valid": task.metadata.get("paper_replay_valid"),
            "provenance_valid": task.metadata.get("provenance_valid"),
            "provenance_violation_count": int(
                task.metadata.get("provenance_violation_count", 0)
            ),
            "project_outcome_valid": task.metadata.get("project_outcome_valid", True),
            "replay_error_rate": task.metadata.get("replay_error_rate", 0.0),
            "replay_num_calls": int(task.metadata.get("replay_num_calls", 0)),
            "replay_num_errors": int(task.metadata.get("replay_num_errors", 0)),
            "teacher_attempt_count": int(
                task.metadata.get("teacher_attempt_count", 0)
            ),
            "teacher_failed_attempt_count": int(
                task.metadata.get("teacher_failed_attempt_count", 0)
            ),
            "required_workflow_projection": json.dumps(
                projection_summary, ensure_ascii=False, default=str,
            ),
            "projection_exact_repeat_dropped": int(
                projection_summary["counts"].get(
                    "exact_no_progress_repeat", 0,
                )
            ),
            "projection_state_transition_noop_dropped": int(
                projection_summary["counts"].get("state_transition_noop", 0)
            ),
            "projection_action_no_net_change_retained": int(
                projection_summary["counts"].get(
                    "action_execution_no_net_change", 0,
                )
            ),
            "teacher_attempt_trace": json.dumps(
                task.metadata.get("teacher_attempt_trace", []),
                ensure_ascii=False,
                default=str,
            ),
            "teacher_round_trace": json.dumps(
                task.metadata.get("teacher_round_trace", []),
                ensure_ascii=False,
                default=str,
            ),
            "criteria_failed_count": task.metadata.get("criteria_failed", 0),
            "success_criteria_provenance": json.dumps(
                task.metadata.get("success_criteria_provenance", []),
                ensure_ascii=False,
                default=str,
            ),
            "unattributed_success_criteria": int(
                task.metadata.get("unattributed_success_criteria", 0)
            ),
            "fsm_final_state": task.metadata.get("fsm_final_state", ""),
            "fsm_transitions": json.dumps(
                task.metadata.get("fsm_transitions", []),
                ensure_ascii=False,
                default=str,
            ),
        }

        row = {
            "prompt": json.dumps(prompt, ensure_ascii=False),
            "data_source": "live_mcp_state_machine",
            "reward_model": {
                "style": "rule",
                "ground_truth": {
                    "oracle_calls": json.dumps(oracle_calls_serialized, ensure_ascii=False, default=str),
                    "success_criteria": success_criteria_json,
                    "required_tools": real_required_tools,
                    "dependency_edges": dependency_edges_json,
                },
            },
            "extra_info": extra_info,
            "uid": extra_info["uid"],
            "group_id": group_id,
            "perturbation_level": perturbation_level,
            "scenario_type": scenario_type,
        }
        rows.append(row)

    if skipped_no_tools > 0:
        logger.warning(
            f"_tasks_to_rows 跳过了 {skipped_no_tools}/{len(tasks)} 个任务 "
            f"（visible_tools 为空）。请检查 task_planner 是否正确产出了 "
            f"visible_tools 字段。"
        )

    return rows
