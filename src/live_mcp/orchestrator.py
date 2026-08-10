"""State-machine task generation over live MCP servers.

Per environment:
  1. Auto-discover tool dependency graph via live MCP probing
  2. State machine alternating LLM decisions and tool execution
     against a live MCP server
  3. Robustness knobs applied before Teacher processing
  4. Replay-validate each perturbed conversation before conversion
"""

from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import os
import random
import re
import tempfile
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from enum import Enum
from pathlib import Path
from typing import Any

from loguru import logger

from src.live_mcp.config import SuiteConfig, project_root
from src.live_mcp.dependency_trace import (
    align_sampled_chain,
    auxiliary_tool_call_indices,
    verify_implicit_edges_counterfactually,
)
from src.live_mcp.contracts.catalog import domain_contract_registry
from src.live_mcp.executor import LiveMCPExecutor
from src.live_mcp.live_state_query_view import (
    compact_sampling_context as _compact_sampling_context,
    live_context_to_prompt_state as _live_context_to_prompt_state,
    teacher_public_action_context as _teacher_public_action_context,
)
from src.live_mcp.registry.environment_metadata import build_environment_metadata
from src.live_mcp.registry.entity_state_contracts import entity_state_is_known
from src.live_mcp.protocol.manager import LiveMCPManager
from src.live_mcp.state_seeder import StateSeeder
from src.live_mcp.protocol.observation import (
    tool_result_envelope,
)
from src.live_mcp.prompt_profiles import resolve_prompt_profile
from src.live_mcp.task_spec import (
    DECISION_STRATA,
    DifficultyVector,
    TaskSpec,
    compile_task_spec,
)
from src.live_mcp.types import LiveTask, OracleCall, OracleProgram, to_plain
from src.live_mcp.registry.tool_semantics import (
    resolve_tool_execution_semantics,
    tool_call_invalidated_by_state_changes,
    unresolved_failed_tool_names,
)
from src.live_mcp.fsm import (
    ConversationFSM,
    FSMStateGroup,
    RobustnessPlan,
    teacher_tool_call_budget,
)
from src.live_mcp.generation.robustness import (
    build_teacher_visible_tools as _build_teacher_visible_tools,
    has_successful_state_transition_noop as _has_successful_state_transition_noop,
    missing_function_has_nonprefix_mutation as _missing_function_has_nonprefix_mutation,
    missing_function_original_round_should_abort as _missing_function_original_round_should_abort,
    normalized_policy_query as _normalized_policy_query,
    profile_scenario_is_valid as _profile_scenario_is_valid,
    successful_conversation_realizes_source_chain as _successful_conversation_realizes_source_chain,
    successful_distractor_names as _successful_distractor_names,
    zero_tool_terminal_is_valid as _zero_tool_terminal_is_valid,
)
from src.live_mcp.generation.turn_loop import run_turn_loop
from src.live_mcp.generation.conversation_runner import (
    run_candidate_conversation,
)
from src.live_mcp.generation.candidate_finalize import (
    FinalizationContext,
    finalize_generated_task,
)
from src.live_mcp.generation.candidate_prepare import prepare_candidate
from src.live_mcp.generation.candidate_pipeline import GenerationCandidateMixin
from src.live_mcp.generation.scenario import (
    classify_scenario,
)
from src.live_mcp.generation.batch import BatchGenerationMixin
from src.live_mcp.generation.irrelevance import IrrelevanceGenerationMixin
from src.live_mcp.generation.context_provider import GenerationContextMixin
from src.live_mcp.generation.chain_scheduler import ChainSchedulerMixin
from src.live_mcp.replay.task_outcome import (
    attribute_success_criteria as _attribute_success_criteria,
    identity_policy_for_domain as _identity_policy_for_domain,
    oracle_deleted_target_ids as _oracle_deleted_target_ids,
    oracle_target_ids as _oracle_target_ids,
    protected_fields_by_resource as _protected_fields_by_resource,
    stable_state_hash as _stable_state_hash,
)
from src.live_mcp.artifact.task_builder import build_live_task

from src.utils import extract_json as _extract_json


from src.live_mcp.dependency_value_flow import (
    _value_is_explicit_in_query,
    _required_arguments_by_tool,
    _novel_dependency_output_fields,
    _dependency_argument_bindings,
    _sampled_chain_edges,
    _operational_dependency_contracts,
    _difficulty_vector_for_chain,
    _decision_stratum,
    _verify_dependency_evidence,
    _verify_realized_chain_dependencies,
)
from src.live_mcp.dependency_cache_contract import DependencyCacheContractMixin
from src.live_mcp.dependency_cache_store import DependencyCacheStoreMixin
from src.live_mcp.dependency_chain_catalog import DependencyChainCatalogMixin
from src.live_mcp.dependency_classifier import DependencyClassifierMixin
from src.live_mcp.dependency_graph_builder import DependencyGraphBuilderMixin
from src.live_mcp.dependency_relation_audit import DependencyRelationAuditMixin


class TaskOrchestrator(
    GenerationCandidateMixin,
    BatchGenerationMixin,
    IrrelevanceGenerationMixin,
    GenerationContextMixin,
    ChainSchedulerMixin,
    DependencyGraphBuilderMixin,
    DependencyChainCatalogMixin,
    DependencyClassifierMixin,
    DependencyCacheStoreMixin,
    DependencyRelationAuditMixin,
    DependencyCacheContractMixin,
):
    """State-machine task generator for live MCP environments.

    1. Auto-discover dependency graph (cached per domain)
    2. Sample robustness plan before state-machine execution
    3. State machine: Teacher operates on perturbed schemas
       (LLM-in-the-loop at every turn, against live MCP server)
    4. Replay-validate the final perturbed conversation against fresh session
    5. Provenance check on final conversation

    Usage:
        client = LLMClient(
            mode="openai", model_path="Gemma-4-31B-it", api_base="...",
        )
        orch = TaskOrchestrator(suite_config, manager, executor, client)
        tasks = orch.generate_many("all", count=100, seed=42)
    """

    # Refresh compact context after the configured number of conversations.
    # Each sampling epoch clones one deterministic initial-state seed into
    # isolated sessions, so the readonly context is reusable for K candidates
    # without sharing mutations. A new epoch/state profile/schema refreshes it.
    SAMPLING_CONTEXT_REFRESH_K: int = 10
    DEPENDENCY_CACHE_VERSION: int = 8
    DEPENDENCY_PAIR_BATCH_SIZE: int = 2
    DEPENDENCY_CLASSIFICATION_MAX_TOKENS: int = 512
    DEPENDENCY_FAILURE_RETRY_SECONDS: float = 60.0
    CHAIN_SAMPLING_JACCARD_THRESHOLD: float = 0.70
    # Tests may override this root explicitly; production remains checkout-anchored.
    DEPENDENCY_CACHE_ROOT: Path | None = None

    def __init__(
        self,
        suite_config: SuiteConfig,
        manager: LiveMCPManager,
        executor: LiveMCPExecutor,
        client: Any,
        *,
        prompt_profile: str = "paper_generation_baseline_v1",
        chain_sampling_stats: dict[str, dict[str, dict[str, int]]] | None = None,
        chain_sampling_sequences: dict[str, dict[str, tuple[str, ...]]] | None = None,
        chain_sampling_lock: threading.RLock | None = None,
    ):
        self.suite_config = suite_config
        self.manager = manager
        self.executor = executor
        self.client = client
        self.prompt_profile = resolve_prompt_profile(prompt_profile)
        # In-process artifacts must use the same schema/classifier identity as
        # the on-disk cache.  A domain-only key can silently retain old chains
        # after a live MCP schema or Teacher model changes.
        self._domain_graphs: dict[tuple[str, str, str], dict] = {}
        self._domain_chains: dict[tuple[str, str, str], list] = {}
        self._chain_filter_stats: dict[
            tuple[str, str, str], dict[str, int]
        ] = {}
        self._dependency_graph_lock = threading.RLock()
        self._dependency_graph_failures: dict[tuple[str, str, str], tuple[float, str]] = {}
        # Process-local candidate scheduling state. This changes only which
        # live-feasible chain is attempted next; it is not a corpus filter and
        # is intentionally not shared across shard processes.
        self._chain_sampling_stats = (
            chain_sampling_stats if chain_sampling_stats is not None else {}
        )
        self._chain_sampling_sequences = (
            chain_sampling_sequences if chain_sampling_sequences is not None else {}
        )
        self._chain_sampling_lock = chain_sampling_lock or threading.RLock()
        # Sampling-context cache per domain.
        # Each entry: {"context": dict, "call_count": int, "session_id": str}
        self._sampling_context_cache: dict[tuple[str, str, str], dict] = {}
        self._sampling_context_lock = threading.RLock()

    def _record_round_trace(
        self,
        *,
        teacher: Any,
        session_id: str,
        server_name: str,
        round_idx: int,
        phase: str,
        current_query: str,
        visible_tools: list[dict[str, Any]],
        public_context: dict[str, Any],
        oracle_calls: list[Any] | None = None,
    ) -> None:
        """Record the round boundary and optional true state for audit only.

        The debug state is never returned to the Teacher and never enters the
        oracle/reward path.  It is captured only when the already opt-in Teacher
        trace also enables ``LIVEMCP_TEACHER_TRACE_INCLUDE_STATE``.
        """
        recorder = getattr(teacher, "record_environment_event", None)
        if not callable(recorder):
            return
        recorder(
            "round_boundary",
            phase=phase,
            round_idx=round_idx,
            current_query=current_query,
            visible_tool_names=[
                str(tool.get("name") or "") for tool in visible_tools
            ],
            public_context=public_context,
            oracle_calls=[
                {
                    "action": getattr(call, "action", "tool_call"),
                    "tool_name": getattr(call, "tool_name", ""),
                    "server_name": getattr(call, "server_name", ""),
                    "arguments": dict(getattr(call, "arguments", {}) or {}),
                }
                for call in (oracle_calls or [])
            ],
        )
        if not bool(getattr(teacher, "trace_includes_state", False)):
            return
        try:
            state_envelope = self.manager.get_state(session_id, server_name)
            state = state_envelope.get(server_name, state_envelope)
            recorder(
                "entity_state_snapshot",
                phase=phase,
                round_idx=round_idx,
                state_hash=_stable_state_hash(state),
                state=state,
            )
        except Exception as exc:
            recorder(
                "entity_state_snapshot",
                phase=phase,
                round_idx=round_idx,
                error=f"{type(exc).__name__}: {exc}",
            )

    _run_turn_loop = run_turn_loop

    def _to_live_task(self, server_name: str, query: str, session_id: str, seed: int,
                      all_tools: list[dict], oracle_program, required_tools: list[str],
                      difficulty: str, task_id: str,
                      conversation_queries: list[str] | None = None,
                      oracle_calls_per_round: list[list] | None = None,
                      execution_history_per_round: list[list] | None = None,
                      sampling_context: dict[str, Any] | None = None) -> LiveTask:
        return build_live_task(
            suite_config=self.suite_config,
            registry=self.manager.registry,
            teacher_client=self.client,
            server_name=server_name,
            query=query,
            session_id=session_id,
            seed=seed,
            all_tools=all_tools,
            oracle_program=oracle_program,
            required_tools=required_tools,
            difficulty=difficulty,
            task_id=task_id,
            conversation_queries=conversation_queries,
            oracle_calls_per_round=oracle_calls_per_round,
            execution_history_per_round=execution_history_per_round,
            sampling_context=sampling_context,
        )
