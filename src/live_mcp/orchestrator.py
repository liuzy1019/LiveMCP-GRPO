"""PROVE-style state-machine task generation.

Per environment:
  1. Auto-discover tool dependency graph via live MCP probing
  2. State machine alternating LLM decisions and tool execution
     against a live MCP server
  3. Robustness knobs applied BEFORE Teacher processing (PROVE §3.2 Figure 2)
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
import time
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from loguru import logger

from src.live_mcp.config import SuiteConfig
from src.live_mcp.executor import LiveMCPExecutor
from src.live_mcp.manager import LiveMCPManager
from src.live_mcp.types import LiveTask, OracleCall, OracleProgram, to_plain
from src.utils import extract_json as _extract_json


class FSMStateGroup(str, Enum):
    """The five state groups published in PROVE §3.2."""

    QUERY = "query"
    TURN = "turn"
    TOOL_EXECUTION = "tool_execution"
    RESPONSE = "response"
    CONTINUATION = "continuation"


@dataclass
class ConversationFSM:
    """Auditable PROVE state-machine state for one synthesized conversation."""

    state: FSMStateGroup = FSMStateGroup.QUERY
    transitions: list[dict[str, Any]] = field(default_factory=list)

    def transition(
        self,
        target: FSMStateGroup,
        event: str,
        **evidence: Any,
    ) -> None:
        self.transitions.append({
            "from": self.state.value,
            "to": target.value,
            "event": event,
            **evidence,
        })
        self.state = target


@dataclass
class RobustnessPlan:
    """Immutable robustness perturbation plan, sampled before Teacher.

    Sampled once per task seed. Distractors, enum stripping, and
    missing-function affect the generation-time Teacher contract.
    """
    inject_distractors: bool = False
    distractor_tools: list[dict] = field(default_factory=list)
    strip_enums: bool = False
    missing_function: bool = False
    hidden_tool: str | None = None
    irrelevance: bool = False

    @classmethod
    def sample(
        cls,
        seed: int,
        all_tools_pool: list[dict],
        domain_tools: list[dict],
        distractor_rate: float,
        strip_enums_rate: float,
        missing_function_rate: float,
        irrelevance: bool = False,
    ) -> "RobustnessPlan":
        """Sample a deterministic robustness plan from the given seed."""
        rng = random.Random(seed)
        known_names = {t["name"] for t in domain_tools}

        # Distractor: sample 3-8 tools from other domains
        distractors: list[dict] = []
        inject_distractors = bool(all_tools_pool and rng.random() < distractor_rate)
        if inject_distractors:
            unique_candidates: dict[str, dict] = {}
            for tool in all_tools_pool:
                name = str(tool.get("name") or "")
                if name and name not in known_names and name not in unique_candidates:
                    unique_candidates[name] = tool
            candidates = list(unique_candidates.values())
            if candidates:
                n = min(len(candidates), rng.randint(3, 8))
                distractors = [dict(t) for t in rng.sample(candidates, n)]

        # Enum stripping
        strip_enums = rng.random() < strip_enums_rate

        # Missing function
        missing_function = rng.random() < missing_function_rate

        return cls(
            inject_distractors=bool(distractors),
            distractor_tools=distractors,
            strip_enums=strip_enums,
            missing_function=missing_function,
            irrelevance=irrelevance,
        )


def _strip_enums_from_schemas(tools: list[dict]) -> list[dict]:
    """Return new list of tool dicts with enum values removed from input_schema."""
    def _strip(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: _strip(child)
                for key, child in value.items()
                if key != "enum"
            }
        if isinstance(value, list):
            return [_strip(child) for child in value]
        return value

    result: list[dict] = []
    for tool in tools:
        t = dict(tool)
        t["input_schema"] = _strip(tool.get("input_schema", {}))
        result.append(t)
    return result


def _build_teacher_visible_tools(
    domain_tools: list[dict],
    plan: RobustnessPlan,
) -> list[dict]:
    """Build the tool schemas that the Teacher LLM sees.

    Distractors are included in the Teacher candidate set. For
    missing-function tasks, the hidden tool is removed so the Teacher generates
    clarification/abstention.
    """
    tools = [dict(t) for t in domain_tools]
    existing_names = {str(tool.get("name") or "") for tool in tools}
    for distractor in plan.distractor_tools:
        name = str(distractor.get("name") or "")
        if name and name not in existing_names:
            tools.append(dict(distractor))
            existing_names.add(name)
    # 1. Strip enums from Teacher-visible schemas only
    if plan.strip_enums:
        tools = _strip_enums_from_schemas(tools)
    # 2. Hide the missing-function tool
    if plan.missing_function and plan.hidden_tool:
        tools = [t for t in tools if t["name"] != plan.hidden_tool]
    return tools


class TaskOrchestrator:
    """PROVE-style state-machine task generator.

    1. Auto-discover dependency graph (cached per domain)
    2. Sample robustness plan BEFORE state machine (PROVE §3.2 Figure 2)
    3. State machine: Teacher operates on perturbed schemas
       (LLM-in-the-loop at every turn, against live MCP server)
    4. Replay-validate the final perturbed conversation against fresh session
    5. Provenance check on final conversation

    Usage:
        client = LLMClient(mode="openai", model_path="Qwen3-32B", api_base="...")
        orch = TaskOrchestrator(suite_config, manager, executor, client)
        tasks = orch.generate_many("all", count=100, seed=42)
    """

    # PROVE refreshes compact context every k conversations. This repository's
    # seeded sessions can contain different entity IDs, so cross-session reuse
    # is unsafe: the cache is invalidated on session change and force-refreshed
    # after in-session writes. K only bounds reuse inside an unchanged session.
    SAMPLING_CONTEXT_REFRESH_K: int = 10
    DEPENDENCY_CACHE_VERSION: int = 4
    DEPENDENCY_SEMANTICS_VERSION: int = 10
    DEPENDENCY_PAIR_BATCH_SIZE: int = 8
    DEPENDENCY_CLASSIFICATION_MAX_TOKENS: int = 512
    DEPENDENCY_FAILURE_RETRY_SECONDS: float = 60.0

    def __init__(
        self,
        suite_config: SuiteConfig,
        manager: LiveMCPManager,
        executor: LiveMCPExecutor,
        client: Any,
    ):
        self.suite_config = suite_config
        self.manager = manager
        self.executor = executor
        self.client = client
        self._domain_graphs: dict[str, dict] = {}     # cached dependency graphs per domain
        self._domain_chains: dict[str, list] = {}     # cached length-2 to length-5 chains
        self._dependency_graph_lock = threading.RLock()
        self._dependency_graph_failures: dict[tuple[str, str], tuple[float, str]] = {}
        # PROVE §3.2 Step 2: sampling context cache per domain.
        # Each entry: {"context": dict, "call_count": int, "session_id": str}
        self._sampling_context_cache: dict[str, dict] = {}

    @staticmethod
    def _chain_progress_for_calls(oracle_calls: list, chain_seed: list[str] | None) -> int:
        """Compute how many chain_seed prefix steps are satisfied by oracle_calls."""
        if not chain_seed:
            return 0
        progress = 0
        for call in oracle_calls:
            if getattr(call, "action", "tool_call") != "tool_call":
                continue
            if progress < len(chain_seed) and call.tool_name == chain_seed[progress]:
                progress += 1
        return progress

    def _run_turn_loop(
        self,
        teacher,
        current_query: str,
        server_tools: list[dict],
        server_name: str,
        session_id: str,
        difficulty: str,
        round_idx: int,
        turn_budget: int = 8,
        reference_date: str = "",
        max_calls_this_round: int = 0,
        chain_context: dict[str, Any] | None = None,
        blocked_tools: set[str] | None = None,
        missing_function_contract: bool = False,
        prior_execution_history: list[dict[str, Any]] | None = None,
        allow_direct_answer: bool = False,
        fsm: ConversationFSM | None = None,
    ) -> tuple[list, list[dict], list[Any], set[str], list[OracleCall], list[Any]]:
        """Run one conversation round of teacher-driven tool execution.

        max_calls_this_round: optional diagnostic/ablation cap on real tool
        calls. Baseline passes 0 (no per-round cap).

        chain_context: live-probed entity values for hallucination prevention
        in decide_action. Extracted from _extract_chain_context in generate_one.

        Returns (oracle_calls, execution_history, oracle_observations, required_tools).
        oracle_observations is 1:1 aligned with oracle_calls — each entry is the
        raw tool observation dict for the corresponding oracle call, or {} for
        terminal actions (ask_clarification, final_answer, report_error).
        """
        from src.live_mcp.types import ToolCall

        oracle_calls: list = []
        oracle_observations: list[Any] = []
        execution_history: list[dict[str, Any]] = []
        prior_history = list(prior_execution_history or [])
        required_tools: set[str] = set()
        attempt_calls: list[OracleCall] = []
        attempt_observations: list[Any] = []

        def _record_attempt(
            name: str,
            arguments: dict[str, Any],
            observation: Any,
            success: bool,
            owner_domain: str = "",
        ) -> None:
            attempt_calls.append(OracleCall(
                tool_name=name,
                arguments=dict(arguments),
                server_name=owner_domain or server_name,
                expected_success=bool(success),
            ))
            attempt_observations.append(observation if observation is not None else {})

        def _owner_domain(name: str) -> str:
            for schema in server_tools:
                if schema.get("name") == name:
                    return str(schema.get("_server_name") or server_name)
            return server_name

        # A continuation round must see the real prior execution history. The
        # marker is only a defensive fallback for legacy callers that do not
        # provide it; it prevents first-turn prompting without pretending the
        # previous tool observations are available.
        if round_idx > 0 and not prior_history:
            prior_history.append({
                "tool_name": "__prior_round__",
                "arguments": {},
                "observation": {},
                "success": True,
                "execution_status": "SUCCESS",
            })

        def _add_oracle(call: OracleCall) -> bool:
            """Append the state-machine oracle without intra-trace pruning."""
            oracle_calls.append(call)
            return True

        max_turns = max(1, int(turn_budget))

        attempt = 0          # raw LLM call count (for temperature scaling)
        _turn: int = 0       # real turn count (tool exec + terminal)

        while _turn < max_turns:
            if fsm is not None:
                fsm.transition(
                    FSMStateGroup.TURN,
                    "teacher_decision_requested",
                    round_idx=round_idx,
                    action_idx=_turn,
                )
            try:
                action = teacher.decide_action(
                    tool_schemas=server_tools,
                    user_query=current_query,
                    execution_history=prior_history + execution_history,
                    attempt=attempt,
                    difficulty=difficulty,
                    reference_date=reference_date,
                    chain_context=chain_context,
                    blocked_tools=blocked_tools,
                    missing_function=missing_function_contract,
                    allow_direct_answer=allow_direct_answer,
                )
            except RuntimeError:
                logger.debug(
                    "_run_turn_loop: decide_action exhausted retries; "
                    "breaking turn loop."
                )
                break

            if action.action == "ask_clarification":
                if fsm is not None:
                    fsm.transition(
                        FSMStateGroup.RESPONSE,
                        "teacher_terminal",
                        action="ask_clarification",
                    )
                _add_oracle(OracleCall(
                    tool_name="ask_clarification",
                    arguments={"question": action.text},
                    action="ask_clarification",
                ))
                oracle_observations.append({})
                break

            if action.action in ("final_answer", "report_error"):
                if fsm is not None:
                    fsm.transition(
                        FSMStateGroup.RESPONSE,
                        "teacher_terminal",
                        action=action.action,
                    )
                _add_oracle(OracleCall(
                    tool_name=action.action,
                    arguments={"text": action.text},
                    action=action.action,
                ))
                oracle_observations.append({})
                break

            if action.action != "tool_call" or not action.tool_name:
                continue

            tool_name = action.tool_name
            tool_name = _fuzzy_match_tool(tool_name, {t["name"] for t in server_tools}) or tool_name

            # P0: missing-function — block hidden tools at execution layer.
            # Teacher may still output the tool name (it's in chain_seed hints),
            # but the executor must never run it.  Treat as a schema-unknown call
            # so the LLM sees an error and can produce ask_clarification.
            if blocked_tools and tool_name in blocked_tools:
                if fsm is not None:
                    fsm.transition(
                        FSMStateGroup.TOOL_EXECUTION,
                        "tool_call_blocked",
                        tool_name=tool_name,
                    )
                blocked_observation = {
                    "error": f"Tool '{tool_name}' is not available in the current environment."
                }
                _record_attempt(tool_name, action.arguments, blocked_observation, False, _owner_domain(tool_name))
                execution_history.append({
                    "tool_name": tool_name,
                    "arguments": dict(action.arguments),
                    "observation": blocked_observation,
                    "success": False,
                    "execution_status": "BLOCKED",
                })
                if fsm is not None:
                    fsm.transition(
                        FSMStateGroup.RESPONSE,
                        "tool_outcome",
                        tool_name=tool_name,
                        outcome="FAILURE",
                    )
                _turn += 1
                attempt += 1
                continue

            required_tools.add(tool_name)

            execution_domain = _owner_domain(tool_name)
            if fsm is not None:
                fsm.transition(
                    FSMStateGroup.TOOL_EXECUTION,
                    "tool_call_dispatched",
                    tool_name=tool_name,
                    owner_domain=execution_domain,
                )
            result = self.executor.execute(
                session_id,
                ToolCall(tool_name, dict(action.arguments), call_id=f"sm_{_turn}"),
                domain=execution_domain,
            )
            _record_attempt(
                tool_name, action.arguments, result.observation, result.success, execution_domain,
            )

            if fsm is not None:
                fsm.transition(
                    FSMStateGroup.RESPONSE,
                    "tool_outcome",
                    tool_name=tool_name,
                    outcome=result.execution_status,
                )

            if result.execution_status == "FAILURE":
                logger.debug(
                    f"_run_turn_loop: tool '{tool_name}' execution failed for "
                    f"{server_name} (error_type={result.error_type}, "
                    f"msg={result.error_message[:80]})"
                )
                execution_history.append({
                    "tool_name": tool_name,
                    "arguments": dict(action.arguments),
                    "observation": result.observation,
                    "success": False,
                    "execution_status": result.execution_status,
                })

                # If this is the first tool call and it failed, don't give up
                # immediately — let the LLM try again with a different approach.
                # Only invoke decide_recovery if we already have at least one
                # successful oracle call, or if we've exhausted multiple attempts.
                real_oracle_count_now = sum(1 for oc in oracle_calls if oc.action == "tool_call")
                if real_oracle_count_now == 0 and attempt < max_turns - 1:
                    # First tool call failed — continue the loop so the LLM
                    # can see the error in execution_history and try again.
                    _turn += 1
                    attempt += 1
                    continue

                recovery = teacher.decide_recovery(
                    last_tool_name=tool_name,
                    last_arguments=dict(action.arguments),
                    error_observation={"error": str(result.observation)},
                    tool_schemas=server_tools,
                    execution_history=prior_history + execution_history,
                )
                rec_action = recovery.get("action", "give_up")

                if rec_action == "give_up":
                    reason = str(
                        recovery.get("reason")
                        or "The request cannot be completed with the available tools and state."
                    )
                    if fsm is not None:
                        fsm.transition(
                            FSMStateGroup.RESPONSE,
                            "recovery_give_up",
                            action="report_error",
                        )
                    _add_oracle(OracleCall(
                        tool_name="report_error",
                        arguments={"text": reason},
                        action="report_error",
                    ))
                    oracle_observations.append({})
                    break
                elif rec_action in ("retry", "retry_same"):
                    corrected = recovery.get("corrected_args", dict(action.arguments))
                    if fsm is not None:
                        fsm.transition(
                            FSMStateGroup.TOOL_EXECUTION,
                            "recovery_retry_dispatched",
                            tool_name=tool_name,
                        )
                    retry_result = self.executor.execute(
                        session_id,
                        ToolCall(tool_name, corrected, call_id=f"sm_recover_{_turn}"),
                        domain=execution_domain,
                    )
                    _record_attempt(
                        tool_name, corrected, retry_result.observation,
                        retry_result.success, execution_domain,
                    )
                    execution_history.append({
                        "tool_name": tool_name,
                        "arguments": corrected,
                        "observation": retry_result.observation if retry_result.observation is not None else {},
                        "success": bool(retry_result.success),
                        "execution_status": retry_result.execution_status,
                    })
                    if fsm is not None:
                        fsm.transition(
                            FSMStateGroup.RESPONSE,
                            "tool_outcome",
                            tool_name=tool_name,
                            outcome=retry_result.execution_status,
                            recovery="retry",
                        )
                    if retry_result.success:
                        if _add_oracle(OracleCall(
                            tool_name=tool_name,
                            arguments=corrected,
                        )):
                            oracle_observations.append(
                                retry_result.observation if retry_result.observation is not None else {}
                            )
                elif rec_action == "retry_alt":
                    alt_tool = recovery.get("tool_name", "")
                    if alt_tool and alt_tool in {t["name"] for t in server_tools}:
                        alt_domain = _owner_domain(alt_tool)
                        if fsm is not None:
                            fsm.transition(
                                FSMStateGroup.TOOL_EXECUTION,
                                "recovery_alternative_dispatched",
                                tool_name=alt_tool,
                                owner_domain=alt_domain,
                            )
                        alt_result = self.executor.execute(
                            session_id,
                            ToolCall(alt_tool, recovery.get("arguments", {}), call_id=f"sm_alt_{_turn}"),
                            domain=alt_domain,
                        )
                        _record_attempt(
                            alt_tool, recovery.get("arguments", {}), alt_result.observation,
                            alt_result.success, alt_domain,
                        )
                        execution_history.append({
                            "tool_name": alt_tool,
                            "arguments": recovery.get("arguments", {}),
                            "observation": alt_result.observation if alt_result.observation is not None else {},
                            "success": bool(alt_result.success),
                            "execution_status": alt_result.execution_status,
                        })
                        if fsm is not None:
                            fsm.transition(
                                FSMStateGroup.RESPONSE,
                                "tool_outcome",
                                tool_name=alt_tool,
                                outcome=alt_result.execution_status,
                                recovery="alternative",
                            )
                        if alt_result.success:
                            required_tools.add(alt_tool)
                            if _add_oracle(OracleCall(
                                tool_name=alt_tool,
                                arguments=recovery.get("arguments", {}),
                            )):
                                oracle_observations.append(
                                    alt_result.observation if alt_result.observation is not None else {}
                                )
                _turn += 1
                attempt += 1
                # PROVE §3.2 multi-round schedule: after recovery added an
                # oracle call, check if we hit the per-round tool-call limit.
                if max_calls_this_round > 0:
                    real_this_round = sum(
                        1 for oc in oracle_calls if oc.action == "tool_call"
                    )
                    if real_this_round >= max_calls_this_round:
                        break
                continue

            # PROVE exposes PARTIAL_SUCCESS as a distinct execution outcome.
            # It is not a recovery failure, but the next Teacher decision sees
            # the exact outcome in execution_history instead of a folded SUCCESS.
            execution_history.append({
                "tool_name": tool_name,
                "arguments": dict(action.arguments),
                "observation": result.observation if result.observation is not None else {},
                "success": True,
                "execution_status": result.execution_status,
            })

            if _add_oracle(OracleCall(
                tool_name=tool_name,
                arguments=dict(action.arguments),
            )):
                oracle_observations.append(
                    result.observation if result.observation is not None else {}
                )

            _turn += 1
            attempt += 1

            # Optional diagnostic bound.  Baseline passes zero because the
            # dependency chain is an atomic task seed, not a per-turn schedule.
            if max_calls_this_round > 0:
                real_this_round = sum(
                    1 for oc in oracle_calls if oc.action == "tool_call"
                )
                if real_this_round >= max_calls_this_round:
                    break

        # A completed round must contain a Teacher-emitted terminal action.
        # Never fabricate success when the action budget is exhausted.
        if (any(oc.action == "tool_call" for oc in oracle_calls)
                and not any(oc.action in ("final_answer", "report_error", "ask_clarification")
                            for oc in oracle_calls)):
            raise RuntimeError(
                "Teacher exhausted the per-round action budget without a "
                "terminal response"
            )
        if not oracle_calls:
            raise RuntimeError(
                "Teacher produced no action for the current conversation round"
            )

        return (
            oracle_calls,
            execution_history,
            oracle_observations,
            required_tools,
            attempt_calls,
            attempt_observations,
        )

    def generate_one(
        self,
        server_name: str,
        seed: int,
        difficulty: str,
        max_turns: int = 8,
        robustness_plan: RobustnessPlan | None = None,
    ) -> LiveTask:
        """PROVE-style state-machine generation with LLM-in-the-loop.

        1. Sample dependency chain seed (PROVE §6 step 2)
        2. Apply robustness plan to Teacher-visible schemas BEFORE Teacher
           processing (PROVE §3.2 Figure 2: knobs are inside state machine)
        3. LLM generates user_query with persona + reference_date (PROVE §4)
        4. State-machine loop: LLM decides next action → execute → recovery →
           record.  The inner step budget is ``max_turns`` (default 8).
        5. Derive success criteria from state delta
        6. Replay validate the perturbed conversation against fresh session

        robustness_plan: if None, defaults to clean (no perturbations).
        When provided, distractor/enum-stripping/missing-function are applied
        to Teacher-visible schemas only.  Executor always uses clean schemas.

        Retries with different seed if oracle_calls is empty or replay fails.
        """
        from src.live_mcp.task_planner import (
            TaskPlanner, derive_success_criteria, derive_progress_predicates,
            replay_validate,
            _PERSONA_TEMPLATES, _REFERENCE_DATES, ContinuationPolicy,
            provenance_check, _is_mutating_tool,
        )
        from src.live_mcp.types import ToolCall

        rng = random.Random(seed)

        # ── Sample diversity injectors (PROVE §4) ──
        persona = _PERSONA_TEMPLATES[seed % len(_PERSONA_TEMPLATES)]
        reference_date = _REFERENCE_DATES[(seed // len(_PERSONA_TEMPLATES)) % len(_REFERENCE_DATES)]

        # ── Sample dependency chain seed (PROVE §6 step 2) ──
        # PROVE §3.2 Step 2 guard: defer chain selection until live state is
        # available so we can filter out chains whose first step has no entity.

        # ── Conversation-level continuation ──
        # The dependency chain is one atomic user goal. Continuation is useful
        # only while that goal remains incomplete; forcing another user turn
        # after the chain is complete leaves no capability to bind the new
        # query to and produces unrelated or unsupported requests.

        # ── Defensive initialisation (all variables reused after the retry loop).
        # Candidate-level regeneration is intentionally one-shot by default.
        # PROVE performs recovery inside the state machine, then filters the
        # completed candidate with replay/provenance. Re-generating the whole
        # conversation several times hides rejection rates and wastes Teacher
        # requests; pool-level oversampling/recovery replaces it.
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
        progress_predicates: list = []
        teacher = None
        scenario_type = ""
        identity_policy = ""
        final_teacher_visible_tools: list[dict] = []
        chain_seed: list[str] | None = None
        source_chain_seed: list[str] | None = None
        all_attempt_calls: list[OracleCall] = []
        all_attempt_observations: list[Any] = []
        all_attempt_round_indices: list[int] = []
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

        # ── Retry with different seed if LLM refuses to call tools ──
        for retry_attempt in range(max_task_attempts):
            local_seed = seed + retry_attempt * 1000
            local_rng = random.Random(local_seed)

            teacher = TaskPlanner(self.client, server_name, seed=local_seed)
            conversation_fsm = ConversationFSM()

            session = self.manager.create_session(seed=local_seed)
            session_id = session.session_id
            all_tools = self.manager.discover_tools(session_id)
            server_tools = self.manager.registry.server_tools(server_name)

            # ── Robustness plan (PROVE §3.2 Figure 2) ──
            # Sampled once per retry; applied BEFORE Teacher processing so
            # the Teacher operates on the perturbed schemas and the resulting
            # conversation is replay-validated as-is.
            # Copy to avoid mutating the caller's plan across retries.
            if robustness_plan is not None:
                plan = RobustnessPlan(
                    inject_distractors=robustness_plan.inject_distractors,
                    distractor_tools=list(robustness_plan.distractor_tools),
                    strip_enums=robustness_plan.strip_enums,
                    missing_function=robustness_plan.missing_function,
                    hidden_tool=None,  # reset per retry — chain may differ
                    irrelevance=robustness_plan.irrelevance,
                )
            else:
                plan = RobustnessPlan()

            # Build the generation-time candidate set, including distractors.
            # Teacher sees perturbed schemas; executor uses clean server_tools.
            # For missing_function: we select the chain first with full tools
            # (for feasibility), pick hidden_tool from chain_seed later, then
            # rebuild teacher_visible_tools without the hidden tool.
            teacher_visible_tools = _build_teacher_visible_tools(server_tools, plan)

            try:
                grounded_state = self.manager.get_state(session_id)
                domain_state = grounded_state.get(server_name, {})
                initial_state_snapshot = copy.deepcopy(domain_state)
                live_sampling_context = self._get_live_sampling_context(
                    session_id=session_id,
                    server_name=server_name,
                    server_tools=server_tools,
                )
                dep_hints = self._get_graph_hints(server_name)

                # ── Chain selection after live state is available ──
                # Deferred from above: we need the real state to filter
                # infeasible chains whose first step has no entity.
                all_chains = self._get_chains(server_name)
                feasible_chains = self._filter_feasible_chains(
                    all_chains, server_name, live_sampling_context,
                ) if all_chains else []
                chain_seed: list[str] | None = None
                if feasible_chains:
                    chain_seed = local_rng.choice(feasible_chains)
                elif all_chains:
                    # Feasible chains exist but none passed live-state filter.
                    # Retry with a fresh session/state (PROVE: Step 2 requires
                    # executable chains). Optional candidate regeneration is
                    # controlled explicitly; default is one attempt.
                    if retry_attempt + 1 < max_task_attempts:
                        logger.debug(
                            f"No feasible chain for {server_name} "
                            f"(attempt {retry_attempt + 1}/{max_task_attempts}), re-sampling session"
                        )
                        self.manager.close_session(session_id)
                        continue
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
                query_chain_context = (
                    _extract_chain_context(
                        chain_seed=source_chain_seed,
                        server_name=server_name,
                        live_context=live_sampling_context,
                    )
                    if source_chain_seed else {}
                )
                # PROVE Step 2: the query generator must see the same
                # chain-specific, handler-feasible entity view used for
                # grounding.  Passing the full live state here lets it select
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
                # PROVE Step 1/3: the complete 2--5 step dependency chain seeds
                # one grounded task.  Continuation is a later interaction
                # mechanism; it must not be used to turn individual chain nodes
                # into separate user requests.
                query_generation_chain = source_chain_seed
                generated_query = teacher.generate_query(
                    tool_schemas=server_tools,
                    grounded_state=query_grounding_state,
                    difficulty=difficulty,
                    rng=local_rng,
                    dep_hints=dep_hints,
                    persona=persona,
                    reference_date=reference_date,
                    chain_seed=query_generation_chain,
                    chain_context=query_chain_context,
                )
                user_query = generated_query.user_query
                conversation_fsm.transition(
                    FSMStateGroup.TURN,
                    "query_generated",
                    round_idx=0,
                    difficulty=difficulty,
                    query_generation_attempts=generated_query.attempts,
                    query_target_capability=generated_query.target_capability,
                    query_chain_supported=generated_query.chain_supported,
                )
                # Missing-function query generation must use the complete chain.
                # Only the Teacher execution contract receives the hidden version.
                blocked_tools_set: set[str] | None = None
                chain_context = query_chain_context
                if plan.missing_function:
                    if not source_chain_seed:
                        logger.debug(
                            f"Missing-function requires a dependency chain for {server_name}; "
                            f"retrying with a fresh seed."
                        )
                        continue
                    else:
                        hidden_tool = source_chain_seed[-1]
                        plan.hidden_tool = hidden_tool
                        teacher_visible_tools = _build_teacher_visible_tools(server_tools, plan)
                        blocked_tools_set = {hidden_tool}
                        chain_seed = None
                        chain_context = {}

                # Accumulators across conversation rounds (PROVE CONTINUATION)
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
                task_id = f"{server_name}_{local_seed}_{local_rng.randint(0, 99999)}"
                retry_label = f" (retry {retry_attempt})" if retry_attempt > 0 else ""

                current_query = user_query
                previous_assistant_response = ""

                logger.debug(
                    f"CONTINUATION: {server_name} task {task_id} "
                    f"starting state-machine continuation "
                    f"(max={ContinuationPolicy.MAX_CONVERSATION_ROUNDS})"
                )

                round_idx = 0
                decision = "follow_up"  # dummy, overwritten on round_idx==0 path below
                while True:
                    # PROVE exposes sampled state to query generation.  The action
                    # planner receives the resulting user query, schemas, and real
                    # execution history; giving it sampler-private IDs would bypass
                    # discovery tools.  IDs must come from the user message or a
                    # prior tool observation.
                    round_action_context: dict[str, Any] = {}
                    if round_idx > 0:
                        followup_live_context = self._get_live_sampling_context(
                            session_id=session_id,
                            server_name=server_name,
                            server_tools=server_tools,
                            force_refresh=True,
                        )
                        if decision == "clarification":
                            current_query = teacher.generate_clarification(
                                tool_schemas=teacher_visible_tools,
                                grounded_state=_live_context_to_prompt_state(followup_live_context),
                                previous_query=current_query,
                                difficulty=difficulty,
                                rng=local_rng,
                                persona=persona,
                                reference_date=reference_date,
                                previous_response=previous_assistant_response,
                            )
                        else:
                            current_query = teacher.generate_followup(
                                tool_schemas=teacher_visible_tools,
                                grounded_state=_live_context_to_prompt_state(followup_live_context),
                                previous_query=current_query,
                                difficulty=difficulty,
                                rng=local_rng,
                                persona=persona,
                                reference_date=reference_date,
                                chain_seed=None,
                                chain_progress=0,
                                previous_response=previous_assistant_response,
                            )
                        conversation_queries.append(current_query)
                    else:
                        # round_idx == 0: first round, no decision yet
                        decision = "follow_up"  # dummy for the first iteration

                    max_calls_r = 0

                    (
                        round_ocs,
                        round_hist,
                        round_obs,
                        round_reqs,
                        round_attempts,
                        round_attempt_obs,
                    ) = self._run_turn_loop(
                        teacher=teacher,
                        current_query=current_query,
                        server_tools=teacher_visible_tools,  # P0: Teacher sees perturbed schemas
                        server_name=server_name,
                        session_id=session_id,
                        difficulty=difficulty,
                        round_idx=round_idx,
                        turn_budget=max_turns,
                        reference_date=reference_date,
                        max_calls_this_round=max_calls_r,
                        chain_context=round_action_context,
                        blocked_tools=blocked_tools_set,
                        missing_function_contract=plan.missing_function,
                        prior_execution_history=all_execution_history,
                        allow_direct_answer=(round_idx > 0 and decision == "clarification"),
                        fsm=conversation_fsm,
                    )

                    if round_idx == 0:
                        _real_round = [c for c in round_ocs if getattr(c, "action", "tool_call") == "tool_call"]
                        _clar_round = [c for c in round_ocs if getattr(c, "action", "tool_call") == "ask_clarification"]
                        _abstain_round = [
                            c for c in round_ocs
                            if getattr(c, "action", "tool_call") in ("ask_clarification", "report_error")
                        ]
                        allow_zero_tool = bool(
                            (difficulty == "missing" and _clar_round)
                            or (plan.missing_function and _abstain_round)
                            or (difficulty == "minimal" and _clar_round)
                        )
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
                    # machine. PROVE deduplicates conversations by Jaccard; it
                    # does not delete repeated reads/calls inside recovery or a
                    # later user round.
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

                    # PROVE §3.2 continuation is a conversation-level decision,
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
                        break
                    conversation_fsm.transition(
                        FSMStateGroup.QUERY,
                        "continuation_selected",
                        decision=decision,
                        round_idx=round_idx,
                    )
                    # follow_up or clarification: continue loop

                # If we broke out of conversation loop early (first round failed)
                _real_now = [c for c in all_oracle_calls if getattr(c, "action", "tool_call") == "tool_call"]
                _clar_now = [c for c in all_oracle_calls if getattr(c, "action", "tool_call") == "ask_clarification"]
                _abstain_now = [
                    c for c in all_oracle_calls
                    if getattr(c, "action", "tool_call") in ("ask_clarification", "report_error")
                ]
                if plan.missing_function:
                    if not _abstain_now:
                        self.manager.close_session(session_id)
                        continue
                elif not _real_now and not (difficulty == "missing" and _clar_now):
                    self.manager.close_session(session_id)
                    continue  # retry loop

                realized_chain_seed: list[str] = []
                if chain_seed and not plan.missing_function:
                    completed_chain_steps = self._chain_progress_for_calls(
                        all_oracle_calls, chain_seed,
                    )
                    if completed_chain_steps == len(chain_seed):
                        realized_chain_seed = list(chain_seed)
                    else:
                        logger.debug(
                            f"Teacher completed the bound user goal without the full "
                            f"seed chain for {server_name} task {task_id}: "
                            f"{completed_chain_steps}/{len(chain_seed)}. Keeping "
                            f"source_chain_seed for audit and omitting OVAL dependency "
                            f"edges; PROVE does not publish full-chain coverage as a "
                            f"corpus rejection gate."
                        )

                distractor_names = {
                    str(tool.get("name") or "") for tool in plan.distractor_tools
                }
                oracle_distractors = {
                    call.tool_name for call in _real_now
                    if call.tool_name in distractor_names
                }
                if oracle_distractors:
                    self.manager.close_session(session_id)
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
                    self.manager.close_session(session_id)
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
                progress_predicates = derive_progress_predicates(
                    oracle_calls=all_oracle_calls,
                    domain=server_name,
                )

# ── Replay validate (PROVE: ≤30% error rate tolerance) ──
                valid, error_rate, num_errors, num_calls, criteria_ok, criteria_failed = replay_validate(
                    oracle_calls=all_attempt_calls,
                    manager=self.manager,
                    executor=self.executor,
                    seed=local_seed,
                    domain=server_name,
                    success_criteria=success_criteria,
                    blocked_tools=blocked_tools_set,
                )
                if not valid:
                    if retry_attempt + 1 < max_task_attempts:
                        logger.debug(
                            f"Replay validation failed for {server_name}: "
                            f"{num_errors}/{num_calls} errors ({error_rate:.0%}), "
                            f"retrying (attempt {retry_attempt + 1}/3)"
                        )
                        self.manager.close_session(session_id)
                        continue
                    raise RuntimeError(
                        f"Replay validation failed for {server_name} task {task_id}: "
                        f"{num_errors}/{num_calls} errors ({error_rate:.0%})"
                    )
                # Criteria quality gate (independent of tool-error 30% filter).
                # PROVE does NOT merge criteria into replay error rate.
                if not criteria_ok:
                    logger.warning(
                        f"Replay criteria check failed for {server_name} "
                        f"task {task_id}: state not reproduced. "
                        f"Accepting — R_coverage uses pure tool-call matching."
                    )

                # ── Provenance check (PROVE §3.2 Step 5: sensitive params) ──
                prov_ok, prov_violations = provenance_check(
                    oracle_calls=all_attempt_calls,
                    user_query=conversation_queries[0],
                    aligned_observations=all_attempt_observations,
                    user_queries=conversation_queries,
                    call_round_indices=all_attempt_round_indices,
                )
                if not prov_ok:
                    if retry_attempt + 1 < max_task_attempts:
                        logger.debug(
                            f"Provenance check failed for {server_name}: "
                            f"{len(prov_violations)} untraceable sensitive params "
                            f"(e.g., {prov_violations[0]['param']} in {prov_violations[0]['tool']}), "
                            f"retrying (attempt {retry_attempt + 1}/3)"
                        )
                        self.manager.close_session(session_id)
                        continue
                    raise RuntimeError(
                        f"Provenance check failed for {server_name} task {task_id}: "
                        f"{len(prov_violations)} untraceable sensitive params"
                    )

                # ── Scenario classification (metadata only, not a gate) ──
                _real = [c for c in all_oracle_calls
                         if getattr(c, "action", "tool_call") == "tool_call"]
                _terminal = next(
                    (c.action for c in reversed(all_oracle_calls)
                     if c.action != "tool_call"),
                    "final_answer",
                )
                scenario_type = _classify_scenario(
                    server_name=server_name,
                    oracle_calls=_real,
                    execution_history=all_execution_history,
                    terminal_action=_terminal,
                    seed=local_seed,
                )
                if plan.missing_function:
                    scenario_type = (
                        "clarification_required"
                        if _terminal == "ask_clarification"
                        else "missing_function"
                    )

                # ── Guard: teacher-generated traces with empty success_criteria ──
                # Teacher models occasionally produce oracle traces that yield
                # empty success_criteria — either because the oracle used only
                # readonly tools, or because a mutating call didn't change
                # tracked state (e.g. cancel already-cancelled order).
                #
                # PROVE does NOT reject these: R_coverage is based on matching
                # oracle tool-call sequences, not on state-diff criteria (§3.3).
                # Empty success_criteria means the coverage reward operates in
                # pure tool-call-match mode, which is correct.
                # We log a warning to help diagnose pipeline health but allow
                # the task through.
                if scenario_type in frozenset({"normal_safe_success", "tool_error_recovery"}) and not success_criteria:
                    logger.warning(
                        f"Empty success_criteria for {scenario_type} task {task_id} "
                        f"(oracle has {len(_real)} tool call(s)). "
                        f"Accepting — R_coverage will use pure tool-call matching."
                    )

                # ── Success ──
                final_teacher_visible_tools = teacher_visible_tools
                generation_succeeded = True
                break

            finally:
                self.manager.close_session(session_id)

        if not generation_succeeded:
            raise RuntimeError(
                f"Teacher generation exhausted {max_task_attempts} attempt(s) for {server_name} "
                f"without a replay-valid completed user goal"
            )

        # ── Final guard: ensure the oracle matches the task type ──
        # Exception: difficulty="missing" expects clarification-only behavior
        # (PROVE missing-required information level). If the oracle has at
        # least one ask_clarification, that's a valid task — don't raise.
        real_calls = [c for c in all_oracle_calls
                      if getattr(c, "action", "tool_call") == "tool_call"]
        clarification_calls = [c for c in all_oracle_calls
                               if getattr(c, "action", "tool_call") == "ask_clarification"]
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
        elif not real_calls and not (
            difficulty in ("missing", "minimal") and clarification_calls
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

        live_task = self._to_live_task(
            server_name=server_name, query=user_query,
            session_id=session_id, seed=local_seed,
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
            },
        )
        # Preserve the same candidate contract used by the Teacher. Distractors
        # were already added before generation; missing tools remain hidden.
        live_task.visible_tools = list(
            final_teacher_visible_tools or live_task.visible_tools
        )
        # P0: hidden_tools must be set on the LiveTask object so generate_data.py
        # serialises it into Parquet and livemcp_oval_loop.py can build blocked_tools.
        if plan.hidden_tool:
            live_task.hidden_tools = [plan.hidden_tool]
            live_task.task_type = "missing_function"
        target_ids = _oracle_target_ids(real_calls)
        identity_policy = _identity_policy_for_domain(server_name)
        deleted_targets = _oracle_deleted_target_ids(real_calls)
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
        live_task.metadata.update({
            "initial_state_hash": _stable_state_hash(initial_state_snapshot),
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
            "chain_seed": [] if plan.missing_function else realized_chain_seed,
            "source_chain_seed": list(source_chain_seed) if source_chain_seed else [],
            "query_generation_attempts": generated_query.attempts,
            "query_target_capability": generated_query.target_capability,
            "query_chain_supported": generated_query.chain_supported,
            "generation_mode": "chain_seeded" if source_chain_seed else "unseeded_fallback",
            # P0-3: data quality signals from replay validation.
            # paper_replay_valid = schema/execution error rate ≤30% (PROVE §3.2 Step 5).
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
            "criteria_failed": criteria_failed,
            "robustness_applied_before_replay": True,
            "distractor_injection_stage": "pre_teacher",
            "has_distractors": plan.inject_distractors,
            "distractor_count": len(plan.distractor_tools),
            "strip_enums": plan.strip_enums,
            "has_missing_function": plan.missing_function,
            # P1-3: continuation decision schedule (local defaults, not PROVE-published)
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
        return live_task

    def generate_many(self, server_name: str, count: int, seed: int,
                      difficulty_mix: dict[str, float] | None = None,
                      irrelevance_ratio: float = 0.05,
                      distractor_rate: float = 0.40,
                      missing_function_rate: float = 1500 / (10895 + 1500),
                      ) -> list[LiveTask]:
        tasks: list[LiveTask] = []
        if server_name == "all":
            servers = self.manager.server_names
        elif "," in server_name:
            servers = [s.strip() for s in server_name.split(",") if s.strip()]
        else:
            servers = [server_name]
        if not servers:
            raise ValueError("no enabled Live MCP servers available")
        unknown = [s for s in servers if s not in self.manager.server_names]
        if unknown:
            raise ValueError(f"unknown servers: {unknown}")
        # Small shards may have fewer rows than domains. Rotate which domains
        # receive the remainder using the launcher client seed stride so the
        # merged candidate pool remains domain-balanced.
        if server_name == "all" and len(servers) > 1:
            stride_raw = os.environ.get("GENERATION_CLIENT_SEED_STRIDE", "1000000")
            try:
                stride = max(1, int(stride_raw))
            except ValueError:
                stride = 1000000
            rotation = (seed // stride) % len(servers)
            servers = servers[rotation:] + servers[:rotation]

        effective_mix = difficulty_mix or {"complete": 0.6, "missing": 0.2, "minimal": 0.2}

        # Sample per candidate instead of rounding per process. Rounding small
        # shards independently makes a 5% global ratio collapse to zero.
        n_irrelevant = sum(
            random.Random(seed + i).random() < irrelevance_ratio
            for i in range(count)
        ) if irrelevance_ratio > 0 else 0
        n_normal = count - n_irrelevant

        # Per-domain budget: each domain gets its fair share (PROVE uniform distribution)
        per_domain = n_normal // len(servers)
        remainder = n_normal % len(servers)
        global_seed_offset = 0
        failed = 0
        # ── tqdm progress bar ──
        try:
            from tqdm import tqdm as _tqdm
            pbar = _tqdm(total=n_normal, desc="[generate_many]", unit="task",
                         dynamic_ncols=True, mininterval=1.0, miniters=1,
                         bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")
        except ImportError:
            pbar = None
        _gen_start = time.time()
        _last_log = 0  # last logged count for rate calc

        # ── Pre-compute task specs for parallel generation ──
        # Build per-domain task specifications with round-robin interleaving
        # so that every domain gets workers immediately, avoiding starvation
        # of later domains by a burst of same-domain tasks.
        # Each spec: (server_name, seed, difficulty).
        per_domain_specs: dict[str, list[tuple[str, int, str]]] = {}
        domain_quotas: dict[str, int] = {}
        domain_max_failures: dict[str, int] = {}
        max_specs_per_domain = 0
        for si, current_server in enumerate(servers):
            domain_target = per_domain + (1 if si < remainder else 0)
            domain_quotas[current_server] = domain_target
            # Failure budget: allow extra attempts proportional to target.
            # For large counts (≥100/domain) allow 50% extra; for small counts
            # use exact target.  At 50% pipeline yield this gives a 25% safety
            # margin on large runs while keeping small runs from exploding.
            extra = domain_target // 2 if domain_target >= 100 else 0
            domain_max_failures[current_server] = domain_target + extra
            specs = []
            for _ in range(domain_target + domain_max_failures[current_server]):
                task_seed = seed + global_seed_offset
                global_seed_offset += 1
                difficulty = self._pick_difficulty(task_seed, effective_mix)
                specs.append((current_server, task_seed, difficulty))
            per_domain_specs[current_server] = specs
            max_specs_per_domain = max(max_specs_per_domain, len(specs))

        # Round-robin interleave so workers pick up tasks from different domains
        task_specs: list[tuple[str, int, str]] = []
        for i in range(max_specs_per_domain):
            for s in servers:
                if i < len(per_domain_specs[s]):
                    task_specs.append(per_domain_specs[s][i])

        # ── Parallel generation with ThreadPoolExecutor ──
        configured_workers_raw = os.environ.get("LIVEMCP_GENERATION_MAX_WORKERS", "8")
        try:
            configured_workers = int(configured_workers_raw)
        except ValueError as exc:
            raise ValueError(
                "LIVEMCP_GENERATION_MAX_WORKERS must be an integer, got "
                f"{configured_workers_raw!r}"
            ) from exc
        if configured_workers < 1:
            raise ValueError(
                "LIVEMCP_GENERATION_MAX_WORKERS must be >= 1, got "
                f"{configured_workers}"
            )
        max_workers = min(configured_workers, max(1, len(servers)))
        domain_ok: dict[str, int] = {s: 0 for s in servers}
        domain_failed_count: dict[str, int] = {s: 0 for s in servers}
        submitted_futures = 0
        completed_futures = 0

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures: dict[Any, tuple[str, int]] = {}
            spec_index = 0

            def submit_until_full() -> None:
                nonlocal spec_index, submitted_futures
                while len(futures) < max_workers and spec_index < len(task_specs):
                    current_server, task_seed, difficulty = task_specs[spec_index]
                    spec_index += 1
                    if domain_ok[current_server] >= domain_quotas[current_server]:
                        continue
                    if domain_failed_count[current_server] >= domain_max_failures[current_server]:
                        continue
                    fut = executor.submit(
                        self._generate_task_with_postprocess,
                        current_server, task_seed, difficulty,
                        distractor_rate, missing_function_rate,
                    )
                    futures[fut] = (current_server, task_seed)
                    submitted_futures += 1

            submit_until_full()
            while futures:
                done, _ = wait(tuple(futures), return_when=FIRST_COMPLETED)
                for fut in done:
                    completed_futures += 1
                    current_server, task_seed = futures.pop(fut)
                    try:
                        task = fut.result()
                    except Exception as e:
                        failed += 1
                        domain_failed_count[current_server] += 1
                        logger.warning(
                            f"generate failed for {current_server} "
                            f"(seed={task_seed}, {domain_failed_count[current_server]}x): {e}"
                        )
                        continue
                    if task is None:
                        failed += 1
                        domain_failed_count[current_server] += 1
                        continue
                    if domain_ok[current_server] >= domain_quotas[current_server]:
                        continue
                    tasks.append(task)
                    domain_ok[current_server] += 1
                    if pbar:
                        pbar.update(1)
                        pbar.set_postfix_str(f"fail={failed}")
                    elapsed = time.time() - _gen_start
                    pct = len(tasks) * 100.0 / n_normal if n_normal > 0 else 0
                    if len(tasks) - _last_log >= 1:
                        _last_log = len(tasks)
                        logger.info(
                            f"[generate_many] {len(tasks)}/{n_normal} ({pct:.0f}%) "
                            f"| submitted={submitted_futures} completed={completed_futures} "
                            f"| {failed} fail | elapsed={elapsed:.0f}s "
                            f"| rate={len(tasks)/elapsed:.2f} task/s"
                        )
                submit_until_full()

        if pbar:
            pbar.close()

        # ── Warn about domains that fell short ──
        for s in servers:
            shortfall = domain_quotas[s] - domain_ok[s]
            if shortfall > 0:
                logger.warning(
                    f"{s}: fell short by {shortfall} tasks "
                    f"(got {domain_ok[s]}/{domain_quotas[s]}, "
                    f"{domain_failed_count[s]} failures)"
                )

        # ── irrelevance tasks (5%) ──
        irr = self._generate_irrelevant_tasks(n_irrelevant, seed + 9999, servers)
        tasks.extend(irr)

        # ── Dedup is deferred to _stratified_task_split (generate_data.py) ──
        # which applies Jaccard 0.70 after _filter_training_eligible_tasks.
        # Calling it here is redundant and wasteful O(n²).
        removed = 0
        before = len(tasks)

        # Surface low yield to the caller. With irrelevance_ratio<1, the
        # contractual target is `count` rows; falling far short usually means
        # the teacher LLM/MCP server pipeline is broken. Warn at <50% and
        # raise at 0% so callers do not silently write empty Parquet files.
        # Skip the guard entirely when the caller explicitly asked for 0
        # tasks (e.g. val-only or train-only generation).
        if count > 0 and not tasks:
            raise RuntimeError(
                f"generate_many produced 0 tasks (target {count}, "
                f"failures={failed}, dedup_removed={removed}). "
                f"Check teacher LLM connectivity and MCP servers."
            )
        if count > 0 and len(tasks) < max(1, count // 2):
            logger.error(
                f"generate_many SEVERE under-yield: got {len(tasks)}/{count} "
                f"({failed} failures, {removed} dedup_removed). "
                f"Inspect logs for repeated teacher errors."
            )

        # ── Data quality statistics (P0-3) ──
        n_paper_invalid = sum(
            1 for t in tasks
            if not t.metadata.get("paper_replay_valid", True)
        )
        n_outcome_invalid = sum(
            1 for t in tasks
            if not t.metadata.get("project_outcome_valid", True)
        )
        if n_paper_invalid or n_outcome_invalid:
            logger.warning(
                f"Data quality: {n_paper_invalid} tasks failed paper replay, "
                f"{n_outcome_invalid} tasks failed outcome criteria "
                f"(out of {len(tasks)} total). "
                f"Recommend filtering by paper_replay_valid for baseline; "
                f"outcome-invalid tasks should be isolated for analysis."
            )

        logger.info(
            f"LLM teacher: {len(tasks)} tasks (target {count}, {failed} failures, "
            f"submitted={submitted_futures}, completed={completed_futures}, "
            f"{removed} dedup removed)"
        )
        return tasks

    def _generate_task_with_postprocess(
        self, server_name: str, seed: int, difficulty: str,
        distractor_rate: float, missing_function_rate: float,
    ) -> LiveTask | None:
        """Thread-safe single-task generation.

        Samples a robustness plan from seed, then passes it to generate_one
        so all perturbations are applied BEFORE Teacher processing and
        Replay validation (PROVE §3.2 Figure 2).

        Returns None if generate_one raises, so the caller can count failures
        and retry with a new seed.
        """
        all_tools_pool = self.manager.registry.all_tools_with_servers()
        domain_tools = self.manager.registry.server_tools(server_name)

        plan = RobustnessPlan.sample(
            seed=seed,
            all_tools_pool=all_tools_pool,
            domain_tools=domain_tools,
            distractor_rate=distractor_rate,
            strip_enums_rate=0.30,
            missing_function_rate=missing_function_rate,
        )
        return self.generate_one(
            server_name, seed=seed, difficulty=difficulty,
            robustness_plan=plan,
        )

    def _to_live_task(self, server_name: str, query: str, session_id: str, seed: int,
                      all_tools: list[dict], oracle_program, required_tools: list[str],
                      difficulty: str, task_id: str,
                      conversation_queries: list[str] | None = None,
                      oracle_calls_per_round: list[list] | None = None,
                      execution_history_per_round: list[list] | None = None,
                      sampling_context: dict[str, Any] | None = None) -> LiveTask:
        if required_tools:
            # Show all domain tools — model must figure out which to use.
            # Don't leak required_tools by only showing those.
            all_domain_tools = self.manager.registry.server_tools(server_name)
            visible_tools = all_domain_tools if all_domain_tools else [t for t in all_tools if t["name"] in required_tools]
        else:
            # Clarification-only tasks (missing difficulty): expose domain tools
            # only — showing all cross-domain tools bloats the prompt and adds
            # irrelevant noise. The agent only needs to see what it CAN use to
            # determine what parameter is missing.
            visible_tools = self.manager.registry.server_tools(server_name)
        return LiveTask(
            task_id=task_id, source="live_mcp_task_planner",
            suite_name=self.suite_config.suite_name, user_prompt=query,
            session_id=session_id, session_seed=seed, target_servers=[server_name],
            visible_tools=visible_tools, required_tools=list(required_tools),
            expected_outcome={"success_criteria": oracle_program.success_criteria},
            success_criteria=list(oracle_program.success_criteria),
            oracle_program=oracle_program, sampling_context=sampling_context or {},
            max_turns=int(self.suite_config.rollout.get("max_turns", 8)),
            difficulty=difficulty, task_type="task_planner",
            metadata={"generation_method": "task_planner"},
            conversation_queries=conversation_queries or [],
            oracle_calls_per_round=oracle_calls_per_round or [],
            execution_history_per_round=execution_history_per_round or [],
        )

    def _generate_irrelevant_tasks(
        self,
        n: int,
        seed: int,
        allowed_servers: list[str] | None = None,
    ) -> list[LiveTask]:
        """Generate tasks whose query is unrelated to any available tool.

        The expected model behavior is to ``report_error`` (cannot be done).
        Goes through unified Replay + Provenance pipeline (PROVE §3.2).
        """
        if n <= 0:
            return []
        from src.live_mcp.task_planner import (
            TaskPlanner, replay_validate, provenance_check,
        )

        rng = random.Random(seed)
        tasks: list[LiveTask] = []

        servers = allowed_servers or self.manager.server_names
        if not servers:
            raise ValueError("irrelevant task generation requires at least one server")

        for i in range(n):
            server_name = rng.choice(servers)
            task_id = f"{server_name}_irrelevant_{seed}_{i}"

            # Ask teacher for an impossible query using a modified prompt
            query = self._generate_irrelevant_query(server_name, seed + i)
            if not query:
                logger.warning(
                    f"Irrelevance Teacher query generation failed for {task_id}; "
                    "rejecting candidate instead of substituting a template"
                )
                continue

            session = self.manager.create_session(seed=seed + i)
            fsm = ConversationFSM()
            try:
                self.manager.discover_tools(session.session_id)
                server_tools = self.manager.registry.server_tools(server_name)
                teacher = TaskPlanner(self.client, server_name, seed=seed + i)
                fsm.transition(
                    FSMStateGroup.TURN,
                    "irrelevant_query_generated",
                    round_idx=0,
                )
                (
                    oracle_calls,
                    execution_history,
                    oracle_observations,
                    _required_tools,
                    attempt_calls,
                    attempt_observations,
                ) = self._run_turn_loop(
                    teacher=teacher,
                    current_query=query,
                    server_tools=server_tools,
                    server_name=server_name,
                    session_id=session.session_id,
                    difficulty="minimal",
                    round_idx=0,
                    turn_budget=int(self.suite_config.rollout.get("max_turns", 8)),
                    fsm=fsm,
                )
            except RuntimeError as exc:
                logger.warning(
                    f"Irrelevance Teacher FSM failed for {task_id}: {exc}"
                )
                continue
            finally:
                self.manager.close_session(session.session_id)

            real_calls = [
                call for call in oracle_calls if call.action == "tool_call"
            ]
            terminals = [
                call for call in oracle_calls
                if call.action in ("report_error", "ask_clarification")
            ]
            # The completed oracle must not claim a useful tool action for an
            # impossible request. Failed Teacher attempts remain in the replay
            # trace and are governed by PROVE's 30% error-rate gate; do not add
            # a stricter unpublished zero-attempt corpus filter.
            if real_calls or len(terminals) != 1:
                logger.warning(
                    f"Irrelevance Teacher FSM rejected {task_id}: "
                    f"attempt_calls={len(attempt_calls)}, "
                    f"oracle_tool_calls={len(real_calls)}, terminals={len(terminals)}"
                )
                continue

            # ── Replay + Provenance (PROVE §3.2 unified pipeline) ──
            # The Teacher emitted a zero-tool terminal, so replay/provenance are
            # still run through the same completed-conversation pipeline.
            _valid, _err_rate, _n_err, n_calls, _criteria_ok, _criteria_failed = (
                replay_validate(
                    oracle_calls=attempt_calls,
                    manager=self.manager,
                    executor=self.executor,
                    seed=seed + i,
                    domain=server_name,
                    success_criteria=[],
                )
            )
            _prov_ok, _prov_violations = provenance_check(
                oracle_calls=attempt_calls,
                user_query=query,
                aligned_observations=attempt_observations,
            )
            # P2: use real Replay/provenance results instead of hardcoded True.
            # Zero-call oracle always passes Replay; provenance is trivially OK.
            # Discard if Replay unexpectedly fails (shouldn't happen for zero calls).
            if not _valid or not _prov_ok:
                logger.warning(
                    f"Irrelevance task {task_id} failed validation "
                    f"(replay={_valid}, provenance={_prov_ok}, "
                    f"err_rate={_err_rate:.2f}) — skipping."
                )
                continue
            task = LiveTask(
                task_id=task_id,
                source="live_mcp_task_planner",
                suite_name=self.suite_config.suite_name,
                user_prompt=query,
                session_id="",
                session_seed=seed + i,
                target_servers=[server_name],
                visible_tools=self.manager.registry.server_tools(server_name),
                required_tools=[],
                expected_outcome={"abstain": True},
                success_criteria=[],
                oracle_program=OracleProgram(
                    task_id=task_id,
                    calls=oracle_calls,
                    success_criteria=[],
                ),
                sampling_context={},
                max_turns=int(self.suite_config.rollout.get("max_turns", 8)),
                difficulty="minimal",
                task_type="irrelevant",
                metadata={
                    "generation_method": "irrelevant_teacher_fsm",
                    "irrelevant": True,
                    "scenario_type": "no_tool_or_abstention",
                    # P2: use real Replay/provenance results (not hardcoded True)
                    "paper_replay_valid": _valid,
                    "project_outcome_valid": _criteria_ok,
                    "replay_error_rate": _err_rate,
                    "replay_num_errors": _n_err,
                    "replay_num_calls": n_calls,
                    "criteria_failed": _criteria_failed,
                    "provenance_valid": _prov_ok,
                    "robustness_applied_before_replay": True,
                    "teacher_attempt_count": len(attempt_calls),
                    "fsm_final_state": fsm.state.value,
                    "fsm_transitions": list(fsm.transitions),
                },
            )
            tasks.append(task)

        return tasks

    def _generate_irrelevant_query(self, server_name: str, seed: int) -> str | None:
        """Ask LLM teacher to generate a query unrelated to the server's tools."""
        from src.live_mcp.task_planner import DOMAIN_DESCRIPTIONS
        domain_desc = DOMAIN_DESCRIPTIONS.get(server_name, "")

        prompt = (
            f"You are generating training data for an AI agent.\n\n"
            f"The agent has tools for: {domain_desc}\n\n"
            f"Generate ONE user query that is COMPLETELY UNRELATED to these tools — "
            f"something the agent cannot possibly do with them. "
            f"The query should sound natural, like a real user request.\n\n"
            f"Examples:\n"
            f'- "What movies are playing this weekend?" (when tools are for banking/scheduling)\n'
            f'- "Can you recommend a good Italian restaurant?" (when tools are for file management)\n'
            f'- "Tell me a joke" (when tools are for shopping/email)\n\n'
            f"Output ONLY the query string, nothing else. Do NOT prefix, do NOT wrap in quotes."
        )
        try:
            raw = self.client.generate_chat(
                [{"role": "user", "content": prompt}],
                temperature=0.8,
            )
            return raw.strip().strip('"\'')
        except Exception as e:
            logger.warning(f"Irrelevant query generation failed for {server_name}: {e}")
            return None

    @staticmethod
    def _fallback_irrelevant_query(server_name: str, rng: random.Random) -> str:
        """Fallback templates when LLM teacher fails to generate."""
        templates = [
            "What's the weather like today?",
            "Tell me a fun fact about space.",
            "Can you recommend a good book to read?",
            "What's the latest news?",
            "How do I cook pasta?",
            "What's your favorite color?",
            "Can you solve this math problem: 42 * 17?",
            "Tell me a joke.",
            "What movies are playing near me?",
            "Can you translate 'hello' to French?",
            "What's the capital of Bhutan?",
            "How many calories in an apple?",
            "Explain quantum computing in simple terms.",
            "What are the rules of chess?",
            "How do I change a flat tire?",
            "What's the population of Tokyo?",
            "Can you summarize the plot of Hamlet?",
            "What causes the Northern Lights?",
            "How do plants make energy from sunlight?",
            "What's the difference between HTTP and HTTPS?",
            "Who painted the Mona Lisa?",
            "How far is the moon from Earth?",
            "What's the best way to learn a new language?",
            "Can you explain how blockchain works?",
        ]
        return rng.choice(templates)

    @staticmethod
    def _pick_difficulty(seed: int, difficulty_mix: dict[str, float]) -> str:
        if not difficulty_mix:
            return "complete"
        rng = random.Random(seed)
        threshold = rng.random()
        cumulative = 0.0
        for name, weight in sorted(difficulty_mix.items()):
            cumulative += weight
            if threshold <= cumulative:
                return name
        return next(iter(sorted(difficulty_mix)))

    @staticmethod
    def _tool_schema_hash(
        server_tools: list[dict], server_name: str | None = None,
    ) -> str:
        """Hash tool schemas plus dependency-classification semantics.

        PROVE caches the pairwise LLM graph against the tool schema. Handler
        implementation changes affect live feasibility/execution, not the
        cached classifier output.
        """
        schema_payload = []
        for tool in sorted(server_tools, key=lambda t: str(t.get("name", ""))):
            schema_payload.append({
                "name": tool.get("name"),
                "description": tool.get("description", ""),
                "input_schema": tool.get("input_schema", {}),
                "annotations": tool.get("annotations", {}),
            })
        payload: dict[str, Any] = {
            "schema": schema_payload,
            "dependency_semantics_version": TaskOrchestrator.DEPENDENCY_SEMANTICS_VERSION,
        }
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _graph_cache_path(server_name: str, schema_hash: str) -> Path:
        return Path("data/dependency_graphs") / f"{server_name}_{schema_hash}.json"

    @staticmethod
    @contextmanager
    def _graph_cache_file_lock(cache_path: Path):
        """Serialize one domain/schema cache build across Python processes."""
        lock_path = cache_path.with_name(f"{cache_path.name}.lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a+", encoding="utf-8") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def _valid_cached_graph(
        graph: Any,
        expected_tool_names: list[str] | None = None,
    ) -> bool:
        if not isinstance(graph, dict) or not graph:
            return False
        expected_tool_set: set[str] | None = None
        if expected_tool_names is not None:
            if len(expected_tool_names) != len(set(expected_tool_names)):
                return False
            expected_tool_set = set(expected_tool_names)
            if set(graph.keys()) != expected_tool_set:
                return False
        for edges in graph.values():
            if not isinstance(edges, dict):
                return False
            for relation in ("explicit", "implicit"):
                targets = edges.get(relation)
                if not isinstance(targets, list):
                    return False
                if len(targets) != len(set(targets)):
                    return False
                if expected_tool_set is not None and any(t not in expected_tool_set for t in targets):
                    return False
        return True

    @staticmethod
    def _normalize_cached_graph(
        graph: dict,
        expected_tool_names: list[str],
    ) -> dict:
        expected_tool_set = set(expected_tool_names)
        normalized: dict[str, dict[str, list[str]]] = {}
        for tool_name in expected_tool_names:
            edge_groups = graph.get(tool_name, {}) if isinstance(graph, dict) else {}
            explicit: list[str] = []
            implicit: list[str] = []
            for relation, target_list in (("explicit", explicit), ("implicit", implicit)):
                raw_targets = edge_groups.get(relation, []) if isinstance(edge_groups, dict) else []
                if not isinstance(raw_targets, list):
                    continue
                for target in raw_targets:
                    if (
                        isinstance(target, str)
                        and target in expected_tool_set
                        and target != tool_name
                        and target not in target_list
                    ):
                        target_list.append(target)
            explicit_set = set(explicit)
            normalized[tool_name] = {
                "explicit": explicit,
                "implicit": [target for target in implicit if target not in explicit_set],
            }
        return normalized

    def _maybe_load_cached_graph(
        self,
        server_name: str,
        schema_hash: str,
        server_tools: list[dict],
    ) -> dict | None:
        """Load the schema-hash dependency graph cache used by PROVE Step 1."""
        cache_path = self._graph_cache_path(server_name, schema_hash)
        if not cache_path.exists():
            return None
        try:
            payload = json.loads(cache_path.read_text())
            graph = payload.get("graph") if isinstance(payload, dict) else None
            expected_tool_names = sorted(t.get("name", "") for t in server_tools)
            cached_tool_names = payload.get("tool_names") if isinstance(payload, dict) else None
            expected_pair_count = len(expected_tool_names) * (len(expected_tool_names) - 1) // 2
            classification_complete = bool(
                isinstance(payload, dict)
                and payload.get("cache_version") == self.DEPENDENCY_CACHE_VERSION
                and payload.get("dependency_semantics_version")
                    == self.DEPENDENCY_SEMANTICS_VERSION
                and payload.get("classification_complete") is True
                and payload.get("expected_pair_count") == expected_pair_count
                and payload.get("classified_pair_count") == expected_pair_count
            )
            if isinstance(graph, dict) and cached_tool_names == expected_tool_names:
                graph = self._normalize_cached_graph(graph, expected_tool_names)
            if (
                isinstance(payload, dict)
                and payload.get("schema_hash") == schema_hash
                and payload.get("server_name") == server_name
                and cached_tool_names == expected_tool_names
                and classification_complete
                and self._valid_cached_graph(graph, expected_tool_names)
            ):
                logger.info(f"Loaded dependency graph cache: {cache_path}")
                return graph
            logger.warning(
                f"Ignoring legacy/incomplete dependency graph cache: {cache_path}; "
                f"a complete pairwise LLM classification is required"
            )
        except Exception as e:
            logger.warning(f"Failed to load dependency graph cache {cache_path}: {e}")
        return None

    def _save_cached_graph(
        self,
        server_name: str,
        schema_hash: str,
        server_tools: list[dict],
        graph: dict,
    ) -> None:
        """Persist PROVE's per-environment graph cache keyed by tool schema."""
        expected_tool_names = sorted(t.get("name", "") for t in server_tools)
        graph = self._normalize_cached_graph(graph, expected_tool_names)
        # Preserve the complete normalized pairwise LLM classification.
        if not self._valid_cached_graph(graph, expected_tool_names):
            logger.warning(f"Skipping invalid dependency graph cache for {server_name}")
            return
        cache_path = self._graph_cache_path(server_name, schema_hash)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "cache_version": self.DEPENDENCY_CACHE_VERSION,
            "dependency_semantics_version": self.DEPENDENCY_SEMANTICS_VERSION,
            "server_name": server_name,
            "schema_hash": schema_hash,
            "tool_names": expected_tool_names,
            "graph": graph,
            "tool_count": len(expected_tool_names),
            "expected_pair_count": len(expected_tool_names) * (len(expected_tool_names) - 1) // 2,
            "classified_pair_count": len(expected_tool_names) * (len(expected_tool_names) - 1) // 2,
            "classification_complete": True,
        }
        serialized = json.dumps(payload, indent=2, ensure_ascii=False)
        temp_fd, temp_name = tempfile.mkstemp(
            dir=cache_path.parent,
            prefix=f".{cache_path.name}.",
            suffix=".tmp",
        )
        temp_path = Path(temp_name)
        try:
            with os.fdopen(temp_fd, "w", encoding="utf-8") as temp_file:
                temp_file.write(serialized)
                temp_file.flush()
                os.fsync(temp_file.fileno())
            os.replace(temp_path, cache_path)
            directory_fd = os.open(cache_path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            if temp_path.exists():
                temp_path.unlink()
        logger.info(f"Saved dependency graph cache: {cache_path}")

    def _probe_dependency_graph(self, server_name: str) -> dict:
        """PROVE Step 1: auto-discover tool dependencies.

        Probe the live MCP server for tool schemas, load the schema-hash graph
        cache when available, and otherwise ask the teacher LLM to classify
        every unordered C(n,2) tool pair as explicit, implicit, or none.
        """
        session = self.manager.create_session(seed=0)
        try:
            self.manager.discover_tools(session.session_id)
            server_tools = self.manager.registry.server_tools(server_name)
        except Exception as e:
            logger.debug(f"_probe_dependency_graph: tool discovery failed for {server_name}: {e}")
            self.manager.close_session(session.session_id)
            return {}

        if len(server_tools) < 2:
            self.manager.close_session(session.session_id)
            return {}

        schema_hash = self._tool_schema_hash(server_tools, server_name)
        try:
            cached = self._maybe_load_cached_graph(server_name, schema_hash, server_tools)
            if cached is not None:
                return cached

            failure_key = (server_name, schema_hash)
            failures = getattr(self, "_dependency_graph_failures", None)
            if failures is None:
                failures = {}
                self._dependency_graph_failures = failures
            previous_failure = failures.get(failure_key)
            if previous_failure is not None:
                failed_at, failure_message = previous_failure
                retry_after = self.DEPENDENCY_FAILURE_RETRY_SECONDS - (
                    time.monotonic() - failed_at
                )
                if retry_after > 0:
                    raise RuntimeError(
                        f"Recent dependency classification failure for {server_name}; "
                        f"retry suppressed for {retry_after:.1f}s: {failure_message}"
                    )
                failures.pop(failure_key, None)

            cache_path = self._graph_cache_path(server_name, schema_hash)
            with self._graph_cache_file_lock(cache_path):
                # Another process may have completed the graph while this
                # process waited for the lock. Never classify before reloading.
                cached = self._maybe_load_cached_graph(
                    server_name, schema_hash, server_tools,
                )
                if cached is not None:
                    return cached

                graph = self._classify_edges_llm(server_tools, server_name)
                if graph is None:
                    failure_message = (
                        f"Pairwise dependency classification incomplete for {server_name}; "
                        f"refusing incomplete graph"
                    )
                    failures[failure_key] = (time.monotonic(), failure_message)
                    raise RuntimeError(failure_message)

                self._save_cached_graph(
                    server_name, schema_hash, server_tools, graph,
                )
                return graph
        finally:
            self.manager.close_session(session.session_id)

    def _classify_edges_llm(
        self,
        server_tools: list[dict],
        server_name: str,
    ) -> dict | None:
        """PROVE §3.2 Step 1: LLM-based pairwise tool relationship classification.

        Sends each unordered C(n,2) tool pair to the LLM once. The LLM selects
        source and target when the relationship is directed, then classifies it
        as explicit, implicit, or none.

        Returns a graph dict with the same structure as _probe_dependency_graph,
        or None if LLM classification fails.
        """
        tool_names = [t["name"] for t in server_tools]
        n = len(tool_names)
        if n < 2:
            return None

        # Build compact tool descriptions for the LLM
        tool_descs: list[str] = []
        for t in server_tools:
            name = t["name"]
            desc = t.get("description", "")
            props = t.get("input_schema", {}).get("properties", {})
            required = t.get("input_schema", {}).get("required", [])
            param_lines = []
            for pk, pv in props.items():
                req_mark = "*" if pk in required else ""
                ptype = pv.get("type", "?")
                pdesc = pv.get("description", "")
                param_lines.append(f"    {pk}{req_mark} ({ptype}){': ' + pdesc if pdesc else ''}")
            params_str = "\n".join(param_lines) if param_lines else "    (none)"
            tool_descs.append(
                f"Tool: {name}\n"
                f"  Description: {desc}\n"
                f"  Parameters:\n{params_str}"
            )

        tool_desc_by_name = {
            str(tool.get("name") or ""): desc
            for tool, desc in zip(server_tools, tool_descs)
        }

        pairs = [
            (tool_names[i], tool_names[j])
            for i in range(len(tool_names))
            for j in range(i + 1, len(tool_names))
        ]
        logger.debug(
            f"_classify_edges_llm: {server_name} classifying {len(pairs)} "
            f"unordered tool pairs"
        )

        # Batch pairs to fit LLM context and bound single-request decode time.
        # This remains PROVE-style pairwise LLM classification over all C(n,2)
        # pairs; each request carries only the schemas for its batch.
        BATCH_SIZE = self.DEPENDENCY_PAIR_BATCH_SIZE
        all_classifications: dict[str, str] = {}  # "A → B" → "explicit"|"implicit"
        classified_pairs: set = set()

        all_pair_keys: set[tuple[str, str]] = set(pairs)
        expected_pair_count = len(all_pair_keys)
        BATCH_RETRIES = 2
        tool_order = {name: index for index, name in enumerate(tool_names)}

        def _canonical_pair(a_name: str, b_name: str) -> tuple[str, str]:
            return (
                (a_name, b_name)
                if tool_order[a_name] < tool_order[b_name]
                else (b_name, a_name)
            )

        def _consume_classifications(
            data: Any,
            valid_pairs: set[tuple[str, str]],
        ) -> None:
            if not isinstance(data, dict):
                return
            entries = data.get("classifications", [])
            if not isinstance(entries, list):
                return
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                relation = str(entry.get("relation", "none")).strip().lower()
                if relation not in ("explicit", "implicit", "none"):
                    continue
                source = str(entry.get("source") or "").strip()
                target = str(entry.get("target") or "").strip()
                pair_text = str(entry.get("pair") or "")
                pair_parts = [
                    part.strip()
                    for part in re.split(r"\s*→\s*", pair_text, maxsplit=1)
                ]
                pair_members = (
                    pair_parts
                    if len(pair_parts) == 2
                    and all(part in tool_desc_by_name for part in pair_parts)
                    and pair_parts[0] != pair_parts[1]
                    else []
                )
                if not pair_members and (
                    source in tool_desc_by_name
                    and target in tool_desc_by_name
                    and source != target
                ):
                    pair_members = [source, target]
                if not pair_members:
                    continue
                pair_key = _canonical_pair(pair_members[0], pair_members[1])
                if pair_key not in valid_pairs or pair_key in classified_pairs:
                    continue

                if relation == "none":
                    classified_pairs.add(pair_key)
                    continue

                if source not in tool_desc_by_name or target not in tool_desc_by_name:
                    continue
                if source == target or _canonical_pair(source, target) != pair_key:
                    continue
                classified_pairs.add(pair_key)
                all_classifications[f"{source} → {target}"] = relation

        for batch_start in range(0, len(pairs), BATCH_SIZE):
            batch_pairs = pairs[batch_start:batch_start + BATCH_SIZE]
            valid_batch_pairs = set(batch_pairs)

            system = (
                "You are analyzing tool dependencies for an MCP server. "
                "For each unordered tool pair {A, B}, decide whether a dependency "
                "exists and, if so, choose its source and target:\n"
                '- "explicit": source produces output that is a REQUIRED INPUT of target '
                "(e.g., source returns an entity ID that target needs as a parameter).\n"
                '- "implicit": source must execute BEFORE target to establish state, '
                "but source's output is not a direct input to target.\n"
                '- "none": neither listed direction is a required dependency.\n\n'
                "Classification rules:\n"
                "- If A creates/returns something that B's required parameters "
                "reference, mark explicit.\n"
                "- Mark implicit only when source establishes live server state that target "
                "cannot succeed without, such as create_draft → send_draft, "
                "add_to_cart → checkout, or schedule_transfer → cancel_transfer.\n"
                "- If B can succeed with the same arguments and pre-existing server state "
                "without running A first, classify A → B as none. Mere topical relevance, "
                "a useful recommendation, or a common workflow order is none.\n"
                "- A read-only A is not an implicit dependency merely because its result is "
                "helpful. It is explicit only when B requires a value produced by A.\n"
                "- Prefer explicit over implicit when both could apply.\n"
                "- Only mark implicit if there is a genuine required state dependency."
            )
            batch_complete = False
            for batch_attempt in range(BATCH_RETRIES + 1):
                # Ask only for pairs that are still missing. Re-sending the full
                # batch makes deterministic teachers repeat the same prefix and
                # never fill truncated/omitted tail entries.
                pending_pairs = [
                    pair for pair in batch_pairs
                    if pair not in classified_pairs
                ]
                pending_tool_names = sorted({
                    name for pair in pending_pairs for name in pair
                })
                pending_tools_text = "\n\n".join(
                    tool_desc_by_name[name]
                    for name in pending_tool_names
                    if name in tool_desc_by_name
                )
                pending_pairs_text = "\n".join(
                    f"{i + 1}. {a_name} → {b_name}"
                    for i, (a_name, b_name) in enumerate(pending_pairs)
                )
                user = (
                    f"## Server: {server_name}\n\n"
                    f"## Tools\n{pending_tools_text}\n\n"
                    f"## Pairs to Classify\n{pending_pairs_text}\n\n"
                    f"Classify every listed pair exactly once. The displayed A → B "
                    f"only identifies the pair; for explicit/implicit you may return "
                    f"either A as source and B as target or the reverse direction. "
                    f"Do not omit any pair.\n\n"
                    f"## Output Format\n"
                    f'{{"classifications": [\n'
                    f'  {{"pair": "tool_a → tool_b", "source": "tool_a", "target": "tool_b", "relation": "explicit"}},\n'
                    f'  {{"pair": "tool_c → tool_d", "source": "tool_c", "target": "tool_d", "relation": "implicit"}},\n'
                    f'  {{"pair": "tool_e → tool_f", "source": "tool_e", "target": "tool_f", "relation": "none"}}\n'
                    f']}}\n\n'
                    f"Output ONLY the JSON, nothing else:"
                )
                try:
                    raw = self.client.generate_chat(
                        [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
                        temperature=0.1 + 0.05 * batch_attempt,
                        max_tokens=self.DEPENDENCY_CLASSIFICATION_MAX_TOKENS,
                    )
                    _consume_classifications(_extract_json(raw), valid_batch_pairs)
                except Exception as e:
                    logger.debug(
                        f"_classify_edges_llm batch {batch_start // BATCH_SIZE + 1} "
                        f"attempt {batch_attempt + 1}/{BATCH_RETRIES + 1} "
                        f"failed for {server_name}: {e}"
                    )
                if valid_batch_pairs <= classified_pairs:
                    batch_complete = True
                    break
                missing_count = len(valid_batch_pairs - classified_pairs)
                logger.debug(
                    f"_classify_edges_llm batch {batch_start // BATCH_SIZE + 1} "
                    f"missing {missing_count}/{len(valid_batch_pairs)} pair(s) after "
                    f"attempt {batch_attempt + 1}"
                )
            if not batch_complete:
                logger.warning(
                    f"_classify_edges_llm batch {batch_start // BATCH_SIZE + 1} "
                    f"incomplete after {BATCH_RETRIES + 1} attempts for {server_name}"
                )
                return None

        # P1-1: completeness gate — refuse to return an incomplete graph.
        # Expected: C(n,2) unordered pairs, each classified exactly once.
        # classification (explicit, implicit, or none).
        if len(classified_pairs) != expected_pair_count:
            logger.warning(
                f"_classify_edges_llm completeness check FAILED for {server_name}: "
                f"classified {len(classified_pairs)}/{expected_pair_count} pairs. "
                f"Discarding partial graph — it will NOT be cached."
            )
            return None

        # Build graph from classifications
        graph: dict[str, dict] = {}
        for t in server_tools:
            graph[t["name"]] = {"explicit": [], "implicit": []}

        for pair_key, relation in all_classifications.items():
            parts = pair_key.split(" → ")
            if len(parts) == 2:
                a_name, b_name = parts
                if a_name in graph and b_name in graph:
                    if relation == "explicit":
                        graph[a_name]["explicit"].append(b_name)
                    elif relation == "implicit":
                        graph[a_name]["implicit"].append(b_name)

        # Return the complete pairwise LLM classification. Build/load preserves
        # these LLM edges; handler facts are applied only when candidate chains
        # are checked against live state and execution.
        return graph

    def _extract_dependency_chains(self, server_name: str) -> list[list[str]]:
        """PROVE §6 step 2: extract length-2 to length-5 tool chains from dependency graph.

        Depth-first search through the dependency graph to find all valid tool chains.
        """
        graph = self._domain_graphs.get(server_name) or self._probe_dependency_graph(server_name)
        self._domain_graphs[server_name] = graph
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

        # Exact dedup only. PROVE extracts length-2 to length-5 paths from the
        # discovered graph; do not add local caps or relation-preference ranking
        # that would bias the sampled chain distribution.
        deduped: list[list[str]] = []
        seen: set[tuple] = set()
        for c in chains:
            key = tuple(c)
            if key not in seen and _chain_respects_state_preconditions(server_name, c):
                seen.add(key)
                deduped.append(c)

        logger.debug(
            f"_extract_dependency_chains: {server_name} → {len(deduped)} "
            f"chains discovered"
        )
        return deduped

    def _get_chains(self, server_name: str) -> list[list[str]]:
        """Return cached dependency chains for *server_name*, extracting if needed."""
        with self._dependency_graph_lock:
            if server_name not in self._domain_chains:
                self._domain_chains[server_name] = self._extract_dependency_chains(server_name)
            return self._domain_chains[server_name]

    def _probe_live_sampling_context(
        self,
        session_id: str,
        server_name: str,
        server_tools: list[dict],
    ) -> dict[str, Any]:
        """PROVE §3.2 Step 2: enumerate real entities through read-only tools.

        This is intentionally separate from ``debug/get_state``.  The sampler
        should only expose entities that a policy could discover through the
        live MCP interface.  Tools with required entity IDs are skipped here;
        they become usable after a list/search/get_cart style probe has surfaced
        concrete IDs.
        """
        from src.live_mcp.types import ToolCall

        entity_ids: list[dict[str, str]] = []
        entity_summaries: list[str] = []
        entity_records: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        probe_results: list[dict[str, Any]] = []

        def add_entity(eid: str, etype: str, edata: dict[str, Any] | None = None) -> None:
            if not eid or not etype:
                return
            key = (etype, eid)
            if key in seen:
                return
            seen.add(key)
            entity_ids.append({"id": eid, "type": etype})
            entity_summaries.append(_format_entity_summary(eid, etype, edata))
            entity_records.append({
                "id": eid,
                "type": etype,
                "data": dict(edata) if isinstance(edata, dict) else {},
            })

        for tool in sorted(server_tools, key=lambda t: str(t.get("name", ""))):
            tool_name = str(tool.get("name") or "")
            if not tool_name or not _is_readonly_discovery_tool(tool):
                continue
            args = _readonly_probe_args(tool, server_name)
            if args is None:
                continue
            result = self.executor.execute(
                session_id,
                ToolCall(tool_name, args, call_id=f"live_probe_{tool_name}"),
                domain=server_name,
            )
            probe_results.append({
                "tool": tool_name,
                "arguments": args,
                "success": bool(result.success),
                "state_changed": bool(result.state_changed),
                "error_type": result.error_type,
            })
            if not result.success or result.state_changed:
                continue
            _extract_probe_entities(
                result.observation,
                add_entity,
                server_name=server_name,
                tool_name=tool_name,
            )

        # ── P0-1 Fix: two-stage enrichment for entities needing sub-probes ──
        # Some quality predicates (e.g. food_delivery restaurant → menu) depend
        # on data not returned by top-level discovery tools.  After the primary
        # probe pass, enrich entity records with additional readonly queries.
        if server_name == "food_delivery":
            _enrich_restaurant_menus(
                self.executor, session_id, server_name, entity_records,
            )

        # ── P0-1: Filter entities by domain-specific data-quality predicates ──
        filter_reasons: dict[str, list[str]] = {}
        qualified_ids: list[dict[str, str]] = []
        qualified_records: list[dict[str, Any]] = []
        for record in entity_records:
            eid = record["id"]
            etype = record["type"]
            ok, reason_str = _entity_record_qualifies(
                server_name, etype, record.get("data", {}),
            )
            if ok:
                qualified_ids.append({"id": eid, "type": etype})
                qualified_records.append(record)
            else:
                filter_reasons.setdefault(reason_str or "unqualified", []).append(
                    f"{etype}/{eid}"
                )

        probed_count = len(entity_ids)
        qualified_count = len(qualified_ids)
        if probed_count > 0 and qualified_count < probed_count:
            logger.info(
                f"_probe_live_sampling_context [{server_name}]: "
                f"probed={probed_count} qualified={qualified_count} "
                f"({qualified_count * 100 // probed_count}%) "
                f"filter_reasons={dict(filter_reasons)}"
            )

        # Build a summary list aligned with qualified_ids (not raw entity_ids).
        # _live_context_to_prompt_state iterates qualified_entity_ids and needs
        # summaries at the same index — using raw entity_summaries would give
        # wrong summaries after filtering.
        entity_key_to_summary: dict[tuple[str, str], str] = {
            (
                str(entity_ids[i].get("type", "")),
                str(entity_ids[i].get("id", "")),
            ): entity_summaries[i]
            for i in range(min(len(entity_ids), len(entity_summaries)))
        }
        qualified_summaries: list[str] = [
            entity_key_to_summary.get(
                (str(q.get("type", "")), str(q.get("id", ""))), "",
            )
            for q in qualified_ids
        ]

        return {
            "source": "live_readonly_probe",
            "entity_ids": entity_ids,
            "entity_summaries": entity_summaries,
            "entity_records": entity_records,
            "entity_types": sorted({item["type"] for item in entity_ids}),
            "probe_results": probe_results,
            # P0-1: qualified entities for chain feasibility and context
            "qualified_entity_ids": qualified_ids,
            "qualified_entity_records": qualified_records,
            "qualified_entity_summaries": qualified_summaries,
            "qualified_entity_types": sorted({item["type"] for item in qualified_ids}),
            "entity_filter_reasons": {
                k: len(v) for k, v in filter_reasons.items()
            },
            "probed_entity_count": probed_count,
            "qualified_entity_count": qualified_count,
        }

    def _filter_feasible_chains(
        self,
        chains: list[list[str]],
        server_name: str,
        live_context: dict[str, Any],
    ) -> list[list[str]]:
        """PROVE §3.2 Step 2 guard: keep only chains executable in live state.

        The chain seed is not just a hint for prompt text. It determines which
        IDs the teacher can ground tool arguments on, so an infeasible chain must
        be removed before query/action generation. Returning the original chains
        when all checks fail reintroduces hallucination pressure, so this method
        deliberately returns [] in that case and lets generation proceed without
        a forced dependency chain.
        """
        if not chains:
            return chains

        # P0-1: use qualified entities for feasibility; fall back to raw if
        # qualified entity lists are absent (legacy cached contexts).
        qualified_entity_ids = live_context.get("qualified_entity_ids")
        chain_context = live_context
        if qualified_entity_ids is not None:
            chain_context = {
                **live_context,
                "entity_ids": qualified_entity_ids,
            }

        feasible: list[list[str]] = []
        drop_reasons: dict[str, int] = {}
        drop_examples: dict[str, list[str]] = {}
        for chain in chains:
            ok, reason = _chain_is_feasible(
                chain, server_name, chain_context
            )
            if ok:
                feasible.append(chain)
            else:
                reason_key = reason or "unknown"
                drop_reasons[reason_key] = drop_reasons.get(reason_key, 0) + 1
                drop_examples.setdefault(reason_key, chain)

        # P0-1: log before/after counts with qualification info
        probed_count = live_context.get("probed_entity_count", 0)
        qualified_count = live_context.get("qualified_entity_count", probed_count)
        logger.info(
            f"_filter_feasible_chains [{server_name}]: "
            f"probed_entities={probed_count} qualified_entities={qualified_count} "
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
            if server_name not in self._domain_graphs:
                graph = self._probe_dependency_graph(server_name)
                self._domain_graphs[server_name] = graph
            return _format_graph_hints(self._domain_graphs[server_name])

    def _get_live_sampling_context(
        self,
        session_id: str,
        server_name: str,
        server_tools: list[dict],
        force_refresh: bool = False,
    ) -> dict[str, Any]:
        """PROVE §3.2 Step 2: return sampling context for the current session.

        Each session has its own seed-determined initial state, so the context
        MUST be probed per-session.  We cache within the same session_id to
        avoid redundant probes during multi-round continuations, but invalidate
        whenever the session_id changes.
        """
        entry = self._sampling_context_cache.get(server_name)

        # Invalidate cache if session changed (different seed → different state)
        if entry is not None and entry.get("session_id") != session_id:
            entry = None

        if entry is None:
            entry = {"context": {}, "call_count": 0, "session_id": session_id}
            self._sampling_context_cache[server_name] = entry

        if force_refresh or entry["call_count"] % self.SAMPLING_CONTEXT_REFRESH_K == 0:
            # Refresh: probe the live server for current entity state.
            fresh = self._probe_live_sampling_context(
                session_id=session_id,
                server_name=server_name,
                server_tools=server_tools,
            )
            entry["context"] = fresh
            entry["session_id"] = session_id
            logger.debug(
                f"_get_live_sampling_context: {server_name} refreshed "
                f"(call #{entry['call_count']}, "
                f"{len(fresh.get('entity_ids', []))} entities)"
            )
        else:
            logger.debug(
                f"_get_live_sampling_context: {server_name} using cached context "
                f"(call #{entry['call_count']}, "
                f"next refresh at #{(entry['call_count'] // self.SAMPLING_CONTEXT_REFRESH_K + 1) * self.SAMPLING_CONTEXT_REFRESH_K})"
            )

        entry["call_count"] += 1
        return entry["context"]


def _stable_state_hash(state: dict[str, Any]) -> str:
    import hashlib

    raw = json.dumps(state, sort_keys=True, ensure_ascii=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


def _identity_policy_for_domain(domain: str) -> str:
    return {
        "calendar": "preserve",
        "banking": "preserve",
        "payments": "preserve",
        "crm": "preserve",
        "issue_tracker": "preserve",
        "email": "append_only",
        "team_chat": "append_only",
        "shopping": "create_new",
        "food_delivery": "create_new",
        "filesystem": "domain_defined",
    }.get(domain, "domain_defined")


def _oracle_target_ids(calls: list[OracleCall]) -> list[str]:
    ids: set[str] = set()
    for call in calls:
        for key, value in (call.arguments or {}).items():
            key_lower = key.lower()
            if not (
                key_lower.endswith("_id")
                or key_lower in (
                    "path", "source", "destination", "from_account", "to_account"
                )
            ):
                continue
            if isinstance(value, str) and value:
                ids.add(value)
    return sorted(ids)


def _oracle_deleted_target_ids(calls: list[OracleCall]) -> list[str]:
    delete_prefixes = ("delete_", "remove_", "cancel_", "clear_", "archive_", "rm")
    ids: set[str] = set()
    for call in calls:
        if not call.tool_name.lower().startswith(delete_prefixes):
            continue
        ids.update(_oracle_target_ids([call]))
    return sorted(ids)


def _protected_fields_by_resource(
    initial_state: dict[str, Any],
    target_ids: list[str],
    success_criteria: list[dict[str, Any]],
) -> dict[str, list[str]]:
    intended: dict[str, set[str]] = {resource_id: set() for resource_id in target_ids}
    for criterion in success_criteria:
        path = str(criterion.get("path", ""))
        path_parts = criterion.get("path_parts")
        parts = path_parts if isinstance(path_parts, list) else path.split(".")
        for resource_id in target_ids:
            if resource_id in parts and len(parts) > parts.index(resource_id) + 1:
                intended[resource_id].add(parts[-1])

    protected: dict[str, list[str]] = {}
    for container in initial_state.values():
        if not isinstance(container, dict):
            continue
        for resource_id in target_ids:
            entity = container.get(resource_id)
            if not isinstance(entity, dict):
                continue
            fields = {
                field for field in entity
                if not field.endswith("_id") and field not in intended[resource_id]
            }
            if fields:
                protected[resource_id] = sorted(fields)
    return protected


_UNSAFE_SHORTCUT_TOOLS: dict[str, set[str]] = {
    "calendar": {"update_event", "delete_event", "create_event"},
    "banking": {"transfer", "wire_transfer", "bill_pay"},
    "filesystem": {"mv", "cp", "rm", "chmod", "chown"},
    "payments": {"pay_invoice", "refund_invoice", "cancel_payment"},
    "shopping": {"checkout", "return_order", "clear_cart"},
    "email": {"send_email", "reply_email", "forward_email",
              "mark_read", "mark_unread", "archive_email",
              "add_label", "remove_label", "move_to_thread"},
    "team_chat": {"send_message", "send_dm", "archive_channel"},
    "crm": {"convert_lead", "update_deal", "update_lead"},
    "issue_tracker": {"transition_issue", "update_issue", "assign_issue"},
    "food_delivery": {"create_order", "cancel_order", "update_order_status"},
}


_READ_INTENT_MARKERS = {
    "what", "what's", "whats", "which", "who", "when", "where", "show",
    "list", "check", "view", "get", "find", "search", "look up", "lookup",
    "confirm", "status", "details", "schedule", "agenda", "calendar",
    "history", "balance", "track", "report", "summary",
}


_WRITE_INTENT_BY_PREFIX: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (
        ("create_", "mkdir", "touch"),
        ("create", "make", "new", "book", "draft", "register", "open"),
    ),
    (
        ("update_", "edit_", "change_", "set_", "mark_", "transition_", "assign_",
         "complete_", "time_track"),
        ("update", "edit", "change", "modify", "set", "mark", "move", "reschedule",
         "rename", "assign", "complete", "transition", "log"),
    ),
    (
        ("delete_", "remove_", "rm", "cancel_", "archive_", "clear_"),
        ("delete", "remove", "cancel", "archive", "clear", "drop"),
    ),
    (
        ("add_", "apply_", "react_", "rate_"),
        ("add", "apply", "attach", "invite", "react", "rate", "review", "label"),
    ),
    (
        ("send_", "reply_", "forward_", "pay_", "refund_", "transfer", "wire_",
         "checkout", "return_", "reorder", "deposit", "withdraw", "freeze_",
         "unfreeze_", "convert_"),
        ("send", "reply", "forward", "pay", "refund", "transfer", "wire", "checkout",
         "return", "reorder", "deposit", "withdraw", "freeze", "unfreeze", "convert"),
    ),
)


def _classify_scenario(
    server_name: str,
    oracle_calls: list[OracleCall],
    execution_history: list[dict[str, Any]],
    terminal_action: str,
    seed: int,
) -> str:
    """Classify scenario based on actual trajectory behavior, not random assignment.

    OVAL-MCP §11.5: scenario_type must reflect the real trajectory, not a
    statistical label injection.  Random assignment (old: 30% unsafe_temptation,
    45% missing_dependency) produces mislabeled training data that pollutes the
    reward signal.

    Detection rules (ordered, first match wins):
      1. clarification_required: terminal is ask_clarification, zero real calls
      2. tool_error_recovery:   any non-reject execution failure in history
      3. unsafe_temptation:     oracle trace has delete+create duplicate pattern
                                on the same entity type (true unsafe shortcut)
      4. missing_dependency:    2+ step oracle where a later tool needs an entity
                                that should have been produced by an earlier step
                                but its predecessor is absent
      5. normal_safe_success:   everything else
    """
    real_calls_only = [c for c in oracle_calls
                       if getattr(c, "action", "tool_call") == "tool_call"]

    if terminal_action == "ask_clarification":
        return "clarification_required"

    if any(not step.get("success", True) and step.get("tool_name") != "__reject__"
           for step in execution_history):
        return "tool_error_recovery"

    if not real_calls_only:
        return "normal_safe_success"

    # ── Trajectory-based unsafe_temptation detection ──
    if _detect_duplicate_side_effect(real_calls_only):
        return "unsafe_temptation"

    # ── Trajectory-based missing_dependency detection ──
    # _detect_missing_dependency internally exempts creators (create_*/mkdir/touch)
    # and self-contained tools (send_email, deposit, etc.).  Single-step traces
    # like [update_event] with no preceding read ARE missing a dependency.
    if _detect_missing_dependency(real_calls_only, server_name):
        return "missing_dependency"

    return "normal_safe_success"


def _detect_duplicate_side_effect(oracle_calls: list[OracleCall]) -> bool:
    """Detect delete+create duplicate pattern (PROVE unsafe shortcut).

    Pattern: the oracle trace deletes/removes/cancels a *specific* entity and
    then creates/adds a similar entity — functionally a "recreate" shortcut
    that destroys identity.  This is the canonical unsafe_temptation in
    OVAL-MCP §11.5.

    Does NOT flag:
      - rm (filesystem: path-based, no entity type)
      - clear_* (bulk operation, legitimate state reset)
      - remove_from_cart + add_to_cart (different items, legitimate)
    """
    # Entity-specific delete prefixes only; rm and clear_ are excluded because
    # rm targets a path (entity="" matches everything), and clear_* is a bulk
    # reset that's always legitimate when followed by new items.
    delete_prefixes = ("delete_", "remove_", "cancel_", "archive_")
    create_prefixes = ("create_", "add_", "mkdir", "touch", "send_")

    deleted_entity_types: set[str] = set()
    for call in oracle_calls:
        name = call.tool_name.lower()
        for prefix in delete_prefixes:
            if name.startswith(prefix):
                entity = name[len(prefix):].lstrip("_")
                if entity:
                    deleted_entity_types.add(entity)
                break

    if not deleted_entity_types:
        return False

    found_delete = False
    for call in oracle_calls:
        name = call.tool_name.lower()
        is_delete = any(name.startswith(p) for p in delete_prefixes)
        if is_delete:
            found_delete = True
            continue
        if not found_delete:
            continue
        is_create = any(name.startswith(p) for p in create_prefixes)
        if not is_create:
            continue
        # Exact entity match: delete_event → create_event (both entity="event").
        # "add_to_cart" after "remove_from_cart" would match via entity="from_cart"
        # vs entity="to_cart" → NOT equal.  Only flag when the tool entity
        # *suffix* (not prefix substring) matches.
        create_entity = _tool_entity(name)
        if create_entity in deleted_entity_types:
            return True

    return False


def _detect_missing_dependency(
    oracle_calls: list[OracleCall],
    server_name: str,
) -> bool:
    """Detect if the oracle trace skips a dependency step.

    A dependency is missing when a write/mutate tool is called on an entity
    without a preceding read OR create tool that resolves/produces that
    entity's identity.  Create tools themselves are exempt — they produce
    new entities and don't need preceding reads.

    Also exempted: standalone creators (mkdir, touch) and tools that operate
    on their own domain (apply_loan: self-contained account operation,
    send_email: compose new message, etc.).
    """
    read_prefixes = (
        "list_", "search_", "get_", "find_", "lookup_", "check_", "verify_",
        "view_", "browse_", "ls", "cat", "pwd", "stat", "head", "tail",
        "find", "grep", "tree", "du", "df",
    )
    write_prefixes = tuple(
        prefix
        for prefixes, _markers in _WRITE_INTENT_BY_PREFIX
        for prefix in prefixes
    ) + (
        "comment_", "time_track", "schedule_", "respond_", "archive_", "clear_", "return_", "reorder",
        "freeze_", "unfreeze_", "dispute_", "react_", "contact_", "move_to_thread",
        "bill_pay", "mv", "cp", "rm", "chmod", "chown",
    )

    for i, call in enumerate(oracle_calls):
        # Step 0 may use entity IDs from task context (user query / initial
        # state snapshot).  Only skip the dependency check when the arguments
        # actually contain entity IDs — empty-arg single-step write tools are
        # genuine missing dependencies.
        if i == 0 and len(oracle_calls) > 1 and _oracle_target_ids([call]):
            continue

        tool_name = call.tool_name.lower()
        if not any(tool_name.startswith(prefix) for prefix in write_prefixes):
            continue

        requirements = _tool_existing_entity_requirements(tool_name, server_name)
        if not requirements:
            continue

        available_entities: set[str] = set()
        for prev in oracle_calls[:i]:
            prev_name = prev.tool_name.lower()
            prev_is_read = any(prev_name.startswith(p) for p in read_prefixes)
            if prev_is_read:
                available_entities.update(_tool_relevant_entity_types(prev_name, server_name))
                available_entities.update(_TOOL_SECONDARY_ENTITIES.get(prev_name, set()))
                continue

            available_entities.update(_CREATED_ENTITY_BY_TOOL.get(prev_name, set()))

        if not requirements <= available_entities:
            return True

    return False


def _has_entity_keyword(name: str) -> bool:
    """Check if tool name contains any known entity keyword."""
    for et in ("event", "order", "account", "email", "invoice",
                "issue", "lead", "deal", "product", "restaurant",
                "channel", "message", "file", "contact", "payment",
                "menu", "cart", "transfer", "transaction"):
        if et in name:
            return True
    return False

# ── Tool-to-conceptual-entity override ──
# Some tools operate on entities whose name cannot be extracted from the tool
# name itself (e.g. "checkout" → cart, "transfer" → account).  Without these
# overrides _detect_missing_dependency falsely flags get_cart→checkout and
# get_balance→transfer as missing a dependency.
_TOOL_ENTITY_OVERRIDE: dict[str, str] = {
    # Shopping domain
    "checkout": "order",          # checkout completes an order
    "get_cart": "order",          # cart IS the order-in-progress
    "clear_cart": "order",
    "add_to_cart": "order",
    "remove_from_cart": "order",
    "update_cart_quantity": "order",
    "rate_order": "order",
    "return_order": "order",
    "reorder": "order",
    "apply_coupon": "order",
    # Banking domain
    "get_balance": "account",     # balance is a property of account
    "get_history": "account",     # transaction history belongs to account
    "get_statement": "account",   # statement belongs to account
    "transfer": "account",        # transfer moves money between accounts
    "wire_transfer": "account",
    "deposit": "account",
    "withdraw": "account",
    "apply_loan": "account",
    "bill_pay": "account",
    # Payments domain
    "pay_invoice": "invoice",
    "get_invoice": "invoice",
    "refund_invoice": "invoice",
    "cancel_payment": "payment",
    "get_payment": "invoice",    # Calendar / email domains
    "add_to_wishlist": "wishlist",  # shopping: wishlist entity (not in keyword list)
    "move_to_thread": "email",    # email threading
    "list_inbox": "email",        # inbox listing resolves email entities
    "get_thread": "email",        # thread lookup resolves email entities (for reply/forward)
    "get_attachments": "email",   # attachment lookup resolves email entities
    "mark_read": "email",         # operates on email entities
    "mark_unread": "email",       # operates on email entities
    "add_label": "email",         # labels an email entity
    "remove_label": "email",      # removes label from email entity
    # Filesystem domain — cmd tools operate on file/directory entities
    "chmod": "file",
    "chown": "file",
    "cp": "file",
    "rm": "file",
    "mv": "file",
    "mkdir": "file",
    "touch": "file",
}

_DOMAIN_TOOL_ENTITY_OVERRIDE: dict[str, dict[str, str]] = {
    "banking": {
        "schedule_transfer": "scheduled_transfer",
        "cancel_transfer": "scheduled_transfer",
    },
    "issue_tracker": {
        "add_label": "issue",
        "remove_label": "issue",
        "add_watcher": "issue",
        "remove_watcher": "issue",
        "set_milestone": "issue",
        "time_track": "issue",
        "add_to_sprint": "issue",
        "remove_from_sprint": "issue",
        "create_subtask": "issue",
    },
    "team_chat": {
        "get_thread": "thread",
        "get_user_status": "user",
        "send_dm": "user",
    },
    "calendar": {
        "add_attendee": "event",
        "remove_attendee": "event",
        "set_reminder": "event",
        "create_recurring": "event",
    },
    "food_delivery": {
        "get_menu": "restaurant",
        "filter_by_dietary": "restaurant",
        "get_popular_items": "restaurant",
        "add_tip": "order",
        "contact_support": "order",
        "rate_order": "order",
    },
    "filesystem": {
        "ls": "file",
        "cat": "file",
        "stat": "file",
        "head": "file",
        "tail": "file",
        "find": "file",
        "grep": "file",
        "tree": "file",
        "pwd": "file",
        "du": "file",
        "df": "file",
    },
}


# ── Secondary entity resolution ──
# Some "get" tools return JOINed entities — e.g. get_deal returns
# {deal, contact, lead}.  These tools resolve more than their primary
# entity, so _detect_missing_dependency should credit them as valid
# preceding reads for all resolved entities.
_TOOL_SECONDARY_ENTITIES: dict[str, set[str]] = {
    # CRM: get_deal returns linked contact + lead alongside the deal itself
    "get_deal": {"lead", "contact"},
}


def _tool_entity(name: str, domain: str = "") -> str:
    """Extract the conceptual entity a tool operates on.

    Checks override map first, then entity keyword list, then fallback.
    """
    tool = name.lower()
    server = domain.lower()
    if server and tool in _DOMAIN_TOOL_ENTITY_OVERRIDE.get(server, {}):
        return _DOMAIN_TOOL_ENTITY_OVERRIDE[server][tool]
    if tool in _TOOL_ENTITY_OVERRIDE:
        return _TOOL_ENTITY_OVERRIDE[tool]
    for et in ("event", "order", "account", "email", "invoice",
                "issue", "lead", "deal", "product", "restaurant",
                "channel", "message", "file", "contact", "payment",
                "menu", "cart", "transfer", "transaction"):
        if et in name:
            return et
    return name.split("_")[-1] if "_" in name else name


def _format_graph_hints(graph: dict) -> str:
    if not graph:
        return ""
    lines = ["## Tool Dependency Hints"]
    for tool, deps in sorted(graph.items()):
        parts = []
        if deps.get("explicit"):
            parts.append("→ " + ", ".join(deps["explicit"]))
        if deps.get("implicit") and not deps.get("explicit"):
            parts.append("~ " + ", ".join(deps["implicit"][:3]))
        if parts:
            lines.append(f"  {tool} {'; '.join(parts)}")
    return "\n".join(lines)


def _fuzzy_match_tool(raw: str, valid_names: set[str]) -> str | None:
    """Try to fix a hallucinated tool name by finding the closest valid match."""
    raw_lower = raw.lower()
    for name in valid_names:
        if name.lower() == raw_lower:
            return name
    if raw_lower.endswith("s"):
        singular = raw_lower[:-1]
        for name in valid_names:
            if name.lower() == singular:
                return name
    for name in valid_names:
        nl = name.lower()
        if raw_lower in nl or nl in raw_lower:
            return name
    raw_words = set(raw_lower.replace("_", " ").split())
    best_name, best_overlap = None, 0
    for name in valid_names:
        name_words = set(name.lower().replace("_", " ").split())
        overlap = len(raw_words & name_words)
        if overlap > best_overlap and overlap >= max(1, len(raw_words) - 1):
            best_overlap, best_name = overlap, name
    return best_name


_CREATED_ENTITY_BY_TOOL: dict[str, set[str]] = {
    "create_event": {"event"},
    "create_recurring": {"event"},
    "create_invoice": {"invoice"},
    "pay_invoice": {"payment"},
    "refund_invoice": {"refund"},
    "dispute_invoice": {"dispute"},
    "schedule_transfer": {"scheduled_transfer"},
    "create_webhook": {"webhook"},
    # NOTE: send_email intentionally omitted. It produces an outgoing message,
    # not an inbox email that subsequent tools (get_email/reply/archive/mark_read/…)
    # can operate on. Including it here caused the PROVE filter to accept
    # send_email → {get_email, reply_email, forward_email, …} edges which are
    # semantically reversed. Chains needing "send then observe" are not
    # expressible in this domain's tool surface.
    "create_draft": {"draft"},
    "create_filter": {"filter"},
    "mkdir": {"file"},
    "touch": {"file"},
    "cp": {"file"},
    "mv": {"file"},
    "create_lead": {"lead"},
    "create_contact": {"contact"},
    "convert_lead": {"contact"},
    "create_deal": {"deal"},
    "create_task": {"task"},
    "add_note": {"note"},
    "create_issue": {"issue"},
    "create_sprint": {"sprint"},
    "create_subtask": {"subtask"},
    "time_track": {"time_entry"},
    "add_to_cart": {"cart_item"},
    "checkout": {"order"},
    "create_order": {"order"},
    "reorder": {"order"},
    "return_order": {"return"},
    "create_channel": {"channel"},
    "send_message": {"message"},
    "create_thread": {"thread"},
    "send_dm": {"dm"},
    "contact_support": {"ticket"},
}


_DOMAIN_TOOL_REQUIREMENTS: dict[str, dict[str, set[str]]] = {
    "calendar": {
        "get_event": {"event"},
        "update_event": {"event"},
        "delete_event": {"event"},
        "add_attendee": {"event"},
        "remove_attendee": {"event"},
        "set_reminder": {"event"},
        "respond_to_event": {"event"},
    },
    "banking": {
        "get_balance": {"account"},
        "get_history": {"account"},
        "get_statement": {"account"},
        "deposit": {"account"},
        "withdraw": {"account"},
        "transfer": {"account"},
        "wire_transfer": {"account"},
        "bill_pay": {"account"},
        "schedule_transfer": {"account"},
        "freeze_account": {"account"},
        "unfreeze_account": {"account"},
        "cancel_transfer": {"scheduled_transfer"},
    },
    "payments": {
        "get_invoice": {"invoice"},
        "list_invoices": {"invoice"},
        "pay_invoice": {"invoice"},
        "refund_invoice": {"invoice"},
        "dispute_invoice": {"invoice"},
        "cancel_payment": {"payment"},
        "list_webhooks": {"webhook"},
        "delete_webhook": {"webhook"},
    },
    "email": {
        "get_email": {"email"},
        "list_inbox": {"email"},
        "search_emails": {"email"},
        "mark_read": {"email"},
        "mark_unread": {"email"},
        "archive_email": {"email"},
        "add_label": {"email"},
        "remove_label": {"email"},
        "reply_email": {"email"},
        "forward_email": {"email"},
        "get_attachments": {"email"},
        "move_to_thread": {"email", "thread"},
        "get_thread": {"email"},
        "create_draft": {"draft"},
        "create_filter": {"filter"},
    },
    "filesystem": {
        "cat": {"file"},
        "stat": {"file"},
        "head": {"file"},
        "tail": {"file"},
        "grep": {"file"},
        "file_info": {"file"},
        "md5sum": {"file"},
        "sha256sum": {"file"},
        "wc": {"file"},
        "xxd": {"file"},
        "chmod": {"file"},
        "chown": {"file"},
        "rm": {"file"},
        "cp": {"file"},
        "mv": {"file"},
    },
    "crm": {
        "list_leads": {"lead"},
        "update_lead": {"lead"},
        "convert_lead": {"lead"},
        "delete_lead": {"lead"},
        "list_deals": {"deal"},
        "get_deal": {"deal"},
        "update_deal": {"deal"},
        "list_tasks": {"task"},
        "complete_task": {"task"},
    },
    "issue_tracker": {
        "get_issue": {"issue"},
        "list_issues": {"issue"},
        "assign_issue": {"issue", "user"},
        "comment_issue": {"issue"},
        "transition_issue": {"issue"},
        "add_label": {"issue"},
        "remove_label": {"issue"},
        "add_watcher": {"issue", "user"},
        "remove_watcher": {"issue", "user"},
        "set_milestone": {"issue"},
        "add_to_sprint": {"issue", "sprint"},
        "remove_from_sprint": {"issue"},
        "create_subtask": {"issue"},
        "list_subtasks": {"issue"},
        "time_track": {"issue"},
        "list_sprints": {"sprint"},
        "create_sprint": {"sprint"},
    },
    "shopping": {
        "get_product": {"product"},
        "compare_products": {"product"},
        "add_to_cart": {"product"},
        "update_cart_quantity": {"cart_item"},
        "remove_from_cart": {"cart_item"},
        "clear_cart": {"cart_item"},
        "checkout": {"cart_item"},
        "get_order": {"order"},
        "track_order": {"order"},
        "return_order": {"order"},
        "get_return_status": {"return"},
        "add_review": {"product"},
        "get_reviews": {"product"},
        "add_to_wishlist": {"product"},
        "remove_from_wishlist": {"product"},
    },
    "team_chat": {
        "get_channel": {"channel"},
        "send_message": {"channel"},
        "archive_channel": {"channel"},
        "react_message": {"channel", "message"},
        "create_thread": {"channel", "message"},
        "get_thread": {"thread"},
        "send_dm": {"user"},
    },
    "food_delivery": {
        "get_restaurant": {"restaurant"},
        "get_menu": {"restaurant"},
        "filter_by_dietary": {"restaurant"},
        "get_popular_items": {"restaurant"},
        "create_order": {"restaurant"},
        "get_order": {"order"},
        "list_orders": {"order"},
        "get_estimated_time": {"order"},
        "track_rider": {"order"},
        "cancel_order": {"order"},
        "add_tip": {"order"},
        "reorder": {"order"},
        "rate_order": {"order"},
        "contact_support": {"order"},
    },
}


# ── Domain-specific entity data-quality predicates ──────────────────
# Each predicate returns (qualified: bool, reason: str).  Only domains
# with explicit predicates are filtered; domains not listed here use a
# conservative all-pass fallback.

def _entity_record_qualifies(
    server_name: str,
    etype: str,
    record: dict[str, Any],
) -> tuple[bool, str]:
    """Check whether an entity record can support tool chains for its type.

    Returns (qualified, reason).  Qualified entities are those whose data
    meets the minimum requirements to serve as inputs for chain operations
    (e.g. a banking account with zero balance cannot support transfer chains).
    """
    domain_filters = DOMAIN_ENTITY_QUALITY_FILTERS.get(server_name, {})
    predicate = domain_filters.get(etype)
    if predicate is None:
        return True, ""
    try:
        qualified, reason = predicate(record)
        return bool(qualified), str(reason) if not qualified else ""
    except Exception as exc:
        import traceback
        from loguru import logger
        logger.warning(
            f"_entity_record_qualifies [{server_name}/{etype}] "
            f"predicate error for record {record.get('id', '?')}: {exc}. "
            f"Failing closed — entity will be filtered."
        )
        return False, "quality_predicate_error"


def _enrich_restaurant_menus(
    executor,
    session_id: str,
    server_name: str,
    entity_records: list[dict[str, Any]],
) -> None:
    """P0-1 Fix: two-stage enrichment for food_delivery restaurants.

    Primary discovery probes (list_restaurants, search_restaurants) return
    restaurant entities without menu data.  get_menu requires a restaurant_id
    arg and is not a parameterless discovery tool, so it never runs in the
    first pass.  Without menu/items, all restaurants fail the quality filter.

    This function calls get_menu for each restaurant (up to a cap) and merges
    the result back into entity_records so the quality predicate has data.
    """
    MAX_RESTAURANTS_TO_ENRICH = 8  # cap to avoid excessive probe overhead

    restaurant_indices = [
        i for i, rec in enumerate(entity_records)
        if rec.get("type") == "restaurant" and not rec.get("data", {}).get("menu")
    ]
    if not restaurant_indices:
        return

    from src.live_mcp.types import ToolCall

    enriched = 0
    for i in restaurant_indices[:MAX_RESTAURANTS_TO_ENRICH]:
        rec = entity_records[i]
        rid = rec.get("id")
        if not rid:
            continue
        result = executor.execute(
            session_id,
            ToolCall("get_menu", {"restaurant_id": rid}, call_id=f"enrich_menu_{rid}"),
            domain=server_name,
        )
        if not result.success or result.state_changed:
            continue
        obs = result.observation
        if isinstance(obs, dict):
            menu_items = obs.get("items", obs.get("menu", []))
            if isinstance(menu_items, list):
                rec["data"]["menu"] = menu_items
                # Also add a lightweight summary for the quality predicate
                rec["data"]["items"] = menu_items
                enriched += 1
        elif isinstance(obs, list):
            rec["data"]["menu"] = obs
            rec["data"]["items"] = obs
            enriched += 1

    if enriched > 0:
        from loguru import logger
        logger.debug(
            f"_enrich_restaurant_menus: enriched {enriched} "
            f"restaurants with menu data for {server_name}"
        )


DOMAIN_ENTITY_QUALITY_FILTERS: dict[str, dict[str, Callable[..., tuple[bool, str]]]] = {
    # Global entity predicates deliberately remain empty. Whether a zero-
    # balance account, out-of-stock product, or menu-less restaurant is usable
    # depends on the selected tool chain. Chain-specific handler facts are
    # enforced by _entity_record_satisfies_chain; unknown fields pass through
    # to live execution/replay.
}


_DOMAIN_TOOL_RELEVANT: dict[str, dict[str, set[str]]] = {
    "calendar": {"list_events": {"event"}, "search_events": {"event"}, "get_free_busy": {"event"}, "check_conflicts": {"event"}},
    "banking": {"list_accounts": {"account"}, "list_transactions": {"account"}, "get_exchange_rate": {"account"}, "apply_loan": {"account"}},
    "payments": {"get_invoice": {"invoice", "payment"}, "list_invoices": {"invoice", "payment"}, "list_webhooks": {"webhook"}},
    "email": {"list_inbox": {"email"}, "search_emails": {"email"}, "list_threads": {"thread"}, "list_drafts": {"draft"}, "create_draft": {"email", "draft"}},
    "filesystem": {name: {"file"} for name in (
        "pwd", "ls", "find", "tree", "du", "df", "sort", "uniq", "cut",
        "sed", "awk", "split", "diff", "readlink",
    )},
    "crm": {
        "list_leads": {"lead"}, "list_contacts": {"contact"},
        "list_deals": {"deal"}, "list_tasks": {"task"},
        "search_contacts": {"contact"},
        "create_deal": {"lead", "contact", "deal"},
        "create_task": {"deal", "contact", "task"},
        "add_note": {"lead", "contact", "deal", "note"},
    },
    "issue_tracker": {"list_issues": {"issue"}, "search_issues": {"issue"}, "list_sprints": {"sprint"}},
    "shopping": {
        "search_products": {"product"}, "list_categories": {"product"},
        "get_coupons": {"product"}, "apply_coupon": {"cart_item"},
        "get_cart": {"cart_item", "product"},
        "get_wishlist": {"wishlist", "product"},
    },
    "team_chat": {"list_channels": {"channel"}, "get_user_status": {"user"}, "search_messages": {"channel", "message"}},
    "food_delivery": {"search_restaurants": {"restaurant"}, "list_orders": {"order"}},
}


_DISCOVERY_TOOL_PREFIXES = (
    "list_", "search_", "get_free_busy", "check_conflicts",
    "get_working_hours", "change_timezone", "export_calendar",
    "get_exchange_rate", "list_categories", "get_coupons", "apply_coupon",
    "get_wishlist", "get_cart", "get_time_report", "get_user_status",
    "pwd", "ls", "cat", "stat", "head", "tail", "find", "grep",
    "tree", "du", "df", "file_info", "md5sum", "sha256sum", "wc",
    "xxd", "sort", "uniq", "cut", "sed", "awk", "split", "diff",
)


_ENTITY_ID_FIELD_TYPES: dict[str, str] = {
    "event_id": "event", "account_id": "account", "from_account": "account",
    "to_account": "account", "invoice_id": "invoice", "payment_id": "payment",
    "webhook_id": "webhook", "refund_id": "refund", "dispute_id": "dispute",
    "email_id": "email", "thread_id": "thread", "draft_id": "draft",
    "filter_id": "filter", "lead_id": "lead", "contact_id": "contact",
    "deal_id": "deal", "task_id": "task", "note_id": "note",
    "issue_id": "issue", "sprint_id": "sprint", "subtask_id": "subtask",
    "entry_id": "time_entry", "restaurant_id": "restaurant", "order_id": "order",
    "return_id": "return", "product_id": "product", "channel_id": "channel",
    "message_id": "message", "dm_id": "dm", "ticket_id": "ticket",
    "transfer_id": "scheduled_transfer", "scheduled_txn_id": "scheduled_transfer",
}


_READONLY_REQUIRED_PROBE_ARGS: dict[str, dict[str, dict[str, Any]]] = {
    "calendar": {
        "search_events": {"query": ""},
        "get_free_busy": {
            "emails": ["current_user@example.com"],
            "start_time": "2026-06-20T00:00",
            "end_time": "2026-06-30T23:59",
        },
        "check_conflicts": {
            "start_time": "2026-06-25T10:00",
            "end_time": "2026-06-25T11:00",
        },
    },
    "team_chat": {
        "search_messages": {"query": ""},
    },
    "food_delivery": {
        "search_restaurants": {"query": ""},
    },
}


def _is_readonly_discovery_tool(tool_schema: dict[str, Any]) -> bool:
    annotations = tool_schema.get("annotations") or {}
    if not bool(annotations.get("readonly")) or bool(annotations.get("mutating")):
        return False
    name = str(tool_schema.get("name") or "")
    return name.startswith(("list_", "search_")) or name in {
        "get_cart", "get_wishlist", "get_coupons", "get_user_status",
        "get_working_hours", "get_exchange_rate", "pwd", "ls", "tree", "df",
    }


def _readonly_probe_args(tool_schema: dict[str, Any], server_name: str) -> dict[str, Any] | None:
    name = str(tool_schema.get("name") or "")
    required = list(tool_schema.get("input_schema", {}).get("required", []) or [])
    if not required:
        return {}
    args = _READONLY_REQUIRED_PROBE_ARGS.get(server_name, {}).get(name)
    if args is None:
        return None
    if not set(required) <= set(args):
        return None
    return dict(args)


def _format_entity_summary(
    eid: str,
    etype: str,
    edata: dict[str, Any] | None = None,
) -> str:
    key_fields = {}
    if isinstance(edata, dict):
        for fk in ("name", "title", "subject", "owner", "status", "type",
                   "balance", "amount", "price", "stage", "priority",
                   "author", "content", "timestamp", "sender", "recipient",
                   "cuisine", "rating", "total", "member_count"):
            if fk in edata:
                val = edata[fk]
                if isinstance(val, str) and len(val) > 40:
                    val = val[:37] + "..."
                key_fields[fk] = val
    if key_fields:
        return f"  {eid} ({etype}): {key_fields}"
    return f"  {eid} ({etype})"


def _extract_probe_entities(
    obj: Any,
    add_entity,
    *,
    server_name: str,
    tool_name: str,
) -> None:
    if isinstance(obj, dict):
        if server_name == "payments":
            payment_id = obj.get("payment_id")
            if isinstance(payment_id, str) and payment_id:
                add_entity(
                    payment_id,
                    "payment",
                    {
                        "payment_id": payment_id,
                        "invoice_id": obj.get("invoice_id", ""),
                        "status": obj.get("payment_status", obj.get("status", "")),
                        "amount": obj.get("amount"),
                    },
                )

        if server_name == "filesystem":
            cwd = obj.get("cwd")
            if isinstance(cwd, str) and cwd:
                add_entity(cwd, "file", obj)
            path = obj.get("path")
            if isinstance(path, str) and path:
                add_entity(path, "file", obj)
                entries = obj.get("entries")
                if isinstance(entries, list):
                    import posixpath
                    for entry in entries:
                        if not isinstance(entry, dict):
                            continue
                        name = entry.get("name")
                        if not isinstance(name, str) or not name:
                            continue
                        child_path = posixpath.normpath(posixpath.join(path, name))
                        add_entity(child_path, "file", {**entry, "path": child_path})

        for field, etype in _ENTITY_ID_FIELD_TYPES.items():
            value = obj.get(field)
            if isinstance(value, str) and value:
                add_entity(value, etype, obj)

        if server_name == "shopping" and tool_name == "get_cart":
            for item in obj.get("cart") or []:
                if isinstance(item, dict) and item.get("product_id"):
                    add_entity(str(item["product_id"]), "cart_item", item)

        if server_name == "team_chat":
            for member in obj.get("members") or []:
                if isinstance(member, str) and member:
                    add_entity(member, "user", None)
            statuses = obj.get("statuses")
            if isinstance(statuses, dict):
                for user_id, status in statuses.items():
                    add_entity(str(user_id), "user", {"status": status})

        if server_name == "issue_tracker":
            for field in ("assignee", "user", "author"):
                value = obj.get(field)
                if isinstance(value, str) and value:
                    add_entity(value, "user", obj)
            for watcher in obj.get("watchers") or []:
                if isinstance(watcher, str) and watcher:
                    add_entity(watcher, "user", None)

        if server_name == "email":
            thread_id = obj.get("thread_id")
            if isinstance(thread_id, str) and thread_id:
                add_entity(thread_id, "thread", obj)

        for value in obj.values():
            if isinstance(value, (dict, list)):
                _extract_probe_entities(
                    value,
                    add_entity,
                    server_name=server_name,
                    tool_name=tool_name,
                )
    elif isinstance(obj, list):
        for item in obj:
            _extract_probe_entities(
                item,
                add_entity,
                server_name=server_name,
                tool_name=tool_name,
            )


def _compact_sampling_context(live_context: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": live_context.get("source", "live_readonly_probe"),
        "entity_ids": list(live_context.get("entity_ids", []))[:50],
        "entity_summaries": list(live_context.get("entity_summaries", []))[:50],
        "entity_records": list(live_context.get("entity_records", []))[:50],
        "entity_types": list(live_context.get("entity_types", [])),
        "probe_results": list(live_context.get("probe_results", [])),
    }


def _live_context_to_prompt_state(live_context: dict[str, Any]) -> dict[str, Any]:
    """Convert read-only probe entities into the compact state formatter shape.

    P1: Prefer qualified_entity_ids / qualified_entity_records so that
    zero-balance accounts, out-of-stock products, etc. are excluded from
    the Teacher's Current State view.  Fall back to raw entity_ids only
    when the qualified fields are completely absent (old cache without
    supporting-data filtering).  An explicitly empty qualified list is
    respected — it is NOT treated as "absent".
    """
    prompt_state: dict[str, dict[str, Any]] = {}
    if "qualified_entity_ids" in live_context:
        entity_source = live_context["qualified_entity_ids"]
        summaries = list(live_context.get("qualified_entity_summaries",
                                          live_context.get("entity_summaries", [])))
    else:
        entity_source = live_context.get("entity_ids", [])
        summaries = list(live_context.get("entity_summaries", []))
    for idx, item in enumerate(entity_source):
        if not isinstance(item, dict):
            continue
        eid = str(item.get("id") or "")
        etype = str(item.get("type") or "entity")
        if not eid:
            continue
        container = f"live_probe_{etype}s"
        summary = summaries[idx] if idx < len(summaries) else ""
        prompt_state.setdefault(container, {})[eid] = {
            "id": eid,
            "type": etype,
            "source": "live_readonly_probe",
            "summary": summary,
        }
    return prompt_state


def _live_context_has_entity_type(
    live_context: dict[str, Any] | None,
    entity_type: str,
    created_types: set[str] | None = None,
) -> bool:
    if created_types and entity_type in created_types:
        return True
    if not live_context:
        return False
    return any(
        isinstance(item, dict) and item.get("type") == entity_type
        for item in live_context.get("entity_ids", [])
    )


def _tool_existing_entity_requirements(tool_name: str, server_name: str = "") -> set[str]:
    tool = tool_name.lower()
    server = server_name.lower()

    if server:
        domain_requirements = _DOMAIN_TOOL_REQUIREMENTS.get(server, {})
        if tool in domain_requirements:
            requirements = set(domain_requirements[tool])
            created = set(_CREATED_ENTITY_BY_TOOL.get(tool, set()))
            if server == "filesystem" and tool in {"cp", "mv"}:
                created.discard("file")
            if server == "filesystem" and tool == "readlink":
                created.discard("file")
            # reorder consumes an existing order and creates a different order.
            # Its input requirement must not be erased by the shared type.
            if server == "food_delivery" and tool == "reorder":
                created.discard("order")
            requirements.difference_update(created)
            return requirements
    if tool.startswith(_DISCOVERY_TOOL_PREFIXES):
        return set()
    if tool.startswith("create_") and tool not in {
        "create_task", "create_deal", "create_thread", "create_subtask", "create_order"
    }:
        return set()
    if tool in {"send_email", "send_dm", "mkdir", "touch"}:
        return set()

    requirements: set[str] = set()
    if "event" in tool or "attendee" in tool or "reminder" in tool or "recurring" in tool:
        requirements.add("event")
    if "account" in tool or tool in {
        "get_balance", "get_history", "get_statement", "deposit", "withdraw",
        "transfer", "bill_pay", "freeze_account", "unfreeze_account", "apply_loan",
        "schedule_transfer",
    }:
        requirements.add("account")
    if "invoice" in tool or tool in {"pay_invoice", "dispute_invoice"}:
        requirements.add("invoice")
    if tool in {"refund_invoice", "cancel_payment"}:
        requirements.add("payment")
    if "webhook" in tool and not tool.startswith(("create_", "list_")):
        requirements.add("webhook")
    if "email" in tool or tool in {
        "mark_read", "mark_unread", "archive_email", "reply_email",
        "forward_email", "get_attachments",
    }:
        requirements.add("email")
    if "draft" in tool and not tool.startswith("create_"):
        requirements.add("draft")
    if "filter" in tool and not tool.startswith(("create_", "list_")):
        requirements.add("filter")
    if "lead" in tool or tool == "convert_lead":
        requirements.add("lead")
    if "contact" in tool:
        requirements.add("contact")
    if "deal" in tool or tool == "add_note":
        requirements.add("deal")
    if tool in {"update_task", "complete_task"}:
        requirements.add("task")
    if "issue" in tool or tool in {
        "assign_issue", "comment_issue", "transition_issue", "add_watcher",
        "remove_watcher", "set_milestone", "add_label", "remove_label",
    }:
        requirements.add("issue")
    if tool in {"assign_issue", "add_watcher", "remove_watcher"}:
        requirements.add("user")
    if "sprint" in tool:
        requirements.add("sprint")
    if tool == "add_to_sprint":
        requirements.update({"issue", "sprint"})
    if "subtask" in tool and not tool.startswith("create_"):
        requirements.add("subtask")
    if "product" in tool or tool in {
        "add_to_cart", "compare_products", "add_review", "get_reviews",
    }:
        requirements.add("product")
    if tool in {"checkout", "update_cart_quantity", "remove_from_cart", "clear_cart"}:
        requirements.add("cart_item")
    if "order" in tool or tool in {
        "track_order", "return_order", "get_estimated_time", "add_tip",
        "cancel_order", "reorder", "rate_order", "contact_support",
    }:
        requirements.add("order")
    if "restaurant" in tool or "menu" in tool or tool in {
        "filter_by_dietary", "get_popular_items", "create_order",
    }:
        requirements.add("restaurant")
    if "channel" in tool or tool in {"send_message", "archive_channel", "create_thread"}:
        requirements.add("channel")
    if tool in {"react_message", "create_thread"}:
        requirements.add("message")
    if "message" in tool and tool not in {"send_message", "search_messages"}:
        requirements.add("message")
    if tool in {"chmod", "chown", "rm", "cp", "mv"}:
        requirements.add("file")
    if tool in {"grep", "sed", "awk", "wc", "sort", "uniq", "cut", "head", "tail",
                "cat", "stat", "md5sum", "sha256sum", "xxd", "diff", "split",
                "join", "readlink", "file_info"}:
        requirements.add("file")

    if server_name == "shopping" and tool == "create_order":
        requirements.discard("order")
    if server_name == "team_chat" and tool == "send_message":
        requirements.discard("message")
    created = set(_CREATED_ENTITY_BY_TOOL.get(tool, set()))
    if server_name == "filesystem" and tool in {"cp", "mv"}:
        created.discard("file")
    requirements.difference_update(created)
    # readlink returns a target path, but that output does not create the
    # symlink node consumed by the handler. It always requires an existing
    # filesystem entity whose record type is symlink.
    if server_name == "filesystem" and tool == "readlink":
        requirements.add("file")
    return requirements


def _tool_relevant_entity_types(tool_name: str, server_name: str = "") -> set[str]:
    tool = tool_name.lower()
    server = server_name.lower()
    relevant = set(_tool_existing_entity_requirements(tool, server_name))
    relevant.update(_DOMAIN_TOOL_RELEVANT.get(server, {}).get(tool, set()))
    relevant.update(_DOMAIN_TOOL_REQUIREMENTS.get(server, {}).get(tool, set()))
    relevant.update(_CREATED_ENTITY_BY_TOOL.get(tool, set()))

    if tool.startswith("list_") or tool.startswith("search_"):
        noun = tool.removeprefix("list_").removeprefix("search_")
        if noun.endswith("ies"):
            relevant.add(noun[:-3] + "y")
        elif noun.endswith("s"):
            relevant.add(noun[:-1])
        else:
            relevant.add(noun)

    if "product" in tool or tool in {"add_to_cart", "get_wishlist", "checkout"}:
        relevant.add("product")
    if "order" in tool or tool in {
        "checkout", "track_order", "return_order", "add_tip", "contact_support",
    }:
        relevant.add("order")
    if any(k in tool for k in ("restaurant", "menu", "popular", "dietary")):
        relevant.add("restaurant")
    if "deal" in tool:
        relevant.add("deal")
    if "contact" in tool:
        relevant.add("contact")
    if "lead" in tool:
        relevant.add("lead")
    if tool in {"create_task", "add_note"}:
        relevant.add("deal")
    if "issue" in tool or tool in {"add_to_sprint", "remove_from_sprint"}:
        relevant.add("issue")
    if tool in {"assign_issue", "add_watcher", "remove_watcher"}:
        relevant.add("user")
    if "sprint" in tool or tool in {"add_to_sprint", "remove_from_sprint"}:
        relevant.add("sprint")
    if "channel" in tool or tool in {"send_message", "create_thread", "react_message"}:
        relevant.add("channel")
    if "message" in tool or tool in {"create_thread", "react_message"}:
        relevant.add("message")
    return relevant


def _is_observable_chain_start(tool_name: str) -> bool:
    tool = tool_name.lower()
    if tool.startswith((
        "list_", "search_", "filter_", "get_", "find_", "lookup_", "check_",
        "view_", "browse_", "read_",
    )):
        return True
    if tool in {"pwd", "ls", "tree", "cat", "stat", "head", "tail", "df", "find"}:
        return True
    return False


def _chain_is_feasible(
    chain: list[str],
    server_name: str,
    live_context: dict[str, Any],
) -> tuple[bool, str]:
    entity_source = (
        live_context.get("qualified_entity_ids")
        if "qualified_entity_ids" in live_context
        else live_context.get("entity_ids", [])
    ) or []
    record_source = (
        live_context.get("qualified_entity_records")
        if "qualified_entity_records" in live_context
        else live_context.get("entity_records", [])
    ) or []
    records_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for item in record_source:
        if not isinstance(item, dict):
            continue
        etype = str(item.get("type") or "")
        eid = str(item.get("id") or "")
        data = item.get("data")
        if etype and eid:
            records_by_key[(etype, eid)] = data if isinstance(data, dict) else {}

    live_ids_by_type: dict[str, set[str]] = {}
    for item in entity_source:
        if isinstance(item, dict) and item.get("type") and item.get("id"):
            etype = str(item["type"])
            eid = str(item["id"])
            if _entity_record_satisfies_chain(
                server_name=server_name,
                chain_seed=chain,
                etype=etype,
                record=records_by_key.get((etype, eid), {}),
            ):
                live_ids_by_type.setdefault(etype, set()).add(eid)
    created_counts: dict[str, int] = {}
    minimum_cardinality = {
        ("banking", "transfer", "account"): 2,
        ("banking", "schedule_transfer", "account"): 2,
        ("filesystem", "join", "file"): 2,
        ("shopping", "compare_products", "product"): 2,
    }
    for idx, tool_name in enumerate(chain):
        tool = tool_name.lower()
        requirements = _tool_existing_entity_requirements(tool, server_name)
        # PROVE §3.2 Step 2: chain is feasible if live_context already has
        # the required entities (probed via read-only discovery tools).
        # A chain can start with a mutating tool (e.g., update_event) as
        # long as the entity exists in the live state.
        missing = []
        for etype in sorted(requirements):
            required_count = minimum_cardinality.get((server_name, tool, etype), 1)
            available_count = len(live_ids_by_type.get(etype, set())) + created_counts.get(etype, 0)
            if available_count < required_count:
                missing.append(f"{etype}({available_count}/{required_count})")
        if missing:
            return False, f"{tool} requires missing entity types {missing}"
        for etype in _CREATED_ENTITY_BY_TOOL.get(tool, set()):
            created_counts[etype] = created_counts.get(etype, 0) + 1
    return True, "ok"


def _chain_respects_state_preconditions(server_name: str, chain: list[str]) -> bool:
    if server_name == "payments":
        if (
            "pay_invoice" in chain and "cancel_payment" in chain
            and chain.index("pay_invoice") < chain.index("cancel_payment")
        ):
            # pay_invoice creates a settled payment; cancel_payment accepts
            # pending payments only.
            return False
        if "create_invoice" in chain and "refund_invoice" in chain:
            create_idx = chain.index("create_invoice")
            refund_idx = chain.index("refund_invoice")
            if create_idx < refund_idx and "pay_invoice" not in chain[create_idx + 1:refund_idx]:
                return False
    if server_name == "team_chat":
        if "create_channel" in chain:
            channel_idx = chain.index("create_channel")
            for message_tool in ("create_thread", "react_message"):
                if message_tool in chain:
                    message_idx = chain.index(message_tool)
                    if channel_idx < message_idx and "send_message" not in chain[channel_idx + 1:message_idx]:
                        return False

    if server_name == "food_delivery":
        # Both creators return a newly placed order.  With unique tools in a
        # dependency path, at most one update_order_status can follow, which
        # can only advance placed -> confirmed.  The new order therefore
        # cannot reach delivering/delivered for track_rider/rate_order.
        for creator in ("create_order", "reorder"):
            if creator not in chain:
                continue
            create_idx = chain.index(creator)
            for target in ("track_rider", "rate_order"):
                if target in chain and create_idx < chain.index(target):
                    return False
        if "cancel_order" in chain:
            cancel_idx = chain.index("cancel_order")
            for status_tool in ("rate_order", "track_rider", "update_order_status"):
                if status_tool in chain and chain.index(status_tool) > cancel_idx:
                    return False
    return True


def _extract_chain_context(
    chain_seed: list[str],
    server_name: str,
    live_context: dict[str, Any],
) -> dict[str, Any]:
    """PROVE §3.2 Step 2: extract chain-relevant IDs from live read-only probes.

    This provides a compact, chain-aligned subset of the live state that the teacher
    LLM can use as grounded reference for parameter generation.  The anti-hallucination
    constraint in generate_query prevents the teacher from inventing IDs outside this
    context.

    Returns:
        dict with:
          - "entity_ids": list of relevant entity IDs mapped to type
          - "entity_summaries": compact list of (id, type, key_fields) tuples
    """
    if not live_context or not chain_seed:
        return {}

    tools = set(chain_seed)
    # A parameter-free discovery predecessor is supposed to reveal the opaque
    # identifier consumed downstream.  Keep that ID in the validation view but
    # hide it from query synthesis so the generated user message does not erase
    # the dependency it was seeded from.
    discovery_hidden_types: set[str] = set()
    for idx, tool_name in enumerate(chain_seed[:-1]):
        lowered = tool_name.lower()
        if not lowered.startswith(("list_", "search_", "filter_", "browse_")):
            continue
        downstream_requirements: set[str] = set()
        for later_tool in chain_seed[idx + 1:]:
            downstream_requirements.update(
                _tool_existing_entity_requirements(later_tool, server_name)
            )
        discovery_hidden_types.update(
            _tool_relevant_entity_types(lowered, server_name)
            & downstream_requirements
        )
    relevant_types: set[str] = set()
    for tool_name in chain_seed:
        relevant_types.update(_tool_relevant_entity_types(tool_name, server_name))
    # When an earlier chain step creates the entity consumed downstream, the
    # query must ask for that creation rather than grounding the downstream
    # action on an unrelated pre-existing ID of the same type.
    creator_supplied_types: set[str] = set()
    for tool_name in chain_seed[:-1]:
        creator_supplied_types.update(
            _CREATED_ENTITY_BY_TOOL.get(tool_name.lower(), set())
        )
    relevant_types -= creator_supplied_types
    if server_name == "payments" and "cancel_payment" not in chain_seed:
        relevant_types.discard("payment")

    entity_ids: list[dict] = []
    entity_summaries: list[str] = []
    entity_records: list[dict[str, Any]] = []
    query_visible_entity_ids: list[dict[str, str]] = []
    query_visible_entity_summaries: list[str] = []
    query_grounding_summaries: list[str] = []
    seen_entity_keys: set[tuple[str, str]] = set()

    summaries_by_key: dict[tuple[str, str], str] = {}
    records_by_key: dict[tuple[str, str], dict[str, Any]] = {}

    # P0-1: prefer qualified entities when available; fall back to raw probes
    # only when qualified fields are ABSENT (not when they are explicit empty lists).
    if "qualified_entity_ids" in live_context:
        source_entity_ids = live_context["qualified_entity_ids"]
    else:
        source_entity_ids = live_context.get("entity_ids", [])
    if "qualified_entity_records" in live_context:
        source_entity_records = live_context["qualified_entity_records"]
    else:
        source_entity_records = live_context.get("entity_records", [])

    for item, summary in zip(
        live_context.get("entity_ids", []),
        live_context.get("entity_summaries", []),
    ):
        if isinstance(item, dict):
            summaries_by_key[(str(item.get("type")), str(item.get("id")))] = str(summary)
    for record in live_context.get("entity_records", []):
        if not isinstance(record, dict):
            continue
        key = (str(record.get("type")), str(record.get("id")))
        data = record.get("data")
        records_by_key[key] = data if isinstance(data, dict) else {}
    for item in source_entity_ids:
        if not isinstance(item, dict):
            continue
        eid = str(item.get("id") or "")
        etype = str(item.get("type") or "")
        if not eid or etype not in relevant_types:
            continue
        key = (etype, eid)
        if key in seen_entity_keys:
            continue
        record = records_by_key.get(key, {})
        if not _entity_record_satisfies_chain(
            server_name=server_name,
            chain_seed=chain_seed,
            etype=etype,
            record=record,
        ):
            continue
        seen_entity_keys.add(key)
        entity_ids.append({"id": eid, "type": etype})
        summary = summaries_by_key.get(key, f"  {eid} ({etype})")
        if server_name == "payments" and etype == "invoice":
            amount = record.get("amount")
            total_refunded = record.get("total_refunded", 0)
            if isinstance(amount, (int, float)):
                summary = f"{summary}; payable_amount={amount}"
                if "refund_invoice" in tools:
                    try:
                        remaining = max(0.0, float(amount) - float(total_refunded or 0))
                        summary = f"{summary}; remaining_refundable={remaining}"
                    except (TypeError, ValueError):
                        pass
            if record.get("payment_id"):
                summary = (
                    f"{summary}; linked_payment_id={record['payment_id']}"
                    f"; payment_status={record.get('payment_status', 'unknown')}"
                )
        elif server_name == "payments" and etype == "payment":
            summary = (
                f"{summary}; resource_type=payment"
                f"; linked_invoice_id={record.get('invoice_id', '')}"
            )
        if server_name == "food_delivery" and etype == "order" and "update_order_status" in tools:
            lifecycle = {
                "placed": ["confirmed", "cancelled"],
                "confirmed": ["preparing", "cancelled"],
                "preparing": ["delivering"],
                "delivering": ["delivered"],
                "delivered": [],
                "cancelled": [],
            }
            status = str(record.get("status", ""))
            summary = f"{summary}; allowed_next_status={lifecycle.get(status, [])}"
        entity_summaries.append(summary)
        entity_records.append({"id": eid, "type": etype, "data": record})
        if etype in discovery_hidden_types:
            selector_fields: dict[str, Any] = {}
            for field_name in (
                "name", "title", "subject", "customer", "owner", "status",
                "amount", "currency", "due_date", "category", "balance",
                "price", "quantity", "stage", "priority",
            ):
                if field_name in record:
                    selector_fields[field_name] = record[field_name]
            query_grounding_summaries.append(
                f"  grounded {etype} candidate: {selector_fields}"
            )
        else:
            query_visible_entity_ids.append({"id": eid, "type": etype})
            query_visible_entity_summaries.append(summary)
            query_grounding_summaries.append(summary)
    return {
        "entity_ids": entity_ids[:30],  # cap to avoid prompt overflow
        "entity_summaries": entity_summaries[:30],
        "entity_records": entity_records[:30],
        "query_visible_entity_ids": query_visible_entity_ids[:30],
        "query_visible_entity_summaries": query_visible_entity_summaries[:30],
        "query_grounding_summaries": query_grounding_summaries[:30],
        "opaque_id_hidden_types": sorted(discovery_hidden_types),
        "relevant_types": sorted(relevant_types),
        "source": "live_readonly_probe",
    }


def _entity_record_satisfies_chain(
    *,
    server_name: str,
    chain_seed: list[str],
    etype: str,
    record: dict[str, Any],
) -> bool:
    """Filter chain context entities by hard state preconditions.

    This does not decide whether the chain itself is valid; it keeps the
    anti-hallucination entity list from presenting IDs that the target tool
    cannot execute on under the current handler logic.
    """
    tools = set(chain_seed)
    if not record:
        return True
    if server_name == "payments" and etype == "invoice":
        status = str(record.get("status", ""))
        if "cancel_payment" in tools and "payment_status" in record and str(record["payment_status"]) != "pending":
            return False
        pay_before_refund = (
            "pay_invoice" in tools
            and "refund_invoice" in tools
            and chain_seed.index("pay_invoice") < chain_seed.index("refund_invoice")
        )
        if "pay_invoice" in tools and status in {"paid", "refunded", "partially_refunded"}:
            return False
        if "refund_invoice" in tools and not pay_before_refund and "status" in record:
            if status not in {"paid", "partially_refunded"}:
                return False
        if "dispute_invoice" in tools and "status" in record:
            if status not in {"paid", "pending"}:
                return False
    if server_name == "payments" and etype == "payment":
        if "cancel_payment" in tools and "status" in record and str(record["status"]) != "pending":
            return False
    if server_name == "food_delivery" and etype == "order":
        status = str(record.get("status", ""))
        # reorder consumes this existing order, then downstream order tools use
        # the newly-created placed order.  Source attributes such as status/tip
        # must not be applied to that new entity.
        reorder_idx = chain_seed.index("reorder") if "reorder" in tools else -1
        if status and reorder_idx < 0:
            lifecycle = {
                "placed": {"confirmed", "cancelled"},
                "confirmed": {"preparing", "cancelled"},
                "preparing": {"delivering"},
                "delivering": {"delivered"},
                "delivered": set(),
                "cancelled": set(),
            }
            possible_statuses = {status}
            for tool_name in chain_seed:
                if tool_name == "update_order_status":
                    possible_statuses = {
                        next_status
                        for current in possible_statuses
                        for next_status in lifecycle.get(current, set())
                    }
                elif tool_name == "cancel_order":
                    possible_statuses = (
                        {"cancelled"}
                        if possible_statuses & {"placed", "confirmed"}
                        else set()
                    )
                elif tool_name == "track_rider":
                    possible_statuses &= {"delivering"}
                elif tool_name == "rate_order":
                    possible_statuses &= {"delivered"}
                if not possible_statuses:
                    return False
        if "add_tip" in tools:
            add_tip_idx = chain_seed.index("add_tip")
            if reorder_idx < 0 or add_tip_idx < reorder_idx:
                if bool(record.get("tip", 0)):
                    return False
    if server_name == "shopping" and etype == "order":
        status = str(record.get("status", ""))
        if "return_order" in tools and status in {"returning", "returned"}:
            return False
    if server_name == "shopping" and etype == "product":
        if any(t in tools for t in {"add_to_cart", "add_to_wishlist", "checkout"}):
            for field in ("stock", "available", "in_stock"):
                if field in record and not bool(record[field]):
                    return False
                if field in record:
                    break
    if server_name == "food_delivery" and etype == "restaurant":
        if "create_order" in tools:
            menu = record.get("menu", record.get("items"))
            if ("menu" in record or "items" in record) and not menu:
                return False
    if server_name == "banking" and etype == "account":
        frozen = bool(record.get("frozen", False))
        unfreeze_after_freeze = (
            "freeze_account" in tools and "unfreeze_account" in tools
            and chain_seed.index("freeze_account") < chain_seed.index("unfreeze_account")
        )
        if "unfreeze_account" in tools and not unfreeze_after_freeze and "frozen" in record and not frozen:
            return False
        if any(t in tools for t in {"withdraw", "deposit", "transfer"}) and frozen:
            return False
        if any(t in tools for t in {"withdraw", "transfer", "wire_transfer", "bill_pay"}):
            if "balance" in record and float(record.get("balance", 0)) <= 0:
                return False
    if server_name == "calendar" and etype == "event":
        if "get_recurring_info" in tools and "recurrence" in record and not record.get("recurrence"):
            return False
        attendee_produced = "add_attendee" in tools
        if any(t in tools for t in {"remove_attendee", "respond_to_event"}):
            if not attendee_produced and "attendees" in record and not record.get("attendees"):
                return False
    if server_name == "filesystem" and "readlink" in tools:
        if "type" in record and str(record["type"]) not in {"symlink", "link"}:
            return False
    if server_name == "filesystem" and "tar_extract" in tools and etype == "file":
        if "type" in record and str(record["type"]) != "file":
            return False
    if server_name == "filesystem" and "join" in tools and etype == "file":
        if "type" in record and str(record["type"]) not in {"file", "symlink", "link"}:
            return False
    if server_name == "issue_tracker" and etype == "issue":
        if "remove_watcher" in tools and "add_watcher" not in tools:
            if "watchers" in record and not record.get("watchers"):
                return False
        if "remove_label" in tools and "add_label" not in tools:
            if "labels" in record and not record.get("labels"):
                return False
        if "remove_from_sprint" in tools and "add_to_sprint" not in tools:
            if "sprint_id" in record and not record.get("sprint_id"):
                return False
        if "transition_issue" in tools and str(record.get("state", "")) in {"closed", "cancelled"}:
            return False
    if server_name == "email" and etype == "email":
        if "remove_label" in tools and "labels" in record and not record.get("labels"):
            return False
        if "mark_read" in tools and "read" in record and bool(record["read"]):
            return False
        if "mark_unread" in tools and "read" in record and not bool(record["read"]):
            return False
        if "archive_email" in tools and "archived" in record and bool(record["archived"]):
            return False
    if server_name == "crm" and etype == "task":
        if "complete_task" in tools and str(record.get("status", "")) == "completed":
            return False
    return True
