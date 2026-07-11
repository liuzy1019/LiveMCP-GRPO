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
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

from src.live_mcp.config import SuiteConfig
from src.live_mcp.executor import LiveMCPExecutor
from src.live_mcp.manager import LiveMCPManager
from src.live_mcp.types import LiveTask, OracleCall, OracleProgram, to_plain
from src.utils import extract_json as _extract_json


@dataclass
class RobustnessPlan:
    """Immutable robustness perturbation plan, sampled before Teacher.

    Sampled once per task seed so that Teacher-visible schemas, Replay, and
    Parquet metadata all refer to the same configuration.  Rollout never
    re-randomizes perturbations for baseline.

    PROVE §3.2 Figure 2: robustness knobs are applied during generation,
    before the completed conversation is replay-validated.
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
    result: list[dict] = []
    for tool in tools:
        t = dict(tool)
        props = t.get("input_schema", {}).get("properties", {})
        if props:
            stripped: dict[str, dict] = {}
            for k, v in props.items():
                stripped[k] = {kk: vv for kk, vv in v.items() if kk != "enum"}
            t["input_schema"] = {**t.get("input_schema", {}), "properties": stripped}
        result.append(t)
    return result


def _build_teacher_visible_tools(
    domain_tools: list[dict],
    plan: RobustnessPlan,
) -> list[dict]:
    """Build the tool schemas that the Teacher LLM sees.

    Distractors are added BEFORE enum stripping (PROVE §3.2: distractor
    schemas also have enums stripped if that knob is active).  For
    missing-function tasks, the hidden tool is removed from Teacher-visible
    schemas so the Teacher generates a clarification/abstention trajectory.
    """
    tools = [dict(t) for t in domain_tools]
    # 1. Add distractor tools
    if plan.inject_distractors:
        tools.extend([dict(t) for t in plan.distractor_tools])
    # 2. Strip enums from Teacher-visible schemas only
    if plan.strip_enums:
        tools = _strip_enums_from_schemas(tools)
    # 3. Hide the missing-function tool
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

    # PROVE §3.2 Step 2: refresh sampling context every k conversations.
    # k=10 balances freshness (state changes after writes) vs. probe overhead.
    SAMPLING_CONTEXT_REFRESH_K: int = 10
    DEPENDENCY_CACHE_VERSION: int = 2
    DEPENDENCY_PAIR_BATCH_SIZE: int = 12
    DEPENDENCY_FAILURE_RETRY_SECONDS: float = 60.0

    # PROVE §3.2 Step 1 builds the graph from pairwise LLM classifications.
    # Deterministic augmentation is kept as an explicit ablation hook, but is
    # disabled for the baseline because it changes the LLM-discovered graph.
    ENABLE_DETERMINISTIC_GRAPH_AUGMENTATION: bool = False

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

    @staticmethod
    def _round_goal_satisfied(round_calls: list, goal_tool: str) -> bool:
        """Return whether a round completed its bound capability or abstained."""
        if not goal_tool:
            return True
        for call in round_calls:
            action = getattr(call, "action", "tool_call")
            if action in ("ask_clarification", "report_error"):
                return True
            if action == "tool_call" and getattr(call, "tool_name", "") == goal_tool:
                return True
        return False

    def _run_turn_loop(
        self,
        teacher,
        current_query: str,
        server_tools: list[dict],
        server_name: str,
        session_id: str,
        difficulty: str,
        dep_hints: str,
        local_rng: random.Random,
        chain_seed: list[str] | None,
        round_idx: int,
        reference_date: str = "",
        chain_progress_start: int = 0,
        max_calls_this_round: int = 0,
        chain_context: dict[str, Any] | None = None,
        blocked_tools: set[str] | None = None,
        missing_function_contract: bool = False,
        prior_execution_history: list[dict[str, Any]] | None = None,
    ) -> tuple[list, list[dict], list[Any], set[str], list[OracleCall], list[Any]]:
        """Run one conversation round of teacher-driven tool execution.

        chain_progress_start: cumulative chain_seed steps already satisfied.
        Baseline initial turns should complete the atomic chain goal; this value
        remains useful for retry/recovery and defensive validation.

        max_calls_this_round: optional diagnostic/ablation cap on real tool
        calls. Baseline passes 0 (no per-round cap); it must not distribute
        dependency-chain nodes across user turns.

        chain_context: live-probed entity values for hallucination prevention
        in decide_action. Extracted from _extract_chain_context in generate_one.

        Returns (oracle_calls, execution_history, oracle_observations, required_tools).
        oracle_observations is 1:1 aligned with oracle_calls — each entry is the
        raw tool observation dict for the corresponding oracle call, or {} for
        terminal actions (ask_clarification, final_answer, report_error).
        """
        from src.live_mcp.task_planner import ContinuationPolicy
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

        # Dedup oracle tool calls within a round. Same (tool_name, args_repr)
        # must not appear twice because repeated oracle calls inflate the ground
        # truth without adding new task progress.
        seen_oracle_keys: set[tuple[str, str]] = set()
        # LIST-class discovery tools are one-shot collection reads.
        seen_read_tools: set[str] = set()

        def _oracle_key(name: str, args: dict) -> tuple[str, str]:
            try:
                args_repr = json.dumps(args, sort_keys=True, default=str, ensure_ascii=False)
            except Exception:
                args_repr = repr(sorted(args.items()) if isinstance(args, dict) else args)
            return (name, args_repr)

        def _add_oracle(call: OracleCall) -> bool:
            """Append call to oracle_calls if not duplicate / not over budget.
            Returns True if appended.
            """
            # Terminal actions are part of the oracle contract but do not
            # consume the 2-5 tool-call budget and are not deduplicated.
            if call.action != "tool_call":
                oracle_calls.append(call)
                return True

            # Hard cap on real tool calls per task.
            real_count = sum(1 for oc in oracle_calls if oc.action == "tool_call")
            if real_count >= 5:
                return False
            # LIST-class tools (one-shot collections): dedup by name only.
            # get_/search_/find_ are entity reads — different entities are
            # legitimate, so they fall through to (name,args) dedup below.
            tname = call.tool_name or ""
            if tname.startswith("list_"):
                if tname in seen_read_tools:
                    return False
                seen_read_tools.add(tname)
            # All other (write-class) tools dedup by (name, args).
            key = _oracle_key(call.tool_name, call.arguments or {})
            if key in seen_oracle_keys:
                return False
            seen_oracle_keys.add(key)
            oracle_calls.append(call)
            return True

        if round_idx == 0:
            round_chain_len = len(chain_seed) if chain_seed else 3
        else:
            # Use actual remaining chain steps instead of a random number.
            # chain_progress_start is the cumulative steps done in prior rounds.
            # Remaining = total chain length - steps already done.
            # Fallback to 2 if no chain (matches PROVE §3.2 min_turns=2).
            if chain_seed:
                remaining = max(1, len(chain_seed) - chain_progress_start)
                round_chain_len = remaining
            else:
                round_chain_len = 2
        target_turns = ContinuationPolicy.target_turns(round_chain_len, local_rng)
        max_turns = min(target_turns + 2, 8)

        # PROVE limits oracle chains to at most five real tool calls.
        MAX_ORACLE_CALLS_PER_TASK = 5

        attempt = 0          # raw LLM call count (for temperature scaling)
        _turn: int = 0       # real turn count (tool exec + terminal)

        def _round_chain_progress() -> int:
            progress = chain_progress_start
            if chain_seed and progress < len(chain_seed):
                for previous in oracle_calls:
                    if previous.action != "tool_call":
                        continue
                    if progress < len(chain_seed) and previous.tool_name == chain_seed[progress]:
                        progress += 1
            elif not chain_seed:
                progress = chain_progress_start + sum(
                    1 for oc in oracle_calls if oc.action == "tool_call"
                )
            return progress

        while _turn < max_turns:
            # Progress means satisfying the seeded dependency chain in order,
            # not merely calling the same number of arbitrary unique tools.
            chain_progress = _round_chain_progress()

            # Stop emitting new oracle entries once the PROVE call budget is full.
            real_oracle_count = sum(1 for oc in oracle_calls if oc.action == "tool_call")
            if real_oracle_count >= MAX_ORACLE_CALLS_PER_TASK:
                break

            try:
                action = teacher.decide_action(
                    tool_schemas=server_tools,
                    user_query=current_query,
                    execution_history=prior_history + execution_history,
                    attempt=attempt,
                    dep_hints=dep_hints,
                    difficulty=difficulty,
                    chain_seed=chain_seed,
                    chain_progress=chain_progress,
                    reference_date=reference_date,
                    chain_context=chain_context,
                    blocked_tools=blocked_tools,
                    missing_function=missing_function_contract,
                )
            except RuntimeError:
                logger.debug(
                    f"_run_turn_loop: decide_action exhausted retries "
                    f"(chain_progress={chain_progress}/{len(chain_seed) if chain_seed else 0}), "
                    f"breaking turn loop."
                )
                break

            if action.action == "ask_clarification":
                _add_oracle(OracleCall(
                    tool_name="ask_clarification",
                    arguments={"question": action.text},
                    action="ask_clarification",
                ))
                oracle_observations.append({})
                break

            if action.action in ("final_answer", "report_error"):
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

            if _has_stale_year(action.arguments, reference_date):
                rejection = {
                    "error": f"Arguments use a year earlier than the reference date {reference_date}."
                }
                _record_attempt(tool_name, action.arguments, rejection, False, _owner_domain(tool_name))
                execution_history.append({
                    "tool_name": "__reject__",
                    "arguments": dict(action.arguments),
                    "observation": rejection,
                    "success": False,
                    "execution_status": "FAILURE",
                })
                _turn += 1
                attempt += 1
                continue

            # P0: missing-function — block hidden tools at execution layer.
            # Teacher may still output the tool name (it's in chain_seed hints),
            # but the executor must never run it.  Treat as a schema-unknown call
            # so the LLM sees an error and can produce ask_clarification.
            if blocked_tools and tool_name in blocked_tools:
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
                _turn += 1
                attempt += 1
                continue

            required_tools.add(tool_name)

            execution_domain = _owner_domain(tool_name)
            result = self.executor.execute(
                session_id,
                ToolCall(tool_name, dict(action.arguments), call_id=f"sm_{_turn}"),
                domain=execution_domain,
            )
            _record_attempt(
                tool_name, action.arguments, result.observation, result.success, execution_domain,
            )

            if not result.success:
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
                    break
                elif rec_action in ("retry", "retry_same"):
                    corrected = recovery.get("corrected_args", dict(action.arguments))
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

            if not ContinuationPolicy.should_continue(
                _turn, target_turns, result.success,
                sum(1 for oc in oracle_calls if oc.action == "tool_call"),
            ):
                break

        # A successful tool trace always has an explicit terminal contract.
        # This also covers turn-decay / budget exits where the teacher did not
        # get another generation turn to emit final_answer.
        if (any(oc.action == "tool_call" for oc in oracle_calls)
                and not any(oc.action in ("final_answer", "report_error", "ask_clarification")
                            for oc in oracle_calls)):
            _add_oracle(OracleCall(
                tool_name="final_answer",
                arguments={"text": "Task completed."},
                action="final_answer",
            ))
            oracle_observations.append({})

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

        # ── Conversation-level continuation (PROVE §3.2 Step 3.5) ──
        # PROVE §3.2 Step 3.5 bounds CONVERSATION ROUNDS (user turns) to 2..3.
        # This is unrelated to _run_turn_loop's max_turns (state-machine step
        # budget inside a single round) which stays at 8.
        # The training rollout consumes conversation_queries[1:] as live
        # follow-up messages.
        enable_continuation = getattr(self, 'enable_continuation', True)
        min_conversation_rounds = (
            ContinuationPolicy.MIN_CONVERSATION_ROUNDS
            if enable_continuation else 1
        )
        # max_conversation_rounds is sampled per-task inside the retry loop
        # (PROVE turn-decay schedule §3.2 Step 3.5).

        # ── Defensive initialisation (all variables reused after the retry loop).
        # Python guarantees range(3) iterates at least once, but Pylance cannot
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

        # ── Retry with different seed if LLM refuses to call tools ──
        for retry_attempt in range(3):
            local_seed = seed + retry_attempt * 1000
            local_rng = random.Random(local_seed)

            max_conversation_rounds = (
                ContinuationPolicy.conversation_rounds(local_rng)
                if enable_continuation else 1
            )

            teacher = TaskPlanner(self.client, server_name, seed=local_seed)

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

            # ── Build Teacher-visible tools (with knob set A, before enum stripping and distractor) ──
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
                teacher_grounding_state = _live_context_to_prompt_state(live_sampling_context)

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
                    # executable chains). After 3 retries, raise so the caller
                    # retries with a fresh seed — no unseeded fallback in baseline.
                    if retry_attempt < 2:
                        logger.debug(
                            f"No feasible chain for {server_name} "
                            f"(retry {retry_attempt + 1}/3), re-sampling session"
                        )
                        self.manager.close_session(session_id)
                        continue
                    raise RuntimeError(
                        f"No feasible chain for {server_name} after 3 retries; "
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
                # PROVE Step 1/3: the complete 2--5 step dependency chain seeds
                # one grounded task.  Continuation is a later interaction
                # mechanism; it must not be used to turn individual chain nodes
                # into separate user requests.
                query_goal_tool = source_chain_seed[-1] if source_chain_seed else ""
                query_generation_chain = source_chain_seed
                user_query = teacher.generate_query(
                    tool_schemas=server_tools,
                    grounded_state=teacher_grounding_state,
                    difficulty=difficulty,
                    rng=local_rng,
                    dep_hints=dep_hints,
                    persona=persona,
                    reference_date=reference_date,
                    chain_seed=query_generation_chain,
                    chain_context=query_chain_context,
                )
                query_intent_ok = (
                    not query_goal_tool
                    or _query_requires_hidden_capability(
                        user_query, query_goal_tool, server_name,
                    )
                )
                query_grounding_ok = (
                    not query_goal_tool
                    or _query_has_required_live_entity_grounding(
                        query=user_query,
                        tool_name=query_goal_tool,
                        domain=server_name,
                        difficulty=difficulty,
                        tool_schemas=server_tools,
                        live_context=query_chain_context,
                    )
                )
                if not query_grounding_ok:
                    logger.debug(
                        f"Initial query is not grounded for chain capability "
                        f"'{query_goal_tool}' in {server_name}; retrying: "
                        f"{user_query}"
                    )
                    continue
                if not query_intent_ok:
                    logger.debug(
                        f"Initial query lexical capability diagnostic failed for "
                        f"'{query_goal_tool}' in {server_name}; accepting because "
                        f"live grounding/execution/replay are authoritative: {user_query}"
                    )

                # Missing-function query generation must use the complete chain.
                # Only the Teacher execution contract receives the hidden version.
                blocked_tools_set: set[str] | None = None
                teacher_dep_hints = dep_hints
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
                        if not _query_satisfies_chain_capability(
                            query=user_query,
                            tool_name=hidden_tool,
                            domain=server_name,
                            difficulty=difficulty,
                            tool_schemas=server_tools,
                            live_context=query_chain_context,
                        ):
                            logger.debug(
                                f"Generated query does not require hidden capability "
                                f"'{hidden_tool}' for {server_name}; retrying: {user_query}"
                            )
                            continue
                        plan.hidden_tool = hidden_tool
                        teacher_visible_tools = _build_teacher_visible_tools(server_tools, plan)
                        blocked_tools_set = {hidden_tool}
                        teacher_dep_hints = _remove_tool_from_dependency_hints(
                            dep_hints, hidden_tool,
                        )
                        chain_seed = None
                        chain_context = {}

                # Accumulators across conversation rounds (PROVE CONTINUATION)
                # (re-assigned each retry; types declared before the loop)
                all_oracle_calls = []
                all_execution_history = []
                all_aligned_observations: list[Any] = []
                all_attempt_calls = []
                all_attempt_observations = []
                all_required_tools = set()
                conversation_queries = [user_query]  # track all user messages
                oracle_calls_per_round = []  # per-round for prompt construction
                execution_history_per_round = []
                task_id = f"{server_name}_{local_seed}_{local_rng.randint(0, 99999)}"
                retry_label = f" (retry {retry_attempt})" if retry_attempt > 0 else ""

                current_query = user_query

                logger.debug(
                    f"CONTINUATION: {server_name} task {task_id} "
                    f"starting state-machine continuation (min={min_conversation_rounds}, "
                    f"max={ContinuationPolicy.MAX_CONVERSATION_ROUNDS})"
                )

                round_idx = 0
                decision = "follow_up"  # dummy, overwritten on round_idx==0 path below
                current_round_goal_tool = query_goal_tool
                round_goal_failed = False
                while True:
                    if round_idx > 0:
                        followup_live_context = self._get_live_sampling_context(
                            session_id=session_id,
                            server_name=server_name,
                            server_tools=server_tools,
                            force_refresh=True,
                        )
                        followup_chain_progress = self._chain_progress_for_calls(all_oracle_calls, chain_seed)
                        if decision == "clarification":
                            current_round_goal_tool = (
                                chain_seed[followup_chain_progress]
                                if chain_seed and followup_chain_progress < len(chain_seed)
                                else ""
                            )
                            current_query = teacher.generate_clarification(
                                tool_schemas=teacher_visible_tools,
                                grounded_state=_live_context_to_prompt_state(followup_live_context),
                                previous_query=current_query,
                                difficulty=difficulty,
                                rng=local_rng,
                                persona=persona,
                                reference_date=reference_date,
                            )
                        else:
                            next_chain_tool = (
                                chain_seed[followup_chain_progress]
                                if chain_seed and followup_chain_progress < len(chain_seed)
                                else ""
                            )
                            current_round_goal_tool = next_chain_tool
                            for followup_attempt in range(3):
                                candidate_query = teacher.generate_followup(
                                    tool_schemas=teacher_visible_tools,
                                    grounded_state=_live_context_to_prompt_state(followup_live_context),
                                    previous_query=current_query,
                                    difficulty=difficulty,
                                    rng=local_rng,
                                    persona=persona,
                                    reference_date=reference_date,
                                    chain_seed=chain_seed,
                                    chain_progress=followup_chain_progress,
                                )
                                grounding_ok = (
                                    not next_chain_tool
                                    or _query_has_required_live_entity_grounding(
                                        query=candidate_query,
                                        tool_name=next_chain_tool,
                                        domain=server_name,
                                        difficulty=difficulty,
                                        tool_schemas=teacher_visible_tools,
                                        live_context=followup_live_context,
                                    )
                                )
                                if grounding_ok:
                                    current_query = candidate_query
                                    if (
                                        next_chain_tool
                                        and not _query_requires_hidden_capability(
                                            candidate_query, next_chain_tool, server_name,
                                        )
                                    ):
                                        logger.debug(
                                            f"Follow-up query lexical capability diagnostic "
                                            f"failed for '{next_chain_tool}' in {server_name}; "
                                            f"accepting grounded query: {candidate_query}"
                                        )
                                    break
                                logger.debug(
                                    f"Follow-up query does not express next chain "
                                    f"capability '{next_chain_tool}' for {server_name} "
                                    f"(attempt {followup_attempt + 1}/3): "
                                    f"{candidate_query}"
                                )
                            else:
                                raise RuntimeError(
                                    f"Failed to generate follow-up requiring "
                                    f"'{next_chain_tool}' for {server_name}"
                                )
                        conversation_queries.append(current_query)
                    else:
                        # round_idx == 0: first round, no decision yet
                        decision = "follow_up"  # dummy for the first iteration

                    current_chain_progress = self._chain_progress_for_calls(all_oracle_calls, chain_seed)

                    # The dependency chain is an atomic task seed, not a
                    # per-turn schedule.  The Teacher may execute the whole
                    # chain in this turn (subject to the global five-call cap).
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
                        dep_hints=teacher_dep_hints,
                        local_rng=local_rng,
                        chain_seed=chain_seed,
                        round_idx=round_idx,
                        reference_date=reference_date,
                        chain_progress_start=current_chain_progress,
                        max_calls_this_round=max_calls_r,
                        chain_context=chain_context,
                        blocked_tools=blocked_tools_set,
                        missing_function_contract=plan.missing_function,
                        prior_execution_history=all_execution_history,
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
                            if retry_attempt < 2:
                                logger.debug(
                                    f"No tool calls recorded for {server_name}{retry_label}, "
                                    f"retrying with new seed ({retry_attempt + 1}/3)"
                                )
                                break  # break conversation loop → continue retry loop
                            raise RuntimeError(
                                f"No tool calls recorded for {server_name} task {task_id} "
                                f"(LLM answered without using tools)"
                            )

                    # Cross-round dedup + total length cap to align with PROVE
                    # red lines.  _run_turn_loop dedups within a single round,
                    # but seen_read_tools / seen_oracle_keys reset between
                    # rounds, so a 4-round task can still emit list_invoices
                    # 4 times.  Apply the same rules globally here.
                    #
                    # Preserve the last round's calls as the supervised target
                    # while enforcing the global five-call oracle budget.
                    global_seen_read = {oc.tool_name for oc in all_oracle_calls
                                        if getattr(oc, "action", "tool_call") == "tool_call"
                                        and (oc.tool_name or "").startswith("list_")}
                    global_seen_keys = set()
                    for _oc in all_oracle_calls:
                        try:
                            _args_repr = json.dumps(_oc.arguments or {}, sort_keys=True, default=str, ensure_ascii=False)
                        except Exception:
                            _args_repr = repr(_oc.arguments)
                        global_seen_keys.add((_oc.tool_name, _args_repr))

                    is_last_round = (round_idx == max_conversation_rounds - 1)
                    real_so_far = sum(1 for oc in all_oracle_calls if getattr(oc, "action", "tool_call") == "tool_call")
                    filtered_round_ocs = []
                    filtered_round_obs = []
                    for i, oc in enumerate(round_ocs):
                        action = getattr(oc, "action", "tool_call")
                        if action != "tool_call":
                            filtered_round_ocs.append(oc)
                            filtered_round_obs.append(round_obs[i])
                            continue
                        if not is_last_round and real_so_far >= 5:
                            break
                        if is_last_round and real_so_far >= 5:
                            # PROVE hard cap: oracle chain ≤ 5 across ALL rounds.
                            break
                        tname = oc.tool_name or ""
                        if tname.startswith("list_"):
                            if tname in global_seen_read:
                                continue
                            global_seen_read.add(tname)
                        try:
                            args_repr = json.dumps(oc.arguments or {}, sort_keys=True, default=str, ensure_ascii=False)
                        except Exception:
                            args_repr = repr(oc.arguments)
                        key = (tname, args_repr)
                        if key in global_seen_keys:
                            continue
                        global_seen_keys.add(key)
                        filtered_round_ocs.append(oc)
                        filtered_round_obs.append(round_obs[i])
                        real_so_far += 1

                    all_oracle_calls.extend(filtered_round_ocs)
                    all_aligned_observations.extend(filtered_round_obs)
                    all_execution_history.extend(round_hist)
                    all_attempt_calls.extend(round_attempts)
                    all_attempt_observations.extend(round_attempt_obs)
                    all_required_tools |= round_reqs
                    oracle_calls_per_round.append(list(filtered_round_ocs))
                    execution_history_per_round.append(list(round_hist))

                    if not self._round_goal_satisfied(
                        filtered_round_ocs, current_round_goal_tool,
                    ):
                        logger.debug(
                            f"Round {round_idx} did not complete bound capability "
                            f"'{current_round_goal_tool}' for {server_name}; "
                            f"rejecting task instead of advancing continuation"
                        )
                        round_goal_failed = True
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

                    completed_rounds = round_idx + 1
                    completed_chain_progress = self._chain_progress_for_calls(
                        all_oracle_calls, chain_seed,
                    )
                    if (
                        chain_seed
                        and completed_chain_progress >= len(chain_seed)
                        and completed_rounds >= min_conversation_rounds
                    ):
                        # The seeded user goal has been completed. Generating an
                        # extra round here produces confirmation chatter or an
                        # unrelated request rather than useful continuation data.
                        break

                    # P1-3: per-turn continuation decision (PROVE §3.2 Step 3.5).
                    # Replaces the pre-sampled max_rounds approach with a true
                    # per-turn end / follow_up / clarification decision.
                    round_idx += 1
                    decision = ContinuationPolicy.sample_continuation_decision(
                        round_idx, local_rng,
                    )
                    if decision == "end":
                        break
                    # follow_up or clarification: continue loop

                if round_goal_failed:
                    continue

                # If we broke out of conversation loop early (first round failed)
                _real_now = [c for c in all_oracle_calls if getattr(c, "action", "tool_call") == "tool_call"]
                _clar_now = [c for c in all_oracle_calls if getattr(c, "action", "tool_call") == "ask_clarification"]
                _abstain_now = [
                    c for c in all_oracle_calls
                    if getattr(c, "action", "tool_call") in ("ask_clarification", "report_error")
                ]
                if plan.missing_function:
                    if _real_now or not _abstain_now:
                        self.manager.close_session(session_id)
                        continue
                    all_required_tools.clear()
                elif not _real_now and not (difficulty == "missing" and _clar_now):
                    self.manager.close_session(session_id)
                    continue  # retry loop

                if chain_seed and not plan.missing_function:
                    # Allow incomplete chain for "missing" difficulty (Teacher
                    # produced ask_clarification before calling all chain tools).
                    # missing_function perturbation also skips — zero tools expected.
                    if difficulty != "missing":
                        completed_chain_steps = self._chain_progress_for_calls(
                            all_oracle_calls, chain_seed,
                        )
                        if completed_chain_steps != len(chain_seed):
                            logger.debug(
                                f"Incomplete Teacher chain for {server_name} task "
                                f"{task_id}: completed {completed_chain_steps}/"
                                f"{len(chain_seed)}; retrying with a fresh seed"
                            )
                            continue

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
                    if retry_attempt < 2:
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
                    user_query="\n".join(conversation_queries),
                    aligned_observations=all_attempt_observations,
                )
                if not prov_ok:
                    if retry_attempt < 2:
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
                f"Teacher generation exhausted 3 retries for {server_name} "
                f"without a complete replay-valid dependency chain"
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
            if real_calls or not abstention_calls:
                raise RuntimeError(
                    f"Invalid missing-function oracle for {server_name} task {task_id}: "
                    f"real_calls={len(real_calls)} terminals={len(abstention_calls)}"
                )
        elif not real_calls and not (difficulty == "missing" and clarification_calls):
            raise RuntimeError(
                f"No real tool_call recorded for {server_name} task {task_id} "
                f"after 3 retries (LLM only produced clarifications/refusals)"
            )
        if real_calls and not (1 <= len(real_calls) <= 8):
            raise RuntimeError(
                f"Oracle chain length {len(real_calls)} outside required 1-8 "
                f"for {server_name} task {task_id}"
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
        # P0: visible_tools = actual Teacher-visible schemas (perturbed).
        # This ensures Parquet candidate set == RL rollout candidate set.
        live_task.visible_tools = final_teacher_visible_tools or live_task.visible_tools
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
            "chain_seed": [] if plan.missing_function else (list(chain_seed) if chain_seed else []),
            "source_chain_seed": list(source_chain_seed) if source_chain_seed else [],
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
            # P0: robustness contract — applied before Replay, not after
            "robustness_applied_before_replay": True,
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
        })
        return live_task

    def generate_many(self, server_name: str, count: int, seed: int,
                      difficulty_mix: dict[str, float] | None = None,
                      irrelevance_ratio: float = 0.05,
                      distractor_rate: float = 0.40,
                      missing_function_rate: float = 0.20,
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

        effective_mix = difficulty_mix or {"complete": 0.6, "missing": 0.2, "minimal": 0.2}

        # Pre-count irrelevance tasks (proportional, no forced minimum)
        n_irrelevant = round(count * irrelevance_ratio) if irrelevance_ratio > 0 else 0
        n_normal = count - n_irrelevant

        # Per-domain budget: each domain gets its fair share (PROVE uniform distribution)
        per_domain = n_normal // len(servers)
        remainder = n_normal % len(servers)
        global_seed_offset = 0
        failed = 0
        # Deduplicate by first user query string before sequence-level dedup,
        # since identical user requests are semantically duplicate tasks even
        # if later oracle details differ slightly.
        seen_queries: set[str] = set()
        dropped_dup_query = 0

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
        seen_lock = threading.Lock()

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures: dict[Any, tuple[str, int]] = {}
            for server_name, task_seed, difficulty in task_specs:
                fut = executor.submit(
                    self._generate_task_with_postprocess,
                    server_name, task_seed, difficulty,
                    distractor_rate, missing_function_rate,
                )
                futures[fut] = (server_name, task_seed)

            for fut in as_completed(futures):
                server_name, task_seed = futures[fut]

                # Skip domains that already hit quota or max failures
                if domain_ok[server_name] >= domain_quotas[server_name]:
                    continue
                if domain_failed_count[server_name] >= domain_max_failures[server_name]:
                    continue

                try:
                    task = fut.result()
                except Exception as e:
                    failed += 1
                    domain_failed_count[server_name] += 1
                    if pbar:
                        pbar.set_postfix_str(f"fail={failed}")
                    logger.warning(
                        f"generate failed for {server_name} "
                        f"(seed={task_seed}, {domain_failed_count[server_name]}x): {e}"
                    )
                    continue

                if task is None:
                    failed += 1
                    domain_failed_count[server_name] += 1
                    if pbar:
                        pbar.set_postfix_str(f"fail={failed}")
                    continue

                q_key = (task.user_prompt or "").strip().lower()
                with seen_lock:
                    if q_key and q_key in seen_queries:
                        dropped_dup_query += 1
                        logger.debug(
                            f"{server_name}: dropping duplicate query "
                            f"(seen #{dropped_dup_query}): {q_key[:80]}"
                        )
                        continue
                    if q_key:
                        seen_queries.add(q_key)

                tasks.append(task)
                domain_ok[server_name] += 1
                if pbar:
                    pbar.update(1)
                    pbar.set_postfix_str(f"fail={failed}")
                # Loguru progress: every task with elapsed time & completion %
                elapsed = time.time() - _gen_start
                pct = len(tasks) * 100.0 / n_normal if n_normal > 0 else 0
                if len(tasks) - _last_log >= 1:
                    _last_log = len(tasks)
                    logger.info(
                        f"[generate_many] {len(tasks)}/{n_normal} ({pct:.0f}%) "
                        f"| {failed} fail | elapsed={elapsed:.0f}s "
                        f"| rate={len(tasks)/elapsed:.2f} task/s"
                    )

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
            f"{removed} dedup removed, {dropped_dup_query} dup-query dropped)"
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
        from src.live_mcp.task_planner import replay_validate, provenance_check

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
                query = self._fallback_irrelevant_query(server_name, rng)

            oracle_calls = [OracleCall(
                tool_name="report_error",
                arguments={"text": "No available tool can satisfy this request."},
                action="report_error",
            )]

            # ── Replay + Provenance (PROVE §3.2 unified pipeline) ──
            # Zero tool-call oracle → num_calls=0, num_errors=0, passed=True.
            # provenance_check on zero tool calls is trivially OK.
            _valid, _err_rate, _n_err, n_calls, _criteria_ok, _criteria_failed = (
                replay_validate(
                    oracle_calls=oracle_calls,
                    manager=self.manager,
                    executor=self.executor,
                    seed=seed + i,
                    domain=server_name,
                    success_criteria=[],
                )
            )
            _prov_ok, _prov_violations = provenance_check(
                oracle_calls=oracle_calls,
                user_query=query,
                aligned_observations=[],
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
                    "generation_method": "irrelevant_template",
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
    def _tool_schema_hash(server_tools: list[dict]) -> str:
        """Stable schema hash for PROVE dependency-graph caching."""
        schema_payload = []
        for tool in sorted(server_tools, key=lambda t: str(t.get("name", ""))):
            schema_payload.append({
                "name": tool.get("name"),
                "description": tool.get("description", ""),
                "input_schema": tool.get("input_schema", {}),
                "annotations": tool.get("annotations", {}),
            })
        raw = json.dumps(schema_payload, sort_keys=True, ensure_ascii=True, default=str)
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
                and payload.get("classification_complete") is True
                and payload.get("expected_pair_count") == expected_pair_count
                and payload.get("classified_pair_count") == expected_pair_count
                and payload.get("deterministic_augmentation")
                    == self.ENABLE_DETERMINISTIC_GRAPH_AUGMENTATION
            )
            if isinstance(graph, dict) and cached_tool_names == expected_tool_names:
                graph = self._normalize_cached_graph(graph, expected_tool_names)
                _break_graph_cycles(graph)
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
        # The LLM-classified edges were already normalized and filtered inside
        # _classify_edges_llm. Optional deterministic augmentation, when enabled
        # for an ablation, is recorded explicitly in cache metadata.
        if not self._valid_cached_graph(graph, expected_tool_names):
            logger.warning(f"Skipping invalid dependency graph cache for {server_name}")
            return
        cache_path = self._graph_cache_path(server_name, schema_hash)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "cache_version": self.DEPENDENCY_CACHE_VERSION,
            "server_name": server_name,
            "schema_hash": schema_hash,
            "tool_names": expected_tool_names,
            "graph": graph,
            "tool_count": len(expected_tool_names),
            "expected_pair_count": len(expected_tool_names) * (len(expected_tool_names) - 1) // 2,
            "classified_pair_count": len(expected_tool_names) * (len(expected_tool_names) - 1) // 2,
            "classification_complete": True,
            "deterministic_augmentation": self.ENABLE_DETERMINISTIC_GRAPH_AUGMENTATION,
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
        every unordered tool pair as explicit, implicit, or none, with the
        classifier choosing the directed edge when a dependency exists.
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

        schema_hash = self._tool_schema_hash(server_tools)
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
                        f"refusing deterministic-only fallback"
                    )
                    failures[failure_key] = (time.monotonic(), failure_message)
                    raise RuntimeError(failure_message)

                det = (
                    _deterministic_schema_edges(server_tools, server_name)
                    if self.ENABLE_DETERMINISTIC_GRAPH_AUGMENTATION
                    else {}
                )
                for src, edge_info in det.items():
                    if src not in graph:
                        graph[src] = edge_info
                    else:
                        ex = set(graph[src].get("explicit", []))
                        im = set(graph[src].get("implicit", []))
                        graph[src] = {
                            "explicit": sorted(ex | set(edge_info.get("explicit", []))),
                            "implicit": sorted(
                                (im | set(edge_info.get("implicit", []))) - ex
                            ),
                        }

                # Pairwise decisions can still form cycles across batches.
                _break_graph_cycles(graph, det)
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

        Sends all unordered nC2 tool pairs to the LLM in batches. For each
        pair, the classifier chooses a single dependency direction or none,
        yielding a directed graph.

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
        # This remains PROVE-style pairwise LLM classification over all nC2
        # pairs; each request just carries the schemas needed for its batch.
        BATCH_SIZE = self.DEPENDENCY_PAIR_BATCH_SIZE
        all_classifications: dict[str, str] = {}  # "A → B" → "explicit"|"implicit"
        classified_pairs: set = set()

        all_pair_keys: set[tuple[str, str]] = {
            tuple(sorted((a, b))) for a, b in pairs
        }
        expected_pair_count = len(all_pair_keys)
        BATCH_RETRIES = 2

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
                    for part in re.split(r"\s*(?:↔|→)\s*", pair_text, maxsplit=1)
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
                pair_key = tuple(sorted(pair_members))
                if pair_key not in valid_pairs or pair_key in classified_pairs:
                    continue

                # A none relation is undirected. Some models correctly emit
                # source="none", target="none"; pair identifies the tools.
                if relation == "none":
                    classified_pairs.add(pair_key)
                    continue

                if source not in tool_desc_by_name or target not in tool_desc_by_name:
                    continue
                if source == target or {source, target} != set(pair_members):
                    continue
                classified_pairs.add(pair_key)
                all_classifications[f"{source} → {target}"] = relation

        for batch_start in range(0, len(pairs), BATCH_SIZE):
            batch_pairs = pairs[batch_start:batch_start + BATCH_SIZE]
            valid_batch_pairs = {tuple(sorted(pair)) for pair in batch_pairs}

            system = (
                "You are analyzing tool dependencies for an MCP server. "
                "For each unordered tool pair {A, B}, choose at most ONE directed "
                "dependency edge:\n"
                '- "explicit": source produces output that is a REQUIRED INPUT of target '
                "(e.g., source returns an entity ID that target needs as a parameter).\n"
                '- "implicit": source must execute BEFORE target to establish state, '
                "but source's output is not a direct input to target.\n"
                '- "none": neither direction is a dependency.\n\n'
                "Classification rules:\n"
                "- If one tool creates/returns something that the other tool's "
                "required parameters reference, choose that direction and mark explicit.\n"
                "- Mark implicit only when source establishes live server state that target "
                "cannot succeed without, such as create_draft → send_draft, "
                "add_to_cart → checkout, or schedule_transfer → cancel_transfer.\n"
                "- Prefer explicit over implicit when both could apply.\n"
                "- Direction matters: if B is merely a later read, verification, "
                "history lookup, or same-entity follow-up after A, mark none unless "
                "B truly requires state created by A.\n"
                "- Read-after-write/audit-after-write is usually none: "
                "pay_invoice → get_invoice, dispute_invoice → get_invoice, "
                "bill_pay → get_history, mark_read → get_email, and "
                "add_attendee → list_events are none unless B requires a new ID "
                "that only A created.\n"
                "- Same entity type alone is NOT a dependency.\n"
                "- Only mark implicit if there is a genuine required state dependency."
            )
            batch_complete = False
            for batch_attempt in range(BATCH_RETRIES + 1):
                # Ask only for pairs that are still missing. Re-sending the full
                # batch makes deterministic teachers repeat the same prefix and
                # never fill truncated/omitted tail entries.
                pending_pairs = [
                    pair for pair in batch_pairs
                    if tuple(sorted(pair)) not in classified_pairs
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
                    f"{i + 1}. {a_name} ↔ {b_name}"
                    for i, (a_name, b_name) in enumerate(pending_pairs)
                )
                user = (
                    f"## Server: {server_name}\n\n"
                    f"## Tools\n{pending_tools_text}\n\n"
                    f"## Pairs to Classify\n{pending_pairs_text}\n\n"
                    f"Classify every listed pair exactly once. Do not omit any pair.\n\n"
                    f"## Output Format\n"
                    f'{{"classifications": [\n'
                    f'  {{"pair": "tool_a ↔ tool_b", "source": "tool_a", "target": "tool_b", "relation": "explicit"}},\n'
                    f'  {{"pair": "tool_c ↔ tool_d", "source": "tool_d", "target": "tool_c", "relation": "implicit"}},\n'
                    f'  {{"pair": "tool_e ↔ tool_f", "source": "tool_e", "target": "tool_f", "relation": "none"}}\n'
                    f']}}\n\n'
                    f"Output ONLY the JSON, nothing else:"
                )
                try:
                    raw = self.client.generate_chat(
                        [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
                        temperature=0.1 + 0.05 * batch_attempt,
                        max_tokens=2048,
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
        # Expected: n(n-1)/2 unordered pairs, each must get exactly one
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

        self._apply_prove_dependency_definition_filter(graph, server_tools, server_name)
        return graph

    @staticmethod
    def _apply_prove_dependency_definition_filter(
        graph: dict[str, dict],
        server_tools: list[dict],
        server_name: str,
    ) -> None:
        """Enforce PROVE's explicit/implicit dependency definitions.

        The LLM classifier proposes edges, but a same-entity read after a write
        is not a dependency unless the read requires an entity that the write
        created. Likewise, a target with no required state/input cannot depend
        on a prior tool under PROVE's definitions.
        """
        tool_by_name = {str(tool.get("name") or ""): tool for tool in server_tools}

        def is_mutating(tool_name: str) -> bool:
            annotations = tool_by_name.get(tool_name, {}).get("annotations") or {}
            return bool(annotations.get("mutating")) and not bool(annotations.get("readonly"))

        def is_readonly(tool_name: str) -> bool:
            annotations = tool_by_name.get(tool_name, {}).get("annotations") or {}
            return bool(annotations.get("readonly")) and not bool(annotations.get("mutating"))

        for source, edge_groups in graph.items():
            source_created = _CREATED_ENTITY_BY_TOOL.get(source.lower(), set())
            source_relevant = _tool_relevant_entity_types(source, server_name)
            for relation in ("explicit", "implicit"):
                kept: list[str] = []
                for target in edge_groups.get(relation, []):
                    target_requirements = _tool_existing_entity_requirements(target, server_name)
                    state_edge = (server_name, source, target) in _PROVE_STATE_DEPENDENCY_EDGES

                    # State-dependency edges bypass entity checks entirely
                    if state_edge:
                        kept.append(target)
                        continue

                    if not target_requirements:
                        continue

                    source_satisfies_target = bool(source_created & target_requirements)
                    observable_read_satisfies_target = (
                        is_readonly(source)
                        and bool(source_relevant & target_requirements)
                    )

                    if not (source_satisfies_target or observable_read_satisfies_target or state_edge):
                        continue

                    if is_mutating(source) and is_readonly(target) and not source_satisfies_target:
                        continue

                    kept.append(target)
                edge_groups[relation] = kept

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
        id_to_summary: dict[str, str] = {
            str(entity_ids[i].get("id", "")): entity_summaries[i]
            for i in range(min(len(entity_ids), len(entity_summaries)))
        }
        qualified_summaries: list[str] = [
            id_to_summary.get(str(q.get("id", "")), "")
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


def _has_stale_year(arguments: dict[str, Any], reference_date: str) -> bool:
    import re

    match = re.search(r"\b(20\d{2})\b", reference_date or "")
    if not match:
        return False
    reference_year = int(match.group(1))
    raw = json.dumps(arguments, ensure_ascii=False, default=str)
    return any(int(year) < reference_year for year in re.findall(r"\b(20\d{2})[-/]", raw))


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


def _tool_annotations(task: LiveTask, tool_name: str) -> dict[str, Any]:
    for tool in task.visible_tools:
        if tool.get("name") == tool_name:
            annotations = tool.get("annotations") or {}
            return annotations if isinstance(annotations, dict) else {}
    return {}


def _query_has_read_intent(query: str) -> bool:
    q = f" {query.lower()} "
    return any(f" {marker} " in q or marker in q for marker in _READ_INTENT_MARKERS)


def _query_has_write_intent_for_tool(query: str, tool_name: str) -> bool:
    q = f" {query.lower().replace('_', ' ')} "
    name = tool_name.lower()
    for prefixes, markers in _WRITE_INTENT_BY_PREFIX:
        if any(name.startswith(prefix) for prefix in prefixes):
            return any(f" {marker} " in q or marker in q for marker in markers)

    # Unknown mutating tool shape.  Fall back to direct token overlap with the
    # tool name so explicit requests such as "run foo_bar" are still eligible.
    tool_tokens = [part for part in name.replace("_", " ").split() if len(part) > 2]
    return bool(tool_tokens and all(token in q for token in tool_tokens))


def _missing_function_candidate_is_semantically_required(
    task: LiveTask,
    hidden_tool: str,
) -> bool:
    """Return whether hiding hidden_tool creates a valid abstention task.

    PROVE-style missing_function samples must be impossible because a required
    tool is absent.  They must not turn a read-only request into report_error
    just because the teacher trajectory happened to include an unrelated write
    tool later in the conversation.
    """
    if not hidden_tool:
        return False

    oracle_calls = getattr(task.oracle_program, "calls", None) or []
    if hidden_tool not in {
        call.tool_name for call in oracle_calls
        if getattr(call, "action", "tool_call") == "tool_call"
    }:
        return False

    annotations = _tool_annotations(task, hidden_tool)
    is_mutating = bool(annotations.get("mutating")) and not bool(annotations.get("readonly"))
    if not is_mutating:
        return True

    # Check ALL conversation queries, not just the first round.
    # In multi-round tasks, the write operation may only appear in a later
    # query (e.g., round 1 is read-only, round 2 requests a payment).
    queries_text: str = " ".join([
        q for q in ([task.user_prompt or ""] + list(task.conversation_queries or []))
        if q
    ])

    if _query_has_read_intent(queries_text) and not _query_has_write_intent_for_tool(queries_text, hidden_tool):
        return False

    return _query_has_write_intent_for_tool(queries_text, hidden_tool)


def _remove_tool_from_dependency_hints(dep_hints: str, hidden_tool: str) -> str:
    """Remove every dependency-hint line that exposes a hidden tool name."""
    if not dep_hints or not hidden_tool:
        return dep_hints
    return "\n".join(
        line for line in dep_hints.splitlines()
        if hidden_tool not in line
    )


def _query_requires_hidden_capability(
    query: str,
    hidden_tool: str,
    domain: str = "",
) -> bool:
    """Conservative lexical gate that proves the query asks for hidden capability."""
    from src.live_mcp.task_planner import _chain_goal_phrase

    q = query.lower().replace("_", " ")
    name = hidden_tool.lower()
    phrase = _chain_goal_phrase(name).lower()
    entity_tokens = [
        token for token in (name.replace("_", " ") + " " + phrase).split()
        if len(token) > 3 and token not in {
            "create", "update", "delete", "remove", "search", "list",
            "check", "complete", "place", "existing", "details", "from",
            "with", "into", "account", "calendar",
        }
    ]
    entity_ok = any(
        re.search(r'\b' + re.escape(token.rstrip("s")) + r'\b', q)
        for token in entity_tokens
    )
    entity = _tool_entity(name, domain)
    entity_aliases = {
        "event": (
            "event", "call", "meeting", "appointment", "invite",
            "reservation", "session", "review",
        ),
        "order": ("order", "purchase", "delivery"),
        "email": ("email", "mail", "message", "thread"),
        "account": ("account", "card", "checking", "savings"),
        "invoice": ("invoice", "bill", "payment"),
        "issue": ("issue", "bug", "ticket", "task"),
        "file": ("file", "folder", "directory", "path"),
        "channel": ("channel", "room", "chat"),
    }
    if entity in entity_aliases:
        entity_ok = entity_ok or any(alias in q for alias in entity_aliases[entity])

    # Command-style filesystem tools do not follow verb_object naming, so the
    # generic prefix gate below cannot recognize ordinary user wording.
    if domain == "filesystem":
        command_markers: dict[str, tuple[str, ...]] = {
            "ls": ("list", "show", "what", "inside", "contents"),
            "find": ("find", "search", "locate"),
            "mkdir": ("create", "make", "new folder", "new directory"),
            "touch": ("create", "make", "empty file", "new file"),
            "cat": ("show", "read", "display", "content"),
            "head": ("show", "read", "first", "top"),
            "tail": ("show", "read", "last", "bottom"),
            "sort": ("sort", "alphabetize", "order"),
            "join": ("join", "merge", "combine"),
            "readlink": ("link", "point", "target", "where"),
            "grep": ("grep", "find", "search", "match"),
            "pwd": ("current directory", "working directory", "where am i"),
        }
        markers = command_markers.get(name)
        if markers is not None:
            return any(marker in q for marker in markers)

    action_groups: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
        (("create_",), (
            "create", "make", "start", "set up", "place", "schedule", "book", "plan",
        )),
        (("update_", "set_", "assign_", "transition_"), ("update", "change", "set", "assign", "move")),
        (("delete_", "remove_", "clear_"), ("delete", "remove", "clear")),
        (("cancel_",), ("cancel", "stop")),
        (("add_",), ("add", "include", "put")),
        (("send_", "reply_", "forward_"), ("send", "reply", "forward")),
        (("pay_",), ("pay", "settle")),
        (("refund_", "return_"), ("refund", "return")),
        (("get_", "list_", "search_", "find_"), (
            "get", "show", "list", "find", "search", "check", "what", "which",
            "when", "next", "give me", "tell me", "details",
        )),
    )
    action_ok = False
    for prefixes, markers in action_groups:
        if any(name.startswith(prefix) for prefix in prefixes):
            action_ok = any(marker in q for marker in markers)
            break
    if name == "checkout":
        action_ok = any(marker in q for marker in ("checkout", "place", "buy", "order", "purchase"))
        entity_ok = "cart" in q or "order" in q or "item" in q
    elif not action_ok and phrase:
        action_ok = any(
            token in q for token in phrase.split()
            if len(token) > 4
        )
    if name.startswith(("get_", "list_", "search_", "find_")):
        # Read requests can identify the object colloquially ("that call",
        # "my order") without repeating the schema's entity noun.
        return bool(action_ok)
    return bool(action_ok and entity_ok)


def _query_satisfies_chain_capability(
    *,
    query: str,
    tool_name: str,
    domain: str,
    difficulty: str,
    tool_schemas: list[dict[str, Any]],
    live_context: dict[str, Any],
) -> bool:
    """Check action intent and complete-level live-ID grounding for one round."""
    if not _query_requires_hidden_capability(query, tool_name, domain):
        return False

    return _query_has_required_live_entity_grounding(
        query=query,
        tool_name=tool_name,
        domain=domain,
        difficulty=difficulty,
        tool_schemas=tool_schemas,
        live_context=live_context,
    )


def _query_has_required_live_entity_grounding(
    *,
    query: str,
    tool_name: str,
    domain: str,
    difficulty: str,
    tool_schemas: list[dict[str, Any]],
    live_context: dict[str, Any],
) -> bool:
    """Enforce only PROVE-relevant live-ID grounding, independent of wording."""

    schema = next(
        (tool for tool in tool_schemas if tool.get("name") == tool_name),
        {},
    )
    required = list(schema.get("input_schema", {}).get("required", []) or [])

    # Detect entity-reference fields: _id suffix OR field names that match
    # known entity types for this tool's domain (e.g., "from_account" in banking,
    # "product_ids" in shopping, "invoice_id" cross-domain).
    entity_types = _tool_relevant_entity_types(tool_name, domain)
    entity_patterns: set[str] = set()
    for et in entity_types:
        entity_patterns.update({et, et + "s", et + "_id"})
    requires_entity_ref = any(
        str(field).endswith("_id")
        or any(pattern in str(field).lower() for pattern in entity_patterns)
        for field in required
    )
    if not requires_entity_ref:
        return True  # tool has no entity-reference params → gate passes

    if difficulty != "complete":
        # Non-complete tasks: still verify basic intent (done above).
        # Don't enforce live-ID grounding since these tasks may use
        # vague entity references by design.
        return True

    live_entity_ids = (
        live_context.get("qualified_entity_ids")
        if "qualified_entity_ids" in live_context
        else live_context.get("entity_ids")
    ) or []
    known_ids = [
        str(item.get("id") or "")
        for item in live_entity_ids
        if isinstance(item, dict) and item.get("id")
    ]
    return any(entity_id in query for entity_id in known_ids)


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

    if terminal_action == "ask_clarification" and not real_calls_only:
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
    "add_to_wishlist": {"wishlist"},
    "checkout": {"order"},
    "create_order": {"order"},
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
        "remove_from_wishlist": {"wishlist"},
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
    "banking": {
        "account": lambda r: (
            float(r.get("balance", 0)) > 0 and not bool(r.get("frozen", False)),
            (
                "account frozen"
                if r.get("frozen")
                else "balance is zero"
            ),
        ),
    },
    "shopping": {
        "product": lambda r: (
            bool(r.get("stock", r.get("available", r.get("in_stock", True)))) if any(
                k in r for k in ("stock", "available", "in_stock")
            ) else True,
            "product out of stock or unavailable",
        ),
    },
    "food_delivery": {
        "restaurant": lambda r: (
            bool(r.get("menu", r.get("items"))),
            "restaurant menu is empty",
        ),
    },
}


_PROVE_STATE_DEPENDENCY_EDGES: set[tuple[str, str, str]] = {
    ("banking", "freeze_account", "unfreeze_account"),
    ("calendar", "add_attendee", "remove_attendee"),
    ("payments", "pay_invoice", "refund_invoice"),
    ("shopping", "add_to_wishlist", "remove_from_wishlist"),
    ("issue_tracker", "add_label", "remove_label"),
    ("issue_tracker", "add_watcher", "remove_watcher"),
    # issue_tracker: sprint state ops on the same issue
    ("issue_tracker", "add_to_sprint", "remove_from_sprint"),
    # issue_tracker: time_track creates time_entry state; get_time_report reads
    # it but has no required entity input so the filter would drop the edge.
    ("issue_tracker", "time_track", "get_time_report"),
    # email: draft is a state precondition for send_email; send_email's
    # required inputs are to/subject/body (no draft_id), so this is implicit.
    ("email", "create_draft", "send_email"),
    # shopping: coupon → cart flow. get_coupons outputs `code` needed by
    # apply_coupon; apply_coupon establishes cart discount state needed by
    # checkout. Neither pair matches the entity table since apply_coupon
    # doesn't create a cart_item and checkout requires cart_item.
    ("shopping", "get_coupons", "apply_coupon"),
    ("shopping", "apply_coupon", "checkout"),
    # filesystem: read→compute chains that the entity filter incorrectly prunes
    # because computational tools (grep/sed/awk) have file entity requirements
    # but the source_relevant computation doesn't propagate correctly in all paths
    ("filesystem", "ls", "cat"),
    ("filesystem", "find", "cat"),
    ("filesystem", "cat", "grep"),
    ("filesystem", "cat", "sed"),
    ("filesystem", "cat", "awk"),
    ("filesystem", "touch", "cat"),
    ("filesystem", "cd", "ls"),
    ("filesystem", "cd", "pwd"),
}


_DOMAIN_TOOL_RELEVANT: dict[str, dict[str, set[str]]] = {
    "calendar": {
        "list_events": {"event"},
        "search_events": {"event"},
        "get_free_busy": {"event"},
        "check_conflicts": {"event"},
    },
    "banking": {
        "list_accounts": {"account"},
        "list_transactions": {"account"},
        "get_exchange_rate": {"account"},
        "apply_loan": {"account"},
    },
    "payments": {
        "get_invoice": {"invoice", "payment"},
        "list_invoices": {"invoice", "payment"},
        "list_webhooks": {"webhook"},
    },
    "email": {
        "list_inbox": {"email"},
        "search_emails": {"email"},
        "list_threads": {"thread"},
        "list_drafts": {"draft"},
        "create_draft": {"email", "draft"},
    },
    "filesystem": {
        "pwd": {"file"},
        "ls": {"file"},
        "find": {"file"},
        "tree": {"file"},
        "du": {"file"},
        "df": {"file"},
        "sort": {"file"},
        "uniq": {"file"},
        "cut": {"file"},
        "sed": {"file"},
        "awk": {"file"},
        "split": {"file"},
        "diff": {"file"},
        "readlink": {"file"},
    },
    "crm": {
        "list_leads": {"lead"},
        "list_contacts": {"contact"},
        "list_deals": {"deal"},
        "list_tasks": {"task"},
        "search_contacts": {"contact"},
        "create_deal": {"lead", "contact", "deal"},
        "create_task": {"deal", "contact", "task"},
        "add_note": {"lead", "contact", "deal", "note"},
    },
    "issue_tracker": {
        "list_issues": {"issue"},
        "search_issues": {"issue"},
        "list_sprints": {"sprint"},
    },
    "shopping": {
        "search_products": {"product"},
        "list_categories": {"product"},
        "get_coupons": {"product"},
        "apply_coupon": {"cart_item"},
        "get_cart": {"cart_item", "product"},
        "get_wishlist": {"wishlist", "product"},
    },
    "team_chat": {
        "list_channels": {"channel"},
        "get_user_status": {"user"},
        "search_messages": {"channel", "message"},
    },
    "food_delivery": {
        "search_restaurants": {"restaurant"},
        "list_orders": {"order"},
    },
}


_DISCOVERY_TOOL_PREFIXES = (
    "list_",
    "search_",
    "get_free_busy",
    "check_conflicts",
    "get_working_hours",
    "change_timezone",
    "export_calendar",
    "get_exchange_rate",
    "list_categories",
    "get_coupons",
    "apply_coupon",
    "get_wishlist",
    "get_cart",
    "get_time_report",
    "get_user_status",
    "pwd",
    "ls",
    "cat",
    "stat",
    "head",
    "tail",
    "find",
    "grep",
    "tree",
    "du",
    "df",
    "file_info",
    "md5sum",
    "sha256sum",
    "wc",
    "xxd",
    "sort",
    "uniq",
    "cut",
    "sed",
    "awk",
    "split",
    "diff",
    "readlink",
)


_ENTITY_ID_FIELD_TYPES: dict[str, str] = {
    "event_id": "event",
    "account_id": "account",
    "from_account": "account",
    "to_account": "account",
    "invoice_id": "invoice",
    "payment_id": "payment",
    "webhook_id": "webhook",
    "refund_id": "refund",
    "dispute_id": "dispute",
    "email_id": "email",
    "thread_id": "thread",
    "draft_id": "draft",
    "filter_id": "filter",
    "lead_id": "lead",
    "contact_id": "contact",
    "deal_id": "deal",
    "task_id": "task",
    "note_id": "note",
    "issue_id": "issue",
    "sprint_id": "sprint",
    "subtask_id": "subtask",
    "entry_id": "time_entry",
    "restaurant_id": "restaurant",
    "order_id": "order",
    "return_id": "return",
    "product_id": "product",
    "channel_id": "channel",
    "message_id": "message",
    "dm_id": "dm",
    "ticket_id": "ticket",
    "transfer_id": "scheduled_transfer",
    "scheduled_txn_id": "scheduled_transfer",
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

    if tool.startswith(_DISCOVERY_TOOL_PREFIXES):
        return set()
    if server:
        domain_requirements = _DOMAIN_TOOL_REQUIREMENTS.get(server, {})
        if tool in domain_requirements:
            requirements = set(domain_requirements[tool])
            created = set(_CREATED_ENTITY_BY_TOOL.get(tool, set()))
            if server == "filesystem" and tool in {"cp", "mv"}:
                created.discard("file")
            requirements.difference_update(created)
            return requirements
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
    created_types: set[str] = set()
    for idx, tool_name in enumerate(chain):
        tool = tool_name.lower()
        requirements = _tool_existing_entity_requirements(tool, server_name)
        # PROVE §3.2 Step 2: chain is feasible if live_context already has
        # the required entities (probed via read-only discovery tools).
        # A chain can start with a mutating tool (e.g., update_event) as
        # long as the entity exists in the live state.
        missing = [
            etype for etype in sorted(requirements)
            if not _live_context_has_entity_type(live_context, etype, created_types)
        ]
        if missing:
            return False, f"{tool} requires missing entity types {missing}"
        created_types.update(_CREATED_ENTITY_BY_TOOL.get(tool, set()))
    return True, "ok"


def _break_graph_cycles(
    graph: dict[str, dict],
    deterministic: dict[str, dict[str, list[str]]] | None = None,
) -> None:
    """Remove edges that create cycles, making the graph a DAG.

    LLM pairwise classification can introduce bidirectional edges
    (e.g. ls↔cd, mark_read↔mark_unread) that form 2-cycles, and
    longer chains that loop back.  This function detects and breaks
    all cycles by removing the weakest edge from each.

    Strategy:
      1. For bidirectional pairs (A↔B): when *deterministic* is provided,
         prefer keeping deterministic edges over non-deterministic ones.
         When neither or both directions are deterministic, fall back to
         keeping the direction from the node with *fewer* total outgoing
         edges (spoke→hub is more plausible than hub→spoke).
      2. For longer cycles, iteratively run Kahn's topological sort
         and remove one back edge at a time.
    """
    # ── Step 1: break bidirectional pairs ──
    # Collect all edges
    edges: set[tuple[str, str]] = set()
    for src, node in graph.items():
        for rel in ("explicit", "implicit"):
            for tgt in node.get(rel, []):
                if tgt in graph:
                    edges.add((src, tgt))

    # Build deterministic edge set for bidirectional preference
    det_edges: set[tuple[str, str]] = set()
    if deterministic:
        for src, edge_info in deterministic.items():
            for tgt in edge_info.get("explicit", []):
                if src in graph and tgt in graph:
                    det_edges.add((src, tgt))
            for tgt in edge_info.get("implicit", []):
                if src in graph and tgt in graph:
                    det_edges.add((src, tgt))

    # Find bidirectional pairs
    bidir_pairs: set[tuple[str, str]] = set()  # (a, b) with a < b
    for a, b in edges:
        if (b, a) in edges:
            bidir_pairs.add((min(a, b), max(a, b)))

    for a, b in bidir_pairs:
        # Preference order:
        # 1. One direction is deterministic → keep that direction
        # 2. Both or neither deterministic → keep direction with fewer outgoing edges
        a_to_b_det: bool = (a, b) in det_edges
        b_to_a_det: bool = (b, a) in det_edges
        out_a = sum(
            len(graph[a].get(rel, [])) for rel in ("explicit", "implicit")
        )
        out_b = sum(
            len(graph[b].get(rel, [])) for rel in ("explicit", "implicit")
        )

        if a_to_b_det and not b_to_a_det:
            # Keep a → b (deterministic), remove b → a
            _remove_edge(graph, b, a)
        elif b_to_a_det and not a_to_b_det:
            # Keep b → a (deterministic), remove a → b
            _remove_edge(graph, a, b)
        elif out_a <= out_b:
            # Neither or both deterministic: keep a → b, remove b → a
            _remove_edge(graph, b, a)
        else:
            # Keep b → a, remove a → b
            _remove_edge(graph, a, b)

    # ── Step 2: break longer cycles via Kahn's algorithm ──
    for _ in range(20):  # safety limit
        in_degree: dict[str, int] = {n: 0 for n in graph}
        adj: dict[str, list[str]] = {n: [] for n in graph}
        for src, node in graph.items():
            for rel in ("explicit", "implicit"):
                for tgt in node.get(rel, []):
                    if tgt in in_degree:
                        in_degree[tgt] += 1
                        adj[src].append(tgt)

        # Kahn's topological sort
        queue = [n for n, d in in_degree.items() if d == 0]
        visited = 0
        while queue:
            n = queue.pop(0)
            visited += 1
            for nb in adj[n]:
                in_degree[nb] -= 1
                if in_degree[nb] == 0:
                    queue.append(nb)

        if visited >= len(graph):
            break  # DAG achieved

        # Find a back edge in the remaining cycle
        remaining = [n for n in graph if in_degree.get(n, 0) > 0]
        for n in remaining:
            for rel in ("implicit", "explicit"):  # prefer removing implicit
                targets = list(graph[n].get(rel, []))
                for tgt in targets:
                    if tgt in remaining:
                        _remove_edge(graph, n, tgt)
                        break
                else:
                    continue
                break
            else:
                continue
            break


def _remove_edge(graph: dict[str, dict], src: str, tgt: str) -> None:
    """Remove a directed edge from src to tgt."""
    for rel in ("explicit", "implicit"):
        if tgt in graph.get(src, {}).get(rel, []):
            graph[src][rel] = [x for x in graph[src][rel] if x != tgt]
            return


def _chain_respects_state_preconditions(server_name: str, chain: list[str]) -> bool:
    if server_name == "shopping":
        # cart_item state can be established by an explicit add_to_cart, or
        # observed via readonly get_cart, or acted on by apply_coupon — any of
        # these earlier in the chain means the cart is non-empty for
        # subsequent cart-consuming ops.
        cart_state_producers = {"add_to_cart", "get_cart", "apply_coupon"}
        cart_tools = {"checkout", "update_cart_quantity", "remove_from_cart", "clear_cart"}
        for cart_tool in cart_tools:
            if cart_tool in chain:
                target_idx = chain.index(cart_tool)
                if not any(p in chain[:target_idx] for p in cart_state_producers):
                    return False

    if server_name == "payments":
        if "create_invoice" in chain and "refund_invoice" in chain:
            create_idx = chain.index("create_invoice")
            refund_idx = chain.index("refund_invoice")
            if create_idx < refund_idx and "pay_invoice" not in chain[create_idx + 1:refund_idx]:
                return False
        # cancel_payment is the reversal of an *initiated* payment, so it
        # must appear *after* a pay_invoice (or list/get_invoice observing
        # an already-paid invoice). It cannot appear before pay_invoice, nor
        # right after create_invoice without any payment initiation.
        if "cancel_payment" in chain:
            cancel_idx = chain.index("cancel_payment")
            pay_producers = {"pay_invoice"}
            if not any(p in chain[:cancel_idx] for p in pay_producers):
                # No pay_invoice earlier → the chain reads as observation-only
                # for cancel_payment; require at least list_invoices or
                # get_invoice earlier to plausibly reference an existing
                # payment id.
                observers = {"list_invoices", "get_invoice"}
                if not any(o in chain[:cancel_idx] for o in observers):
                    return False

    if server_name == "crm":
        if "complete_task" in chain:
            complete_idx = chain.index("complete_task")
            before = chain[:complete_idx]
            if "create_task" not in before and "list_tasks" not in before:
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
        if "cancel_order" in chain:
            cancel_idx = chain.index("cancel_order")
            if "create_order" not in chain[:cancel_idx]:
                return False
        if "track_rider" in chain:
            track_idx = chain.index("track_rider")
            if "create_order" not in chain[:track_idx]:
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

    relevant_types: set[str] = set()
    for tool_name in chain_seed:
        relevant_types.update(_tool_relevant_entity_types(tool_name, server_name))
    if server_name == "payments" and "cancel_payment" not in chain_seed:
        relevant_types.discard("payment")

    entity_ids: list[dict] = []
    entity_summaries: list[str] = []
    entity_records: list[dict[str, Any]] = []
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
        entity_summaries.append(summaries_by_key.get(key, f"  {eid} ({etype})"))
        entity_records.append({"id": eid, "type": etype, "data": record})
    return {
        "entity_ids": entity_ids[:30],  # cap to avoid prompt overflow
        "entity_summaries": entity_summaries[:30],
        "entity_records": entity_records[:30],
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
        if "cancel_payment" in tools and str(record.get("payment_status", "")) != "pending":
            return False
        pay_before_refund = (
            "pay_invoice" in tools
            and "refund_invoice" in tools
            and chain_seed.index("pay_invoice") < chain_seed.index("refund_invoice")
        )
        if "pay_invoice" in tools and status in {"paid", "refunded", "partially_refunded"}:
            return False
        if "refund_invoice" in tools and not pay_before_refund:
            if status not in {"paid", "partially_refunded"}:
                return False
    if server_name == "payments" and etype == "payment":
        if "cancel_payment" in tools and str(record.get("status", "")) != "pending":
            return False
    if server_name == "food_delivery" and etype == "order":
        status = str(record.get("status", ""))
        update_before_track = (
            "update_order_status" in tools
            and "track_rider" in tools
            and chain_seed.index("update_order_status") < chain_seed.index("track_rider")
        )
        if "rate_order" in tools and status != "delivered":
            return False
        if "track_rider" in tools and not update_before_track and status != "delivering":
            return False
        if "cancel_order" in tools and status not in {"placed", "confirmed"}:
            return False
        if "update_order_status" in tools and status not in {"placed", "confirmed", "preparing", "delivering"}:
            return False
        if "add_tip" in tools and bool(record.get("tip", 0)):
            return False
    if server_name == "shopping" and etype == "order":
        status = str(record.get("status", ""))
        if "return_order" in tools and status in {"returning", "returned"}:
            return False
    return True


def _deterministic_schema_edges(
    server_tools: list[dict],
    server_name: str,
) -> dict[str, dict[str, list[str]]]:
    """Schema-based deterministic dependency edges (PROVE §3.2 Step 1 pre-pass).

    LLM pairwise classification can miss obvious read-before-write edges.
    This function injects known dependencies deterministically, ensuring
    critical chains (get_lead→convert_lead, get_cart→checkout, etc.) are
    never lost.
    """
    graph: dict[str, dict[str, list[str]]] = {}
    names = set(t["name"] for t in server_tools)

    def _add_explicit(src: str, dst: str) -> None:
        if src in names and dst in names and src != dst:
            node = graph.setdefault(src, {"explicit": [], "implicit": []})
            if dst not in node["explicit"]:
                node["explicit"].append(dst)

    # Universal read-before-write patterns: get_E / list_Es / search_Es → {update_E, delete_E, ...}
    _WRITE_SUFFIXES = {
        "_event": ["update_event", "delete_event", "cancel_event"],
        "_order": ["cancel_order", "return_order", "reorder"],
        "_account": ["transfer", "wire_transfer", "bill_pay", "withdraw"],
        "_invoice": ["pay_invoice", "refund_invoice", "cancel_payment"],
        "_product": ["add_to_cart"],
        "_issue": ["transition_issue", "update_issue", "assign_issue"],
        "_lead": ["convert_lead", "update_lead"],
        "_deal": ["update_deal"],
        "_channel": ["archive_channel", "send_message"],
        "_email": ["reply_email", "forward_email", "archive_email"],
        "_contact": ["update_contact", "delete_contact"],
        "_message": ["react_message", "create_thread"],
        "_task": ["complete_task"],
    }
    for t in server_tools:
        tname = t["name"]
        entity_name: str | None = None
        for prefix in ("get_", "list_", "search_", "create_"):
            if tname.startswith(prefix):
                entity_name = tname[len(prefix):]
                break
        if entity_name is None:
            continue
        # Normalize plural forms: list_events→event, list_categories→category
        normalized = entity_name
        if entity_name.endswith("ies"):
            normalized = entity_name[:-3] + "y"
        elif entity_name.endswith("s") and not entity_name.endswith("ss"):
            normalized = entity_name[:-1]
        for suffix, targets in _WRITE_SUFFIXES.items():
            entity_type = suffix[1:]  # strip leading "_"
            if normalized == entity_type or entity_name == entity_type:
                for wt in targets:
                    # For create_* tools, skip self-referential edges like create_event→update_event
                    # when the target starts with the same prefix
                    if tname.startswith("create_") and wt.startswith("create_"):
                        continue
                    _add_explicit(tname, wt)
                break

    # Domain-specific compound edges.
    #
    # All edges here are derived from ground-truth input schemas (see
    # /tmp/tools_schema.json). Each edge satisfies at least one of:
    #   * explicit: source output contains an id that is a required input of target
    #   * implicit: source establishes an entity/state that target requires
    # Edges that would violate _apply_prove_dependency_definition_filter are
    # not added here; state-only edges go into _PROVE_STATE_DEPENDENCY_EDGES.
    _EDGES: dict[str, list[tuple[str, str]]] = {
        "shopping": [
            ("search_products", "get_product"),
            ("search_products", "get_reviews"),
            ("search_products", "add_review"),
            ("search_products", "compare_products"),
            ("search_products", "add_to_cart"),
            ("search_products", "add_to_wishlist"),
            ("get_product", "get_reviews"),
            ("get_product", "add_review"),
            ("get_product", "compare_products"),
            ("get_product", "add_to_cart"),
            ("get_product", "add_to_wishlist"),
            ("get_cart", "checkout"),
            ("get_cart", "clear_cart"),
            ("get_cart", "remove_from_cart"),
            ("get_cart", "update_cart_quantity"),
            ("add_to_cart", "update_cart_quantity"),
            ("add_to_cart", "remove_from_cart"),
            ("add_to_cart", "checkout"),
            ("checkout", "get_order"),
            ("checkout", "track_order"),
            ("checkout", "return_order"),
            ("list_orders", "get_order"),
            ("list_orders", "track_order"),
            ("list_orders", "return_order"),
            ("get_order", "track_order"),
            ("get_order", "return_order"),
            ("return_order", "get_return_status"),
            ("get_wishlist", "remove_from_wishlist"),
            # State-dependency edges (whitelisted in _PROVE_STATE_DEPENDENCY_EDGES):
            # coupon → cart flow. get_coupons outputs `code` consumed by
            # apply_coupon; apply_coupon establishes cart discount state that
            # checkout depends on.
            ("get_coupons", "apply_coupon"),
            ("apply_coupon", "checkout"),
        ],
        "calendar": [
            ("search_events", "get_event"),
            ("search_events", "update_event"),
            ("search_events", "delete_event"),
            ("search_events", "add_attendee"),
            ("search_events", "remove_attendee"),
            ("search_events", "set_reminder"),
            ("search_events", "respond_to_event"),
            ("search_events", "get_recurring_info"),
            ("list_events", "get_event"),
            ("list_events", "get_recurring_info"),
            ("get_event", "update_event"),
            ("get_event", "delete_event"),
            ("get_event", "add_attendee"),
            ("get_event", "remove_attendee"),
            ("get_event", "set_reminder"),
            ("get_event", "respond_to_event"),
            ("get_event", "get_recurring_info"),
            ("create_event", "get_event"),
            ("create_event", "update_event"),
            ("create_event", "delete_event"),
            ("create_event", "add_attendee"),
            ("create_event", "set_reminder"),
            ("create_event", "respond_to_event"),
            ("create_recurring", "get_event"),
            ("create_recurring", "get_recurring_info"),
            ("create_recurring", "add_attendee"),
            ("create_recurring", "set_reminder"),
            ("create_recurring", "respond_to_event"),
            ("create_recurring", "update_event"),
            ("create_recurring", "delete_event"),
            ("add_attendee", "remove_attendee"),
        ],
        "email": [
            ("list_inbox", "get_email"),
            ("list_inbox", "reply_email"),
            ("list_inbox", "forward_email"),
            ("list_inbox", "archive_email"),
            ("list_inbox", "mark_read"),
            ("list_inbox", "mark_unread"),
            ("list_inbox", "add_label"),
            ("list_inbox", "remove_label"),
            ("list_inbox", "get_attachments"),
            ("list_inbox", "move_to_thread"),
            ("search_emails", "get_email"),
            ("search_emails", "reply_email"),
            ("search_emails", "forward_email"),
            ("search_emails", "archive_email"),
            ("search_emails", "mark_read"),
            ("search_emails", "mark_unread"),
            ("search_emails", "add_label"),
            ("search_emails", "remove_label"),
            ("search_emails", "get_attachments"),
            ("search_emails", "get_thread"),
            ("search_emails", "move_to_thread"),
            ("get_email", "reply_email"),
            ("get_email", "forward_email"),
            ("get_email", "archive_email"),
            ("get_email", "mark_read"),
            ("get_email", "mark_unread"),
            ("get_email", "add_label"),
            ("get_email", "remove_label"),
            ("get_email", "get_attachments"),
            ("get_email", "move_to_thread"),
            ("add_label", "remove_label"),
            ("mark_unread", "mark_read"),
            ("mark_read", "mark_unread"),
            # State-dependency edge (whitelisted):
            # draft is a precondition for send_email even though send_email's
            # required schema args are to/subject/body (no draft_id).
            ("create_draft", "send_email"),
        ],
        "crm": [
            ("list_leads", "update_lead"),
            ("list_leads", "convert_lead"),
            ("list_leads", "delete_lead"),
            ("list_leads", "add_note"),
            ("list_deals", "update_deal"),
            ("list_deals", "get_deal"),
            ("list_deals", "add_note"),
            ("list_deals", "create_task"),
            ("list_tasks", "complete_task"),
            ("get_deal", "update_deal"),
            ("get_deal", "add_note"),
            ("get_deal", "create_task"),
            ("create_lead", "convert_lead"),
            ("create_lead", "update_lead"),
            ("create_lead", "add_note"),
            ("create_contact", "add_note"),
            ("create_contact", "create_deal"),
            ("create_contact", "update_contact"),
            ("create_contact", "delete_contact"),
            ("create_deal", "add_note"),
            ("create_deal", "update_deal"),
            ("create_deal", "create_task"),
            ("create_task", "complete_task"),
            ("convert_lead", "create_deal"),
            ("convert_lead", "add_note"),
            ("convert_lead", "update_contact"),
            ("convert_lead", "delete_contact"),
        ],
        "issue_tracker": [
            ("search_issues", "get_issue"),
            ("search_issues", "update_issue"),
            ("search_issues", "assign_issue"),
            ("search_issues", "transition_issue"),
            ("search_issues", "comment_issue"),
            ("search_issues", "add_label"),
            ("search_issues", "remove_label"),
            ("search_issues", "add_watcher"),
            ("search_issues", "remove_watcher"),
            ("search_issues", "add_to_sprint"),
            ("search_issues", "remove_from_sprint"),
            ("search_issues", "create_subtask"),
            ("search_issues", "list_subtasks"),
            ("search_issues", "time_track"),
            ("search_issues", "set_milestone"),
            ("list_issues", "get_issue"),
            ("list_issues", "update_issue"),
            ("list_issues", "assign_issue"),
            ("list_issues", "transition_issue"),
            ("list_issues", "comment_issue"),
            ("list_issues", "add_label"),
            ("list_issues", "add_watcher"),
            ("list_issues", "add_to_sprint"),
            ("list_issues", "create_subtask"),
            ("list_issues", "list_subtasks"),
            ("list_issues", "time_track"),
            ("list_issues", "set_milestone"),
            ("get_issue", "update_issue"),
            ("get_issue", "assign_issue"),
            ("get_issue", "transition_issue"),
            ("get_issue", "comment_issue"),
            ("get_issue", "add_label"),
            ("get_issue", "add_watcher"),
            ("get_issue", "add_to_sprint"),
            ("get_issue", "create_subtask"),
            ("get_issue", "list_subtasks"),
            ("get_issue", "time_track"),
            ("get_issue", "set_milestone"),
            ("create_issue", "get_issue"),
            ("create_issue", "assign_issue"),
            ("create_issue", "transition_issue"),
            ("create_issue", "comment_issue"),
            ("create_issue", "add_label"),
            ("create_issue", "add_watcher"),
            ("create_issue", "add_to_sprint"),
            ("create_issue", "create_subtask"),
            ("create_issue", "list_subtasks"),
            ("create_issue", "time_track"),
            ("create_issue", "set_milestone"),
            ("create_issue", "update_issue"),
            ("create_subtask", "list_subtasks"),
            ("list_sprints", "add_to_sprint"),
            ("create_sprint", "add_to_sprint"),
            # State-dependency edges (whitelisted):
            # sprint membership state, and time-tracking state feeding reports.
            ("add_to_sprint", "remove_from_sprint"),
            ("time_track", "get_time_report"),
        ],
        "filesystem": [
            # ls / find / tree → any readonly analyzer that needs a path
            ("ls", "cat"), ("ls", "head"), ("ls", "tail"), ("ls", "wc"),
            ("ls", "stat"), ("ls", "file_info"), ("ls", "md5sum"),
            ("ls", "sha256sum"), ("ls", "xxd"), ("ls", "sort"), ("ls", "uniq"),
            ("ls", "cut"), ("ls", "sed"), ("ls", "awk"), ("ls", "split"),
            ("ls", "readlink"), ("ls", "chmod"), ("ls", "chown"),
            ("ls", "mv"), ("ls", "cp"), ("ls", "rm"), ("ls", "cd"),
            ("ls", "truncate"), ("ls", "tar_create"), ("ls", "zip"),
            ("find", "cat"), ("find", "head"), ("find", "tail"), ("find", "wc"),
            ("find", "stat"), ("find", "file_info"), ("find", "md5sum"),
            ("find", "sha256sum"), ("find", "xxd"), ("find", "sort"),
            ("find", "uniq"), ("find", "cut"), ("find", "sed"), ("find", "awk"),
            ("find", "split"), ("find", "readlink"), ("find", "chmod"),
            ("find", "chown"), ("find", "mv"), ("find", "cp"), ("find", "rm"),
            ("find", "truncate"), ("find", "tar_create"), ("find", "zip"),
            ("ls", "symlink"), ("find", "symlink"),
            ("ls", "join"), ("find", "join"),
            ("tree", "cat"), ("tree", "cd"), ("tree", "ls"), ("tree", "rm"),
            ("tree", "du"),
            # cd / pwd navigation
            ("cd", "ls"), ("cd", "pwd"), ("cd", "find"), ("cd", "tree"),
            # cat / head / tail / stat as read-followed-by-write on the same file
            ("cat", "rm"), ("cat", "chmod"), ("cat", "chown"), ("cat", "mv"),
            ("cat", "cp"), ("cat", "grep"), ("cat", "sed"), ("cat", "awk"),
            ("cat", "wc"),
            ("stat", "chmod"), ("stat", "chown"), ("stat", "rm"), ("stat", "mv"),
            ("head", "wc"), ("tail", "wc"),
            # writer → observer on the created path
            ("touch", "cat"), ("touch", "head"), ("touch", "tail"),
            ("touch", "wc"), ("touch", "stat"), ("touch", "file_info"),
            ("touch", "chmod"), ("touch", "chown"), ("touch", "rm"),
            ("touch", "truncate"), ("touch", "md5sum"), ("touch", "sha256sum"),
            ("mkdir", "ls"), ("mkdir", "cd"), ("mkdir", "touch"),
            ("mkdir", "chmod"), ("mkdir", "chown"), ("mkdir", "rm"),
            ("mkdir", "cp"), ("mkdir", "mv"),
            ("cp", "ls"), ("cp", "cat"), ("cp", "stat"), ("cp", "rm"),
            ("cp", "md5sum"), ("cp", "sha256sum"), ("cp", "diff"),
            ("mv", "ls"), ("mv", "cat"), ("mv", "stat"), ("mv", "rm"),
            ("mv", "chmod"), ("mv", "chown"),
            # archive lifecycle
            ("tar_create", "tar_extract"), ("tar_create", "md5sum"),
            ("tar_create", "sha256sum"), ("tar_create", "rm"),
            ("tar_create", "stat"),
            ("zip", "unzip"), ("zip", "md5sum"), ("zip", "sha256sum"),
            ("zip", "rm"), ("zip", "stat"),
            # compute pipeline
            ("sort", "uniq"), ("sort", "head"), ("sort", "tail"), ("sort", "wc"),
            ("split", "ls"), ("split", "cat"), ("split", "wc"),
            ("readlink", "cat"), ("readlink", "stat"), ("readlink", "ls"),
            # misc
            ("du", "rm"),
        ],
        "banking": [
            ("list_accounts", "get_account_info"),
            ("list_accounts", "get_balance"),
            ("list_accounts", "get_history"),
            ("list_accounts", "get_statement"),
            ("list_accounts", "verify_account"),
            ("list_accounts", "deposit"),
            ("list_accounts", "withdraw"),
            ("list_accounts", "transfer"),
            ("list_accounts", "wire_transfer"),
            ("list_accounts", "bill_pay"),
            ("list_accounts", "schedule_transfer"),
            ("list_accounts", "freeze_account"),
            ("list_accounts", "apply_loan"),
            ("get_account_info", "get_balance"),
            ("get_account_info", "get_history"),
            ("get_account_info", "get_statement"),
            ("get_account_info", "verify_account"),
            ("get_account_info", "deposit"),
            ("get_account_info", "withdraw"),
            ("get_account_info", "transfer"),
            ("get_account_info", "bill_pay"),
            ("get_account_info", "freeze_account"),
            ("get_balance", "withdraw"),
            ("get_balance", "transfer"),
            ("get_balance", "bill_pay"),
            ("verify_account", "wire_transfer"),
            ("verify_account", "transfer"),
            ("schedule_transfer", "cancel_transfer"),
            ("freeze_account", "unfreeze_account"),
        ],
        "payments": [
            ("list_invoices", "get_invoice"),
            ("list_invoices", "pay_invoice"),
            ("list_invoices", "refund_invoice"),
            ("list_invoices", "dispute_invoice"),
            ("get_invoice", "pay_invoice"),
            ("get_invoice", "refund_invoice"),
            ("get_invoice", "dispute_invoice"),
            ("create_invoice", "get_invoice"),
            ("create_invoice", "pay_invoice"),
            ("create_invoice", "refund_invoice"),
            ("create_invoice", "dispute_invoice"),
            ("pay_invoice", "refund_invoice"),
            ("pay_invoice", "cancel_payment"),
            ("list_webhooks", "delete_webhook"),
            ("create_webhook", "delete_webhook"),
        ],
        "team_chat": [
            ("list_channels", "get_channel"),
            ("list_channels", "archive_channel"),
            ("list_channels", "send_message"),
            ("list_channels", "search_messages"),
            ("get_channel", "send_message"),
            ("get_channel", "archive_channel"),
            ("create_channel", "get_channel"),
            ("create_channel", "send_message"),
            ("create_channel", "archive_channel"),
            ("search_messages", "react_message"),
            ("search_messages", "create_thread"),
            ("send_message", "react_message"),
            ("send_message", "create_thread"),
            ("create_thread", "get_thread"),
        ],
        "food_delivery": [
            ("search_restaurants", "get_restaurant"),
            ("search_restaurants", "get_menu"),
            ("search_restaurants", "filter_by_dietary"),
            ("search_restaurants", "get_popular_items"),
            ("search_restaurants", "create_order"),
            ("list_restaurants", "get_restaurant"),
            ("list_restaurants", "get_menu"),
            ("list_restaurants", "create_order"),
            ("get_restaurant", "get_menu"),
            ("get_restaurant", "filter_by_dietary"),
            ("get_restaurant", "get_popular_items"),
            ("get_restaurant", "create_order"),
            ("get_menu", "create_order"),
            ("get_popular_items", "create_order"),
            ("filter_by_dietary", "create_order"),
            ("create_order", "get_order"),
            ("create_order", "get_estimated_time"),
            ("create_order", "track_rider"),
            ("create_order", "cancel_order"),
            ("create_order", "add_tip"),
            ("create_order", "rate_order"),
            ("create_order", "contact_support"),
            ("create_order", "update_order_status"),
            ("list_orders", "get_order"),
            ("list_orders", "cancel_order"),
            ("list_orders", "reorder"),
            ("list_orders", "rate_order"),
            ("list_orders", "contact_support"),
            ("list_orders", "update_order_status"),
            ("list_orders", "track_rider"),
            ("list_orders", "get_estimated_time"),
            ("list_orders", "add_tip"),
            ("get_order", "cancel_order"),
            ("get_order", "add_tip"),
            ("get_order", "rate_order"),
            ("get_order", "track_rider"),
            ("get_order", "get_estimated_time"),
            ("get_order", "reorder"),
            ("get_order", "contact_support"),
            ("get_order", "update_order_status"),
        ],
    }
    for src, dst in _EDGES.get(server_name, []):
        _add_explicit(src, dst)

    return graph
