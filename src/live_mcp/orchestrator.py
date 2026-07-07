"""PROVE-style state-machine task generation.

Per environment:
  1. Auto-discover tool dependency graph via live MCP probing
  2. State machine alternating LLM decisions and tool execution
     against a live MCP server
  3. Replay-validate each conversation before conversion
"""

from __future__ import annotations

import copy
import hashlib
import json
import random
from pathlib import Path
from typing import Any

from loguru import logger

from src.live_mcp.config import SuiteConfig
from src.live_mcp.dedup import dedup_tasks
from src.live_mcp.executor import LiveMCPExecutor
from src.live_mcp.manager import LiveMCPManager
from src.live_mcp.types import LiveTask, OracleCall, OracleProgram, to_plain
from src.utils import extract_json as _extract_json


class TaskOrchestrator:
    """PROVE-style state-machine task generator.

    1. Auto-discover dependency graph (cached per domain)
    2. State machine: query generation → tool execution → continuation decisions
       (LLM-in-the-loop at every turn, against live MCP server)
    3. Replay-validate against fresh session
    4. Robustness knobs applied post-generation

    Usage:
        client = LLMClient(mode="openai", model_path="Qwen3-32B", api_base="...")
        orch = TaskOrchestrator(suite_config, manager, executor, client)
        tasks = orch.generate_many("all", count=100, seed=42)
    """

    # PROVE §3.2 Step 2: refresh sampling context every k conversations.
    # k=10 balances freshness (state changes after writes) vs. probe overhead.
    SAMPLING_CONTEXT_REFRESH_K: int = 10

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
        dep_hints: str,
        local_rng: random.Random,
        chain_seed: list[str] | None,
        round_idx: int,
        reference_date: str = "",
        chain_progress_start: int = 0,
        max_calls_this_round: int = 0,
        chain_context: dict[str, Any] | None = None,
    ) -> tuple[list, list[dict], set[str]]:
        """Run one conversation round of teacher-driven tool execution.

        chain_progress_start: cumulative chain_seed steps satisfied in previous
        rounds. Used for cross-round chain enforcement (PROVE continuation).

        max_calls_this_round: if > 0, stop after this many real tool_calls so
        the chain execution spans multiple conversation rounds (PROVE §3.2
        min_turns=2 multi-round schedule). 0 = no limit (single-round).

        chain_context: live-probed entity values for hallucination prevention
        in decide_action. Extracted from _extract_chain_context in generate_one.

        Returns (oracle_calls, execution_history, required_tools).
        """
        from src.live_mcp.task_planner import ContinuationPolicy, apply_perturbation
        from src.live_mcp.types import ToolCall

        oracle_calls: list = []
        execution_history: list[dict[str, Any]] = []
        required_tools: set[str] = set()

        # For multi-round continuation rounds (round_idx > 0), signal to
        # decide_action that this is NOT a first turn.  Without this,
        # decide_action applies its "first turn" guidance, which allows
        # ask_clarification.  The LLM may then produce ask_clarification
        # instead of continuing the chain, producing an oracle trace with
        # too few tool calls.  MARKER: seed-entity avoids __reject__ checks.
        if round_idx > 0:
            execution_history.append({
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
                    execution_history=execution_history,
                    attempt=attempt,
                    dep_hints=dep_hints,
                    difficulty=difficulty,
                    chain_seed=chain_seed,
                    chain_progress=chain_progress,
                    reference_date=reference_date,
                    chain_context=chain_context,
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
                break

            if action.action in ("final_answer", "report_error"):
                _add_oracle(OracleCall(
                    tool_name=action.action,
                    arguments={"text": action.text},
                    action=action.action,
                ))
                break

            if action.action != "tool_call" or not action.tool_name:
                continue

            tool_name = action.tool_name
            tool_name = _fuzzy_match_tool(tool_name, {t["name"] for t in server_tools}) or tool_name

            if _has_stale_year(action.arguments, reference_date):
                execution_history.append({
                    "tool_name": "__reject__",
                    "arguments": dict(action.arguments),
                    "observation": {
                        "error": f"Arguments use a year earlier than the reference date {reference_date}."
                    },
                    "success": False,
                    "execution_status": "FAILURE",
                })
                _turn += 1
                attempt += 1
                continue
            required_tools.add(tool_name)

            result = self.executor.execute(
                session_id,
                ToolCall(tool_name, dict(action.arguments), call_id=f"sm_{_turn}"),
                domain=server_name,
            )

            # Perturbations that modify the observation payload (pagination,
            # incomplete_intermediate, partial_batch_failure) are safe on both
            # read and write calls — they alter the returned data shape but not
            # the underlying state.  Only the "retry" perturbation (intermittent
            # error) is blocked on writes because it would trigger a re-execution
            # after state already changed.
            perturbed_obs = apply_perturbation(result.observation, server_name, local_rng)
            if result.success and isinstance(perturbed_obs, dict) and perturbed_obs.get("retry") and result.state_changed:
                perturbed_obs = None

            if isinstance(perturbed_obs, dict) and perturbed_obs.get("retry"):
                execution_history.append({
                    "tool_name": tool_name,
                    "arguments": dict(action.arguments),
                    "observation": perturbed_obs,
                    "success": False,
                })
                # Retry triggered: don't add failed call to oracle.
                # If the next turn succeeds, the success path will add it once.
                _turn += 1
                attempt += 1
                continue

            if not result.success:
                logger.debug(
                    f"_run_turn_loop: tool '{tool_name}' execution failed for "
                    f"{server_name} (error_type={result.error_type}, "
                    f"msg={result.error_message[:80]})"
                )
                execution_history.append({
                    "tool_name": tool_name,
                    "arguments": dict(action.arguments),
                    "observation": perturbed_obs if perturbed_obs is not None else result.observation,
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
                    error_observation=perturbed_obs if perturbed_obs is not None else {"error": str(result.observation)},
                    tool_schemas=server_tools,
                    execution_history=execution_history,
                )
                rec_action = recovery.get("action", "give_up")

                if rec_action == "give_up":
                    break
                elif rec_action in ("retry", "retry_same"):
                    corrected = recovery.get("corrected_args", dict(action.arguments))
                    retry_result = self.executor.execute(
                        session_id,
                        ToolCall(tool_name, corrected, call_id=f"sm_recover_{_turn}"),
                        domain=server_name,
                    )
                    if retry_result.success:
                        execution_history.append({
                            "tool_name": tool_name,
                            "arguments": corrected,
                            "observation": retry_result.observation if retry_result.observation is not None else {},
                            "success": True,
                            "execution_status": retry_result.execution_status,
                        })
                        _add_oracle(OracleCall(
                            tool_name=tool_name,
                            arguments=corrected,
                        ))
                elif rec_action == "retry_alt":
                    alt_tool = recovery.get("tool_name", "")
                    if alt_tool and alt_tool in {t["name"] for t in server_tools}:
                        alt_result = self.executor.execute(
                            session_id,
                            ToolCall(alt_tool, recovery.get("arguments", {}), call_id=f"sm_alt_{_turn}"),
                            domain=server_name,
                        )
                        if alt_result.success:
                            required_tools.add(alt_tool)
                            execution_history.append({
                                "tool_name": alt_tool,
                                "arguments": recovery.get("arguments", {}),
                                "observation": alt_result.observation if alt_result.observation is not None else {},
                                "success": True,
                                "execution_status": alt_result.execution_status,
                            })
                            _add_oracle(OracleCall(
                                tool_name=alt_tool,
                                arguments=recovery.get("arguments", {}),
                            ))
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

            obs_to_record = perturbed_obs if perturbed_obs is not None else result.observation
            execution_history.append({
                "tool_name": tool_name,
                "arguments": dict(action.arguments),
                "observation": obs_to_record if obs_to_record is not None else {},
                "success": True,
                "execution_status": result.execution_status,
            })

            _add_oracle(OracleCall(
                tool_name=tool_name,
                arguments=dict(action.arguments),
            ))

            _turn += 1
            attempt += 1

            # PROVE §3.2 multi-round schedule: limit tool calls per round so
            # the dependency chain spans multiple user turns (min_turns=2).
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

        return oracle_calls, execution_history, required_tools

    def generate_one(
        self,
        server_name: str,
        seed: int,
        difficulty: str,
        max_turns: int = 8,
    ) -> LiveTask:
        """PROVE-style state-machine generation with LLM-in-the-loop.

        1. Sample dependency chain seed (PROVE §6 step 2)
        2. LLM generates user_query with persona + reference_date (PROVE §4)
        3. Turn-decay loop (min_turns≈chain_len, max_turns≈chain_len+2)
           LLM decides next action → execute → apply perturbation → recovery → record
        4. Derive success criteria from state delta
        5. Replay validate against fresh session

        Retries with different seed if oracle_calls is empty or replay fails.
        """
        from src.live_mcp.task_planner import (
            TaskPlanner, derive_success_criteria, derive_progress_predicates,
            replay_validate, apply_perturbation,
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
        # PROVE uses a continuation decision module bounded by max_turns=3.
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

                # ── Chain-aligned entity context (PROVE §3.2 Step 2) ──
                # Extract real entity IDs from live state that are relevant to the
                # sampled chain.  This constrains the teacher LLM to use only
                # grounded IDs, preventing hallucination.
                chain_context = {}
                if chain_seed:
                    chain_context = _extract_chain_context(
                        chain_seed=chain_seed,
                        server_name=server_name,
                        live_context=live_sampling_context,
                    )

                user_query = teacher.generate_query(
                    tool_schemas=server_tools,
                    grounded_state=teacher_grounding_state,
                    difficulty=difficulty,
                    rng=local_rng,
                    dep_hints=dep_hints,
                    persona=persona,
                    reference_date=reference_date,
                    chain_seed=chain_seed,
                    chain_context=chain_context,
                )
                if chain_seed:
                    query_ok, query_reason = _query_aligns_with_chain(
                        user_query=user_query,
                        chain_seed=chain_seed,
                        is_mutating_tool=_is_mutating_tool,
                    )
                    if not query_ok:
                        logger.debug(
                            f"Query-chain mismatch for {server_name}: {query_reason}; "
                            f"retrying generate_one with next seed"
                        )
                        self.manager.close_session(session_id)
                        continue

                # Accumulators across conversation rounds (PROVE CONTINUATION)
                # (re-assigned each retry; types declared before the loop)
                all_oracle_calls = []
                all_execution_history = []
                all_required_tools = set()
                conversation_queries = [user_query]  # track all user messages
                oracle_calls_per_round = []  # per-round for prompt construction
                execution_history_per_round = []
                task_id = f"{server_name}_{local_seed}_{local_rng.randint(0, 99999)}"
                retry_label = f" (retry {retry_attempt})" if retry_attempt > 0 else ""

                current_query = user_query

                logger.debug(
                    f"CONTINUATION: {server_name} task {task_id} "
                    f"starting state-machine continuation with max "
                    f"{max_conversation_rounds} round(s)"
                )

                for round_idx in range(max_conversation_rounds):
                    if round_idx > 0:
                        logger.debug(
                            f"CONTINUATION: {server_name} round {round_idx + 1}/{max_conversation_rounds} "
                            f"generating follow-up query"
                        )
                        followup_live_context = self._get_live_sampling_context(
                            session_id=session_id,
                            server_name=server_name,
                            server_tools=server_tools,
                        )
                        followup_chain_progress = self._chain_progress_for_calls(all_oracle_calls, chain_seed)
                        current_query = teacher.generate_followup(
                            tool_schemas=server_tools,
                            grounded_state=_live_context_to_prompt_state(followup_live_context),
                            previous_query=current_query,
                            difficulty=difficulty,
                            rng=local_rng,
                            persona=persona,
                            reference_date=reference_date,
                            chain_seed=chain_seed,
                            chain_progress=followup_chain_progress,
                        )
                        conversation_queries.append(current_query)

                    current_chain_progress = self._chain_progress_for_calls(all_oracle_calls, chain_seed)

                    # PROVE §3.2 multi-round schedule: distribute the dependency
                    # chain across conversation rounds (min_turns=2, max_turns=3).
                    # Each round gets ceil(remaining_chain / remaining_rounds)
                    # tool calls so the chain naturally spans multiple rounds.
                    if chain_seed and max_conversation_rounds > 1:
                        remaining_rounds = max_conversation_rounds - round_idx
                        remaining_chain = len(chain_seed) - current_chain_progress
                        if remaining_rounds > 1 and remaining_chain > 1:
                            max_calls_r = max(1, int(
                                (remaining_chain + remaining_rounds - 1) // remaining_rounds
                            ))
                        else:
                            max_calls_r = 0  # last round — no limit
                    else:
                        max_calls_r = 0

                    round_ocs, round_hist, round_reqs = self._run_turn_loop(
                        teacher=teacher,
                        current_query=current_query,
                        server_tools=server_tools,
                        server_name=server_name,
                        session_id=session_id,
                        difficulty=difficulty,
                        dep_hints=dep_hints,
                        local_rng=local_rng,
                        chain_seed=chain_seed,
                        round_idx=round_idx,
                        reference_date=reference_date,
                        chain_progress_start=current_chain_progress,
                        max_calls_this_round=max_calls_r,
                        chain_context=chain_context,
                    )

                    if round_idx == 0:
                        _real_round = [c for c in round_ocs if getattr(c, "action", "tool_call") == "tool_call"]
                        _clar_round = [c for c in round_ocs if getattr(c, "action", "tool_call") == "ask_clarification"]
                        if not _real_round and not (difficulty == "missing" and _clar_round):
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
                    for oc in round_ocs:
                        action = getattr(oc, "action", "tool_call")
                        if action != "tool_call":
                            filtered_round_ocs.append(oc)
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
                        real_so_far += 1

                    all_oracle_calls.extend(filtered_round_ocs)
                    all_execution_history.extend(round_hist)
                    all_required_tools |= round_reqs
                    oracle_calls_per_round.append(list(filtered_round_ocs))
                    execution_history_per_round.append(list(round_hist))

                    # Only break the conversation on ask_clarification (genuinely
                    # stuck — no tool_call possible).  final_answer / report_error
                    # are legitimate round terminals; the continuation decision is
                    # made by should_continue_conversation below (PROVE §3.2 Step 3.5).
                    if any(
                        getattr(oc, "action", "tool_call") == "ask_clarification"
                        for oc in filtered_round_ocs
                    ):
                        break

                    chain_after_round = self._chain_progress_for_calls(all_oracle_calls, chain_seed)
                    if not ContinuationPolicy.should_continue_conversation(
                        rounds_done=round_idx + 1,
                        max_rounds=max_conversation_rounds,
                        chain_seed=chain_seed,
                        chain_progress=chain_after_round,
                        rng=local_rng,
                        min_rounds=min_conversation_rounds,
                    ):
                        break

                # If we broke out of conversation loop early (first round failed)
                _real_now = [c for c in all_oracle_calls if getattr(c, "action", "tool_call") == "tool_call"]
                _clar_now = [c for c in all_oracle_calls if getattr(c, "action", "tool_call") == "ask_clarification"]
                if not _real_now and not (difficulty == "missing" and _clar_now):
                    self.manager.close_session(session_id)
                    continue  # retry loop

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
                valid, error_rate, num_errors, num_calls = replay_validate(
                    oracle_calls=all_oracle_calls,
                    manager=self.manager,
                    executor=self.executor,
                    seed=local_seed,
                    domain=server_name,
                    success_criteria=success_criteria,
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

                # ── Provenance check (PROVE §3.2 Step 5: sensitive params) ──
                prov_ok, prov_violations = provenance_check(
                    oracle_calls=all_oracle_calls,
                    user_query="\n".join(conversation_queries),
                    execution_history=all_execution_history,
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

                # ── Success ──
                break

            finally:
                self.manager.close_session(session_id)

        # ── Final guard: ensure the oracle matches the task type ──
        # Exception: difficulty="missing" expects clarification-only behavior
        # (PROVE missing-required information level). If the oracle has at
        # least one ask_clarification, that's a valid task — don't raise.
        real_calls = [c for c in all_oracle_calls
                      if getattr(c, "action", "tool_call") == "tool_call"]
        clarification_calls = [c for c in all_oracle_calls
                               if getattr(c, "action", "tool_call") == "ask_clarification"]
        if not real_calls and not (difficulty == "missing" and clarification_calls):
            raise RuntimeError(
                f"No real tool_call recorded for {server_name} task {task_id} "
                f"after 3 retries (LLM only produced clarifications/refusals)"
            )
        if real_calls and not (2 <= len(real_calls) <= 5):
            raise RuntimeError(
                f"Oracle chain length {len(real_calls)} outside required 2-5 "
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
            "strip_enums": bool(getattr(teacher, "_strip_enums", False)),
            "reference_date": reference_date,
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
                         dynamic_ncols=True, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")
        except ImportError:
            pbar = None

        # ── normal task generation: per-domain budget ──
        for si, current_server in enumerate(servers):
            domain_target = per_domain + (1 if si < remainder else 0)
            domain_ok = 0
            domain_failed = 0
            # Strict 2-5 chain and replay gates intentionally reject fluent but
            # unverifiable teacher traces; allow enough attempts to replenish
            # the requested domain quota instead of silently under-yielding.
            max_domain_failures = max(domain_target * 4, 10)

            for _attempt in range(domain_target + max_domain_failures):
                if domain_ok >= domain_target:
                    break
                if domain_failed >= max_domain_failures:
                    logger.warning(
                        f"{current_server}: gave up after {domain_failed} failures, "
                        f"got {domain_ok}/{domain_target}"
                    )
                    break

                task_seed = seed + global_seed_offset
                global_seed_offset += 1
                difficulty = self._pick_difficulty(task_seed, effective_mix)
                try:
                    task = self.generate_one(
                        current_server, seed=task_seed, difficulty=difficulty,
                    )
                    q_key = (task.user_prompt or "").strip().lower()
                    if q_key and q_key in seen_queries:
                        dropped_dup_query += 1
                        logger.debug(
                            f"{current_server}: dropping duplicate query "
                            f"(seen #{dropped_dup_query}): {q_key[:80]}"
                        )
                        continue  # don't count as ok or failed; let attempt budget cover it
                    if q_key:
                        seen_queries.add(q_key)

                    rng_knob = random.Random(task_seed)
                    if rng_knob.random() < distractor_rate:
                        self._apply_distractors(task)
                    if rng_knob.random() < missing_function_rate:
                        # Try to apply missing_function perturbation. If the
                        # semantic filter rejects this candidate (the hidden tool
                        # isn't required by the user query), keep the task as-is
                        # rather than discarding valid generated work.
                        _applied = self._apply_missing_function(task)
                        if not _applied:
                            logger.debug(
                                f"{task.task_id}: missing_function semantic "
                                f"filter rejected; keeping task without perturbation"
                            )
                    tasks.append(task)
                    domain_ok += 1
                    if pbar:
                        pbar.update(1)
                        pbar.set_postfix_str(f"fail={failed}")
                    elif len(tasks) % 10 == 0:
                        print(f"[generate_many] {len(tasks)}/{n_normal} tasks, {failed} failures", flush=True)
                    if len(tasks) % 10 == 0:
                        logger.info(f"generate_many progress: {len(tasks)}/{n_normal} tasks, {failed} failures")
                except Exception as e:
                    failed += 1
                    domain_failed += 1
                    if pbar:
                        pbar.set_postfix_str(f"fail={failed}")
                    logger.warning(
                        f"generate failed for {current_server} "
                        f"({domain_failed}x): {e}"
                    )

        if pbar:
            pbar.close()

        # ── irrelevance tasks (5%) ──
        irr = self._generate_irrelevant_tasks(n_irrelevant, seed + 9999, servers)
        tasks.extend(irr)

        # ── dedup across all generated tasks ──
        before = len(tasks)
        tasks = dedup_tasks(tasks, threshold=0.70)
        removed = before - len(tasks)

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

        logger.info(
            f"LLM teacher: {len(tasks)} tasks (target {count}, {failed} failures, "
            f"{removed} dedup removed, {dropped_dup_query} dup-query dropped)"
        )
        return tasks

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

    def _apply_distractors(self, task: LiveTask) -> None:
        known = {t["name"] for t in task.visible_tools}
        candidates = [t for t in self.manager.registry.all_tools() if t["name"] not in known]
        # Use deterministic seed via hashlib (Python hash() is randomized by PYTHONHASHSEED)
        import hashlib
        seed_bytes = hashlib.md5(task.task_id.encode()).digest()
        rng = random.Random(int.from_bytes(seed_bytes[:8], "big"))
        selected = rng.sample(candidates, min(len(candidates), rng.randint(3, 8)))
        task.visible_tools.extend(selected)
        task.metadata["has_distractors"] = True
        task.metadata["distractor_count"] = len(selected)

    def _apply_missing_function(self, task: LiveTask) -> bool:
        """Apply missing_function perturbation. Returns True if applied, False if
        the task is not a valid abstention candidate (skip it in generate_many).
        """
        if not task.required_tools:
            return False
        # Pick the LAST tool actually invoked in the oracle chain (the terminal
        # action), not required_tools[-1] which is alphabetically sorted and
        # would often select a lookup tool instead of the executor. Hiding the
        # terminal/executor matches the intended "abstain" semantics: the model
        # has all the lookup tools to inspect state but cannot complete the
        # action, so the correct behavior is report_error with a clear reason.
        oracle_calls = getattr(task.oracle_program, "calls", None) or []
        tool_oracle_calls = [
            call for call in oracle_calls
            if getattr(call, "action", "tool_call") == "tool_call"
        ]
        if tool_oracle_calls:
            hidden = tool_oracle_calls[-1].tool_name
        else:
            # Fallback: oracle empty (shouldn't happen for non-irrelevant tasks),
            # use last required tool which preserves prior behavior.
            hidden = task.required_tools[-1]
        if hidden not in task.required_tools:
            # Defensive: if oracle's last tool somehow not in required_tools
            # (e.g. filtered earlier), fall back to required_tools[-1].
            hidden = task.required_tools[-1]
        if not _missing_function_candidate_is_semantically_required(task, hidden):
            logger.debug(
                f"{task.task_id}: skip missing_function perturbation because "
                f"{hidden} is not required by the visible user request"
            )
            return False
        missing = {"type": "missing_function", "server": task.target_servers[0], "tool": hidden}
        task.metadata["original_required_tools"] = list(task.required_tools)
        task.metadata["original_success_criteria"] = list(task.success_criteria)
        task.metadata["original_oracle_program"] = to_plain(task.oracle_program)
        task.hidden_tools.append(hidden)
        task.visible_tools = [t for t in task.visible_tools if t["name"] != hidden]

        # Guard: ensure visible_tools never empty — otherwise _tasks_to_rows
        # silently drops the task. Add cross-domain distractor tools as fallback.
        if not task.visible_tools:
            import hashlib
            known = set(task.hidden_tools) | {hidden}
            candidates = [t for t in self.manager.registry.all_tools()
                          if t["name"] not in known]
            seed_bytes = hashlib.md5(task.task_id.encode()).digest()
            rng = random.Random(int.from_bytes(seed_bytes[:8], "big"))
            if candidates:
                selected = rng.sample(candidates, min(len(candidates), rng.randint(3, 8)))
                task.visible_tools = selected
                task.metadata["has_distractors"] = True
                task.metadata["distractor_count"] = len(selected)

        task.required_tools = []
        task.success_criteria = [missing]
        task.expected_outcome = {"success_criteria": [missing], "abstain": True}
        task.oracle_program.calls = [OracleCall(
            tool_name="report_error",
            arguments={"text": f"Required tool '{hidden}' is unavailable."},
            action="report_error",
        )]
        task.oracle_program.success_criteria = [missing]
        task.task_type = "missing_function"
        task.metadata["has_missing_function"] = True
        task.metadata["unavailable_required_tool"] = hidden
        task.metadata["scenario_type"] = "missing_function"

        # missing_function semantics demand report_error without invoking
        # tools. Collapse the task to single-turn abstain shape so prompt,
        # oracle, and reward target remain the same MDP.
        task.oracle_calls_per_round = []
        task.execution_history_per_round = []
        if task.conversation_queries:
            task.conversation_queries = [task.conversation_queries[0]]
        # user_prompt already holds the first query; nothing to change there.
        return True

    def _generate_irrelevant_tasks(
        self,
        n: int,
        seed: int,
        allowed_servers: list[str] | None = None,
    ) -> list[LiveTask]:
        """Generate tasks whose query is unrelated to any available tool.

        The expected model behavior is to ``report_error`` (cannot be done).
        These tasks have an empty oracle program and ``missing_function``-type
        success criteria.
        """
        if n <= 0:
            return []
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

            missing = {"type": "missing_function", "server": server_name, "tool": "all"}
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
                success_criteria=[missing],
                oracle_program=OracleProgram(
                    task_id=task_id,
                    calls=[OracleCall(
                        tool_name="report_error",
                        arguments={"text": "No available tool can satisfy this request."},
                        action="report_error",
                    )],
                    success_criteria=[missing],
                ),
                sampling_context={},
                max_turns=int(self.suite_config.rollout.get("max_turns", 8)),
                difficulty="minimal",
                task_type="irrelevant",
                metadata={
                    "generation_method": "irrelevant_template",
                    "irrelevant": True,
                    "scenario_type": "no_tool_or_abstention",
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
            if isinstance(graph, dict) and cached_tool_names == expected_tool_names:
                graph = self._normalize_cached_graph(graph, expected_tool_names)
                self._apply_prove_dependency_definition_filter(graph, server_tools, server_name)
            if (
                isinstance(payload, dict)
                and payload.get("schema_hash") == schema_hash
                and payload.get("server_name") == server_name
                and cached_tool_names == expected_tool_names
                and self._valid_cached_graph(graph, expected_tool_names)
            ):
                logger.info(f"Loaded dependency graph cache: {cache_path}")
                return graph
            logger.warning(f"Ignoring invalid dependency graph cache: {cache_path}")
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
        self._apply_prove_dependency_definition_filter(graph, server_tools, server_name)
        if not self._valid_cached_graph(graph, expected_tool_names):
            logger.warning(f"Skipping invalid dependency graph cache for {server_name}")
            return
        cache_path = self._graph_cache_path(server_name, schema_hash)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "server_name": server_name,
            "schema_hash": schema_hash,
            "tool_names": expected_tool_names,
            "graph": graph,
        }
        cache_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
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
        cached = self._maybe_load_cached_graph(server_name, schema_hash, server_tools)
        if cached is not None:
            self.manager.close_session(session.session_id)
            return cached

        try:
            graph = self._classify_edges_llm(server_tools, server_name) or {}

            # ── P3c: Schema-based deterministic edges as a pre-pass ──
            # Merge deterministic edges into the LLM-classified graph so
            # critical read-before-write chains (get_lead→convert_lead,
            # get_cart→checkout, etc.) are never lost, even if the LLM
            # classifier misses them.
            det = _deterministic_schema_edges(server_tools, server_name)
            for src, edge_info in det.items():
                if src not in graph:
                    graph[src] = edge_info
                else:
                    ex = set(graph[src].get("explicit", []))
                    im = set(graph[src].get("implicit", []))
                    graph[src] = {
                        "explicit": sorted(ex | set(edge_info.get("explicit", []))),
                        "implicit": sorted((im | set(edge_info.get("implicit", []))) - ex),
                    }

            self._save_cached_graph(server_name, schema_hash, server_tools, graph)
        finally:
            self.manager.close_session(session.session_id)

        return graph

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
        pair_labels = [f"{a_name} ↔ {b_name}" for a_name, b_name in pairs]
        logger.debug(
            f"_classify_edges_llm: {server_name} classifying {len(pairs)} "
            f"unordered tool pairs"
        )

        # Batch pairs to fit LLM context and bound single-request decode time.
        # This remains PROVE-style pairwise LLM classification over all nC2
        # pairs; each request just carries the schemas needed for its batch.
        BATCH_SIZE = 24
        all_classifications: dict[str, str] = {}  # "A → B" → "explicit"|"implicit"
        classified_pairs: set = set()

        for batch_start in range(0, len(pairs), BATCH_SIZE):
            batch_pairs = pairs[batch_start:batch_start + BATCH_SIZE]
            batch_labels = pair_labels[batch_start:batch_start + BATCH_SIZE]
            valid_batch_pairs = {tuple(sorted(pair)) for pair in batch_pairs}

            batch_tool_names = sorted({name for pair in batch_pairs for name in pair})
            batch_tools_text = "\n\n".join(
                tool_desc_by_name[name]
                for name in batch_tool_names
                if name in tool_desc_by_name
            )
            pairs_text = "\n".join(f"{i+1}. {label}" for i, label in enumerate(batch_labels))

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
            user = (
                f"## Server: {server_name}\n\n"
                f"## Tools\n{batch_tools_text}\n\n"
                f"## Pairs to Classify\n{pairs_text}\n\n"
                f"## Output Format\n"
                f'{{"classifications": [\n'
                f'  {{"pair": "tool_a ↔ tool_b", "source": "tool_a", "target": "tool_b", "relation": "explicit"}},\n'
                f'  {{"pair": "tool_c ↔ tool_d", "source": "tool_d", "target": "tool_c", "relation": "implicit"}},\n'
                f'  {{"pair": "tool_e ↔ tool_f", "relation": "none"}},\n'
                f'  ...\n'
                f']}}\n\n'
                f"Output ONLY the JSON, nothing else:"
            )

            try:
                raw = self.client.generate_chat(
                    [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
                    temperature=0.1,
                    max_tokens=1024,
                )
                data = _extract_json(raw)
                for entry in data.get("classifications", []):
                    relation = entry.get("relation", "none")
                    if relation not in ("explicit", "implicit"):
                        continue
                    source = str(entry.get("source") or "")
                    target = str(entry.get("target") or "")
                    if not source or not target:
                        pair_text = str(entry.get("pair", ""))
                        parts = pair_text.split(" → ")
                        if len(parts) == 2:
                            source, target = parts
                    if source not in tool_desc_by_name or target not in tool_desc_by_name:
                        continue
                    if source == target:
                        continue
                    pair_key = tuple(sorted((source, target)))
                    if pair_key not in valid_batch_pairs:
                        continue
                    if pair_key in classified_pairs:
                        continue
                    classified_pairs.add(pair_key)
                    all_classifications[f"{source} → {target}"] = relation
            except Exception as e:
                logger.debug(
                    f"_classify_edges_llm batch {batch_start // BATCH_SIZE + 1} "
                    f"failed for {server_name}: {e}"
                )
                # Continue with other batches; partial results are better than none

        if not all_classifications:
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

        return {
            "source": "live_readonly_probe",
            "entity_ids": entity_ids,
            "entity_summaries": entity_summaries,
            "entity_records": entity_records,
            "entity_types": sorted({item["type"] for item in entity_ids}),
            "probe_results": probe_results,
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

        feasible: list[list[str]] = []
        drop_reasons: dict[str, int] = {}
        drop_examples: dict[str, list[str]] = {}
        for chain in chains:
            ok, reason = _chain_is_feasible(
                chain, server_name, live_context
            )
            if ok:
                feasible.append(chain)
            else:
                reason_key = reason or "unknown"
                drop_reasons[reason_key] = drop_reasons.get(reason_key, 0) + 1
                drop_examples.setdefault(reason_key, chain)

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
        if server_name not in self._domain_graphs:
            graph = self._probe_dependency_graph(server_name)
            self._domain_graphs[server_name] = graph
        return _format_graph_hints(self._domain_graphs[server_name])

    def _get_live_sampling_context(
        self,
        session_id: str,
        server_name: str,
        server_tools: list[dict],
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

        if entry["call_count"] % self.SAMPLING_CONTEXT_REFRESH_K == 0:
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


def _query_aligns_with_chain(
    user_query: str,
    chain_seed: list[str],
    is_mutating_tool,
) -> tuple[bool, str]:
    """Check that a chain-seeded query asks for the final mutating action.

    PROVE samples user queries from dependency-graph chains. If the natural
    query only asks for an early read step while decide_action is later forced
    by chain_guidance to execute a write, the oracle target no longer matches
    the user request.  Read-only final steps are exempt because lookup chains
    can be expressed by broad information requests.
    """
    if not user_query or not chain_seed:
        return True, "no chain/query"
    final_tool = chain_seed[-1]
    if not is_mutating_tool(final_tool):
        return True, "final tool is read-only"
    if _query_has_write_intent_for_tool(user_query, final_tool):
        return True, "ok"
    return (
        False,
        f"query does not request final mutating chain tool {final_tool}: {user_query!r}",
    )


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
    read_prefixes = ("list_", "search_", "get_", "find_", "lookup_", "check_",
                     "view_", "browse_", "ls", "cat", "pwd", "stat", "head", "tail")
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
    "send_email": {"email"},
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


_PROVE_STATE_DEPENDENCY_EDGES: set[tuple[str, str, str]] = {
    ("banking", "freeze_account", "unfreeze_account"),
    ("calendar", "add_attendee", "remove_attendee"),
    ("payments", "pay_invoice", "refund_invoice"),
    ("shopping", "add_to_wishlist", "remove_from_wishlist"),
    ("issue_tracker", "add_label", "remove_label"),
    ("issue_tracker", "add_watcher", "remove_watcher"),
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
    """Convert read-only probe entities into the compact state formatter shape."""
    prompt_state: dict[str, dict[str, Any]] = {}
    summaries = list(live_context.get("entity_summaries", []))
    for idx, item in enumerate(live_context.get("entity_ids", [])):
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


def _chain_respects_state_preconditions(server_name: str, chain: list[str]) -> bool:
    if server_name == "shopping":
        cart_tools = {"checkout", "update_cart_quantity", "remove_from_cart", "clear_cart"}
        for cart_tool in cart_tools:
            if cart_tool in chain:
                target_idx = chain.index(cart_tool)
                if "add_to_cart" not in chain[:target_idx]:
                    return False

    if server_name == "payments":
        if "create_invoice" in chain and "refund_invoice" in chain:
            create_idx = chain.index("create_invoice")
            refund_idx = chain.index("refund_invoice")
            if create_idx < refund_idx and "pay_invoice" not in chain[create_idx + 1:refund_idx]:
                return False
        if "cancel_payment" in chain:
            cancel_idx = chain.index("cancel_payment")
            if "create_invoice" in chain and chain.index("create_invoice") < cancel_idx:
                return False
            if "pay_invoice" in chain and chain.index("pay_invoice") < cancel_idx:
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
            if "update_order_status" not in chain[:track_idx]:
                return False
        if "create_order" in chain:
            create_idx = chain.index("create_order")
            for status_tool in ("rate_order", "track_rider"):
                if status_tool in chain and chain.index(status_tool) > create_idx:
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
    for item in live_context.get("entity_ids", []):
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

    # Domain-specific compound edges
    _EDGES: dict[str, list[tuple[str, str]]] = {
        "shopping": [
            ("search_products", "get_product"), ("get_cart", "checkout"),
            ("get_cart", "clear_cart"), ("get_cart", "remove_from_cart"),
        ],
        "calendar": [("search_events", "get_event")],
        "email": [("list_inbox", "get_email")],
        "crm": [
            ("list_leads", "update_lead"), ("list_leads", "convert_lead"),
            ("list_leads", "delete_lead"), ("list_deals", "update_deal"),
            ("list_deals", "get_deal"), ("list_tasks", "complete_task"),
        ],
        "issue_tracker": [("search_issues", "get_issue")],
        "filesystem": [
            ("find", "mv"), ("find", "rm"), ("find", "chmod"), ("find", "chown"),
            ("find", "cp"), ("find", "cat"),
            ("ls", "mv"), ("ls", "chmod"), ("ls", "rm"), ("ls", "cp"),
            ("ls", "cd"), ("ls", "cat"),
            ("cd", "ls"), ("cd", "pwd"),
            ("cat", "rm"), ("cat", "grep"), ("cat", "sed"), ("cat", "awk"),
            ("touch", "chmod"), ("touch", "cat"),
            ("mkdir", "cd"), ("mkdir", "touch"),
            ("mv", "ls"), ("cp", "ls"),
            ("du", "rm"),
        ],
        "banking": [("search_transactions", "get_transaction")],
        "payments": [("list_invoices", "get_invoice")],
        "team_chat": [("search_messages", "get_message")],
        "food_delivery": [
            ("search_restaurants", "get_restaurant"),
            ("get_cart", "create_order"), ("get_cart", "cancel_order"),
        ],
    }
    for src, dst in _EDGES.get(server_name, []):
        _add_explicit(src, dst)

    return graph
