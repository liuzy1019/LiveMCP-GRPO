"""PROVE-style state-machine task generation.

Per environment:
  1. Auto-discover tool dependency graph via live MCP probing
  2. State machine alternating LLM decisions and tool execution
     against a live MCP server
  3. Replay-validate each conversation before conversion

No replay filtering needed — oracle trace was built by actual execution.
"""

from __future__ import annotations

import copy
import json
import random
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
        client = LLMClient(mode="openai", model_path="Qwen3-32B-Instruct", api_base="...")
        orch = TaskOrchestrator(suite_config, manager, executor, client)
        tasks = orch.generate_many("all", count=100, seed=42)
    """

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
    ) -> tuple[list, list[dict], set[str]]:
        """Run one conversation round of teacher-driven tool execution.

        Returns (oracle_calls, execution_history, required_tools).
        """
        from src.live_mcp.task_planner import ContinuationPolicy, apply_perturbation
        from src.live_mcp.types import ToolCall

        oracle_calls: list = []
        execution_history: list[dict[str, Any]] = []
        required_tools: set[str] = set()

        # BUG-C fix: dedup oracle tool calls within a round.
        # Same (tool_name, args_repr) must not appear twice — the LLM occasionally
        # "forgets" prior progress and re-issues the same call, which inflates the
        # oracle chain past PROVE's len-5 limit and leaks repeated tools into the
        # ground truth.
        seen_oracle_keys: set[tuple[str, str]] = set()
        # BUG-3 fix: read-class tools deduped by name only.
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
            # Hard cap on real (non-clarification) calls per task.
            real_count = sum(1 for oc in oracle_calls if oc.action != "clarification")
            if call.action != "clarification" and real_count >= 5:
                return False
            # LIST-class tools (one-shot collections): dedup by name only.
            # get_/search_/find_ are entity reads — different entities are
            # legitimate, so they fall through to (name,args) dedup below.
            tname = call.tool_name or ""
            if call.action != "clarification" and tname.startswith("list_"):
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
            round_chain_len = local_rng.randint(1, 3)
        target_turns = ContinuationPolicy.target_turns(round_chain_len, local_rng)
        max_turns = min(target_turns + 2, 8)

        # BUG-2 fix (PROVE alignment): hard chain-length cap at 5 across all
        # rounds.  PROVE limits oracle chain to len-5; without a hard cap the
        # turn loop keeps appending until the LLM emits final_answer, producing
        # 6-9 length chains for greedy LLMs.
        MAX_ORACLE_CALLS_PER_TASK = 5

        for turn in range(max_turns):
            chain_progress = len({oc.tool_name for oc in oracle_calls if oc.action != "clarification"})

            # BUG-2 hard cap: stop emitting new oracle entries once we hit 5.
            real_oracle_count = sum(1 for oc in oracle_calls if oc.action != "clarification")
            if real_oracle_count >= MAX_ORACLE_CALLS_PER_TASK:
                break

            action = teacher.decide_action(
                tool_schemas=server_tools,
                user_query=current_query,
                execution_history=execution_history,
                attempt=turn,
                dep_hints=dep_hints,
                difficulty=difficulty,
                chain_seed=chain_seed if round_idx == 0 else None,
                chain_progress=chain_progress,
            )

            if action.action == "ask_clarification":
                if difficulty == "missing":
                    _add_oracle(OracleCall(
                        tool_name="ask_clarification",
                        arguments={"question": action.text},
                        action="clarification",
                    ))
                break

            if action.action in ("final_answer", "report_error"):
                if (round_idx == 0 and chain_seed and len(chain_seed) >= 2
                        and chain_progress < len(chain_seed)
                        and turn < max_turns - 1):
                    remaining = chain_seed[chain_progress:]
                    execution_history.append({
                        "tool_name": "__reject__",
                        "arguments": {},
                        "observation": {
                            "error": (
                                f"Task incomplete — {chain_progress}/{len(chain_seed)} "
                                f"steps done. Remaining tools (approximate): "
                                f"{', '.join(remaining)}. Continue making tool calls."
                            ),
                        },
                        "success": False,
                    })
                    continue
                break

            if action.action != "tool_call" or not action.tool_name:
                continue

            tool_name = action.tool_name
            tool_name = _fuzzy_match_tool(tool_name, {t["name"] for t in server_tools}) or tool_name
            required_tools.add(tool_name)

            result = self.executor.execute(
                session_id,
                ToolCall(tool_name, dict(action.arguments), call_id=f"sm_{turn}"),
                domain=server_name,
            )

            perturbed_obs = apply_perturbation(
                result.observation, server_name, local_rng,
            )

            if isinstance(perturbed_obs, dict) and perturbed_obs.get("retry"):
                execution_history.append({
                    "tool_name": tool_name,
                    "arguments": dict(action.arguments),
                    "observation": perturbed_obs,
                    "success": False,
                })
                # Retry triggered: don't add failed call to oracle.
                # If the next turn succeeds, the success path will add it once.
                continue

            if not result.success:
                execution_history.append({
                    "tool_name": tool_name,
                    "arguments": dict(action.arguments),
                    "observation": perturbed_obs if perturbed_obs is not None else result.observation,
                    "success": False,
                })

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
                        ToolCall(tool_name, corrected, call_id=f"sm_recover_{turn}"),
                        domain=server_name,
                    )
                    if retry_result.success:
                        execution_history.append({
                            "tool_name": tool_name,
                            "arguments": corrected,
                            "observation": retry_result.observation if retry_result.observation is not None else {},
                            "success": True,
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
                            ToolCall(alt_tool, recovery.get("arguments", {}), call_id=f"sm_alt_{turn}"),
                            domain=server_name,
                        )
                        if alt_result.success:
                            required_tools.add(alt_tool)
                            execution_history.append({
                                "tool_name": alt_tool,
                                "arguments": recovery.get("arguments", {}),
                                "observation": alt_result.observation if alt_result.observation is not None else {},
                                "success": True,
                            })
                            _add_oracle(OracleCall(
                                tool_name=alt_tool,
                                arguments=recovery.get("arguments", {}),
                            ))
                continue

            obs_to_record = perturbed_obs if perturbed_obs is not None else result.observation
            execution_history.append({
                "tool_name": tool_name,
                "arguments": dict(action.arguments),
                "observation": obs_to_record if obs_to_record is not None else {},
                "success": True,
            })

            _add_oracle(OracleCall(
                tool_name=tool_name,
                arguments=dict(action.arguments),
            ))

            if not ContinuationPolicy.should_continue(
                turn, target_turns, result.success,
                sum(1 for oc in oracle_calls if oc.action != "clarification"),
            ):
                break

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
            TaskPlanner, derive_success_criteria, replay_validate, apply_perturbation,
            _PERSONA_TEMPLATES, _REFERENCE_DATES, ContinuationPolicy,
            provenance_check,
        )
        from src.live_mcp.types import ToolCall

        rng = random.Random(seed)

        # ── Sample diversity injectors (PROVE §4) ──
        persona = _PERSONA_TEMPLATES[seed % len(_PERSONA_TEMPLATES)]
        reference_date = _REFERENCE_DATES[(seed // len(_PERSONA_TEMPLATES)) % len(_REFERENCE_DATES)]

        # ── Sample dependency chain seed (PROVE §6 step 2) ──
        chains = self._get_chains(server_name)
        chain_seed: list[str] | None = None
        if chains and rng.random() < 0.80:  # 80% of tasks have a chain seed
            chain_seed = rng.choice(chains)

        # ── Conversation-level continuation (PROVE §3.2 Step 3.5) ──
        conversation_rounds = ContinuationPolicy.conversation_rounds(rng)

        # ── Retry with different seed if LLM refuses to call tools ──
        for retry_attempt in range(3):
            local_seed = seed + retry_attempt * 1000
            local_rng = random.Random(local_seed)

            teacher = TaskPlanner(self.client, server_name, seed=local_seed)

            session = self.manager.create_session(seed=local_seed)
            session_id = session.session_id
            all_tools = self.manager.discover_tools(session_id)
            server_tools = self.manager.registry.server_tools(server_name)

            try:
                grounded_state = self.manager.get_state(session_id)
                domain_state = grounded_state.get(server_name, {})
                initial_state_snapshot = copy.deepcopy(domain_state)

                dep_hints = self._get_graph_hints(server_name)

                user_query = teacher.generate_query(
                    tool_schemas=server_tools,
                    grounded_state=domain_state,
                    difficulty=difficulty,
                    rng=local_rng,
                    dep_hints=dep_hints,
                    persona=persona,
                    reference_date=reference_date,
                    chain_seed=chain_seed,
                )

                # Accumulators across conversation rounds (PROVE CONTINUATION)
                all_oracle_calls: list[OracleCall] = []
                all_execution_history: list[dict[str, Any]] = []
                all_required_tools: set[str] = set()
                conversation_queries: list[str] = [user_query]  # track all user messages
                oracle_calls_per_round: list[list[OracleCall]] = []  # per-round for prompt construction
                execution_history_per_round: list[list[dict]] = []
                task_id = f"{server_name}_{local_seed}_{local_rng.randint(0, 99999)}"
                retry_label = f" (retry {retry_attempt})" if retry_attempt > 0 else ""

                current_query = user_query

                logger.debug(
                    f"CONTINUATION: {server_name} task {task_id} "
                    f"starting {conversation_rounds} conversation round(s)"
                )

                for round_idx in range(conversation_rounds):
                    if round_idx > 0:
                        logger.debug(
                            f"CONTINUATION: {server_name} round {round_idx + 1}/{conversation_rounds} "
                            f"generating follow-up query"
                        )
                        grounded_state_update = self.manager.get_state(session_id)
                        domain_state_update = grounded_state_update.get(server_name, {})
                        current_query = teacher.generate_followup(
                            tool_schemas=server_tools,
                            grounded_state=domain_state_update,
                            previous_query=user_query,
                            execution_history=all_execution_history,
                            difficulty=difficulty,
                            rng=local_rng,
                            persona=persona,
                            reference_date=reference_date,
                        )
                        conversation_queries.append(current_query)

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
                    )

                    if round_idx == 0:
                        _real_round = [c for c in round_ocs if getattr(c, "action", "tool_call") != "clarification"]
                        _clar_round = [c for c in round_ocs if getattr(c, "action", "tool_call") == "clarification"]
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
                    global_seen_read = {oc.tool_name for oc in all_oracle_calls
                                        if getattr(oc, "action", "tool_call") != "clarification"
                                        and (oc.tool_name or "").startswith("list_")}
                    global_seen_keys = set()
                    for _oc in all_oracle_calls:
                        try:
                            _args_repr = json.dumps(_oc.arguments or {}, sort_keys=True, default=str, ensure_ascii=False)
                        except Exception:
                            _args_repr = repr(_oc.arguments)
                        global_seen_keys.add((_oc.tool_name, _args_repr))

                    real_so_far = sum(1 for oc in all_oracle_calls if getattr(oc, "action", "tool_call") != "clarification")
                    filtered_round_ocs = []
                    for oc in round_ocs:
                        action = getattr(oc, "action", "tool_call")
                        if action == "clarification":
                            filtered_round_ocs.append(oc)
                            continue
                        if real_so_far >= 5:
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

                # If we broke out of conversation loop early (first round failed)
                _real_now = [c for c in all_oracle_calls if getattr(c, "action", "tool_call") != "clarification"]
                _clar_now = [c for c in all_oracle_calls if getattr(c, "action", "tool_call") == "clarification"]
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

                # ── Replay validate (PROVE §3.2 Step 5: ≤30% error rate) ──
                valid, error_rate, num_errors, num_calls = replay_validate(
                    oracle_calls=all_oracle_calls,
                    manager=self.manager,
                    executor=self.executor,
                    seed=local_seed,
                    domain=server_name,
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
                    user_query=user_query,
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

                # ── Success ──
                break

            finally:
                self.manager.close_session(session_id)

        # ── Final guard: ensure oracle has at least 1 real tool_call ──
        # BUG-D fix: retry loop may have exited via fall-through with empty
        # oracle (e.g., every retry the LLM only emitted ask_clarification at
        # round 0). task_planner-typed tasks must execute tools; if not, raise
        # so generate_many counts it as a failure rather than emitting a 0-call
        # row that pollutes the dataset.
        #
        # Exception: difficulty="missing" expects clarification-only behavior
        # (PROVE missing-required information level). If the oracle has at
        # least one ask_clarification, that's a valid task — don't raise.
        real_calls = [c for c in all_oracle_calls
                      if getattr(c, "action", "tool_call") != "clarification"]
        clarification_calls = [c for c in all_oracle_calls
                               if getattr(c, "action", "tool_call") == "clarification"]
        if not real_calls and not (difficulty == "missing" and clarification_calls):
            raise RuntimeError(
                f"No real tool_call recorded for {server_name} task {task_id} "
                f"after 3 retries (LLM only produced clarifications/refusals)"
            )

        # ── Build final task ──
        oracle_program = OracleProgram(
            task_id=task_id,
            calls=all_oracle_calls,
            success_criteria=success_criteria,
        )

        return self._to_live_task(
            server_name=server_name, query=user_query,
            session_id=session_id, seed=local_seed,
            all_tools=all_tools, oracle_program=oracle_program,
            required_tools=sorted(all_required_tools),
            difficulty=difficulty, task_id=task_id,
            conversation_queries=conversation_queries,
            oracle_calls_per_round=oracle_calls_per_round,
            execution_history_per_round=execution_history_per_round,
        )

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
        # BUG-B fix: dedup tasks by first user query string. The teacher LLM
        # tends to emit identical queries ("link deal_0001 to contact_0001")
        # across different seeds because its query-generation prompt is
        # template-y. Same query → same task semantically; we drop duplicates
        # before they reach Jaccard-on-tool-seq dedup (which only catches
        # tool-sequence overlap, not query-string repetition).
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
            max_domain_failures = max(domain_target * 2, 5)

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
                    # BUG-B fix: drop duplicate first-user-query tasks early.
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
                        self._apply_missing_function(task)
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
        irr = self._generate_irrelevant_tasks(n_irrelevant, seed + 9999)
        tasks.extend(irr)

        # ── dedup across all generated tasks ──
        before = len(tasks)
        tasks = dedup_tasks(tasks, threshold=0.70)
        removed = before - len(tasks)

        # P1-6: surface low yield to the caller. With irrelevance_ratio<1,
        # the contractual target is `count` rows; falling far short almost
        # always means the teacher LLM/MCP server pipeline is broken. We
        # warn loudly at <50% and raise at 0% so that callers don't
        # silently write empty Parquet files.
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
                      execution_history_per_round: list[list] | None = None) -> LiveTask:
        if required_tools:
            # Show all domain tools — model must figure out which to use.
            # Don't leak required_tools by only showing those.
            all_domain_tools = self.manager.registry.server_tools(server_name)
            visible_tools = all_domain_tools if all_domain_tools else [t for t in all_tools if t["name"] in required_tools]
        else:
            # Clarification-only tasks (missing difficulty): expose all tools
            # so the agent can see what's available to identify the missing param
            visible_tools = all_tools
        return LiveTask(
            task_id=task_id, source="live_mcp_task_planner",
            suite_name=self.suite_config.suite_name, user_prompt=query,
            session_id=session_id, session_seed=seed, target_servers=[server_name],
            visible_tools=visible_tools, required_tools=list(required_tools),
            expected_outcome={"success_criteria": oracle_program.success_criteria},
            success_criteria=list(oracle_program.success_criteria),
            oracle_program=oracle_program, sampling_context={},
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

    def _apply_missing_function(self, task: LiveTask) -> None:
        if not task.required_tools:
            return
        # Pick the LAST tool actually invoked in the oracle chain (the terminal
        # action), not required_tools[-1] which is alphabetically sorted and
        # would often select a lookup tool instead of the executor. Hiding the
        # terminal/executor matches the intended "abstain" semantics: the model
        # has all the lookup tools to inspect state but cannot complete the
        # action, so the correct behavior is report_error with a clear reason.
        oracle_calls = getattr(task.oracle_program, "calls", None) or []
        if oracle_calls:
            hidden = oracle_calls[-1].tool_name
        else:
            # Fallback: oracle empty (shouldn't happen for non-irrelevant tasks),
            # use last required tool which preserves prior behavior.
            hidden = task.required_tools[-1]
        if hidden not in task.required_tools:
            # Defensive: if oracle's last tool somehow not in required_tools
            # (e.g. filtered earlier), fall back to required_tools[-1].
            hidden = task.required_tools[-1]
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

        task.required_tools = [n for n in task.required_tools if n != hidden]
        task.success_criteria = [missing]
        task.expected_outcome = {"success_criteria": [missing], "abstain": True}
        task.oracle_program.calls = []
        task.oracle_program.success_criteria = [missing]
        task.task_type = "missing_function"
        task.metadata["has_missing_function"] = True
        task.metadata["unavailable_required_tool"] = hidden

        # BUG-1 fix (PROVE alignment): missing_function semantics demand the
        # model abstain (report_error) without invoking tools.  However the
        # task was generated *with* a successful trajectory still cached in
        # ``oracle_calls_per_round`` / ``execution_history_per_round`` /
        # ``conversation_queries`` (multi-turn followups).  If we keep those,
        # generate_data.py renders a multi-turn prompt that *shows* the model
        # successfully calling tools — but extra_info[oracle_calls] is now
        # empty.  Result: prompt and oracle disagree, model trains on noise.
        # Collapse the task to single-turn abstain shape so prompt matches
        # oracle.
        task.oracle_calls_per_round = []
        task.execution_history_per_round = []
        if task.conversation_queries:
            task.conversation_queries = [task.conversation_queries[0]]
        # user_prompt already holds the first query; nothing to change there.

    def _generate_irrelevant_tasks(self, n: int, seed: int) -> list[LiveTask]:
        """Generate tasks whose query is unrelated to any available tool.

        The expected model behavior is to ``report_error`` (cannot be done).
        These tasks have an empty oracle program and ``missing_function``-type
        success criteria.
        """
        if n <= 0:
            return []
        rng = random.Random(seed)
        tasks: list[LiveTask] = []

        for i in range(n):
            server_name = rng.choice(self.manager.server_names)
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
                    task_id=task_id, calls=[], success_criteria=[missing],
                ),
                sampling_context={},
                max_turns=int(self.suite_config.rollout.get("max_turns", 8)),
                difficulty="minimal",
                task_type="irrelevant",
                metadata={
                    "generation_method": "irrelevant_template",
                    "irrelevant": True,
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

    def _maybe_load_cached_graph(self, server_name: str) -> dict | None:
        """Load precomputed LLM dependency graph from disk cache.

        Cache files live in ``data/dependency_graphs/{domain}.json``.
        These are produced by running ``_classify_edges_llm`` offline
        (e.g., via ``scripts/precompute_graphs.py``).

        Returns None if cache doesn't exist or is invalid.
        """
        from pathlib import Path
        cache_dir = Path("data/dependency_graphs")
        cache_path = cache_dir / f"{server_name}.json"
        if not cache_path.exists():
            return None
        try:
            import json as _json
            data = _json.loads(cache_path.read_text())
            if not isinstance(data, dict) or not data:
                return None
            # Validate structure
            for tool_name, edges in data.items():
                if not isinstance(edges, dict):
                    return None
                if "explicit" not in edges or "implicit" not in edges:
                    return None
            logger.info(f"Loaded cached LLM dependency graph: {cache_path}")
            return data
        except Exception as e:
            logger.warning(f"Failed to load cached graph {cache_path}: {e}")
            return None

    def precompute_llm_graph(self, server_name: str) -> bool:
        """Run LLM pairwise classification and save result to disk cache.

        Call this once per domain as an offline preprocessing step.
        The cached graph is loaded automatically by ``_probe_dependency_graph``
        on subsequent runs.

        Returns True if graph was computed and saved successfully.
        """
        from pathlib import Path
        import json as _json

        session = self.manager.create_session(seed=0)
        try:
            all_tools = self.manager.discover_tools(session.session_id)
            server_tools = [
                t for t in all_tools
                if self.manager.registry.server_for_tool(t["name"]) == server_name
            ]
        finally:
            self.manager.close_session(session.session_id)

        if len(server_tools) < 2:
            logger.warning(f"precompute_llm_graph: {server_name} has < 2 tools, skipping")
            return False

        try:
            graph = self._classify_edges_llm(server_tools, server_name)
            if not graph:
                return False
        except Exception as e:
            logger.error(f"precompute_llm_graph failed for {server_name}: {e}")
            return False

        cache_dir = Path("data/dependency_graphs")
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{server_name}.json"
        cache_path.write_text(_json.dumps(graph, indent=2, ensure_ascii=False))
        logger.info(
            f"Saved LLM dependency graph: {cache_path} "
            f"(explicit: {sum(len(v['explicit']) for v in graph.values())}, "
            f"implicit: {sum(len(v['implicit']) for v in graph.values())})"
        )
        return True

    def _probe_dependency_graph(self, server_name: str) -> dict:
        """PROVE Step 1: auto-discover tool dependencies.

        Uses live MCP probing + rule-based classification (field intersection for
        explicit edges, entity-type clustering for implicit edges). This is
        deterministic, zero-cost, and produces valid dependency graphs for
        chain extraction.

        Optional: LLM-based pairwise classification can be precomputed offline
        and saved to ``data/dependency_graphs/{domain}.json``. If the cache file
        exists, it takes precedence over rule-based results (higher quality
        implicit edge detection).
        """
        # ── Try loading precomputed LLM graph first ──
        cached = self._maybe_load_cached_graph(server_name)
        if cached:
            logger.debug(f"_probe_dependency_graph: using cached LLM graph for {server_name}")
            self._domain_graphs[server_name] = cached
            return cached

        session = self.manager.create_session(seed=0)
        try:
            all_tools = self.manager.discover_tools(session.session_id)
            server_tools = [
                t for t in all_tools
                if self.manager.registry.server_for_tool(t["name"]) == server_name
            ]
        except Exception as e:
            logger.debug(f"_probe_dependency_graph: tool discovery failed for {server_name}: {e}")
            self.manager.close_session(session.session_id)
            return {}

        if len(server_tools) < 2:
            self.manager.close_session(session.session_id)
            return {}

        graph: dict[str, dict] = {}
        for tool in server_tools:
            graph[tool["name"]] = {"explicit": [], "implicit": []}

        # ── Rule-based classification (primary path, instantaneous) ──
        query_outputs: dict[str, set[str]] = {}
        try:
            for tool in server_tools:
                name = tool["name"]
                if not _is_query_tool(name):
                    continue
                try:
                    from src.live_mcp.types import ToolCall
                    probe_args = _minimal_args(tool)
                    result = self.executor.execute(
                        session.session_id,
                        ToolCall(name, probe_args, call_id=f"probe_{name}"),
                        domain=server_name,
                    )
                    if result.success and isinstance(result.observation, dict):
                        fields: set[str] = set()
                        _collect_fields(result.observation, fields)
                        query_outputs[name] = fields
                except Exception as e:
                    logger.debug(f"_probe_dependency_graph: probe failed for {name}: {e}")

            # Explicit edges: A's output field names match B's required params
            for a_tool, a_fields in query_outputs.items():
                for b_tool_info in server_tools:
                    b_name = b_tool_info["name"]
                    if b_name == a_tool:
                        continue
                    b_required = set(
                        b_tool_info.get("input_schema", {}).get("required", [])
                    )
                    if a_fields & b_required:
                        graph[a_tool]["explicit"].append(b_name)

            # Implicit edges: A and B operate on the same entity type
            for a_name in graph:
                if not graph[a_name]["explicit"]:
                    a_entity = _tool_entity(a_name)
                    for b_name in graph:
                        if b_name != a_name and _tool_entity(b_name) == a_entity:
                            graph[a_name]["implicit"].append(b_name)
        finally:
            self.manager.close_session(session.session_id)

        # ── Merge yaml pre-defined dependency_graph.edges (supplements rule-based) ──
        _merge_yaml_dependency_edges(graph, server_name, self.suite_config)

        return graph

    def _classify_edges_llm(
        self,
        server_tools: list[dict],
        server_name: str,
    ) -> dict | None:
        """PROVE §3.2 Step 1: LLM-based pairwise tool relationship classification.

        Sends all nC2 tool pairs to the LLM in batches, asking it to classify
        each directed edge as explicit, implicit, or none.

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

        tools_text = "\n\n".join(tool_descs)

        # Generate all directed pairs (A → B, A ≠ B)
        pairs: list[tuple[str, str]] = []
        pair_labels: list[str] = []
        for a_name in tool_names:
            for b_name in tool_names:
                if a_name != b_name:
                    pairs.append((a_name, b_name))
                    pair_labels.append(f"{a_name} → {b_name}")

        # Batch pairs to fit LLM context (~20 pairs per call)
        BATCH_SIZE = 20
        all_classifications: dict[str, str] = {}  # "A → B" → "explicit"|"implicit"|"none"

        for batch_start in range(0, len(pairs), BATCH_SIZE):
            batch_pairs = pairs[batch_start:batch_start + BATCH_SIZE]
            batch_labels = pair_labels[batch_start:batch_start + BATCH_SIZE]

            pairs_text = "\n".join(f"{i+1}. {label}" for i, label in enumerate(batch_labels))

            system = (
                "You are analyzing tool dependencies for an MCP server. "
                "For each directed tool pair (A → B), classify the relationship:\n"
                '- "explicit": tool A produces output that is a REQUIRED INPUT of tool B '
                "(e.g., A returns an entity ID that B needs as a parameter).\n"
                '- "implicit": tool A must execute BEFORE tool B to establish state, '
                "but A's output is not a direct input to B.\n"
                '- "none": no dependency — B can execute without A.\n\n'
                "Classification rules:\n"
                "- If A's description mentions creating/returning something that B's "
                "required parameters reference, it is explicit.\n"
                "- If A and B operate on the same entity type (e.g., both deal with "
                "'orders' or 'events') but A's output isn't a required param of B, "
                "it may be implicit.\n"
                "- Prefer explicit over implicit when both could apply.\n"
                "- Only mark implicit if there is a genuine state dependency."
            )
            user = (
                f"## Server: {server_name}\n\n"
                f"## Tools\n{tools_text}\n\n"
                f"## Pairs to Classify\n{pairs_text}\n\n"
                f"## Output Format\n"
                f'{{"classifications": [\n'
                f'  {{"pair": "tool_a → tool_b", "relation": "explicit"}},\n'
                f'  {{"pair": "tool_a → tool_c", "relation": "implicit"}},\n'
                f'  ...\n'
                f']}}\n\n'
                f"Output ONLY the JSON, nothing else:"
            )

            try:
                raw = self.client.generate_chat(
                    [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
                    temperature=0.1,
                    max_tokens=2048,
                )
                import json as _json
                data = _extract_json(raw)
                for entry in data.get("classifications", []):
                    pair_key = entry.get("pair", "")
                    relation = entry.get("relation", "none")
                    if relation in ("explicit", "implicit"):
                        all_classifications[pair_key] = relation
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
            for neighbor in graph.get(current, {}).get("explicit", []):
                if neighbor in visited:
                    continue
                new_path = path + [neighbor]
                if len(new_path) >= 2:
                    chains.append(new_path)
                _dfs(neighbor, new_path, visited | {neighbor})
            # Also explore implicit edges
            for neighbor in graph.get(current, {}).get("implicit", [])[:2]:
                if neighbor in visited:
                    continue
                new_path = path + [neighbor]
                if len(new_path) >= 2:
                    chains.append(new_path)
                _dfs(neighbor, new_path, visited | {neighbor})

        for start_node in graph:
            _dfs(start_node, [start_node], {start_node})

        # Deduplicate by sorted tuple
        unique: list[list[str]] = []
        seen: set[tuple] = set()
        for c in chains:
            key = tuple(c)
            if key not in seen:
                seen.add(key)
                unique.append(c)

        logger.debug(f"_extract_dependency_chains: {server_name} → {len(unique)} chains")
        return unique

    def _get_chains(self, server_name: str) -> list[list[str]]:
        """Return cached dependency chains for *server_name*, extracting if needed."""
        if server_name not in self._domain_chains:
            self._domain_chains[server_name] = self._extract_dependency_chains(server_name)
        return self._domain_chains[server_name]

    def _get_graph_hints(self, server_name: str) -> str:
        """Return cached dependency hints for *server_name*, probing if needed."""
        if server_name not in self._domain_graphs:
            graph = self._probe_dependency_graph(server_name)
            self._domain_graphs[server_name] = graph
        return _format_graph_hints(self._domain_graphs[server_name])


def _minimal_args(tool_schema: dict) -> dict[str, Any]:
    """Build minimal valid arguments for a tool's required parameters."""
    args: dict[str, Any] = {}
    props = tool_schema.get("input_schema", {}).get("properties", {})
    required = tool_schema.get("input_schema", {}).get("required", [])
    for param in required:
        info = props.get(param, {})
        ptype = info.get("type", "string")
        if "enum" in info:
            args[param] = info["enum"][0]
        elif ptype == "string":
            args[param] = ""
        elif ptype in ("integer", "number"):
            args[param] = 0
        elif ptype == "boolean":
            args[param] = False
        elif ptype == "array":
            args[param] = []
        elif ptype == "object":
            args[param] = {}
        else:
            args[param] = ""
    return args


def _is_query_tool(name: str) -> bool:
    return any(w in name for w in (
        "list", "search", "get", "view", "read", "show", "find", "query",
        "lookup", "check", "browse",
    )) or name in ("ls", "cat", "pwd", "stat", "head", "tail")


def _tool_entity(name: str) -> str:
    for et in ("event", "order", "account", "email", "invoice",
                "issue", "lead", "deal", "product", "restaurant",
                "channel", "message", "file", "contact", "payment",
                "menu", "cart", "transfer", "transaction"):
        if et in name:
            return et
    return name.split("_")[-1] if "_" in name else name


def _collect_fields(obj: Any, fields: set[str], prefix: str = "") -> None:
    """Recursively collect field names from a nested dict/list."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            full = f"{prefix}{k}"
            fields.add(k)
            fields.add(full)
            if isinstance(v, (dict, list)):
                _collect_fields(v, fields, f"{full}.")
    elif isinstance(obj, list) and len(obj) > 0:
        _collect_fields(obj[0], fields, prefix)


def _merge_yaml_dependency_edges(
    graph: dict[str, dict],
    server_name: str,
    suite_config: Any,
) -> None:
    """Merge pre-defined dependency edges from domain yaml into *graph* (mutates in-place).

    Edges are defined in configs/live_mcp/{domain}.yaml under dependency_graph.edges.
    Each entry has source_tool, target_tool, and relation (explicit|implicit).
    """
    if suite_config is None:
        return
    for server_cfg in getattr(suite_config, "servers", []):
        if getattr(server_cfg, "name", "") != server_name:
            continue
        edges = getattr(server_cfg, "dependency_graph", {}).get("edges", [])
        if not edges:
            return
        n_added = 0
        for edge in edges:
            src = edge.get("source_tool", "")
            tgt = edge.get("target_tool", "")
            rel = edge.get("relation", "implicit")
            if not src or not tgt:
                continue
            if src not in graph:
                graph[src] = {"explicit": [], "implicit": []}
            if tgt not in graph:
                graph[tgt] = {"explicit": [], "implicit": []}
            if tgt not in graph[src][rel]:
                graph[src][rel].append(tgt)
                n_added += 1
        if n_added > 0:
            logger.debug(
                f"_merge_yaml_dependency_edges: {server_name} "
                f"+{n_added} yaml-defined edges"
            )
        return


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
