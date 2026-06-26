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
import random
from typing import Any

from loguru import logger

from src.live_mcp.config import SuiteConfig
from src.live_mcp.dedup import dedup_tasks
from src.live_mcp.executor import LiveMCPExecutor
from src.live_mcp.manager import LiveMCPManager
from src.live_mcp.types import LiveTask, OracleCall, OracleProgram, to_plain


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
        self._domain_graphs: dict[str, dict] = {}  # cached dependency graphs per domain

    def generate_one(
        self,
        server_name: str,
        seed: int,
        difficulty: str,
        max_turns: int = 8,
        max_retries: int = 3,
    ) -> LiveTask:
        """PROVE-style state-machine generation with LLM-in-the-loop.

        1. LLM generates user_query
        2. Loop: LLM decides next action (tool_call with args, or terminal)
           → execute tool against live MCP → apply perturbation → record
        3. Derive success criteria from state delta
        4. Replay validate against fresh session
        """
        from src.live_mcp.task_planner import (
            TaskPlanner, derive_success_criteria, replay_validate, apply_perturbation,
        )
        from src.live_mcp.types import ToolCall

        teacher = TaskPlanner(self.client, server_name)
        rng = random.Random(seed)

        # ── PROVE Step 1: auto-discover dependency graph (cached per domain) ──
        dep_hints = self._get_graph_hints(server_name)

        # ── Step 2: generate user query ──
        session = self.manager.create_session(seed=seed)
        session_id = session.session_id
        all_tools = self.manager.discover_tools(session_id)
        server_tools = [
            tool for tool in all_tools
            if self.manager.registry.server_for_tool(tool["name"]) == server_name
        ]

        try:
            grounded_state = self.manager.get_state(session_id)
            domain_state = grounded_state.get(server_name, {})
            initial_state_snapshot = copy.deepcopy(domain_state)

            user_query = teacher.generate_query(
                tool_schemas=server_tools,
                grounded_state=domain_state,
                difficulty=difficulty,
                rng=rng,
                dep_hints=dep_hints,
            )

            # ── Step 2-N: state-machine loop ──
            oracle_calls: list[OracleCall] = []
            execution_history: list[dict[str, Any]] = []
            required_tools: set[str] = set()
            task_id = f"{server_name}_{seed}_{rng.randint(0, 99999)}"

            for turn in range(max_turns):
                action = teacher.decide_action(
                    tool_schemas=server_tools,
                    user_query=user_query,
                    execution_history=execution_history,
                    attempt=turn,
                    dep_hints=dep_hints,
                )

                if action.action in ("final_answer", "report_error", "ask_clarification"):
                    break  # terminal

                if action.action != "tool_call" or not action.tool_name:
                    continue  # unparseable → skip, LLM will retry next turn

                tool_name = action.tool_name
                tool_name = _fuzzy_match_tool(tool_name, {t["name"] for t in server_tools}) or tool_name
                required_tools.add(tool_name)

                # Execute the tool call
                result = self.executor.execute(
                    session_id,
                    ToolCall(tool_name, dict(action.arguments), call_id=f"sm_{turn}"),
                )

                # Apply execution perturbation (PROVE-style)
                perturbed_obs = apply_perturbation(
                    result.observation, server_name, rng,
                )
                if isinstance(perturbed_obs, dict) and perturbed_obs.get("retry"):
                    # Intermittent error → oracle retries, don't record as oracle call yet
                    execution_history.append({
                        "tool_name": tool_name,
                        "arguments": dict(action.arguments),
                        "observation": perturbed_obs,
                        "success": False,
                    })
                    continue

                execution_history.append({
                    "tool_name": tool_name,
                    "arguments": dict(action.arguments),
                    "observation": perturbed_obs if perturbed_obs is not None else result.observation,
                    "success": result.success,
                })
                if not result.success:
                    continue  # LLM sees the failure, can retry. Not part of oracle.

                oracle_calls.append(OracleCall(
                    tool_name=tool_name,
                    arguments=dict(action.arguments),
                ))

            # ── Step: derive success criteria from state delta ──
            final_state_full = self.manager.get_state(session_id)
            final_state = final_state_full.get(server_name, {})
            success_criteria = derive_success_criteria(
                initial_state=initial_state_snapshot,
                final_state=final_state,
                oracle_calls=oracle_calls,
                domain=server_name,
            )

            # ── Step: replay validate ──
            valid = replay_validate(
                oracle_calls=oracle_calls,
                manager=self.manager,
                executor=self.executor,
                seed=seed,
                domain=server_name,
            )
            if not valid and oracle_calls:
                raise RuntimeError(
                    f"Replay validation failed for {server_name} task {task_id}"
                )

        finally:
            self.manager.close_session(session_id)

        oracle_program = OracleProgram(
            task_id=task_id,
            calls=oracle_calls,
            success_criteria=success_criteria,
        )

        return self._to_live_task(
            server_name=server_name, query=user_query,
            session_id=session_id, seed=seed,
            all_tools=all_tools, oracle_program=oracle_program,
            required_tools=sorted(required_tools),
            difficulty=difficulty, task_id=task_id,
        )

    def generate_many(self, server_name: str, count: int, seed: int,
                      difficulty_mix: dict[str, float] | None = None,
                      irrelevance_ratio: float = 0.05,
                      distractor_rate: float = 0.40,
                      missing_function_rate: float = 0.20,
                      ) -> list[LiveTask]:
        tasks: list[LiveTask] = []
        servers = self.manager.server_names if server_name == "all" else [server_name]
        if not servers:
            raise ValueError("no enabled Live MCP servers available")
        unknown = [s for s in servers if s not in self.manager.server_names]
        if unknown:
            raise ValueError(f"unknown servers: {unknown}")

        effective_mix = difficulty_mix or {"complete": 0.6, "missing": 0.2, "minimal": 0.2}

        # Pre-count irrelevance tasks (proportional, no forced minimum)
        n_irrelevant = round(count * irrelevance_ratio) if irrelevance_ratio > 0 else 0
        n_normal = count - n_irrelevant

        idx = 0
        failed = 0
        domain_failures: dict[str, int] = {}
        domain_cooldown_until: dict[str, int] = {}
        # After 3 consecutive failures, skip the domain for this many rounds
        # before retrying (increases with each cooldown cycle)
        base_cooldown = len(servers) * 2

        # ── normal task generation ──
        while len(tasks) < n_normal and failed < count * 2:
            current_server = servers[idx % len(servers)]
            cooldown_end = domain_cooldown_until.get(current_server, 0)
            if idx < cooldown_end:
                idx += 1
                continue
            if idx >= cooldown_end and cooldown_end > 0:
                # Cooldown expired, reset failure count and let it retry
                domain_cooldown_until.pop(current_server, None)
                domain_failures[current_server] = 0
            difficulty = self._pick_difficulty(seed + idx, effective_mix)
            try:
                task = self.generate_one(
                    current_server, seed=seed + idx, difficulty=difficulty,
                )
                # Robustness knobs at configurable rates
                rng_knob = random.Random(seed + idx)
                if rng_knob.random() < distractor_rate:
                    self._apply_distractors(task)
                if rng_knob.random() < missing_function_rate:
                    self._apply_missing_function(task)
                tasks.append(task)
                if len(tasks) % 10 == 0:
                    logger.info(
                        f"generate_many progress: {len(tasks)}/{n_normal} tasks, "
                        f"{failed} failures"
                    )
            except Exception as e:
                failed += 1
                domain_failures[current_server] = domain_failures.get(current_server, 0) + 1
                logger.warning(
                    f"generate failed for {current_server} "
                    f"({domain_failures[current_server]}x): {e}"
                )
                if domain_failures[current_server] >= 3:
                    cooldown = base_cooldown * (2 ** (domain_failures[current_server] // 3 - 1))
                    domain_cooldown_until[current_server] = idx + cooldown
                    logger.warning(
                        f"cooling down {current_server} for {cooldown} rounds "
                        f"(retry at round {idx + cooldown})"
                    )
            idx += 1

        # ── irrelevance tasks (5%) ──
        irr = self._generate_irrelevant_tasks(n_irrelevant, seed + 9999)
        tasks.extend(irr)

        # ── dedup across all generated tasks ──
        before = len(tasks)
        tasks = dedup_tasks(tasks, threshold=0.70)
        removed = before - len(tasks)

        logger.info(
            f"LLM teacher: {len(tasks)} tasks (target {count}, {failed} failures, "
            f"{removed} dedup removed)"
        )
        return tasks

    def _to_live_task(self, server_name: str, query: str, session_id: str, seed: int,
                      all_tools: list[dict], oracle_program, required_tools: list[str],
                      difficulty: str, task_id: str) -> LiveTask:
        visible_tools = [t for t in all_tools if t["name"] in required_tools]
        return LiveTask(
            task_id=f"{server_name}_{seed}_{task_id}", source="live_mcp_task_planner",
            suite_name=self.suite_config.suite_name, user_prompt=query,
            session_id=session_id, session_seed=seed, target_servers=[server_name],
            visible_tools=visible_tools, required_tools=list(required_tools),
            expected_outcome={"success_criteria": oracle_program.success_criteria},
            success_criteria=list(oracle_program.success_criteria),
            oracle_program=oracle_program, sampling_context={},
            max_turns=int(self.suite_config.rollout.get("max_turns", 8)),
            difficulty=difficulty, task_type="task_planner",
            metadata={"generation_method": "task_planner"},
        )

    def _apply_distractors(self, task: LiveTask) -> None:
        known = {t["name"] for t in task.visible_tools}
        candidates = [t for t in self.manager.registry.all_tools() if t["name"] not in known]
        rng = random.Random(hash(task.task_id))
        selected = rng.sample(candidates, min(len(candidates), rng.randint(3, 8)))
        task.visible_tools.extend(selected)
        task.metadata["has_distractors"] = True
        task.metadata["distractor_count"] = len(selected)

    def _apply_missing_function(self, task: LiveTask) -> None:
        if not task.required_tools:
            return
        hidden = task.required_tools[-1]
        missing = {"type": "missing_function", "server": task.target_servers[0], "tool": hidden}
        task.metadata["original_required_tools"] = list(task.required_tools)
        task.metadata["original_success_criteria"] = list(task.success_criteria)
        task.metadata["original_oracle_program"] = to_plain(task.oracle_program)
        task.hidden_tools.append(hidden)
        task.visible_tools = [t for t in task.visible_tools if t["name"] != hidden]
        task.required_tools = [n for n in task.required_tools if n != hidden]
        task.success_criteria = [missing]
        task.expected_outcome = {"success_criteria": [missing], "abstain": True}
        task.oracle_program.calls = []
        task.oracle_program.success_criteria = [missing]
        task.task_type = "missing_function"
        task.metadata["has_missing_function"] = True
        task.metadata["unavailable_required_tool"] = hidden

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
                visible_tools=self.manager.registry.all_tools(),
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
        from src.live_mcp.task_planner import TaskPlanner
        domain_desc = TaskPlanner.DOMAIN_DESCRIPTIONS.get(server_name, "")

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


    def _probe_dependency_graph(self, server_name: str) -> dict:
        """PROVE Step 1: auto-discover tool dependencies via live MCP probing.

        Executes query/list tools against a fresh session, records which entity
        IDs appear in each tool's output, then classifies directed edges:
          explicit: tool A's output fields match tool B's required params
          implicit: A and B operate on the same entity type
        """
        session = self.manager.create_session(seed=0)
        try:
            all_tools = self.manager.discover_tools(session.session_id)
            server_tools = [
                t for t in all_tools
                if self.manager.registry.server_for_tool(t["name"]) == server_name
            ]
        except Exception:
            self.manager.close_session(session.session_id)
            return {}

        graph: dict[str, dict] = {}
        query_outputs: dict[str, set[str]] = {}

        try:
            # Execute read-only query tools and capture output field names
            for tool in server_tools:
                name = tool["name"]
                graph[name] = {"explicit": [], "implicit": []}
                if not _is_query_tool(name):
                    continue
                try:
                    from src.live_mcp.types import ToolCall
                    probe_args = _minimal_args(tool)
                    result = self.executor.execute(
                        session.session_id,
                        ToolCall(name, probe_args, call_id=f"probe_{name}"),
                    )
                    if result.success and isinstance(result.observation, dict):
                        fields: set[str] = set()
                        _collect_fields(result.observation, fields)
                        query_outputs[name] = fields
                except Exception:
                    pass

            # Classify edges
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
            for a_name in graph:
                if not graph[a_name]["explicit"]:
                    a_entity = _tool_entity(a_name)
                    for b_name in graph:
                        if b_name != a_name and _tool_entity(b_name) == a_entity:
                            graph[a_name]["implicit"].append(b_name)
        finally:
            self.manager.close_session(session.session_id)

        return graph

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
    ))


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
