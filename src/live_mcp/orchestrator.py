"""Finite state-machine orchestration for LiveTask generation."""

from __future__ import annotations

import random
from typing import Any

from src.live_mcp.config import SuiteConfig
from src.live_mcp.dependency_graph import DependencyGraphBuilder, edges_from_config
from src.live_mcp.executor import LiveMCPExecutor
from src.live_mcp.manager import LiveMCPManager
from src.live_mcp.oracle import OraclePlanner, OracleValidator
from src.live_mcp.query_generator import QueryGenerator, StructuredTask
from src.live_mcp.sampler import CalendarSampler, ShoppingSampler
from src.live_mcp.teacher import DeterministicTeacherAdapter
from src.live_mcp.types import LiveTask, to_plain


class StateMachineOrchestrator:
    STATES = [
        "INIT_SESSION",
        "DISCOVER_TOOLS",
        "BUILD_DEPENDENCY_GRAPH",
        "SAMPLE_STATE",
        "SELECT_CHAIN",
        "BIND_SLOTS",
        "RENDER_QUERY",
        "PLAN_ORACLE",
        "VALIDATE_ORACLE",
        "TEACHER_PROCESSING",
        "TOOL_EXECUTION",
        "RESPONSE_CLASSIFICATION",
        "RECOVERY_OR_CONTINUATION",
        "REPLAY_VALIDATE_TRACE",
        "WRITE_TASK",
    ]

    def __init__(
        self,
        suite_config: SuiteConfig,
        manager: LiveMCPManager,
        executor: LiveMCPExecutor,
    ):
        self.suite_config = suite_config
        self.manager = manager
        self.executor = executor
        self.graph_builder = DependencyGraphBuilder()
        self.query_generator = QueryGenerator(self._templates_by_task())
        self.teacher = DeterministicTeacherAdapter(self.query_generator)
        self.planner = OraclePlanner()
        self.validator = OracleValidator()
        self.samplers = {"calendar": CalendarSampler(), "shopping": ShoppingSampler()}

    def generate_one(self, server_name: str, seed: int, difficulty: str) -> LiveTask:
        rng = random.Random(seed)
        session = self.manager.create_session(seed=seed)
        tools = self.manager.discover_tools(session.session_id)
        try:
            server_tools = [tool for tool in tools if self.manager.registry.server_for_tool(tool["name"]) == server_name]
            server_config = self._server_config(server_name)
            graph = self.graph_builder.build(
                server_name,
                server_tools,
                edges_from_config(server_config.dependency_graph.get("edges", [])),
            )
            chains = self.graph_builder.extract_chains(graph)
            structured = self.samplers[server_name].sample_task(session.session_id, self.manager, difficulty, rng)
            if chains:
                structured.tool_chain = chains[0]
            query = self.teacher.generate_query(
                structured,
                sampling_context=structured.slots,
                persona="default",
                reference_date="2026-06-22",
                information_level="grounded",
            )
            validation = self.query_generator.validate_query(query, structured)
            if not validation.valid:
                raise ValueError(f"query validation failed: {validation}")
            oracle_program = self.planner.plan(structured)
            oracle_validation = self.validator.validate(
                structured,
                oracle_program,
                self.manager,
                self.executor,
                seed=seed,
            )
            if not oracle_validation.valid:
                raise ValueError(f"oracle validation failed: {oracle_validation.error}")
            task = self._to_live_task(structured, query, session.session_id, seed, tools, oracle_program)
            task.metadata["orchestrator_states"] = list(self.STATES)
            task.metadata["oracle_validated"] = True
            return task
        finally:
            self.manager.close_session(session.session_id)

    def generate_many(
        self,
        server_name: str,
        count: int,
        seed: int,
        difficulty_mix: dict[str, float],
    ) -> list[LiveTask]:
        tasks: list[LiveTask] = []
        servers = self.manager.server_names if server_name == "all" else [server_name]
        if not servers:
            raise ValueError("no enabled Live MCP servers available")
        unknown = [name for name in servers if name not in self.manager.server_names]
        if unknown:
            raise ValueError(f"unknown or disabled Live MCP server(s): {unknown}")
        idx = 0
        while len(tasks) < count:
            current_server = servers[idx % len(servers)]
            difficulty = self._pick_difficulty(seed + idx, difficulty_mix)
            task = self.generate_one(current_server, seed + idx, difficulty)
            if idx % 5 == 3:
                self._apply_distractors(task)
            if idx % 5 == 4:
                self._apply_missing_function(task)
            tasks.append(task)
            idx += 1
        return tasks

    def _to_live_task(
        self,
        structured: StructuredTask,
        query: str,
        session_id: str,
        seed: int,
        all_tools: list[dict[str, Any]],
        oracle_program: Any,
    ) -> LiveTask:
        visible_tools = [tool for tool in all_tools if tool["name"] in structured.required_tools]
        return LiveTask(
            task_id=f"{structured.server_name}_{seed}_{structured.task_id.split(':', 1)[0]}",
            source="live_mcp_state_machine",
            suite_name=self.suite_config.suite_name,
            user_prompt=query,
            session_id=session_id,
            session_seed=seed,
            target_servers=[structured.server_name],
            visible_tools=visible_tools,
            required_tools=list(structured.required_tools),
            expected_outcome={"success_criteria": structured.success_criteria},
            success_criteria=list(structured.success_criteria),
            oracle_program=oracle_program,
            sampling_context=dict(structured.slots),
            max_turns=int(self.suite_config.rollout.get("max_turns", 8)),
            difficulty=structured.difficulty,
            task_type=structured.task_id.split(":", 1)[0],
        )

    def _apply_distractors(self, task: LiveTask) -> None:
        known = {tool["name"] for tool in task.visible_tools}
        for tool in self.manager.registry.all_tools():
            if tool["name"] not in known and tool["name"] not in task.required_tools:
                task.visible_tools.append(tool)
                known.add(tool["name"])
            if len(task.visible_tools) >= len(task.required_tools) + 3:
                break
        task.metadata["has_distractors"] = True

    def _apply_missing_function(self, task: LiveTask) -> None:
        if not task.required_tools:
            return
        hidden = task.required_tools[-1]
        missing_criterion = {"type": "missing_function", "server": task.target_servers[0], "tool": hidden}
        task.metadata["original_required_tools"] = list(task.required_tools)
        task.metadata["original_success_criteria"] = list(task.success_criteria)
        task.metadata["original_oracle_program"] = to_plain(task.oracle_program)
        task.hidden_tools.append(hidden)
        task.visible_tools = [tool for tool in task.visible_tools if tool["name"] != hidden]
        task.required_tools = [tool_name for tool_name in task.required_tools if tool_name != hidden]
        task.success_criteria = [missing_criterion]
        task.expected_outcome = {"success_criteria": [missing_criterion], "abstain": True}
        task.oracle_program.calls = []
        task.oracle_program.success_criteria = [missing_criterion]
        task.task_type = "missing_function"
        task.metadata["has_missing_function"] = True
        task.metadata["unavailable_required_tool"] = hidden

    def _templates_by_task(self) -> dict[str, list[str]]:
        templates: dict[str, list[str]] = {}
        for cfg in self.suite_config.servers:
            templates.update(cfg.query_templates)
        return templates

    def _server_config(self, server_name: str):
        for cfg in self.suite_config.servers:
            if cfg.name == server_name:
                return cfg
        raise KeyError(server_name)

    def _pick_difficulty(self, seed: int, difficulty_mix: dict[str, float]) -> str:
        if not difficulty_mix:
            return "easy"
        rng = random.Random(seed)
        threshold = rng.random()
        cumulative = 0.0
        for name, weight in difficulty_mix.items():
            cumulative += weight
            if threshold <= cumulative:
                return name
        return next(iter(difficulty_mix))
