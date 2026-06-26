"""Public facade for the optional Live MCP branch.

This module is intentionally not imported by the replay GRPO training path.
Use it from explicit live-MCP scripts or tests only.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from src.live_mcp.agent_loop import AgentLoopConfig, MCPToolsAgentLoop, OracleGenerationBackend
from src.live_mcp.config import SuiteConfig, load_suite_config
from src.live_mcp.executor import LiveMCPExecutor
from src.live_mcp.manager import LiveMCPManager
from src.live_mcp.trace import TraceRecorder
from src.live_mcp.types import LiveTask, RolloutTrace, live_task_from_dict, to_plain
from src.reward.action_parser import ActionParser


@dataclass
class TaskGenerationSummary:
    tasks_requested: int
    tasks_written: int
    oracle_valid_rate: float
    has_calendar_tasks: bool
    has_shopping_tasks: bool
    has_missing_function_tasks: bool
    has_distractor_tasks: bool
    output: str

    def to_dict(self) -> dict[str, object]:
        return {
            "tasks_requested": self.tasks_requested,
            "tasks_written": self.tasks_written,
            "oracle_valid_rate": self.oracle_valid_rate,
            "has_calendar_tasks": self.has_calendar_tasks,
            "has_shopping_tasks": self.has_shopping_tasks,
            "has_missing_function_tasks": self.has_missing_function_tasks,
            "has_distractor_tasks": self.has_distractor_tasks,
            "output": self.output,
        }


@dataclass
class LiveSmokeSummary:
    sessions_created: int
    rollouts_finished: int
    trace_files_written: int
    subprocess_stdio_used: bool
    reward_score_mean: float
    reward_score_mean_is_finite: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "sessions_created": self.sessions_created,
            "rollouts_finished": self.rollouts_finished,
            "trace_files_written": self.trace_files_written,
            "subprocess_stdio_used": self.subprocess_stdio_used,
            "reward_score_mean": self.reward_score_mean,
            "reward_score_mean_is_finite": self.reward_score_mean_is_finite,
        }


class LiveMCPBranch:
    """Optional live-MCP branch facade.

    The class owns live server lifecycle. It is safe to instantiate from live
    scripts, but should not be imported by the default replay training path.
    """

    def __init__(self, suite_config: SuiteConfig):
        self.suite_config = suite_config
        self.manager = LiveMCPManager(suite_config)
        self._started = False
        self.executor: LiveMCPExecutor | None = None

    @classmethod
    def from_suite(cls, suite_path: str | Path) -> "LiveMCPBranch":
        return cls(load_suite_config(suite_path))

    def __enter__(self) -> "LiveMCPBranch":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    def start(self) -> None:
        if self._started:
            return
        self.manager.start_suite()
        self.executor = LiveMCPExecutor(self.manager, self.manager.registry)
        self._started = True

    def stop(self) -> None:
        if not self._started:
            return
        self.manager.stop_suite()
        self.executor = None
        self._started = False

    def generate_tasks_llm(
        self,
        *,
        server_name: str,
        count: int,
        seed: int,
        difficulty_mix: dict[str, float] | None = None,
        model_path: str = "models/Qwen3-4B",
        api_base: str | None = None,
        irrelevance_ratio: float = 0.05,
        distractor_rate: float = 0.40,
        missing_function_rate: float = 0.20,
    ) -> list[LiveTask]:
        """Generate tasks with PROVE-style two-phase teacher."""
        self._require_started()
        assert self.executor is not None

        from src.live_mcp.llm_client import LLMClient
        from src.live_mcp.orchestrator import TaskOrchestrator

        if api_base:
            client = LLMClient(
                mode="openai", model_path=model_path, api_base=api_base,
            )
        else:
            client = LLMClient(mode="local", model_path=model_path)

        orchestrator = TaskOrchestrator(
            self.suite_config, self.manager, self.executor, client,
        )
        return orchestrator.generate_many(
            server_name=server_name,
            count=count,
            seed=seed,
            difficulty_mix=difficulty_mix or {"complete": 0.6, "missing": 0.2, "minimal": 0.2},
            distractor_rate=distractor_rate,
            missing_function_rate=missing_function_rate,
            irrelevance_ratio=irrelevance_ratio,
        )

    def run_oracle_smoke(
        self,
        *,
        tasks: Iterable[LiveTask],
        server_name: str,
        num_tasks: int,
        seed: int,
        trace_dir: str | Path | None = None,
    ) -> tuple[list[RolloutTrace], LiveSmokeSummary]:
        self._require_started()
        assert self.executor is not None
        selected = [
            task
            for task in tasks
            if server_name == "all" or server_name in task.target_servers
        ][:num_tasks]
        recorder = TraceRecorder(
            trace_dir
            or self.suite_config.live.get("trace_dir")
            or self.suite_config.trace.get("output_dir", "data/live_mcp/traces")
        )
        loop = MCPToolsAgentLoop(
            manager=self.manager,
            executor=self.executor,
            parser=ActionParser(strict=False),
            trace_recorder=recorder,
            config=AgentLoopConfig(max_turns=int(self.suite_config.rollout.get("max_turns", 8))),
        )
        traces: list[RolloutTrace] = []
        for idx, task in enumerate(selected):
            session = self.manager.create_session(seed=seed + idx)
            self.manager.discover_tools(session.session_id)
            task.session_id = session.session_id
            task.session_seed = seed + idx
            task.visible_tools = _visible_existing_tools(
                task.visible_tools,
                self.manager.registry.all_tools(),
            )
            try:
                traces.append(loop.rollout(task, OracleGenerationBackend(task)))
            finally:
                self.manager.close_session(session.session_id)
        return traces, summarize_smoke_traces(
            traces,
            subprocess_stdio_used=self.manager.subprocess_stdio_used,
        )

    def _require_started(self) -> None:
        if not self._started:
            raise RuntimeError("LiveMCPBranch.start() must be called before use")


def save_live_tasks(tasks: Iterable[LiveTask], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for task in tasks:
            f.write(json.dumps(to_plain(task), ensure_ascii=True) + "\n")
    return path


def load_live_tasks(path: str | Path) -> list[LiveTask]:
    tasks: list[LiveTask] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                tasks.append(live_task_from_dict(json.loads(line)))
    return tasks


def summarize_generated_tasks(
    tasks: list[LiveTask],
    tasks_requested: int,
    output_path: str | Path,
) -> TaskGenerationSummary:
    return TaskGenerationSummary(
        tasks_requested=tasks_requested,
        tasks_written=len(tasks),
        oracle_valid_rate=1.0 if tasks else 0.0,
        has_calendar_tasks=any("calendar" in task.target_servers for task in tasks),
        has_shopping_tasks=any("shopping" in task.target_servers for task in tasks),
        has_missing_function_tasks=any(task.task_type == "missing_function" for task in tasks),
        has_distractor_tasks=any(task.metadata.get("has_distractors") for task in tasks),
        output=str(output_path),
    )


def summarize_smoke_traces(
    traces: list[RolloutTrace],
    subprocess_stdio_used: bool = False,
) -> LiveSmokeSummary:
    scores = [trace.reward.get("score", 0.0) for trace in traces]
    return LiveSmokeSummary(
        sessions_created=len(traces),
        rollouts_finished=len(traces),
        trace_files_written=len(traces),
        subprocess_stdio_used=subprocess_stdio_used,
        reward_score_mean=sum(scores) / len(scores) if scores else 0.0,
        reward_score_mean_is_finite=all(math.isfinite(score) for score in scores),
    )


def _visible_existing_tools(
    task_tools: list[dict[str, object]],
    all_tools: list[dict[str, object]],
) -> list[dict[str, object]]:
    all_by_name = {tool["name"]: tool for tool in all_tools if "name" in tool}
    return [
        all_by_name[tool["name"]]
        for tool in task_tools
        if tool.get("name") in all_by_name
    ]
