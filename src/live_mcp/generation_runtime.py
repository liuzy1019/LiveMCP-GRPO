"""Runtime facade used by Teacher data-generation and cache scripts."""

from __future__ import annotations

from pathlib import Path

from src.live_mcp.config import SuiteConfig, load_suite_config
from src.live_mcp.executor import LiveMCPExecutor
from src.live_mcp.manager import LiveMCPManager
from src.live_mcp.types import LiveTask


class TeacherGenerationRuntime:
    """Own the Live MCP server lifecycle used during Teacher generation."""

    def __init__(self, suite_config: SuiteConfig):
        self.suite_config = suite_config
        self.manager = LiveMCPManager(suite_config)
        self._started = False
        self.executor: LiveMCPExecutor | None = None

    @classmethod
    def from_suite(cls, suite_path: str | Path) -> "TeacherGenerationRuntime":
        return cls(load_suite_config(suite_path))

    def __enter__(self) -> "TeacherGenerationRuntime":
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

    def generate_tasks(
        self,
        *,
        server_name: str,
        count: int,
        seed: int,
        difficulty_mix: dict[str, float] | None = None,
        model_path: str = "models/Qwen/Qwen3-4B",
        api_base: str | None = None,
        device: int | None = None,
        irrelevance_ratio: float = 0.05,
        distractor_rate: float = 0.40,
        missing_function_rate: float = 1500 / (10895 + 1500),
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
            client = LLMClient(mode="local", model_path=model_path, device=device)

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

    def _require_started(self) -> None:
        if not self._started:
            raise RuntimeError(
                "TeacherGenerationRuntime.start() must be called before use"
            )
