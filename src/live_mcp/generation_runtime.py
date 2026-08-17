"""Runtime facade used by Teacher data-generation and cache scripts."""

from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from src.live_mcp.config import SuiteConfig, load_suite_config
from src.live_mcp.executor import LiveMCPExecutor
from src.live_mcp.generation.mix_policy import default_difficulty_mix
from src.live_mcp.protocol.manager import LiveMCPManager
from src.live_mcp.types import LiveTask
from src.live_mcp.generation.chain_scheduler import chain_fingerprint


class TeacherGenerationRuntime:
    """Own the Live MCP server lifecycle used during Teacher generation."""

    def __init__(self, suite_config: SuiteConfig):
        self.suite_config = suite_config
        self.manager = LiveMCPManager(suite_config)
        self._started = False
        self.executor: LiveMCPExecutor | None = None
        # Shared across generate_tasks() recovery calls in this runtime. Shard
        # processes remain isolated and final diversity is still merge-owned.
        self._chain_sampling_stats: dict[str, dict[str, dict[str, int]]] = {}
        self._chain_sampling_sequences: dict[str, dict[str, tuple[str, ...]]] = {}
        self._chain_sampling_lock = threading.RLock()

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

    def preload_retained_sequences(
        self, sequences_by_domain: dict[str, list[list[str]]],
    ) -> None:
        """Seed scheduling state with sequences retained by the prior merge."""
        if self._started:
            raise RuntimeError("retained sequences must be loaded before runtime start")
        for domain, sequences in sequences_by_domain.items():
            if domain not in self.manager.server_names:
                raise ValueError(f"unknown retained-sequence domain: {domain!r}")
            domain_stats = self._chain_sampling_stats.setdefault(domain, {})
            domain_sequences = self._chain_sampling_sequences.setdefault(domain, {})
            for raw_sequence in sequences:
                if not isinstance(raw_sequence, list) or not raw_sequence or not all(
                    isinstance(name, str) and name for name in raw_sequence
                ):
                    raise ValueError(
                        f"invalid retained tool sequence for {domain}: {raw_sequence!r}"
                    )
                fingerprint = chain_fingerprint(domain, raw_sequence)
                domain_sequences[fingerprint] = tuple(raw_sequence)
                domain_stats[fingerprint] = {
                    "attempted": 1,
                    "accepted": 1,
                    "rejected_goal": 0,
                }

    def generate_tasks(
        self,
        *,
        server_name: str,
        count: int,
        seed: int,
        difficulty_mix: dict[str, float] | None = None,
        model_path: str = "models/Google/Gemma-4-31B-it",
        teacher_model_id: str | None = None,
        api_base: str | None = None,
        device: int | None = None,
        irrelevance_ratio: float = 0.05,
        irrelevance_count: int | None = None,
        distractor_rate: float = 0.40,
        missing_function_rate: float = 1500 / (10895 + 1500),
        prompt_profile: str = "paper_generation_baseline_v1",
        fixed_attempt_budget: bool | None = None,
        progress_callback: Callable[[list[LiveTask]], None] | None = None,
        failure_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> list[LiveTask]:
        """Generate tasks with the two-phase teacher."""
        self._require_started()
        assert self.executor is not None

        from src.live_mcp.llm_client import LLMClient
        from src.live_mcp.orchestrator import TaskOrchestrator

        if api_base:
            client = LLMClient(
                mode="openai",
                model_path=model_path,
                contract_model_id=teacher_model_id,
                api_base=api_base,
            )
        else:
            client = LLMClient(
                mode="local",
                model_path=model_path,
                contract_model_id=teacher_model_id,
                device=device,
            )

        orchestrator = TaskOrchestrator(
            self.suite_config,
            self.manager,
            self.executor,
            client,
            prompt_profile=prompt_profile,
            chain_sampling_stats=self._chain_sampling_stats,
            chain_sampling_sequences=self._chain_sampling_sequences,
            chain_sampling_lock=self._chain_sampling_lock,
        )
        return orchestrator.generate_many(
            server_name=server_name,
            count=count,
            seed=seed,
            difficulty_mix=difficulty_mix or default_difficulty_mix(),
            distractor_rate=distractor_rate,
            missing_function_rate=missing_function_rate,
            irrelevance_ratio=irrelevance_ratio,
            irrelevance_count=irrelevance_count,
            fixed_attempt_budget=fixed_attempt_budget,
            progress_callback=progress_callback,
            failure_callback=failure_callback,
        )

    def _require_started(self) -> None:
        if not self._started:
            raise RuntimeError(
                "TeacherGenerationRuntime.start() must be called before use"
            )
