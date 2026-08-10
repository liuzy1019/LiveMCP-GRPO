"""Canonical in-memory LiveTask projection from an accepted candidate."""

from __future__ import annotations

from typing import Any

from src.live_mcp.types import LiveTask


def build_live_task(
    *,
    suite_config: Any,
    registry: Any,
    teacher_client: Any,
    server_name: str,
    query: str,
    session_id: str,
    seed: int,
    all_tools: list[dict],
    oracle_program: Any,
    required_tools: list[str],
    difficulty: str,
    task_id: str,
    conversation_queries: list[str] | None = None,
    oracle_calls_per_round: list[list] | None = None,
    execution_history_per_round: list[list] | None = None,
    sampling_context: dict[str, Any] | None = None,
) -> LiveTask:
    domain_tools = registry.server_tools(server_name)
    visible_tools = (
        domain_tools
        if domain_tools or not required_tools
        else [tool for tool in all_tools if tool["name"] in required_tools]
    )
    return LiveTask(
        task_id=task_id,
        source="live_mcp_task_planner",
        suite_name=suite_config.suite_name,
        user_prompt=query,
        session_id=session_id,
        session_seed=seed,
        target_servers=[server_name],
        visible_tools=visible_tools,
        required_tools=list(required_tools),
        expected_outcome={
            "success_criteria": oracle_program.success_criteria,
        },
        success_criteria=list(oracle_program.success_criteria),
        oracle_program=oracle_program,
        sampling_context=sampling_context or {},
        max_turns=int(suite_config.rollout.get("max_turns", 8)),
        difficulty=difficulty,
        task_type="task_planner",
        metadata={
            "generation_method": "task_planner",
            "teacher_model_id": str(
                getattr(
                    teacher_client,
                    "contract_model_id",
                    getattr(teacher_client, "model_path", "unknown"),
                )
            ),
        },
        conversation_queries=conversation_queries or [],
        oracle_calls_per_round=oracle_calls_per_round or [],
        execution_history_per_round=execution_history_per_round or [],
    )
