"""Query generation and missing-function action-contract setup."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Callable

from loguru import logger

from src.live_mcp.fsm import ConversationFSM, FSMStateGroup, RobustnessPlan
from src.live_mcp.generation.robustness import (
    build_teacher_visible_tools as _build_teacher_visible_tools,
)
from src.live_mcp.task_spec import TaskSpec


@dataclass
class GeneratedQueryContract:
    generated_query: Any
    user_query: str
    blocked_tools: set[str] | None
    chain_seed: list[str] | None
    chain_context: dict[str, Any]
    teacher_visible_tools: list[dict[str, Any]]
    retry_candidate: bool = False


def generate_query_contract(
    *,
    teacher: Any,
    conversation_fsm: ConversationFSM,
    teacher_visible_tools: list[dict[str, Any]],
    query_teacher_visible_tools: list[dict[str, Any]],
    query_grounding_state: dict[str, Any],
    difficulty: str,
    local_rng: random.Random,
    dep_hints: Any,
    persona: str,
    reference_date: str,
    source_chain_seed: list[str] | None,
    query_chain_context: dict[str, Any],
    task_spec: TaskSpec | None,
    plan: RobustnessPlan,
    server_tools: list[dict[str, Any]],
    server_name: str,
    trace_generation: Callable[..., None],
) -> GeneratedQueryContract:
    generated_query = teacher.generate_query(
        tool_schemas=teacher_visible_tools,
        grounded_state=query_grounding_state,
        difficulty=difficulty,
        rng=local_rng,
        dep_hints=dep_hints,
        persona=persona,
        reference_date=reference_date,
        chain_seed=source_chain_seed,
        chain_context=query_chain_context,
        task_spec=(task_spec.to_prompt_contract() if task_spec is not None else None),
        initial_goal_spec=None,
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

    blocked_tools: set[str] | None = None
    chain_seed = source_chain_seed
    chain_context = query_chain_context
    if plan.missing_function:
        if not source_chain_seed:
            logger.debug(
                "Missing-function requires a dependency chain for {}; "
                "retrying with a fresh seed.",
                server_name,
            )
            return GeneratedQueryContract(
                generated_query=generated_query,
                user_query=user_query,
                blocked_tools=None,
                chain_seed=None,
                chain_context={},
                teacher_visible_tools=teacher_visible_tools,
                retry_candidate=True,
            )
        hidden_tool = source_chain_seed[-1]
        plan.hidden_tool = hidden_tool
        teacher_visible_tools = _build_teacher_visible_tools(server_tools, plan)
        blocked_tools = {hidden_tool}
        chain_seed = None
        chain_context = {}

    trace_generation(
        "action_candidate_contract",
        query_candidate_tool_names=[
            str(tool.get("name") or "") for tool in query_teacher_visible_tools
        ],
        action_candidate_tools=teacher_visible_tools,
        hidden_tools=sorted(blocked_tools or set()),
        enum_stripped=plan.strip_enums,
        distractor_tool_names=[
            str(tool.get("name") or "") for tool in plan.distractor_tools
        ],
    )
    return GeneratedQueryContract(
        generated_query=generated_query,
        user_query=user_query,
        blocked_tools=blocked_tools,
        chain_seed=chain_seed,
        chain_context=chain_context,
        teacher_visible_tools=teacher_visible_tools,
    )
