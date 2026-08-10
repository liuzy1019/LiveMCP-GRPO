"""Per-attempt Teacher, session, and robustness setup."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from src.live_mcp.fsm import ConversationFSM, RobustnessPlan
from src.live_mcp.generation.robustness import (
    build_teacher_visible_tools as _build_teacher_visible_tools,
)
from src.live_mcp.task_planner import TaskPlanner


@dataclass
class CandidateAttemptSetup:
    teacher: Any
    conversation_fsm: ConversationFSM
    session: Any
    session_id: str
    all_tools: list[dict[str, Any]]
    server_tools: list[dict[str, Any]]
    plan: RobustnessPlan
    teacher_visible_tools: list[dict[str, Any]]
    query_teacher_visible_tools: list[dict[str, Any]]
    trace_generation: Callable[..., None]


def setup_candidate_attempt(
    *,
    orchestrator: Any,
    server_name: str,
    local_seed: int,
    sampling_state_seed: int,
    robustness_plan: RobustnessPlan | None,
    retry_attempt: int,
    difficulty: str,
) -> CandidateAttemptSetup:
    teacher = TaskPlanner(
        orchestrator.client,
        server_name,
        seed=local_seed,
        max_observation_chars=int(
            orchestrator.suite_config.rollout.get(
                "observation_max_chars", 4096,
            )
        ),
        prompt_profile=orchestrator.prompt_profile,
    )
    conversation_fsm = ConversationFSM()

    def trace_generation(stage: str, **payload: Any) -> None:
        recorder = getattr(teacher, "record_environment_event", None)
        if callable(recorder):
            recorder(stage, **payload)

    session_servers = [server_name]
    if robustness_plan is not None:
        session_servers.extend(
            str(tool.get("_server_name") or "")
            for tool in robustness_plan.distractor_tools
            if str(tool.get("_server_name") or "")
        )
    session = orchestrator.manager.create_session(
        seed=sampling_state_seed,
        server_names=list(dict.fromkeys(session_servers)),
    )
    session_id = session.session_id
    all_tools = orchestrator.manager.discover_tools(session_id)
    server_tools = orchestrator.manager.registry.server_tools(server_name)

    if robustness_plan is None:
        plan = RobustnessPlan()
    else:
        plan = RobustnessPlan(
            inject_distractors=robustness_plan.inject_distractors,
            distractor_tools=list(robustness_plan.distractor_tools),
            strip_enums=robustness_plan.strip_enums,
            missing_function=robustness_plan.missing_function,
            hidden_tool=None,
            irrelevance=robustness_plan.irrelevance,
        )
    teacher_visible_tools = _build_teacher_visible_tools(server_tools, plan)
    query_teacher_visible_tools = list(teacher_visible_tools)
    trace_generation(
        "generation_setup",
        retry_attempt=retry_attempt,
        session_id=session_id,
        server_name=server_name,
        difficulty=difficulty,
        robustness_plan={
            "inject_distractors": plan.inject_distractors,
            "distractor_tool_names": [
                str(tool.get("name") or "")
                for tool in plan.distractor_tools
            ],
            "strip_enums": plan.strip_enums,
            "missing_function": plan.missing_function,
            "irrelevance": plan.irrelevance,
        },
        clean_server_tools=server_tools,
        query_candidate_tools=query_teacher_visible_tools,
    )
    return CandidateAttemptSetup(
        teacher=teacher,
        conversation_fsm=conversation_fsm,
        session=session,
        session_id=session_id,
        all_tools=all_tools,
        server_tools=server_tools,
        plan=plan,
        teacher_visible_tools=teacher_visible_tools,
        query_teacher_visible_tools=query_teacher_visible_tools,
        trace_generation=trace_generation,
    )
