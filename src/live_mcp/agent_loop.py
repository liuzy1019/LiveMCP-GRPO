"""Offline multi-turn Live MCP agent loop.

.. deprecated::
    This module is kept for backward compatibility with ``src.live_mcp.api``
    (offline replay / validation path).  The primary GRPO rollout path uses
    ``src.agent_loop.livemcp_oval_loop.LiveMCPOvalLoop`` instead.
    Do not add new features here; migrate consumers to the Oval loop.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from typing import Protocol

from src.live_mcp import errors
from src.live_mcp.executor import LiveMCPExecutor
from src.live_mcp.manager import LiveMCPManager
from src.live_mcp.reward import RewardComposer
from src.live_mcp.trace import TraceRecorder, prompt_hash
from src.live_mcp.types import LiveTask, RolloutTrace, ToolCall, TraceTurn
from src.reward.action_parser import ActionParser


class GenerationBackend(Protocol):
    model_name: str

    def generate(self, prompt: str) -> str: ...


@dataclass
class AgentLoopConfig:
    max_turns: int = 8
    observation_max_chars: int = 4096
    stop_on_parse_error: bool = True
    stop_on_execution_error: bool = False
    allow_parallel_tool_calls: bool = False


class OracleGenerationBackend:
    model_name = "deterministic_oracle"

    def __init__(self, task: LiveTask):
        self.task = task
        self.idx = 0

    def generate(self, prompt: str) -> str:
        if self.task.task_type == "missing_function":
            self.idx += 1
            missing_tool = self.task.metadata.get("unavailable_required_tool") or (self.task.hidden_tools[0] if self.task.hidden_tools else "required tool")
            return f"<report_error>Required tool {missing_tool} is not available.</report_error>"
        if self.idx < len(self.task.oracle_program.calls):
            call = self.task.oracle_program.calls[self.idx]
            self.idx += 1
            return f"<tool_call>{json.dumps({'name': call.tool_name, 'arguments': call.arguments}, ensure_ascii=True)}</tool_call>"
        return "<final_answer>Done.</final_answer>"


class MCPToolsAgentLoop:
    def __init__(
        self,
        manager: LiveMCPManager,
        executor: LiveMCPExecutor,
        parser: ActionParser,
        trace_recorder: TraceRecorder,
        config: AgentLoopConfig,
        reward_composer: RewardComposer | None = None,
    ):
        warnings.warn(
            "MCPToolsAgentLoop is deprecated. "
            "Use src.agent_loop.livemcp_oval_loop.LiveMCPOvalLoop for GRPO rollout.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.manager = manager
        self.executor = executor
        self.parser = parser
        self.trace_recorder = trace_recorder
        self.config = config
        self.reward_composer = reward_composer or RewardComposer()

    def rollout(self, task: LiveTask, model: GenerationBackend) -> RolloutTrace:
        trace = self.trace_recorder.start(task, getattr(model, "model_name", "model"))
        observations: list[str] = []
        final_status = "max_turns"
        for turn_idx in range(self.config.max_turns):
            prompt = self._build_prompt(task, observations)
            output = model.generate(prompt)
            parsed = self.parser.parse(output)
            tool_calls: list[ToolCall] = []
            execution_results = []
            done = False
            observation_text = ""
            if parsed.action_type == "unparseable":
                final_status = errors.PARSE_ERROR
                done = self.config.stop_on_parse_error
            elif parsed.action_type == "final_answer":
                final_status = "model_final"
                done = True
            elif parsed.action_type == "tool_call":
                if len(parsed.tool_calls) > 1 and not self.config.allow_parallel_tool_calls:
                    final_status = errors.PARALLEL_NOT_SUPPORTED
                    done = True
                else:
                    for idx, call in enumerate(parsed.tool_calls):
                        tool_calls.append(
                            ToolCall(
                                name=call["name"],
                                arguments=call.get("arguments", {}),
                                call_id=f"{turn_idx}_{idx}",
                                raw_text=output,
                            )
                        )
                    execution_results = self.executor.execute_many(task.session_id, tool_calls)
                    observation_text = json.dumps(
                        [r.observation if r.success else {"error": r.error_type, "message": r.error_message} for r in execution_results],
                        ensure_ascii=True,
                    )[: self.config.observation_max_chars]
                    observations.append(observation_text)
                    if any(not r.success for r in execution_results) and self.config.stop_on_execution_error:
                        final_status = "execution_failed"
                        done = True
            else:
                final_status = parsed.action_type
                done = True
            self.trace_recorder.append_turn(
                trace,
                TraceTurn(
                    turn_idx=turn_idx,
                    prompt_hash=prompt_hash(prompt),
                    model_output=output,
                    parsed_action_type=parsed.action_type,
                    tool_calls=tool_calls,
                    execution_results=execution_results,
                    observation_text=observation_text,
                    done=done,
                ),
            )
            if done:
                break
        final_state = self.manager.get_state(task.session_id)
        reward = self.reward_composer.compute(task, trace, final_state=final_state)
        if _expects_abstention(task) and reward.get("component_abstention", 0.0) >= 1.0:
            final_status = "success"
        elif final_status == "model_final" and reward["component_coverage"] >= 1.0:
            final_status = "success"
        elif final_status == "model_final":
            final_status = "model_final_incorrect"
        self.trace_recorder.finish(trace, final_status, reward)
        self.trace_recorder.save(trace)
        return trace

    def _build_prompt(self, task: LiveTask, observations: list[str]) -> str:
        tools = json.dumps(task.visible_tools, ensure_ascii=True)
        obs = "\n".join(observations)
        return f"User: {task.user_prompt}\nTools: {tools}\nObservations:\n{obs}\nAssistant:"


def _expects_abstention(task: LiveTask) -> bool:
    return task.task_type == "missing_function" or any(criterion.get("type") == "missing_function" for criterion in task.success_criteria)
