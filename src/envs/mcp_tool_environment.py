"""MCPToolEnvironment — verl rollout 环境接口。

将 EpisodeSeed + ReplayMCPExecutor + ActionParser + ComponentReward 组合为
verl 可调用的环境。

职责：
  1. 管理 episode 生命周期（reset → step → done）
  2. 构建 prompt（system + tools_snapshot + initial_messages + history）
  3. 接收 model output → parse → 调 executor → 返回 observation
  4. 收集 step-level reward info 供 TrajectoryVerifier 使用
  5. 支持 batch 操作（多个 episode 并行）

与 verl 的集成点：
  - verl 的 rollout worker 调用 env.reset() 获取初始 prompt
  - 每次 model 生成后调用 env.step(output) 获取 observation + done
  - episode 结束后调用 env.get_reward_info() 获取 reward 计算所需信息
"""

import copy
from dataclasses import dataclass, field
from typing import Any, Optional

from src.data.episode_schema import ActionType, EpisodeSeed, OracleStep
from src.envs.replay_mcp_executor import ExecutionResult, MatchConfig, ReplayMCPExecutor
from src.reward.action_parser import ActionParser, ParsedAction, parse_action
from src.reward.component_reward import OracleAction, SampleMetadata


@dataclass
class StepInfo:
    """单步信息（用于 trajectory-level reward 计算）。"""
    step_idx: int
    model_output: str                   # 模型原始输出
    parsed_action: Optional[ParsedAction] = None
    execution_result: Optional[ExecutionResult] = None
    oracle_step: Optional[OracleStep] = None
    # reward 计算所需
    oracle_action: Optional[OracleAction] = None
    metadata: Optional[SampleMetadata] = None


@dataclass
class EnvState:
    """单个 episode 的环境状态。"""
    episode: EpisodeSeed
    executor: ReplayMCPExecutor
    step_infos: list[StepInfo] = field(default_factory=list)
    messages: list[dict] = field(default_factory=list)  # 完整对话历史
    done: bool = False
    truncated: bool = False             # 超过 max_turns 被截断


@dataclass
class EnvConfig:
    """环境配置。"""
    # parser 模式：RL rollout 默认 strict
    strict_parse: bool = True
    # replay rollout 默认要求参数值匹配，避免错误参数释放 oracle observation。
    strict_replay_values: bool = True
    # 最大轮数（超过后强制截断）
    max_turns: int = 5
    # system prompt 模板
    system_prompt_template: str = (
        "You are a helpful assistant with access to the following tools. "
        "Use them when needed to answer the user's question.\n\n"
        "Available tools:\n{tools_description}\n\n"
        "Response format:\n"
        "- To call a tool: <tool_call>{{\"name\": \"tool_name\", \"arguments\": {{...}}}}</tool_call>\n"
        "- To give final answer: <final_answer>your answer</final_answer>\n"
        "- To report error: <report_error>error description</report_error>\n"
        "- To ask clarification: <ask_clarification>your question</ask_clarification>"
    )
    # observation 模板
    observation_template: str = "Tool output:\n{observation}"
    # match config
    match_config: Optional[MatchConfig] = None


class MCPToolEnvironment:
    """verl rollout 环境。

    Usage:
        env = MCPToolEnvironment(config)
        prompt = env.reset(episode_seed)
        while not env.done:
            model_output = model.generate(prompt)
            observation, done = env.step(model_output)
            if not done:
                prompt = env.build_next_prompt(observation)
        reward_info = env.get_reward_info()
    """

    def __init__(self, config: Optional[EnvConfig] = None):
        self.config = config or EnvConfig()
        self.parser = ActionParser(strict=self.config.strict_parse)
        self._state: Optional[EnvState] = None

    @property
    def done(self) -> bool:
        return self._state.done if self._state else True

    @property
    def current_step(self) -> int:
        return len(self._state.step_infos) if self._state else 0

    def reset(self, episode: EpisodeSeed) -> str:
        """重置环境，返回初始 prompt。

        Args:
            episode: EpisodeSeed 实例。

        Returns:
            完整的初始 prompt 字符串（system + tools + user message）。
        """
        match_config = (
            copy.deepcopy(self.config.match_config)
            if self.config.match_config is not None
            else MatchConfig(strict_values=self.config.strict_replay_values)
        )
        # 从 episode metadata 中提取 name_map / enum_map（如果有）
        if episode.metadata.get("name_map"):
            match_config.name_map = episode.metadata["name_map"]
        if episode.metadata.get("enum_map"):
            match_config.enum_map = episode.metadata["enum_map"]

        executor = ReplayMCPExecutor(
            episode=episode,
            match_config=match_config,
        )

        # 构建初始 messages
        messages = self._build_initial_messages(episode)

        self._state = EnvState(
            episode=episode,
            executor=executor,
            messages=list(messages),
        )

        return self._messages_to_prompt(messages)

    def step(self, model_output: str) -> tuple[str, bool]:
        """执行一步。

        Args:
            model_output: 模型生成的原始文本。

        Returns:
            (observation_or_empty, done)
            - 如果 done=True，observation 为空
            - 如果 done=False，observation 是下一轮的 tool output
        """
        if self._state is None or self._state.done:
            return "", True

        state = self._state

        # 检查 max_turns
        if self.current_step >= self.config.max_turns:
            state.done = True
            state.truncated = True
            return "", True

        # 解析 model output
        parsed = self.parser.parse(model_output)

        # 获取当前 oracle step（如果还有的话）
        oracle_step = None
        if state.executor.current_step < state.executor.total_steps:
            oracle_step = state.episode.oracle_trace[state.executor.current_step]

        # 构建 executor 输入
        action_dict = self._parsed_to_action_dict(parsed)

        # 执行
        exec_result = state.executor.step(action_dict)

        # 记录 step info
        step_info = StepInfo(
            step_idx=self.current_step,
            model_output=model_output,
            parsed_action=parsed,
            execution_result=exec_result,
            oracle_step=oracle_step,
            oracle_action=self._oracle_step_to_action(oracle_step) if oracle_step else None,
            metadata=self._build_sample_metadata(),
        )
        state.step_infos.append(step_info)

        # 更新 messages
        state.messages.append({"role": "assistant", "content": model_output})

        if exec_result.done:
            state.done = True
            return "", True

        # 未结束 → 返回 observation
        observation = self.config.observation_template.format(
            observation=exec_result.observation
        )
        state.messages.append({"role": "tool", "content": observation})

        return observation, False

    def get_reward_info(self) -> dict[str, Any]:
        """获取 reward 计算所需的全部信息。

        Returns:
            包含 step_infos、episode metadata、truncated 等信息的 dict。
        """
        if self._state is None:
            return {"step_infos": [], "done": True, "truncated": False}

        state = self._state
        return {
            "episode_id": state.episode.episode_id,
            "episode_type": state.episode.episode_type,
            "step_infos": state.step_infos,
            "total_steps": len(state.step_infos),
            "oracle_total_steps": state.executor.total_steps,
            "done": state.done,
            "truncated": state.truncated,
            "all_matched": all(
                si.execution_result.matched
                for si in state.step_infos
                if si.execution_result is not None
            ),
            "perturbation_level": state.episode.metadata.get("perturbation_level", "none"),
            "scenario_type": state.episode.episode_type,
            "metadata": {
                "name_map": state.executor.match_config.name_map,
                "enum_map": state.executor.match_config.enum_map,
            },
        }

    def get_messages(self) -> list[dict]:
        """获取完整对话历史。"""
        return self._state.messages if self._state else []

    def _build_initial_messages(self, episode: EpisodeSeed) -> list[dict]:
        """构建初始 messages（system + user）。"""
        messages = []

        # System prompt（含 tools description）
        tools_desc = self._format_tools_description(episode.tools_snapshot)
        system_content = self.config.system_prompt_template.format(
            tools_description=tools_desc
        )
        messages.append({"role": "system", "content": system_content})

        # Initial messages（来自 episode）
        for msg in episode.initial_messages:
            messages.append(msg)

        return messages

    def _format_tools_description(self, tools: list[dict]) -> str:
        """格式化工具描述。"""
        lines = []
        for tool in tools:
            name = tool.get("name", tool.get("function", {}).get("name", "unknown"))
            desc = tool.get("description", tool.get("function", {}).get("description", ""))
            params = tool.get("parameters", tool.get("function", {}).get("parameters", {}))

            lines.append(f"- {name}: {desc}")
            if params and isinstance(params, dict):
                properties = params.get("properties", {})
                required = params.get("required", [])
                for pname, pinfo in properties.items():
                    req_mark = " (required)" if pname in required else ""
                    ptype = pinfo.get("type", "any")
                    pdesc = pinfo.get("description", "")
                    lines.append(f"    - {pname} ({ptype}{req_mark}): {pdesc}")

        return "\n".join(lines)

    def _parsed_to_action_dict(self, parsed: ParsedAction) -> dict[str, Any]:
        """将 ParsedAction 转为 executor 需要的 action dict。"""
        action = {"action_type": parsed.action_type}

        if parsed.action_type == "tool_call":
            action["tool_calls"] = parsed.tool_calls
        elif parsed.action_type in ("final_answer", "report_error", "ask_clarification"):
            action["content"] = parsed.content
        # unparseable → 保持 action_type="unparseable"，executor 会处理

        return action

    def _oracle_step_to_action(self, oracle_step: OracleStep) -> OracleAction:
        """将 OracleStep 转为 ComponentReward 需要的 OracleAction。"""
        if oracle_step.action_type == ActionType.TOOL_CALL.value:
            return OracleAction(
                action_type="tool_call",
                tool_calls=[{
                    "name": oracle_step.tool_name,
                    "arguments": oracle_step.arguments,
                }],
                match_mode="set",
            )
        elif oracle_step.action_type == ActionType.PARALLEL_TOOL_CALL.value:
            return OracleAction(
                action_type="tool_call",
                tool_calls=[
                    {"name": c["tool_name"], "arguments": c.get("arguments", {})}
                    for c in oracle_step.calls
                ],
                match_mode=oracle_step.match_mode,
            )
        elif oracle_step.action_type == ActionType.FINAL_ANSWER.value:
            return OracleAction(
                action_type="final_answer",
                final_answer=oracle_step.expected_content,
            )
        elif oracle_step.action_type == ActionType.REPORT_ERROR.value:
            return OracleAction(
                action_type="report_error",
                error_info=oracle_step.expected_content,
            )
        elif oracle_step.action_type == ActionType.ASK_CLARIFICATION.value:
            return OracleAction(
                action_type="ask_clarification",
            )
        else:
            return OracleAction(action_type=oracle_step.action_type)

    def _build_sample_metadata(self) -> SampleMetadata:
        """构建当前 step 的 SampleMetadata。"""
        state = self._state
        return SampleMetadata(
            name_map=state.executor.match_config.name_map,
            enum_map=state.executor.match_config.enum_map,
            perturbation_level=state.episode.metadata.get("perturbation_level", "none"),
            scenario_type=state.episode.episode_type,
        )

    @staticmethod
    def _messages_to_prompt(messages: list[dict]) -> str:
        """将 messages 列表转为单个 prompt 字符串。

        这是一个简化实现；实际 verl 集成时会使用 tokenizer 的 chat template。
        """
        parts = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "system":
                parts.append(f"<|system|>\n{content}")
            elif role == "user":
                parts.append(f"<|user|>\n{content}")
            elif role == "assistant":
                parts.append(f"<|assistant|>\n{content}")
            elif role == "tool":
                parts.append(f"<|tool|>\n{content}")
            else:
                parts.append(content)
        return "\n".join(parts)
