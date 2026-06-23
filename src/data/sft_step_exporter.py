"""SFT Step Exporter — 将 EpisodeSeed 展开为 SFT 训练格式。

Phase 3 核心产出：将 EpisodeSeed 的 oracle_trace 展开为 SFT 训练样本。

设计原则（来自 AGENTS.md）：
  - SFT cold start 只用于 action format / JSON / schema-following 对齐
  - 样本必须从 episode_seed 的可见上下文导出
  - 不能把 oracle_trace 放进模型输入
  - 模型输出格式必须与 action_parser.py 一致

展开策略：
  对于一个 N 步的 episode，展开为 N 条训练样本：
    - 样本 0: prompt = [system, user] + tools → completion = step_0 action
    - 样本 1: prompt = [system, user, step_0_action, observation_0] + tools → completion = step_1 action
    - 样本 k: prompt = [system, user, ..., step_{k-1}_action, observation_{k-1}] + tools → completion = step_k action

  这样模型学会：
    1. 正确的 <tool_call>...</tool_call> 格式
    2. 正确的 JSON 参数填充
    3. 读取 observation 后决定下一步
    4. 并行调用的格式
    5. final_answer 的格式
"""

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from loguru import logger

from src.data.episode_schema import (
    ActionType,
    EpisodeSeed,
    EpisodeType,
    OracleStep,
    load_episodes,
)


# ============================================================
# 配置
# ============================================================

# System prompt 模板：告诉模型它是一个 MCP tool-use agent
SYSTEM_PROMPT = """You are a helpful assistant with access to MCP tools. When you need to use a tool, output your action in the following format:

For a single tool call:
<tool_call>{"name": "tool_name", "arguments": {"param1": "value1", "param2": "value2"}}</tool_call>

For parallel tool calls (multiple tools in one turn):
<tool_call>[{"name": "tool1", "arguments": {...}}, {"name": "tool2", "arguments": {...}}]</tool_call>

When you have enough information to answer the user's question directly:
<final_answer>Your answer here</final_answer>

When you need more information from the user:
<ask_clarification>Your question here</ask_clarification>

When a tool returns an error and you cannot proceed:
<report_error>Description of the error and what went wrong</report_error>

Important rules:
- Always use the exact tool names from the available tools list.
- Fill all required parameters with appropriate values.
- Read tool outputs carefully before deciding your next action.
- Do not call tools unnecessarily if you already have the answer."""


@dataclass
class ExporterConfig:
    """SFT Step Exporter 配置。"""
    # 输入
    episode_seeds_path: str = "data/toucan/episode_seeds.jsonl"
    # 输出
    output_path: str = "data/sft/sft_train.jsonl"
    stats_output_path: str = "data/sft/export_stats.json"
    # 格式选项
    include_system_prompt: bool = True
    # 工具 schema 展示格式
    tools_in_system: bool = True  # 工具 schema 放在 system prompt 中（当前唯一支持的模式）
    max_tools_in_prompt: int = 20  # 最多展示多少个工具 schema
    # 过滤
    max_completion_length: int = 2048  # completion 最大字符数
    max_prompt_length: int = 10240  # prompt 最大字符数（估算）
    skip_no_tool_episodes: bool = False  # 是否跳过 no_tool 类型
    # 多步展开
    expand_all_steps: bool = True  # True: 展开所有步骤; False: 只取第一步
    # observation 截断
    max_observation_in_context: int = 1024  # 上下文中 observation 的最大长度


# ============================================================
# 格式化工具
# ============================================================


def _format_tools_section(tools_snapshot: list[dict], max_tools: int = 20, oracle_tool_names: list[str] | None = None) -> str:
    """将工具 schema 列表格式化为文本。

    保证 oracle_tool_names 中的工具一定在可见列表中。
    """
    if not tools_snapshot:
        return ""

    # 优先包含 oracle tools，然后填充其余
    if oracle_tool_names:
        oracle_set = set(oracle_tool_names)
        oracle_tools = []
        other_tools = []
        for tool in tools_snapshot:
            func = tool.get("function", tool)
            name = func.get("name", "")
            if name in oracle_set:
                oracle_tools.append(tool)
            else:
                other_tools.append(tool)
        # oracle tools 优先，剩余 slots 填充 other tools
        remaining_slots = max(0, max_tools - len(oracle_tools))
        tools_to_show = oracle_tools + other_tools[:remaining_slots]
    else:
        tools_to_show = tools_snapshot[:max_tools]
    lines = ["## Available Tools\n"]

    for tool in tools_to_show:
        func = tool.get("function", tool)
        name = func.get("name", "unknown")
        desc = func.get("description", "")
        params = func.get("parameters", {})

        lines.append(f"### {name}")
        if desc:
            lines.append(f"{desc}")

        # 参数
        properties = params.get("properties", {})
        required = params.get("required", [])
        if properties:
            lines.append("Parameters:")
            for param_name, param_info in properties.items():
                param_type = param_info.get("type", "any")
                param_desc = param_info.get("description", "")
                req_mark = " (required)" if param_name in required else ""
                enum_vals = param_info.get("enum", [])
                enum_str = f" [enum: {', '.join(str(v) for v in enum_vals)}]" if enum_vals else ""
                lines.append(f"  - {param_name}: {param_type}{req_mark}{enum_str} — {param_desc}")

        lines.append("")

    if len(tools_snapshot) > max_tools:
        lines.append(f"... and {len(tools_snapshot) - max_tools} more tools")

    return "\n".join(lines)


def _format_tool_call_completion(step: OracleStep) -> str:
    """将 tool_call OracleStep 格式化为模型 completion。"""
    if step.action_type == ActionType.TOOL_CALL.value:
        call_obj = {"name": step.tool_name, "arguments": step.arguments}
        return f'<tool_call>{json.dumps(call_obj, ensure_ascii=False)}</tool_call>'

    elif step.action_type == ActionType.PARALLEL_TOOL_CALL.value:
        calls = []
        for call in step.calls:
            calls.append({
                "name": call["tool_name"],
                "arguments": call.get("arguments", {}),
            })
        return f'<tool_call>{json.dumps(calls, ensure_ascii=False)}</tool_call>'

    elif step.action_type == ActionType.FINAL_ANSWER.value:
        return f'<final_answer>{step.expected_content}</final_answer>'

    elif step.action_type == ActionType.ASK_CLARIFICATION.value:
        return f'<ask_clarification>{step.expected_content}</ask_clarification>'

    elif step.action_type == ActionType.REPORT_ERROR.value:
        return f'<report_error>{step.expected_content}</report_error>'

    else:
        return f'<final_answer>{step.expected_content}</final_answer>'


def _format_observation(step: OracleStep, max_length: int = 1024) -> str:
    """格式化 step 的 observation 为上下文文本。"""
    if step.action_type == ActionType.TOOL_CALL.value:
        obs = step.replay_observation
    elif step.action_type == ActionType.PARALLEL_TOOL_CALL.value:
        # 并行调用：合并所有 observation
        parts = []
        for i, obs_text in enumerate(step.replay_observations):
            call_name = step.calls[i]["tool_name"] if i < len(step.calls) else f"call_{i}"
            parts.append(f"[{call_name}]: {obs_text}")
        obs = "\n".join(parts)
    else:
        return ""

    # 截断
    if len(obs) > max_length:
        keep_each = (max_length - 50) // 2
        obs = f"{obs[:keep_each]}\n[...truncated {len(obs) - keep_each * 2} chars...]\n{obs[-keep_each:]}"

    return obs


# ============================================================
# SFT 样本数据结构
# ============================================================


@dataclass
class SFTSample:
    """一条 SFT 训练样本。

    使用 OpenAI chat format:
      messages = [system, user, assistant, tool, assistant, ...]
      最后一条 assistant message 的 content 是 completion（训练目标）
    """
    # 完整的 messages 列表（包含 completion）
    messages: list[dict] = field(default_factory=list)
    # 元数据
    episode_id: str = ""
    step_idx: int = 0
    action_type: str = ""
    episode_type: str = ""
    source: str = ""

    def to_dict(self) -> dict:
        """序列化为训练格式 dict。"""
        return {
            "messages": self.messages,
            "metadata": {
                "episode_id": self.episode_id,
                "step_idx": self.step_idx,
                "action_type": self.action_type,
                "episode_type": self.episode_type,
                "source": self.source,
            },
        }

    @property
    def prompt_messages(self) -> list[dict]:
        """除最后一条 assistant 外的所有 messages（prompt 部分）。"""
        return self.messages[:-1]

    @property
    def completion(self) -> str:
        """最后一条 assistant message 的 content（训练目标）。"""
        if self.messages and self.messages[-1].get("role") == "assistant":
            return self.messages[-1].get("content", "")
        return ""


# ============================================================
# Exporter 主类
# ============================================================


class SFTStepExporter:
    """将 EpisodeSeed 展开为 SFT 训练样本。

    Usage:
        exporter = SFTStepExporter(config)
        samples = exporter.export()
        exporter.save(samples)
    """

    def __init__(self, config: Optional[ExporterConfig] = None):
        self.config = config or ExporterConfig()
        self.stats = Counter()

    def export(self) -> list[SFTSample]:
        """从 episode seeds 文件导出 SFT 样本。

        Returns:
            SFTSample 列表。
        """
        episodes = load_episodes(self.config.episode_seeds_path)
        logger.info(f"加载 {len(episodes)} 条 episode seeds")
        return self.export_from_episodes(episodes)

    def export_from_episodes(self, episodes: list[EpisodeSeed]) -> list[SFTSample]:
        """从 EpisodeSeed 列表导出 SFT 样本。

        Args:
            episodes: EpisodeSeed 列表。

        Returns:
            SFTSample 列表。
        """
        samples: list[SFTSample] = []

        for ep in episodes:
            ep_samples = self._export_episode(ep)
            samples.extend(ep_samples)

        logger.info(
            f"导出完成: {len(samples)} 条 SFT 样本 "
            f"(来自 {len(episodes)} 条 episodes)"
        )
        self._log_stats()
        return samples

    def export_single(self, episode: EpisodeSeed) -> list[SFTSample]:
        """导出单个 episode 的所有 SFT 样本。"""
        return self._export_episode(episode)

    def save(self, samples: list[SFTSample]) -> None:
        """保存 SFT 样本到 JSONL 文件。"""
        path = Path(self.config.output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            for sample in samples:
                f.write(json.dumps(sample.to_dict(), ensure_ascii=False) + "\n")

        logger.info(f"保存 {len(samples)} 条 SFT 样本到 {path}")

        # 保存统计
        stats_path = Path(self.config.stats_output_path)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_report = self._build_stats_report(samples)
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats_report, f, ensure_ascii=False, indent=2)
        logger.info(f"保存统计到 {stats_path}")

    def _export_episode(self, episode: EpisodeSeed) -> list[SFTSample]:
        """展开单个 episode 为 SFT 样本列表。"""
        self.stats["episodes_processed"] += 1

        # 跳过 no_tool（可选）
        if self.config.skip_no_tool_episodes and episode.episode_type == EpisodeType.NO_TOOL.value:
            self.stats["skip_no_tool"] += 1
            return []

        # 构建 system prompt（含工具 schema）
        system_content = self._build_system_content(episode)

        # 逐步展开
        samples: list[SFTSample] = []
        context_messages: list[dict] = []  # 累积的对话上下文

        # 初始 user message（从 initial_messages 中提取，过滤 pre-tool assistant message）
        for msg in episode.initial_messages:
            # P1 修复：过滤第一个 tool_call 前的 assistant message，
            # 避免泄露前置话术干扰 SFT 目标
            if msg.get("role") == "assistant":
                continue
            context_messages.append(msg)

        # 展开每个 oracle step
        steps_to_export = episode.oracle_trace
        if not self.config.expand_all_steps:
            steps_to_export = episode.oracle_trace[:1]

        # 构建 tools_snapshot 中所有可用工具名集合（用于一致性校验）
        available_tool_names = set()
        for tool in (episode.tools_snapshot or []):
            func = tool.get("function", tool)
            available_tool_names.add(func.get("name", ""))

        for step_idx, step in enumerate(steps_to_export):
            # 一致性校验：step 使用的工具必须在 tools_snapshot 中
            step_tool_names = []
            if step.action_type == ActionType.TOOL_CALL.value:
                step_tool_names = [step.tool_name]
            elif step.action_type == ActionType.PARALLEL_TOOL_CALL.value:
                step_tool_names = [c.get("tool_name", "") for c in step.calls]

            missing_tools = [tn for tn in step_tool_names if tn and tn not in available_tool_names]
            if missing_tools:
                self.stats["skip_missing_tool_schema"] += 1
                self.stats["skip_missing_tool_calls"] += len(missing_tools)
                logger.debug(
                    f"截断 episode 于 step {step_idx} (episode={episode.episode_id}): "
                    f"工具 {missing_tools} 不在 tools_snapshot 中"
                )
                # 从该点截断：不导出该 step 及后续所有 step
                # 因为把不可见工具调用加入 context 会污染后续样本的 prompt 历史
                break
            # 构建 completion
            completion = _format_tool_call_completion(step)

            # 检查 completion 长度
            if len(completion) > self.config.max_completion_length:
                self.stats["skip_completion_too_long"] += 1
                break

            # 构建完整 messages
            messages = []

            # System prompt
            if self.config.include_system_prompt:
                messages.append({"role": "system", "content": system_content})

            # 对话上下文
            messages.extend(context_messages)

            # 最后一条 assistant message（completion / 训练目标）
            messages.append({"role": "assistant", "content": completion})

            # 检查 prompt 长度（粗略估算）
            prompt_length = sum(len(m.get("content", "")) for m in messages[:-1])
            if prompt_length > self.config.max_prompt_length:
                self.stats["skip_prompt_too_long"] += 1
                break

            # 创建样本
            sample = SFTSample(
                messages=messages,
                episode_id=episode.episode_id,
                step_idx=step_idx,
                action_type=step.action_type,
                episode_type=episode.episode_type,
                source=episode.source,
            )
            samples.append(sample)
            self.stats["samples_exported"] += 1
            self.stats[f"action_{step.action_type}"] += 1

            # 更新上下文：添加当前 action 和 observation
            # 模型的 action 作为 assistant message
            context_messages.append({"role": "assistant", "content": completion})

            # observation 作为 tool response（如果有）
            obs = _format_observation(step, self.config.max_observation_in_context)
            if obs:
                context_messages.append({"role": "tool", "content": obs})

        return samples

    def _build_system_content(self, episode: EpisodeSeed) -> str:
        """构建 system prompt 内容（含工具 schema）。"""
        parts = [SYSTEM_PROMPT]

        if self.config.tools_in_system and episode.tools_snapshot:
            # 收集 oracle trace 中使用的所有工具名，保证它们在可见列表中
            oracle_tool_names = episode.tool_names_used
            tools_section = _format_tools_section(
                episode.tools_snapshot,
                max_tools=self.config.max_tools_in_prompt,
                oracle_tool_names=oracle_tool_names,
            )
            parts.append(tools_section)

        return "\n\n".join(parts)

    def _log_stats(self) -> None:
        """输出统计信息。"""
        logger.info("=" * 60)
        logger.info("SFT Step Exporter 统计:")
        for key, value in sorted(self.stats.items()):
            logger.info(f"  {key}: {value}")
        logger.info("=" * 60)

    def _build_stats_report(self, samples: list[SFTSample]) -> dict:
        """构建统计报告。"""
        action_dist = Counter(s.action_type for s in samples)
        episode_type_dist = Counter(s.episode_type for s in samples)
        step_dist = Counter(s.step_idx for s in samples)

        # 长度统计
        completion_lengths = [len(s.completion) for s in samples]
        prompt_lengths = [
            sum(len(m.get("content", "")) for m in s.prompt_messages)
            for s in samples
        ]

        return {
            "total_samples": len(samples),
            "exporter_stats": dict(self.stats),
            "action_type_distribution": dict(action_dist),
            "episode_type_distribution": dict(episode_type_dist),
            "step_index_distribution": {str(k): v for k, v in sorted(step_dist.items())},
            "completion_length": {
                "min": min(completion_lengths) if completion_lengths else 0,
                "max": max(completion_lengths) if completion_lengths else 0,
                "mean": sum(completion_lengths) / len(completion_lengths) if completion_lengths else 0,
            },
            "prompt_length": {
                "min": min(prompt_lengths) if prompt_lengths else 0,
                "max": max(prompt_lengths) if prompt_lengths else 0,
                "mean": sum(prompt_lengths) / len(prompt_lengths) if prompt_lengths else 0,
            },
        }


# ============================================================
# CLI 入口
# ============================================================


def main():
    """命令行入口。"""
    import argparse

    parser = argparse.ArgumentParser(description="从 EpisodeSeed 导出 SFT 训练数据")
    parser.add_argument(
        "--input", "-i",
        default="data/toucan/episode_seeds.jsonl",
        help="EpisodeSeed JSONL 路径",
    )
    parser.add_argument(
        "--output", "-o",
        default="data/sft/sft_train.jsonl",
        help="输出 SFT JSONL 路径",
    )
    parser.add_argument(
        "--no-system",
        action="store_true",
        help="不包含 system prompt",
    )
    parser.add_argument(
        "--first-step-only",
        action="store_true",
        help="只导出每个 episode 的第一步",
    )
    parser.add_argument(
        "--skip-no-tool",
        action="store_true",
        help="跳过 no_tool 类型的 episode",
    )
    args = parser.parse_args()

    config = ExporterConfig(
        episode_seeds_path=args.input,
        output_path=args.output,
        include_system_prompt=not args.no_system,
        expand_all_steps=not args.first_step_only,
        skip_no_tool_episodes=args.skip_no_tool,
    )

    exporter = SFTStepExporter(config)
    samples = exporter.export()
    exporter.save(samples)

    print(f"\n{'='*60}")
    print(f"  SFT 数据导出完成")
    print(f"  输出: {args.output}")
    print(f"  总数: {len(samples)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
