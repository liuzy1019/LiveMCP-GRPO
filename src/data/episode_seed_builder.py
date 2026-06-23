"""EpisodeSeed Builder — 从 Toucan 数据构建 EpisodeSeed。

Phase 2 核心产出：将 Toucan 原始数据转换为标准化的 EpisodeSeed 格式。

转换逻辑:
  1. 解析 Toucan 的 messages/tools 字段
  2. 提取 oracle trace（tool_call + tool_response 对）
  3. 分类 episode_type
  4. 提取 MCP server 信息
  5. 构建 initial_messages（模型可见部分）
  6. 验证并输出

过滤规则:
  - 必须 parseable（messages + tools 都能解析）
  - 必须有 target_tools（知道正确答案）
  - tool_call 和 tool_response 必须配对
  - arguments 必须可解析
  - irrelevant 子集单独处理为 no_tool 类型
"""

import ast
import json
import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from loguru import logger

from src.data.episode_schema import (
    ActionType,
    DataSource,
    EpisodeSeed,
    EpisodeType,
    OracleStep,
    SplitKeys,
    VerifierConfig,
    save_episodes,
    validate_episodes,
)


# ============================================================
# 配置
# ============================================================


@dataclass
class BuilderConfig:
    """EpisodeSeed Builder 配置。"""
    # 输入路径
    toucan_data_path: str = "data/toucan/toucan_sft_subset_5000.jsonl"
    # 输出路径
    output_path: str = "data/toucan/episode_seeds.jsonl"
    stats_output_path: str = "data/toucan/builder_stats.json"
    # 过滤
    max_decision_turns: int = 5       # 最大决策轮数（超过的截断或丢弃）
    min_decision_turns: int = 1       # 最小决策轮数
    require_paired: bool = True       # 要求 tool_call/response 配对
    require_args_parseable: bool = True  # 要求 arguments 可解析
    # 处理选项
    include_irrelevant_as_no_tool: bool = True  # irrelevant 子集作为 no_tool
    truncate_long_episodes: bool = True  # 截断超长 episode（否则丢弃）
    max_observation_length: int = 2048  # replay_observation 最大长度
    # 随机种子
    seed: int = 42


# ============================================================
# 解析工具（复用 inspect_toucan.py 的逻辑）
# ============================================================


def _parse_json_field(value: Any) -> Any:
    """解析可能是 JSON 字符串的字段。"""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, ValueError):
            pass
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            pass
    return value


def _parse_tool_call_content(content: str) -> Optional[dict]:
    """解析 tool_call 消息的 content。

    Toucan 格式: {'name': 'tool_name', 'arguments': '{...}'}
    """
    parsed = _parse_json_field(content)
    if isinstance(parsed, dict) and "name" in parsed:
        args = parsed.get("arguments", {})
        if isinstance(args, str):
            try:
                parsed["arguments_parsed"] = json.loads(args)
            except (json.JSONDecodeError, ValueError):
                try:
                    parsed["arguments_parsed"] = ast.literal_eval(args)
                except (ValueError, SyntaxError):
                    parsed["arguments_parsed"] = None
        else:
            parsed["arguments_parsed"] = args
        return parsed
    return None


def _extract_mcp_server(tool_name: str) -> str:
    """从工具名中提取 MCP server。

    Toucan 工具名格式: server-name-tool_name
    """
    parts = tool_name.rsplit("-", 1)
    if len(parts) >= 2:
        return parts[0]
    return tool_name


def _truncate_observation(obs: str, max_length: int) -> str:
    """截断过长的 observation。

    保留头尾，中间用 truncation marker 替代。
    """
    if len(obs) <= max_length:
        return obs

    keep_each = (max_length - 50) // 2  # 50 chars for marker
    head = obs[:keep_each]
    tail = obs[-keep_each:]
    truncated_chars = len(obs) - keep_each * 2
    return f"{head}\n[...truncated {truncated_chars} chars...]\n{tail}"


# ============================================================
# Builder 主类
# ============================================================


class EpisodeSeedBuilder:
    """从 Toucan 数据构建 EpisodeSeed。

    Usage:
        builder = EpisodeSeedBuilder(config)
        episodes = builder.build()
        builder.save(episodes)
    """

    def __init__(self, config: Optional[BuilderConfig] = None):
        self.config = config or BuilderConfig()
        self.stats = Counter()
        self.rng = random.Random(self.config.seed)

    def build(self) -> list[EpisodeSeed]:
        """从 Toucan 数据构建 EpisodeSeed 列表。

        Returns:
            验证通过的 EpisodeSeed 列表。
        """
        # 加载数据
        data = self._load_data()
        logger.info(f"加载 Toucan 数据: {len(data)} 条")

        # 逐条转换
        episodes: list[EpisodeSeed] = []
        for item in data:
            ep = self._convert_item(item)
            if ep is not None:
                episodes.append(ep)

        success_rate = (len(episodes) / len(data) * 100) if data else 0.0
        logger.info(
            f"构建完成: {len(episodes)} / {len(data)} 条成功 "
            f"({success_rate:.1f}%)"
        )
        self._log_stats()

        return episodes

    def build_from_data(self, data: list[dict]) -> list[EpisodeSeed]:
        """从已加载的数据列表构建 EpisodeSeed。

        Args:
            data: Toucan 原始数据列表。

        Returns:
            验证通过的 EpisodeSeed 列表。
        """
        episodes: list[EpisodeSeed] = []
        for item in data:
            ep = self._convert_item(item)
            if ep is not None:
                episodes.append(ep)
        return episodes

    def save(self, episodes: list[EpisodeSeed]) -> None:
        """保存 episodes 和统计信息。"""
        # 保存 episodes
        count = save_episodes(episodes, self.config.output_path)
        logger.info(f"保存 {count} 条 episode seeds 到 {self.config.output_path}")

        # 保存统计
        stats_path = Path(self.config.stats_output_path)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_dict = self._build_stats_report(episodes)
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats_dict, f, ensure_ascii=False, indent=2)
        logger.info(f"保存统计到 {self.config.stats_output_path}")

    def _load_data(self) -> list[dict]:
        """加载 Toucan JSONL 数据。"""
        path = Path(self.config.toucan_data_path)
        if not path.exists():
            raise FileNotFoundError(f"Toucan 数据文件不存在: {path}")

        data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        return data

    def _convert_item(self, item: dict) -> Optional[EpisodeSeed]:
        """将单条 Toucan 样本转换为 EpisodeSeed。

        Returns:
            EpisodeSeed 或 None（如果不满足条件）。
        """
        self.stats["total_processed"] += 1

        # 解析 messages
        messages = _parse_json_field(item.get("messages"))
        if not isinstance(messages, list):
            self.stats["skip_messages_unparseable"] += 1
            return None

        # 解析 tools
        tools = _parse_json_field(item.get("tools"))
        if not isinstance(tools, list) or not tools:
            self.stats["skip_tools_unparseable"] += 1
            return None

        # 获取元数据
        uuid = item.get("uuid", "")
        subset_name = item.get("subset_name", "")
        target_tools = item.get("target_tools", "")

        # irrelevant 子集特殊处理
        if subset_name == "irrelevant":
            if self.config.include_irrelevant_as_no_tool:
                return self._build_no_tool_episode(uuid, messages, tools, item)
            else:
                self.stats["skip_irrelevant"] += 1
                return None

        # 非 irrelevant 子集必须有 target_tools
        if not target_tools:
            self.stats["skip_no_target_tools"] += 1
            return None

        # 提取 tool_call/response 对
        trace_result = self._extract_oracle_trace(messages)
        if trace_result is None:
            self.stats["skip_trace_extraction_failed"] += 1
            return None

        oracle_steps, initial_msgs = trace_result

        # 过滤条件
        n_steps = len(oracle_steps)
        if n_steps < self.config.min_decision_turns:
            self.stats["skip_too_few_turns"] += 1
            return None

        if n_steps > self.config.max_decision_turns:
            if self.config.truncate_long_episodes:
                oracle_steps = oracle_steps[:self.config.max_decision_turns]
                # 重新编号
                for i, step in enumerate(oracle_steps):
                    step.step = i
                self.stats["truncated_episodes"] += 1
            else:
                self.stats["skip_too_many_turns"] += 1
                return None

        # 确定 episode_type
        episode_type = self._classify_episode_type(oracle_steps, messages)

        # 提取 MCP server 信息
        mcp_servers = self._extract_mcp_servers(oracle_steps)

        # 构建 EpisodeSeed
        episode = EpisodeSeed(
            episode_id=f"toucan_{uuid}" if uuid else f"toucan_{self.stats['total_processed']}",
            source=DataSource.TOUCAN.value,
            episode_type=episode_type,
            scenario_tags=[],
            mcp_servers=mcp_servers,
            tools_snapshot=tools,
            initial_messages=initial_msgs,
            oracle_trace=oracle_steps,
            max_turns=min(n_steps + 1, self.config.max_decision_turns),
            split_keys=SplitKeys(
                server=mcp_servers[0] if mcp_servers else "",
                domain=self._infer_domain(mcp_servers),
                tool_names=[s.tool_name for s in oracle_steps
                            if s.action_type == ActionType.TOOL_CALL.value],
            ),
            metadata={
                "subset_name": subset_name,
                "target_tools": target_tools,
                "original_uuid": uuid,
            },
        )

        # 验证
        errors = episode.validate()
        if errors:
            self.stats["skip_validation_failed"] += 1
            return None

        self.stats["success"] += 1
        self.stats[f"type_{episode_type}"] += 1
        return episode

    def _build_no_tool_episode(
        self,
        uuid: str,
        messages: list[dict],
        tools: list[dict],
        item: dict,
    ) -> Optional[EpisodeSeed]:
        """构建 no_tool 类型的 episode（来自 irrelevant 子集）。"""
        # 提取 user message
        user_msgs = [m for m in messages if m.get("role") == "user"]
        if not user_msgs:
            self.stats["skip_no_user_message"] += 1
            return None

        # 找最后一个 assistant 回复作为 expected content
        last_assistant = None
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                last_assistant = msg.get("content", "")
                break

        # initial_messages: 只保留第一个 user message
        initial_msgs = [{"role": "user", "content": user_msgs[0].get("content", "")}]

        # Oracle: final_answer（不调用工具）
        oracle_step = OracleStep(
            step=0,
            action_type=ActionType.FINAL_ANSWER.value,
            expected_content=last_assistant or "",
            verifier=VerifierConfig(type="exact"),
        )

        episode = EpisodeSeed(
            episode_id=f"toucan_{uuid}" if uuid else f"toucan_notool_{self.stats['total_processed']}",
            source=DataSource.TOUCAN.value,
            episode_type=EpisodeType.NO_TOOL.value,
            scenario_tags=[],
            mcp_servers=[],
            tools_snapshot=tools,
            initial_messages=initial_msgs,
            oracle_trace=[oracle_step],
            max_turns=1,
            split_keys=SplitKeys(),
            metadata={
                "subset_name": "irrelevant",
                "original_uuid": uuid,
            },
        )

        errors = episode.validate()
        if errors:
            self.stats["skip_no_tool_validation_failed"] += 1
            return None

        self.stats["success"] += 1
        self.stats["type_no_tool"] += 1
        return episode

    def _extract_oracle_trace(
        self,
        messages: list[dict],
    ) -> Optional[tuple[list[OracleStep], list[dict]]]:
        """从 Toucan messages 中提取 oracle trace 和 initial messages。

        Toucan 消息格式:
          - role: user / assistant / tool_call / tool_response

        转换规则:
          - user message → initial_messages
          - assistant (before first tool_call) → initial_messages
          - tool_call + tool_response → OracleStep
          - 连续多个 tool_call (无 response 间隔) → parallel_tool_call
          - 最后的 assistant → final_answer step

        Returns:
            (oracle_steps, initial_messages) 或 None
        """
        initial_msgs: list[dict] = []
        oracle_steps: list[OracleStep] = []
        step_idx = 0

        # 状态机
        i = 0
        found_first_tool_call = False

        while i < len(messages):
            msg = messages[i]
            role = msg.get("role", "")
            content = msg.get("content", "")

            if role == "user":
                if not found_first_tool_call:
                    initial_msgs.append({"role": "user", "content": content})
                # 多轮中后续的 user message 忽略（不影响 oracle trace）
                i += 1

            elif role == "assistant" and not found_first_tool_call:
                # 第一个 tool_call 之前的 assistant 消息放入 initial
                initial_msgs.append({"role": "assistant", "content": content})
                i += 1

            elif role == "assistant" and found_first_tool_call:
                # tool_call 之后的 assistant = final_answer
                # 只取最后一个 assistant 作为 final answer
                # 先跳过，最后统一处理
                i += 1

            elif role == "tool_call":
                found_first_tool_call = True

                # 先收集连续的 tool_call（不被 tool_response 打断的）
                # 真正的并行 = 连续多个 tool_call 之间没有 tool_response
                parallel_calls = []
                parallel_start = i

                # 向前看：收集连续的 tool_call（不被 response 打断）
                while i < len(messages) and messages[i].get("role") == "tool_call":
                    tc_content = messages[i].get("content", "")
                    parsed_tc = _parse_tool_call_content(tc_content)

                    if parsed_tc is None:
                        return None

                    args = parsed_tc.get("arguments_parsed")
                    if args is None and self.config.require_args_parseable:
                        return None

                    parallel_calls.append({
                        "tool_name": parsed_tc["name"],
                        "arguments": args if args is not None else {},
                    })
                    i += 1

                # 现在收集对应的 tool_response
                parallel_observations = []
                for _ in range(len(parallel_calls)):
                    if i < len(messages) and messages[i].get("role") == "tool_response":
                        obs = messages[i].get("content", "")
                        obs = _truncate_observation(obs, self.config.max_observation_length)
                        parallel_observations.append(obs)
                        i += 1
                    else:
                        if self.config.require_paired:
                            return None
                        parallel_observations.append("")

                # 构建 OracleStep
                if len(parallel_calls) == 1:
                    # 单个 tool_call
                    step = OracleStep(
                        step=step_idx,
                        action_type=ActionType.TOOL_CALL.value,
                        tool_name=parallel_calls[0]["tool_name"],
                        arguments=parallel_calls[0]["arguments"],
                        replay_observation=parallel_observations[0] if parallel_observations else "",
                        verifier=VerifierConfig(type="exact"),
                    )
                else:
                    # 并行 tool_call
                    step = OracleStep(
                        step=step_idx,
                        action_type=ActionType.PARALLEL_TOOL_CALL.value,
                        calls=parallel_calls,
                        match_mode="set",
                        replay_observations=parallel_observations,
                        verifier=VerifierConfig(type="exact"),
                    )

                oracle_steps.append(step)
                step_idx += 1

            elif role == "tool_response":
                # 孤立的 tool_response（前面没有 tool_call），跳过
                self.stats["orphan_tool_response"] += 1
                i += 1

            else:
                i += 1

        # 处理 final_answer：找最后一个 assistant 消息（在 tool_call 之后的）
        if found_first_tool_call:
            last_assistant_content = None
            for msg in reversed(messages):
                if msg.get("role") == "assistant":
                    last_assistant_content = msg.get("content", "")
                    break

            # 如果最后有 assistant 回复，且不是 initial 中的，添加 final_answer step
            if last_assistant_content and oracle_steps:
                # 确认这个 assistant 是在最后一个 tool_response 之后
                last_tool_idx = -1
                for idx, msg in enumerate(messages):
                    if msg.get("role") in ("tool_call", "tool_response"):
                        last_tool_idx = idx

                last_assistant_idx = -1
                for idx, msg in enumerate(messages):
                    if msg.get("role") == "assistant":
                        last_assistant_idx = idx

                if last_assistant_idx > last_tool_idx:
                    final_step = OracleStep(
                        step=step_idx,
                        action_type=ActionType.FINAL_ANSWER.value,
                        expected_content=last_assistant_content,
                        verifier=VerifierConfig(type="exact"),
                    )
                    oracle_steps.append(final_step)

        # 必须有至少一个 oracle step
        if not oracle_steps:
            return None

        # initial_messages 必须有 user message
        if not initial_msgs or not any(m.get("role") == "user" for m in initial_msgs):
            return None

        return oracle_steps, initial_msgs

    def _classify_episode_type(
        self,
        oracle_steps: list[OracleStep],
        messages: list[dict],
    ) -> str:
        """根据 oracle trace 分类 episode type。"""
        if not oracle_steps:
            return EpisodeType.NO_TOOL.value

        action_types = [s.action_type for s in oracle_steps]

        # 只有 tool_call（无 final_answer）
        tool_call_types = {ActionType.TOOL_CALL.value, ActionType.PARALLEL_TOOL_CALL.value}
        if all(t in tool_call_types for t in action_types):
            if len(oracle_steps) == 1:
                return EpisodeType.CALL_ONLY.value
            else:
                return EpisodeType.CALL_THEN_CALL.value

        # 有 tool_call + final_answer
        has_tool = any(t in tool_call_types for t in action_types)
        has_final = ActionType.FINAL_ANSWER.value in action_types

        if has_tool and has_final:
            if len(oracle_steps) == 2:
                return EpisodeType.CALL_THEN_FINAL.value
            else:
                return EpisodeType.CALL_THEN_CALL.value

        # 只有 final_answer
        if has_final and not has_tool:
            return EpisodeType.NO_TOOL.value

        # 有 report_error
        if ActionType.REPORT_ERROR.value in action_types:
            return EpisodeType.ERROR_OUTPUT.value

        return EpisodeType.CALL_ONLY.value

    def _extract_mcp_servers(self, oracle_steps: list[OracleStep]) -> list[str]:
        """从 oracle steps 中提取 MCP server 列表。"""
        servers = set()
        for step in oracle_steps:
            if step.action_type == ActionType.TOOL_CALL.value and step.tool_name:
                server = _extract_mcp_server(step.tool_name)
                servers.add(server)
            elif step.action_type == ActionType.PARALLEL_TOOL_CALL.value:
                for call in step.calls:
                    name = call.get("tool_name", "")
                    if name:
                        servers.add(_extract_mcp_server(name))
        return sorted(servers)

    def _infer_domain(self, mcp_servers: list[str]) -> str:
        """从 MCP server 名推断领域。"""
        domain_keywords = {
            "weather": "weather",
            "finance": "finance",
            "okx": "finance",
            "calculator": "math",
            "math": "math",
            "dictionary": "language",
            "translate": "language",
            "code": "programming",
            "leetcode": "programming",
            "image": "media",
            "flux": "media",
            "drawing": "media",
            "music": "media",
            "book": "knowledge",
            "wiki": "knowledge",
            "search": "search",
            "time": "utility",
            "file": "filesystem",
        }

        for server in mcp_servers:
            server_lower = server.lower()
            for keyword, domain in domain_keywords.items():
                if keyword in server_lower:
                    return domain

        return "general"

    def _log_stats(self) -> None:
        """输出构建统计。"""
        logger.info("=" * 60)
        logger.info("EpisodeSeed Builder 统计:")
        for key, value in sorted(self.stats.items()):
            logger.info(f"  {key}: {value}")
        logger.info("=" * 60)

    def _build_stats_report(self, episodes: list[EpisodeSeed]) -> dict:
        """构建统计报告。"""
        type_dist = Counter(ep.episode_type for ep in episodes)
        source_dist = Counter(ep.source for ep in episodes)
        server_dist = Counter()
        domain_dist = Counter()
        turns_dist = Counter()

        for ep in episodes:
            for s in ep.mcp_servers:
                server_dist[s] += 1
            domain_dist[ep.split_keys.domain] += 1
            turns_dist[ep.decision_turns] += 1

        return {
            "total_episodes": len(episodes),
            "builder_stats": dict(self.stats),
            "episode_type_distribution": dict(type_dist),
            "source_distribution": dict(source_dist),
            "domain_distribution": dict(domain_dist),
            "decision_turns_distribution": {str(k): v for k, v in sorted(turns_dist.items())},
            "top_20_servers": server_dist.most_common(20),
            "has_parallel_calls": sum(1 for ep in episodes if ep.has_parallel_calls),
            "validation_report": validate_episodes(episodes),
        }


# ============================================================
# CLI 入口
# ============================================================


def main():
    """命令行入口。"""
    import argparse

    parser = argparse.ArgumentParser(description="从 Toucan 数据构建 EpisodeSeed")
    parser.add_argument(
        "--input", "-i",
        default="data/toucan/toucan_sft_subset_5000.jsonl",
        help="Toucan JSONL 数据路径",
    )
    parser.add_argument(
        "--output", "-o",
        default="data/toucan/episode_seeds.jsonl",
        help="输出 EpisodeSeed JSONL 路径",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=5,
        help="最大决策轮数",
    )
    parser.add_argument(
        "--no-irrelevant",
        action="store_true",
        help="不包含 irrelevant 子集",
    )
    args = parser.parse_args()

    config = BuilderConfig(
        toucan_data_path=args.input,
        output_path=args.output,
        max_decision_turns=args.max_turns,
        include_irrelevant_as_no_tool=not args.no_irrelevant,
    )

    builder = EpisodeSeedBuilder(config)
    episodes = builder.build()
    builder.save(episodes)

    # 打印摘要
    print(f"\n{'='*60}")
    print(f"  EpisodeSeed 构建完成")
    print(f"  输出: {args.output}")
    print(f"  总数: {len(episodes)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
