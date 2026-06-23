"""Conditioned Decision 构建器。

从 ToolACE 转换后的真实数据中加载 conditioned decision samples。

设计要点（mcp_tools_rl_project_plan.md §6）：
  - 多步轨迹拆成 conditioned decision points
  - 使用真实 tool_output（非合成），保证 oracle 可靠
  - 无法确定唯一 oracle 的样本丢弃
  - 标记 provenance: real

Oracle Propagation（mcp_tools_rl_project_plan.md §6.2）：
  P0 实现为 exact oracle validator（给定 canonical history 后精确匹配 next-action）。

重要变更（2026-06-18）：
  - 删除模板合成逻辑（合成的 tool_output 与 final_answer 之间无真实信息依赖，
    会产生噪声 reward 信号）
  - 改为直接从配置指定的 conditioned_steps.jsonl 加载真实数据
  - 真实数据中 tool_output 和 final_answer 有实际的实体引用关系
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from loguru import logger


@dataclass
class ConditionedSample:
    """一条 conditioned decision 样本。"""
    task_id: str
    tools: list[dict]
    messages: list[dict]  # 包含 tool_call + tool_output history
    ground_truth_action: dict  # next_action oracle
    action_type: str  # "tool_call" / "final_answer"
    step_index: int
    total_steps: int
    provenance: str  # "real"
    scenario_type: str  # "conditioned_tool_call" / "final_answer"


@dataclass
class NoToolSample:
    """一条不调用工具的样本。"""
    task_id: str
    tools: list[dict]
    messages: list[dict]
    ground_truth_action: dict  # {"type": "ask_clarification"/"final_answer", "content": ...}
    action_type: str  # "ask_clarification" / "final_answer"
    provenance: str  # "real"
    scenario_type: str  # "no_tool"
    no_tool_subtype: str = ""  # "ask_clarification" / "no_tool_needed"


@dataclass
class ConditionedBuilderConfig:
    """Conditioned decision 构建配置。"""
    # 数据路径
    conditioned_steps_path: str = "data/toolace/processed/conditioned_steps.jsonl"
    no_tool_steps_path: str = "data/toolace/processed/no_tool_steps.jsonl"
    # 是否加载 no_tool 样本
    include_no_tool: bool = True
    # 最大样本数限制
    max_conditioned_samples: Optional[int] = None
    max_no_tool_samples: Optional[int] = None
    seed: int = 42


class ConditionedDecisionBuilder:
    """Conditioned Decision 构建器。

    从配置指定的真实数据中加载 conditioned decision samples。

    两类数据：
    1. conditioned_steps.jsonl — 真实的 tool_output → next_action 链路
    2. no_tool_steps.jsonl — assistant 选择不调用工具的样本

    Usage:
        builder = ConditionedDecisionBuilder()
        conditioned, no_tool = builder.build()
    """

    def __init__(self, config: Optional[ConditionedBuilderConfig] = None):
        self.config = config or ConditionedBuilderConfig()
        self.stats = {
            "conditioned_tool_call": 0,
            "conditioned_final_answer": 0,
            "no_tool_ask_clarification": 0,
            "no_tool_no_needed": 0,
            "skipped_invalid": 0,
        }

    def build(self) -> tuple[list[ConditionedSample], list[NoToolSample]]:
        """从真实数据构建 conditioned decision 和 no-tool 样本。

        Returns:
            (conditioned_samples, no_tool_samples)
        """
        conditioned = self._load_conditioned()
        no_tool = self._load_no_tool() if self.config.include_no_tool else []

        logger.info(
            f"Conditioned decision 构建完成: "
            f"conditioned_tool_call={self.stats['conditioned_tool_call']}, "
            f"final_answer={self.stats['conditioned_final_answer']}, "
            f"no_tool_ask_clarification={self.stats['no_tool_ask_clarification']}, "
            f"no_tool_no_needed={self.stats['no_tool_no_needed']}, "
            f"skipped={self.stats['skipped_invalid']}"
        )

        return conditioned, no_tool

    def build_from_decision_steps(
        self,
        data_path: str | Path,
        max_samples: Optional[int] = None,
    ) -> list[ConditionedSample]:
        """兼容旧接口：从 decision_steps.jsonl 构建 conditioned samples。

        新实现：直接从 conditioned_steps.jsonl 加载真实数据。
        如果 conditioned_steps.jsonl 不存在，则从 decision_steps.jsonl 中
        提取有 tool_output history 的多步样本。

        Args:
            data_path: decision_steps.jsonl 路径（用于推导 conditioned_steps.jsonl 位置）。
            max_samples: 最大输出样本数。

        Returns:
            ConditionedSample 列表。
        """
        data_path = Path(data_path)
        conditioned_path = data_path.parent / "conditioned_steps.jsonl"

        if conditioned_path.exists():
            # 优先从真实 conditioned_steps.jsonl 加载
            self.config.conditioned_steps_path = str(conditioned_path)
            conditioned = self._load_conditioned()
        else:
            # 回退：从 decision_steps.jsonl 中提取有 tool history 的多步样本
            logger.warning(
                f"conditioned_steps.jsonl 不存在 ({conditioned_path})，"
                f"从 decision_steps.jsonl 提取多步样本"
            )
            conditioned = self._extract_from_decision_steps(data_path)

        if max_samples and len(conditioned) > max_samples:
            import random
            rng = random.Random(self.config.seed)
            conditioned = rng.sample(conditioned, max_samples)

        return conditioned

    def _load_conditioned(self) -> list[ConditionedSample]:
        """从 conditioned_steps.jsonl 加载真实 conditioned decision 样本。"""
        path = Path(self.config.conditioned_steps_path)
        if not path.exists():
            logger.warning(f"conditioned_steps.jsonl 不存在: {path}")
            return []

        logger.info(f"加载真实 conditioned steps: {path}")
        samples = []

        with open(path) as f:
            for line in f:
                d = json.loads(line)

                # 验证必要字段
                if not d.get("ground_truth_action") or not d.get("tools") or not d.get("messages"):
                    self.stats["skipped_invalid"] += 1
                    continue

                action_type = d.get("action_type", "")
                # 归一化：数据中的 "conditioned_tool_call" → "tool_call"
                normalized_action_type = "tool_call" if action_type == "conditioned_tool_call" else action_type
                scenario_type = d.get("scenario_type", "")

                # 验证 messages 中确实有 tool_output（conditioned 的前提）
                has_tool_output = any(
                    m.get("role") == "tool" for m in d.get("messages", [])
                )
                if not has_tool_output:
                    self.stats["skipped_invalid"] += 1
                    continue

                sample = ConditionedSample(
                    task_id=d["task_id"],
                    tools=d["tools"],
                    messages=d["messages"],
                    ground_truth_action=d["ground_truth_action"],
                    action_type=normalized_action_type,
                    step_index=d.get("step_index", 0),
                    total_steps=d.get("total_steps", 2),
                    provenance="real",
                    scenario_type=scenario_type or (
                        "conditioned_tool_call" if action_type == "conditioned_tool_call"
                        else "final_answer"
                    ),
                )
                samples.append(sample)

                if action_type == "conditioned_tool_call":
                    self.stats["conditioned_tool_call"] += 1
                else:
                    self.stats["conditioned_final_answer"] += 1

        if self.config.max_conditioned_samples and len(samples) > self.config.max_conditioned_samples:
            import random
            rng = random.Random(self.config.seed)
            samples = rng.sample(samples, self.config.max_conditioned_samples)

        logger.info(f"  加载 conditioned: {len(samples)} 条")
        return samples

    def _load_no_tool(self) -> list[NoToolSample]:
        """从 no_tool_steps.jsonl 加载不调用工具的样本。"""
        path = Path(self.config.no_tool_steps_path)
        if not path.exists():
            logger.warning(f"no_tool_steps.jsonl 不存在: {path}")
            return []

        logger.info(f"加载真实 no-tool steps: {path}")
        samples = []

        with open(path) as f:
            for line in f:
                d = json.loads(line)

                if not d.get("ground_truth_action") or not d.get("tools") or not d.get("messages"):
                    self.stats["skipped_invalid"] += 1
                    continue

                raw_action_type = d.get("action_type", "no_tool_needed")
                no_tool_subtype = d.get("no_tool_subtype", raw_action_type)
                action_type = (
                    "ask_clarification"
                    if no_tool_subtype == "ask_clarification"
                    else "final_answer"
                )
                ground_truth_action = dict(d["ground_truth_action"])
                ground_truth_action["type"] = action_type
                ground_truth_action["no_tool_subtype"] = no_tool_subtype

                sample = NoToolSample(
                    task_id=d["task_id"],
                    tools=d["tools"],
                    messages=d["messages"],
                    ground_truth_action=ground_truth_action,
                    action_type=action_type,
                    provenance="real",
                    scenario_type="no_tool",
                    no_tool_subtype=no_tool_subtype,
                )
                samples.append(sample)

                if no_tool_subtype == "ask_clarification":
                    self.stats["no_tool_ask_clarification"] += 1
                else:
                    self.stats["no_tool_no_needed"] += 1

        if self.config.max_no_tool_samples and len(samples) > self.config.max_no_tool_samples:
            import random
            rng = random.Random(self.config.seed)
            samples = rng.sample(samples, self.config.max_no_tool_samples)

        logger.info(f"  加载 no-tool: {len(samples)} 条")
        return samples

    def _extract_from_decision_steps(self, data_path: Path) -> list[ConditionedSample]:
        """从 decision_steps.jsonl 中提取有 tool_output history 的多步样本。

        这是 conditioned_steps.jsonl 不存在时的回退逻辑。
        """
        samples = []

        # 按 sample_id 分组
        sample_groups: dict[str, list[dict]] = {}
        with open(data_path) as f:
            for line in f:
                d = json.loads(line)
                task_id = d["task_id"]
                sample_id = "_".join(task_id.split("_")[:2])
                if sample_id not in sample_groups:
                    sample_groups[sample_id] = []
                sample_groups[sample_id].append(d)

        for sample_id, steps in sample_groups.items():
            if len(steps) <= 1:
                continue

            steps.sort(key=lambda x: x.get("step_index", 0))

            for i in range(1, len(steps)):
                current_step = steps[i]
                messages = current_step.get("messages", [])

                # 只取有 tool_output 的
                has_tool_output = any(
                    m.get("role") == "tool" for m in messages
                )
                if not has_tool_output:
                    continue

                gt_calls = current_step.get("ground_truth_calls", [])
                if not gt_calls:
                    self.stats["skipped_invalid"] += 1
                    continue

                sample = ConditionedSample(
                    task_id=f"{current_step['task_id']}___conditioned",
                    tools=current_step["tools"],
                    messages=messages,
                    ground_truth_action={
                        "type": "tool_call",
                        "tool_calls": gt_calls,
                    },
                    action_type="tool_call",
                    step_index=i,
                    total_steps=len(steps),
                    provenance="real",
                    scenario_type="conditioned_tool_call",
                )
                samples.append(sample)
                self.stats["conditioned_tool_call"] += 1

        logger.info(f"  从 decision_steps 提取 conditioned: {len(samples)} 条")
        return samples

    def get_stats(self) -> dict:
        """返回构建统计。"""
        return dict(self.stats)
