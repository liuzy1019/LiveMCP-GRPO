"""
BFCL 多轮任务环境。

封装 BFCL V3 的多轮任务加载、Schema 扰动、工具调用执行。
"""

import json
import copy
import random
from pathlib import Path
from typing import Any, Optional
from dataclasses import dataclass, field
from loguru import logger

from .schema_perturber import (
    SchemaPerturber,
    PerturbationLevel,
    TRAINING_LEVELS,
)


# ──────────────────────────────────────────────
# 数据类型
# ──────────────────────────────────────────────


@dataclass
class BFCLTask:
    """BFCL 单条多轮任务。"""

    id: str
    """任务唯一 ID。"""

    user_turns: list[list[dict]]
    """用户轮次列表，每轮是一个对话历史列表。"""

    functions: list[dict]
    """Tool definition 列表。"""

    initial_config: dict
    """环境初始状态。"""

    path: list[str]
    """预期工具调用顺序（gold path）。"""

    involved_classes: list[str]
    """涉及的 API 域类名。"""

    metadata: dict = field(default_factory=dict)
    """其他元数据。"""


@dataclass
class ToolCallResult:
    """工具调用结果。"""

    success: bool
    """是否成功。"""

    result: str
    """执行结果字符串。"""

    error: Optional[str] = None
    """错误信息。"""


# ──────────────────────────────────────────────
# 数据加载器
# ──────────────────────────────────────────────


CLASS_TO_DOC: dict[str, str] = {
    "GorillaFileSystem": "gorilla_file_system",
    "MathAPI": "math_api",
    "MessageAPI": "message_api",
    "TicketAPI": "ticket_api",
    "TradingBot": "trading_bot",
    "TravelAPI": "travel_booking",
    "TwitterAPI": "posting_api",
    "VehicleControlAPI": "vehicle_control",
}
"""BFCL V3 involved_classes 到 multi_turn_func_doc 文件名映射。"""


class BFCLDataLoader:
    """BFCL 数据加载器。

    从 JSONL 文件加载多轮任务数据，并根据 involved_classes
    从 multi_turn_func_doc/ 加载实际的 tool definition。
    """

    def __init__(self, data_path: str, func_doc_dir: str = ""):
        """
        Args:
            data_path: JSONL 文件路径。
            func_doc_dir: multi_turn_func_doc 目录路径。
                留空时自动在 data/ 下查找。
        """
        self.data_path = Path(data_path)
        if not self.data_path.exists():
            raise FileNotFoundError(f"数据文件不存在: {data_path}")

        # 确定 func_doc 目录
        self.func_doc_dir = Path(func_doc_dir) if func_doc_dir else (
            self.data_path.parent / "multi_turn_func_doc"
        )
        if not self.func_doc_dir.exists():
            logger.warning(
                f"func_doc 目录不存在: {self.func_doc_dir}，"
                f"functions 将被加载为空"
            )

        # 加载所有 func_doc 文件
        self.doc_cache: dict[str, list[dict]] = {}
        if self.func_doc_dir.exists():
            self._load_doc_cache()

        self.tasks: list[BFCLTask] = []
        self._load()

        logger.info(
            f"BFCL 数据加载完成: {len(self.tasks)} 条任务, "
            f"{len(self.doc_cache)} 个 API 域"
        )

    def _load_doc_cache(self) -> None:
        """加载所有 API 域的 tool definition。"""
        for doc_file in sorted(self.func_doc_dir.glob("*.json")):
            try:
                funcs = []
                with open(doc_file) as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            funcs.append(json.loads(line))
                self.doc_cache[doc_file.stem] = funcs
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"加载 func_doc 失败 {doc_file.name}: {e}")

    def _load(self) -> None:
        """从 JSONL 文件加载数据。"""
        with open(self.data_path, "r") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                    task = self._parse_task(raw)
                    self.tasks.append(task)
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"第 {line_num} 行解析失败: {e}")

    def _parse_task(self, raw: dict) -> BFCLTask:
        """将原始 JSON 解析为 BFCLTask。

        函数的来源：优先从 raw 中直接读取（BFCL V4 格式），
        否则从 multi_turn_func_doc/ 中根据 involved_classes 加载。
        """
        task_id = raw.get("id", str(raw.get("group_id", "")))
        functions = raw.get("function") or raw.get("metadata", {}).get("functions", [])
        question = raw.get("question") or raw.get("input", [])
        initial_config = raw.get("initial_config", {})
        path = raw.get("path", [])
        involved_classes = raw.get("involved_classes", [])

        # 如果 functions 为空，从 func_doc 加载
        if not functions and self.doc_cache:
            functions = []
            seen_names: set[str] = set()
            for cls_name in involved_classes:
                doc_name = CLASS_TO_DOC.get(cls_name)
                if doc_name and doc_name in self.doc_cache:
                    for func_def in self.doc_cache[doc_name]:
                        if func_def.get("name") not in seen_names:
                            functions.append(func_def)
                            seen_names.add(func_def.get("name", ""))
            if functions:
                logger.debug(
                    f"从 func_doc 加载了 {len(functions)} 个工具定义"
                    f"（{task_id}）"
                )

        # 标准化 user_turns
        if isinstance(question, list):
            user_turns = question
        else:
            user_turns = [[{"role": "user", "content": str(question)}]]

        # 从 input 字段获取（EvalScope 格式）
        metadata = raw.get("metadata", {})
        if not functions and "turns" in metadata:
            user_turns = metadata["turns"]

        return BFCLTask(
            id=task_id,
            user_turns=user_turns,
            functions=functions,
            initial_config=initial_config,
            path=path,
            involved_classes=involved_classes,
            metadata=metadata,
        )

    def get_task(self, index: int) -> BFCLTask:
        """按索引获取任务。

        Args:
            index: 任务索引。

        Returns:
            BFCLTask 实例。
        """
        return self.tasks[index]

    def sample_tasks(
        self, n: int, seed: Optional[int] = None
    ) -> list[BFCLTask]:
        """随机采样 n 个任务。

        Args:
            n: 采样数量。
            seed: 随机种子。

        Returns:
            采样到的任务列表。
        """
        rng = random.Random(seed)
        return rng.sample(self.tasks, min(n, len(self.tasks)))

    def __len__(self) -> int:
        return len(self.tasks)


# ──────────────────────────────────────────────
# Ground Truth 加载器
# ──────────────────────────────────────────────


class BFCLGroundTruthLoader:
    """BFCL Ground Truth 加载器。"""

    def __init__(self, gt_path: str):
        """
        Args:
            gt_path: Ground truth JSON 或 JSONL 文件路径。
        """
        self.gt_path = Path(gt_path)
        self._data: dict[str, list[str]] = {}

        if not self.gt_path.exists():
            raise FileNotFoundError(f"Ground truth 文件不存在: {gt_path}")

        self._load()

    def _load(self) -> None:
        """加载 ground truth 数据。"""
        with open(self.gt_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                    task_id = raw.get("id", "")
                    ground_truth = raw.get("ground_truth", [])
                    self._data[task_id] = ground_truth
                except json.JSONDecodeError:
                    continue

        logger.info(
            f"Ground truth 加载完成: {len(self._data)} 条"
        )

    def get_ground_truth(self, task_id: str) -> list[str]:
        """获取指定任务的 ground truth。

        Args:
            task_id: 任务 ID。

        Returns:
            Ground truth 函数调用字符串列表。
        """
        return self._data.get(task_id, [])


# ──────────────────────────────────────────────
# 任务生成器（训练用）
# ──────────────────────────────────────────────


