"""数据长度预检（P1-1）。

目标：训练启动时，先扫一遍 train/val parquet，统计 prompt token 长度分布，
与 data.max_prompt_length 比对：
  - max > limit          → fail-fast，否则会被 verl 静默过滤
  - p99 > limit * 0.95   → warn，buffer 不足，下次数据更新就可能破

不在这里做截断也不修数据，纯诊断。修数据的责任在数据构建阶段。
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger


SKIP_ENV = "LIVEMCP_SKIP_LENGTH_CHECK"
SOFT_RATIO = 0.95


@dataclass
class LengthStats:
    n_rows: int
    max_len: int
    p99: int
    p95: int
    p50: int
    n_overflow: int            # > limit
    n_near_limit: int          # > limit * SOFT_RATIO

    def format(self, split: str, limit: int) -> str:
        return (
            f"[{split}] rows={self.n_rows} "
            f"max={self.max_len} p99={self.p99} p95={self.p95} p50={self.p50} "
            f"limit={limit} overflow={self.n_overflow} near_limit={self.n_near_limit}"
        )


def _percentile(sorted_xs: list[int], pct: float) -> int:
    if not sorted_xs:
        return 0
    idx = max(0, min(len(sorted_xs) - 1, int(len(sorted_xs) * pct) - 1))
    return sorted_xs[idx]


from src.utils import iter_prompt_messages


def _encoded_token_length(encoded: Any) -> int:
    """返回单条 chat template 编码结果的 token 数。

    Transformers 5 的 ``apply_chat_template`` 可能返回 BatchEncoding；对它
    直接调用 ``len`` 得到的是字段数（通常为 2），而不是 input_ids 长度。
    这里只接受单条序列，意外的多 batch 返回值交给调用方按预检失败处理。
    """
    if isinstance(encoded, Mapping):
        if "input_ids" not in encoded:
            raise TypeError("chat template encoding is missing input_ids")
        encoded = encoded["input_ids"]

    shape = getattr(encoded, "shape", None)
    if shape is not None:
        dimensions = tuple(int(value) for value in shape)
        if len(dimensions) == 1:
            return dimensions[0]
        if len(dimensions) == 2 and dimensions[0] == 1:
            return dimensions[1]
        raise ValueError(
            "chat template must encode exactly one conversation; "
            f"got shape={dimensions}"
        )

    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if not isinstance(encoded, (list, tuple)):
        raise TypeError(
            "unsupported chat template encoding type: "
            f"{type(encoded).__name__}"
        )
    if encoded and isinstance(encoded[0], (list, tuple)):
        if len(encoded) != 1:
            raise ValueError(
                "chat template must encode exactly one conversation; "
                f"got batch_size={len(encoded)}"
            )
        encoded = encoded[0]
    return len(encoded)


def check_split_length(
    parquet_path: str | Path,
    tokenizer_path: str,
    max_prompt_length: int,
    split: str,
) -> LengthStats:
    """对单个 split 做 fail-fast 长度校验。"""
    from verl.utils.tokenizer import hf_tokenizer

    tokenizer = hf_tokenizer(tokenizer_path)

    # 走一遍计算
    import pyarrow.parquet as pq
    table = pq.read_table(str(parquet_path))
    records = table.to_pylist()

    lens: list[int] = []
    n_template_fail = 0
    for messages in iter_prompt_messages(records):
        try:
            encoded = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
            )
            lens.append(_encoded_token_length(encoded))
        except Exception:
            n_template_fail += 1
            lens.append(max_prompt_length + 1)  # 标记为溢出

    lens.sort()
    n_overflow = sum(1 for x in lens if x > max_prompt_length)
    soft_threshold = int(max_prompt_length * SOFT_RATIO)
    n_near = sum(1 for x in lens if x > soft_threshold)

    stats = LengthStats(
        n_rows=len(lens),
        max_len=lens[-1] if lens else 0,
        p99=_percentile(lens, 0.99),
        p95=_percentile(lens, 0.95),
        p50=_percentile(lens, 0.50),
        n_overflow=n_overflow,
        n_near_limit=n_near,
    )
    logger.info(stats.format(split, max_prompt_length))
    if n_template_fail > 0:
        logger.warning(
            f"[{split}] {n_template_fail} 行 chat_template 渲染失败，已按溢出处理"
        )

    if n_overflow > 0:
        suggested = max(max_prompt_length, stats.max_len + 512)
        # 向上取整到 1024 倍数，便于配置
        suggested = ((suggested + 1023) // 1024) * 1024
        raise RuntimeError(
            f"[{split}] {n_overflow}/{stats.n_rows} 行 prompt 长度超过 "
            f"max_prompt_length={max_prompt_length}（实测 max={stats.max_len}）。"
            f"verl 会静默过滤这些行，导致 batch 缩水或 group 不完整。\n"
            f"  → 建议把 data.max_prompt_length 调到 ≥ {suggested}，"
            f"或离线裁剪超长样本后重新生成 parquet。\n"
            f"  → 临时跳过：export {SKIP_ENV}=1（仅用于线下 debug，不要进 CI）。"
        )

    if stats.p99 > soft_threshold:
        logger.warning(
            f"[{split}] p99={stats.p99} 已经接近 max_prompt_length={max_prompt_length} "
            f"（阈值 {soft_threshold}）。下次数据更新或 schema 扩张可能直接溢出，"
            f"建议预留 buffer（设为 ≥ {((stats.p99 + 1024 + 1023) // 1024) * 1024}）。"
        )

    return stats


def parse_data_args_from_argv(argv: list[str]) -> dict:
    """从 hydra-style argv 解析 data.train_files / data.val_files / model.path /
    data.max_prompt_length。返回 dict，缺失字段不放 key。"""
    out: dict = {}
    for arg in argv:
        if arg.startswith("data.train_files="):
            out["train_files"] = arg.split("=", 1)[1].strip("[]'\"")
        elif arg.startswith("data.val_files="):
            out["val_files"] = arg.split("=", 1)[1].strip("[]'\"")
        elif arg.startswith("actor_rollout_ref.model.path="):
            out["model_path"] = arg.split("=", 1)[1]
        elif arg.startswith("data.max_prompt_length="):
            out["max_prompt_length"] = int(arg.split("=", 1)[1])
    return out


def maybe_run_length_check(argv: list[str]) -> None:
    """训练入口在调 verl 前调用一次。

    默认开启；LIVEMCP_SKIP_LENGTH_CHECK=1 时跳过。
    校验失败抛 RuntimeError，阻止 verl 启动。"""
    if os.environ.get(SKIP_ENV, "0") == "1":
        logger.warning(f"{SKIP_ENV}=1，跳过数据长度预检")
        return
    args = parse_data_args_from_argv(argv)
    train = args.get("train_files")
    val = args.get("val_files")
    model_path = args.get("model_path")
    limit = args.get("max_prompt_length")
    if not (train and model_path and limit):
        logger.warning(
            "数据长度预检：argv 中缺少 train_files / model.path / max_prompt_length，跳过"
        )
        return
    check_split_length(train, model_path, limit, "train")
    if val:
        check_split_length(val, model_path, limit, "val")
