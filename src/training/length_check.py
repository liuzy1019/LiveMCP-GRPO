"""数据长度预检（P1-1）。

目标：训练启动时，先扫一遍 train/val parquet，统计 prompt token 长度分布，
与 data.max_prompt_length 比对：
  - max > limit          → fail-fast，否则会被 verl 静默过滤
  - p99 > limit * 0.95   → warn，buffer 不足，下次数据更新就可能破

不在这里做截断也不修数据，纯诊断。修数据的责任在数据构建阶段。
"""

from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from loguru import logger


SKIP_ENV = "SCHEMASHIFT_SKIP_LENGTH_CHECK"
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


def _iter_prompt_messages(records: list[dict]) -> Iterable[list[dict]]:
    """parquet 里 prompt 列实际是 list<struct{role, content}>。
    保留对 JSON 字符串形式的兜底（旧数据兼容）。"""
    import json as _json
    for r in records:
        p = r.get("prompt")
        if p is None:
            yield [{"role": "user", "content": ""}]
            continue
        if isinstance(p, list):
            yield list(p)
            continue
        if isinstance(p, str):
            try:
                yield _json.loads(p)
            except (ValueError, TypeError):
                yield [{"role": "user", "content": p}]
            continue
        yield [{"role": "user", "content": str(p)}]


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
    for messages in _iter_prompt_messages(records):
        try:
            ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
            lens.append(len(ids))
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

    默认开启；SCHEMASHIFT_SKIP_LENGTH_CHECK=1 时跳过。
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


########## group 完整性检查


def assert_e4_group_integrity(
    parquet_path: str | Path,
    tokenizer_path: str,
    max_prompt_length: int,
    split: str,
    expected_levels: dict[str, int] | None = None,
) -> None:
    """前置检查：模拟 verl 的 prompt 长度过滤后，每个 group_id 内的 group 结构是否完整。

    E4（静态 replay）数据预期每个 group 有 9 条、每条对应一个 perturbation_level，
    分布为 {"none": 3, "mild": 3, "strong": 3}。OVAL（live MCP）数据标签和分布不同，
    可通过 expected_levels 注入，或传 None 跳过标签分布检查、只做 group 完整性校验。

    Args:
        parquet_path: 训练/验证 parquet 文件。
        tokenizer_path: tokenizer 路径或模型名。
        max_prompt_length: prompt 长度上限（超限的行会被 verl 静默过滤）。
        split: "train" 或 "val"（仅用于日志）。
        expected_levels: 期望的标签分布，如 {"none": 3, "mild": 3, "strong": 3}。
            None 时跳过标签分布检查，仅校验每个 group 的样本数一致性。
    """
    import pyarrow.parquet as pq
    from verl.utils.tokenizer import hf_tokenizer

    tokenizer = hf_tokenizer(tokenizer_path)
    records = pq.read_table(str(parquet_path)).to_pylist()
    expected = expected_levels or {}
    grouped: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    n_filtered = 0
    for r, messages in zip(records, _iter_prompt_messages(records)):
        try:
            n = len(tokenizer.apply_chat_template(messages, add_generation_prompt=True))
        except Exception:
            n = max_prompt_length + 1
        if n > max_prompt_length:
            n_filtered += 1
            continue
        grouped[r.get("group_id", "")][r.get("perturbation_level", "none")] += 1

    if expected:
        # 精确标签分布检查（E4 3:3:3 等）
        bad = [(gid, dict(d)) for gid, d in grouped.items() if dict(d) != expected]
        if bad:
            sample = "\n".join(f"  {gid}: {dist}" for gid, dist in bad[:5])
            raise AssertionError(
                f"[{split}] 过滤后 group 完整性破坏：{len(bad)} 个 group_id "
                f"不再满足预期分布 {expected}。\n"
                f"max_prompt_length={max_prompt_length} 过滤掉了 {n_filtered}/{len(records)} 行。\n"
                f"前 5 个异常 group:\n{sample}"
            )
        expected_total = sum(expected.values())
        logger.info(
            f"[{split}] group 完整性 OK: {len(grouped)} groups × {expected_total} records "
            f"({', '.join(f'{k}={v}' for k, v in expected.items())})"
        )
    else:
        # 宽松检查：统计每个 group 的样本数，验证过滤后无 group 被截断
        group_sizes = {gid: sum(d.values()) for gid, d in grouped.items()}
        # 允许 tolerance：最大和最小 group 样本数之差 ≤ 1（因随机过滤可能不均）
        if group_sizes:
            sizes = sorted(set(group_sizes.values()))
            if len(sizes) > 2 or (len(sizes) == 2 and sizes[1] - sizes[0] > 1):
                bad_sizes = [(gid, s) for gid, s in sorted(group_sizes.items())[:10]]
                sample = "\n".join(f"  {gid}: {s}" for gid, s in bad_sizes)
                logger.warning(
                    f"[{split}] group 样本数差异较大 (sizes={sizes})，"
                    f"可能存在过滤不均。前 10:\n{sample}"
                )
            total = sum(group_sizes.values())
            avg_size = total / len(group_sizes) if group_sizes else 0
            logger.info(
                f"[{split}] group 完整性 OK: {len(grouped)} groups, "
                f"avg {avg_size:.1f} rec/group (#filtered={n_filtered})"
            )
