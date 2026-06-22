"""BFCL Reward 函数。

提供 verl 兼容的 reward function，用于 BFCL 多轮工具调用评估。
在 agent loop 中计算 reward_score，此模块提供备用/验证 reward。
使用 AST 宽松匹配（类型容忍），与 eval 模块保持一致。
"""
import json
from typing import Any

from loguru import logger

from src.agent_loop.bfcl_agent_loop import _parse_bfcl_native_args
from src.eval.matching import _args_match


def compute_bfcl_reward(
    func_calls: list[dict],
    ground_truth_json: str,
    **kwargs,
) -> float:
    """计算 BFCL 任务 reward（与 agent loop 内逻辑一致）。

    按轮次顺序匹配 agent 的 tool calls 与 ground truth。
    如果提供了 turn_func_calls，按轮次匹配；否则扁平匹配。
    返回值: 1.0（完全匹配）或 0.0（不匹配）。

    Args:
        func_calls: agent 执行的 tool call 列表，每项为 {"name": str, "arguments": dict}。
        ground_truth_json: BFCL ground truth JSON 字符串。
        **kwargs: turn_func_calls（按轮次分组的调用列表）、num_turns 等。

    Returns:
        float: reward 值。
    """
    try:
        gt_data = json.loads(ground_truth_json) if isinstance(ground_truth_json, str) else ground_truth_json
    except (json.JSONDecodeError, TypeError):
        logger.warning(f"Ground truth JSON 解析失败: {str(ground_truth_json)[:100]}")
        return 0.0

    if not gt_data:
        return 0.0

    # 优先使用按轮次分组的数据
    turn_func_calls = kwargs.get("turn_func_calls")
    if turn_func_calls:
        return _compute_per_turn_reward(turn_func_calls, gt_data)

    # fallback: 扁平匹配
    if not func_calls:
        return 0.0
    return _compute_flat_reward(func_calls, gt_data)


def _compute_per_turn_reward(
    turn_func_calls: list[list[dict]],
    gt_data: list,
) -> float:
    """按轮次匹配（与 agent loop 逻辑一致）。

    长度规则：agent loop 在没有后续 user turn 时会多记一笔空 turn 再退出，
    所以允许 turn_func_calls 末尾出现若干空 turn；但 GT 结束后的任何非空
    工具调用 turn 都应判错。
    """
    if len(turn_func_calls) < len(gt_data):
        return 0.0
    # GT 结束后的额外轮次必须全为空
    for extra in turn_func_calls[len(gt_data):]:
        if extra:
            return 0.0

    for turn_idx, turn_gt in enumerate(gt_data):
        turn_agent = turn_func_calls[turn_idx]

        gt_dicts = []
        for gt_item in (turn_gt if isinstance(turn_gt, list) else [turn_gt]):
            if isinstance(gt_item, str):
                gt_dicts.append(_parse_bfcl_native_args(gt_item))

        if not gt_dicts:
            # 空 GT turn：模型在该轮不应调用任何工具
            if len(turn_agent) != 0:
                return 0.0
            continue

        agent_dicts = [
            (
                fc.get("name", ""),
                fc.get("arguments", {}),
            )
            for fc in turn_agent
        ]

        if len(agent_dicts) != len(gt_dicts):
            return 0.0
        for (a_name, a_args), (gt_name, gt_args) in zip(agent_dicts, gt_dicts):
            if a_name != gt_name or not _args_match(a_args, gt_args):
                return 0.0

    return 1.0


def _compute_flat_reward(
    func_calls: list[dict],
    gt_data: list,
) -> float:
    """扁平匹配（向后兼容，不推荐用于多轮任务）。"""
    gt_flat = []
    for turn_gt in gt_data:
        items = turn_gt if isinstance(turn_gt, list) else [turn_gt]
        for item in items:
            if isinstance(item, str):
                gt_flat.append(_parse_bfcl_native_args(item))

    if not gt_flat:
        return 0.0

    agent_dicts = [
        (
            fc.get("name", ""),
            fc.get("arguments", {}),
        )
        for fc in func_calls
    ]

    if len(agent_dicts) != len(gt_flat):
        return 0.0
    for (a_name, a_args), (gt_name, gt_args) in zip(agent_dicts, gt_flat):
        if a_name != gt_name or not _args_match(a_args, gt_args):
            return 0.0
    return 1.0
