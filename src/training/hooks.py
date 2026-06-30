"""训练回调钩子：lambda_safe 更新 + non_tensor_batch 数据规范化。

从 register_estimator.py 拆分出来的独立职责模块。
lambda_safe 更新通过 LambdaState.atomic_update() 实现跨进程安全。
"""

import numpy as np
from loguru import logger

LAMBDA_UPDATE_DIAGNOSED = False


# ── lambda_safe 跨 batch 更新 ──────────────────────────────────────

def update_lambda_safe(non_tensor_batch, batch_size: int) -> bool:
    """从 batch 的 c_safety 值更新 file-backed LambdaState（跨进程安全）。

    通过 fcntl 文件锁保护 load→update→save 原子性，
    防止 Ray 多 actor 并发更新时的 write-after-read 覆盖。

    Returns True if update succeeded, False if c_safety unavailable.
    """
    global LAMBDA_UPDATE_DIAGNOSED

    c_safety_values: list[int] = []
    possible_keys = ["c_safety", "C_safety", "is_unsafe"]

    for key in possible_keys:
        if non_tensor_batch and key in non_tensor_batch:
            raw = non_tensor_batch[key]
            if isinstance(raw, np.ndarray):
                if raw.ndim > 0:
                    c_safety_values = [int(v) for v in raw.tolist()]
            elif isinstance(raw, list):
                c_safety_values = [int(v) for v in raw]
            break

    if not c_safety_values:
        # 尝试从 reward_extra_info 提取
        if non_tensor_batch and "reward_extra_info" in non_tensor_batch:
            extra = non_tensor_batch["reward_extra_info"]
            for item in (extra.tolist() if isinstance(extra, np.ndarray) else extra):
                if isinstance(item, dict) and "c_safety" in item:
                    c_safety_values.append(int(item["c_safety"]))

    if not c_safety_values:
        if not LAMBDA_UPDATE_DIAGNOSED:
            logger.debug(
                "[lambda_safe] c_safety 不在 non_tensor_batch 中，"
                "lambda_safe 保持固定值（LambdaState 不可用）"
            )
            LAMBDA_UPDATE_DIAGNOSED = True
        return False

    try:
        from src.oval_mcp.training.lambda_state import LambdaState

        state, old_lambda, new_lambda, skipped = LambdaState.atomic_update(c_safety_values)

        if not hasattr(update_lambda_safe, '_log_step'):
            update_lambda_safe._log_step = 0
        update_lambda_safe._log_step += 1

        if skipped:
            logger.warning(
                f"[lambda_safe STALL] step={state.step} streak={state.stall_streak} "
                f"hat_C={sum(c_safety_values)/len(c_safety_values):.3f} "
                f"lambda FROZEN at {state.lambda_safe:.4f}"
            )
        elif state.is_stall_frozen:
            logger.info(
                f"[lambda_safe FROZEN] step={state.step} "
                f"lambda={state.lambda_safe:.4f} (decrease allowed)"
            )
        elif update_lambda_safe._log_step % 10 == 1:
            logger.info(
                f"[lambda_safe] step={state.step} "
                f"hat_C={sum(c_safety_values)/len(c_safety_values):.3f} "
                f"lambda: {old_lambda:.4f} → {new_lambda:.4f}"
            )
        return True
    except Exception as e:
        logger.warning(f"[lambda_safe] 更新失败: {e}")
        return False


# ── non_tensor_batch 字段规范化 ────────────────────────────────────

def normalize_livemcp_non_tensor_batch(non_tensor_batch, batch_size: int):
    """将 LiveMCP 字段从 extra_info 提升到顶层 non_tensor_batch。

    verl 可能在 generation/reward 流程中保留整个 extra_info dict 但丢失顶层
    parquet 列。estimator 有 extra_info fallback，但提前提升字段能保持诊断
    日志真实，避免静默降级。
    """
    if not non_tensor_batch:
        return non_tensor_batch

    required = {
        "episode_id",
        "group_id",
        "perturbation_level",
        "scenario_type",
        "action_type",
        "tool_name",
    }
    if required.issubset(non_tensor_batch.keys()):
        return non_tensor_batch
    if "extra_info" not in non_tensor_batch:
        return non_tensor_batch

    extra_infos = non_tensor_batch["extra_info"]
    if isinstance(extra_infos, np.ndarray) and extra_infos.ndim > 0:
        extras = extra_infos.tolist()
    elif isinstance(extra_infos, (list, tuple)):
        extras = list(extra_infos)
    else:
        extras = [extra_infos] * batch_size

    from src.utils import normalize_extra_info
    normalized_extras = [normalize_extra_info(e) for e in extras]
    extras = normalized_extras

    if not extras or not isinstance(extras[0], dict):
        return non_tensor_batch
    if len(extras) == 1 and batch_size > 1:
        extras = extras * batch_size

    normalized = dict(non_tensor_batch)

    def _values(field: str, default):
        return np.array(
            [e.get(field, default(i, e) if callable(default) else default) for i, e in enumerate(extras)],
            dtype=object,
        )

    if "episode_id" not in normalized:
        normalized["episode_id"] = _values("episode_id", lambda i, e: e.get("uid", f"unk_{i}"))
    if "group_id" not in normalized:
        normalized["group_id"] = _values("group_id", lambda i, e: e.get("episode_id", f"unk_{i}"))
    if "perturbation_level" not in normalized:
        normalized["perturbation_level"] = _values("perturbation_level", "none")
    if "scenario_type" not in normalized:
        normalized["scenario_type"] = _values("scenario_type", "single_step")
    if "action_type" not in normalized:
        normalized["action_type"] = _values("action_type", "")
    if "tool_name" not in normalized:
        normalized["tool_name"] = _values("tool_name", "")

    return normalized


__all__ = [
    "update_lambda_safe",
    "normalize_livemcp_non_tensor_batch",
]