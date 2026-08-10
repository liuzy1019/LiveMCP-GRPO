"""训练回调钩子：lambda_safe 更新 + non_tensor_batch 数据规范化。

从 estimator.py 拆分出来的独立职责模块。
lambda_safe 更新通过 LambdaState.atomic_update() 实现跨进程安全。
"""

import numpy as np
from loguru import logger


# ── lambda_safe 跨 batch 更新 ──────────────────────────────────────

def update_lambda_safe(non_tensor_batch, batch_size: int) -> bool:
    """从 batch 的 c_safety 值更新 file-backed LambdaState（跨进程安全）。

    通过 fcntl 文件锁保护 load→update→save 原子性，
    防止 Ray 多 actor 并发更新时的 write-after-read 覆盖。

    Returns True if update succeeded. ``prove_baseline`` does not use the
    safety multiplier and returns False.  In ``oval_full`` missing or corrupt
    safety evidence is an integrity error, not a fixed-lambda fallback.
    """
    from src.training.hyperparams import get_config

    config = get_config()
    if config.reward_profile == "prove_baseline":
        return False

    c_safety_values: list[int] = []
    if non_tensor_batch and "c_safety" in non_tensor_batch:
        raw = non_tensor_batch["c_safety"]
        if isinstance(raw, np.ndarray):
            if raw.ndim > 0:
                c_safety_values = [int(v) for v in raw.tolist()]
        elif isinstance(raw, (list, tuple)):
            c_safety_values = [int(v) for v in raw]

    if not c_safety_values:
        raise ValueError(
            "oval_full requires canonical top-level c_safety for every batch"
        )
    if len(c_safety_values) != batch_size:
        raise ValueError(
            "c_safety length mismatch: "
            f"values={len(c_safety_values)}, batch_size={batch_size}"
        )

    try:
        from src.oval_mcp.training.lambda_state import LambdaState

        state, old_lambda, new_lambda, skipped = LambdaState.atomic_update(
            c_safety_values,
            path=config.lambda_state_path,
            k_stall=config.k_stall,
            tau_unsafe_stall=config.tau_unsafe_stall,
        )

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
        raise RuntimeError(f"lambda_safe update failed: {e}") from e


# ── non_tensor_batch 字段规范化 ────────────────────────────────────

def validate_livemcp_non_tensor_batch(non_tensor_batch, batch_size: int):
    """Require canonical top-level estimator metadata without reconstruction."""
    if not non_tensor_batch:
        raise ValueError("livemcp_grpo requires non_tensor_batch metadata")
    required = {"group_id", "perturbation_level", "scenario_type"}
    missing = sorted(required - set(non_tensor_batch))
    if missing:
        raise ValueError(f"livemcp_grpo missing canonical metadata: {missing}")
    for field in sorted(required):
        values = non_tensor_batch[field]
        if isinstance(values, np.ndarray):
            size = int(values.shape[0]) if values.ndim else 1
        elif isinstance(values, (list, tuple)):
            size = len(values)
        else:
            size = 1
        if size != batch_size:
            raise ValueError(
                f"livemcp_grpo metadata length mismatch: "
                f"{field}={size}, batch_size={batch_size}"
            )
    return non_tensor_batch


__all__ = [
    "update_lambda_safe",
    "validate_livemcp_non_tensor_batch",
]
