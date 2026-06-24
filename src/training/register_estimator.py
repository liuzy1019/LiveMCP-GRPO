"""
verl 集成入口：注册 schemashift_grpo estimator + patch verl 传递 non_tensor_batch
+ lambda_safe 跨 batch 更新（file-backed LambdaState）。

reference verl (ray_trainer.py:242-258) 的自定义 estimator 路径只传
token_level_rewards/response_mask/index(uid)/config，不传 non_tensor_batch。
这里 monkey-patch compute_advantage 使 schemashift_grpo 也收到 non_tensor_batch。

lambda_safe 更新流：
  1. oval_reward_fn.compute_score 输出 score(=J), c_safety, r_task 等
  2. verl 将 score 存入 token_level_rewards，c_safety 可能保留在 non_tensor_batch
  3. schemashift_grpo estimator 在 batch 边界调用 LambdaState.update()
  4. 更新后的 lambda_safe 被 oval_reward_fn 读取（通过 LambdaState.load_or_default）
"""

import importlib
import functools
from typing import Optional

import numpy as np
from loguru import logger

LAMBDA_UPDATE_DIAGNOSED = False


def _update_lambda_safe_safe(non_tensor_batch, batch_size: int) -> bool:
    """Attempt to update lambda_safe from batch c_safety values.

    Returns True if update succeeded, False if c_safety unavailable.
    """

    # 尝试从 non_tensor_batch 获取 c_safety 值
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
        global LAMBDA_UPDATE_DIAGNOSED
        if not LAMBDA_UPDATE_DIAGNOSED:
            logger.debug(
                "[lambda_safe] c_safety 不在 non_tensor_batch 中，"
                "lambda_safe 保持固定值（Phase 1 默认行为）"
            )
            LAMBDA_UPDATE_DIAGNOSED = True
        return False

    try:
        from src.oval_mcp.training.lambda_state import LambdaState
        state = LambdaState.load_or_default()
        old_lambda = state.lambda_safe
        new_lambda, skipped = state.update(c_safety_values)
        state.save()

        if not hasattr(_update_lambda_safe_safe, '_log_step'):
            _update_lambda_safe_safe._log_step = 0
        _update_lambda_safe_safe._log_step += 1

        # stall warning
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
        elif _update_lambda_safe_safe._log_step % 10 == 1:
            logger.info(
                f"[lambda_safe] step={state.step} "
                f"hat_C={sum(c_safety_values)/len(c_safety_values):.3f} "
                f"lambda: {old_lambda:.4f} → {new_lambda:.4f}"
            )
        return True
    except Exception as e:
        logger.warning(f"[lambda_safe] 更新失败: {e}")
        return False


def _normalize_schemashift_non_tensor_batch(non_tensor_batch, batch_size: int):
    """Promote SchemaShift fields from extra_info to top-level non_tensor_batch.

    verl may preserve the whole extra_info dict while dropping some top-level
    parquet columns during generation/reward plumbing. The estimator supports an
    extra_info fallback, but promoting fields here keeps diagnostics honest and
    avoids silently degrading the SchemaShift path.
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

    # P1-2: extra_info 元素可能是 JSON 字符串（pyarrow 序列化后）
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


def register_schemashift_estimator(config: Optional[dict] = None) -> bool:
    """注册 schemashift_grpo estimator + patch verl 传递 non_tensor_batch。"""
    cfg = config or {}
    if not cfg.get("use_schemashift", True):
        logger.info("SchemaShift 已禁用")
        return False

    try:
        from src.training import schemashift_grpo_estimator  # noqa: F401
        logger.info("schemashift_grpo estimator 已注册")
    except Exception as e:
        logger.error(f"estimator 注册失败: {e}")
        return False

    # Patch verl 的 compute_advantage 使 schemashift_grpo 收到 non_tensor_batch
    try:
        mod = importlib.import_module("verl.trainer.ppo.ray_trainer")
        core_algos = importlib.import_module("verl.trainer.ppo.core_algos")

        original_fn = mod.compute_advantage

        @functools.wraps(original_fn)
        def patched_compute_advantage(data, adv_estimator, *args, **kwargs):
            from verl.trainer.ppo.core_algos import get_adv_estimator_fn

            # 处理 schemashift_grpo（走自定义 estimator 路径，注入 non_tensor_batch）
            if str(adv_estimator) == "schemashift_grpo":
                adv_estimator_fn = get_adv_estimator_fn(adv_estimator)
                bsz = data.batch["token_level_rewards"].shape[0]
                non_tensor_batch = _normalize_schemashift_non_tensor_batch(
                    data.non_tensor_batch, bsz
                )
                adv_kwargs = {
                    "token_level_rewards": data.batch["token_level_rewards"],
                    "response_mask": data.batch["response_mask"],
                    "config": kwargs.get("config"),
                    "norm_adv_by_std_in_grpo": kwargs.get("norm_adv_by_std_in_grpo", True),
                }
                if non_tensor_batch and "uid" in non_tensor_batch:
                    adv_kwargs["index"] = non_tensor_batch["uid"]
                else:
                    adv_kwargs["index"] = np.arange(bsz)
            # 注入 non_tensor_batch（包含 perturbation_level, group_id）
                # 注：这些字段被 _get_gen_batch 保留在 batch.non_tensor_batch 中
                # （见 verl/trainer/ppo/ray_trainer.py _get_gen_batch 的修改）
                adv_kwargs["non_tensor_batch"] = non_tensor_batch
                if "reward_baselines" in data.batch:
                    adv_kwargs["reward_baselines"] = data.batch["reward_baselines"]

                # ── lambda_safe 更新（batch 边界） ──
                _update_lambda_safe_safe(non_tensor_batch, bsz)

                # 诊断：检查关键字段是否传递成功
                nb = non_tensor_batch
                nb_keys = set(nb.keys()) if nb else set()
                has_fields = {"perturbation_level", "group_id", "scenario_type"}.issubset(nb_keys)
                if not hasattr(patched_compute_advantage, '_diagnosed'):
                    logger.info(
                        f"schemashift_grpo monkey-patch: "
                        f"batch_size={bsz}, "
                        f"non_tensor_batch_keys={nb_keys}, "
                        f"has_perturbation_level={has_fields}"
                    )
                    patched_compute_advantage._diagnosed = True

                advantages, returns = adv_estimator_fn(**adv_kwargs)
                data.batch["advantages"] = advantages
                data.batch["returns"] = returns
                return data
            else:
                return original_fn(data, adv_estimator, *args, **kwargs)

        mod.compute_advantage = patched_compute_advantage
        logger.info("verl compute_advantage 已 patch（schemashift_grpo 可接收 non_tensor_batch）")
    except (ImportError, AttributeError) as e:
        raise RuntimeError(
            f"verl compute_advantage patch 失败，SchemaShift estimator 无法工作: {e}"
        ) from e

    return True
