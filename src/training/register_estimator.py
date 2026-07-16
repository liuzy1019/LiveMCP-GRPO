"""
verl 集成入口：注册 livemcp_grpo estimator + patch verl 传递 non_tensor_batch。

reference verl (ray_trainer.py:242-258) 的自定义 estimator 路径只传
token_level_rewards/response_mask/index(uid)/config，不传 non_tensor_batch。
这里 monkey-patch compute_advantage 使 livemcp_grpo 也收到 non_tensor_batch。

lambda_safe 更新和数据规范化逻辑在 hooks.py 中。
"""

import importlib
import functools
from typing import Optional

import numpy as np
from loguru import logger

from src.training.hooks import update_lambda_safe, validate_livemcp_non_tensor_batch


def register_livemcp_estimator(config: Optional[dict] = None) -> bool:
    """注册 livemcp_grpo estimator + patch verl 传递 non_tensor_batch。"""
    cfg = config or {}
    if not cfg.get("use_livemcp", True):
        logger.info("LiveMCP 已禁用")
        return False

    try:
        importlib.import_module("src.training.livemcp_grpo_estimator")
        logger.info("livemcp_grpo estimator 已注册")
    except Exception as e:
        logger.error(f"estimator 注册失败: {e}")
        return False

    # Patch verl 的 compute_advantage 使 livemcp_grpo 收到 non_tensor_batch
    try:
        mod = importlib.import_module("verl.trainer.ppo.ray_trainer")
        core_algos = importlib.import_module("verl.trainer.ppo.core_algos")

        original_fn = mod.compute_advantage

        @functools.wraps(original_fn)
        def patched_compute_advantage(data, adv_estimator, *args, **kwargs):
            from verl.trainer.ppo.core_algos import get_adv_estimator_fn

            # 处理 livemcp_grpo（走自定义 estimator 路径，注入 non_tensor_batch）
            if str(adv_estimator) == "livemcp_grpo":
                adv_estimator_fn = get_adv_estimator_fn(adv_estimator)
                bsz = data.batch["token_level_rewards"].shape[0]
                non_tensor_batch = validate_livemcp_non_tensor_batch(
                    data.non_tensor_batch, bsz
                )

                _config = kwargs.get("config")

                adv_kwargs = {
                    "token_level_rewards": data.batch["token_level_rewards"],
                    "response_mask": data.batch["response_mask"],
                    "config": _config,
                    "norm_adv_by_std_in_grpo": kwargs.get("norm_adv_by_std_in_grpo", True),
                }
                if non_tensor_batch and "uid" in non_tensor_batch:
                    adv_kwargs["index"] = non_tensor_batch["uid"]
                else:
                    adv_kwargs["index"] = np.arange(bsz)
                adv_kwargs["non_tensor_batch"] = non_tensor_batch
                if "reward_baselines" in data.batch:
                    adv_kwargs["reward_baselines"] = data.batch["reward_baselines"]

                # lambda_safe 更新（batch 边界，fcntl 文件锁保护）
                update_lambda_safe(non_tensor_batch, bsz)

                # 诊断：检查关键字段是否传递成功
                nb = non_tensor_batch
                nb_keys = set(nb.keys()) if nb else set()
                has_fields = {"perturbation_level", "group_id", "scenario_type"}.issubset(nb_keys)
                if not hasattr(patched_compute_advantage, '_diagnosed'):
                    logger.info(
                        f"livemcp_grpo monkey-patch: "
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
        logger.info("verl compute_advantage 已 patch（livemcp_grpo 可接收 non_tensor_batch）")
        if not callable(mod.compute_advantage):
            raise RuntimeError("verl compute_advantage patch verification failed: not callable")
        logger.debug("verl compute_advantage smoke check passed")
    except (ImportError, AttributeError) as e:
        raise RuntimeError(
            f"verl compute_advantage patch 失败，LiveMCP estimator 无法工作: {e}"
        ) from e

    return True


__all__ = [
    "register_livemcp_estimator",
]
