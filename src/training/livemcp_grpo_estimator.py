"""
LiveMCP-GRPO 自定义 advantage estimator。

注册为 verl 的 "livemcp_grpo" estimator。
只从 non_tensor_batch（由 _get_gen_batch 保留，register_estimator.py 转发）读取
perturbation_level、scenario_type 和 group_id；缺字段 fail-closed。
"""

from collections import defaultdict
import os

import numpy as np
import torch
from loguru import logger

try:
    from verl.trainer.ppo.core_algos import register_adv_est
except ImportError as e:
    raise ImportError(
        "verl 未安装或版本不兼容，livemcp_grpo estimator 无法注册。"
        f"原始错误: {e}"
    )

from src.training.advantage_core import compute_standard_grpo_advantages


# ── LATA helpers ────────────────────────────────────────────────────

_LATA_WARNED = False


def _resolve_lata_mode(config) -> str:
    """从 config 或环境变量解析 LATA 模式。

    优先级: config > get_config() > env vars > default "none".
    禁用 sentinel: "0", "false", "off", "none" 均视为禁用.
    """
    _DISABLED = frozenset({"0", "false", "off", "none", ""})
    # 1) hydra config（如果已通过 register_estimator 注入）
    if config:
        val = str(config.get("lata_mode", config.get("lata", "none"))).lower()
        return "none" if val in _DISABLED else val
    # 2) 统一配置（优先使用 LIVEMCP_LATA / OVAL_LATA 但统一到 get_config）
    try:
        from src.training.livemcp_hyperparams import get_config
        lata = get_config().lata_mode
        return "none" if lata.lower() in _DISABLED else lata.lower()
    except ImportError:
        pass
    # 3) 直接环境变量兜底
    for key in ("LIVEMCP_LATA", "OVAL_LATA"):
        val = os.environ.get(key, "")
        if val:
            return "none" if val.lower() in _DISABLED else val.lower()
    return "none"


def _apply_lata(
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    mode: str,
    config,
) -> torch.Tensor:
    """Apply LATA allocation to trajectory-level advantages, fail-closed."""
    global _LATA_WARNED
    try:
        from src.oval_mcp.training.lata import LATAAllocator, LATAConfig
        allocator = LATAAllocator(LATAConfig(mode=mode))
        result = allocator.allocate_from_mask(advantages, response_mask)
        mean_len = result.mean_length
        scale_range = (min(result.per_token_scale), max(result.per_token_scale))
        if not _LATA_WARNED:
            logger.info(
                f"[LATA] mode={mode} | mean_len={mean_len:.1f} "
                f"| scale_range=[{scale_range[0]:.3f}, {scale_range[1]:.3f}]"
            )
            _LATA_WARNED = True

        # P1-13: 跟踪 mean_response_length 趋势。LATA 的 sqrt(L) 归一化
        # 可能使模型偏好更短的回复。如果 mean_len 持续下降且 R_task 也下降，
        # 说明 LATA 过强，应回退到 "none" 或降低归一化强度。
        if not hasattr(_apply_lata, '_len_history'):
            _apply_lata._len_history: list[float] = []
        _apply_lata._len_history.append(mean_len)
        if len(_apply_lata._len_history) >= 50:
            recent = _apply_lata._len_history[-50:]
            if recent:
                # Simple trend: slope of last 50 points via linear regression
                n = len(recent)
                x_mean = (n - 1) / 2
                y_mean = sum(recent) / n
                numerator = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(recent))
                denominator = sum((i - x_mean) ** 2 for i in range(n))
                slope = numerator / denominator if denominator > 0 else 0
                if slope < -0.01:
                    logger.warning(
                        f"[LATA LENGTH TREND] mean_response_length 正在下降 "
                        f"(slope={slope:.3f}/step over last 50 batches, "
                        f"current mean={mean_len:.1f})。"
                        f"如果 R_task 也同步下降，建议将 LATA 模式改为 'none'。"
                    )
            _apply_lata._len_history = _apply_lata._len_history[-100:]  # keep last 100
        return result.token_advantages
    except Exception as e:
        raise RuntimeError(f"LATA allocation failed in mode {mode!r}: {e}") from e


def _diagnose_batch(index, non_tensor_batch, task_ids, levels, scenario_types, scores):
    """记录诊断信息，帮助排查数据流问题。"""
    n = len(scores)
    nunique_task = len(set(task_ids))
    nunique_level = len(set(levels))
    nunique_scenario = len(set(scenario_types))
    nb_fields = set(non_tensor_batch.keys()) if non_tensor_batch else set()
    logger.info(
        f"[livemcp_grpo] batch={n}, tasks={nunique_task}, "
        f"levels={nunique_level}, scenarios={nunique_scenario}, "
        f"nb_fields={nb_fields}, "
        f"score_range=[{scores.min().item():.4f}, {scores.max().item():.4f}]"
    )

    logger.debug(
        "livemcp_grpo groups are per-prompt rollout groups; "
        "perturbation/scenario labels are integrity metadata"
    )


@register_adv_est("livemcp_grpo")
def compute_livemcp_grpo_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config=None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-prompt GRPO advantage with optional OVAL saturation/LATA.

    与 verl 集成：
    - verl 的 ray_trainer 通过 adv_kwargs 传入 token_level_rewards, response_mask, index, config
    - non_tensor_batch 通过 register_estimator.py 的 monkey-patch 注入
    - group_id=task_id; each group contains N rollouts of one prompt.
    - perturbation_level/scenario_type are integrity diagnostics, not
      within-group strata. Mixed labels indicate invalid grouping.
    """
    from src.training.livemcp_hyperparams import get_config

    hp = get_config()
    scores = token_level_rewards.sum(dim=-1)
    bsz = scores.shape[0]

    non_tensor_batch = kwargs.get("non_tensor_batch")
    if non_tensor_batch is None:
        raise ValueError("livemcp_grpo requires non_tensor_batch metadata")
    required_metadata = {"group_id", "perturbation_level", "scenario_type"}
    missing_metadata = sorted(required_metadata - set(non_tensor_batch))
    if missing_metadata:
        raise ValueError(
            f"livemcp_grpo missing canonical metadata: {missing_metadata}"
        )

    def _to_list(val):
        if isinstance(val, np.ndarray) and val.ndim > 0:
            return val.tolist()
        if isinstance(val, (list, tuple)):
            return list(val)
        return [str(val)]

    task_ids = _to_list(non_tensor_batch["group_id"])
    levels = _to_list(non_tensor_batch["perturbation_level"])
    scenario_types = _to_list(non_tensor_batch["scenario_type"])
    source = "non_tensor_batch"

    # P1-8: 校验 metadata 数组长度等于 batch size
    if len(task_ids) != bsz or len(levels) != bsz or len(scenario_types) != bsz:
        raise ValueError(
            f"[livemcp_grpo] metadata 长度不匹配 batch size: "
            f"task_ids={len(task_ids)}, levels={len(levels)}, "
            f"scenario_types={len(scenario_types)}, bsz={bsz}, source={source}"
        )

    with torch.no_grad():
        advantages = torch.zeros(bsz, device=scores.device)

        # 按 task_id 分组
        task2indices = defaultdict(list)
        for i, tid in enumerate(task_ids):
            task2indices[tid].append(i)

        group_sizes = {tid: len(indices) for tid, indices in task2indices.items()}
        distinct_sizes = set(group_sizes.values())
        if len(distinct_sizes) > 1:
            raise ValueError(
                "livemcp_grpo received unequal per-prompt rollout groups: "
                + ", ".join(
                    f"{tid}:{size}" for tid, size in list(group_sizes.items())[:5]
                )
            )

        # A batch without repeated rollout groups cannot implement the declared
        # LiveMCP estimator. Never silently switch to batch-level GRPO.
        max_group_size = max(len(v) for v in task2indices.values())
        if max_group_size == 1:
            raise ValueError(
                "livemcp_grpo requires at least two rollouts per group in each batch"
            )
        else:
            for tid, indices in task2indices.items():
                idx_tensor = torch.tensor(indices, device=scores.device)
                group_levels = {str(levels[i]) for i in indices}
                group_scenarios = {str(scenario_types[i]) for i in indices}
                if len(group_levels) != 1 or len(group_scenarios) != 1:
                    raise ValueError(
                        "one GRPO group must contain rollouts of one prompt; "
                        f"group_id={tid!r}, levels={sorted(group_levels)}, "
                        f"scenarios={sorted(group_scenarios)}"
                    )
                group_scores = scores[idx_tensor]
                if norm_adv_by_std_in_grpo:
                    advantages[idx_tensor] = compute_standard_grpo_advantages(
                        group_scores, epsilon=epsilon,
                    )
                else:
                    advantages[idx_tensor] = group_scores - group_scores.mean()

        # 首次执行时打诊断日志
        if not hasattr(compute_livemcp_grpo_advantage, '_diagnosed'):
            _diagnose_batch(index, non_tensor_batch, task_ids, levels, scenario_types, scores)
            compute_livemcp_grpo_advantage._diagnosed = True

        # ── 饱和组检测与跳过（§9.2-9.3） ──
        # std(J) < min_group_std → advantage = 0，不产生 policy gradient
        # 饱和组 rollout 仍参与 lambda_safe 的 hat_C_batch（在 register_estimator 层已处理）
        min_group_std = float(
            config.get("min_group_std", hp.min_group_std)
            if config else hp.min_group_std
        )
        n_saturated_groups = 0
        n_total_groups = len(task2indices)
        saturated_group_ids: list[str] = []

        if max_group_size >= 2:
            for tid, indices in task2indices.items():
                if len(indices) < 2:
                    continue
                group_j = scores[torch.tensor(indices, device=scores.device)]
                g_mean = group_j.mean()
                g_var = ((group_j - g_mean) ** 2).mean()
                g_std = g_var.sqrt()
                if g_std < min_group_std:
                    advantages[torch.tensor(indices, device=scores.device)] = 0.0
                    n_saturated_groups += 1
                    saturated_group_ids.append(tid)

        if n_saturated_groups > 0 and not hasattr(compute_livemcp_grpo_advantage, '_sat_warned'):
            logger.warning(
                f"[livemcp_grpo] SATURATION: {n_saturated_groups}/{n_total_groups} groups skipped "
                f"(std(J) < {min_group_std:.0e}). "
                f"saturated_ids (前5): {saturated_group_ids[:5]}"
            )
            compute_livemcp_grpo_advantage._sat_warned = True

        # ── LATA: Length-Aware Token Allocation ──
        # 当全部组饱和（advantage 已全零）时跳过 LATA 分配，避免无效计算
        lata_mode = _resolve_lata_mode(config)
        if lata_mode != "none" and n_saturated_groups < n_total_groups:
            advantages = _apply_lata(advantages, response_mask, lata_mode, config)
        else:
            advantages = advantages.unsqueeze(-1) * response_mask

    return advantages, advantages.clone()
