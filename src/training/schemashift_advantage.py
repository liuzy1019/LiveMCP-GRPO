"""
SchemaShift Advantage 计算（参考实现 / 单元测试 oracle）。

⚠️  生产路径不使用本模块。verl 训练流程实际调用的是
    ``src/training/schemashift_grpo_estimator.py`` 中通过
    ``register_estimator.py`` 注册的版本，后者在 ``task_id × level``
    两级上做分层（本模块只在全 batch 同 level 上做分层），语义不等价。

本模块保留用于：
    1. 算法说明 / 公式推导的最小可读实现；
    2. ``tests/test_advantage.py`` 中以独立函数形式做单元化校验。
请勿在新代码中调用，更不要从这里复制逻辑迁移到生产路径。
"""

import torch
from loguru import logger

# 扰动强度级别（从已删除的 src/envs/schema_perturber 提取，OVAL 路线用于分层 advantage 计算）
PerturbationLevel = str
LEVEL_NONE: PerturbationLevel = "none"
LEVEL_MILD: PerturbationLevel = "mild"
LEVEL_MODERATE: PerturbationLevel = "moderate"
LEVEL_STRONG: PerturbationLevel = "strong"

TRAINING_LEVELS = [LEVEL_NONE, LEVEL_MILD, LEVEL_STRONG]


def compute_schemashift_advantages(
    rewards: torch.Tensor,
    perturbation_levels: list[PerturbationLevel],
    beta: float = 0.25,
    eps: float = 1e-8,
) -> torch.Tensor:
    """计算 SchemaShift advantage：A = strat_z + beta * global_z。

    分层归一化：每层内独立 z-score。
    全局残差：beta * global_z，确保跨层梯度流动。

    Args:
        rewards: [N] 每个 rollout 的 scalar reward。
        perturbation_levels: [N] 每个 rollout 对应的扰动强度。
        beta: 全局残差强度。0 = 纯分层，0.25 = 默认。
        eps: 数值稳定常数。

    Returns:
        [N] 每个 rollout 的最终 advantage。

    Raises:
        ValueError: rewards 和 perturbation_levels 长度不匹配。

    Example:
        >>> rewards = torch.tensor([0.9, 0.8, 0.7, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0])
        >>> levels = ["none", "none", "none", "mild", "mild", "mild", "strong", "strong", "strong"]
        >>> adv = compute_schemashift_advantages(rewards, levels)
    """
    N = len(rewards)
    if len(perturbation_levels) != N:
        raise ValueError(
            f"rewards ({N}) 和 perturbation_levels ({len(perturbation_levels)}) "
            f"长度不匹配"
        )

    device = rewards.device
    advantages = torch.zeros(N, device=device)

    # ── Step 1: 按扰动强度分层归一化 ──
    levels_present = set(perturbation_levels)
    unsupported = levels_present - set(TRAINING_LEVELS)
    if unsupported:
        logger.warning(
            f"扰动强度包含非训练级别: {unsupported}，"
            f"将不回退到层内归一化（仅 global_z 生效）"
        )
    logger.debug(
        f"分层归一化: 扰动强度分布={levels_present}"
    )

    for level in TRAINING_LEVELS:
        if level not in levels_present:
            continue

        # 找到该强度的 rollout 索引
        mask = torch.tensor(
            [pl == level for pl in perturbation_levels],
            device=device,
            dtype=torch.bool,
        )
        idxs = torch.where(mask)[0]
        num_samples = len(idxs)

        if num_samples < 2:
            logger.warning(
                f"扰动强度 '{level}' 只有 {num_samples} 个样本，"
                f"回退到全局归一化"
            )
            continue

        level_rewards = rewards[idxs]
        level_mean = level_rewards.mean()
        level_std = level_rewards.std(unbiased=False).clamp(min=eps)

        level_advantages = (level_rewards - level_mean) / level_std
        advantages[idxs] = level_advantages

        logger.debug(
            f"  层 '{level}': {num_samples} 个样本, "
            f"mean={level_mean:.4f}, std={level_std:.4f}, "
            f"adv=[{level_advantages[0]:.4f}, ...]"
        )

    # ── Step 2: 全局残差校正 ──
    global_mean = rewards.mean()
    global_std = rewards.std(unbiased=False).clamp(min=eps)
    global_z = (rewards - global_mean) / global_std

    # ── Step 3: 融合（加法形式）──
    # A = strat_z + beta * global_z
    # beta=0 → 纯分层；beta=0.25 → 加全局残差
    final_advantages = advantages + beta * global_z

    logger.debug(
        f"Advantage 融合: beta={beta}, "
        f"adv_range=[{final_advantages.min():.4f}, {final_advantages.max():.4f}]"
    )

    return final_advantages


def compute_standard_grpo_advantages(
    rewards: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """标准 GRPO advantage 计算（基线对照用）。

    Args:
        rewards: [N] 每个 rollout 的 scalar reward。
        eps: 数值稳定常数。

    Returns:
        [N] advantage 向量。
    """
    mean = rewards.mean()
    std = rewards.std(unbiased=False).clamp(min=eps)
    advantages = (rewards - mean) / std
    return advantages


# ============================================================
# MCP-RL Full: 2 维 stratum 支持
# ============================================================

# 支持的 scenario_type 值
SCENARIO_TYPES = (
    "single_step",
    "distractor",
    "conditioned_tool_call",
    "final_answer",
    "output_error",
)


def compute_stratified_advantage(
    rewards: torch.Tensor,
    perturbation_levels: list[str],
    scenario_types: list[str],
    beta: float = 0.25,
    min_stratum_size: int = 3,
    eps: float = 1e-8,
) -> torch.Tensor:
    """MCP-RL Full 分层 advantage：stratum = (perturbation_level, scenario_type)。

    Fallback 逻辑（mcp_tools_rl_project_plan.md §12）：
      - stratum 内 >= min_stratum_size 样本：正常 z-score
      - stratum 内 == 2 样本：只减均值，std 设为 1.0
      - stratum 内 == 1 样本：回退到 scenario-level（忽略 perturbation_level）
      - scenario-level 也只有 1 样本：回退到 global advantage

    Args:
        rewards: [N] 每个 rollout 的 scalar reward。
        perturbation_levels: [N] 每个 rollout 的扰动强度。
        scenario_types: [N] 每个 rollout 的场景类型。
        beta: 全局残差强度。
        min_stratum_size: 正常 z-score 所需的最小样本数。
        eps: 数值稳定常数。

    Returns:
        [N] 每个 rollout 的最终 advantage。

    Raises:
        ValueError: 输入长度不匹配。
    """
    N = len(rewards)
    if len(perturbation_levels) != N or len(scenario_types) != N:
        raise ValueError(
            f"rewards ({N}), perturbation_levels ({len(perturbation_levels)}), "
            f"scenario_types ({len(scenario_types)}) 长度不匹配"
        )

    device = rewards.device
    strat_advs = torch.zeros(N, device=device)

    # 构建 stratum → indices 映射
    from collections import defaultdict
    stratum_indices: dict[tuple[str, str], list[int]] = defaultdict(list)
    scenario_indices: dict[str, list[int]] = defaultdict(list)

    for i in range(N):
        stratum_key = (perturbation_levels[i], scenario_types[i])
        stratum_indices[stratum_key].append(i)
        scenario_indices[scenario_types[i]].append(i)

    # 对每个 stratum 计算层内 advantage
    for stratum_key, indices in stratum_indices.items():
        idx_tensor = torch.tensor(indices, device=device)
        stratum_rewards = rewards[idx_tensor]
        n = len(indices)

        if n >= min_stratum_size:
            # 正常 z-score
            mean = stratum_rewards.mean()
            std = stratum_rewards.std(unbiased=False).clamp(min=eps)
            strat_advs[idx_tensor] = (stratum_rewards - mean) / std
        elif n == 2:
            # 只减均值，std 设为 1.0
            mean = stratum_rewards.mean()
            strat_advs[idx_tensor] = stratum_rewards - mean
        elif n == 1:
            # 回退到 scenario-level
            scenario = stratum_key[1]
            scenario_idxs = scenario_indices[scenario]
            if len(scenario_idxs) >= 2:
                sc_tensor = torch.tensor(scenario_idxs, device=device)
                sc_rewards = rewards[sc_tensor]
                sc_mean = sc_rewards.mean()
                if len(scenario_idxs) >= min_stratum_size:
                    sc_std = sc_rewards.std(unbiased=False).clamp(min=eps)
                    # 只设置当前这个样本的 advantage
                    strat_advs[idx_tensor] = (stratum_rewards - sc_mean) / sc_std
                else:
                    strat_advs[idx_tensor] = stratum_rewards - sc_mean
            else:
                # scenario-level 也只有 1 样本，回退到 global
                global_mean = rewards.mean()
                global_std = rewards.std(unbiased=False).clamp(min=eps)
                strat_advs[idx_tensor] = (stratum_rewards - global_mean) / global_std

    # 全局 z-score
    global_mean = rewards.mean()
    global_std = rewards.std(unbiased=False).clamp(min=eps)
    global_z = (rewards - global_mean) / global_std

    # 融合
    final_advantages = strat_advs + beta * global_z

    return final_advantages
