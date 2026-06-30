"""
分层 advantage 纯函数 — 单一权威来源（Single Source of Truth）。

从 livemcp_advantage.py 和 livemcp_grpo_estimator.py 提取出的共享逻辑。
不依赖 verl/Ray/logger，可独立测试。
"""

from collections import defaultdict

import torch

# ── 标准 GRPO（batch-level z-score） ───────────────────────────────

def compute_standard_grpo_advantages(
    rewards: torch.Tensor,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """标准 GRPO：batch 内 z-score。

    所有 reward 相同时返回全零。
    """
    if rewards.numel() < 2:
        return torch.zeros_like(rewards)
    mean = rewards.mean()
    std = rewards.std(unbiased=False)
    if std < epsilon:
        return torch.zeros_like(rewards)
    return (rewards - mean) / std


# ── 分层 advantage（单组） ──────────────────────────────────────────

def compute_per_group_stratified_advantages(
    group_scores: torch.Tensor,
    group_levels: list[str],
    group_scenarios: list[str],
    beta: float = 0.25,
    epsilon: float = 1e-6,
    min_stratum_size: int = 3,
    norm_by_std: bool = True,
) -> torch.Tensor:
    """单组的分层 advantage 计算。

    对 group 内按 (level, scenario) 二维分层做层内 z-score，
    再加 beta 加权的全局残差：
        A = strat_z + beta * global_z

    层内样本不足 min_stratum_size 时逐级回退：
      stratum → scenario → group

    Args:
        group_scores: (n,) tensor，组内每个样本的 reward。
        group_levels: 长度 n 的 perturbation_level 列表。
        group_scenarios: 长度 n 的 scenario_type 列表。
        beta: 全局残差权重（0=纯层内，1=全 global）。
        epsilon: std 下限，防止除零。
        min_stratum_size: 层内最小样本数，低于此值不除 std。
        norm_by_std: 是否除以 std（False 时只减均值）。

    Returns:
        (n,) tensor，优势值。
    """
    n_group = len(group_scores)
    if not (len(group_levels) == n_group and len(group_scenarios) == n_group):
        raise ValueError(
            f"length mismatch: group_scores={n_group}, "
            f"group_levels={len(group_levels)}, "
            f"group_scenarios={len(group_scenarios)}"
        )

    # 二维分层索引
    stratum2local = defaultdict(list)
    scenario2local = defaultdict(list)
    for local_i, (level, scenario) in enumerate(
        zip(group_levels, group_scenarios)
    ):
        stratum2local[(level, scenario)].append(local_i)
        scenario2local[scenario].append(local_i)

    strat_advs = torch.zeros(n_group, device=group_scores.device)

    # Step 1: 层内归一化（带 fallback）
    for stratum_key, loc_indices in stratum2local.items():
        loc_tensor = torch.tensor(loc_indices, device=group_scores.device)
        stratum_scores = group_scores[loc_tensor]
        n_stratum = len(loc_indices)

        if n_stratum >= min_stratum_size:
            s_mean = stratum_scores.mean()
            if norm_by_std:
                s_std = stratum_scores.std(unbiased=False).clamp(min=epsilon)
                strat_advs[loc_tensor] = (stratum_scores - s_mean) / s_std
            else:
                strat_advs[loc_tensor] = stratum_scores - s_mean
        elif n_stratum == 2:
            # 只减均值，不除 std
            strat_advs[loc_tensor] = stratum_scores - stratum_scores.mean()
        elif n_stratum == 1:
            # 回退到 scenario-level
            scenario = stratum_key[1]
            sc_indices = scenario2local[scenario]
            if len(sc_indices) >= 2:
                sc_tensor = torch.tensor(sc_indices, device=group_scores.device)
                sc_scores = group_scores[sc_tensor]
                sc_mean = sc_scores.mean()
                if len(sc_indices) >= min_stratum_size and norm_by_std:
                    sc_std = sc_scores.std(unbiased=False).clamp(min=epsilon)
                    strat_advs[loc_tensor] = (stratum_scores - sc_mean) / sc_std
                else:
                    strat_advs[loc_tensor] = stratum_scores - sc_mean
            else:
                # 回退到 group-level global
                g_mean = group_scores.mean()
                if n_group >= min_stratum_size and norm_by_std:
                    g_std = group_scores.std(unbiased=False).clamp(min=epsilon)
                    strat_advs[loc_tensor] = (stratum_scores - g_mean) / g_std
                else:
                    strat_advs[loc_tensor] = stratum_scores - g_mean

    # Step 2: 全局 z-score 残差
    group_mean = group_scores.mean()
    if norm_by_std and n_group >= 2:
        group_std = group_scores.std(unbiased=False).clamp(min=epsilon)
        global_z = (group_scores - group_mean) / group_std
    else:
        global_z = group_scores - group_mean

    # Step 3: A = strat_z + beta * global_z
    return strat_advs + beta * global_z


# ── 便捷封装 ────────────────────────────────────────────────────────

def compute_livemcp_advantages(
    rewards: torch.Tensor,
    levels: list[str],
    scenario_types: list[str] | None = None,
    beta: float = 0.25,
    epsilon: float = 1e-6,
    min_stratum_size: int = 3,
) -> torch.Tensor:
    """全 batch 作为单一 group 计算分层 advantage（向后兼容接口）。"""
    if scenario_types is None:
        scenario_types = ["single_step"] * len(rewards)

    if len(rewards) != len(levels):
        raise ValueError(
            f"length mismatch: rewards={len(rewards)}, levels={len(levels)}"
        )
    if len(scenario_types) != len(rewards):
        raise ValueError(
            f"length mismatch: rewards={len(rewards)}, "
            f"scenario_types={len(scenario_types)}"
        )

    return compute_per_group_stratified_advantages(
        group_scores=rewards,
        group_levels=list(levels),
        group_scenarios=list(scenario_types),
        beta=beta,
        epsilon=epsilon,
        min_stratum_size=min_stratum_size,
        norm_by_std=True,
    )


def compute_stratified_advantage(
    rewards: torch.Tensor,
    levels: list[str],
    scenario_types: list[str],
    beta: float = 0.25,
    epsilon: float = 1e-6,
    min_stratum_size: int = 3,
) -> torch.Tensor:
    """compute_livemcp_advantages 的显式别名（要求 scenario_types 必传）。"""
    return compute_livemcp_advantages(
        rewards=rewards,
        levels=levels,
        scenario_types=scenario_types,
        beta=beta,
        epsilon=epsilon,
        min_stratum_size=min_stratum_size,
    )


__all__ = [
    "compute_standard_grpo_advantages",
    "compute_per_group_stratified_advantages",
    "compute_livemcp_advantages",
    "compute_stratified_advantage",
]