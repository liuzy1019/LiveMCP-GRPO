"""
分层 advantage 纯函数 — 单一权威来源（Single Source of Truth）。

供 livemcp_grpo_estimator.py 和训练入口共用的权威实现。
不依赖 verl/Ray/logger，可独立测试。
"""

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


__all__ = [
    "compute_standard_grpo_advantages",
]
