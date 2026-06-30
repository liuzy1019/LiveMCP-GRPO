"""Group saturation diagnostics per OVAL-MCP §9.2-9.3.

每个 training step 记录:
  - std(J_i) within group
  - std(C_safety_i) within group
  - all_success_group_rate
  - all_failure_group_rate
  - all_safe_group_rate
  - all_unsafe_group_rate
  - mixed_safety_group_rate
  - unsafe_success_rate

saturated group 处理规则 (§9.3):
  - 不产生 policy gradient (在 estimator 中 skip)
  - 仍参与 lambda_safe 的 hat_C_batch 计算
  - lambda stall protection: 连续 K_stall 步 unsafe 主导时冻结 lambda_safe
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GroupSaturation:
    """Per-group saturation state."""

    group_id: str = ""
    n_total: int = 0          # G (group size)
    j_values: list[float] = field(default_factory=list)
    c_values: list[int] = field(default_factory=list)
    r_values: list[float] = field(default_factory=list)

    # computed
    mean_j: float = 0.0
    std_j: float = 0.0
    mean_c: float = 0.0
    std_c: float = 0.0
    is_saturated: bool = False  # std_j < min_group_std

    # classification
    all_success: bool = False     # all R_task > threshold
    all_failure: bool = False
    all_safe: bool = False        # all C_safety = 0
    all_unsafe: bool = False      # all C_safety = 1
    mixed_safety: bool = False    # some safe, some unsafe
    unsafe_success: bool = False  # at least one R>0.5 and C=1
    unsafe_success_count: int = 0


@dataclass
class SaturationSummary:
    """Batch-level saturation diagnostic summary."""

    n_groups: int = 0
    n_saturated: int = 0          # groups skipped due to std_j < threshold
    n_valid_gradient: int = 0     # groups that produced gradient

    all_success_rate: float = 0.0
    all_failure_rate: float = 0.0
    all_safe_rate: float = 0.0
    all_unsafe_rate: float = 0.0
    mixed_safety_rate: float = 0.0
    unsafe_success_rate: float = 0.0

    saturated_group_unsafe_rate: float = 0.0

    # per-group detail
    groups: list[GroupSaturation] = field(default_factory=list)


class SaturationDiagnostics:
    """Compute per-group and batch-level saturation diagnostics."""

    def __init__(
        self,
        min_group_std: float = 1e-6,
        success_threshold: float = 0.5,
        k_stall: int = 10,
        tau_unsafe_stall: float = 0.5,
    ):
        self.min_group_std = min_group_std
        self.success_threshold = success_threshold
        self.k_stall = k_stall
        self.tau_unsafe_stall = tau_unsafe_stall
        self._lambda_increase_streak: int = 0

    def diagnose_groups(
        self,
        groups: dict[str, list[tuple[float, int, float]]],
        # groups: {group_id: [(r_task, c_safety, j), ...]}
    ) -> SaturationSummary:
        """Compute per-group saturation state and batch summary."""
        summary = SaturationSummary()
        summary.n_groups = len(groups)

        n_success_groups = 0
        n_failure_groups = 0
        n_safe_groups = 0
        n_unsafe_groups = 0
        n_mixed_safety = 0
        n_unsafe_success = 0  # count of groups with any unsafe success
        n_saturated = 0
        saturated_unsafe_count = 0
        saturated_total = 0

        for gid, values in groups.items():
            gs = self._diagnose_group(gid, values)
            summary.groups.append(gs)
            saturated_total += gs.n_total
            if gs.is_saturated:
                n_saturated += 1
                saturated_unsafe_count += sum(gs.c_values)
            else:
                summary.n_valid_gradient += 1

            if gs.all_success:
                n_success_groups += 1
            if gs.all_failure:
                n_failure_groups += 1
            if gs.all_safe:
                n_safe_groups += 1
            if gs.all_unsafe:
                n_unsafe_groups += 1
            if gs.mixed_safety:
                n_mixed_safety += 1
            if gs.unsafe_success:
                n_unsafe_success += 1

        ng = max(len(groups), 1)
        summary.n_saturated = n_saturated
        summary.all_success_rate = n_success_groups / ng
        summary.all_failure_rate = n_failure_groups / ng
        summary.all_safe_rate = n_safe_groups / ng
        summary.all_unsafe_rate = n_unsafe_groups / ng
        summary.mixed_safety_rate = n_mixed_safety / ng
        summary.unsafe_success_rate = n_unsafe_success / ng
        summary.saturated_group_unsafe_rate = (
            saturated_unsafe_count / max(saturated_total, 1)
        )

        return summary

    def _diagnose_group(
        self,
        group_id: str,
        values: list[tuple[float, int, float]],
    ) -> GroupSaturation:
        gs = GroupSaturation(group_id=group_id, n_total=len(values))
        if not values:
            gs.is_saturated = True
            return gs

        r_vals = [v[0] for v in values]
        c_vals = [v[1] for v in values]
        j_vals = [v[2] for v in values]
        gs.r_values = r_vals
        gs.c_values = c_vals
        gs.j_values = j_vals

        import math
        gs.mean_j = sum(j_vals) / len(j_vals)
        gs.mean_c = sum(c_vals) / len(c_vals)

        if len(j_vals) >= 2:
            var_j = sum((j - gs.mean_j) ** 2 for j in j_vals) / len(j_vals)
            gs.std_j = math.sqrt(var_j)
            var_c = sum((c - gs.mean_c) ** 2 for c in c_vals) / len(c_vals)
            gs.std_c = math.sqrt(var_c)

        gs.is_saturated = gs.std_j < self.min_group_std

        gs.all_success = all(r > self.success_threshold for r in r_vals)
        gs.all_failure = all(r <= self.success_threshold for r in r_vals)
        gs.all_safe = all(c == 0 for c in c_vals)
        gs.all_unsafe = all(c == 1 for c in c_vals)
        gs.mixed_safety = not gs.all_safe and not gs.all_unsafe
        gs.unsafe_success = any(
            r > self.success_threshold and c == 1
            for r, c in zip(r_vals, c_vals)
        )
        gs.unsafe_success_count = sum(
            1 for r, c in zip(r_vals, c_vals)
            if r > self.success_threshold and c == 1
        )

        return gs

    def check_lambda_stall(
        self,
        lambda_increased: bool,
        summary: SaturationSummary,
    ) -> bool:
        """[Legacy] 检查 lambda_safe 是否应冻结。

        注意：此方法仅用于测试/离线分析。
        训练生产代码应使用 LambdaState.update()（含原子文件锁和持久化）。
        两套实现独立维护 stall streak，同时使用会导致状态发散。
        """
        all_unsafe_mask = summary.all_unsafe_rate > self.tau_unsafe_stall

        if lambda_increased and all_unsafe_mask:
            self._lambda_increase_streak += 1
        else:
            self._lambda_increase_streak = max(0, self._lambda_increase_streak - 1)

        return self._lambda_increase_streak >= self.k_stall

    def reset_stall_counter(self) -> None:
        """Reset lambda increase streak (e.g. after data sampling adjustment)."""
        self._lambda_increase_streak = 0


__all__ = [
    "GroupSaturation",
    "SaturationDiagnostics",
    "SaturationSummary",
]
