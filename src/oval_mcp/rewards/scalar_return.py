"""Scalar return: J = R_task - lambda_safe * C_safety + optional terms.

OVAL-MCP §9 Phase 1:
  J_i = R_task(tau_i) - lambda_safe * C_safety(tau_i)
  A_i = (J_i - mean(J_1...J_G)) / (std(J_1...J_G) + eps)

Phase 1: I_process = 0, I_shape = 0.

F_gamma 与 R_coverage 的关系（Phase 2 消融必须验证）：
  - γ=1 时 F_gamma = Phi(m_T) = completed_required_states / total_required_states
  - required_states ⊆ coverage_predicates，因此 F_gamma 与 R_coverage 共享底层信息
  - M4+F vs M4 的提升来源：
    * 若 partial_R2(F_gamma | R_coverage) < 0.05：F_gamma 的作用是放大 progress
      信号在 J 中的相对权重（progress emphasis），而非提供新信号源。
      效果可通过调大 R_coverage 权重近似复现。
    * 若 partial_R2 >= 0.05：F_gamma 提供了 R_coverage 之外的独特信息，
      提升可归因于 shaping 的信用分配改善。
  - 默认解释：γ=1 时 F_gamma 提供 progress emphasis，不是独立信用分配。

P_process 的作用边界（Phase 2 消融必须说明）：
  - trajectory-level P_process（Phase 2）在典型情况下对组内 J 方差贡献 < 1%
  - 只在 R_task 完全饱和时（组内所有 tau 的 R_task 相同）才会产生可量化的效果
  - 作用：反饱和（防止 J 方差为零时 gradient signal 消失）
  - 不提供 step-level 信用分配改善——信用分配改善等到 Phase 3 LATA
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ScalarReturnResult:
    """Scalarized return for a single trajectory."""

    j: float = 0.0  # scalar return J_i
    r_task: float = 0.0
    c_safety: int = 0
    lambda_safe: float = 1.0
    f_gamma: float = 0.0
    p_process: float = 0.0

    # Diagnostic info
    task_id: str = ""
    session_id: str = ""

    def to_dict(self) -> dict[str, float]:
        return {
            "j": self.j,
            "r_task": self.r_task,
            "c_safety": float(self.c_safety),
            "lambda_safe": self.lambda_safe,
            "f_gamma": self.f_gamma,
            "p_process": self.p_process,
        }


class ScalarReturn:
    """Compute scalar return J_i for Phase 1 constrained GRPO.

    Phase 1:
      I_process = 0
      I_shape = 0
      J_i = R_task(tau_i) - lambda_safe * C_safety(tau_i)
    """

    def __init__(
        self,
        lambda_safe: float = 1.0,
        i_shape: int = 0,
        i_process: int = 0,
        lambda_shape: float = 0.0,
        lambda_process: float = 0.0,
        min_group_std: float = 1e-6,
    ):
        self.lambda_safe = lambda_safe
        self.i_shape = i_shape
        self.i_process = i_process
        self.lambda_shape = lambda_shape
        self.lambda_process = lambda_process
        self.min_group_std = min_group_std

    @classmethod
    def phase1_default(cls) -> "ScalarReturn":
        """Phase 1 default: only R_task - lambda_safe * C_safety."""
        return cls(
            lambda_safe=1.0,
            i_shape=0,
            i_process=0,
            lambda_shape=0.0,
            lambda_process=0.0,
        )

    def compute_single(
        self,
        r_task: float,
        c_safety: int,
        f_gamma: float = 0.0,
        p_process: float = 0.0,
        task_id: str = "",
        session_id: str = "",
    ) -> ScalarReturnResult:
        """Compute J_i for a single trajectory."""
        j = (
            r_task
            + self.i_shape * self.lambda_shape * f_gamma
            + self.i_process * self.lambda_process * p_process
            - self.lambda_safe * c_safety
        )
        return ScalarReturnResult(
            j=j,
            r_task=r_task,
            c_safety=c_safety,
            lambda_safe=self.lambda_safe,
            f_gamma=f_gamma,
            p_process=p_process,
            task_id=task_id,
            session_id=session_id,
        )

    def compute_group_advantages(
        self,
        j_values: list[float],
    ) -> tuple[list[float], bool]:
        """Compute group-relative advantages A_i from J_i values.

        A_i = (J_i - mean(J)) / (std(J) + eps)

        Returns:
          advantages: list of A_i for each trajectory
          saturated: True if std(J) < min_group_std (no gradient produced)
        """
        if not j_values:
            return [], True

        import math

        mean_j = sum(j_values) / len(j_values)
        variance = sum((j - mean_j) ** 2 for j in j_values) / len(j_values)
        std_j = math.sqrt(variance)

        if std_j < self.min_group_std:
            return [0.0] * len(j_values), True

        advantages = [(j - mean_j) / std_j for j in j_values]
        return advantages, False

    def update_lambda_safe(
        self,
        c_safety_values: list[int],
        alpha_lambda: float = 0.01,
        epsilon: float = 0.05,
        lambda_safe_max: float = 10.0,
    ) -> float:
        """Lagrangian multiplier update.

        hat_C_batch = mean(C_safety over batch)
        lambda_safe = clip(lambda_safe + alpha_lambda * (hat_C_batch - epsilon), 0, lambda_safe_max)
        """
        if not c_safety_values:
            return self.lambda_safe

        hat_c_batch = sum(c_safety_values) / len(c_safety_values)
        new_lambda = self.lambda_safe + alpha_lambda * (hat_c_batch - epsilon)
        self.lambda_safe = max(0.0, min(new_lambda, lambda_safe_max))
        return self.lambda_safe


__all__ = ["ScalarReturn", "ScalarReturnResult"]
