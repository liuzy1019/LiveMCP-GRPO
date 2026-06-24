"""LATA: Length-Aware Token Allocation per OVAL-MCP §9.1.

问题：
  GRPO 的 standard token allocation 是对 trajectory 内所有 token 分配相同的
  trajectory-level advantage A_i： a_{i,t} = A_i，所有 t。
  这导致长回复的 per-token 信号被稀释——即使 J_i 相同，长回复的每个 token
  收到的 gradient 强度弱于短回复。

方案：
  √L normalization：短回复获得更高的 per-token advantage，长回复反之。
  保持 total contribution = A_i * sqrt(L_i)，而非 A_i * L_i。

三种模式：
  "none":    a_{i,t} = A_i                      （standard GRPO）
  "sqrt_l":  a_{i,t} = A_i / sqrt(L_i)          （推荐默认）
  "norm":    a_{i,t} = A_i * sqrt(L_ref / L_i)  （batch 归一化，L_ref = mean(L)）

Phase 1 推荐 "none"（保持 baseline 可比），Phase 3 切换到 "sqrt_l"。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch


@dataclass
class LATAConfig:
    mode: str = "none"  # none | sqrt_l | norm
    min_length: int = 1  # floor for L_i to avoid division by zero


@dataclass
class LATAResult:
    """Per-token advantage allocation."""

    token_advantages: torch.Tensor | np.ndarray  # [bsz, max_len] or [bsz]
    trajectory_advantages: torch.Tensor | np.ndarray  # [bsz]
    response_lengths: list[int]  # per-trajectory
    mode: str = "none"
    mean_length: float = 0.0
    per_token_scale: list[float] = field(default_factory=list)  # 1/sqrt(L_i) per trajectory


class LATAAllocator:
    """Allocate trajectory-level advantages across tokens with length normalization."""

    def __init__(self, config: Optional[LATAConfig] = None):
        self.config = config or LATAConfig()

    def allocate_from_mask(
        self,
        advantages: torch.Tensor,
        response_mask: torch.Tensor,
    ) -> LATAResult:
        """Allocate advantages across tokens given response_mask.

        Args:
          advantages: [bsz] trajectory-level advantages
          response_mask: [bsz, max_len] boolean/int mask (1=response token, 0=padding/tool)

        Returns:
          LATAResult with token_advantages [bsz, max_len] matching response_mask shape.
        """
        bsz = advantages.shape[0]
        device = advantages.device

        # compute per-trajectory response lengths from mask
        if isinstance(response_mask, torch.Tensor):
            lengths = response_mask.sum(dim=-1).long().cpu().tolist()
        else:
            lengths = [int(np.sum(m)) for m in response_mask]

        lengths_clamped = [max(l, self.config.min_length) for l in lengths]
        L_ref = float(np.mean(lengths_clamped)) if lengths_clamped else 1.0

        if self.config.mode == "none":
            # standard: repeat A_i across all response tokens
            token_adv = advantages.unsqueeze(-1) * response_mask.float()
            per_token_scale = [1.0] * bsz

        elif self.config.mode == "sqrt_l":
            # a_{i,t} = A_i / sqrt(L_i)  for each token in response i
            scale = torch.tensor(
                [1.0 / math.sqrt(L) for L in lengths_clamped],
                dtype=advantages.dtype, device=device,
            )
            per_token_scale = scale.cpu().tolist()
            token_adv = (advantages * scale).unsqueeze(-1) * response_mask.float()

        elif self.config.mode == "norm":
            # a_{i,t} = A_i * sqrt(L_ref / L_i)
            scale = torch.tensor(
                [math.sqrt(L_ref / L) for L in lengths_clamped],
                dtype=advantages.dtype, device=device,
            )
            per_token_scale = scale.cpu().tolist()
            token_adv = (advantages * scale).unsqueeze(-1) * response_mask.float()

        else:
            raise ValueError(f"unknown LATA mode: {self.config.mode}")

        return LATAResult(
            token_advantages=token_adv,
            trajectory_advantages=advantages,
            response_lengths=lengths,
            mode=self.config.mode,
            mean_length=L_ref,
            per_token_scale=per_token_scale,
        )

    def allocate_from_lengths(
        self,
        advantages: torch.Tensor,
        response_lengths: List[int],
    ) -> LATAResult:
        """Allocate advantages given explicit response lengths (no mask).

        Use when you only have trajectory-level advantages and lengths.
        Returns token_advantages [bsz, max_len] padded to max(response_lengths).
        """
        bsz = advantages.shape[0]
        device = advantages.device
        max_len = max(response_lengths)
        lengths_clamped = [max(l, self.config.min_length) for l in response_lengths]
        L_ref = float(np.mean(lengths_clamped))

        # build mask
        if isinstance(advantages, torch.Tensor):
            idx = torch.arange(max_len, device=device).unsqueeze(0).expand(bsz, -1)
            len_tensor = torch.tensor(response_lengths, device=device).unsqueeze(1)
            mask = (idx < len_tensor).float()
        else:
            mask = np.zeros((bsz, max_len))
            for i, L in enumerate(response_lengths):
                mask[i, :L] = 1.0

        if self.config.mode == "none":
            token_adv = advantages.unsqueeze(-1) * mask
        elif self.config.mode == "sqrt_l":
            scale = torch.tensor(
                [1.0 / math.sqrt(L) for L in lengths_clamped],
                dtype=advantages.dtype, device=device,
            )
            token_adv = (advantages * scale).unsqueeze(-1) * mask
        elif self.config.mode == "norm":
            scale = torch.tensor(
                [math.sqrt(L_ref / L) for L in lengths_clamped],
                dtype=advantages.dtype, device=device,
            )
            token_adv = (advantages * scale).unsqueeze(-1) * mask

        return LATAResult(
            token_advantages=token_adv,
            trajectory_advantages=advantages,
            response_lengths=response_lengths,
            mode=self.config.mode,
            mean_length=L_ref,
        )


__all__ = ["LATAAllocator", "LATAConfig", "LATAResult"]
