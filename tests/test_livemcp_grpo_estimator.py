from __future__ import annotations

import numpy as np
import torch
from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage

from src.training.grpo_estimator import (
    _action_signature,
    _resolve_lata_mode,
    compute_livemcp_grpo_advantage,
)


def _metadata() -> dict[str, np.ndarray]:
    return {
        "group_id": np.asarray(["task-a", "task-a"], dtype=object),
        "perturbation_level": np.asarray(["none", "none"], dtype=object),
        "scenario_type": np.asarray(["standard", "standard"], dtype=object),
        "audit_events": np.asarray(
            [
                [{"action_type": "tool_call", "tool_name": "read", "tool_arguments": {"id": 1}}],
                [{"action_type": "tool_call", "tool_name": "write", "tool_arguments": {"id": 1}}],
            ],
            dtype=object,
        ),
    }


def test_saturation_uses_raw_j_not_post_kl_rewards() -> None:
    token_rewards = torch.tensor([[0.0, 0.0], [0.0, 0.5]])
    response_mask = torch.ones_like(token_rewards)

    advantages, returns = compute_livemcp_grpo_advantage(
        token_level_rewards=token_rewards,
        response_mask=response_mask,
        index=np.asarray(["task-a", "task-a"], dtype=object),
        config={"min_group_std": 1e-6, "lata_mode": "none"},
        raw_j=torch.tensor([0.25, 0.25]),
        non_tensor_batch=_metadata(),
    )

    assert torch.count_nonzero(advantages) == 0
    assert torch.equal(advantages, returns)


def test_standard_grpo_exactly_saturated_group_has_zero_advantage() -> None:
    token_rewards = torch.zeros((16, 2), dtype=torch.float32)
    token_rewards[:, -1] = 1.175
    response_mask = torch.ones_like(token_rewards)

    advantages, returns = compute_grpo_outcome_advantage(
        token_level_rewards=token_rewards,
        response_mask=response_mask,
        index=np.asarray(["task-a"] * 16, dtype=object),
    )

    assert torch.count_nonzero(advantages) == 0
    assert torch.equal(advantages, returns)


def test_unsaturated_raw_j_produces_group_advantages() -> None:
    token_rewards = torch.tensor([[0.0, 0.0], [0.0, 1.0]])
    response_mask = torch.ones_like(token_rewards)

    advantages, _ = compute_livemcp_grpo_advantage(
        token_level_rewards=token_rewards,
        response_mask=response_mask,
        index=np.asarray(["task-a", "task-a"], dtype=object),
        config={"min_group_std": 1e-6, "lata_mode": "none"},
        raw_j=torch.tensor([0.0, 1.0]),
        non_tensor_batch=_metadata(),
    )

    assert advantages[0, 0] < 0
    assert advantages[1, 0] > 0


def test_action_signature_is_canonical() -> None:
    left = [{"action_type": "tool_call", "tool_name": "read", "tool_arguments": {"b": 2, "a": 1}}]
    right = [{"tool_arguments": {"a": 1, "b": 2}, "tool_name": "read", "action_type": "tool_call"}]
    assert _action_signature(left) == _action_signature(right)


def test_nonempty_hydra_config_without_lata_uses_runtime_setting(
    monkeypatch,
) -> None:
    monkeypatch.setenv("LIVEMCP_LATA", "norm")

    assert _resolve_lata_mode({"norm_adv_by_std_in_grpo": True}) == "norm"


def test_explicit_hydra_lata_setting_remains_authoritative(
    monkeypatch,
) -> None:
    monkeypatch.setenv("LIVEMCP_LATA", "norm")

    assert _resolve_lata_mode({"lata_mode": "none"}) == "none"
    assert _resolve_lata_mode({"lata": "sqrt_l"}) == "sqrt_l"


def test_runtime_lata_is_applied_when_hydra_config_has_no_lata_key(
    monkeypatch,
) -> None:
    monkeypatch.setenv("LIVEMCP_LATA", "norm")
    token_rewards = torch.tensor([[0.0, 0.0], [0.0, 1.0]])
    response_mask = torch.tensor([[1.0, 0.0], [1.0, 1.0]])

    advantages, _ = compute_livemcp_grpo_advantage(
        token_level_rewards=token_rewards,
        response_mask=response_mask,
        index=np.asarray(["task-a", "task-a"], dtype=object),
        config={"min_group_std": 1e-6, "norm_adv_by_std_in_grpo": True},
        raw_j=torch.tensor([0.0, 1.0]),
        non_tensor_batch=_metadata(),
    )

    assert torch.allclose(
        advantages,
        torch.tensor(
            [
                [-np.sqrt(1.5), 0.0],
                [np.sqrt(0.75), np.sqrt(0.75)],
            ],
            dtype=torch.float32,
        ),
        atol=1e-6,
    )
