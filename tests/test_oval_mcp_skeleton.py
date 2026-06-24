"""Unit tests for F_gamma, P_process, saturation, and LambdaState."""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.oval_mcp.verifier.events import EventLog, AuditEvent
from src.oval_mcp.rewards.f_gamma import ProgressTracker, FGammaResult
from src.oval_mcp.rewards.p_process import ProcessScorer, ProcessScoreResult
from src.oval_mcp.training.saturation import SaturationDiagnostics


# ═══════════════════════════════════════════════════════════════════════
# F_gamma tests
# ═══════════════════════════════════════════════════════════════════════

def test_f_gamma_empty_log():
    tracker = ProgressTracker()
    log = EventLog(session_id="s1", task_id="t1")
    result = tracker.compute(log, task={})
    assert result.f_gamma == 0.0
    assert result.phi_initial == 0.0
    assert result.phi_final == 0.0
    print("  PASS test_f_gamma_empty_log")


def test_f_gamma_full_progress():
    """全部 predicate 完成 → F_gamma ≈ 1.0（gamma=1）"""
    tracker = ProgressTracker()
    log = EventLog(session_id="s1", task_id="t1")
    # resolved + transition
    log.append(AuditEvent(event_id="e1", session_id="s1", step=1,
        action_type="tool_call", operation="query", target_type="calendar_event",
        execution_success=True, schema_valid=True, state_changed=True))
    # transition completed
    log.append(AuditEvent(event_id="e2", session_id="s1", step=2,
        action_type="tool_call", operation="update", target_type="calendar_event",
        execution_success=True, state_changed=True))
    # terminal → verified + produced
    log.append(AuditEvent(event_id="e3", session_id="s1", step=3,
        action_type="final_answer", operation="terminal"))
    result = tracker.compute(log, task={}, gamma=1.0)
    assert result.f_gamma > 0.0, f"expected F>0, got {result.f_gamma}"
    assert result.phi_final > 0.0
    print(f"  PASS test_f_gamma_full_progress: F_gamma={result.f_gamma:.3f}, phi_final={result.phi_final:.3f}")


def test_f_gamma_gamma_lt_1():
    """gamma < 1 时早期 progress 权重更大"""
    tracker = ProgressTracker()
    log = EventLog(session_id="s1", task_id="t1")
    log.append(AuditEvent(event_id="e1", session_id="s1", step=1,
        action_type="tool_call", operation="query", target_type="calendar_event",
        execution_success=True, state_changed=True))
    log.append(AuditEvent(event_id="e2", session_id="s1", step=2,
        action_type="tool_call", operation="update", target_type="calendar_event",
        execution_success=True, state_changed=True))
    log.append(AuditEvent(event_id="e3", session_id="s1", step=3,
        action_type="final_answer", operation="terminal"))

    r1 = tracker.compute(log, task={}, gamma=1.0)
    r09 = tracker.compute(log, task={}, gamma=0.9)
    # gamma < 1 的 F_gamma 应该 <= gamma=1 的 F_gamma（折扣后更小）
    assert r09.f_gamma <= r1.f_gamma, \
        f"gamma=0.9: {r09.f_gamma}, gamma=1.0: {r1.f_gamma}"
    print(f"  PASS test_f_gamma_gamma_lt_1: F(γ=0.9)={r09.f_gamma:.3f}, F(γ=1.0)={r1.f_gamma:.3f}")


def test_f_gamma_no_progress():
    """无执行成功事件 → F_gamma = 0"""
    tracker = ProgressTracker()
    log = EventLog(session_id="s1", task_id="t1")
    log.append(AuditEvent(event_id="e1", session_id="s1", step=1,
        action_type="tool_call", operation="query", target_type="calendar_event",
        execution_success=False))
    result = tracker.compute(log, task={}, gamma=1.0)
    assert result.f_gamma == 0.0
    print("  PASS test_f_gamma_no_progress")


# ═══════════════════════════════════════════════════════════════════════
# P_process tests
# ═══════════════════════════════════════════════════════════════════════

def test_p_process_empty():
    scorer = ProcessScorer()
    log = EventLog(session_id="s1", task_id="t1")
    result = scorer.compute(log)
    assert result.p_process == 0.0
    assert result.n_steps == 0
    print("  PASS test_p_process_empty")


def test_p_process_clean_success():
    """全部成功 → P_process > 0"""
    scorer = ProcessScorer()
    log = EventLog(session_id="s1", task_id="t1")
    log.append(AuditEvent(event_id="e1", session_id="s1", step=1,
        action_type="tool_call", operation="query",
        execution_success=True, schema_valid=True, state_changed=True))
    log.append(AuditEvent(event_id="e2", session_id="s1", step=2,
        action_type="tool_call", operation="update",
        execution_success=True, schema_valid=True, state_changed=True))
    result = scorer.compute(log)
    assert result.p_process > 0, f"expected P>0, got {result.p_process}"
    assert result.total_bonus > 0
    assert result.n_forbidden_steps == 0
    print(f"  PASS test_p_process_clean_success: P={result.p_process:.3f}, bonus={result.total_bonus:.3f}")


def test_p_process_forbidden_clamping():
    """forbidden event step 的 p_t 不能为正数，即使 bonus 更大"""
    scorer = ProcessScorer()
    log = EventLog(session_id="s1", task_id="t1")
    # 成功的操作，但同时触发 forbidden_transition
    log.append(AuditEvent(event_id="e1", session_id="s1", step=1,
        action_type="tool_call", operation="delete",
        execution_success=True, schema_valid=True, state_changed=True,
        forbidden_transition="forbidden_transition_delete_recreate"))
    result = scorer.compute(log)
    # 即使 execution_success 触发 bonus，forbidden 应 clamping 到负值
    assert result.per_step[0].p_clamped <= -0.07, \
        f"forbidden step p should be <= -0.08, got {result.per_step[0].p_clamped}"
    assert result.n_forbidden_steps == 1
    print(f"  PASS test_p_process_forbidden_clamping: p={result.per_step[0].p_clamped:.3f}")


def test_p_process_clip_to_range():
    """P_process 应该在 [-p_max, p_max] 内"""
    scorer = ProcessScorer(p_max=0.3)
    log = EventLog(session_id="s1", task_id="t1")
    # 大量成功操作 → P 应该被 clip 到 p_max
    for i in range(10):
        log.append(AuditEvent(event_id=f"e{i}", session_id="s1", step=i+1,
            action_type="tool_call", operation="update",
            execution_success=True, schema_valid=True, state_changed=True))
    result = scorer.compute(log)
    assert result.p_process <= scorer.p_max, \
        f"P should be <= {scorer.p_max}, got {result.p_process}"
    print(f"  PASS test_p_process_clip_to_range: P={result.p_process:.3f} (max={scorer.p_max})")


def test_p_process_schema_invalid_penalty():
    """schema_invalid → 触发 PEN_invalid_tool_schema penalty"""
    scorer = ProcessScorer()
    log = EventLog(session_id="s1", task_id="t1")
    log.append(AuditEvent(event_id="e1", session_id="s1", step=1,
        action_type="tool_call", operation="query",
        execution_success=True, schema_valid=False))  # schema 无效
    result = scorer.compute(log)
    assert result.total_penalty < 0, f"expected penalty, got {result.total_penalty}"
    print(f"  PASS test_p_process_schema_invalid_penalty: penalty={result.total_penalty:.3f}")


# ═══════════════════════════════════════════════════════════════════════
# Saturation tests
# ═══════════════════════════════════════════════════════════════════════

def test_saturation_all_success_all_safe():
    diag = SaturationDiagnostics()
    groups = {
        "g1": [(0.8, 0, 0.8), (0.9, 0, 0.9), (0.7, 0, 0.7)],
    }
    summary = diag.diagnose_groups(groups)
    assert summary.all_success_rate == 1.0
    assert summary.all_safe_rate == 1.0
    assert summary.n_saturated == 0
    print("  PASS test_saturation_all_success_all_safe")


def test_saturation_mixed_safety():
    diag = SaturationDiagnostics()
    groups = {
        "g1": [(0.8, 0, 0.8), (0.7, 1, -0.3), (0.9, 0, 0.9)],
    }
    summary = diag.diagnose_groups(groups)
    assert summary.mixed_safety_rate == 1.0
    assert summary.unsafe_success_rate == 1.0
    print("  PASS test_saturation_mixed_safety")


def test_saturation_within_group_std():
    """组内 J 无方差 → saturated"""
    diag = SaturationDiagnostics(min_group_std=1e-6)
    groups = {
        "g1": [(0.5, 0, 0.5), (0.5, 0, 0.5), (0.5, 0, 0.5)],
    }
    summary = diag.diagnose_groups(groups)
    assert summary.n_saturated == 1
    assert summary.groups[0].is_saturated
    print("  PASS test_saturation_within_group_std")


def test_saturation_lambda_stall_check():
    """lambda 连续增大且 unsafe 主导 → stall protection 触发"""
    diag = SaturationDiagnostics(k_stall=3, tau_unsafe_stall=0.5)
    # 模拟 lambda 一直在增大 + 大部分 group unsafe
    triggered = False
    for _ in range(5):
        summary = diag.diagnose_groups({
            f"g{i}": [(0.3, 1, -0.7)] * 3 for i in range(3)
        })
        if diag.check_lambda_stall(lambda_increased=True, summary=summary):
            triggered = True
            break
    assert triggered, "stall protection 应该在第 3 步后触发"
    print("  PASS test_saturation_lambda_stall_check")


# ═══════════════════════════════════════════════════════════════════════
# LambdaState tests (incl. stall protection)
# ═══════════════════════════════════════════════════════════════════════

def test_lambda_state_basic():
    from src.oval_mcp.training.lambda_state import LambdaState

    tmp_path = "/tmp/test_ssgrpo_lambda.json"
    LambdaState.reset(tmp_path)

    state = LambdaState.load_or_default(path=tmp_path)
    assert state.lambda_safe == 1.0
    assert state.step == 0

    # 无违规 → lambda 减小; api 返回 (new_lambda, skipped=False)
    new_l, skipped = state.update([0, 0, 0, 0])
    assert not skipped
    assert new_l < 1.0, f"lambda should decrease, got {new_l}"
    assert state.lambda_safe == new_l
    state.save()

    # 重新加载
    state2 = LambdaState.load_or_default(path=tmp_path)
    assert abs(state2.lambda_safe - state.lambda_safe) < 1e-10

    # 有违规 → lambda 增大
    new_l2, skipped2 = state2.update([1, 1, 1])
    assert not skipped2
    assert new_l2 > 1.0, f"lambda should increase, got {new_l2}"

    LambdaState.reset(tmp_path)
    print(f"  PASS test_lambda_state_basic: final lambda={state2.lambda_safe:.4f}")


def test_lambda_state_clip():
    """lambda_safe 不应超出 [0, lambda_safe_max]"""
    from src.oval_mcp.training.lambda_state import LambdaState

    tmp_path = "/tmp/test_ssgrpo_lambda2.json"
    LambdaState.reset(tmp_path)
    state = LambdaState.load_or_default(path=tmp_path, lambda_safe_max=5.0)

    # 持续违规 → lambda 单调增
    for _ in range(50):
        state.update([1] * 10)
    assert state.lambda_safe >= 1.0
    assert state.lambda_safe <= state.lambda_safe_max

    # 长期零违规 → lambda 应下降
    state2 = LambdaState.load_or_default(path=tmp_path, lambda_safe_max=5.0)
    for _ in range(200):
        state2.update([0] * 10)
    assert state2.lambda_safe >= 0.0
    assert state2.lambda_safe < 1.0

    LambdaState.reset(tmp_path)
    print(f"  PASS test_lambda_state_clip: 增后={state.lambda_safe:.4f}, 降后={state2.lambda_safe:.4f}")


def test_lambda_state_stall_protection():
    """stall protection: 连续 unsafe 主导时冻结 lambda 增大"""
    from src.oval_mcp.training.lambda_state import LambdaState

    tmp_path = "/tmp/test_ssgrpo_stall.json"
    LambdaState.reset(tmp_path)
    state = LambdaState.load_or_default(path=tmp_path)

    # 初始 λ=1.0, 连续 batch 全是 unsafe (high hat_C)
    for step_idx in range(8):
        new_l, skipped = state.update(
            [1] * 10,
            k_stall=5,
            tau_unsafe_stall=0.5,
        )

    # hat_C=1.0 >> 0.5 → unsafe dominant, streak≥5 → should be frozen
    assert state.is_stall_frozen, f"step 8 should be frozen"
    assert state.stall_streak >= 5

    # Frozen: increase should be skipped
    old_l = state.lambda_safe
    new_l, skipped = state.update(
        [1] * 10,
        k_stall=5,
        tau_unsafe_stall=0.5,
    )
    assert skipped, "frozen state should skip increase"
    assert new_l == old_l, "frozen lambda should not change on increase"

    # Decrease should still work even when frozen
    new_l2, skipped2 = state.update(
        [0] * 10,
        k_stall=5,
        tau_unsafe_stall=0.5,
    )
    assert not skipped2, "decrease should not be skipped"
    assert new_l2 < old_l, f"lambda should decrease: {old_l} → {new_l2}"

    # After enough safe batches, should unfreeze
    for _ in range(4):
        state.update([0] * 10, k_stall=5, tau_unsafe_stall=0.5)
    assert not state.is_stall_frozen, "should unfreeze after safe batches"

    LambdaState.reset(tmp_path)
    print(f"  PASS test_lambda_state_stall_protection: frozen_lambda={old_l:.4f}")


def test_lambda_state_stall_persists():
    """stall state persists across save/load (跨进程重启不丢失)"""
    from src.oval_mcp.training.lambda_state import LambdaState

    tmp_path = "/tmp/test_ssgrpo_stall2.json"
    LambdaState.reset(tmp_path)
    state = LambdaState.load_or_default(path=tmp_path)

    # trigger freeze
    for _ in range(6):
        state.update([1] * 10, k_stall=5, tau_unsafe_stall=0.5)
    assert state.is_stall_frozen
    state.save()

    # simulate process restart
    state2 = LambdaState.load_or_default(path=tmp_path)
    assert state2.is_stall_frozen, "stall should persist across load"
    assert state2.stall_streak == state.stall_streak

    LambdaState.reset(tmp_path)
    print(f"  PASS test_lambda_state_stall_persists")


# ═══════════════════════════════════════════════════════════════════════
# LATA tests
# ═══════════════════════════════════════════════════════════════════════

def test_lata_none_mode():
    """none 模式: a_{i,t} = A_i（与 standard GRPO 一致）"""
    from src.oval_mcp.training.lata import LATAAllocator, LATAConfig
    import torch as _t

    allocator = LATAAllocator(LATAConfig(mode="none"))
    advantages = _t.tensor([0.5, -0.3, 0.8])
    response_mask = _t.tensor([
        [1, 1, 0, 0],
        [1, 1, 1, 1],
        [1, 0, 0, 0],
    ])
    result = allocator.allocate_from_mask(advantages, response_mask)

    assert result.mode == "none"
    assert result.response_lengths == [2, 4, 1]
    assert abs(result.token_advantages[0, 0].item() - 0.5) < 1e-5
    assert abs(result.token_advantages[0, 1].item() - 0.5) < 1e-5
    assert result.token_advantages[0, 2].item() == 0.0
    assert abs(result.token_advantages[1, 0].item() - (-0.3)) < 1e-5
    print("  PASS test_lata_none_mode")


def test_lata_sqrt_l_mode():
    """sqrt_l 模式: a_{i,t} = A_i / sqrt(L_i), 长回复 per-token advantage 更小"""
    from src.oval_mcp.training.lata import LATAAllocator, LATAConfig

    allocator = LATAAllocator(LATAConfig(mode="sqrt_l"))
    advantages = torch.tensor([0.5, 0.5])  # same trajectory advantage
    response_mask = torch.tensor([
        [1, 1, 0, 0],   # L=2 → scale=1/sqrt(2)=0.707
        [1, 1, 1, 1],   # L=4 → scale=1/sqrt(4)=0.500
    ])
    result = allocator.allocate_from_mask(advantages, response_mask)

    # shorter response (L=2) gets higher per-token advantage
    short_token_adv = result.token_advantages[0, 0].item()
    long_token_adv = result.token_advantages[1, 0].item()
    assert short_token_adv > long_token_adv, \
        f"shorter ({short_token_adv:.3f}) should > longer ({long_token_adv:.3f})"
    assert abs(short_token_adv - 0.5 / 1.414) < 0.01
    assert abs(long_token_adv - 0.5 / 2.0) < 0.01
    print(f"  PASS test_lata_sqrt_l_mode: short={short_token_adv:.3f}, long={long_token_adv:.3f}")


def test_lata_norm_mode():
    """norm 模式: a_{i,t} = A_i * sqrt(L_ref / L_i), batch 归一化"""
    from src.oval_mcp.training.lata import LATAAllocator, LATAConfig

    allocator = LATAAllocator(LATAConfig(mode="norm"))
    advantages = torch.tensor([1.0, 1.0])
    response_mask = torch.tensor([
        [1, 0, 0, 0],   # L=1 → L_ref=2.5, scale=sqrt(2.5/1)=1.581
        [1, 1, 1, 1],   # L=4 → scale=sqrt(2.5/4)=0.791
    ])
    result = allocator.allocate_from_mask(advantages, response_mask)

    short_adv = result.token_advantages[0, 0].item()
    long_adv = result.token_advantages[1, 0].item()
    assert short_adv > long_adv, \
        f"norm should amplify short: short={short_adv:.3f}, long={long_adv:.3f}"
    # L_ref = (1+4)/2 = 2.5
    assert abs(short_adv - 1.581) < 0.005
    assert abs(long_adv - 0.791) < 0.005
    print(f"  PASS test_lata_norm_mode: short={short_adv:.3f}, long={long_adv:.3f}")


def test_lata_zero_mask():
    """空 mask 不应报错"""
    from src.oval_mcp.training.lata import LATAAllocator, LATAConfig

    allocator = LATAAllocator(LATAConfig(mode="sqrt_l"))
    advantages = torch.tensor([0.5])
    response_mask = torch.tensor([[0, 0, 0]])  # all zeros → min_length=1
    result = allocator.allocate_from_mask(advantages, response_mask)
    assert result.token_advantages.shape == (1, 3)
    assert result.response_lengths == [0]  # clamped to 1
    print("  PASS test_lata_zero_mask")


if __name__ == "__main__":
    print("OVAL-MCP 骨架模块测试")
    print("=" * 40)
    tests = [
        test_f_gamma_empty_log,
        test_f_gamma_full_progress,
        test_f_gamma_gamma_lt_1,
        test_f_gamma_no_progress,
        test_p_process_empty,
        test_p_process_clean_success,
        test_p_process_forbidden_clamping,
        test_p_process_clip_to_range,
        test_p_process_schema_invalid_penalty,
        test_saturation_all_success_all_safe,
        test_saturation_mixed_safety,
        test_saturation_within_group_std,
        test_saturation_lambda_stall_check,
        test_lambda_state_basic,
        test_lambda_state_clip,
        test_lambda_state_stall_protection,
        test_lambda_state_stall_persists,
        test_lata_none_mode,
        test_lata_sqrt_l_mode,
        test_lata_norm_mode,
        test_lata_zero_mask,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"  FAIL {t.__name__}: {e}")
    print(f"\n{passed}/{len(tests)} passed")
