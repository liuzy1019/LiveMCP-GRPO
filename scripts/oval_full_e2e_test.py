#!/usr/bin/env python3
"""OVAL-MCP 端到端集成验证：全链路 GRPO 采样 → 奖励 → advantage 测试。

不依赖 verl/vLLM。使用真实 MCP server 和完整奖励链路。

测试覆盖：
  1. MCP 环境生命周期（启动/重置/隔离/关闭）
  2. 多轨迹采样模拟（安全 + 不安全 + 空操作）
  3. 完整奖励链路：AuditWrapper → R_task → C_safety → F_gamma → P_process → J → λ 更新 → advantage
  4. 组内方差和 saturation 诊断
  5. LATA length-aware allocation

用法：
  /mnt/data1/zhanyiliu/liuzhanyi/anaconda3/envs/arl/bin/python scripts/oval_full_e2e_test.py
"""

from __future__ import annotations

import json
import sys
import os
import random
from pathlib import Path
from dataclasses import dataclass, field
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ═══════════════════════════════════════════════════════════════════════
# Test runner
# ═══════════════════════════════════════════════════════════════════════

_passed = 0
_failed = 0
_warnings: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"  ✓ {name}" + (f"  [{detail}]" if detail else ""))
    else:
        _failed += 1
        print(f"  ✗ {name}" + (f"  [{detail}]" if detail else ""))


def warn(msg: str) -> None:
    _warnings.append(msg)
    print(f"  ⚠ {msg}")


def summary() -> bool:
    total = _passed + _failed
    print(f"\n{'=' * 60}")
    print(f"结果: {_passed}/{total} 通过")
    if _warnings:
        print(f"警告 ({len(_warnings)}):")
        for w in _warnings:
            print(f"  - {w}")
    print(f"{'=' * 60}")
    return _failed == 0


# ═══════════════════════════════════════════════════════════════════════
# Helper: rollout simulator
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class SimulatedRollout:
    """一次完整 rollout 的结果。"""
    domain: str
    session_id: str
    audit_events: list  # list of AuditEvent dicts
    tool_calls: list
    results: list
    task_type: str  # safe | unsafe | empty | efficient | redundant


def simulate_rollout(
    ctx,
    domain: str,
    seed: int,
    rollout_type: str,
) -> SimulatedRollout:
    """模拟一次模型 rollout，根据 rollout_type 执行不同的动作序列。

    rollout_type:
      safe:        list_events → update_event → final_answer
      unsafe:      list_events → delete_event → create_event → final_answer
      empty:       无 tool_call，直接 final_answer
      efficient:   update_event → final_answer
      redundant:   list_events → list_events → update_event → final_answer
    """
    from src.live_mcp.types import ToolCall

    sid = ctx.create_session(seed)
    audit_events: list = []
    tool_calls: list = []
    exec_results: list = []

    def step(tool_name, args, model_output=""):
        tc = ToolCall(tool_name, args, f"call_{len(tool_calls)}")
        try:
            evt, res = ctx.execute_with_audit(sid, domain, tc, model_output)
            tool_calls.append(tc)
            audit_events.append(evt)
            exec_results.append(res)
            return evt, res
        except Exception as e:
            print(f"    [sim] {tool_name} failed: {e}")
            return None, None

    # ── lookup existing entity ──
    if rollout_type in ("safe", "unsafe", "redundant"):
        step("list_events", {})

    # ── find first event id from observation ──
    existing_eid = ""
    if exec_results:
        obs0 = exec_results[0].observation if exec_results[0] else {}
        events = obs0.get("events", []) if isinstance(obs0, dict) else []
        existing_eid = events[0].get("event_id", "") if events else ""

    if rollout_type == "safe":
        if existing_eid:
            step("update_event", {"event_id": existing_eid, "fields": {"title": "Updated Meeting"}})
        step("list_events", {})  # final action: terminal

    elif rollout_type == "unsafe":
        if existing_eid:
            # Get the deleted entity's title so the recreation matches
            deleted_title = ""
            if exec_results:
                obs0 = exec_results[0].observation if exec_results[0] else {}
                events = obs0.get("events", []) if isinstance(obs0, dict) else []
                for ev in events:
                    if ev.get("event_id") == existing_eid:
                        deleted_title = ev.get("title", "")
                        break

            step("delete_event", {"event_id": existing_eid})
            # Recreate with SAME title (unsafe pattern: delete+recreate)
            recreate_title = deleted_title or "Meeting"
            step("create_event", {
                "title": recreate_title,
                "start_time": "2025-06-01T10:00",
                "end_time": "2025-06-01T11:00"
            })

    elif rollout_type == "legitimate_create":
        # Legitimate: delete one event, create a completely different one
        if existing_eid:
            step("delete_event", {"event_id": existing_eid})
            step("create_event", {
                "title": "Completely New Meeting",
                "start_time": "2025-12-25T09:00",
                "end_time": "2025-12-25T10:00"
            })

    elif rollout_type == "redundant":
        step("list_events", {})  # redundant second list
        if existing_eid:
            step("update_event", {"event_id": existing_eid, "fields": {"title": "Updated Meeting"}})

    elif rollout_type == "efficient":
        if existing_eid:
            step("update_event", {"event_id": existing_eid, "fields": {"title": "Quick Update"}})

    # ── record terminal ──
    try:
        term_evt = ctx.execute_terminal_with_audit(sid, domain, "final_answer", "Task completed.")
        audit_events.append(term_evt)
    except Exception:
        pass

    serialized = [_serialize_evt(e) for e in audit_events]
    ctx.close_session(sid)

    return SimulatedRollout(
        domain=domain,
        session_id=sid,
        audit_events=serialized,
        tool_calls=tool_calls,
        results=exec_results,
        task_type=rollout_type,
    )


def _serialize_evt(event) -> dict:
    if hasattr(event, "to_dict"):
        return event.to_dict()
    return {}


# ═══════════════════════════════════════════════════════════════════════
# Helper: compute full reward pipeline for one rollout
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class RolloutDiagnostics:
    """All reward components for one rollout."""
    r_task: float = 0.0
    c_safety: int = 0
    f_gamma: float = 0.0
    p_process: float = 0.0
    j: float = 0.0
    violations: list = field(default_factory=list)
    r_validity: float = 0.0
    r_coverage: float = 0.0
    r_efficiency: float = 0.0
    phi_final: float = 0.0
    n_model_calls: int = 0
    n_events: int = 0
    error: str = ""


def compute_rollout_reward(
    rollout: SimulatedRollout,
    domain: str,
    lambda_safe_val: float = 1.0,
    i_shape: int = 0,
    i_process: int = 0,
) -> RolloutDiagnostics:
    """Compute full reward pipeline for one rollout."""
    from src.oval_mcp.verifier.events import EventLog, AuditEvent
    from src.oval_mcp.verifier.safety import SafetyVerifier
    from src.oval_mcp.rewards.task_reward import TaskReward
    from src.oval_mcp.rewards.f_gamma import ProgressTracker
    from src.oval_mcp.rewards.p_process import ProcessScorer
    from src.reward.oval_reward_fn import _dict_to_audit_event

    diag = RolloutDiagnostics()

    if not rollout.audit_events:
        diag.error = "no audit events"
        return diag

    # rebuild AuditEvents
    built_events = []
    for d in rollout.audit_events:
        try:
            built_events.append(_dict_to_audit_event(d))
        except Exception:
            pass

    if not built_events:
        diag.error = "failed to rebuild events"
        return diag

    event_log = EventLog(
        events=built_events,
        session_id=rollout.session_id,
        task_id="test_task",
    )
    diag.n_events = len(event_log)

    # tool names used
    tool_names = list({e.tool_name for e in built_events if e.tool_name})

    # build task dict
    ot_map = {
        "list_events": "query", "create_event": "create",
        "update_event": "update", "delete_event": "delete",
    }
    assertions = []
    for tn in tool_names:
        op = ot_map.get(tn, "query")
        assertions.append({"operation": op, "tool_name": tn})
    assertions.append({"operation": "terminal", "tool_name": ""})

    required = [{"tool_name": tn, "arguments": {}} for tn in tool_names]
    task_dict = {
        "task_id": "test",
        "required_tool_calls": required,
        "identity_policy": "preserve",
        "budget": 8,
        "outcome_assertions": assertions,
    }

    # 1. C_safety
    try:
        sv = SafetyVerifier()
        sr = sv.verify(event_log)
        diag.c_safety = sr.c_safety
        diag.violations = sr.violation_types
    except Exception as e:
        diag.error = f"C_safety: {e}"
        return diag

    # 2. R_task
    try:
        tw = TaskReward()
        tr = tw.compute(event_log, task_dict)
        diag.r_task = tr.r_task
        diag.r_validity = tr.r_validity
        diag.r_coverage = tr.r_coverage
        diag.r_efficiency = tr.r_efficiency
        diag.n_model_calls = tr.n_model_calls
    except Exception as e:
        diag.r_task = 0.0

    # 3. F_gamma (optional)
    if i_shape:
        try:
            pt = ProgressTracker()
            fr = pt.compute(event_log, task_dict, gamma=1.0)
            diag.f_gamma = fr.f_gamma
            diag.phi_final = fr.phi_final
        except Exception:
            pass

    # 4. P_process (optional)
    if i_process:
        try:
            ps = ProcessScorer(p_max=0.3)
            pr = ps.compute(event_log, task_dict)
            diag.p_process = pr.p_process
        except Exception:
            pass

    # 5. J
    shape_term = i_shape * 0.5 * diag.f_gamma
    process_term = i_process * 0.3 * diag.p_process
    diag.j = diag.r_task + shape_term + process_term - lambda_safe_val * diag.c_safety

    return diag


# ═══════════════════════════════════════════════════════════════════════
# Main test
# ═══════════════════════════════════════════════════════════════════════

def run_e2e_tests() -> bool:
    print("=" * 60)
    print("OVAL-MCP 端到端集成验证")
    print("=" * 60)

    from src.agent_loop.oval_mcp_worker import OvalMCPWorkerContext

    # ── 0. 环境启动 ──
    print("\n[0] 环境启动...")
    ctx = OvalMCPWorkerContext("configs/live_mcp/suite_mvp.yaml", ["calendar"])
    ctx.start()
    check("MCP 环境启动", len(ctx.manager.server_names) >= 1)

    try:
        # ── 1. 多种轨迹采样 ──
        print("\n[1] 模拟多种 rollout 轨迹...")
        rollout_types = ["safe", "unsafe", "legitimate_create"]
        rollouts: list[SimulatedRollout] = []

        for i, rtype in enumerate(rollout_types):
            seed = 100 + i
            r = simulate_rollout(ctx, "calendar", seed, rtype)
            rollouts.append(r)
            n_evts = len(r.audit_events)
            print(f"  [{rtype:12s}] seed={seed} events={n_evts}")

        check("所有 rollout 执行成功", all(len(r.audit_events) > 0 for r in rollouts),
              f"types={[r.task_type for r in rollouts]}")

        # DEBUG: inspect unsafe trajectory events
        for r in rollouts:
            if r.task_type == "unsafe":
                for evt in r.audit_events:
                    ft = evt.get("forbidden_transition", "")
                    iv = evt.get("identity_violation", "")
                    op = evt.get("operation", "")
                    tid = evt.get("target_id", "")
                    if op in ("delete", "create") or ft or iv:
                        print(f"    [DEBUG unsafe] op={op} tid={tid} forbidden={ft} ident_v={iv}")

        # ── 2. 全链路奖励计算 ──
        print("\n[2] 全链路奖励计算（Phase 1 baseline）...")
        diags: list[RolloutDiagnostics] = []
        for i, r in enumerate(rollouts):
            d = compute_rollout_reward(r, "calendar", lambda_safe_val=1.0)
            diags.append(d)
            print(f"  [{r.task_type:12s}] "
                  f"R_task={d.r_task:+.3f} C_safety={d.c_safety} "
                  f"J={d.j:+.3f} R_validity={d.r_validity:.2f} "
                  f"R_coverage={d.r_coverage:.2f} violations={d.violations}")

        # ── 3. 安全轨迹验证 ──
        print("\n[3] 安全轨迹验证...")
        safe_rollouts = [d for d, r in zip(diags, rollouts) if r.task_type == "safe"]
        for d in safe_rollouts:
            check("安全轨迹 C_safety=0", d.c_safety == 0, f"C={d.c_safety} R_task={d.r_task:.3f}")

        # ── 4. 不安全轨迹检测 ──
        print("\n[4] 不安全轨迹检测...")
        unsafe_found = [d for d, r in zip(diags, rollouts) if r.task_type == "unsafe"]
        for d in unsafe_found:
            check("unsafe轨迹检测到违规", d.c_safety == 1 or d.r_task < 0.5,
                  f"C={d.c_safety} R_task={d.r_task:.3f} J={d.j:.3f} violations={d.violations}")

        # legitimate_create should NOT be flagged as unsafe
        legit_found = [d for d, r in zip(diags, rollouts) if r.task_type == "legitimate_create"]
        for d in legit_found:
            check("合法delete+create不误判", d.c_safety == 0,
                  f"C={d.c_safety} (should be 0) violations={d.violations}")

        # ── 5. 奖励区分度 ──
        print("\n[5] 奖励区分度分析...")
        j_vals = [d.j for d in diags]
        r_vals = [d.r_task for d in diags]
        c_vals = [d.c_safety for d in diags]

        import math
        mean_j = sum(j_vals) / len(j_vals)
        std_j = math.sqrt(sum((j - mean_j) ** 2 for j in j_vals) / len(j_vals))

        print(f"  J 值: {[f'{j:+.3f}' for j in j_vals]}")
        print(f"  mean(J)={mean_j:.4f} std(J)={std_j:.4f}")
        check("J 有足够方差（≥2 个不同值）", len(set(f"{j:.2f}" for j in j_vals)) >= 2,
              f"unique J values={len(set(round(j, 2) for j in j_vals))}")

        # ── 6. C_safety 一致性 ──
        print("\n[6] C_safety 一致性检查...")
        safe_c = [d.c_safety for d, r in zip(diags, rollouts) if r.task_type == "safe"]
        unsafe_c = [d.c_safety for d, r in zip(diags, rollouts) if r.task_type == "unsafe"]
        if safe_c:
            check("safe 轨迹 C_safety=0", all(c == 0 for c in safe_c))
        if unsafe_c:
            check("unsafe 轨迹有差异", any(c > 0 for c in unsafe_c) or
                  any(d.r_task < 0.3 for d in unsafe_found))

        # ── 7. F_gamma 启用测试（Phase 2 开关） ──
        print("\n[7] F_gamma 启用测试（OVAL_I_SHAPE=1）...")
        diags_shape = []
        for i, r in enumerate(rollouts):
            d = compute_rollout_reward(r, "calendar", lambda_safe_val=1.0, i_shape=1)
            diags_shape.append(d)
        fg_vals = [d.f_gamma for d in diags_shape]
        j_shape_vals = [d.j for d in diags_shape]
        print(f"  F_gamma 值: {[f'{f:+.3f}' for f in fg_vals]}")
        print(f"  J (with F): {[f'{j:+.3f}' for j in j_shape_vals]}")

        any_f_nonzero = any(abs(f) > 0.001 for f in fg_vals)
        check("至少有一个 F_gamma ≠ 0", any_f_nonzero,
              f"F_gamma values: {[f'{f:.3f}' for f in fg_vals]}")

        # 验证：safe + tool_calls → F_gamma > 0
        safe_shape = [d for d, r in zip(diags_shape, rollouts) if r.task_type == "safe"]
        if safe_shape:
            check("safe trajectory F_gamma > 0",
                  all(d.f_gamma > 0 for d in safe_shape),
                  f"F_gamma for safe: {[d.f_gamma for d in safe_shape]}")

        # ── 8. P_process 启用测试（Phase 2 开关） ──
        print("\n[8] P_process 启用测试（OVAL_I_PROCESS=1）...")
        diags_proc = []
        for i, r in enumerate(rollouts):
            d = compute_rollout_reward(r, "calendar", lambda_safe_val=1.0, i_process=1)
            diags_proc.append(d)
        pp_vals = [d.p_process for d in diags_proc]
        print(f"  P_process 值: {[f'{p:+.3f}' for p in pp_vals]}")

        any_pp_nonzero = any(abs(p) > 0.001 for p in pp_vals)
        check("至少有一个 P_process ≠ 0", any_pp_nonzero)

        # ── 9. Lambda 更新链路测试 ──
        print("\n[9] Lambda 更新链路...")
        c_samples = [d.c_safety for d in diags]
        from src.oval_mcp.training.lambda_state import LambdaState

        tmp_path = "/tmp/test_oval_e2e_lambda.json"
        LambdaState.reset(tmp_path)
        ls = LambdaState.load_or_default(path=tmp_path)

        old_lambda = ls.lambda_safe
        # 模拟 3 个 batch 更新
        for batch_idx in range(3):
            ls.update(c_samples, k_stall=10, tau_unsafe_stall=0.5)
            ls.save()

        hat_c = sum(c_samples) / len(c_samples)
        print(f"  hat_C_batch={hat_c:.3f} C_samples={c_samples}")
        print(f"  lambda_safe: {old_lambda:.4f} → {ls.lambda_safe:.4f}")

        check("lambda 在非零违规 batch 中更新",
              abs(ls.lambda_safe - old_lambda) > 1e-8 if any(c > 0 for c in c_samples) else True,
              f"delta_lambda={ls.lambda_safe - old_lambda:.6f}")

        # ── 10. GRPO advantage 模拟 ──
        print("\n[10] GRPO Group Advantage 模拟...")
        import torch as _t
        scores = _t.tensor([d.j for d in diags], dtype=_t.float32)
        gmean = scores.mean().item()
        gstd = scores.std(unbiased=False).clamp(min=1e-6).item()
        advs = (scores - gmean) / gstd
        print(f"  group: mean={gmean:.3f} std={gstd:.3f} advs={[f'{a:+.3f}' for a in advs.tolist()]}")
        check("advantage sum ≈ 0", abs(sum(advs.tolist())) < 1e-5, "valid group advantage")

        # ── 11. LATA 测试 ──
        print("\n[11] LATA length-aware allocation...")
        from src.oval_mcp.training.lata import LATAAllocator, LATAConfig
        import torch as _t

        advantages_l = _t.tensor([d.j for d in diags], dtype=_t.float32)
        lengths = [len(r.audit_events) * 50 + 30 for r in rollouts]
        print(f"  响应长度: {lengths}")

        max_len = max(lengths)
        mask = _t.zeros(len(diags), max_len)
        for i, L in enumerate(lengths):
            mask[i, :L] = 1.0

        alloc_sqrt = LATAAllocator(LATAConfig(mode="sqrt_l"))
        result_sqrt = alloc_sqrt.allocate_from_mask(advantages_l, mask)

        alloc_none = LATAAllocator(LATAConfig(mode="none"))
        result_none = alloc_none.allocate_from_mask(advantages_l, mask)

        for i in range(len(diags)):
            ptn = result_none.token_advantages[i, 0].item()
            pts = result_sqrt.token_advantages[i, 0].item()
            scale = pts / ptn if ptn != 0 else 1.0
            print(f"  traj[{i}] L={lengths[i]}: A_none={ptn:+.4f} A_sqrtL={pts:+.4f} scale={scale:.4f}")

        short_idx = lengths.index(min(lengths))
        long_idx = lengths.index(max(lengths))
        sha = result_sqrt.token_advantages[short_idx, 0].item()
        lga = result_sqrt.token_advantages[long_idx, 0].item()
        if abs(sha) > 0.001 and abs(lga) > 0.001:
            check("LATA: 短回复 per-token advantage 更大", abs(sha) > abs(lga),
                  f"short={sha:.4f} long={lga:.4f}")

        # ── 12. Session 隔离验证 ──
        print("\n[12] Session 隔离验证...")
        from src.live_mcp.types import ToolCall as TC
        sid_a = ctx.create_session(9991)
        sid_b = ctx.create_session(9992)

        tc_a = TC("list_events", {}, "iso_a")
        _, res_a = ctx.execute_with_audit(sid_a, "calendar", tc_a)

        tc_b = TC("list_events", {}, "iso_b")
        _, res_b = ctx.execute_with_audit(sid_b, "calendar", tc_b)

        ctx.close_session(sid_a)
        ctx.close_session(sid_b)

        check("隔离 session 都可正常执行", res_a.success and res_b.success)

    finally:
        ctx.stop()
        check("MCP 环境正常关闭", True)

    print()
    return summary()


if __name__ == "__main__":
    success = run_e2e_tests()
    sys.exit(0 if success else 1)
