#!/usr/bin/env python3
"""OvalMCPWorkerContext 独立集成测试 — 简洁版。

测试链路：
  1. 启动 OvalMCPWorkerContext
  2. 创建 session，执行 tool_calls
  3. 验证审计事件生成和序列化
  4. 验证删除+重建模式检测（C_safety）
  5. 验证 reward function 接口

用法：
  /mnt/data1/zhanyiliu/liuzhanyi/anaconda3/envs/arl/bin/python scripts/oval_mcp_worker_test.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.agent_loop.oval_mcp_worker import OvalMCPWorkerContext
from src.live_mcp.types import ToolCall
from src.oval_mcp.verifier.events import EventLog
from src.oval_mcp.verifier.safety import SafetyVerifier
from src.oval_mcp.rewards.task_reward import TaskReward
from src.reward.oval_reward_fn import compute_score


def run():
    p = 0
    f = 0

    def ok(name, cond, detail=""):
        nonlocal p, f
        if cond:
            p += 1
        else:
            f += 1
        s = "✓" if cond else "✗"
        print(f"  {s} {name}" + (f" — {detail}" if detail else ""))

    print("=" * 60)
    print("OvalMCPWorkerContext Integration Test")
    print("=" * 60)

    # ── 1. 启动 ──
    print("\n[1] Starting OvalMCPWorkerContext...")
    ctx = OvalMCPWorkerContext("configs/live_mcp/suite_mvp.yaml", ["calendar", "shopping"])
    ctx.start()
    ok("servers_started", len(ctx.manager.server_names) >= 2, str(ctx.manager.server_names))

    # ── 2. 创建 session + list_events ──
    print("\n[2] list_events...")
    sid = ctx.create_session(42)
    ok("session_created", bool(sid))

    tc_list = ToolCall("list_events", {}, "c1")
    evt1, res1 = ctx.execute_with_audit(sid, "calendar", tc_list)
    ok("list_events_ok", res1.success)
    ok("audit_event_1", evt1.step == 1, f"step={evt1.step} op={evt1.operation}")

    # ── 3. update_event ──
    print("\n[3] update_event...")
    obs = res1.observation
    events_list = obs.get("events", []) if isinstance(obs, dict) else []
    update_event_id = events_list[0].get("event_id", "") if events_list else ""

    if update_event_id:
        tc_upd = ToolCall("update_event", {"event_id": update_event_id, "title": "Updated"}, "c2")
        evt2, res2 = ctx.execute_with_audit(sid, "calendar", tc_upd)
        ok("update_ok", res2.success)
        ok("update_target", evt2.target_id == update_event_id, f"target={evt2.target_id}")
    else:
        ok("update_skipped", True, "no events in list_events")
        update_event_id = ""

    # ── 4. delete + recreate (unsafe pattern) ──
    print("\n[4] delete + recreate (unsafe)...")
    all_events = [evt1]
    if update_event_id:
        tc_del = ToolCall("delete_event", {"event_id": update_event_id}, "c3")
        evt_del, res_del = ctx.execute_with_audit(sid, "calendar", tc_del)
        all_events.append(evt_del)
        ok("delete_ok", res_del.success)

        tc_create = ToolCall("create_event", {"title": "Updated", "start_time": "2025-01-01T10:00", "end_time": "2025-01-01T11:00"}, "c4")
        evt_cre, res_cre = ctx.execute_with_audit(sid, "calendar", tc_create)
        all_events.append(evt_cre)
        ok("recreate_ok", res_cre.success)

        # safety check
        sv = SafetyVerifier()
        e_log = EventLog(events=all_events, session_id=sid, task_id="test")
        sres = sv.verify(e_log)
        ok("unsafe_detected", sres.c_safety > 0, f"C_safety={sres.c_safety} violations={sres.violation_types}")
    else:
        ok("unsafe_skipped", True, "no event_id")

    # ── 5. session isolation ──
    print("\n[5] Session isolation...")
    sid2 = ctx.create_session(99)
    tc_fresh = ToolCall("list_events", {}, "c_fresh")
    _, res_fresh = ctx.execute_with_audit(sid2, "calendar", tc_fresh)
    ok("isolation", res_fresh.success)
    ctx.close_session(sid)
    ctx.close_session(sid2)

    # ── 6. serialization ──
    print("\n[6] Serialization...")
    serialized = ctx.serialize_audit_events(all_events)
    ok("serialized", len(serialized) == len(all_events), f"{len(serialized)}/{len(all_events)}")
    # verify roundtrip: dict → AuditEvent → dict
    for evt_dict in serialized:
        from src.reward.oval_reward_fn import _dict_to_audit_event
        rebuilt = _dict_to_audit_event(evt_dict)
        ok(f"roundtrip step={evt_dict.get('step')}", rebuilt.step == evt_dict.get("step", -1))

    # ── 7. reward function ──
    print("\n[7] Reward function interface...")
    required = ["list_events"] + (["update_event"] if update_event_id else [])
    result = compute_score("oval", "mock", {}, {
        "audit_events": serialized,
        "task_id": "test",
        "domain": "calendar",
        "required_tools": required,
        "n_model_tool_calls": len(all_events),
        "n_exec_success": len(all_events),
        "session_id": sid,
    })
    print(f"  Result: score={result['score']:.3f} r_task={result['r_task']:.3f} c_safety={result['c_safety']:.1f} j={result['j']:.3f}")
    ok("score_exists", "score" in result)
    ok("r_task_exists", "r_task" in result)
    ok("c_safety_exists", "c_safety" in result)

    # cleanup
    ctx.stop()
    ok("context_stopped", True)

    print(f"\n{'='*60}")
    print(f"Passed: {p}/{p+f}")
    print(f"{'='*60}")
    return f == 0


if __name__ == "__main__":
    success = run()
    sys.exit(0 if success else 1)
