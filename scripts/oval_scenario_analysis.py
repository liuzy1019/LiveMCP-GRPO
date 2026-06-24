#!/usr/bin/env python3
"""OVAL-MCP 全场景奖励反馈测试与分析。

Design rationale:
  R_task = clip(R_positive/Z_pos + w_eff*R_efficiency, -0.2, 1.0)
  R_positive = 0.5*R_validity + 0.5*R_coverage + 0.2*R_name + 0.1*R_arg
  J = R_task + I_shape*λ_shape*F_gamma + I_process*λ_process*P_process - λ_safe*C_safety

测试场景矩阵：
  ┌──────────────────────┬────────────────────────────────────────────────────┐
  │ Dimension            │ Values                                              │
  ├──────────────────────┼────────────────────────────────────────────────────┤
  │ Safety               │ safe / unsafe(new_entity) / unsafe(delete+recreate) │
  │ Calls count          │ 0 / 1 / 2 / 3 / 4 / 5 / 6 (redundant)              │
  │ Execution success    │ all_pass / partial_fail / all_fail                  │
  │ Identity policy      │ preserve / create_new                               │
  │ Required tools       │ 1 / 2 / 3                                           │
  │ Terminal action      │ final_answer / ask_clarification / report_error     │
  └──────────────────────┴────────────────────────────────────────────────────┘
"""

import sys
import math
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.agent_loop.oval_mcp_worker import OvalMCPWorkerContext
from src.live_mcp.types import ToolCall
from src.oval_mcp.verifier.events import EventLog
from src.oval_mcp.verifier.safety import SafetyVerifier, SafetyResult
from src.oval_mcp.rewards.task_reward import TaskReward, TaskRewardResult
from src.oval_mcp.rewards.f_gamma import ProgressTracker, FGammaResult
from src.oval_mcp.rewards.p_process import ProcessScorer, ProcessScoreResult
from src.reward.oval_reward_fn import _dict_to_audit_event


# ═══════════════════════════════════════════════════════════════════════
# Scenario runner
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class ScenarioConfig:
    name: str
    domain: str
    steps: list[dict]  # [{tool_name, args, expect_fail?, action_type}, ...]
    task_overrides: dict = field(default_factory=dict)


@dataclass
class ScenarioResult:
    config: ScenarioConfig
    r_task: float = 0.0
    r_validity: float = 0.0
    r_coverage: float = 0.0
    r_name: float = 0.0
    r_arg: float = 0.0
    r_efficiency: float = 0.0
    r_positive: float = 0.0
    z_pos: float = 1.0
    c_safety: int = 0
    c_details: str = ""
    f_gamma: float = 0.0
    phi_final: float = 0.0
    per_turn_f: str = ""
    p_process: float = 0.0
    p_per_step: str = ""
    j_phase1: float = 0.0
    j_m4f: float = 0.0
    j_m4p: float = 0.0
    j_m4fp: float = 0.0
    n_events: int = 0
    n_tool_calls: int = 0
    n_exec_ok: int = 0
    error: str = ""


def run_scenario(
    ctx: OvalMCPWorkerContext,
    cfg: ScenarioConfig,
    lambda_safe: float = 1.0,
) -> ScenarioResult:
    result = ScenarioResult(config=cfg)

    sid = ctx.create_session(seed=hash(cfg.name) % 10000 + 100)
    audit_events: list = []
    last_created_id: str = ""  # Track for self-contradiction scenarios
    last_deleted_title: str = ""  # Track for delete+recreate detection
    last_deleted_start: str = ""
    last_deleted_end: str = ""
    last_event_id: str = ""  # Track for need_event_id patterns

    try:
        executed_tool_names: list[str] = []
        n_exec_ok = 0

        for i, step in enumerate(cfg.steps):
            action_type = step.get("action_type", "tool_call")
            args = dict(step.get("args", {}))

            # Runtime dynamic arg injection (within same session)
            if step.get("use_last_created_id") and last_created_id:
                args["event_id"] = last_created_id

            if step.get("use_deleted_entity_title") and last_deleted_title:
                args["title"] = last_deleted_title
                if last_deleted_start:
                    args["start_time"] = last_deleted_start
                if last_deleted_end:
                    args["end_time"] = last_deleted_end

            if step.get("need_event_id") and last_event_id:
                args["event_id"] = last_event_id
                args.pop("need_event_id", None)

            if action_type == "terminal":
                evt = ctx.execute_terminal_with_audit(
                    sid, cfg.domain, step["tool_name"],
                )
                audit_events.append(evt)
            else:
                tc = ToolCall(step["tool_name"], args, f"call_{i}")
                evt, exec_res = ctx.execute_with_audit(sid, cfg.domain, tc)
                audit_events.append(evt)
                executed_tool_names.append(step["tool_name"])
                if exec_res.success:
                    n_exec_ok += 1

                # Track created entity ID from observation for self-contradiction test
                if step["tool_name"] == "create_event" and exec_res.success:
                    obs = exec_res.observation if exec_res else {}
                    if isinstance(obs, dict):
                        evt_data = obs.get("event", obs)
                        eid = evt_data.get("event_id", "")
                        if eid:
                            last_created_id = eid

                # Track deleted entity title for unsafe recreate detection
                if step["tool_name"] == "delete_event" and exec_res.success:
                    # Title was captured from list_events step (see below)
                    pass

                # Snapshot entity info from list_events for later recreate detection
                # and for need_event_id patterns
                if step["tool_name"] in ("list_events", "get_event") and exec_res.success:
                    obs = exec_res.observation if exec_res else {}
                    events_list = obs.get("events", []) if isinstance(obs, dict) else []
                    # Also handle single event response from get_event
                    single_event = obs.get("event", None) if isinstance(obs, dict) else None
                    if single_event:
                        events_list = [single_event]
                    if events_list:
                        first = events_list[0]
                        last_event_id = first.get("event_id", "")
                        if step.get("capture_for_recreate"):
                            last_deleted_title = first.get("title", "")
                            last_deleted_start = first.get("start_time", "")
                            last_deleted_end = first.get("end_time", "")

        result.n_events = len(audit_events)
        result.n_tool_calls = sum(1 for e in audit_events if hasattr(e, "tool_name") and e.tool_name)
        result.n_exec_ok = n_exec_ok

        # Serialize + rebuild (simulates verl reward function path)
        serialized = [e.to_dict() for e in audit_events if hasattr(e, "to_dict")]
        built_events = [_dict_to_audit_event(d) for d in serialized]

        event_log = EventLog(events=built_events, session_id=sid, task_id=cfg.name)

        # ── C_safety ──
        sv = SafetyVerifier()
        safety_result = sv.verify(event_log)
        result.c_safety = safety_result.c_safety
        result.c_details = ",".join(safety_result.violation_types) if safety_result.violation_types else "none"

        # ── R_task ──
        tw = TaskReward()
        task_op_map = {
            "list_events": "query", "get_event": "query",
            "create_event": "create", "update_event": "update", "delete_event": "delete",
            "search_products": "query", "add_to_cart": "update",
            "remove_from_cart": "update", "checkout": "create", "get_order": "query",
            "final_answer": "terminal", "ask_clarification": "terminal", "report_error": "terminal",
        }
        assertions = []
        for tn in set(executed_tool_names):
            op = task_op_map.get(tn, "query")
            assertions.append({"operation": op, "tool_name": tn})
        assertions.append({"operation": "terminal", "tool_name": ""})

        task_dict = {
            "task_id": cfg.name,
            "required_tool_calls": [{"tool_name": tn, "arguments": {}} for tn in set(executed_tool_names)],
            "identity_policy": cfg.task_overrides.get("identity_policy", "preserve"),
            "budget": cfg.task_overrides.get("budget", 8),
            "outcome_assertions": assertions,
        }
        tr = tw.compute(event_log, task_dict)
        result.r_task = tr.r_task
        result.r_validity = tr.r_validity
        result.r_coverage = tr.r_coverage
        result.r_name = tr.r_name
        result.r_arg = tr.r_arg
        result.r_efficiency = tr.r_efficiency
        result.r_positive = tr.r_positive
        result.z_pos = tr.z_pos

        # ── F_gamma ──
        pt = ProgressTracker()
        fgr = pt.compute(event_log, task_dict, gamma=1.0)
        result.f_gamma = fgr.f_gamma
        result.phi_final = fgr.phi_final
        result.per_turn_f = ",".join(f"{f:+.3f}" for f in fgr.per_turn_f)

        # ── P_process ──
        ps = ProcessScorer(p_max=0.3)
        psr = ps.compute(event_log, task_dict)
        result.p_process = psr.p_process
        result.p_per_step = ",".join(f"{s.p_clamped:+.3f}" for s in psr.per_step)

        # ── J ──
        result.j_phase1 = result.r_task - lambda_safe * result.c_safety
        result.j_m4f = result.r_task + 0.5 * result.f_gamma - lambda_safe * result.c_safety
        result.j_m4p = result.r_task + 0.3 * result.p_process - lambda_safe * result.c_safety
        result.j_m4fp = result.r_task + 0.5 * result.f_gamma + 0.3 * result.p_process - lambda_safe * result.c_safety

    except Exception as e:
        result.error = str(e)
    finally:
        ctx.close_session(sid)

    return result


# ═══════════════════════════════════════════════════════════════════════
# Scenario definitions
# ═══════════════════════════════════════════════════════════════════════

def build_scenarios() -> list[ScenarioConfig]:
    return [
        # ── Minimal / empty ──
        ScenarioConfig("no_tool_terminal", "calendar", [
            {"action_type": "terminal", "tool_name": "final_answer"},
        ], {"budget": 5}),

        # ── Safe: 1 OP ──
        ScenarioConfig("safe_1_list", "calendar", [
            {"tool_name": "list_events", "args": {}},
            {"action_type": "terminal", "tool_name": "final_answer"},
        ]),

        ScenarioConfig("safe_1_update", "calendar", [
            {"tool_name": "list_events", "args": {}},
            {"tool_name": "update_event", "args": {"fields": {"title": "Updated Meeting"}}, "need_event_id": True},
            {"action_type": "terminal", "tool_name": "final_answer"},
        ]),

        ScenarioConfig("safe_1_create", "calendar", [
            {"tool_name": "create_event", "args": {"title": "New Meeting", "start_time": "2025-06-01T10:00", "end_time": "2025-06-01T11:00"}},
            {"action_type": "terminal", "tool_name": "final_answer"},
        ], {"identity_policy": "create_new"}),

        # ── Safe: 2 OP ──
        ScenarioConfig("safe_2_update_list", "calendar", [
            {"tool_name": "list_events", "args": {}},
            {"tool_name": "update_event", "args": {"fields": {"title": "Updated Twice"}}, "need_event_id": True},
            {"tool_name": "list_events", "args": {}},
            {"action_type": "terminal", "tool_name": "final_answer"},
        ]),

        ScenarioConfig("unsafe_create_delete_own", "calendar", [
            {"tool_name": "create_event", "args": {"title": "Temp", "start_time": "2025-06-01T10:00", "end_time": "2025-06-01T11:00"}},
            {"tool_name": "delete_event", "args": {}, "use_last_created_id": True},
            {"action_type": "terminal", "tool_name": "final_answer"},
        ], {"identity_policy": "create_new"}),

        # ── Safe: 3 OP ──
        ScenarioConfig("safe_3_chain", "calendar", [
            {"tool_name": "list_events", "args": {}},
            {"tool_name": "get_event", "args": {}, "need_event_id": True},
            {"tool_name": "update_event", "args": {"fields": {"title": "Chain Updated"}}, "need_event_id": True},
            {"action_type": "terminal", "tool_name": "final_answer"},
        ]),

        # ── Unsafe: delete+recreate ──
        ScenarioConfig("unsafe_delete_recreate", "calendar", [
            {"tool_name": "list_events", "args": {}, "capture_for_recreate": True},
            {"tool_name": "delete_event", "args": {}, "need_event_id": True},
            {"tool_name": "create_event", "args": {}, "use_deleted_entity_title": True},
            {"action_type": "terminal", "tool_name": "final_answer"},
        ]),

        # ── Unsafe: self-contradiction ──
        ScenarioConfig("unsafe_self_contradict", "calendar", [
            {"tool_name": "create_event", "args": {"title": "Temp", "start_time": "2025-06-01T10:00", "end_time": "2025-06-01T11:00"}},
            {"tool_name": "delete_event", "args": {}, "use_last_created_id": True},
            {"action_type": "terminal", "tool_name": "final_answer"},
        ], {"identity_policy": "create_new"}),

        # ── Error: partial fail ──
        ScenarioConfig("error_partial_fail", "calendar", [
            {"tool_name": "list_events", "args": {}},
            {"tool_name": "update_event", "args": {"event_id": "NONEXISTENT", "fields": {"title": "Won't Work"}}},
            {"tool_name": "list_events", "args": {}},
            {"action_type": "terminal", "tool_name": "final_answer"},
        ]),

        # ── Redundant calls ──
        ScenarioConfig("redundant_3x_list", "calendar", [
            {"tool_name": "list_events", "args": {}},
            {"tool_name": "list_events", "args": {}},
            {"tool_name": "list_events", "args": {}},
            {"action_type": "terminal", "tool_name": "final_answer"},
        ]),

        ScenarioConfig("redundant_5_calls", "calendar", [
            {"tool_name": "list_events", "args": {}},
            {"tool_name": "list_events", "args": {}},
            {"tool_name": "list_events", "args": {}},
            {"tool_name": "list_events", "args": {}},
            {"tool_name": "list_events", "args": {}},
            {"action_type": "terminal", "tool_name": "final_answer"},
        ]),

        # ── Terminal variants ──
        ScenarioConfig("terminal_ask", "calendar", [
            {"tool_name": "list_events", "args": {}},
            {"action_type": "terminal", "tool_name": "ask_clarification"},
        ]),

        ScenarioConfig("terminal_error", "calendar", [
            {"tool_name": "list_events", "args": {}},
            {"action_type": "terminal", "tool_name": "report_error"},
        ]),

        # ── Shopping domain ──
        ScenarioConfig("shopping_2_ops", "shopping", [
            {"tool_name": "search_products", "args": {"query": "laptop"}},
            {"tool_name": "add_to_cart", "args": {"product_id": "prod_001", "quantity": 1}},
            {"action_type": "terminal", "tool_name": "final_answer"},
        ], {"identity_policy": "create_new"}),

        ScenarioConfig("shopping_checkout", "shopping", [
            {"tool_name": "search_products", "args": {"query": "laptop"}},
            {"tool_name": "add_to_cart", "args": {"product_id": "prod_001", "quantity": 1}},
            {"tool_name": "checkout", "args": {}},
            {"action_type": "terminal", "tool_name": "final_answer"},
        ], {"identity_policy": "create_new"}),

        # ── Legitimate create after delete (different entity) ──
        ScenarioConfig("legit_delete_create_diff", "calendar", [
            {"tool_name": "list_events", "args": {}},
            {"tool_name": "delete_event", "args": {}, "need_event_id": True},
            {"tool_name": "create_event", "args": {"title": "Completely Different", "start_time": "2026-01-01T09:00", "end_time": "2026-01-01T10:00"}},
            {"action_type": "terminal", "tool_name": "final_answer"},
        ]),
    ]


# ═══════════════════════════════════════════════════════════════════════
# Smart arg resolution for scenarios that need dynamic IDs
# ═══════════════════════════════════════════════════════════════════════

def resolve_scenario_args(scenario: ScenarioConfig, ctx: OvalMCPWorkerContext) -> ScenarioConfig:
    """For scenarios with 'recreate_from_deleted', pre-run a list_events
    in an ephemeral session to get the title/start/end of the entity to recreate.
    
    `need_event_id` and `need_created_id` are now handled within run_scenario
    (same session), so the resolver should NOT touch those.
    """
    
    has_recreate = any(
        step.get("recreate_from_deleted")
        for step in scenario.steps
    )
    if not has_recreate:
        return scenario

    sid = ctx.create_session(seed=hash(scenario.name + "_resolver") % 10000 + 500)
    try:
        tc = ToolCall("list_events", {}, "resolver_0")
        _, res = ctx.execute_with_audit(sid, scenario.domain, tc)
        obs = res.observation if res else {}
        events_list = obs.get("events", []) if isinstance(obs, dict) else []
        existing_eid = events_list[0].get("event_id", "") if events_list else ""
        existing_title = events_list[0].get("title", "") if events_list else ""
        existing_start = events_list[0].get("start_time", "") if events_list else ""
        existing_end = events_list[0].get("end_time", "") if events_list else ""

        for step in scenario.steps:
            if step.get("recreate_from_deleted"):
                # Need the deleted entity's title for the recreate
                # But actual delete happens in run_scenario, so we just
                # store the entity info for later use
                step["args"] = {
                    "title": existing_title,
                    "start_time": existing_start,
                    "end_time": existing_end,
                }
                del step["recreate_from_deleted"]
    finally:
        ctx.close_session(sid)

    return scenario


# ═══════════════════════════════════════════════════════════════════════
# Analysis helpers
# ═══════════════════════════════════════════════════════════════════════

def analyze_results(results: list[ScenarioResult]) -> dict:
    """Produce summary statistics across all scenarios."""
    stats = {}

    # Overall J distribution
    j_vals = [r.j_phase1 for r in results]
    stats["n_scenarios"] = len(results)
    stats["mean_j"] = sum(j_vals) / len(j_vals)
    stats["std_j"] = math.sqrt(sum((j - stats["mean_j"]) ** 2 for j in j_vals) / len(j_vals))
    stats["min_j"] = min(j_vals)
    stats["max_j"] = max(j_vals)
    stats["j_range"] = stats["max_j"] - stats["min_j"]
    stats["unique_j"] = len(set(round(j, 3) for j in j_vals))
    stats["saturated_rate"] = sum(1 for r in results if abs(r.j_phase1 - stats["mean_j"]) < 0.001) / len(results)

    # Safety stats
    stats["n_unsafe"] = sum(1 for r in results if r.c_safety > 0)
    stats["unsafe_rate"] = stats["n_unsafe"] / len(results)
    stats["mean_r_task"] = sum(r.r_task for r in results) / len(results)
    stats["mean_r_task_safe"] = (
        sum(r.r_task for r in results if r.c_safety == 0) / max(1, sum(1 for r in results if r.c_safety == 0))
    )
    stats["mean_r_task_unsafe"] = (
        sum(r.r_task for r in results if r.c_safety > 0) / max(1, sum(1 for r in results if r.c_safety > 0))
    )

    # F_gamma stats
    fg_vals = [r.f_gamma for r in results]
    stats["mean_f_gamma"] = sum(fg_vals) / len(fg_vals)
    stats["std_f_gamma"] = math.sqrt(sum((f - stats["mean_f_gamma"]) ** 2 for f in fg_vals) / len(fg_vals))

    # P_process stats
    pp_vals = [r.p_process for r in results]
    stats["mean_p_process"] = sum(pp_vals) / len(pp_vals)
    stats["std_p_process"] = math.sqrt(sum((p - stats["mean_p_process"]) ** 2 for p in pp_vals) / len(pp_vals))

    # R_task decomposition averages
    stats["mean_r_validity"] = sum(r.r_validity for r in results) / len(results)
    stats["mean_r_coverage"] = sum(r.r_coverage for r in results) / len(results)
    stats["mean_r_name"] = sum(r.r_name for r in results) / len(results)
    stats["mean_r_arg"] = sum(r.r_arg for r in results) / len(results)
    stats["mean_r_efficiency"] = sum(r.r_efficiency for r in results) / len(results)

    # Phase 2 J variants
    stats["mean_j_m4f"] = sum(r.j_m4f for r in results) / len(results)
    stats["mean_j_m4p"] = sum(r.j_m4p for r in results) / len(results)
    stats["mean_j_m4fp"] = sum(r.j_m4fp for r in results) / len(results)

    # J range widening from F_gamma / P_process
    m4f_range = max(r.j_m4f for r in results) - min(r.j_m4f for r in results)
    m4p_range = max(r.j_m4p for r in results) - min(r.j_m4p for r in results)
    m4fp_range = max(r.j_m4fp for r in results) - min(r.j_m4fp for r in results)
    stats["j_range_widen_f"] = (m4f_range - stats["j_range"]) / max(stats["j_range"], 0.01)
    stats["j_range_widen_p"] = (m4p_range - stats["j_range"]) / max(stats["j_range"], 0.01)
    stats["j_range_widen_fp"] = (m4fp_range - stats["j_range"]) / max(stats["j_range"], 0.01)

    # Safety penalty magnitude
    unsafe_results = [r for r in results if r.c_safety > 0]
    if unsafe_results:
        stats["mean_j_drop_unsafe"] = sum(
            (r.r_task - r.j_phase1) for r in unsafe_results
        ) / len(unsafe_results)
    else:
        stats["mean_j_drop_unsafe"] = 0.0

    return stats


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 110)
    print("OVAL-MCP 全场景奖励反馈分析")
    print("=" * 110)

    ctx = OvalMCPWorkerContext("configs/live_mcp/suite_mvp.yaml", ["calendar", "shopping"])
    ctx.start()
    print(f"  环境启动: {list(ctx.manager.server_names)}")

    try:
        scenarios = build_scenarios()
        results: list[ScenarioResult] = []

        print(f"\n{'场景':<30} {'R_task':>7} {'R_val':>6} {'R_cov':>6} {'R_eff':>7} {'C_saf':>5} {'F_gam':>7} {'P_proc':>7} {'J_ph1':>7} {'J_F':>7} {'J_P':>7} {'J_FP':>7} {'违规明细'}")
        print("-" * 110)

        for sc in scenarios:
            resolved = resolve_scenario_args(sc, ctx)
            r = run_scenario(ctx, resolved)
            results.append(r)

            flags = ""
            if r.c_safety > 0:
                flags += " ⚠UNSAFE"
            if r.error:
                flags += " ❌ERR"
            if r.r_efficiency < -0.01:
                flags += " 📉EFF"

            print(
                f"{r.config.name:<30} "
                f"{r.r_task:>+7.3f} {r.r_validity:>6.2f} {r.r_coverage:>6.2f} {r.r_efficiency:>+7.3f} "
                f"{r.c_safety:>5d} {r.f_gamma:>+7.3f} {r.p_process:>+7.3f} "
                f"{r.j_phase1:>+7.3f} {r.j_m4f:>+7.3f} {r.j_m4p:>+7.3f} {r.j_m4fp:>+7.3f} "
                f" {r.c_details}{flags}"
            )

        stats = analyze_results(results)

        print(f"\n{'='*110}")
        print("场景维度分析")
        print(f"{'='*110}")

        # Group by safety
        print("\n── 按 Safety 分组 ──")
        safe = [r for r in results if r.c_safety == 0]
        unsafe = [r for r in results if r.c_safety > 0]

        def group_stats(name, group):
            if not group:
                print(f"  {name}: 无数据")
                return
            j = [r.j_phase1 for r in group]
            jf = [r.j_m4f for r in group]
            jp = [r.j_m4p for r in group]
            jfp = [r.j_m4fp for r in group]
            rt = [r.r_task for r in group]
            fg = [r.f_gamma for r in group]
            pp = [r.p_process for r in group]
            m = lambda xs: sum(xs)/len(xs)
            s = lambda xs: math.sqrt(sum((x-m(xs))**2 for x in xs)/len(xs))
            print(f"  {name} (n={len(group)}):")
            print(f"    R_task: μ={m(rt):.4f} σ={s(rt):.4f} [{min(rt):.3f}, {max(rt):.3f}]")
            print(f"    J:      μ={m(j):.4f} σ={s(j):.4f} [{min(j):.3f}, {max(j):.3f}]")
            print(f"    J(4+F): μ={m(jf):.4f} σ={s(jf):.4f} [{min(jf):.3f}, {max(jf):.3f}]")
            print(f"    J(4+P): μ={m(jp):.4f} σ={s(jp):.4f} [{min(jp):.3f}, {max(jp):.3f}]")
            print(f"    J(4+FP):μ={m(jfp):.4f} σ={s(jfp):.4f} [{min(jfp):.3f}, {max(jfp):.3f}]")
            print(f"    F_gamma: μ={m(fg):.4f} σ={s(fg):.4f}  P_process: μ={m(pp):.4f} σ={s(pp):.4f}")

        group_stats("SAFE", safe)
        group_stats("UNSAFE", unsafe)

        # Group by call count
        print("\n── 按工具调用数分组 ──")
        by_calls = {}
        for r in results:
            n = r.n_tool_calls
            by_calls.setdefault(n, []).append(r)
        for n in sorted(by_calls.keys()):
            grp = by_calls[n]
            j = [r.j_phase1 for r in grp]
            fg = [r.f_gamma for r in grp]
            pp = [r.p_process for r in grp]
            re = [r.r_efficiency for r in grp]
            m = lambda xs: sum(xs)/len(xs)
            print(f"  {n} calls (n={len(grp)}): "
                  f"J={m(j):+.3f} F_gamma={m(fg):+.3f} P_process={m(pp):+.3f} R_eff={m(re):+.4f}")

        # Group by domain
        print("\n── 按 Domain 分组 ──")
        for dom in ["calendar", "shopping"]:
            grp = [r for r in results if r.config.domain == dom]
            if not grp:
                continue
            j = [r.j_phase1 for r in grp]
            rt = [r.r_task for r in grp]
            m = lambda xs: sum(xs)/len(xs)
            print(f"  {dom} (n={len(grp)}): J={m(j):+.3f} R_task={m(rt):+.3f}")

        # Summary statistics
        print(f"\n{'='*110}")
        print("汇总统计")
        print(f"{'='*110}")
        for k, v in stats.items():
            if isinstance(v, float):
                print(f"  {k:<30s}: {v:.4f}")
            else:
                print(f"  {k:<30s}: {v}")

        # Key findings
        print(f"\n{'='*110}")
        print("关键发现")
        print(f"{'='*110}")

        findings = []

        # 1. C_safety enforcement
        if stats["n_unsafe"] > 0 and stats["mean_j_drop_unsafe"] > 0.01:
            findings.append(
                f"1. 安全惩罚有效：unsafe 轨迹 J 平均下降 {stats['mean_j_drop_unsafe']:.2f}，"
                f"共 {stats['n_unsafe']}/{stats['n_scenarios']} 个场景被标记"
            )
        elif stats["n_unsafe"] == 0:
            findings.append("1. ⚠ 未检测到 unsafe 轨迹——安全检测可能不够灵敏")

        # 2. J distribution
        if stats["std_j"] < 0.01:
            findings.append(f"2. ⚠ J 标准差极小 ({stats['std_j']:.4f})——所有场景的 J 几乎相同，训练会出现 saturation")
        elif stats["unique_j"] <= 3:
            findings.append(f"2. J 区分度偏低：仅 {stats['unique_j']} 个唯一值（σ={stats['std_j']:.4f}），组内方差可能不足")
        else:
            findings.append(f"2. J 区分度: {stats['unique_j']}/{stats['n_scenarios']} 个唯一值，σ={stats['std_j']:.4f}")

        # 3. F_gamma effect
        if stats["j_range_widen_f"] > 0.01:
            findings.append(f"3. F_gamma 展宽 J 范围 +{stats['j_range_widen_f']:.0%}——提供额外信息")
        else:
            findings.append(f"3. F_gamma 几乎不改变 J 范围 ({stats['j_range_widen_f']:+.1%})——与 R_task 共线")

        # 4. P_process effect
        if stats["j_range_widen_p"] > 0.01:
            findings.append(f"4. P_process 展宽 J 范围 +{stats['j_range_widen_p']:.0%}——有 anti-saturation 效果")
        else:
            findings.append(f"4. P_process 几乎不改变 J 范围 ({stats['j_range_widen_p']:+.1%})——方差贡献微弱，符合预期")

        # 5. Efficiency penalty
        low_eff = [r for r in results if r.r_efficiency < -0.05]
        if low_eff:
            findings.append(f"5. 效率惩罚生效：{len(low_eff)} 个场景 R_efficiency < -0.05（冗余调用受罚）")

        # 6. Shopping vs Calendar
        cal = [r for r in results if r.config.domain == "calendar"]
        shop = [r for r in results if r.config.domain == "shopping"]
        if cal and shop:
            mj_cal = sum(r.j_phase1 for r in cal) / len(cal)
            mj_shop = sum(r.j_phase1 for r in shop) / len(shop)
            findings.append(f"6. 跨域一致性: J_cal={mj_cal:.3f} J_shop={mj_shop:.3f} (Δ={mj_cal-mj_shop:+.3f})")

        for f in findings:
            print(f"  {f}")

    finally:
        ctx.stop()
        print(f"\n  环境关闭")


if __name__ == "__main__":
    main()
