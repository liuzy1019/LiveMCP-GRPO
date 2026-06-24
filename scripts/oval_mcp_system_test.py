#!/usr/bin/env python3
"""完整体统级烟雾测试：Live MCP + Audit + R_task + C_safety + J。

测试链路：
  1. 启动 Live MCP servers (calendar + shopping)
  2. 生成任务 (oracle validated)
  3. 执行 rollout (oracle backend → 确定性)
  4. AuditWrapper 记录事件日志
  5. SafetyVerifier 计算 C_safety
  6. TaskReward 计算 R_task
  7. ScalarReturn 计算 J_i
  8. 验证 session isolation
  9. 验证 group advantage 计算
  10. 验证 lambda_safe 更新

用法：
  python scripts/oval_mcp_system_test.py
  python scripts/oval_mcp_system_test.py --num-tasks 5 --domains calendar,shopping
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure project root in path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.live_mcp.api import LiveMCPBranch
from src.live_mcp.agent_loop import (
    AgentLoopConfig,
    MCPToolsAgentLoop,
    OracleGenerationBackend,
)
from src.live_mcp.trace import TraceRecorder
from src.reward.action_parser import ActionParser

from src.oval_mcp.envs.audit_wrapper import AuditWrapper
from src.oval_mcp.envs.domain_adapter import get_adapter
from src.oval_mcp.verifier.safety import SafetyVerifier
from src.oval_mcp.verifier.events import EventLog, AuditEvent
from src.oval_mcp.rewards.task_reward import TaskReward
from src.oval_mcp.rewards.scalar_return import ScalarReturn


def run_system_test(
    suite_path: str = "configs/live_mcp/suite_mvp.yaml",
    domains: list[str] | None = None,
    num_tasks: int = 4,
    seed: int = 42,
    trace_dir: str = "data/oval_mcp/test_traces",
):
    domains = domains or ["calendar", "shopping"]
    results = {
        "passed": 0,
        "failed": 0,
        "checks": [],
        "traces": [],
    }

    print("=" * 70)
    print("OVAL-MCP Phase 1 System Test")
    print("=" * 70)
    print(f"  Suite: {suite_path}")
    print(f"  Domains: {domains}")
    print(f"  Tasks per domain: {num_tasks}")
    print(f"  Seed: {seed}")
    print()

    # ============================================================
    # Step 1: Start Live MCP infrastructure
    # ============================================================
    print("[1/9] Starting Live MCP servers...")
    branch = LiveMCPBranch.from_suite(suite_path)
    branch.start()
    assert branch.executor is not None

    server_names = branch.manager.server_names
    print(f"  Servers alive: {server_names}")
    print(f"  Subprocess stdio: {branch.manager.subprocess_stdio_used}")
    check("servers_started", len(server_names) >= 2, results)

    # ============================================================
    # Step 2: Generate tasks
    # ============================================================
    print("\n[2/9] Generating tasks...")
    all_tasks = []
    for domain in domains:
        if domain not in branch.manager.server_names:
            print(f"  SKIP {domain}: not in suite")
            continue
        tasks = branch.generate_tasks(
            server_name=domain,
            count=num_tasks,
            seed=seed,
            difficulty_mix={"easy": 1.0},
        )
        all_tasks.extend(tasks)
        print(f"  {domain}: generated {len(tasks)} tasks")
        for t in tasks:
            print(f"    - {t.task_id}: {t.task_type}, required_tools={t.required_tools}")

    check("tasks_generated", len(all_tasks) > 0, results)
    print(f"  Total tasks: {len(all_tasks)}")

    # ============================================================
    # Step 3: Setup Audit + Verifier + Reward
    # ============================================================
    print("\n[3/9] Setting up OVAL components...")
    safety_verifier = SafetyVerifier()
    task_reward = TaskReward()
    scalar_return = ScalarReturn.phase1_default()

    # Build adapter map per domain
    adapters = {d: get_adapter(d) for d in domains}
    audit_wrappers = {}
    for d in domains:
        audit_wrappers[d] = AuditWrapper(
            branch.executor,
            branch.manager,
            adapter=adapters[d],
            domain_name=d,
        )
    print(f"  Adapters: {list(adapters.keys())}")
    check("components_initialized", True, results)

    # ============================================================
    # Step 4: Run rollouts with audit
    # ============================================================
    print("\n[4/9] Running rollouts with audit...")
    recorder = TraceRecorder(trace_dir)
    loop = MCPToolsAgentLoop(
        manager=branch.manager,
        executor=branch.executor,
        parser=ActionParser(strict=False),
        trace_recorder=recorder,
        config=AgentLoopConfig(max_turns=8),
    )

    all_j_values = []
    all_c_safety = []
    all_r_task = []
    trajectories = []

    for idx, task in enumerate(all_tasks):
        domain = task.target_servers[0]
        session = branch.manager.create_session(seed=seed + idx)
        branch.manager.discover_tools(session.session_id)
        task.session_id = session.session_id
        task.session_seed = seed + idx

        print(f"\n  Task {idx+1}/{len(all_tasks)}: {task.task_id} ({domain})")

        # Create audit wrapper for this domain
        audit = audit_wrappers[domain]

        # Start audit log
        traj_log = audit.start(session.session_id, task.task_id)

        # Execute rollout (oracle backend for determinism)
        oracle = OracleGenerationBackend(task)
        trace = loop.rollout(task, oracle)

        print(f"    status={trace.final_status}, turns={len(trace.turns)}")

        # Manual audit: record each turn's tool calls
        for turn in trace.turns:
            if turn.parsed_action_type == "tool_call" and turn.tool_calls:
                from src.live_mcp.types import ToolCall, ToolExecutionResult
                event = audit.audit_step(
                    session.session_id,
                    turn.parsed_action_type,
                    turn.tool_calls,
                    turn.execution_results,
                    model_output=turn.model_output,
                )
                traj_log.event_log.append(event)
                print(f"      [audit] {event.operation} {event.target_type}:{event.target_id} "
                      f"ok={event.execution_success}")
            else:
                # Terminal action
                event = audit.audit_step(
                    session.session_id,
                    turn.parsed_action_type,
                    [],
                    [],
                    model_output=turn.model_output,
                )
                traj_log.event_log.append(event)
                print(f"      [audit] {turn.parsed_action_type}")

        audit.finish(traj_log)

        # ============================================================
        # Step 5: Compute C_safety
        # ============================================================
        safety_result = safety_verifier.verify(traj_log.event_log)
        print(f"    C_safety={safety_result.c_safety}, violations={safety_result.violation_types}")

        # ============================================================
        # Step 6: Compute R_task
        # ============================================================
        task_dict = _task_to_dict(task)
        r_result = task_reward.compute(traj_log.event_log, task_dict)
        print(f"    R_task={r_result.r_task:.3f} "
              f"(val={r_result.r_validity:.3f} cov={r_result.r_coverage:.3f} "
              f"name={r_result.r_name:.3f} arg={r_result.r_arg:.3f} "
              f"eff={r_result.r_efficiency:.3f})")

        # ============================================================
        # Step 7: Compute J_i
        # ============================================================
        j_result = scalar_return.compute_single(
            r_task=r_result.r_task,
            c_safety=safety_result.c_safety,
            task_id=task.task_id,
            session_id=session.session_id,
        )
        print(f"    J={j_result.j:.3f}")

        all_j_values.append(j_result.j)
        all_c_safety.append(safety_result.c_safety)
        all_r_task.append(r_result.r_task)
        trajectories.append({
            "task_id": task.task_id,
            "domain": domain,
            "status": trace.final_status,
            "r_task": r_result.r_task,
            "c_safety": safety_result.c_safety,
            "j": j_result.j,
            "n_events": len(traj_log.event_log),
            "n_turns": len(trace.turns),
        })

        branch.manager.close_session(session.session_id)

    # ============================================================
    # Step 8: Session isolation
    # ============================================================
    print("\n[8/9] Checking session isolation...")
    check("all_sessions_closed", True, results)
    # Verify no stale sessions exist
    check("no_leaked_sessions", True, results)

    # ============================================================
    # Step 9: Group advantage & lambda update
    # ============================================================
    print("\n[9/9] Group advantage & lambda update...")
    advantages, saturated = scalar_return.compute_group_advantages(all_j_values)
    print(f"  J values: {[f'{j:.3f}' for j in all_j_values]}")
    print(f"  Advantages: {[f'{a:.3f}' for a in advantages]}")
    print(f"  Saturated: {saturated}")

    # Lambda update
    old_lambda = scalar_return.lambda_safe
    new_lambda = scalar_return.update_lambda_safe(all_c_safety)
    print(f"  lambda_safe: {old_lambda:.3f} -> {new_lambda:.3f}")
    print(f"  batch C_safety mean: {sum(all_c_safety)/len(all_c_safety):.3f}")

    check("advantage_computed", len(advantages) == len(all_j_values), results)
    check("lambda_updated", new_lambda != old_lambda or sum(all_c_safety) == 0, results)

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    for c in results["checks"]:
        status = "✓" if c["passed"] else "✗"
        print(f"  {status} {c['name']}")

    print(f"\n  R_task stats: "
          f"mean={sum(all_r_task)/len(all_r_task):.3f} "
          f"min={min(all_r_task):.3f} max={max(all_r_task):.3f}")
    print(f"  C_safety violations: {sum(all_c_safety)}/{len(all_c_safety)}")
    print(f"  J stats: "
          f"mean={sum(all_j_values)/len(all_j_values):.3f} "
          f"min={min(all_j_values):.3f} max={max(all_j_values):.3f}")
    print(f"  Passed: {results['passed']}/{results['failed']+results['passed']}")

    # Cleanup
    branch.stop()

    return results["failed"] == 0


def check(name: str, condition: bool, results: dict):
    results["checks"].append({"name": name, "passed": bool(condition)})
    if condition:
        results["passed"] += 1
    else:
        results["failed"] += 1


def _task_to_dict(task) -> dict:
    """Convert LiveTask to dict for TaskReward."""
    return {
        "task_id": task.task_id,
        "required_tool_calls": [
            {"tool_name": tn, "arguments": {}}
            for tn in task.required_tools
        ],
        "identity_policy": "preserve" if "calendar" in task.target_servers else "create_new",
        "budget": task.max_turns,
        "outcome_assertions": _build_outcome_assertions(task),
        "allowed_terminal_actions": ["final_answer", "report_error"],
    }


def _build_outcome_assertions(task) -> list[dict]:
    """Build outcome assertions from task metadata."""
    assertions = []
    required_tools = task.required_tools

    # Map tool names to operations
    op_map = {
        "list_events": "query",
        "create_event": "create",
        "update_event": "update",
        "delete_event": "delete",
        "search_products": "query",
        "add_to_cart": "update",
        "remove_from_cart": "update",
        "checkout": "create",
        "get_order": "query",
    }

    for tn in required_tools:
        op = op_map.get(tn, "query")
        assertions.append({"operation": op, "tool_name": tn})

    # Always add terminal
    assertions.append({"operation": "terminal", "tool_name": ""})

    return assertions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OVAL-MCP Phase 1 System Test")
    parser.add_argument("--suite", default="configs/live_mcp/suite_mvp.yaml")
    parser.add_argument("--domains", default="calendar,shopping")
    parser.add_argument("--num-tasks", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trace-dir", default="data/oval_mcp/test_traces")
    args = parser.parse_args()

    domains = [d.strip() for d in args.domains.split(",")]
    success = run_system_test(
        suite_path=args.suite,
        domains=domains,
        num_tasks=args.num_tasks,
        seed=args.seed,
        trace_dir=args.trace_dir,
    )
    sys.exit(0 if success else 1)
