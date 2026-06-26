"""OvalMCPWorkerContext — 每个 verl rollout worker 的 live MCP + audit 上下文。

职责：
  1. 管理 LiveMCPManager + LiveMCPExecutor + AuditWrapper 生命周期
  2. 为每个 rollout 创建隔离 session
  3. 执行真实 MCP tool call 并产生审计事件
  4. 提供 reset/cleanup 保证 session 隔离

设计约束：
  - 每个 worker 进程一个 context 实例
  - 不依赖 verl 特定 API，可独立测试
  - 线程安全（每个 rollout 使用独立 session）
"""

from __future__ import annotations

import json
from typing import Any

from src.live_mcp.config import load_suite_config
from src.live_mcp.executor import LiveMCPExecutor
from src.live_mcp.manager import LiveMCPManager
from src.live_mcp.types import ToolCall, ToolExecutionResult
from src.oval_mcp.envs.audit_wrapper import AuditWrapper
from src.oval_mcp.envs.domain_adapter import get_adapter
from src.oval_mcp.verifier.events import AuditEvent


class OvalMCPWorkerContext:
    """每个 verl rollout worker 的 live MCP + audit 上下文。

    Usage:
        ctx = OvalMCPWorkerContext(suite_path="configs/live_mcp/suite_mvp.yaml")
        ctx.start()
        try:
            session = ctx.create_session(seed=42)
            event = ctx.execute(session_id, tool_call, domain="calendar")
        finally:
            ctx.stop()
    """

    def __init__(
        self,
        suite_path: str = "configs/live_mcp/suite_mvp.yaml",
        domains: list[str] | None = None,
    ):
        self.suite_config = load_suite_config(suite_path)
        self.domains = domains or [
            "calendar", "shopping", "banking", "email", "filesystem",
            "payments", "crm", "issue_tracker", "team_chat", "food_delivery",
        ]
        self.manager = LiveMCPManager(self.suite_config)
        self.executor: LiveMCPExecutor | None = None
        self._audit_wrappers: dict[str, AuditWrapper] = {}
        self._started = False

    def start(self) -> None:
        """启动 MCP server subprocess。"""
        if self._started:
            return
        self.manager.start_suite()
        self.executor = LiveMCPExecutor(self.manager, self.manager.registry)
        # 初始化 domain adapters + audit wrappers
        for domain in self.domains:
            if domain in self.manager.server_names:
                adapter = get_adapter(domain)
                self._audit_wrappers[domain] = AuditWrapper(
                    self.executor,
                    self.manager,
                    adapter=adapter,
                    domain_name=domain,
                )
        self._started = True

    def stop(self) -> None:
        """停止 MCP server subprocess。"""
        if not self._started:
            return
        self.manager.stop_suite()
        self.executor = None
        self._audit_wrappers.clear()
        self._started = False

    def create_session(self, seed: int = 42) -> str:
        """创建隔离 session，返回 session_id。"""
        session = self.manager.create_session(seed=seed)
        self.manager.discover_tools(session.session_id)
        return session.session_id

    def close_session(self, session_id: str) -> None:
        """关闭 session。"""
        self.manager.close_session(session_id)

    def reset_session(self, session_id: str, seed: int) -> None:
        """重置 session 状态。"""
        self.manager.reset_session(session_id, seed=seed)

    def execute_with_audit(
        self,
        session_id: str,
        domain: str,
        tool_call: ToolCall,
        model_output: str = "",
    ) -> tuple[AuditEvent, ToolExecutionResult]:
        """执行一次 tool_call 并产生审计事件。

        Captures pre_state before execution and post_state after,
        to enable accurate state-diff and cross-event pattern detection.
        """
        audit = self._audit_wrappers.get(domain)
        if audit is None:
            raise ValueError(f"no audit wrapper for domain: {domain}")

        # Capture pre-state BEFORE execution (required for state diff)
        pre_state = None
        try:
            pre_state = self.manager.get_state(session_id)
        except Exception:
            pass

        # Execute via LiveMCPExecutor
        exec_result = self.executor.execute(session_id, tool_call)

        # Capture post-state AFTER execution
        post_state = None
        try:
            post_state = self.manager.get_state(session_id)
        except Exception:
            pass

        # Record event with proper pre/post state
        event = audit.audit_step_with_state(
            session_id=session_id,
            action_type="tool_call",
            tool_calls=[tool_call],
            execution_results=[exec_result],
            model_output=model_output,
            pre_state=pre_state,
            post_state=post_state,
        )

        return event, exec_result

    def execute_terminal_with_audit(
        self,
        session_id: str,
        domain: str,
        action_type: str,
        model_output: str = "",
    ) -> AuditEvent:
        """记录终止动作的审计事件。

        Args:
            session_id: 当前 session
            domain: MCP server 名称
            action_type: final_answer / report_error / ask_clarification
            model_output: 模型原始输出文本

        Returns:
            AuditEvent
        """
        audit = self._audit_wrappers.get(domain)
        if audit is None:
            raise ValueError(f"no audit wrapper for domain: {domain}")

        return audit.audit_step(
            session_id=session_id,
            action_type=action_type,
            tool_calls=[],
            execution_results=[],
            model_output=model_output,
        )

    def get_tool_schemas(self, domain: str) -> list[dict[str, Any]]:
        """获取 domain 的工具 schema 列表（用于构建 prompt）。"""
        return self.manager.registry.server_tools(domain)

    def serialize_audit_events(self, events: list[AuditEvent]) -> list[dict[str, Any]]:
        """序列化 AuditEvent 列表为 JSON-safe dict 列表。

        用于传递给 verl reward function（通过 extra_fields）。
        """
        return [event.to_dict() if hasattr(event, "to_dict") else _event_to_dict(event) for event in events]


def _event_to_dict(event: AuditEvent) -> dict[str, Any]:
    """将 AuditEvent 转为可序列化 dict（fallback，主路径走 event.to_dict()）。"""
    return {
        "step": event.step,
        "operation": event.operation,
        "target_type": event.target_type,
        "target_id": event.target_id,
        "tool_name": event.tool_name,
        "execution_success": event.execution_success,
        "error_message": event.error_message,
        "session_id": event.session_id,
    }
