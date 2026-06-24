"""OVAL-MCP: Event-Verified Constrained GRPO for Long-Horizon MCP Tool Use.

Phase 1: live execution + event-sourced safety + constrained GRPO
"""

from src.oval_mcp.verifier.events import AuditEvent, EventLog, TrajectoryEventLog
from src.oval_mcp.verifier.safety import SafetyVerifier
from src.oval_mcp.rewards.task_reward import TaskReward
from src.oval_mcp.rewards.scalar_return import ScalarReturn

__all__ = [
    "AuditEvent",
    "EventLog",
    "TrajectoryEventLog",
    "SafetyVerifier",
    "TaskReward",
    "ScalarReturn",
]
