"""OVAL-MCP reward layer.

R_task (§7.1), safety cost (§7.2), scalar return (§9),
F_gamma potential shaping (§8), P_process process score (§7.3).
"""

from src.oval_mcp.rewards.task_reward import TaskReward, TaskRewardResult, DEFAULT_WEIGHTS
from src.oval_mcp.rewards.scalar_return import ScalarReturn, ScalarReturnResult
from src.oval_mcp.rewards.f_gamma import FGammaResult, ProgressTracker, PROGRESS_PREDICATE_NAMES
from src.oval_mcp.rewards.p_process import (
    BONUS_PREDICATES,
    FORBIDDEN_PEN_NAMES,
    PENALTY_PREDICATES,
    ProcessScoreResult,
    ProcessScorer,
    StepProcessScore,
)

__all__ = [
    "TaskReward",
    "TaskRewardResult",
    "DEFAULT_WEIGHTS",
    "ScalarReturn",
    "ScalarReturnResult",
    "FGammaResult",
    "ProgressTracker",
    "PROGRESS_PREDICATE_NAMES",
    "BONUS_PREDICATES",
    "FORBIDDEN_PEN_NAMES",
    "PENALTY_PREDICATES",
    "ProcessScoreResult",
    "ProcessScorer",
    "StepProcessScore",
]
