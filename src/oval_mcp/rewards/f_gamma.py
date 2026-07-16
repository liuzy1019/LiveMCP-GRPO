"""F_gamma: Potential-based progress shaping.

OVAL-MCP §8.

progress potential:
  Phi(m_t) = completed_required_states(m_t) / total_required_states

per-turn shaping:
  F_u = gamma * Phi(m_{u+1}) - Phi(m_u)

trajectory shaping:
  F_gamma(tau) = sum_u gamma^u * (gamma * Phi(m_{u+1}) - Phi(m_u))

gamma=1 时的 telescoping 性质:
  F_gamma(tau) = Phi(m_T) - Phi(m_0) = completed/total at final state
  trajectory-level F_gamma 只区分不同终点的轨迹，不提供 step-level 区分。
  step-level 区分由 LATA + F_u 承担。

absorbing failure state 的 Phi:
  Phi(absorbing_failure) = Phi(m_{T-1})  # 继承失败前的 progress
  理由: safety failure 的惩罚由 C_safety 承担，shaping 不重复计数。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from src.oval_mcp.verifier.events import EventLog

# ---------------------------------------------------------------------------
# progress predicate registry (domain-agnostic keys)
# ---------------------------------------------------------------------------
PROGRESS_PREDICATE_NAMES: list[str] = [
    "resolved_required_entity",
    "satisfied_dependency_edge",
    "completed_required_transition",
    "verified_postcondition",
    "produced_required_response",
]


@dataclass
class FGammaResult:
    """F_gamma decomposition for one trajectory."""

    f_gamma: float = 0.0
    phi_initial: float = 0.0
    phi_final: float = 0.0
    per_turn_f: list[float] = field(default_factory=list)
    total_required_states: int = 0
    completed_required_states: int = 0


class ProgressTracker:
    """Track required predicate completion across a trajectory.

    Uses DomainAdapter.evaluate_event() to determine which predicates
    each event satisfies — a single source of truth shared with
    R_coverage and P_process.
    """

    def __init__(self, required_predicates: Optional[list[str]] = None):
        self._required = set(required_predicates or PROGRESS_PREDICATE_NAMES)

    def compute(
        self,
        event_log: EventLog,
        task: dict,
        gamma: float = 1.0,
        domain_adapter: Any = None,
    ) -> FGammaResult:
        """Compute F_gamma from trajectory event log and task definition.

        The canonical ten-domain reward path requires a DomainAdapter.
        """
        result = FGammaResult()
        if domain_adapter is None:
            raise ValueError("ProgressTracker requires a DomainAdapter")

        task_predicates = self._get_task_progress_predicates(task)
        total = len(task_predicates)
        result.total_required_states = total
        if total == 0:
            return result

        result.phi_initial = 0.0
        completed: set[str] = set()
        per_turn_f: list[float] = []

        for event in list(event_log.tool_call_events) + list(event_log.terminal_events):
            prev_phi = len(completed) / total

            new_predicates = self._eval_event(event, task, domain_adapter)
            completed.update(new_predicates)

            curr_phi = len(completed) / total
            f_u = gamma * curr_phi - prev_phi
            per_turn_f.append(f_u)

        result.per_turn_f = per_turn_f
        result.completed_required_states = len(completed)
        result.phi_final = len(completed) / total

        if gamma == 1.0:
            result.f_gamma = result.phi_final - result.phi_initial
        else:
            result.f_gamma = sum(
                (gamma ** u_idx) * fu
                for u_idx, fu in enumerate(per_turn_f)
            )

        return result

    def _get_task_progress_predicates(self, task: dict) -> list[str]:
        custom = task.get("progress_predicates")
        if custom and isinstance(custom, list):
            predicates: list[str] = []
            seen: set[str] = set()
            for item in custom:
                if isinstance(item, str):
                    name = item
                elif isinstance(item, dict):
                    name = str(item.get("type", ""))
                else:
                    continue
                if name in self._required and name not in seen:
                    predicates.append(name)
                    seen.add(name)
            return predicates
        return list(self._required)

    def _eval_event(self, event, task: dict, domain_adapter: Any) -> frozenset[str]:
        """Return predicates satisfied by *event*.

        Adapter errors are reward-integrity failures and must propagate.
        """
        return domain_adapter.evaluate_event(event, task)


__all__ = [
    "FGammaResult",
    "ProgressTracker",
    "PROGRESS_PREDICATE_NAMES",
]
