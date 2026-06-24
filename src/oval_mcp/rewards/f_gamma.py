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
  step-level 区分由 Phase 3 LATA + F_u 承担。

absorbing failure state 的 Phi:
  Phi(absorbing_failure) = Phi(m_{T-1})  # 继承失败前的 progress
  理由: safety failure 的惩罚由 C_safety 承担，shaping 不重复计数。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

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
class ProgressState:
    """Snapshot of which required progress predicates have been completed so far."""

    completed: set[str] = field(default_factory=set)
    total: int = 0

    @property
    def phi(self) -> float:
        """Phi(m_t) ∈ [0, 1]."""
        if self.total == 0:
            return 0.0
        return len(self.completed) / self.total


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

    Reads required_states from the total count of progress predicates,
    then inspects event_log per-turn to determine which predicates
    have been satisfied at each step.

    Phase 1/2 simplified: detect predicate completion by operation
    type coverage (create/update/delete/query/terminal) and
    execution_success flags.  Full predicate semantics require
    DomainAdapter integration (Phase 3).
    """

    def __init__(self, required_predicates: Optional[list[str]] = None):
        self._required = set(required_predicates or PROGRESS_PREDICATE_NAMES)

    def compute(
        self,
        event_log: EventLog,
        task: dict,
        gamma: float = 1.0,
    ) -> FGammaResult:
        """Compute F_gamma from trajectory event log and task definition.

        Simplified Phase 1/2: uses a coarse operation-based heuristic
        to estimate predicate completion at each step.

        Phase 3: replace with DomainAdapter.predicate_evaluator.
        """
        result = FGammaResult()

        # determine which predicates are required for this task
        task_predicates = self._get_task_progress_predicates(task)
        total = len(task_predicates)
        result.total_required_states = total
        if total == 0:
            return result

        result.phi_initial = 0.0
        completed: set[str] = set()
        per_turn_f: list[float] = []

        tools = event_log.tool_call_events
        for step_idx, event in enumerate(tools, start=1):
            prev_phi = len(completed) / total

            # coarse heuristic: each successful operation maps to a predicate
            self._advance_predicates(completed, event, task_predicates, total)

            curr_phi = len(completed) / total

            # F_u = gamma * Phi(m_u) - Phi(m_{u-1})
            f_u = gamma * curr_phi - prev_phi
            per_turn_f.append(f_u)

        # terminal events may also satisfy predicates
        for event in event_log.terminal_events:
            prev_phi = len(completed) / total
            self._advance_predicates(completed, event, task_predicates, total)
            curr_phi = len(completed) / total
            f_u = gamma * curr_phi - prev_phi
            per_turn_f.append(f_u)

        result.per_turn_f = per_turn_f
        result.completed_required_states = len(completed)
        result.phi_final = len(completed) / total

        # F_gamma = sum of per-turn F_u (telescoped with discount)
        if gamma == 1.0:
            result.f_gamma = result.phi_final - result.phi_initial
        else:
            result.f_gamma = sum(
                (gamma ** u_idx) * fu
                for u_idx, fu in enumerate(per_turn_f)
            )

        return result

    def _get_task_progress_predicates(self, task: dict) -> list[str]:
        """Extract task-specific progress predicate set.

        Falls back to all PROGRESS_PREDICATE_NAMES if task doesn't specify.
        """
        custom = task.get("progress_predicates")
        if custom and isinstance(custom, list):
            return [p for p in custom if p in self._required]
        # default: all registered predicates
        return list(self._required)

    def _advance_predicates(
        self,
        completed: set[str],
        event,
        task_predicates: list[str],
        total: int,
    ) -> None:
        """Coarse heuristic: map event operation → predicate completion.

        Phase 3 TODO: replace with DomainAdapter.predicate_evaluator(event, state).
        """
        if not event.execution_success:
            return

        op = event.operation
        if op in ("query", "list_events", "search_products", "get_event", "get_order"):
            completed.add("resolved_required_entity")
        if op in ("create", "update", "delete", "add_to_cart", "remove_from_cart", "checkout"):
            completed.add("completed_required_transition")
            completed.add("resolved_required_entity")
        if op == "terminal" and event.action_type == "final_answer":
            completed.add("verified_postcondition")
            completed.add("produced_required_response")
        if event.state_changed:
            completed.add("satisfied_dependency_edge")


__all__ = [
    "FGammaResult",
    "ProgressState",
    "ProgressTracker",
    "PROGRESS_PREDICATE_NAMES",
]
