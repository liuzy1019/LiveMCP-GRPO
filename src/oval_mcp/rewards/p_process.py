"""P_process: Bounded process score per OVAL-MCP §7.3.

Per-step score:
  B_t = sum(triggered bonus values)    # > 0
  N_t = sum(triggered penalty values)   # <= 0
  p_t_raw = B_t + N_t

  if step triggers forbidden event:
    N_t_forbidden = sum of PEN values mapped to forbidden events
    p_t = min(p_t_raw, -abs(N_t_forbidden))
  else:
    p_t = p_t_raw

Trajectory score:
  P_process(tau) = clip(sum_t p_t, -p_max, p_max)

默认超参:
  p_max = 0.3
  lambda_process = 0.3
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from src.oval_mcp.verifier.events import EventLog

# ── bonus predicates (positive values) ────────────────────────────────
BONUS_PREDICATES: dict[str, float] = {
    "B_resolve_required_entity":       0.05,
    "B_satisfy_dependency_edge":       0.05,
    "B_preserve_required_identity":    0.05,
    "B_complete_required_transition":  0.08,
    "B_verify_postcondition":          0.04,
    "B_recover_from_tool_error":       0.04,
}

# ── penalty predicates (negative values) ──────────────────────────────
PENALTY_PREDICATES: dict[str, float] = {
    "PEN_redundant_no_progress_action":  -0.03,
    "PEN_unresolved_entity_action":      -0.05,
    "PEN_wrong_resource_action":         -0.05,
    "PEN_forbidden_transition_attempt":  -0.08,
    "PEN_missing_required_response":     -0.05,
    "PEN_invalid_tool_schema":           -0.05,
}

# PEN predicates that map to forbidden events (§7.3 mapping table)
# Only these count toward N_t_forbidden
FORBIDDEN_PEN_NAMES: set[str] = {
    "PEN_forbidden_transition_attempt",
}


@dataclass
class StepProcessScore:
    """Per-step process score decomposition."""

    step: int = 0
    bonus_total: float = 0.0
    penalty_total: float = 0.0
    p_raw: float = 0.0
    p_clamped: float = 0.0
    triggered_bonuses: list[str] = field(default_factory=list)
    triggered_penalties: list[str] = field(default_factory=list)
    forbidden_penalty_sum: float = 0.0


@dataclass
class ProcessScoreResult:
    """P_process result for a trajectory."""

    p_process: float = 0.0
    per_step: list[StepProcessScore] = field(default_factory=list)
    total_bonus: float = 0.0
    total_penalty: float = 0.0
    n_steps: int = 0
    n_forbidden_steps: int = 0


FALLBACK_WARNED = False  # module-level flag for one-time warning


class ProcessScorer:
    """Compute per-step process scores and trajectory P_process.

    Uses DomainAdapter.evaluate_event() for predicate satisfaction;
    falls back to generic heuristics when no adapter is available.
    """

    def __init__(
        self,
        p_max: float = 0.3,
        bonus_map: Optional[dict[str, float]] = None,
        penalty_map: Optional[dict[str, float]] = None,
    ):
        self.p_max = p_max
        self._bonus = dict(bonus_map or BONUS_PREDICATES)
        self._penalty = dict(penalty_map or PENALTY_PREDICATES)

    def compute(
        self,
        event_log: EventLog,
        task: Optional[dict[str, Any]] = None,
        domain_adapter: Any = None,
    ) -> ProcessScoreResult:
        """Compute P_process for a complete trajectory."""
        result = ProcessScoreResult()

        tools = event_log.tool_call_events
        if not tools:
            return result

        p_sum = 0.0
        for idx, event in enumerate(tools, start=1):
            score = self._score_step(event, idx, task, domain_adapter)
            result.per_step.append(score)
            p_sum += score.p_clamped
            if score.forbidden_penalty_sum < 0:
                result.n_forbidden_steps += 1
            result.total_bonus += score.bonus_total
            result.total_penalty += score.penalty_total

        result.n_steps = len(tools)
        result.p_process = max(-self.p_max, min(self.p_max, p_sum))
        return result

    # Mapping from predicate names to bonus keys
    _PREDICATE_BONUS_MAP: dict[str, str] = {
        "resolved_required_entity": "B_resolve_required_entity",
        "satisfied_dependency_edge": "B_satisfy_dependency_edge",
        "completed_required_transition": "B_complete_required_transition",
        "verified_postcondition": "B_verify_postcondition",
        "produced_required_response": "B_recover_from_tool_error",
    }

    def _score_step(
        self,
        event,
        step_index: int,
        task: Optional[dict[str, Any]] = None,
        domain_adapter: Any = None,
    ) -> StepProcessScore:
        """Compute p_t for a single tool_call event.

        Uses DomainAdapter.evaluate_event() for predicate satisfaction;
        falls back to generic operation-based heuristics.
        """
        score = StepProcessScore(step=step_index)

        triggered_bonus: list[str] = []
        triggered_penalty: list[str] = []

        # ── bonus detection via DomainAdapter ──
        satisfied: frozenset[str] = frozenset()
        if domain_adapter is not None and task is not None:
            try:
                satisfied = domain_adapter.evaluate_event(event, task)
            except Exception:
                pass

        if satisfied:
            for pred, bonus_key in self._PREDICATE_BONUS_MAP.items():
                if pred in satisfied and bonus_key in self._bonus:
                    triggered_bonus.append(bonus_key)
        else:
            # Fallback: generic heuristics
            if event.execution_success and event.state_changed:
                triggered_bonus.append("B_complete_required_transition")
            if event.execution_success and event.operation == "query":
                triggered_bonus.append("B_resolve_required_entity")

        # ── penalty detection (schema / execution — orthogonal to predicates) ──
        if not event.schema_valid:
            triggered_penalty.append("PEN_invalid_tool_schema")

        if not event.execution_success:
            triggered_penalty.append("PEN_wrong_resource_action")

        if event.forbidden_transition:
            triggered_penalty.append("PEN_forbidden_transition_attempt")

        # ── compute values ──
        bonus_val = sum(self._bonus.get(name, 0.0) for name in triggered_bonus)
        penalty_val = sum(self._penalty.get(name, 0.0) for name in triggered_penalty)
        p_raw = bonus_val + penalty_val
        forbidden_penalty = sum(
            self._penalty.get(name, 0.0)
            for name in triggered_penalty
            if name in FORBIDDEN_PEN_NAMES
        )

        # forbidden event clamping (§7.3)
        if forbidden_penalty < 0:
            p_clamped = min(p_raw, -abs(forbidden_penalty))
        else:
            p_clamped = p_raw

        score.bonus_total = bonus_val
        score.penalty_total = penalty_val
        score.p_raw = p_raw
        score.p_clamped = p_clamped
        score.triggered_bonuses = triggered_bonus
        score.triggered_penalties = triggered_penalty
        score.forbidden_penalty_sum = forbidden_penalty

        return score


__all__ = [
    "BONUS_PREDICATES",
    "FORBIDDEN_PEN_NAMES",
    "PENALTY_PREDICATES",
    "ProcessScoreResult",
    "ProcessScorer",
    "StepProcessScore",
]
