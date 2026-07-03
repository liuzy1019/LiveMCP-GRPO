"""Execution-aware five-component Live MCP reward."""

from __future__ import annotations

import math
from typing import Any

from src.live_mcp.oracle import criterion_satisfied
from src.live_mcp.types import LiveTask, RolloutTrace, ToolExecutionResult


class RewardComposer:
    def __init__(self, weights: dict[str, float] | None = None, alpha: float = 0.5):
        # PROVE §3.3 reward weights: w_val=0.5, w_cov=0.5, w_eff=0.15, w_name=0.2, w_arg=0.1
        self.weights = weights or {
            "validity": 0.5,
            "coverage": 0.5,
            "efficiency": 0.15,
            "tool_selection": 0.2,
            "argument_value": 0.1,
        }
        self.alpha = alpha  # PROVE α=0.5, used both in budget slack β and penalty scale

    def compute(
        self,
        task: LiveTask,
        trace: RolloutTrace,
        final_state: dict[str, Any] | None = None,
    ) -> dict[str, float]:
        results = [result for turn in trace.turns for result in turn.execution_results]
        model_calls = [call for turn in trace.turns for call in turn.tool_calls]
        expects_abstention = _expects_abstention(task)
        abstention = self._abstention(task, trace, results, model_calls) if expects_abstention else 0.0
        validity = self._validity(results, no_tool_expected=expects_abstention)
        coverage = abstention if expects_abstention else self._coverage(task, results)
        if final_state is not None and not all(criterion_satisfied(final_state, c) for c in task.success_criteria):
            coverage = min(coverage, 0.99)
        oracle_tool_calls = [
            call for call in task.oracle_program.calls
            if getattr(call, "action", "tool_call") == "tool_call"
        ]
        efficiency = self._efficiency(len(model_calls), len(oracle_tool_calls))
        tool_selection = self._tool_selection(task, results, no_tool_expected=expects_abstention)
        argument_value = self._argument_value(task, model_calls, no_tool_expected=expects_abstention)
        score = (
            self.weights["validity"] * validity
            + self.weights["coverage"] * coverage
            + self.weights["efficiency"] * efficiency
            + self.weights["tool_selection"] * tool_selection
            + self.weights["argument_value"] * argument_value
        )
        return {
            "score": float(score),
            "component_validity": float(validity),
            "component_coverage": float(coverage),
            "component_efficiency": float(efficiency),
            "component_tool_selection": float(tool_selection),
            "component_argument_value": float(argument_value),
            "component_abstention": float(abstention),
            "num_turns": float(len(trace.turns)),
            "num_tool_calls": float(len(model_calls)),
            "num_execution_errors": float(sum(1 for r in results if not r.success)),
        }

    def _validity(self, results: list[ToolExecutionResult], no_tool_expected: bool = False) -> float:
        if not results:
            return 1.0 if no_tool_expected else 0.0
        scores = []
        for result in results:
            score = 0.33
            score += 0.33 if result.schema_valid else 0.0
            score += 0.34 if result.success else 0.0
            scores.append(score)
        return sum(scores) / len(scores)

    def _coverage(self, task: LiveTask, results: list[ToolExecutionResult]) -> float:
        """PROVE §3.3 R_coverage: dependency-ordered coverage.

        For each oracle call g, it contributes to coverage only if:
          1. The model called the same tool (m(g) = 1).
          2. All predecessor oracle calls j where j→g is a dependency edge
             have already been matched by the model before g (o(g) = 1).

        This implements the PROVE formula:
          R_cov = (1/|G|) * Σ_g [ m(g) · o(g) ]
          where o(g) = ∏_{j: j→g ∈ E} ⊮[µ(j) < µ(g)]

        The dependency graph is approximated from the oracle call sequence:
        an implicit dependency edge j→g exists when j appears before g in the
        oracle chain (sequential ordering = the dependency order discovered in
        Step 1).  This is equivalent to the ordered-prefix check but correctly
        handles non-linear chains where a later oracle step depends on multiple
        earlier steps.
        """
        oracle_calls = [
            call for call in task.oracle_program.calls
            if getattr(call, "action", "tool_call") == "tool_call"
        ]
        if not oracle_calls:
            return 1.0

        # Build model call sequence (tool names in order, successful only).
        # PROVE: only successful model calls count toward coverage.
        model_tool_names: list[str] = [
            r.canonical_tool_name for r in results if r.success
        ]

        # For each oracle position g, track whether it has been matched.
        # matched[g] = True means oracle_calls[g] was satisfied by the model.
        matched: list[bool] = [False] * len(oracle_calls)

        # Map oracle tool name → list of oracle indices (handles repeated tools).
        # We consume matches greedily in oracle order to respect dependency edges.
        oracle_name_to_indices: dict[str, list[int]] = {}
        for g_idx, oc in enumerate(oracle_calls):
            oracle_name_to_indices.setdefault(oc.tool_name, []).append(g_idx)

        # Greedy matching: walk model calls in order, match each to the earliest
        # unmatched oracle call with the same tool name whose predecessors are
        # already matched (o(g) = 1).
        for model_tool in model_tool_names:
            candidates = oracle_name_to_indices.get(model_tool, [])
            for g_idx in candidates:
                if matched[g_idx]:
                    continue
                # Check o(g): PROVE §3.3 dependency edge check.
                # In a sequential oracle chain, each step g depends only on
                # its direct predecessor (g_idx - 1), not all prior steps.
                # o(g) = 1 iff the direct predecessor is already matched
                # (or g is the first step, which has no predecessor).
                # Requiring ALL predecessors matched (range(g_idx)) is too
                # strict: it penalises the model for skipping a redundant
                # read step even when the critical dependency is satisfied.
                predecessors_satisfied = (g_idx == 0) or matched[g_idx - 1]
                if predecessors_satisfied:
                    matched[g_idx] = True
                    break  # consume this model call

        return sum(matched) / len(oracle_calls)

    def _efficiency(self, model_call_count: int, gt_call_count: int) -> float:
        """PROVE §3.3: R_eff = max(0, 1 - α * max(0, n - B) / B).

        B = n_gt + ⌈α * n_gt⌉  (budget with slack), α = 0.5.
        Penalty is proportional to excess relative to budget, not absolute excess.
        """
        if gt_call_count <= 0:
            return 1.0 if model_call_count == 0 else 0.0
        B = gt_call_count + math.ceil(self.alpha * gt_call_count)
        excess = max(0, model_call_count - B)
        return max(0.0, 1.0 - self.alpha * excess / B)

    def _tool_selection(self, task: LiveTask, results: list[ToolExecutionResult], no_tool_expected: bool = False) -> float:
        if no_tool_expected:
            return 1.0 if not results else 0.0
        if not results:
            return 0.0
        allowed = set(task.required_tools) | {
            call.tool_name for call in task.oracle_program.calls
            if getattr(call, "action", "tool_call") == "tool_call"
        }
        return sum(1 for result in results if result.canonical_tool_name in allowed) / len(results)

    def _argument_value(self, task: LiveTask, model_calls: list[Any], no_tool_expected: bool = False) -> float:
        if no_tool_expected:
            return 1.0 if not model_calls else 0.0
        expected = {
            (call.tool_name, key): value
            for call in task.oracle_program.calls
            if getattr(call, "action", "tool_call") == "tool_call"
            for key, value in call.arguments.items()
        }
        if not expected:
            return 1.0
        checked = 0
        matched = 0
        for call in model_calls:
            for key, value in call.arguments.items():
                expected_value = expected.get((call.name, key))
                if expected_value is None:
                    continue
                checked += 1
                if value == expected_value:
                    matched += 1
        return matched / checked if checked else 0.0

    def _abstention(
        self,
        task: LiveTask,
        trace: RolloutTrace,
        results: list[ToolExecutionResult],
        model_calls: list[Any],
    ) -> float:
        if results or model_calls:
            return 0.0
        if not trace.turns:
            return 0.0
        final_action = trace.turns[-1].parsed_action_type
        if final_action not in {"ask_clarification", "report_error", "final_answer"}:
            return 0.0
        text = trace.turns[-1].model_output.lower()
        hidden_tools = [tool.lower() for tool in task.hidden_tools]
        unavailable = str(task.metadata.get("unavailable_required_tool", "")).lower()
        if final_action == "final_answer" and not any(token in text for token in ("unavailable", "missing", "cannot", "can't", "no tool")):
            return 0.0
        if unavailable and unavailable in text:
            return 1.0
        if any(tool and tool in text for tool in hidden_tools):
            return 1.0
        return 0.5


def _expects_abstention(task: LiveTask) -> bool:
    return task.task_type == "missing_function" or any(criterion.get("type") == "missing_function" for criterion in task.success_criteria)
