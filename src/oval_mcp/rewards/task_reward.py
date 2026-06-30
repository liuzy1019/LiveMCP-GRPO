"""Task Reward: R_task per OVAL-MCP §7.1.

Trajectory-level R_task composed of:
  R_positive = w_val * R_validity + w_cov * R_coverage + w_name * R_name + w_arg * R_arg
  R_task = clip(R_positive / Z_pos + w_eff * R_efficiency, -0.2, 1.0)

If required_tool_calls = []: binary R_task (no-tool tasks).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.oval_mcp.verifier.events import EventLog


# Recommended weights from OVAL-MCP §7.1
DEFAULT_WEIGHTS = {
    "w_val": 0.5,
    "w_cov": 0.5,
    "w_eff": 0.15,
    "w_name": 0.2,
    "w_arg": 0.1,
    "w_struct": 0.6,
    "w_exec": 0.4,
    "alpha_eff": 0.5,
    "beta_budget": 0.5,
}


@dataclass
class TaskRewardResult:
    """Decomposed task reward with all components."""

    r_task: float = 0.0
    r_validity: float = 0.0
    r_structural: float = 0.0
    r_execution: float = 0.0
    r_coverage: float = 0.0
    r_name: float = 0.0
    r_arg: float = 0.0
    r_efficiency: float = 0.0
    r_positive: float = 0.0
    z_pos: float = 1.0

    # Diagnostics
    n_model_calls: int = 0
    n_required_calls: int = 0
    completed_predicates: int = 0
    total_predicates: int = 1
    aligned_calls: int = 0
    is_no_tool_task: bool = False

    def to_dict(self) -> dict[str, float]:
        return {
            "r_task": self.r_task,
            "r_validity": self.r_validity,
            "r_structural": self.r_structural,
            "r_execution": self.r_execution,
            "r_coverage": self.r_coverage,
            "r_name": self.r_name,
            "r_arg": self.r_arg,
            "r_efficiency": self.r_efficiency,
            "n_model_calls": float(self.n_model_calls),
            "completed_predicates": float(self.completed_predicates),
            "total_predicates": float(self.total_predicates),
        }


class TaskReward:
    """Compute R_task from trajectory event log and task definition."""

    def __init__(self, weights: dict[str, float] | None = None):
        self.w = {**DEFAULT_WEIGHTS, **(weights or {})}

    def compute(
        self,
        event_log: EventLog,
        task: dict[str, Any],
        domain_adapter: Any = None,
    ) -> TaskRewardResult:
        """Compute R_task for a complete trajectory."""
        result = TaskRewardResult()

        required_tool_calls = task.get("required_tool_calls", [])
        is_no_tool = len(required_tool_calls) == 0
        result.is_no_tool_task = is_no_tool

        if is_no_tool:
            return self._compute_no_tool(event_log, task, result)

        return self._compute_with_tools(event_log, task, required_tool_calls, result, domain_adapter)

    def _compute_no_tool(
        self,
        event_log: EventLog,
        task: dict[str, Any],
        result: TaskRewardResult,
    ) -> TaskRewardResult:
        """No-tool task: binary R_task.

        R_task = 1.0 if no tool calls AND terminal predicate passes, else 0.0
        """
        n_calls = len(event_log.tool_call_events)
        result.n_model_calls = n_calls

        # Check if terminal action satisfies task predicate
        terminal_ok = self._check_terminal_predicate(event_log, task)

        if n_calls == 0 and terminal_ok:
            result.r_task = 1.0
        else:
            result.r_task = 0.0

        return result

    def _compute_with_tools(
        self,
        event_log: EventLog,
        task: dict[str, Any],
        required_tool_calls: list[dict],
        result: TaskRewardResult,
        domain_adapter: Any = None,
    ) -> TaskRewardResult:
        """Tool-required task: full R_task formula."""
        tool_events = event_log.tool_call_events
        n_calls = len(tool_events)
        result.n_model_calls = n_calls

        # 1. R_validity = w_struct * R_structural + w_exec * R_execution
        r_structural = self._compute_structural_validity(tool_events, task)
        r_execution = self._compute_execution_validity(tool_events)
        result.r_structural = r_structural
        result.r_execution = r_execution
        result.r_validity = self.w["w_struct"] * r_structural + self.w["w_exec"] * r_execution

        # Terminal-action whitelist enforcement.
        # Tasks may declare allowed_terminal_actions (e.g. ["ask_clarification"]
        # for clarification scenarios, ["report_error"] for missing_function,
        # ["final_answer"] for normal). Violating the whitelist halves
        # r_validity — strong enough to matter for training, soft enough not
        # to wipe out partial-credit on coverage / name / arg components.
        if not self._check_terminal_predicate(event_log, task):
            result.r_validity *= 0.5

        # 2. R_coverage: completed outcome predicates / total
        # P0-2: Combine outcome_assertions (operation-level) with
        # success_criteria (state-level). Both must be satisfied for full
        # coverage; the denominator is the union count so that adding
        # state criteria does not artificially deflate scores when only
        # ops are required.
        outcome_count = len(task.get("outcome_assertions", []))
        criteria_list = task.get("success_criteria", []) or []
        criteria_count = len(criteria_list)
        total_preds = max(outcome_count + criteria_count, 1)
        completed_ops = self._count_completed_predicates(event_log, task, domain_adapter)
        completed_state = self._count_completed_state_criteria(event_log, criteria_list)
        completed = completed_ops + completed_state
        result.completed_predicates = completed
        result.total_predicates = total_preds
        result.r_coverage = min(1.0, completed / total_preds) if total_preds > 0 else 0.0

        # Check identity violation → R_coverage = 0
        if self._has_identity_violation(event_log, task):
            result.r_coverage = 0.0

        # 3. R_name: precision — fraction of model calls whose name is in GT
        #    PROVE §4.2: R_name = |{c ∈ Ĉ : c.name ∈ GT_names}| / |Ĉ|
        required_names = self._required_tool_names(required_tool_calls)
        model_names = self._model_tool_names(tool_events)
        if n_calls == 0:
            result.r_name = 0.0
        else:
            # 分母是模型调用总数（precision），惩罚调用不在 GT 中的工具
            correct_calls = sum(1 for e in tool_events if e.tool_name in required_names)
            result.r_name = correct_calls / n_calls

        # 4. R_arg: argument value match for aligned calls
        aligned_calls = self._identify_aligned_calls(tool_events, task, required_tool_calls)
        result.aligned_calls = len(aligned_calls)
        result.r_arg = self._compute_arg_score(aligned_calls)

        # 5. R_efficiency
        n_required = self._count_required_calls(required_tool_calls)
        result.n_required_calls = n_required
        result.r_efficiency = self._compute_efficiency(n_calls, n_required)

        # R_positive and R_task
        result.r_positive = (
            self.w["w_val"] * result.r_validity
            + self.w["w_cov"] * result.r_coverage
            + self.w["w_name"] * result.r_name
            + self.w["w_arg"] * result.r_arg
        )
        result.z_pos = (
            self.w["w_val"] + self.w["w_cov"] + self.w["w_name"] + self.w["w_arg"]
        )
        result.r_task = max(
            -0.2,
            min(1.0, result.r_positive / result.z_pos + self.w["w_eff"] * result.r_efficiency),
        )

        return result

    def _compute_structural_validity(
        self,
        tool_events: list,
        task: dict[str, Any],
    ) -> float:
        """R_structural: fraction of tool calls with schema-valid args."""
        if not tool_events:
            return 0.0
        valid = sum(1 for e in tool_events if e.schema_valid)
        return valid / len(tool_events)

    def _compute_execution_validity(
        self,
        tool_events: list,
    ) -> float:
        """R_execution: fraction of tool calls that executed successfully."""
        if not tool_events:
            return 0.0
        success = sum(1 for e in tool_events if e.execution_success)
        return success / len(tool_events)

    def _count_completed_state_criteria(
        self,
        event_log: EventLog,
        criteria: list[dict],
    ) -> int:
        """P0-2: Verify state-level success_criteria against the trajectory.

        Each criterion is a dict like:
            {"type": "state_equals", "server": <domain>, "path": <dotted>, "value": <expected>}
            {"type": "state_exists", "server": <domain>, "path": <dotted>}
            {"type": "file_exists", "server": <domain>, "path": <dotted>}
            {"type": "cart_not_empty", "server": <domain>}
            {"type": "email_count_gte", "server": <domain>, "value": <int>}
            {"type": "missing_function", ...}                # checked elsewhere

        We approximate the post-trajectory state from the LAST tool_call
        event whose observation is a dict (the executor returns the
        post-call state snapshot). When no observation is available we
        fall back to checking that any tool_call with operation matching
        the criterion path exists; safer than always returning 0.

        Returns the number of criteria that hold true.
        """
        if not criteria:
            return 0

        # Build a best-effort "final state" view from the latest event observation
        # whose schema is a dict.
        final_state: dict[str, Any] | None = None
        for ev in reversed(event_log.events):
            obs = getattr(ev, "observation", None)
            if isinstance(obs, dict) and obs:
                final_state = obs
                break

        # Build set of ids that the trajectory created/updated/deleted, so
        # state_exists / state_equals can be approximated even without a
        # final-state snapshot.
        seen_ids: set[str] = set()
        for ev in event_log.events:
            if getattr(ev, "target_id", ""):
                seen_ids.add(ev.target_id)
            for cid in getattr(ev, "created_ids", []) or []:
                seen_ids.add(cid)

        completed = 0
        for c in criteria:
            if not isinstance(c, dict):
                continue
            ctype = c.get("type", "")
            path = c.get("path", "")
            if ctype == "missing_function":
                # Handled by allowed_terminal_actions, not here
                continue
            if ctype == "state_exists":
                if not path:
                    continue
                # path is dotted: e.g. "events.evt_001"
                target = path.rsplit(".", 1)[-1]
                if target in seen_ids or self._lookup_state(final_state, path) is not None:
                    completed += 1
            elif ctype == "state_equals":
                value = c.get("value")
                actual = self._lookup_state(final_state, path)
                if actual is not None and str(actual) == str(value):
                    completed += 1
            elif ctype == "file_exists":
                if self._lookup_state(final_state, path) is not None:
                    completed += 1
            elif ctype == "cart_not_empty":
                cart = self._lookup_state(final_state, "cart") if final_state else None
                if cart:
                    completed += 1
            elif ctype == "email_count_gte":
                emails = self._lookup_state(final_state, "emails") if final_state else None
                if isinstance(emails, dict) and len(emails) >= int(c.get("value", 0)):
                    completed += 1
            else:
                # Unknown criterion type — skip rather than penalise
                continue
        return completed

    @staticmethod
    def _lookup_state(state: dict | None, path: str) -> Any:
        """Walk a dotted path through a state dict; return None if missing."""
        if not state or not isinstance(state, dict) or not path:
            return None
        cur: Any = state
        for part in path.split("."):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                return None
        return cur

    def _count_completed_predicates(
        self,
        event_log: EventLog,
        task: dict[str, Any],
        domain_adapter: Any = None,
    ) -> int:
        """Count how many unique progress predicates were satisfied across the trajectory.

        Uses DomainAdapter.evaluate_event() when available; falls back to
        operation-based counting otherwise.
        """
        if domain_adapter is not None:
            completed: set[str] = set()
            for event in event_log.events:
                try:
                    satisfied = domain_adapter.evaluate_event(event, task)
                    completed.update(satisfied)
                except Exception:
                    pass
            return len(completed)

        # Fallback: operation-based counting
        assertions = task.get("outcome_assertions", [])
        if not assertions:
            return 0
        operations = {e.operation for e in event_log.events if e.operation}
        required_ops = set()
        for a in assertions:
            if isinstance(a, dict):
                op = a.get("operation")
                if op:
                    required_ops.add(op)
        if not required_ops:
            return 0
        return sum(1 for op in required_ops if op in operations)

    def _has_identity_violation(
        self,
        event_log: EventLog,
        task: dict[str, Any],
    ) -> bool:
        """Check if any event has identity violation AND task requires preserve."""
        identity_policy = task.get("identity_policy", "")
        if identity_policy != "preserve":
            return False
        return any(e.identity_violation for e in event_log.events)

    def _required_tool_names(
        self,
        required_tool_calls: list[dict],
    ) -> set[str]:
        """Extract unique required tool names."""
        return {c.get("tool_name", "") for c in required_tool_calls if c.get("tool_name")}

    def _model_tool_names(self, tool_events: list) -> set[str]:
        """Extract unique tool names from model calls."""
        return {e.tool_name for e in tool_events if e.tool_name}

    def _identify_aligned_calls(
        self,
        tool_events: list,
        task: dict[str, Any],
        required_tool_calls: list[dict],
    ) -> list:
        """Identify model calls that are 'aligned' with required tools.

        A call is aligned if:
          1. tool_name matches a required tool
          2. the call advanced at least one predicate (execution_success or state_changed)
        """
        required_names = self._required_tool_names(required_tool_calls)
        aligned = []
        for e in tool_events:
            if e.tool_name in required_names and (e.execution_success or e.state_changed):
                aligned.append((e, self._find_required_call(required_tool_calls, e.tool_name)))
        return aligned

    def _find_required_call(
        self,
        required_tool_calls: list[dict],
        tool_name: str,
    ) -> dict | None:
        for c in required_tool_calls:
            if c.get("tool_name") == tool_name:
                return c
        return None

    def _compute_arg_score(
        self,
        aligned_calls: list,
    ) -> float:
        """R_arg: mean arg_match_score across aligned calls.

        arg_match_score = |matched_arg_values| / |required_arg_values|
        """
        if not aligned_calls:
            return 0.0
        scores = []
        for event, required_call in aligned_calls:
            if required_call is None:
                continue
            required_args = required_call.get("arguments", {})
            if not required_args:
                scores.append(1.0)
                continue
            model_args = event.tool_arguments or {}
            matched = 0
            for key, expected_val in required_args.items():
                actual_val = model_args.get(key)
                if actual_val is None:
                    continue
                if self._args_equal(actual_val, expected_val):
                    matched += 1
            scores.append(matched / len(required_args))
        return sum(scores) / len(scores) if scores else 0.0

    @staticmethod
    def _args_equal(actual: Any, expected: Any) -> bool:
        """Type-aware argument equality.

        - Numbers compared as floats (500 == 500.0 == "500").
        - Booleans compared as bools (True == "true" == "True").
        - dict/list compared via canonical JSON (key-order independent for dicts).
        - Strings compared case-insensitive after strip.
        - Falls back to str().lower() comparison.
        """
        # Numeric comparison: try float on both sides
        try:
            af = float(actual)
            ef = float(expected)
            return abs(af - ef) < 1e-9
        except (TypeError, ValueError):
            pass

        # Bool comparison
        if isinstance(actual, bool) or isinstance(expected, bool):
            def _to_bool(x: Any) -> bool | None:
                if isinstance(x, bool):
                    return x
                if isinstance(x, str) and x.strip().lower() in ("true", "false"):
                    return x.strip().lower() == "true"
                return None
            ab, eb = _to_bool(actual), _to_bool(expected)
            if ab is not None and eb is not None:
                return ab == eb

        # Structured comparison: dict / list
        if isinstance(actual, (dict, list)) or isinstance(expected, (dict, list)):
            try:
                import json as _json
                return _json.dumps(actual, sort_keys=True, ensure_ascii=False) == \
                       _json.dumps(expected, sort_keys=True, ensure_ascii=False)
            except (TypeError, ValueError):
                pass

        # String fallback: case-insensitive + strip
        return str(actual).strip().lower() == str(expected).strip().lower()

    def _compute_efficiency(
        self,
        n_model_calls: int,
        n_required_calls: int,
    ) -> float:
        """R_efficiency: adaptive excess-call penalty.

        B = n_required_calls + ceil(beta_budget * n_required_calls)
        R_efficiency = -alpha_eff * max(0, n_model_calls - B) / max(B, 1)
        """
        import math
        B = n_required_calls + math.ceil(self.w["beta_budget"] * n_required_calls)
        B = max(B, 1)
        excess = max(0, n_model_calls - B)
        return -self.w["alpha_eff"] * excess / B

    def _count_required_calls(self, required_tool_calls: list[dict]) -> int:
        """Count the number of required tool calls (with multiplicity)."""
        return len(required_tool_calls)

    def _check_terminal_predicate(
        self,
        event_log: EventLog,
        task: dict[str, Any],
    ) -> bool:
        """Check if trajectory ends with a valid terminal action type.

        Structural check — actual predicate satisfaction (verified_postcondition,
        produced_required_response) is handled by DomainAdapter.evaluate_event()
        during R_coverage / F_gamma / P_process computation.
        """
        if not event_log.events:
            return False
        last = event_log.events[-1]
        allowed = task.get("allowed_terminal_actions", ["final_answer", "ask_clarification", "report_error"])
        return last.action_type in allowed


__all__ = ["TaskReward", "TaskRewardResult", "DEFAULT_WEIGHTS"]
