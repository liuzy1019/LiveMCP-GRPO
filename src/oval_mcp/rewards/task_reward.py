"""Five-component trajectory task reward.

Trajectory-level reward:
  R_task = w_val*R_validity + w_cov*R_coverage
           + w_eff*R_efficiency + w_name*R_name + w_arg*R_arg

Weights: w_val=0.5, w_cov=0.5, w_eff=0.15, w_name=0.2, w_arg=0.1
The maximum tool-task reward is 1.3 with the default weights. The lower
bound depends on the finite rollout action budget because efficiency is not
clipped by the published formula.

If required_tool_calls = []: binary R_task (no-tool tasks).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.oval_mcp.verifier.events import EventLog


# PROVE task-reward weights; docs/OVAL-MCP.md is the authoritative contract.
DEFAULT_WEIGHTS = {
    "w_val": 0.5,
    "w_cov": 0.5,
    "w_eff": 0.15,
    "w_name": 0.2,
    "w_arg": 0.1,
    "alpha_eff": 0.5,
    "beta_budget": 0.5,
}


@dataclass
class TaskRewardResult:
    """Decomposed task reward with all components."""

    r_task: float = 0.0
    r_validity: float = 0.0
    # 三级分项，各占 1/3
    r_name_exists: float = 0.0    # level-1: function name 在 schema 中存在
    r_args_present: float = 0.0   # level-2: 必需参数存在且类型兼容
    r_execution: float = 0.0      # level-3: live 执行成功
    r_coverage: float = 0.0
    r_name: float = 0.0
    r_arg: float = 0.0
    r_efficiency: float = 0.0

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
            "r_name_exists": self.r_name_exists,
            "r_args_present": self.r_args_present,
            "r_execution": self.r_execution,
            "r_coverage": self.r_coverage,
            "r_name": self.r_name,
            "r_arg": self.r_arg,
            "r_efficiency": self.r_efficiency,
            "n_model_calls": float(self.n_model_calls),
            "n_required_calls": float(self.n_required_calls),
            "completed_predicates": float(self.completed_predicates),
            "total_predicates": float(self.total_predicates),
            "aligned_calls": float(self.aligned_calls),
            "is_no_tool_task": 1.0 if self.is_no_tool_task else 0.0,
        }


class TaskReward:
    """Compute R_task from trajectory event log and task definition."""

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        reward_profile: str = "prove_baseline",
    ):
        if reward_profile not in {"prove_baseline", "oval_full"}:
            raise ValueError(
                "reward_profile must be 'prove_baseline' or 'oval_full', "
                f"got {reward_profile!r}"
            )
        self.w = {**DEFAULT_WEIGHTS, **(weights or {})}
        self.reward_profile = reward_profile

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

        PROVE:
            R_task = 1.0 if no tool calls, else 0.0.

        OVAL:
            Preserve the stricter local terminal contract.
        """
        n_calls = len(event_log.tool_call_events)
        result.n_model_calls = n_calls

        if self.reward_profile == "prove_baseline":
            result.r_task = 1.0 if n_calls == 0 else 0.0
            return result

        terminal_ok = self._check_terminal_predicate(event_log, task)
        result.r_task = 1.0 if n_calls == 0 and terminal_ok else 0.0
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

        # 1. R_validity: 三级等权平均
        #    level-1 (1/3): function name 在 candidate schema 中存在
        #    level-2 (1/3): 所有必需参数存在且 JSON 类型兼容
        #    level-3 (1/3): live 执行成功无错
        #    部分分：名字对参数错 ≈ 0.33；结构正确执行失败 ≈ 0.66
        r_name_exists, r_args_present = self._compute_structural_validity_3level(tool_events, task)
        r_execution = self._compute_execution_validity(tool_events)
        result.r_name_exists = r_name_exists
        result.r_args_present = r_args_present
        result.r_execution = r_execution
        result.r_validity = (r_name_exists + r_args_present + r_execution) / 3.0

        # Terminal-action penalties are enabled only by the oval_full profile.
        # three-level validity definition. The reward profile sets this flag
        # prove_baseline applies no terminal multiplier.
        if (
            task.get("apply_terminal_validity_penalty", False)
            and not self._check_terminal_predicate(event_log, task)
        ):
            result.r_validity *= 0.5

        # 2. R_coverage: dependency-ordered GT step coverage.
        #    R_cov = (1/|G|) * Σ m(g)*o(g)
        #    |G| = len(required_tool_calls)，即 GT steps 数量。
        #    m(g) = 1 if GT step g is matched in model output
        #    o(g) = 1 if all dependency predecessors of g are matched earlier
        #    When dependency_edges are available, use partial-order matching:
        #    non-dependent tools can be called in any order; only dependency
        #    chains enforce ordering constraints.
        dep_edges: list[tuple[int, int]] = task.get("dependency_edges", [])
        if self.reward_profile == "prove_baseline":
            # Published PROVE matching uses name + GT argument keys and only
            # dependency order. Execution success belongs to R_validity, while
            # conversation-round alignment is a local OVAL contract.
            aligned_calls = self._match_required_calls_partial_order(
                tool_events, required_tool_calls, dep_edges,
                required_call_rounds=None,
                require_execution_success=False,
            )
        else:
            required_call_rounds: list[int] = task.get(
                "required_call_rounds", [0] * len(required_tool_calls),
            )
            if len(required_call_rounds) != len(required_tool_calls):
                raise ValueError(
                    "required_call_rounds must align with required_tool_calls"
                )
            if dep_edges:
                aligned_calls = self._match_required_calls_partial_order(
                    tool_events, required_tool_calls, dep_edges,
                    required_call_rounds=required_call_rounds,
                    require_execution_success=True,
                )
            else:
                aligned_calls = self._match_required_calls_in_order(
                    tool_events, required_tool_calls,
                    required_call_rounds=required_call_rounds,
                    require_execution_success=True,
                )
        total_preds = max(len(required_tool_calls), 1)
        completed = len(aligned_calls)
        result.completed_predicates = completed
        result.total_predicates = total_preds
        result.r_coverage = completed / total_preds

        # Identity coverage penalties are project-level safety shaping.
        if (
            task.get("apply_identity_coverage_penalty", False)
            and self._has_identity_violation(event_log, task)
        ):
            result.r_coverage = 0.0

        # 3. R_name: precision — fraction of model calls whose name is in GT
        #    R_name = |{c ∈ Ĉ : c.name ∈ GT_names}| / |Ĉ|
        required_names = self._required_tool_names(required_tool_calls)
        model_names = self._model_tool_names(tool_events)
        if n_calls == 0:
            result.r_name = 0.0
        else:
            # 分母是模型调用总数（precision），惩罚调用不在 GT 中的工具
            correct_calls = sum(1 for e in tool_events if e.tool_name in required_names)
            result.r_name = correct_calls / n_calls

        # 4. R_arg: argument value match for aligned calls
        result.aligned_calls = len(aligned_calls)
        result.r_arg = self._compute_arg_score(aligned_calls)

        # 5. R_efficiency
        n_required = self._count_required_calls(required_tool_calls)
        result.n_required_calls = n_required
        result.r_efficiency = self._compute_efficiency(n_calls, n_required)

        # Direct weighted sum; no additional normalization or clipping.
        result.r_task = (
            self.w["w_val"] * result.r_validity
            + self.w["w_cov"] * result.r_coverage
            + self.w["w_eff"] * result.r_efficiency
            + self.w["w_name"] * result.r_name
            + self.w["w_arg"] * result.r_arg
        )

        return result

    def _compute_structural_validity_3level(
        self,
        tool_events: list,
        task: dict[str, Any],
    ) -> tuple[float, float]:
        """三级 validity 的前两级，返回 (r_name_exists, r_args_present)。

        level-1 r_name_exists: tool_name 在 candidate schema 中存在
            → event.tool_name_known（由 rollout candidate set 设置）
        level-2 r_args_present: 所有必需参数存在且 JSON 类型兼容
            → event.schema_valid（executor 做完整 schema 校验后设置）
              仅当 level-1 通过时才有意义，否则记 0
        """
        if not tool_events:
            return 0.0, 0.0
        n = len(tool_events)
        name_ok = 0
        args_ok = 0
        for e in tool_events:
            # level-1: name 存在性
            l1 = e.tool_name_known
            name_ok += int(bool(l1))
            # level-2: 参数校验（仅 name 存在时才有意义）
            l2 = e.schema_valid if l1 else False
            args_ok += int(bool(l2))
        return name_ok / n, args_ok / n

    def _compute_execution_validity(
        self,
        tool_events: list,
    ) -> float:
        """R_execution: fraction of tool calls that executed successfully."""
        if not tool_events:
            return 0.0
        success = sum(1 for e in tool_events if e.execution_success)
        return success / len(tool_events)

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

    def _match_required_calls_in_order(
        self,
        tool_events: list,
        required_tool_calls: list[dict],
        required_call_rounds: list[int] | None,
        require_execution_success: bool,
    ) -> list:
        """Greedily align oracle calls to later successful model events."""
        aligned: list[tuple[Any, dict]] = []
        cursor = 0
        for required_idx, required in enumerate(required_tool_calls):
            required_round = (
                required_call_rounds[required_idx]
                if required_call_rounds is not None
                else None
            )
            required_name = required.get("tool_name", "")
            required_keys = set((required.get("arguments") or {}).keys())
            for idx in range(cursor, len(tool_events)):
                event = tool_events[idx]
                if (
                    required_round is not None
                    and int(getattr(event, "round_idx", -1)) != required_round
                ):
                    continue
                if event.tool_name != required_name:
                    continue
                if require_execution_success and not event.execution_success:
                    continue
                if not required_keys.issubset(set((event.tool_arguments or {}).keys())):
                    continue
                aligned.append((event, required))
                cursor = idx + 1
                break
        return aligned

    def _match_required_calls_partial_order(
        self,
        tool_events: list,
        required_tool_calls: list[dict],
        dependency_edges: list[tuple[int, int]],
        required_call_rounds: list[int] | None,
        require_execution_success: bool,
    ) -> list:
        """P0-3: dependency partial-order coverage matching with temporal validation.

        dependency_edges is a list of (src_idx, dst_idx) tuples where indices
        refer to positions in required_tool_calls.  Builds preds_by_idx from
        these edges directly — no tool-name lookup.

        Matching rules:
        1. A required step i can only match after all predecessors in
           preds_by_idx[i] have been matched.
        2. The matched event index must be STRICTLY GREATER than the max
           event index of all matched predecessors (temporal ordering).
        3. Non-dependent tools can be called in any order (no predecessor
           constraint → no temporal constraint).

        Returns list of (event, required_dict) for matched calls.
        """
        n_required = len(required_tool_calls)

        # Build predecessor index sets from index-based edges
        preds_by_idx: list[set[int]] = [set() for _ in range(n_required)]
        for src, dst in dependency_edges:
            if 0 <= src < n_required and 0 <= dst < n_required:
                preds_by_idx[dst].add(src)

        matched_preds: set[int] = set()            # indices of matched required calls
        matched_event_idx: dict[int, int] = {}      # required_idx → event_idx
        aligned: list = []
        used_events: set[int] = set()

        # Iterate until no progress (allow reordering for non-dependent tools)
        for _attempt in range(n_required * 2):
            made_progress = False
            for i, required in enumerate(required_tool_calls):
                if i in matched_preds:
                    continue
                preds = preds_by_idx[i]
                if not preds.issubset(matched_preds):
                    continue

                # Compute the earliest allowed event index:
                # must be after all matched predecessor events
                min_event_idx = -1
                if preds:
                    max_pred_event = max(matched_event_idx[p] for p in preds)
                    min_event_idx = max_pred_event

                required_name = required.get("tool_name", "")
                required_keys = set((required.get("arguments") or {}).keys())
                required_round = (
                    required_call_rounds[i]
                    if required_call_rounds is not None
                    else None
                )

                for idx, event in enumerate(tool_events):
                    if idx in used_events:
                        continue
                    if idx <= min_event_idx:
                        continue  # temporal ordering: must be AFTER predecessors
                    if (
                        required_round is not None
                        and int(getattr(event, "round_idx", -1)) != required_round
                    ):
                        continue
                    if event.tool_name != required_name:
                        continue
                    if require_execution_success and not event.execution_success:
                        continue
                    if not required_keys.issubset(set((event.tool_arguments or {}).keys())):
                        continue
                    aligned.append((event, required))
                    used_events.add(idx)
                    matched_preds.add(i)
                    matched_event_idx[i] = idx
                    made_progress = True
                    break

            if not made_progress:
                break

        return aligned

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
        # bool is a subclass of int in Python. Handle it before numeric
        # coercion so True never matches 1 and False never matches 0.
        if isinstance(actual, bool) or isinstance(expected, bool):
            def _to_bool(x: Any) -> bool | None:
                if isinstance(x, bool):
                    return x
                if isinstance(x, str) and x.strip().lower() in ("true", "false"):
                    return x.strip().lower() == "true"
                return None
            ab, eb = _to_bool(actual), _to_bool(expected)
            return ab is not None and eb is not None and ab == eb

        # Numeric comparison: try float on both sides
        try:
            af = float(actual)
            ef = float(expected)
            return abs(af - ef) < 1e-9
        except (TypeError, ValueError):
            pass

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
