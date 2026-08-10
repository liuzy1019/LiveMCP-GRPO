#!/usr/bin/env python3
"""Analyze per-prompt reward saturation from saved VERL rollout JSONL files."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import statistics
from typing import Any, Iterable


_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_TERMINAL_RE = re.compile(
    r"<(final_answer|ask_clarification|report_error)>",
)
_COMPONENT_FIELDS = (
    "r_task",
    "r_validity",
    "r_name_exists",
    "r_args_present",
    "r_execution",
    "r_coverage",
    "r_efficiency",
    "r_name",
    "r_arg",
)


def _json_value(value: Any, default: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return default
    return value if value is not None else default


def _required_tool_count(row: dict[str, Any]) -> int:
    ground_truth = row.get("gts") or {}
    required = ground_truth.get("required_tools") or []
    oracle = _json_value(ground_truth.get("oracle_calls"), [])
    tool_calls = [
        call
        for call in oracle
        if isinstance(call, dict)
        and call.get("action", "tool_call") == "tool_call"
        and call.get("tool_name")
        not in {"final_answer", "ask_clarification", "report_error"}
    ]
    return max(len(required), len(tool_calls))


def _replay_info(row: dict[str, Any]) -> dict[str, Any]:
    value = _json_value(row.get("reward_replay_info"), {})
    return value if isinstance(value, dict) else {}


def _functional_action_signature(output: str) -> str:
    actions: list[list[Any]] = []
    for raw_call in _TOOL_CALL_RE.findall(output):
        try:
            call = json.loads(raw_call)
        except json.JSONDecodeError:
            actions.append(["invalid_tool_call", raw_call.strip()])
            continue
        actions.append([
            "tool_call",
            str(call.get("name", "")),
            call.get("arguments", {}),
        ])
    actions.extend(["terminal", terminal] for terminal in _TERMINAL_RE.findall(output))
    return json.dumps(
        actions,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def load_rows(paths: Iterable[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: row must be an object")
            row["_source_file"] = str(path)
            rows.append(row)
    if not rows:
        raise ValueError("no rollout rows found")
    return rows


def analyze(rows: list[dict[str, Any]], min_group_std: float) -> dict[str, Any]:
    groups: dict[tuple[str, int, str, str], list[dict[str, Any]]] = defaultdict(list)
    grouping_sources: Counter[str] = Counter()
    for row in rows:
        prompt = row.get("input")
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("every rollout row must contain a non-empty input")
        source = str(row.get("_source_file") or "")
        run_id = (
            str(Path(source).parent.parent.resolve())
            if source else "__in_memory__"
        )
        step = int(row.get("step", 0))
        if str(row.get("group_id") or ""):
            grouping_source = "group_id"
            row_group_id = str(row["group_id"])
        elif str(row.get("uid") or ""):
            grouping_source = "uid"
            row_group_id = str(row["uid"])
        else:
            # Historical dumps did not persist the estimator's row identity.
            # Prompt fallback is diagnostic-only and can merge distinct source
            # rows that share the same initial user request.
            grouping_source = "prompt_fallback"
            row_group_id = prompt
        grouping_sources[grouping_source] += 1
        groups[(run_id, step, grouping_source, row_group_id)].append(row)

    summaries: list[dict[str, Any]] = []
    for (run_id, step, grouping_source, row_group_id), group_rows in sorted(groups.items()):
        prompts = {str(row.get("input", "")) for row in group_rows}
        if len(prompts) != 1:
            raise ValueError(
                f"one {grouping_source} group contains {len(prompts)} prompts"
            )
        prompt = next(iter(prompts))
        group_id = hashlib.sha256(
            f"{run_id}\0{step}\0{grouping_source}\0{row_group_id}".encode("utf-8")
        ).hexdigest()[:16]
        scores = [float(row["score"]) for row in group_rows]
        raw_j = [
            float(row["j"]) for row in group_rows if "j" in row
        ]
        post_kl = [
            float(row["post_kl_return"])
            for row in group_rows
            if "post_kl_return" in row
        ]
        required_count = _required_tool_count(group_rows[0])
        raw_std = statistics.pstdev(scores)
        exactly_saturated = raw_std == 0.0
        advantages = [
            float(row["trajectory_advantage"])
            for row in group_rows
            if "trajectory_advantage" in row
        ]
        replay_infos = [_replay_info(row) for row in group_rows]
        domains = {str(info.get("domain") or "unknown") for info in replay_infos}
        scenarios = {
            str(info.get("scenario_type") or "unknown") for info in replay_infos
        }
        if len(domains) != 1 or len(scenarios) != 1:
            raise ValueError(
                "one estimator group contains mixed domain/scenario metadata"
            )
        model_tool_calls = [
            int(info.get("n_model_tool_calls", 0) or 0) for info in replay_infos
        ]
        exec_successes = [
            int(info.get("n_exec_success", 0) or 0) for info in replay_infos
        ]
        round_ok_values = [
            float(row.get("r_round_ok", 0.0)) for row in group_rows
        ]
        summaries.append({
            "group_id": group_id,
            "source_group_id": row_group_id,
            "grouping_source": grouping_source,
            "run_id": run_id,
            "step": step,
            "size": len(group_rows),
            "required_tool_count": required_count,
            "task_kind": "no_tool" if required_count == 0 else "tool",
            "domain": next(iter(domains)),
            "scenario_type": next(iter(scenarios)),
            "raw_j_mean": (
                statistics.fmean(raw_j)
                if len(raw_j) == len(group_rows) else None
            ),
            "raw_j_std": (
                statistics.pstdev(raw_j)
                if len(raw_j) == len(group_rows) else None
            ),
            "raw_j_score_crosscheck_ok": (
                all(abs(score_j - score) < 1e-9
                    for score_j, score in zip(raw_j, scores))
                if len(raw_j) == len(group_rows) else None
            ),
            "post_kl_mean": statistics.fmean(scores),
            "post_kl_std": raw_std,
            "post_kl_return_std": (
                statistics.pstdev(post_kl)
                if len(post_kl) == len(group_rows)
                else None
            ),
            "unique_raw_rewards": len(set(scores)),
            "unique_outputs": len({
                str(row.get("output", "")) for row in group_rows
            }),
            "unique_functional_actions": len({
                _functional_action_signature(str(row.get("output", "")))
                for row in group_rows
            }),
            "saturated": raw_std < min_group_std,
            "exactly_saturated": exactly_saturated,
            "functional_reward_alias": (
                raw_std < min_group_std
                and len({
                    _functional_action_signature(str(row.get("output", "")))
                    for row in group_rows
                }) > 1
            ),
            "exact_functional_reward_alias": (
                exactly_saturated
                and len({
                    _functional_action_signature(str(row.get("output", "")))
                    for row in group_rows
                }) > 1
            ),
            "mean_model_tool_calls": statistics.fmean(model_tool_calls),
            "mean_execution_successes": statistics.fmean(exec_successes),
            "round_contract_success_rate": statistics.fmean(round_ok_values),
            "integrity_failure_rows": sum(
                info.get("trajectory_integrity_ok") is False
                for info in replay_infos
            ),
            "nonzero_trajectory_advantage": (
                max((abs(value) for value in advantages), default=0.0) > 1e-7
                if len(advantages) == len(group_rows)
                else None
            ),
        })

    by_kind: dict[str, dict[str, Any]] = {}
    for kind in ("tool", "no_tool"):
        selected = [summary for summary in summaries if summary["task_kind"] == kind]
        saturated = sum(bool(summary["saturated"]) for summary in selected)
        exactly_saturated = sum(
            bool(summary["exactly_saturated"]) for summary in selected
        )
        by_kind[kind] = {
            "groups": len(selected),
            "saturated_groups": saturated,
            "saturation_rate": saturated / len(selected) if selected else None,
            "exactly_saturated_groups": exactly_saturated,
            "exact_saturation_rate": (
                exactly_saturated / len(selected) if selected else None
            ),
            "mean_unique_functional_actions": (
                statistics.fmean(
                    summary["unique_functional_actions"] for summary in selected
                )
                if selected
                else None
            ),
        }

    by_domain: dict[str, dict[str, Any]] = {}
    for domain in sorted({summary["domain"] for summary in summaries}):
        selected = [
            summary for summary in summaries if summary["domain"] == domain
        ]
        saturated = sum(bool(summary["saturated"]) for summary in selected)
        exactly_saturated = sum(
            bool(summary["exactly_saturated"]) for summary in selected
        )
        by_domain[domain] = {
            "groups": len(selected),
            "rows": sum(int(summary["size"]) for summary in selected),
            "saturated_groups": saturated,
            "saturation_rate": saturated / len(selected),
            "exactly_saturated_groups": exactly_saturated,
            "exact_saturation_rate": exactly_saturated / len(selected),
            "functional_reward_alias_groups": sum(
                bool(summary["functional_reward_alias"])
                for summary in selected
            ),
            "exact_functional_reward_alias_groups": sum(
                bool(summary["exact_functional_reward_alias"])
                for summary in selected
            ),
            "mean_post_kl": statistics.fmean(
                summary["post_kl_mean"] for summary in selected
            ),
            "mean_raw_j": (
                statistics.fmean(
                    summary["raw_j_mean"] for summary in selected
                    if summary["raw_j_mean"] is not None
                )
                if any(s["raw_j_mean"] is not None for s in selected)
                else None
            ),
            "mean_model_tool_calls": statistics.fmean(
                summary["mean_model_tool_calls"] for summary in selected
            ),
            "mean_execution_successes": statistics.fmean(
                summary["mean_execution_successes"] for summary in selected
            ),
            "round_contract_success_rate": statistics.fmean(
                summary["round_contract_success_rate"] for summary in selected
            ),
            "integrity_failure_rows": sum(
                int(summary["integrity_failure_rows"])
                for summary in selected
            ),
        }

    component_means = {
        field: statistics.fmean(
            float(row[field]) for row in rows if field in row
        )
        for field in _COMPONENT_FIELDS
        if any(field in row for row in rows)
    }
    saturated_count = sum(bool(summary["saturated"]) for summary in summaries)
    exactly_saturated_count = sum(
        bool(summary["exactly_saturated"]) for summary in summaries
    )
    raw_saturated_nonzero_advantage = sum(
        bool(summary["saturated"])
        and summary["nonzero_trajectory_advantage"] is True
        for summary in summaries
    )
    return {
        "rows": len(rows),
        "groups": len(summaries),
        "group_size_distribution": dict(sorted(Counter(
            summary["size"] for summary in summaries
        ).items())),
        "grouping_source_rows": dict(sorted(grouping_sources.items())),
        "grouping_exact": set(grouping_sources) <= {"group_id", "uid"},
        "min_group_std": min_group_std,
        "saturated_groups": saturated_count,
        "saturation_rate": saturated_count / len(summaries),
        "exactly_saturated_groups": exactly_saturated_count,
        "exact_saturation_rate": exactly_saturated_count / len(summaries),
        "raw_saturated_nonzero_advantage_groups": (
            raw_saturated_nonzero_advantage
        ),
        "by_task_kind": by_kind,
        "by_domain": by_domain,
        "component_means": component_means,
        "post_kl_available": all(
            summary["post_kl_return_std"] is not None for summary in summaries
        ),
        "groups_detail": summaries,
    }


def _resolve_paths(raw_paths: list[str]) -> list[Path]:
    paths: list[Path] = []
    for raw_path in raw_paths:
        path = Path(raw_path)
        if path.is_dir():
            paths.extend(sorted(path.glob("*.jsonl")))
        else:
            paths.append(path)
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing rollout files: {missing}")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "paths",
        nargs="+",
        help="Rollout JSONL files or directories containing step JSONL files.",
    )
    parser.add_argument("--min-group-std", type=float, default=1e-6)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = analyze(
        load_rows(_resolve_paths(args.paths)),
        min_group_std=args.min_group_std,
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
