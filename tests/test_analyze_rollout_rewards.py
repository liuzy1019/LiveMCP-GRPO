from scripts.analyze_rollout_rewards import (
    _functional_action_signature,
    analyze,
)


def _row(
    prompt: str,
    score: float,
    output: str,
    required_tools: list[str],
    *,
    step: int = 0,
    group_id: str | None = None,
    domain: str = "test",
) -> dict:
    row = {
        "input": prompt,
        "output": output,
        "score": score,
        "j": score,
        "gts": {
            "required_tools": required_tools,
            "oracle_calls": "[]",
        },
        "r_task": score,
        "step": step,
        "reward_replay_info": {
            "domain": domain,
            "scenario_type": "normal_safe_success",
            "n_model_tool_calls": int(bool(required_tools)),
            "n_exec_success": int(bool(required_tools)),
            "trajectory_integrity_ok": True,
        },
    }
    if group_id is not None:
        row["group_id"] = group_id
    return row


def test_functional_signature_ignores_json_key_order() -> None:
    left = '<tool_call>{"name":"read","arguments":{"b":2,"a":1}}</tool_call>'
    right = '<tool_call>{"arguments":{"a":1,"b":2},"name":"read"}</tool_call>'
    assert _functional_action_signature(left) == _functional_action_signature(right)


def test_analysis_separates_tool_and_no_tool_saturation() -> None:
    rows = [
        _row("tool", 0.0, '<tool_call>{"name":"read","arguments":{}}</tool_call>', ["read"]),
        _row("tool", 1.0, '<tool_call>{"name":"write","arguments":{}}</tool_call>', ["read"]),
        _row("no-tool", 1.0, "<report_error>unsupported</report_error>", []),
        _row("no-tool", 1.0, "<report_error>cannot do that</report_error>", []),
    ]
    report = analyze(rows, min_group_std=1e-6)
    assert report["groups"] == 2
    assert report["saturated_groups"] == 1
    assert report["exactly_saturated_groups"] == 1
    assert report["by_task_kind"]["tool"]["saturation_rate"] == 0.0
    assert report["by_task_kind"]["no_tool"]["saturation_rate"] == 1.0
    assert report["post_kl_available"] is False
    assert report["by_domain"]["test"]["groups"] == 2


def test_analysis_separates_exact_saturation_from_low_variance() -> None:
    rows = [
        _row("low", 0.0, "<report_error>a</report_error>", []),
        _row("low", 0.01, "<report_error>b</report_error>", []),
    ]
    report = analyze(rows, min_group_std=0.01)
    assert report["saturated_groups"] == 1
    assert report["exactly_saturated_groups"] == 0
    assert report["groups_detail"][0]["exactly_saturated"] is False


def test_analysis_does_not_merge_training_steps() -> None:
    rows = [
        _row("same-prompt", 0.0, "<report_error>a</report_error>", [], step=1),
        _row("same-prompt", 0.0, "<report_error>b</report_error>", [], step=1),
        _row("same-prompt", 1.0, "<report_error>c</report_error>", [], step=2),
        _row("same-prompt", 1.0, "<report_error>d</report_error>", [], step=2),
    ]
    report = analyze(rows, min_group_std=1e-6)
    assert report["groups"] == 2
    assert report["saturated_groups"] == 2


def test_analysis_uses_estimator_group_id_not_duplicate_prompt_text() -> None:
    rows = [
        _row("same-prompt", 0.0, "<report_error>a</report_error>", [], group_id="task-a"),
        _row("same-prompt", 1.0, "<report_error>b</report_error>", [], group_id="task-a"),
        _row("same-prompt", 1.0, "<report_error>c</report_error>", [], group_id="task-b"),
        _row("same-prompt", 1.0, "<report_error>d</report_error>", [], group_id="task-b"),
    ]
    report = analyze(rows, min_group_std=1e-6)
    assert report["groups"] == 2
    assert report["group_size_distribution"] == {2: 2}
    assert report["saturated_groups"] == 1
    assert report["grouping_source_rows"] == {"group_id": 4}
    assert report["grouping_exact"] is True


def test_historical_prompt_fallback_is_reported_as_inexact() -> None:
    rows = [
        _row("legacy", 0.0, "<report_error>a</report_error>", []),
        _row("legacy", 0.0, "<report_error>b</report_error>", []),
    ]
    report = analyze(rows, min_group_std=1e-6)
    assert report["grouping_source_rows"] == {"prompt_fallback": 2}
    assert report["grouping_exact"] is False
