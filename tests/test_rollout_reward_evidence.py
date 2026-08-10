import numpy as np

from verl.trainer.ppo.ray_trainer import _build_reward_replay_infos


def test_reward_replay_info_merges_row_and_runtime_evidence() -> None:
    batch = {
        "extra_info": np.array([
            {"task_id": "a", "allowed_terminal_actions": ["final_answer"]},
            {"task_id": "b", "allowed_terminal_actions": ["report_error"]},
        ], dtype=object),
        "audit_events": np.array([[{"action_type": "final_answer"}], []], dtype=object),
        "final_state": np.array([{"x": 1}, {"x": 2}], dtype=object),
        "trajectory_integrity_ok": np.array([True, False], dtype=object),
        "reward_profile": np.array(["prove_baseline", "prove_baseline"], dtype=object),
        "unrelated": np.array(["drop", "drop"], dtype=object),
    }
    replay = _build_reward_replay_infos(batch)
    assert replay[0]["task_id"] == "a"
    assert replay[0]["audit_events"] == [{"action_type": "final_answer"}]
    assert replay[1]["final_state"] == {"x": 2}
    assert replay[1]["trajectory_integrity_ok"] is False
    assert replay[0]["reward_profile"] == "prove_baseline"
    assert "unrelated" not in replay[0]


def test_reward_replay_info_requires_original_metadata() -> None:
    assert _build_reward_replay_infos({"audit_events": np.array([], dtype=object)}) == []
