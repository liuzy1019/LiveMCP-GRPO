from __future__ import annotations

from types import SimpleNamespace

import pytest

import verl.trainer.main_ppo as main_ppo


class _RunnerHandle:
    def __init__(self) -> None:
        self.run = SimpleNamespace(remote=lambda config: ("run", config))


class _TaskRunnerClass:
    def __init__(self, runner: _RunnerHandle) -> None:
        self.runner = runner

    def remote(self) -> _RunnerHandle:
        return self.runner


def _config() -> SimpleNamespace:
    return SimpleNamespace(
        ray_kwargs={
            "ray_init": {},
            "timeline_json_file": None,
        },
        transfer_queue=SimpleNamespace(enable=False),
        global_profiler=SimpleNamespace(tool=None, get=lambda *args: None),
    )


@pytest.mark.parametrize("run_error", [None, RuntimeError("training failed")])
def test_run_ppo_always_kills_owned_task_runner(monkeypatch, run_error) -> None:
    runner = _RunnerHandle()
    killed: list[tuple[object, bool]] = []

    monkeypatch.setattr(main_ppo.ray, "is_initialized", lambda: True)
    monkeypatch.setattr(
        main_ppo.ray,
        "get",
        lambda ref: (_ for _ in ()).throw(run_error) if run_error else None,
    )
    monkeypatch.setattr(
        main_ppo.ray,
        "kill",
        lambda actor, no_restart: killed.append((actor, no_restart)),
    )
    monkeypatch.setattr(main_ppo, "is_cuda_available", False)

    if run_error:
        with pytest.raises(RuntimeError, match="training failed"):
            main_ppo.run_ppo(
                _config(),
                task_runner_class=_TaskRunnerClass(runner),
            )
    else:
        main_ppo.run_ppo(
            _config(),
            task_runner_class=_TaskRunnerClass(runner),
        )

    assert killed == [(runner, True)]
