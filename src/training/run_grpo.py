#!/usr/bin/env python3
"""OVAL-MCP GRPO 训练入口。

用法:
    OVAL_REWARD_PROFILE=oval_full python src/training/run_grpo.py \\
        actor_rollout_ref.model.path=models/Qwen/Qwen3-4B-Instruct-2507 \\
        ...
"""

import os
import sys
from pathlib import Path

from loguru import logger

# 确保项目在路径中
PROJECT_DIR = str(Path(__file__).resolve().parent.parent.parent)
VENDORED_VERL_DIR = os.path.join(PROJECT_DIR, "verl")
for path in (PROJECT_DIR, VENDORED_VERL_DIR):
    while path in sys.path:
        sys.path.remove(path)
# Project modules must win over the vendored repository's generic packages
# (notably both repositories contain a top-level ``scripts`` package).
sys.path.insert(0, PROJECT_DIR)
sys.path.insert(1, VENDORED_VERL_DIR)


def _bind_profile_estimator(profile: str, argv: list[str]) -> str:
    """Bind the Hydra estimator to the declared training profile."""
    expected = "grpo" if profile == "prove_baseline" else "livemcp_grpo"
    prefix = "algorithm.adv_estimator="
    normalized = [arg.lstrip("+") for arg in argv]
    values = [arg[len(prefix):] for arg in normalized if arg.startswith(prefix)]
    if len(set(values)) > 1:
        raise ValueError(f"conflicting advantage estimator overrides: {values}")
    if values and values[0] != expected:
        raise ValueError(
            f"reward profile {profile!r} requires adv_estimator={expected!r}, "
            f"got {values[0]!r}"
        )
    if not values:
        argv.append(f"{prefix}{expected}")
    return expected


def _validate_verl_runtime() -> None:
    """Fail early when the vendored verl checkout is not installed completely."""
    try:
        import verl.trainer.main_ppo  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "vendored verl runtime is incomplete in the active Python environment; "
            "install it with that interpreter via `python -m pip install -e ./verl` "
            f"before training (original error: {exc})"
        ) from exc


def _run_length_precheck() -> None:
    """Run the production prompt-length preflight before verl starts."""
    from src.training.length_check import (
        maybe_run_length_check,
    )

    maybe_run_length_check(sys.argv[1:])


def main() -> None:
    # ── P1-9: 从环境变量解析统一配置，导出到 env（Ray worker 继承） ──
    from src.training.hyperparams import LiveMCPHyperparams
    hp = LiveMCPHyperparams.from_env()
    hp.export_env()
    estimator_name = _bind_profile_estimator(hp.reward_profile, sys.argv)
    _validate_verl_runtime()
    logger.info("LiveMCP 超参配置:\n" + hp.summary())
    from src.oval_mcp.training.lambda_state import LambdaState
    # 将配置保存到 LambdaState 路径相邻位置，供 wandb 等外部工具读取
    lambda_state_dir = os.path.dirname(hp.lambda_state_path) or "."
    config_dump_path = os.path.join(lambda_state_dir, "livemcp_config.json")
    import json as _json
    os.makedirs(lambda_state_dir, exist_ok=True)
    with open(config_dump_path, "w") as f:
        _json.dump(hp.to_dict(), f, indent=2, ensure_ascii=False)

    logger.info(f"OVAL-MCP GRPO 训练入口 | adv_estimator={estimator_name}")

    # Prompt-length preflight always executes unless explicitly skipped.
    _run_length_precheck()

    # 注册 agent loop（必须在 verl 启动前 import）
    from src.agent_loop.livemcp_oval_loop import LiveMCPOvalLoop  # noqa: F401
    logger.info("Agent loop LiveMCPOvalLoop 已注册")

    # Only the OVAL extension needs the custom estimator and non-tensor bridge.
    # The prove_baseline profile uses verl's standard GRPO implementation.
    if estimator_name == "livemcp_grpo":
        from src.training.estimator import register_livemcp_estimator
        if not register_livemcp_estimator():
            raise RuntimeError(
                "LiveMCP estimator registration failed; aborting before Ray startup"
            )

    # ── 初始化 LambdaState（lambda_safe file-backed 共享状态） ──
    # 每次训练从干净状态开始（OVAL_KEEP_LAMBDA=1 保留上次状态）
    if (
        hp.reward_profile == "prove_baseline" or not hp.keep_lambda
    ) and os.path.exists(hp.lambda_state_path):
        LambdaState.reset(hp.lambda_state_path)
    if hp.reward_profile == "prove_baseline" or not hp.keep_lambda:
        lambda_state = LambdaState.load_or_default(
            path=hp.lambda_state_path,
            lambda_safe=hp.lambda_safe_default,
            alpha_lambda=hp.alpha_lambda,
            epsilon=hp.lambda_epsilon,
            lambda_safe_max=hp.lambda_safe_max,
        )
    else:
        lambda_state = LambdaState.load_or_default(path=hp.lambda_state_path)
    lambda_state.save()
    logger.info(
        f"lambda_safe 初始化: {lambda_state.lambda_safe} "
        f"(path={hp.lambda_state_path})"
    )

    # ray TaskRunner 跑在独立 actor 进程，主进程注册的 dict / monkey-patch 不会带过去。
    # 通过 task_runner_class hook 在 actor 进程里再注册一次。
    import hydra
    import ray
    from verl.trainer.main_ppo import run_ppo

    from src.training.task_runner import LiveMCPTaskRunner

    @hydra.main(config_path="../../verl/verl/trainer/config", config_name="ppo_trainer", version_base=None)
    def _entry(config):
        # 防止系统默认 temp dir 路径过长导致 AF_UNIX socket path 超限
        import tempfile
        ray_tmp_dir = hp.ray_tmpdir
        os.makedirs(ray_tmp_dir, exist_ok=True)
        os.environ.setdefault("TMPDIR", "/tmp/oval_tmp")
        os.environ.setdefault("RAY_TMPDIR", ray_tmp_dir)
        os.makedirs(os.environ["TMPDIR"], exist_ok=True)
        tempfile.tempdir = os.environ["TMPDIR"]

        from omegaconf import OmegaConf, open_dict
        ray_init = config.ray_kwargs.get("ray_init", {})
        if not ray_init.get("_temp_dir"):
            with open_dict(config):
                OmegaConf.update(
                    config, "ray_kwargs.ray_init._temp_dir",
                    ray_tmp_dir, merge=True, force_add=True,
                )

        task_runner_class = ray.remote(num_cpus=1)(LiveMCPTaskRunner)
        try:
            run_ppo(config, task_runner_class=task_runner_class)
        finally:
            if ray.is_initialized():
                ray.shutdown()

    _entry()


if __name__ == "__main__":
    main()
