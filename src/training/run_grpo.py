#!/usr/bin/env python3
"""OVAL-MCP GRPO 训练入口。

用法:
    OVAL_BETA=0.25 python src/training/run_grpo.py \\
        actor_rollout_ref.model.path=models/Qwen3-4B \\
        ...
"""

import os
import sys
from pathlib import Path

from loguru import logger

# 确保项目在路径中
PROJECT_DIR = str(Path(__file__).resolve().parent.parent.parent)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, "verl"))


def _maybe_run_pre_check(hp) -> None:
    """E4 启动前：先跑通用长度预检（默认开启），再跑 LiveMCP 专属的
    3:3:3 group 完整性（precheck=True 才跑，离线校验代价高）。"""
    from src.training.length_check import (
        assert_e4_group_integrity,
        maybe_run_length_check,
        parse_data_args_from_argv,
    )

    # 长度预检默认开启，由 length_check 自己处理 LIVEMCP_SKIP_LENGTH_CHECK
    maybe_run_length_check(sys.argv[1:])

    if not hp.precheck:
        return
    args = parse_data_args_from_argv(sys.argv[1:])
    train = args.get("train_files")
    val = args.get("val_files")
    model_path = args.get("model_path")
    limit = args.get("max_prompt_length", 10240)
    if train and model_path:
        assert_e4_group_integrity(train, model_path, limit, "train")
    if val and model_path:
        assert_e4_group_integrity(val, model_path, limit, "val")


def main() -> None:
    # ── P1-9: 从环境变量解析统一配置，导出到 env（Ray worker 继承） ──
    from src.training.livemcp_hyperparams import LiveMCPHyperparams
    hp = LiveMCPHyperparams.from_env()
    hp.export_env()
    logger.info("LiveMCP 超参配置:\n" + hp.summary())
    # 将配置保存到 LambdaState 路径相邻位置，供 wandb 等外部工具读取
    config_dump_path = os.path.join(
        os.path.dirname("/tmp/ovalmcp_lambda_state.json"),
        "livemcp_config.json",
    )
    try:
        import json as _json
        os.makedirs(os.path.dirname(config_dump_path), exist_ok=True)
        with open(config_dump_path, "w") as f:
            _json.dump(hp.to_dict(), f, indent=2, ensure_ascii=False)
    except Exception:
        pass

    beta = hp.beta
    logger.info(f"OVAL-MCP GRPO 训练入口 | beta={beta}")

    # 训练前可选：模拟 verl 的 prompt 过滤，验证 group 完整性
    if hp.precheck:
        _maybe_run_pre_check(hp)

    # 注册 agent loop（必须在 verl 启动前 import）
    from src.agent_loop.livemcp_oval_loop import LiveMCPOvalLoop  # noqa: F401
    logger.info("Agent loop LiveMCPOvalLoop 已注册")

    # 注册 livemcp_grpo estimator + patch verl 传递 non_tensor_batch
    # 主进程注册一次，便于 fail-fast；ray actor 内还需重新注册（见下面 LiveMCPTaskRunner）
    from src.training.register_estimator import register_livemcp_estimator
    register_livemcp_estimator()

    # ── 初始化 LambdaState（lambda_safe file-backed 共享状态） ──
    from src.oval_mcp.training.lambda_state import LambdaState, DEFAULT_STATE_PATH
    # 每次训练从干净状态开始（OVAL_KEEP_LAMBDA=1 保留上次状态）
    if not hp.keep_lambda and os.path.exists(DEFAULT_STATE_PATH):
        LambdaState.reset(DEFAULT_STATE_PATH)
    lambda_state = LambdaState.load_or_default()
    lambda_state.save()
    logger.info(f"lambda_safe 初始化: {lambda_state.lambda_safe} (path={DEFAULT_STATE_PATH})")

    # ray TaskRunner 跑在独立 actor 进程，主进程注册的 dict / monkey-patch 不会带过去。
    # 通过 task_runner_class hook 在 actor 进程里再注册一次。
    import hydra
    import ray
    from verl.trainer.main_ppo import run_ppo

    from src.training.livemcp_task_runner import LiveMCPTaskRunner

    @hydra.main(config_path="../../verl/verl/trainer/config", config_name="ppo_trainer", version_base=None)
    def _entry(config):
        # 防止系统默认 temp dir 路径过长导致 AF_UNIX socket path 超限
        import tempfile
        ray_tmp_dir = hp.ray_tmpdir
        os.makedirs(ray_tmp_dir, exist_ok=True)
        os.environ.setdefault("TMPDIR", "/tmp/ssgrpo_tmp")
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
