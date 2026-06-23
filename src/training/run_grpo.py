#!/usr/bin/env python3
"""SchemaShift-GRPO 训练入口。

用法:
    SCHEMASHIFT_BETA=0.25 python src/training/run_grpo.py \
        actor_rollout_ref.model.path=models/Qwen3-4B \
        ...
"""

import os
import sys
from pathlib import Path

# 确保项目在路径中
PROJECT_DIR = str(Path(__file__).resolve().parent.parent.parent)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, "verl"))


def _maybe_run_pre_check() -> None:
    """E4 启动前：先跑通用长度预检（默认开启），再跑 SchemaShift 专属的
    3:3:3 group 完整性（SCHEMASHIFT_PRECHECK=1 才跑，离线校验代价高）。"""
    from src.training.length_check import (
        assert_e4_group_integrity,
        maybe_run_length_check,
        parse_data_args_from_argv,
    )

    # 长度预检默认开启，由 length_check 自己处理 SCHEMASHIFT_SKIP_LENGTH_CHECK
    maybe_run_length_check(sys.argv[1:])

    # group 完整性是 E4 独有，沿用原 SCHEMASHIFT_PRECHECK 开关
    if os.environ.get("SCHEMASHIFT_PRECHECK", "0") != "1":
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
    beta = float(os.environ.get("SCHEMASHIFT_BETA", "0.25"))
    print(f"  E4 SchemaShift-GRPO 训练入口 | beta={beta}")

    # 训练前可选：模拟 verl 的 prompt 过滤，验证 group 完整性（SCHEMASHIFT_PRECHECK=1 触发）
    _maybe_run_pre_check()

    # 注册 agent loop（必须在 verl 启动前 import）
    from src.agent_loop.schemashift_replay_loop import SchemaShiftReplayLoop  # noqa: F401
    print("  Agent loop SchemaShiftReplayLoop 已注册")

    # 保留 BFCLAgentLoop 注册（兼容旧配置）
    try:
        from src.agent_loop.bfcl_agent_loop import BFCLAgentLoop  # noqa: F401
    except ImportError:
        pass

    # 注册 schemashift_grpo estimator + patch verl 传递 non_tensor_batch
    # 主进程注册一次，便于 fail-fast；ray actor 内还需重新注册（见下面 SchemaShiftTaskRunner）
    from src.training.register_estimator import register_schemashift_estimator
    register_schemashift_estimator()

    # ray TaskRunner 跑在独立 actor 进程，主进程注册的 dict / monkey-patch 不会带过去。
    # 通过 task_runner_class hook 在 actor 进程里再注册一次。
    import hydra
    import ray
    from verl.trainer.main_ppo import TaskRunner, run_ppo

    class SchemaShiftTaskRunner(TaskRunner):
        def run(self, config):
            from src.agent_loop.schemashift_replay_loop import SchemaShiftReplayLoop  # noqa: F401
            try:
                from src.agent_loop.bfcl_agent_loop import BFCLAgentLoop  # noqa: F401
            except ImportError:
                pass
            from src.training.register_estimator import register_schemashift_estimator
            register_schemashift_estimator()
            return super().run(config)

    @hydra.main(config_path="../../verl/verl/trainer/config", config_name="ppo_trainer", version_base=None)
    def _entry(config):
        # 防止系统默认 temp dir 路径过长导致 AF_UNIX socket path 超限
        import tempfile
        ray_tmp_dir = os.environ.get("SCHEMASHIFT_RAY_TMPDIR", "/tmp/ssgrpo_ray")
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

        task_runner_class = ray.remote(num_cpus=1)(SchemaShiftTaskRunner)
        try:
            run_ppo(config, task_runner_class=task_runner_class)
        finally:
            if ray.is_initialized():
                ray.shutdown()

    _entry()


if __name__ == "__main__":
    main()
