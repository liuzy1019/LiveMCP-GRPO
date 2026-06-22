#!/usr/bin/env python3
"""E4 SchemaShift-GRPO 训练入口。

用法:
    SCHEMASHIFT_BETA=0.25 python -m src.training.run_exp4 \
        actor_rollout_ref.model.path=Qwen/Qwen2.5-1.5B-Instruct \
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
    limit = args.get("max_prompt_length", 2048)
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
    from src.agent_loop.bfcl_agent_loop import BFCLAgentLoop  # noqa: F401
    print("  Agent loop BFCLAgentLoop 已注册")

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
            from src.agent_loop.bfcl_agent_loop import BFCLAgentLoop  # noqa: F401
            from src.training.register_estimator import register_schemashift_estimator
            register_schemashift_estimator()
            return super().run(config)

    @hydra.main(config_path="../../verl/verl/trainer/config", config_name="ppo_trainer", version_base=None)
    def _entry(config):
        task_runner_class = ray.remote(num_cpus=1)(SchemaShiftTaskRunner)
        run_ppo(config, task_runner_class=task_runner_class)

    _entry()


if __name__ == "__main__":
    main()