"""SchemaShiftTaskRunner — 在 ray actor 内注册 estimator 的 TaskRunner。

继承 verl 的 TaskRunner，在 run() 开始时注册 schemashift_grpo estimator。
由于 verl 的 compute_advantage 在 driver process（TaskRunner.run 所在进程）执行，
在这里注册 patch 就能确保 estimator 在正确的进程中生效。

Usage:
    # 在启动脚本中指定 task_runner_class
    python -c "
    from verl.trainer.main_ppo import run_ppo
    from src.training.schemashift_task_runner import SchemaShiftTaskRunner
    import ray
    task_runner_class = ray.remote(num_cpus=1)(SchemaShiftTaskRunner)
    run_ppo(config, task_runner_class=task_runner_class)
    "
"""

from loguru import logger

from verl.trainer.main_ppo import TaskRunner


class SchemaShiftTaskRunner(TaskRunner):
    """在 ray actor 内注册 schemashift_grpo estimator 的 TaskRunner。"""

    def run(self, config):
        """注册 estimator 后执行标准训练流程。"""
        # 注册 schemashift_grpo estimator + monkey-patch compute_advantage
        from src.training.register_estimator import register_schemashift_estimator

        success = register_schemashift_estimator(
            config={"use_schemashift": True}
        )
        if success:
            logger.info("SchemaShiftTaskRunner: estimator 注册成功")
        else:
            logger.warning("SchemaShiftTaskRunner: estimator 注册失败，将使用标准 GRPO")

        # 执行标准训练流程
        return super().run(config)
