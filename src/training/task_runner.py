"""LiveMCPTaskRunner — 在 ray actor 内注册 estimator 的 TaskRunner。

继承 verl 的 TaskRunner，在 run() 开始时注册 livemcp_grpo estimator。
由于 verl 的 compute_advantage 在 driver process（TaskRunner.run 所在进程）执行，
在这里注册 patch 就能确保 estimator 在正确的进程中生效。

Usage:
    # 在启动脚本中指定 task_runner_class
    python -c "
    from verl.trainer.main_ppo import run_ppo
    from src.training.task_runner import LiveMCPTaskRunner
    import ray
    task_runner_class = ray.remote(num_cpus=1)(LiveMCPTaskRunner)
    run_ppo(config, task_runner_class=task_runner_class)
    "
"""

from loguru import logger

from verl.trainer.main_ppo import TaskRunner


class LiveMCPTaskRunner(TaskRunner):
    """Register the OVAL estimator in the Ray driver process."""

    def run(self, config):
        """注册 estimator 后执行标准训练流程。"""
        estimator_name = str(config.algorithm.adv_estimator)
        if estimator_name == "livemcp_grpo":
            from src.training.estimator import register_livemcp_estimator

            success = register_livemcp_estimator(
                config={"use_livemcp": True}
            )
            if not success:
                raise RuntimeError(
                    "LiveMCPTaskRunner: estimator registration failed inside Ray actor"
                )
            logger.info("LiveMCPTaskRunner: livemcp_grpo estimator registered")
        elif estimator_name == "grpo":
            logger.info("LiveMCPTaskRunner: using verl standard GRPO estimator")
        else:
            raise RuntimeError(
                f"unsupported estimator for LiveMCP training: {estimator_name!r}"
            )

        # 执行标准训练流程
        return super().run(config)
