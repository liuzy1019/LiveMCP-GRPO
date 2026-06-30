"""
LiveMCP 统一超参配置 — 单点真源（Single Source of Truth）。

解决问题：
  - P1-9: 消融开关原分散在 os.environ 中，wandb 不会记录
  - P1-20: 配置分散在 Hydra / env / 代码常量三层，实验复现困难

使用方式:
  # 训练启动时（run_grpo.py）
  config = LiveMCPHyperparams.from_env()
  config.export_env()  # 导出到环境变量，Ray worker 继承
  logger.info(config.summary())

  # 消费端（oval_reward_fn.py / livemcp_oval_loop / estimator）
  from src.training.livemcp_hyperparams import get_config
  cfg = get_config()  # 从环境变量重建，Ray worker 安全

架构约束:
  - 奖励函数运行在 Ray worker 进程中，无法直接访问主进程的 Python 对象
  - 环境变量是跨进程传递配置的唯一可靠通道
  - 本模块作为「环境变量的类型安全包装 + 默认值文档」
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from typing import ClassVar


@dataclass
class LiveMCPHyperparams:
    """LiveMCP 训练的所有超参，单点定义默认值。"""

    # ── 奖励消融开关 ──────────────────────────────────────────────
    # I_shape: 启用 F_gamma 进度塑形项（0=关, 1=开）
    i_shape: int = 0
    # I_process: 启用 P_process 过程质量项（0=关, 1=开）
    i_process: int = 1
    # lambda_shape: F_gamma 权重系数
    lambda_shape: float = 0.5
    # lambda_process: P_process 权重系数
    lambda_process: float = 0.3
    # gamma: F_gamma 衰减因子（1.0 = 无衰减, 仅依赖终点状态）
    gamma: float = 1.0

    # ── 安全约束 ──────────────────────────────────────────────────
    # lambda_safe 初始值
    lambda_safe_default: float = 1.0
    # lambda_safe 学习率（dual ascent）
    alpha_lambda: float = 0.01
    # lambda_safe 目标安全阈值 epsilon
    lambda_epsilon: float = 0.05
    # lambda_safe 上界
    lambda_safe_max: float = 3.0
    # stall protection: 连续多少步 unsafe 后冻结
    k_stall: int = 10
    # stall protection: hat_C 超过此阈值触发 unsafe streak
    tau_unsafe_stall: float = 0.5

    # ── Advantage ─────────────────────────────────────────────────
    # beta: StratAdv 全局残差权重（0=纯层内，1=纯全局）
    beta: float = 0.25
    # min_stratum_size: 层内最小样本数
    min_stratum_size: int = 3
    # min_group_std: 饱和检测阈值
    min_group_std: float = 0.01

    # ── LATA ─────────────────────────────────────────────────────
    # lata_mode: "none" | "sqrt_l" | "norm"
    lata_mode: str = "none"

    # ── Task Reward 权重 ──────────────────────────────────────────
    # 见 OVAL-MCP §7.1
    w_val: float = 0.5
    w_cov: float = 0.5
    w_eff: float = 0.15
    w_name: float = 0.2
    w_arg: float = 0.1
    w_struct: float = 0.6
    w_exec: float = 0.4
    alpha_eff: float = 0.3
    beta_budget: float = 0.3

    # ── P_process 上限 ───────────────────────────────────────────
    p_max: float = 0.3

    # ── Agent Loop ────────────────────────────────────────────────
    suite_path: str = "configs/live_mcp/suite_mvp.yaml"
    domains: str = "calendar,shopping,banking,email,filesystem,payments,crm,issue_tracker,team_chat,food_delivery"

    # ── 训练流程 ──────────────────────────────────────────────────
    # 启动时是否重置 LambdaState
    keep_lambda: bool = False
    # 是否运行 E4 group 完整性预检
    precheck: bool = False
    # Ray temp dir（避免 AF_UNIX socket path 超限）
    ray_tmpdir: str = "/tmp/oval_ray"

    # ── 环境变量名映射表 ──────────────────────────────────────────
    _ENV_MAP: ClassVar[dict[str, str]] = {
        "i_shape":          "OVAL_I_SHAPE",
        "i_process":        "OVAL_I_PROCESS",
        "lambda_shape":     "OVAL_LAMBDA_SHAPE",
        "lambda_process":   "OVAL_LAMBDA_PROCESS",
        "gamma":            "OVAL_GAMMA",
        "alpha_lambda":     "OVAL_ALPHA_LAMBDA",
        "lambda_epsilon":   "OVAL_LAMBDA_EPSILON",
        "lambda_safe_max":  "OVAL_LAMBDA_SAFE_MAX",
        "beta":             "OVAL_BETA",
        "min_stratum_size": "OVAL_MIN_STRATUM_SIZE",
        "min_group_std":    "OVAL_MIN_GROUP_STD",
        "lata_mode":        "LIVEMCP_LATA",
        "p_max":            "OVAL_P_MAX",
        "suite_path":       "OVAL_SUITE_PATH",
        "domains":          "OVAL_DOMAINS",
        "keep_lambda":      "OVAL_KEEP_LAMBDA",
        "precheck":         "OVAL_PRECHECK",
        "ray_tmpdir":       "OVAL_RAY_TMPDIR",
    }

    @classmethod
    def from_env(cls, **overrides) -> "LiveMCPHyperparams":
        """从环境变量读取配置，overrides 优先级最高。"""
        kwargs: dict = {}
        for f in fields(cls):
            if f.name.startswith("_"):
                continue
            env_var = cls._ENV_MAP.get(f.name)
            if env_var:
                raw = os.environ.get(env_var)
                if raw is not None:
                    kwargs[f.name] = _coerce(raw, f.type)
        kwargs.update(overrides)
        return cls(**kwargs)

    def export_env(self) -> None:
        """将所有字段导出到环境变量，确保 Ray worker 进程继承。"""
        for f in fields(self):
            if f.name.startswith("_"):
                continue
            env_var = self._ENV_MAP.get(f.name)
            if env_var:
                os.environ[env_var] = str(getattr(self, f.name))

    def summary(self) -> str:
        """生成人类可读的配置摘要。"""
        lines = ["LiveMCP Hyperparams:"]
        sections = {
            "Reward": ["i_shape", "i_process", "lambda_shape", "lambda_process", "gamma"],
            "Safety": ["lambda_safe_default", "alpha_lambda", "lambda_epsilon", "lambda_safe_max", "k_stall", "tau_unsafe_stall"],
            "Advantage": ["beta", "min_stratum_size", "min_group_std"],
            "LATA": ["lata_mode"],
            "TaskReward": ["w_val", "w_cov", "w_eff", "w_name", "w_arg", "w_struct", "w_exec", "alpha_eff", "beta_budget"],
            "P_process": ["p_max"],
            "Loop": ["suite_path", "domains"],
            "Train": ["keep_lambda", "precheck", "ray_tmpdir"],
        }
        for section, keys in sections.items():
            items = [f"  {k}={getattr(self, k)}" for k in keys if hasattr(self, k)]
            if items:
                lines.append(f"  [{section}]")
                lines.extend(items)
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """转为纯字典，用于 JSON/YAML 序列化和 wandb 记录。"""
        return {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if not f.name.startswith("_")
        }


def _coerce(raw: str, field_type) -> object:
    """将环境变量字符串转为目标类型。"""
    origin = getattr(field_type, "__origin__", None)
    if field_type is bool or str(field_type) == "bool":
        return raw.lower() in ("true", "1", "yes")
    if field_type is int or str(field_type) == "int":
        return int(raw)
    if field_type is float or str(field_type) == "float":
        return float(raw)
    return raw


def get_config() -> LiveMCPHyperparams:
    """从当前环境变量重建配置（Ray worker 安全）。

    在所有消费端统一使用此函数，而非直接调用 os.environ.get。
    """
    return LiveMCPHyperparams.from_env()


__all__ = ["LiveMCPHyperparams", "get_config"]
