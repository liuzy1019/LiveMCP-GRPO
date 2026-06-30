#!/usr/bin/env python3
"""训练配置 —— 参考 PyTorch Lightning Trainer 风格。

集中管理所有训练配置项，支持：
- 单卡 / 多卡 FSDP / DeepSpeed 三种策略
- WandB + Console 双日志
- 自动生成实验名（日期 + 核心配置），目录隔离
- 环境变量覆盖

用法：
    from src.training.trainer_config import TrainerConfig, ExperimentManager

    config = TrainerConfig(
        model_path="models/Qwen3-4B",
        train_file="data/train.parquet",
        val_file="data/val.parquet",
        total_steps=200,
        strategy="fsdp",
        devices=4,
        use_wandb=True,
    )
    exp = ExperimentManager(config)
    exp.setup()
    # → experiments/oval-mcp-grpo/20260629_fsdp_4gpu_b32_lr1e-6/
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from loguru import logger

# ── 项目根目录 ──────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


# ══════════════════════════════════════════════════════════════════════
# TrainerConfig —— 对标 Lightning Trainer 的集中配置
# ══════════════════════════════════════════════════════════════════════

@dataclass
class TrainerConfig:
    """训练配置 —— 所有参数集中管理，可被环境变量覆盖。"""

    # ── 模型 ──
    model_path: str = "models/Qwen3-4B"

    # ── 数据 ──
    train_file: str = "data/train.parquet"
    val_file: str = "data/val.parquet"

    # ── 训练规模 ──
    total_steps: int = 200
    total_epochs: int = 1
    save_freq: int = 50
    val_before_train: bool = False
    test_freq: int = -1

    # ── Batch ──
    train_batch_size: int = 32
    mini_batch_size: int = 8
    micro_batch_size_per_gpu: int = 2
    val_batch_size: Optional[int] = None

    # ── Sequence ──
    max_prompt_length: int = 10240
    max_response_length: int = 2048

    # ── Rollout ──
    rollout_n: int = 8
    rollout_tp: int = 1
    max_num_seqs: int = 64
    temperature: float = 0.7
    top_p: float = 0.95
    log_prob_micro_batch: int = 1

    # ── 优化器 ──
    lr: float = 1e-6
    lr_warmup_ratio: float = 0.1
    grad_clip: float = 1.0
    ppo_epochs: int = 1

    # ── KL ──
    kl_coef: float = 0.01

    # ── 策略（单卡/多卡） ──
    strategy: str = "fsdp"  # "fsdp" | "deepspeed" | "ddp"
    devices: int = 1  # GPU 数量，1=单卡
    nnodes: int = 1

    # ── FSDP 专属 ──
    fsdp_param_offload: bool = False
    model_dtype: str = "bfloat16"

    # ── vLLM 专属 ──
    gpu_mem_util: float = 0.60
    enforce_eager: bool = False
    free_cache_engine: bool = True

    # ── 日志 & 实验管理 ──
    use_wandb: bool = False
    wandb_project: str = "oval-mcp-grpo"
    wandb_entity: Optional[str] = None
    wandb_tags: List[str] = field(default_factory=list)
    log_dir: str = "experiments"  # 所有实验的根目录
    seed: int = 42

    # ── 回调/验证 ──
    agent_loop: str = "livemcp_oval"
    reward_fn_path: str = "src/reward/oval_reward_fn.py"
    reward_fn_name: str = "compute_score"
    adv_estimator: str = "livemcp_grpo"

    # ── 恢复 ──
    resume: bool = False
    resume_from: Optional[str] = None

    # ── Misc ──
    debug: bool = False

    # ── 实验名（None = 自动生成） ──
    run_name: Optional[str] = None

    def generate_run_name(self) -> str:
        """根据配置自动生成实验名：{日期}_{strategy}_{GPU数}gpu_b{batch}_lr{学习率}

        示例: 20260629_fsdp_4gpu_b32_lr1e-6
        """
        if self.run_name is not None:
            return self.run_name

        date_str = datetime.now().strftime("%Y%m%d")

        def _fmt_lr(lr: float) -> str:
            """格式化学习率：1e-6 → 1e-6, 1.5e-6 → 1.5e-6"""
            s = f"{lr:.1e}"
            # 去掉 .0: 1.0e-06 → 1e-06, 5.0e-06 → 5e-06
            s = re.sub(r'\.0+(e)', r'\1', s)
            # 去掉指数前导零: e-06 → e-6
            s = re.sub(r'(e[+-])0+(\d+)', r'\1\2', s)
            return s

        parts = [
            date_str,
            self.strategy,
            f"{self.devices}gpu",
            f"b{self.train_batch_size}",
            f"lr{_fmt_lr(self.lr)}",
        ]

        # 如果开启了 WandB，加标记
        if self.use_wandb:
            parts.append("wb")

        return "_".join(parts)

    def to_hydra_overrides(self) -> List[str]:
        """生成 verl Hydra 命令行覆盖参数列表。"""
        val_batch = self.val_batch_size or self.train_batch_size
        max_num_batched_tokens = self.max_prompt_length + self.max_response_length
        run_name = self.generate_run_name()

        overrides = [
            f"data.train_files={self.train_file}",
            f"data.val_files={self.val_file}",
            f"data.max_prompt_length={self.max_prompt_length}",
            f"data.max_response_length={self.max_response_length}",
            f"data.train_batch_size={self.train_batch_size}",
            f"data.val_batch_size={val_batch}",
            "data.shuffle=True",
            "data.filter_overlong_prompts=True",
            "data.truncation=left",
            "data.reward_fn_key=data_source",
            "data.return_raw_chat=True",
            # 模型
            f"actor_rollout_ref.model.path={self.model_path}",
            # Actor
            f"actor_rollout_ref.actor.strategy={self.strategy}",
            f"actor_rollout_ref.actor.ppo_mini_batch_size={self.mini_batch_size}",
            f"actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu={self.micro_batch_size_per_gpu}",
            f"actor_rollout_ref.actor.ppo_epochs={self.ppo_epochs}",
            f"actor_rollout_ref.actor.grad_clip={self.grad_clip}",
            f"actor_rollout_ref.actor.optim.lr={self.lr}",
            f"actor_rollout_ref.actor.optim.lr_warmup_steps_ratio={self.lr_warmup_ratio}",
            f"actor_rollout_ref.actor.fsdp_config.param_offload={str(self.fsdp_param_offload).lower()}",
            f"actor_rollout_ref.actor.fsdp_config.model_dtype={self.model_dtype}",
            # Ref
            f"actor_rollout_ref.ref.fsdp_config.model_dtype={self.model_dtype}",
            f"actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu={self.log_prob_micro_batch}",
            f"actor_rollout_ref.ref.fsdp_config.param_offload={str(self.fsdp_param_offload).lower()}",
            # Rollout
            "actor_rollout_ref.rollout.name=vllm",
            f"actor_rollout_ref.rollout.tensor_model_parallel_size={self.rollout_tp}",
            f"actor_rollout_ref.rollout.gpu_memory_utilization={self.gpu_mem_util}",
            f"actor_rollout_ref.rollout.free_cache_engine={str(self.free_cache_engine).lower()}",
            f"actor_rollout_ref.rollout.enforce_eager={str(self.enforce_eager).lower()}",
            f"actor_rollout_ref.rollout.n={self.rollout_n}",
            f"actor_rollout_ref.rollout.temperature={self.temperature}",
            f"actor_rollout_ref.rollout.top_p={self.top_p}",
            f"actor_rollout_ref.rollout.prompt_length={self.max_prompt_length}",
            f"actor_rollout_ref.rollout.response_length={self.max_response_length}",
            f"actor_rollout_ref.rollout.max_num_batched_tokens={max_num_batched_tokens}",
            f"actor_rollout_ref.rollout.max_num_seqs={self.max_num_seqs}",
            "actor_rollout_ref.rollout.enable_prefix_caching=False",
            f"actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu={self.log_prob_micro_batch}",
            "actor_rollout_ref.rollout.mode=async",
            f"actor_rollout_ref.rollout.agent.default_agent_loop={self.agent_loop}",
            "actor_rollout_ref.rollout.agent.agent_loop_config_path=configs/agent_loop.yaml",
            f"actor_rollout_ref.rollout.agent.num_workers={self.devices}",
            # Algorithm
            f"algorithm.adv_estimator={self.adv_estimator}",
            "algorithm.use_kl_in_reward=True",
            "algorithm.kl_penalty=kl",
            "algorithm.kl_ctrl.type=fixed",
            f"algorithm.kl_ctrl.kl_coef={self.kl_coef}",
            # Reward
            f"custom_reward_function.path={self.reward_fn_path}",
            f"custom_reward_function.name={self.reward_fn_name}",
            # Trainer
            f"trainer.project_name={self.wandb_project}",
            f"trainer.experiment_name={run_name}",
            f"trainer.total_epochs={self.total_epochs}",
            f"trainer.total_training_steps={self.total_steps}",
            f"trainer.nnodes={self.nnodes}",
            f"trainer.n_gpus_per_node={self.devices}",
            f"trainer.save_freq={self.save_freq}",
            f"trainer.val_before_train={str(self.val_before_train).lower()}",
            f"trainer.test_freq={self.test_freq}",
            f"trainer.resume_mode={'auto' if self.resume else 'disable'}",
            "reward_model.enable=False",
        ]

        if self.resume_from:
            overrides.append(f"trainer.resume_from_path={self.resume_from}")

        return overrides

    def to_logger_list(self) -> str:
        """生成 verl trainer.logger 参数值。"""
        if self.use_wandb:
            return '["console","wandb"]'
        return '["console"]'

    @classmethod
    def from_env(cls, **overrides) -> "TrainerConfig":
        """从环境变量 + 手动覆盖创建配置。环境变量优先级 > 默认值 < 手动覆盖。"""
        env_map = {
            "model_path": "OVAL_MODEL_PATH",
            "train_file": "OVAL_TRAIN_FILE",
            "val_file": "OVAL_VAL_FILE",
            "total_steps": "OVAL_TOTAL_STEPS",
            "train_batch_size": "OVAL_TRAIN_BATCH_SIZE",
            "mini_batch_size": "OVAL_MINI_BATCH_SIZE",
            "micro_batch_size_per_gpu": "OVAL_MICRO_BATCH",
            "max_prompt_length": "OVAL_PROMPT_LENGTH",
            "max_response_length": "OVAL_RESPONSE_LENGTH",
            "rollout_n": "OVAL_ROLLOUT_N",
            "max_num_seqs": "OVAL_MAX_NUM_SEQS",
            "lr": "OVAL_LR",
            "lr_warmup_ratio": "OVAL_LR_WARMUP_RATIO",
            "kl_coef": "OVAL_KL_COEF",
            "ppo_epochs": "OVAL_PPO_EPOCHS",
            "grad_clip": "OVAL_GRAD_CLIP",
            "gpu_mem_util": "OVAL_GPU_MEM_UTIL",
            "fsdp_param_offload": "OVAL_ACTOR_PARAM_OFFLOAD",
            "rollout_tp": "OVAL_ROLLOUT_TP",
            "log_prob_micro_batch": "OVAL_LOG_PROB_MICRO_BATCH",
            "use_wandb": "OVAL_USE_WANDB",
            "wandb_project": "OVAL_WANDB_PROJECT",
            "wandb_entity": "OVAL_WANDB_ENTITY",
            "seed": "OVAL_SEED",
            "agent_loop": "OVAL_AGENT_LOOP",
            "adv_estimator": "OVAL_ADV_ESTIMATOR",
            "strategy": "OVAL_STRATEGY",
            "devices": "OVAL_DEVICES",
            "debug": "OVAL_DEBUG",
        }

        kwargs: Dict[str, Any] = {}
        for field_name, env_var in env_map.items():
            val = os.environ.get(env_var)
            if val is None:
                continue

            field_info = cls.__dataclass_fields__[field_name]
            field_type = field_info.type

            # 类型转换
            origin = getattr(field_type, "__origin__", None)
            if field_type is bool or str(field_type) == "bool":
                kwargs[field_name] = val.lower() in ("true", "1", "yes")
            elif field_type is int or str(field_type) == "int":
                kwargs[field_name] = int(val)
            elif field_type is float or str(field_type) == "float":
                kwargs[field_name] = float(val)
            elif origin is list or str(field_type).startswith("typing.List"):
                kwargs[field_name] = [t.strip() for t in val.split(",") if t.strip()]
            else:
                kwargs[field_name] = val

        kwargs.update(overrides)
        return cls(**kwargs)

    def to_dict(self) -> Dict[str, Any]:
        """转为纯字典，用于保存到 YAML/JSON。"""
        result = {}
        for f in self.__dataclass_fields__:
            val = getattr(self, f)
            if isinstance(val, Path):
                val = str(val)
            result[f] = val
        return result


# ══════════════════════════════════════════════════════════════════════
# ExperimentManager —— 对标 Lightning 的实验管理（目录隔离、保存配置）
# ══════════════════════════════════════════════════════════════════════

class ExperimentManager:
    """管理训练实验的生命周期：创建目录、保存配置、管理 checkpoint 和日志。

    目录结构:
        experiments/
        └── {project}/
            └── {YYYYMMDD}_{strategy}_{N}gpu_b{bs}_lr{lr}[_wb]/
                ├── config.yaml          # 训练配置快照
                ├── git_info.json        # git commit/branch
                ├── checkpoints/         # 模型 checkpoint（verl 自动写入）
                ├── logs/                # 训练日志
                └── wandb/               # WandB 本地缓存（由 WANDB_DIR 控制）

    示例: experiments/oval-mcp-grpo/20260629_fsdp_4gpu_b32_lr1e-6_wb/
    """

    def __init__(self, config: TrainerConfig):
        self.config = config
        self.run_dir: Optional[Path] = None

    @property
    def run_name(self) -> str:
        """实验名：自动从 config 生成。"""
        return self.config.generate_run_name()

    @property
    def run_dir_path(self) -> Path:
        """实验目录绝对路径。"""
        if self.run_dir is None:
            self.run_dir = (
                Path(self.config.log_dir)
                / self.config.wandb_project
                / self.run_name
            ).resolve()
        return self.run_dir

    def setup(self) -> Path:
        """创建实验目录结构，保存配置快照，返回 run_dir。"""
        run_dir = self.run_dir_path

        # 如果已存在同名实验，追加序号
        if run_dir.exists():
            for i in range(1, 100):
                alt = run_dir.with_name(f"{self.run_name}_{i}")
                if not alt.exists():
                    run_dir = alt
                    self.run_dir = run_dir
                    break

        for sub in ["checkpoints", "logs"]:
            (run_dir / sub).mkdir(parents=True, exist_ok=True)

        # 保存配置快照
        config_yaml = run_dir / "config.yaml"
        with open(config_yaml, "w") as f:
            yaml.dump(self.config.to_dict(), f, default_flow_style=False, allow_unicode=True)

        # 保存 git 信息
        self._save_git_info(run_dir)

        logger.info(f"实验目录已创建: {run_dir}")
        logger.info(f"  配置已保存: {config_yaml}")
        return run_dir

    def _save_git_info(self, run_dir: Path) -> None:
        """保存当前 git commit/branch/diff 信息。"""
        git_info: Dict[str, str] = {}
        for key, cmd in [
            ("commit", ["git", "rev-parse", "HEAD"]),
            ("branch", ["git", "rev-parse", "--abbrev-ref", "HEAD"]),
            ("diff", ["git", "diff", "--stat"]),
        ]:
            try:
                git_info[key] = subprocess.check_output(
                    cmd, cwd=PROJECT_ROOT, text=True
                ).strip()
            except Exception:
                git_info[key] = "unknown"

        with open(run_dir / "git_info.json", "w") as f:
            json.dump(git_info, f, indent=2, ensure_ascii=False)

    @staticmethod
    def list_experiments(log_dir: str = "experiments", project: Optional[str] = None) -> List[Path]:
        """列出已有的实验目录。"""
        root = Path(log_dir)
        if not root.exists():
            return []

        if project:
            project_dir = root / project
            if project_dir.exists():
                return sorted(
                    [d for d in project_dir.iterdir() if d.is_dir()],
                    reverse=True,
                )
            return []

        results = []
        for proj_dir in sorted(root.iterdir()):
            if proj_dir.is_dir():
                results.extend(
                    sorted(
                        [d for d in proj_dir.iterdir() if d.is_dir()],
                        reverse=True,
                    )
                )
        return results

    @staticmethod
    def latest_experiment(
        log_dir: str = "experiments", project: Optional[str] = None
    ) -> Optional[Path]:
        """返回最近的实验目录。"""
        exps = ExperimentManager.list_experiments(log_dir, project)
        return exps[0] if exps else None


# ── 便捷函数 ────────────────────────────────────────────────────────

def resolve_gpu_info(requested_devices: Optional[int] = None) -> Tuple[int, str, str]:
    """自动检测 GPU 信息（依赖 gpu_config.sh 已设置 CUDA_VISIBLE_DEVICES）。

    Returns:
        (num_devices, gpu_ids, gpu_model_name)
    """
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cvd:
        ids_list = cvd.split(",")
        if requested_devices and requested_devices < len(ids_list):
            ids_list = ids_list[:requested_devices]
        num = len(ids_list)
        ids_str = cvd if requested_devices is None else ",".join(ids_list)
    else:
        try:
            result = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index,name", "--format=csv,noheader"],
                text=True,
            )
            lines = [l.strip() for l in result.strip().split("\n") if l.strip()]
            num = requested_devices or len(lines)
            ids_str = ",".join(str(i) for i in range(num))
        except Exception:
            num = requested_devices or 1
            ids_str = ",".join(str(i) for i in range(num))

    try:
        gpu_name = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True,
        ).strip().split("\n")[0]
    except Exception:
        gpu_name = "unknown"

    return num, ids_str, gpu_name


def print_config_summary(config: TrainerConfig, gpu_count: int, gpu_model: str) -> None:
    """打印训练配置摘要（类似 Lightning 的启动横幅）。"""
    run_name = config.generate_run_name()
    strategy_label = {
        "fsdp": f"FSDP ({gpu_count} GPUs)",
        "deepspeed": f"DeepSpeed ({gpu_count} GPUs)",
        "ddp": f"DDP ({gpu_count} GPUs)",
    }.get(config.strategy, config.strategy)

    separator = "=" * 60
    print(f"\n{separator}")
    print(f"  OVAL-MCP GRPO Training")
    print(f"{separator}")
    print(f"  Experiment:   {run_name}")
    print(f"  Project:      {config.wandb_project}")
    print(f"  Strategy:     {strategy_label}")
    print(f"  GPU:          {gpu_count}x {gpu_model}")
    print(f"  Model:        {config.model_path}")
    print(f"  Train:        {config.train_file} → {config.total_steps} steps")
    print(f"  Val:          {config.val_file}")
    print(f"  Batch:        {config.train_batch_size} (mini={config.mini_batch_size}, micro={config.micro_batch_size_per_gpu})")
    print(f"  LR:           {config.lr}")
    print(f"  KL coef:      {config.kl_coef}")
    print(f"  Rollout N:    {config.rollout_n}")
    print(f"  Seq:          prompt={config.max_prompt_length} resp={config.max_response_length}")
    print(f"  WandB:        {'on' if config.use_wandb else 'off'}")
    print(f"  Log Dir:      {config.log_dir}/{config.wandb_project}/{run_name}/")
    print(f"{separator}\n")
