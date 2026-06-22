"""SchemaShift GRPO 训练入口。

替代直接调用 verl.trainer.main_ppo，注入 SchemaShiftTaskRunner。

Usage:
    python scripts/train_grpo.py [hydra overrides...]
"""

import os
import sys
import tempfile
import logging
import warnings

# 确保项目根目录在 PYTHONPATH 中
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# vLLM 0.11 CuMemAllocator is incompatible with PyTorch expandable segments.
os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)

# ---- 抑制已知无害警告 ----
# 1. transformers: fix_mistral_regex 对 Qwen2 tokenizer 的误报（vLLM 内部加载不走 hf_tokenizer）
#    用 warnings filter 而非仅 logging filter，确保 Ray worker 子进程也生效
warnings.filterwarnings("ignore", message=r".*incorrect regex pattern.*")
# 2. transformers logger: torch_dtype deprecated（verl 上游兼容性代码）
# 3. torch profiler tool config warning
class _SuppressKnownWarnings(logging.Filter):
    _suppressed = (
        "`torch_dtype` is deprecated",
        "Torch profiler tool config is not fully supported",
    )

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return not any(s in msg for s in self._suppressed)


logging.getLogger("transformers").addFilter(_SuppressKnownWarnings())
# FSDP state_dict_type FutureWarning
warnings.filterwarnings("ignore", message=r".*FSDP\.state_dict_type\(\).*", category=FutureWarning)
# Flash Attention 2 dtype warning（model_dtype=bfloat16 后不再触发，但保留兜底）
warnings.filterwarnings("ignore", message=r".*Flash Attention 2 only supports.*")

import hydra
import ray
from omegaconf import OmegaConf, open_dict

from src.training.length_check import maybe_run_length_check
from src.training.schemashift_task_runner import SchemaShiftTaskRunner
from verl.trainer.main_ppo import run_ppo


def _ensure_short_ray_temp_dir(config) -> str:
    """Set a short Ray temp dir to avoid AF_UNIX socket path length failures."""
    ray_init = config.ray_kwargs.get("ray_init", {})
    configured = ray_init.get("_temp_dir")
    ray_tmp_dir = configured or os.environ.get("SCHEMASHIFT_RAY_TMPDIR") or "/tmp/ssgrpo_ray"
    os.makedirs(ray_tmp_dir, exist_ok=True)

    # Ray also consults tempfile in some paths; keep it short for subprocesses.
    os.environ.setdefault("TMPDIR", "/tmp/ssgrpo_tmp")
    os.environ.setdefault("RAY_TMPDIR", ray_tmp_dir)
    os.makedirs(os.environ["TMPDIR"], exist_ok=True)
    tempfile.tempdir = os.environ["TMPDIR"]

    if not configured:
        with open_dict(config):
            OmegaConf.update(config, "ray_kwargs.ray_init._temp_dir", ray_tmp_dir, merge=True, force_add=True)
    return ray_tmp_dir


@hydra.main(config_path="../verl/verl/trainer/config", config_name="ppo_trainer", version_base=None)
def main(config):
    """SchemaShift GRPO 训练入口。"""
    # 长度预检：拦截超长 prompt，避免 verl 静默过滤导致 batch 缩水
    maybe_run_length_check(sys.argv[1:])

    ray_tmp_dir = _ensure_short_ray_temp_dir(config)
    print(f"SchemaShift Ray temp dir: {ray_tmp_dir}")

    # 使用 SchemaShiftTaskRunner
    task_runner_class = ray.remote(num_cpus=1)(SchemaShiftTaskRunner)
    run_ppo(config, task_runner_class=task_runner_class)


if __name__ == "__main__":
    main()
