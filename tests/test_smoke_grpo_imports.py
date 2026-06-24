"""
依赖契约测试（P1-2）。

目的：在 pytest 阶段就发现版本错位 / 关键符号丢失，避免训练跑到 step 1 才报错。
策略：装了的依赖必须落在 [min, max) 兼容窗口；没装的依赖（verl/vllm 在
非训练机上不一定有）跳过，不算失败。

兼容窗口锚定 arl 环境实测通过的版本组合（见 AGENTS.md 和 docs/project_plan.md）：
torch 2.8 / vllm 0.11 / verl 0.6.1 / flash_attn 2.7.3 / transformers 4.57.6 /
ray 2.x."""

from __future__ import annotations

import importlib

import pytest
from packaging.version import Version


########## helpers


def _version_of(mod_name: str) -> Version | None:
    """优先 importlib.metadata，再回退到 mod.__version__。"""
    try:
        from importlib.metadata import version as _v
        return Version(_v(mod_name))
    except Exception:
        try:
            mod = importlib.import_module(mod_name)
            v = getattr(mod, "__version__", None)
            return Version(v) if v else None
        except Exception:
            return None


def _require_in_range(mod_name: str, min_v: str, max_v: str):
    """装了就校验版本，没装就 skip。max_v 是开区间。"""
    pytest.importorskip(mod_name, reason=f"{mod_name} 未安装，跳过版本契约校验")
    v = _version_of(mod_name)
    assert v is not None, f"{mod_name} 已安装但拿不到版本号"
    lo, hi = Version(min_v), Version(max_v)
    assert lo <= v < hi, (
        f"{mod_name}=={v} 不在兼容窗口 [{min_v}, {max_v}) 内。"
        f"如确认要升级，请同步更新 AGENTS.md / docs/project_plan.md 的版本表，再放宽本测试。"
    )


def _require_symbols(mod_path: str, symbols: list[str]):
    """import mod_path，并断言每个 symbol 都存在。装的 mod 缺 symbol = 真失败。"""
    pytest.importorskip(mod_path, reason=f"{mod_path} 未安装，跳过符号契约校验")
    mod = importlib.import_module(mod_path)
    missing = [s for s in symbols if not hasattr(mod, s)]
    assert not missing, f"{mod_path} 缺少符号 {missing}，可能是版本不兼容"


########## 核心训练栈版本窗口


def test_torch_version():
    _require_in_range("torch", "2.6.0", "3.0.0")


def test_transformers_version():
    _require_in_range("transformers", "4.55.0", "5.0.0")


def test_vllm_version():
    # vllm 0.10 是 verl 0.6.1 接受的下限；0.12 起 LLMEngine API 又会变
    _require_in_range("vllm", "0.10.0", "0.12.0")


def test_verl_version():
    # verl editable 安装在 ./verl，pin 在 0.6.x
    _require_in_range("verl", "0.6.0", "0.7.0")


def test_ray_version():
    _require_in_range("ray", "2.10.0", "3.0.0")


def test_flashinfer_optional_or_disabled():
    """flashinfer 装与不装都行，但装了 0.6.4 就必须配合 VLLM_USE_FLASHINFER_SAMPLER=0
    避开 JIT。这里只校验：如果装了 0.6.x，VLLM_ATTENTION_BACKEND 默认值不能是 FLASHINFER。
    实际开关在 sh 里设，本测试只做事实记录，不做强约束。"""
    v = _version_of("flashinfer_python") or _version_of("flashinfer")
    if v is None:
        pytest.skip("flashinfer 未安装，跳过")
    # 不强约束，只确保不会被误升到 0.7+（verl 0.6.1 还没适配）
    assert v < Version("0.7.0"), (
        f"flashinfer=={v} 已超过 verl 0.6.1 验证过的 0.6.x 窗口；"
        f"如确实要升级，需重新跑一遍训练 smoke。"
    )


########## 关键符号契约（项目源码直接依赖的入口）


def test_flash_attn_importable():
    """verl fsdp_workers 默认 attn_implementation=flash_attention_2，必须能 import。"""
    flash_attn = pytest.importorskip("flash_attn", reason="flash_attn 未安装")
    # 2.x 的入口；如果换 3.x API 会变
    assert hasattr(flash_attn, "flash_attn_func") or hasattr(
        flash_attn, "__version__"
    ), "flash_attn 装了但拿不到任何入口"


def test_vllm_core_symbols():
    _require_symbols("vllm", ["LLM", "SamplingParams"])


def test_verl_entry_symbols():
    """训练入口直接依赖 verl.trainer.main_ppo.run_ppo + TaskRunner。"""
    _require_symbols("verl.trainer.main_ppo", ["run_ppo", "TaskRunner"])


def test_verl_fsdp_config_symbols():
    """3 个训练脚本通过 hydra override 设置这些字段，对应 dataclass 必须存在。"""
    _require_symbols(
        "verl.workers.config",
        ["FSDPActorConfig", "FSDPEngineConfig", "FSDPOptimizerConfig"],
    )


def test_verl_advantage_registry_symbols():
    """训练入口用 register_adv_est 注入 schemashift_grpo，符号丢失会导致整个
    SchemaShift 路径失效。"""
    _require_symbols(
        "verl.trainer.ppo.core_algos",
        ["register_adv_est", "AdvantageEstimator"],
    )





########## 项目自身入口能 import


def test_project_entry_imports():
    """核心模块必须能 import，且不会触发副作用错误。"""
    importlib.import_module("src.training.schemashift_grpo_estimator")
    importlib.import_module("src.training.register_estimator")
    importlib.import_module("src.reward.oval_reward_fn")
    importlib.import_module("src.reward.action_parser")


def test_register_estimator_callable():
    """register 入口能调通；返回 True 才算注册成功。

    注意：register_schemashift_estimator 当前不是幂等的（重复调会叠加 patch 链），
    所以这里只调一次。幂等化属于 P0-2 / P1 范畴，未来加上后再扩展为 idempotent 测试。"""
    pytest.importorskip("verl", reason="verl 未安装")
    from src.training.register_estimator import register_schemashift_estimator
    assert register_schemashift_estimator() is True


def test_register_estimator_disabled_path():
    """显式 use_schemashift=False 时必须返回 False，不抛异常。"""
    pytest.importorskip("verl", reason="verl 未安装")
    from src.training.register_estimator import register_schemashift_estimator
    assert register_schemashift_estimator({"use_schemashift": False}) is False


########## arl 环境一致性提醒（软检查）


def test_env_matches_documented_versions(record_property):
    """记录当前版本到 pytest property，方便 CI 日志里直接看到漂移。
    不做强断言，只记录。"""
    versions = {}
    for name in [
        "torch", "transformers", "vllm", "verl", "ray",
        "flash_attn", "flashinfer_python", "tensordict",
    ]:
        v = _version_of(name) or _version_of(name.replace("_", "-"))
        versions[name] = str(v) if v else "not_installed"
    for k, v in versions.items():
        record_property(f"env.{k}", v)
    # 不 assert，让 CI 日志里有快照供后续 diff
