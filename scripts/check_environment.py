#!/usr/bin/env python3
"""SchemaShift-GRPO 环境兼容性检查脚本。
在目标机器上运行: python check_environment.py
"""

import os
import sys
import platform
from pathlib import Path
from subprocess import run

try:
    from packaging.version import parse as parse_version
except ImportError:
    parse_version = None


def version_at_least(output: str, minimum: str) -> bool:
    """Compare package versions without lexicographic false negatives."""
    if not output:
        return False
    version = output.split()[0]
    if parse_version is None:
        return version >= minimum
    try:
        return parse_version(version) >= parse_version(minimum)
    except Exception:
        return False


def check(header, checks):
    """检查一组依赖项。"""
    print(f"\n{'='*50}")
    print(f"  {header}")
    print(f"{'='*50}")
    all_ok = True
    for label, cmd, expected in checks:
        try:
            result = run(cmd, capture_output=True, text=True, timeout=10)
            output = result.stdout.strip() or result.stderr.strip()
            ok = result.returncode == 0 and expected(output)
            status = "✅" if ok else "❌"
            if not ok:
                all_ok = False
            print(f"  {status} {label:30s} {output[:60] if output else '(empty)'}")
        except Exception as e:
            print(f"  ⚠️  {label:30s} error: {e}")
            all_ok = False
    return all_ok


def main():
    print(f"系统: {platform.platform()}")
    print(f"Hostname: {platform.node()}")
    print(f"Python: {sys.version.split()[0]}")

    # Python 版本
    ok_py = sys.version_info >= (3, 11)
    print(f"{'✅' if ok_py else '❌'} Python >= 3.11: {sys.version.split()[0]}")

    # CUDA
    try:
        import torch
        cuda_ok = torch.cuda.is_available()
    except ImportError:
        print("❌ torch 未安装，无法检测 CUDA")
        torch = None
        cuda_ok = False
    print(f"{'✅' if cuda_ok else '❌'} CUDA 可用: {cuda_ok}")
    if cuda_ok:
        print(f"  GPU 数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            total_mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"    显存: {total_mem:.0f} GB")

    # 核心 Python 包
    py = sys.executable
    ok = check("核心依赖检查", [
        ("torch", [py, "-c", "import torch; print(torch.__version__)"],
         lambda x: version_at_least(x, "2.0.0")),
        ("transformers", [py, "-c", "import transformers; print(transformers.__version__)"],
         lambda x: version_at_least(x, "4.0")),
        ("numpy", [py, "-c", "import numpy; print(numpy.__version__)"],
         lambda x: True),
        ("PyYAML", [py, "-c", "import yaml; print(yaml.__version__)"],
         lambda x: True),
        ("pydantic", [py, "-c", "import pydantic; print(pydantic.__version__)"],
         lambda x: version_at_least(x, "2.0")),
        ("loguru", [py, "-c", "import loguru; print(loguru.__version__)"],
         lambda x: True),
        ("huggingface_hub", [py, "-c", "import huggingface_hub; print(huggingface_hub.__version__)"],
         lambda x: True),
    ])

    # RL 训练框架（可选）
    ok &= check("RL 训练框架检查（可选）", [
        ("vllm", [py, "-c", "import vllm; print(vllm.__version__)"],
         lambda x: version_at_least(x, "0.6.0")),
        ("verl", [py, "-c", "import verl; print('OK')"],
         lambda x: "OK" in x),
    ])

    # GPU 推理测试
    print(f"\n{'='*50}")
    print("  GPU 推理测试")
    print(f"{'='*50}")
    if cuda_ok and torch is not None:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            project_root = Path(__file__).resolve().parent.parent
            default_model = project_root / "models" / "Qwen3-4B"
            model_path = os.environ.get(
                "SCHEMASHIFT_ENV_TEST_MODEL",
                str(default_model) if default_model.exists() else "",
            )
            if not model_path:
                print("  ⚠️ 未找到本地 models/Qwen3-4B，跳过 GPU 推理测试")
                raise RuntimeError("skip_gpu_inference_test")
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype="auto",
                device_map="cuda:0",
            )
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            inputs = tokenizer("Hello", return_tensors="pt").to("cuda:0")
            out = model.generate(**inputs, max_new_tokens=10)
            print(f"  ✅ 模型推理成功: {tokenizer.decode(out[0])[:50]}")
            del model
            import gc; gc.collect()
            torch.cuda.empty_cache()
        except Exception as e:
            if str(e) != "skip_gpu_inference_test":
                print(f"  ⚠️ 推理测试失败: {e}")

    # 总结
    print(f"\n{'='*50}")
    final_ok = ok and ok_py and cuda_ok
    if final_ok:
        print("  ✅ 环境兼容性检查通过")
    else:
        print(f"  {'⚠️' if ok_py and cuda_ok else '❌'} 环境兼容性检查{'部分' if ok_py and cuda_ok else '不'}通过")
        if not cuda_ok:
            print("  CUDA 不可用；请检查 NVIDIA driver/GPU 可见性。")
        if not ok:
            print("  缺少的包可以在验证后安装: pip install <package>")
    print(f"{'='*50}")
    return final_ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
