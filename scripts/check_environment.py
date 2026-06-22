#!/usr/bin/env python3
"""dancegrpo 环境兼容性检查脚本。
在目标机器上运行: python check_environment.py
"""

import sys
import platform
from subprocess import run, PIPE

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


def check(header, cmds, checks):
    """检查一组依赖项。"""
    print(f"\n{'='*50}")
    print(f"  {header}")
    print(f"{'='*50}")
    all_ok = True
    for label, cmd, expected in checks:
        try:
            result = run(cmd, shell=True, capture_output=True, text=True, timeout=10)
            output = result.stdout.strip() or result.stderr.strip()
            ok = expected(output)
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
    ok_py = sys.version_info >= (3, 10)
    print(f"{'✅' if ok_py else '❌'} Python >= 3.10: {sys.version.split()[0]}")

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
    ok = check("核心依赖检查", "", [
        ("torch", f"python -c 'import torch; print(torch.__version__)'",
         lambda x: version_at_least(x, "2.0.0")),
        ("transformers", f"python -c 'import transformers; print(transformers.__version__)'",
         lambda x: version_at_least(x, "4.0")),
        ("numpy", f"python -c 'import numpy; print(numpy.__version__)'",
         lambda x: True),
        ("PyYAML", f"python -c 'import yaml; print(yaml.__version__)'",
         lambda x: True),
        ("pydantic", f"python -c 'import pydantic; print(pydantic.__version__)'",
         lambda x: version_at_least(x, "2.0")),
        ("loguru", f"python -c 'import loguru; print(loguru.__version__)'",
         lambda x: True),
        ("huggingface_hub", f"python -c 'import huggingface_hub; print(huggingface_hub.__version__)'",
         lambda x: True),
    ])

    # RL 训练框架（可选）
    ok &= check("RL 训练框架检查（可选）", "", [
        ("vllm", f"python -c 'import vllm; print(vllm.__version__)'",
         lambda x: version_at_least(x, "0.6.0")),
        ("verl", f"python -c 'import verl; print(\"OK\")'",
         lambda x: "OK" in x),
        ("bfcl executor", f"python -c 'from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import execute_multi_turn_func_call; print(\"OK\")'",
         lambda x: "OK" in x),
    ])

    # GPU 推理测试
    print(f"\n{'='*50}")
    print("  GPU 推理测试")
    print(f"{'='*50}")
    if cuda_ok and torch is not None:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            model = AutoModelForCausalLM.from_pretrained(
                "Qwen/Qwen2.5-1.5B-Instruct",
                torch_dtype="auto",
                device_map="cuda:0",
            )
            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
            inputs = tokenizer("Hello", return_tensors="pt").to("cuda:0")
            out = model.generate(**inputs, max_new_tokens=10)
            print(f"  ✅ 模型推理成功: {tokenizer.decode(out[0])[:50]}")
            del model
            import gc; gc.collect()
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  ⚠️ 推理测试失败: {e}")

    # 总结
    print(f"\n{'='*50}")
    if ok:
        print("  ✅ 环境兼容性检查通过")
    else:
        print(f"  {'⚠️' if ok_py and cuda_ok else '❌'} 环境兼容性检查{'部分' if ok_py and cuda_ok else '不'}通过")
        print("  缺少的包可以在验证后安装: pip install <package>")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
