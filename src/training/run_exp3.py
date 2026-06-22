#!/usr/bin/env python3
"""E3 GRPO 基线训练入口。

只注册 agent loop，不 patch verl。标准 GRPO 训练。
"""

import os
import sys
from pathlib import Path

PROJECT_DIR = str(Path(__file__).resolve().parent.parent.parent)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, "verl"))


def main() -> None:
    # P1-1：先做数据长度预检，避免跑到 step 1 才发现 max_prompt_length 不够
    from src.training.length_check import maybe_run_length_check
    maybe_run_length_check(sys.argv[1:])

    # 注册 agent loop（必须在 verl 启动前 import）
    from src.agent_loop.bfcl_agent_loop import BFCLAgentLoop  # noqa: F401
    print("  Agent loop BFCLAgentLoop 已注册")

    from verl.trainer.main_ppo import main as verl_main
    verl_main()


if __name__ == "__main__":
    main()
