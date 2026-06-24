#!/usr/bin/env python3
"""为 OVAL GRPO 训练生成多样化的 Live MCP 数据。

生成的 parquet 直接用于 verl GRPO rollout（SchemaShiftOvalLoop）。

数据分布覆盖 OVAL-MCP §11.5 要求的类别：
  - normal_safe_success（正常安全成功）
  - distractor_tools（无关工具干扰）
  - missing_function（缺少必要工具）
  - overcall_redundant_read（冗余读操作）

Phase 1 目前基于 orchestrator 的 2 个 DomainAdapter (calendar + shopping)，
生成难度分层数据。Phase 2 将扩展 unsafe success / error recovery 等类别。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd


def generate_oval_data(
    output_path: str = "data/oval_grpo_train.parquet",
    val_output_path: str = "data/oval_grpo_val.parquet",
    num_train: int = 64,
    num_val: int = 16,
    seed: int = 42,
):
    """生成 OVAL 训练数据。

    难度分布：
      easy=50%:    基础任务（单步/两步工具调用）
      medium=30%:  包含 distractor tools
      hard=20%:    包含 missing_function 或复杂多步
    """
    from src.live_mcp.api import LiveMCPBranch

    print(f"Generating OVAL training data: {num_train} train + {num_val} val tasks")
    print(f"  Difficulty mix: easy=50%, medium=30%, hard=20%")
    print(f"  Domains: calendar, shopping")

    branch = LiveMCPBranch.from_suite("configs/live_mcp/suite_mvp.yaml")
    branch.start()

    all_rows = []
    val_rows = []

    difficulty_mix = {"easy": 0.5, "medium": 0.3, "hard": 0.2}

    try:
        for domain in ["calendar", "shopping"]:
            domain_train = num_train // 2
            domain_val = num_val // 2
            print(f"\n  Domain: {domain}")

            tasks = branch.generate_tasks(
                server_name=domain,
                count=domain_train + domain_val,
                seed=seed,
                difficulty_mix=difficulty_mix,
            )
            print(f"  Generated {len(tasks)} tasks")

            for i, task in enumerate(tasks):
                visible_tools = branch.manager.registry.server_tools(domain)

                tools_desc_lines = []
                for t in visible_tools:
                    name = t.get("name", "")
                    desc = t.get("description", "")
                    params = t.get("input_schema", {}).get("properties", {})
                    required = t.get("input_schema", {}).get("required", [])
                    tools_desc_lines.append(f"- {name}: {desc}")
                    for pname, pinfo in params.items():
                        req = " (required)" if pname in required else ""
                        ptype = pinfo.get("type", "any")
                        pdesc = pinfo.get("description", "")
                        tools_desc_lines.append(f"    - {pname} ({ptype}{req}): {pdesc}")

                tools_block = "\n".join(tools_desc_lines)

                system_prompt = (
                    f"You are a helpful assistant with access to the following tools. "
                    f"Use them when needed to answer the user's question.\n\n"
                    f"Available tools:\n{tools_block}\n\n"
                    f"Response format:\n"
                    f'- To call a tool: <tool_call>{{"name": "tool_name", "arguments": {{...}}}}</tool_call>\n'
                    f"- To give final answer: <final_answer>your answer</final_answer>\n"
                    f"- To report error: <report_error>error description</report_error>\n"
                    f"- To ask clarification: <ask_clarification>your question</ask_clarification>"
                )

                prompt = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": task.user_prompt},
                ]

                # 根据 task 元数据确定场景类型
                task_type = task.task_type
                has_distractors = task.metadata.get("has_distractors", False)
                has_missing_func = task.metadata.get("has_missing_function", False)

                if has_missing_func:
                    scenario_type = "missing_function"
                    perturbation_level = "hard"
                elif has_distractors:
                    scenario_type = "distractor"
                    perturbation_level = "medium"
                else:
                    scenario_type = task_type
                    perturbation_level = task.difficulty

                extra_info = {
                    "task_id": task.task_id,
                    "domain": domain,
                    "target_servers": task.target_servers,
                    "required_tools": task.required_tools,
                    "session_seed": seed + i,
                    "budget": task.max_turns,
                    "perturbation_level": perturbation_level,
                    "scenario_type": scenario_type,
                    "group_id": f"oval_{domain}_{i // 4}",  # 每 4 条同一 group（简化版）
                    "uid": f"oval_{task.task_id}",
                    "has_distractors": has_distractors,
                    "has_missing_function": has_missing_func,
                }

                row = {
                    "prompt": prompt,
                    "data_source": "schemashift_oval",
                    "reward_model": {"style": "rule", "ground_truth": {"task_id": task.task_id}},
                    "extra_info": extra_info,
                    "uid": extra_info["uid"],
                    "group_id": extra_info["group_id"],
                    "perturbation_level": perturbation_level,
                    "scenario_type": scenario_type,
                }

                if i < domain_train:
                    all_rows.append(row)
                else:
                    val_rows.append(row)

    finally:
        branch.stop()

    df_train = pd.DataFrame(all_rows)
    df_val = pd.DataFrame(val_rows)

    train_path = Path(output_path)
    val_path = Path(val_output_path)
    train_path.parent.mkdir(parents=True, exist_ok=True)

    df_train.to_parquet(train_path, index=False)
    df_val.to_parquet(val_path, index=False)

    print(f"\nTrain: {len(df_train)} rows → {train_path}")
    print(f"Val:   {len(df_val)} rows → {val_path}")

    # 按 domain + scenario_type 统计分布
    for domain in ["calendar", "shopping"]:
        domain_rows = df_train[df_train["extra_info"].apply(lambda x: x.get("domain") == domain)]
        print(f"\n  {domain}: {len(domain_rows)} rows")
        for stype in ["normal", "distractor", "missing_function"]:
            count = len(domain_rows[domain_rows["scenario_type"] == stype])
            if count > 0:
                print(f"    {stype}: {count}")

    # 验证可读
    print("\nVerifying (first 3 rows)...")
    df_check = pd.read_parquet(train_path)
    for i in range(min(3, len(df_check))):
        ei = df_check.iloc[i]["extra_info"]
        prompt = df_check.iloc[i]["prompt"]
        st = df_check.iloc[i]["scenario_type"]
        print(f"  [{i}] domain={ei['domain']} scenario={st} "
              f"tools={ei['required_tools']} prompt_len={len(str(prompt))}")


if __name__ == "__main__":
    generate_oval_data()
