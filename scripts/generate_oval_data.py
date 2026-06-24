#!/usr/bin/env python3
"""为 OVAL GRPO smoke test 生成训练数据。

从 live MCP servers (calendar+shopping) 生成任务，
转为 verl parquet 格式。
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
    num_train: int = 32,
    num_val: int = 8,
    seed: int = 42,
):
    from src.live_mcp.api import LiveMCPBranch

    print(f"Generating oval training data: {num_train} train + {num_val} val tasks")
    branch = LiveMCPBranch.from_suite("configs/live_mcp/suite_mvp.yaml")
    branch.start()

    all_rows = []
    val_rows = []

    try:
        for domain in ["calendar", "shopping"]:
            print(f"\n  Domain: {domain}")
            tasks = branch.generate_tasks(
                server_name=domain,
                count=num_train + num_val,
                seed=seed,
                difficulty_mix={"easy": 1.0},
            )
            print(f"  Generated {len(tasks)} tasks")

            for i, task in enumerate(tasks):
                # Get tool schemas for the domain
                visible_tools = branch.manager.registry.server_tools(domain)

                # Build tools description for prompt
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

                extra_info = {
                    "task_id": task.task_id,
                    "domain": domain,
                    "target_servers": task.target_servers,
                    "required_tools": task.required_tools,
                    "session_seed": seed + i,
                    "budget": task.max_turns,
                    "perturbation_level": "none",
                    "scenario_type": task.task_type,
                    "group_id": f"oval_{domain}_{i}",
                    "uid": f"oval_{task.task_id}",
                }

                row = {
                    "prompt": prompt,
                    "data_source": "schemashift_oval",
                    "reward_model": {"style": "rule", "ground_truth": {"task_id": task.task_id}},
                    "extra_info": extra_info,
                    "uid": extra_info["uid"],
                    "group_id": extra_info["group_id"],
                    "perturbation_level": "none",
                    "scenario_type": task.task_type,
                }

                if i < num_train:
                    all_rows.append(row)
                else:
                    val_rows.append(row)

    finally:
        branch.stop()

    # 写入 parquet
    df_train = pd.DataFrame(all_rows)
    df_val = pd.DataFrame(val_rows)

    train_path = Path(output_path)
    val_path = Path(val_output_path)
    train_path.parent.mkdir(parents=True, exist_ok=True)

    df_train.to_parquet(train_path, index=False)
    df_val.to_parquet(val_path, index=False)

    print(f"\nTrain: {len(df_train)} rows → {train_path}")
    print(f"Val:   {len(df_val)} rows → {val_path}")
    for domain in ["calendar", "shopping"]:
        n = len(df_train[df_train["extra_info"].apply(lambda x: x.get("domain") == domain)])
        print(f"  {domain}: {n} train rows")

    # 验证
    print("\nVerifying...")
    df_check = pd.read_parquet(train_path)
    for i in range(min(3, len(df_check))):
        ei = df_check.iloc[i]["extra_info"]
        prompt = df_check.iloc[i]["prompt"]
        print(f"  [{i}] domain={ei['domain']} tools={ei['required_tools']} prompt_len={len(str(prompt))}")


if __name__ == "__main__":
    generate_oval_data()
