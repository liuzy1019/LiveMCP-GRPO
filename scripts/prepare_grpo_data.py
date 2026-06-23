"""将 EpisodeSeed 数据转为 verl GRPO 训练所需的 parquet 格式。

verl 的 RLHFDataset 需要 parquet 文件包含以下字段：
  - prompt: list[dict]（chat messages 格式）
  - data_source: str（用于选择 reward function）
  - reward_model.ground_truth: str（oracle 信息，JSON 序列化）
  - extra_info: dict（perturbation_level, name_map 等）

Usage:
    python scripts/prepare_grpo_data.py \
        --episode_seeds data/episode_seeds.jsonl \
        --output data/grpo_train.parquet \
        --max_samples 100
"""

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.episode_schema import EpisodeSeed
from src.envs.schema_perturber import SchemaPerturber, LEVEL_NONE, LEVEL_MILD, LEVEL_STRONG


# E4 SchemaShift 扰动展开配置：每 seed 3 层 × 每层 3 副本 = 9 行
E4_LEVELS = [LEVEL_NONE, LEVEL_MILD, LEVEL_STRONG]
E4_COPIES_PER_LEVEL = 3


def _stable_seed(text: str) -> int:
    """Return a process-stable 31-bit seed for reproducible data generation."""
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & 0x7FFFFFFF


def episode_to_verl_row(
    episode: EpisodeSeed,
    perturbed_tools: list[dict],
    perturbation_level: str,
    name_map: dict[str, str] | None = None,
    enum_map: dict | None = None,
    copy_index: int = 0,
) -> dict:
    """将单个 EpisodeSeed 转为 verl parquet 行（支持扰动变体）。

    Args:
        episode: EpisodeSeed 对象。
        perturbed_tools: 扰动后的 tool definitions（或原始 tools）。
        perturbation_level: 扰动级别 "none"/"mild"/"strong"。
        name_map: perturbed name → canonical name 映射。
        enum_map: perturbed enum → canonical enum 映射。
        copy_index: 同级别内的副本序号（0..E4_COPIES_PER_LEVEL-1）。
    """
    name_map = name_map or {}
    enum_map = enum_map or {}
    # 构建 prompt（system + tools + user messages）
    tools_desc = format_tools_for_prompt(perturbed_tools)
    system_msg = (
        "You are a helpful assistant with access to the following tools. "
        "Use them when needed to answer the user's question.\n\n"
        f"Available tools:\n{tools_desc}\n\n"
        "Response format:\n"
        "- To call a tool: <tool_call>{\"name\": \"tool_name\", \"arguments\": {...}}</tool_call>\n"
        "- To give final answer: <final_answer>your answer</final_answer>\n"
        "- To report error: <report_error>error description</report_error>\n"
        "- To ask clarification: <ask_clarification>your question</ask_clarification>"
    )

    messages = [{"role": "system", "content": system_msg}]
    for msg in episode.initial_messages:
        messages.append(msg)

    # 构建 ground_truth（oracle 信息）
    # oracle_trace 是 dict 列表或 OracleStep 对象
    oracle_actions = []
    first_action_type = ""
    first_tool_name = ""
    for step in episode.oracle_trace:
        action_type = step.get("action_type", "tool_call") if isinstance(step, dict) else getattr(step, "action_type", "tool_call")
        tool_name = step.get("tool_name") if isinstance(step, dict) else getattr(step, "tool_name", None)
        arguments = step.get("arguments", {}) if isinstance(step, dict) else getattr(step, "arguments", {})
        calls = step.get("calls", []) if isinstance(step, dict) else getattr(step, "calls", [])
        expected_content = step.get("expected_content", "") if isinstance(step, dict) else getattr(step, "expected_content", "")
        match_mode = step.get("match_mode", "set") if isinstance(step, dict) else getattr(step, "match_mode", "set")

        # 获取 replay_observation（多步 reward 需要）
        replay_observation = step.get("replay_observation", "") if isinstance(step, dict) else getattr(step, "replay_observation", "")
        replay_observations = step.get("replay_observations", []) if isinstance(step, dict) else getattr(step, "replay_observations", [])

        action_data = {"action_type": action_type}

        if action_type == "tool_call" and tool_name:
            action_data["tool_calls"] = [{"name": tool_name, "arguments": arguments}]
            action_data["match_mode"] = "ordered"
            # 始终保留 replay_observation（交互式静态 rollout 需要）
            action_data["replay_observation"] = replay_observation or ""
            first_tool_name = first_tool_name or tool_name
        elif action_type == "parallel_tool_call" and calls:
            # 转换 parallel calls 为 reward 可识别的 tool_calls 格式
            action_data["action_type"] = "tool_call"
            action_data["tool_calls"] = [
                {"name": c.get("tool_name", c.get("name", "")), "arguments": c.get("arguments", {})}
                for c in calls
            ]
            action_data["match_mode"] = match_mode
            # 始终保留 replay_observations
            if replay_observations:
                action_data["replay_observations"] = replay_observations
                # 合并为单个 observation 供交互式 replay 使用
                action_data["replay_observation"] = "\n".join(replay_observations)
            elif replay_observation:
                action_data["replay_observation"] = replay_observation
            else:
                action_data["replay_observation"] = ""
            first_tool_name = first_tool_name or ",".join(
                c.get("tool_name", c.get("name", "")) for c in calls
            )
        elif action_type in ("final_answer", "report_error", "ask_clarification"):
            if expected_content:
                action_data["final_answer"] = expected_content

        first_action_type = first_action_type or action_data["action_type"]
        oracle_actions.append(action_data)

    ground_truth = json.dumps({
        "oracle_actions": oracle_actions,
        "episode_type": episode.episode_type,
    }, ensure_ascii=False)

    scenario_type = episode.episode_type
    group_id = episode.episode_id
    uid = f"{group_id}___{perturbation_level}___{scenario_type}___r{copy_index}"

    # 构建 extra_info（去掉空 dict 字段避免 parquet 空 struct 问题）
    # 注意：嵌套的 list/dict 必须序列化为 JSON 字符串，否则 pyarrow 无法处理混合类型
    replay_data_json = json.dumps({
        "oracle_actions": oracle_actions,
        "episode_type": episode.episode_type,
    }, ensure_ascii=False)

    extra_info = {
        "perturbation_level": perturbation_level,
        "scenario_type": scenario_type,
        "episode_id": episode.episode_id,
        "group_id": group_id,
        "uid": uid,
        "action_type": first_action_type,
        "tool_name": first_tool_name,
        # 交互式静态 replay 所需：verl rl_dataset 会提取 tools_kwargs
        "need_tools_kwargs": True,
        "tools_kwargs": json.dumps({
            "replay_data": {
                "create_kwargs": {
                    "oracle_actions": oracle_actions,
                    "episode_type": episode.episode_type,
                },
            },
        }, ensure_ascii=False),
        # 同时在 extra_info 顶层放一份 replay_data（兼容 agent loop 直接读取）
        "replay_data": replay_data_json,
    }
    # 只在非空时添加 map 字段
    if name_map:
        extra_info["name_map"] = name_map
    if enum_map:
        extra_info["enum_map"] = enum_map

    return {
        "prompt": messages,
        "data_source": "schemashift",
        "reward_model": {"style": "rule", "ground_truth": ground_truth},
        "extra_info": extra_info,
        # Keep these at top-level so verl's non_tensor_batch can preserve them
        # through _get_gen_batch and the schemashift_grpo estimator.
        "uid": uid,
        "group_id": group_id,
        "perturbation_level": perturbation_level,
        "scenario_type": scenario_type,
        "action_type": first_action_type,
        "tool_name": first_tool_name,
    }


def format_tools_for_prompt(tools: list[dict]) -> str:
    """格式化工具描述。"""
    lines = []
    for tool in tools:
        name = tool.get("name", tool.get("function", {}).get("name", "unknown"))
        desc = tool.get("description", tool.get("function", {}).get("description", ""))
        params = tool.get("parameters", tool.get("function", {}).get("parameters", {}))

        lines.append(f"- {name}: {desc}")
        if params and isinstance(params, dict):
            properties = params.get("properties", {})
            required = params.get("required", [])
            for pname, pinfo in properties.items():
                req_mark = " (required)" if pname in required else ""
                ptype = pinfo.get("type", "any")
                pdesc = pinfo.get("description", "")
                lines.append(f"    - {pname} ({ptype}{req_mark}): {pdesc}")

    return "\n".join(lines)


def main():
    logger.remove()
    logger.add(sys.stderr, level="WARNING")

    parser = argparse.ArgumentParser(description="EpisodeSeed → verl parquet")
    # 输入输出
    parser.add_argument("--episode_seeds", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--val_split", type=float, default=0.05)
    parser.add_argument("--val_output", type=str, default=None)
    parser.add_argument("--expect_e4_groups", action="store_true",
                        help="校验每个 group_id 是否满足 E4 3:3:3 结构")
    parser.add_argument("--records_per_group", type=int, default=9,
                        help="E4 每个 group 的期望记录数")
    args = parser.parse_args()

    # 读取 episode seeds
    episodes = []
    input_path = Path(args.episode_seeds)
    if input_path.suffix == ".jsonl":
        with open(input_path) as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    episodes.append(EpisodeSeed(**data))
    elif input_path.suffix == ".json":
        with open(input_path) as f:
            data_list = json.load(f)
            for data in data_list:
                episodes.append(EpisodeSeed(**data))
    else:
        raise ValueError(f"Unsupported file format: {input_path.suffix}")

    if args.max_samples > 0:
        episodes = episodes[:args.max_samples]

    print(f"Loaded {len(episodes)} episodes")

    # 转换：E4 SchemaShift 扰动展开
    #   只展开有工具调用的 episode（call_then_call / call_then_final / call_only）
    #   跳过 no_tool（无工具）和 error_output（无工具调用）
    skip_types = {"no_tool", "error_output"}
    rows: list[dict] = []
    for ep in episodes:
        if ep.episode_type in skip_types or not ep.oracle_trace:
            continue
        if not ep.tools_snapshot:
            continue

        group_id = ep.episode_id

        # 去重 tools_snapshot（部分 EpisodeSeed 含重复工具定义）
        seen_names: set[str] = set()
        deduped_tools: list[dict] = []
        for t in ep.tools_snapshot:
            name = t.get("function", t).get("name", "")
            if name and name not in seen_names:
                seen_names.add(name)
                deduped_tools.append(t)
        if not deduped_tools:
            continue

        for level in E4_LEVELS:
            # 每层独立 SchemaPerturber，避免 name_map/enum_map 跨层污染
            base_seed = _stable_seed(f"{group_id}:{level}")

            if level == LEVEL_NONE:
                tools = copy.deepcopy(deduped_tools)
                nm = {}
                em = {}
            else:
                # 扰动可能因工具名冲突失败，递增 seed 重试
                tools = None
                MAX_RETRIES = 20
                for retry in range(MAX_RETRIES):
                    try:
                        perturber = SchemaPerturber(seed=base_seed + retry)
                        tools = perturber.perturb(copy.deepcopy(deduped_tools), level)
                        nm = dict(perturber.name_map)
                        em = dict(perturber.enum_map)
                        break
                    except ValueError as e:
                        if retry == MAX_RETRIES - 1:
                            raise RuntimeError(
                                f"工具名扰动冲突，{MAX_RETRIES}次重试均失败: "
                                f"episode={group_id}, level={level}, last_error={e}"
                            ) from e
                        continue

            for copy_i in range(E4_COPIES_PER_LEVEL):
                row = episode_to_verl_row(ep, tools, level, nm, em, copy_i)
                rows.append(row)

    df = pd.DataFrame(rows)
    n_groups = df["group_id"].nunique() if "group_id" in df.columns else 0
    skipped = len(episodes) - n_groups
    records_per_group = (len(rows) // n_groups) if n_groups else 0
    print(
        f"E4 展开: {len(episodes)} seeds → {n_groups} trainable groups "
        f"({skipped} skipped) → {len(rows)} rows ({records_per_group} rows/group)"
    )

    # P1-4: E4 group 完整性校验
    if args.expect_e4_groups and "group_id" in df.columns:
        from collections import Counter
        group_counts = Counter(df["group_id"])
        bad_groups = {g: c for g, c in group_counts.items() if c != args.records_per_group}
        if bad_groups:
            n_bad = len(bad_groups)
            examples = dict(list(bad_groups.items())[:5])
            raise ValueError(
                f"E4 group 完整性校验失败: {n_bad} 个 group 的记录数 != {args.records_per_group}。"
                f"示例: {examples}"
            )
        # 校验 perturbation_level 分布
        if "perturbation_level" in df.columns:
            for gid, gdf in df.groupby("group_id"):
                level_counts = Counter(gdf["perturbation_level"])
                expected_per_level = args.records_per_group // 3
                for level in ("none", "mild", "strong"):
                    actual = level_counts.get(level, 0)
                    if actual != expected_per_level:
                        raise ValueError(
                            f"E4 group '{gid}' perturbation_level 分布异常: "
                            f"期望每级 {expected_per_level} 条，实际 {dict(level_counts)}"
                        )
        print(f"E4 group 完整性校验通过: {len(group_counts)} groups × {args.records_per_group} records")

    # 分割 train/val（按 group_id 分割，确保同一 group 的所有行在同一 split 中）
    if args.val_split > 0 and args.val_output:
        # 获取唯一的 group_id 列表（保持原始顺序）
        if "group_id" in df.columns:
            unique_groups = df["group_id"].unique().tolist()
        else:
            # fallback：如果没有 group_id 列，按行分割
            unique_groups = list(range(len(df)))

        n_val_groups = max(1, int(len(unique_groups) * args.val_split))
        val_groups = set(unique_groups[-n_val_groups:])

        if "group_id" in df.columns:
            val_mask = df["group_id"].isin(val_groups)
            val_df = df[val_mask].reset_index(drop=True)
            train_df = df[~val_mask].reset_index(drop=True)
        else:
            n_val = max(1, int(len(df) * args.val_split))
            val_df = df.tail(n_val)
            train_df = df.head(len(df) - n_val)

        train_df.to_parquet(args.output, index=False)
        val_df.to_parquet(args.val_output, index=False)
        print(f"Train: {len(train_df)} rows ({len(unique_groups) - n_val_groups} groups) → {args.output}")
        print(f"Val: {len(val_df)} rows ({n_val_groups} groups) → {args.val_output}")
    else:
        df.to_parquet(args.output, index=False)
        print(f"Output: {len(df)} → {args.output}")


if __name__ == "__main__":
    main()
