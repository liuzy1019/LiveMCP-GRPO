#!/usr/bin/env python3
"""[DEPRECATED] 预生成 BFCL V3 schema 变体（仅用于调试/可视化）。

主路径已迁移到 ``scripts/build_parquet.py``。该脚本遵循每 (task, level)
独立 perturber 的设计，避免跨 task ``name_map`` 污染，是训练/评估
的唯一依据。

本脚本仅作为预生成 JSON dump 使用（供 debug、快照、可视化），
不要再进入训练流程。
"""

import json
import re
import sys
from pathlib import Path
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.envs.bfcl_env import BFCLDataLoader
from src.envs.schema_perturber import SchemaPerturber, LEVEL_MILD, LEVEL_STRONG


def load_all_tasks(data_dir: str) -> list:
    """加载所有 multi-turn 任务的函数定义和 GT。"""
    data_dir = Path(data_dir)
    gt_dir = data_dir / "possible_answer"

    # 文件名对应
    task_files = [
        "BFCL_v3_multi_turn_base.json",
        "BFCL_v3_multi_turn_composite.json",
        "BFCL_v3_multi_turn_long_context.json",
        "BFCL_v3_multi_turn_miss_func.json",
        "BFCL_v3_multi_turn_miss_param.json",
    ]

    # 加载所有任务
    tasks_by_id = {}
    for fname in task_files:
        fpath = data_dir / fname
        if not fpath.exists():
            logger.warning(f"文件不存在: {fpath}")
            continue
        loader = BFCLDataLoader(str(fpath))
        for t in loader.tasks:
            tasks_by_id[t.id] = {
                "task": t,
                "source_file": fname,
            }

    # 加载所有 GT
    for fname in task_files:
        gt_path = gt_dir / fname
        if not gt_path.exists():
            continue
        with open(gt_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    gt_data = json.loads(line)
                    tid = gt_data.get("id", "")
                    if tid in tasks_by_id:
                        tasks_by_id[tid]["ground_truth"] = gt_data.get("ground_truth", [])
                except json.JSONDecodeError:
                    continue

    logger.info(f"加载 {len(tasks_by_id)} 个任务（含 GT）")
    return list(tasks_by_id.values())


def generate_perturbations(
    data_dir: str,
    output_dir: str,
    seed: int = 42,
):
    """生成所有扰动数据。"""
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 加载数据
    all_entries = load_all_tasks(str(data_dir))

    # 2. 生成扰动：每 (task, level) 独立 perturber，避免 name_map / enum_map 跨任务污染
    output = []
    name_map_global: dict[str, str] = {}
    for entry in all_entries:
        task = entry["task"]
        gt_raw = entry.get("ground_truth", [])
        if not task.functions:
            continue

        for level in [LEVEL_MILD, LEVEL_STRONG]:
            # 独立 perturber，不复用
            level_seed = seed if level == LEVEL_MILD else seed + 9999
            perturber = SchemaPerturber(seed=level_seed)
            perturbed_funcs = perturber.perturb(task.functions, level)

            # 扰动 GT 中的函数名
            reverse_map = {v: k for k, v in perturber.name_map.items()}
            perturbed_gt = []
            for gt_item in gt_raw:
                # GT item 可能是字符串或 dict
                if isinstance(gt_item, str):
                    # 替换函数名（使用词边界避免 find 误伤 find_flights）
                    result = gt_item
                    for orig_name, pert_name in reverse_map.items():
                        result = re.sub(
                            rf'\b{re.escape(orig_name)}\b',
                            pert_name,
                            result,
                        )
                    perturbed_gt.append(result)
                elif isinstance(gt_item, dict):
                    # BFCL V4 格式: {"func_name": {params}}
                    new_item = {}
                    for k, v in gt_item.items():
                        new_key = reverse_map.get(k, k)
                        new_item[new_key] = v
                    perturbed_gt.append(new_item)
                else:
                    perturbed_gt.append(gt_item)

            output.append({
                "id": task.id,
                "source": entry.get("source_file", ""),
                "level": level,
                "original_functions": task.functions,
                "perturbed_functions": perturbed_funcs,
                "original_gt": gt_raw,
                "perturbed_gt": perturbed_gt,
                "initial_config": task.initial_config,
                "involved_classes": task.involved_classes,
                "path": task.path,
                "name_map": dict(perturber.name_map),
                "enum_map": dict(perturber.enum_map),
            })
            # 仅用于全局 dump，跨 task 重复的 key 不影响训练主路径
            name_map_global.update(perturber.name_map)

    logger.info(f"生成 {len(output)} 个扰动样本")
    logger.info(f"name_map_global: {len(name_map_global)} 条映射（仅供调试）")

    # 3. 保存
    output_file = output_dir / "bfcl_v3_perturbed.json"
    with open(output_file, "w") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    logger.info(f"已保存: {output_file} ({len(output)} 条)")

    # 4. 保存 name_map（仅调试用，不是训练数据主路径）
    reverse_map = {v: k for k, v in name_map_global.items()}
    map_file = output_dir / "name_map.json"
    with open(map_file, "w") as f:
        json.dump({
            "forward": name_map_global,
            "reverse": reverse_map,
            "total_mappings": len(name_map_global),
            "_warning": "仅供调试。跨 task name_map 可能有 key 冲突，训练主路径请使用 build_parquet.py 生成的 per-record name_map_json",
        }, f, ensure_ascii=False, indent=2)
    logger.info(f"已保存映射: {map_file}")


if __name__ == "__main__":
    logger.warning(
        "[DEPRECATED] 本脚本仅用于调试/快照。训练/评估主路径请使用 scripts/build_parquet.py。"
    )
    base = Path(__file__).resolve().parent.parent / "data"
    generate_perturbations(
        data_dir=str(base),
        output_dir=str(base),
        seed=42,
    )
