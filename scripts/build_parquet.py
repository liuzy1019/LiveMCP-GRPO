#!/usr/bin/env python3
"""BFCL 数据 → verl parquet 格式转换。

prompt 以 JSON 消息列表格式存储，agent loop 直接反序列化使用。
"""

import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.envs.bfcl_env import BFCLDataLoader, BFCLGroundTruthLoader
from src.envs.schema_perturber import SchemaPerturber, LEVEL_NONE, LEVEL_MILD, LEVEL_STRONG


MULTI_TURN_FILES = [
    "BFCL_v3_multi_turn_base.json",
    "BFCL_v3_multi_turn_composite.json",
    "BFCL_v3_multi_turn_long_context.json",
    "BFCL_v3_multi_turn_miss_func.json",
    "BFCL_v3_multi_turn_miss_param.json",
]

LEVEL_DISTRIBUTION = [LEVEL_NONE, LEVEL_MILD, LEVEL_STRONG]
# E4 数据布局：1 task -> 9 records (3 none + 3 mild + 3 strong)，训练时 rollout.n=1、batch 是 9 的倍数
E4_LEVEL_ASSIGNMENTS = [LEVEL_NONE] * 3 + [LEVEL_MILD] * 3 + [LEVEL_STRONG] * 3


def build_messages(functions: list[dict], user_turns: list) -> list[dict]:
    """构造多轮对话消息列表。

    格式: system(含工具定义) + user(第一轮)。
    后续轮次由 agent loop 在运行时追加。

    Returns:
        [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
    """
    tools_str = json.dumps(functions, ensure_ascii=False, indent=2)
    first_turn = user_turns[0] if user_turns else []
    user_text = ""
    for msg in first_turn:
        if isinstance(msg, dict) and msg.get("role") == "user":
            user_text = msg.get("content", "")
            break

    return [
        {
            "role": "system",
            "content": f"You have access to the following tools:\n{tools_str}\n\n"
                       f'Call tools using: <tool_call>{{"name": "func_name", "arguments": {{...}}}}</tool_call>'
        },
        {"role": "user", "content": user_text},
    ]


def infer_max_turns(user_turns: list) -> int:
    """根据 user_turns 推断需要的最大轮次。"""
    return min(len(user_turns) + 2, 15)  # 比用户轮数多 2 轮保险


def write_parquet(records: list[dict], path: Path) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        logger.error("需要 pyarrow: pip install pyarrow")
        sys.exit(1)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(records)
    pq.write_table(table, str(path))
    logger.info(f"  写入 {path.name} ({len(records)} 条)")


# ═══════════════════════════════════════════
# E3: GRPO 基线 — 原始 schema
# ═══════════════════════════════════════════

def _load_ground_truth(data_dir: str) -> dict[str, list]:
    """加载所有多轮任务的 ground truth。"""
    gt = {}
    gt_dir = Path(data_dir) / "possible_answer"
    for fname in MULTI_TURN_FILES:
        fpath = gt_dir / fname
        if fpath.exists():
            loader = BFCLGroundTruthLoader(str(fpath))
            gt.update(loader._data)
    return gt


# ═══════════════════════════════════════════
# E4: SchemaShift — 混合扰动
# ═══════════════════════════════════════════

def prepare_exp4(data_dir: str, out_dir: str, seed: int = 42, val_ratio: float = 0.1):
    """E4 SchemaShift 数据：1 task → 9 records (3 none + 3 mild + 3 strong)。

    rollout.n=1 per record。uid 编码 task_id 和 level，estimator 按 group_id/level 做分层 advantage。
    batch_size 需为 9 的倍数，shuffle=False，确保同 task 9 条记录在同一 batch。
    """
    logger.info("=== E4: SchemaShift (9 records/task) ===")
    gt_all = _load_ground_truth(data_dir)

    # Step 1: 先按 task_id 做 train/val split
    all_task_ids = []
    for fname in MULTI_TURN_FILES:
        loader = BFCLDataLoader(str(Path(data_dir) / fname))
        for task in loader.tasks:
            all_task_ids.append(task.id)
    rng = random.Random(seed)
    rng.shuffle(all_task_ids)
    split = int(len(all_task_ids) * (1 - val_ratio))
    train_ids = set(all_task_ids[:split])
    val_ids = set(all_task_ids[split:])

    # Step 2: 为每个 task 生成 9 条记录（3 none + 3 mild + 3 strong）
    train_records, val_records = [], []
    for fname in MULTI_TURN_FILES:
        loader = BFCLDataLoader(str(Path(data_dir) / fname))
        for task in loader.tasks:
            is_train = task.id in train_ids

            # 每层独立 perturber，避免 name_map/enum_map 跨层污染
            perturber_mild = SchemaPerturber(seed=seed)
            perturber_strong = SchemaPerturber(seed=seed + 9999)
            mild_funcs, mild_nm = perturber_mild.perturb(task.functions, LEVEL_MILD), dict(perturber_mild.name_map)
            strong_funcs, strong_nm = perturber_strong.perturb(task.functions, LEVEL_STRONG), dict(perturber_strong.name_map)
            strong_enum_map = dict(perturber_strong.enum_map)

            schemas = {
                LEVEL_NONE: (task.functions, {}, {}),
                LEVEL_MILD: (mild_funcs, mild_nm, dict(perturber_mild.enum_map)),
                LEVEL_STRONG: (strong_funcs, strong_nm, strong_enum_map),
            }

            for level in E4_LEVEL_ASSIGNMENTS:
                funcs, nm, em = schemas[level]
                record = {
                    "prompt": build_messages(funcs, task.user_turns),
                    "data_source": "bfcl",
                    "functions_json": json.dumps(funcs),
                    "initial_config": json.dumps(task.initial_config),
                    "involved_classes": json.dumps(task.involved_classes),
                    "user_turns_json": json.dumps(task.user_turns[1:]),
                    "ground_truth_json": json.dumps(gt_all.get(task.id, [])),
                    "uid": f"{task.id}___{level}",
                    "perturbation_level": level,
                    "name_map_json": json.dumps(nm),
                    "enum_map_json": json.dumps(em),
                    "task_id": task.id,
                    "group_id": task.id,
                    "max_turns": infer_max_turns(task.user_turns),
                }
                if is_train:
                    train_records.append(record)
                else:
                    val_records.append(record)

    # Shuffle task 顺序但保持同 task 的 9 条记录相邻（batch_size 需为 9 的倍数）
    # 以 task 为单位 shuffle，每个 task 的 9 条记录保持连续
    def shuffle_by_task(records, rng):
        grouped = defaultdict(list)
        for r in records:
            grouped[r["group_id"]].append(r)
        task_ids = list(grouped.keys())
        rng.shuffle(task_ids)
        result = []
        for tid in task_ids:
            result.extend(grouped[tid])
        return result
    train_records = shuffle_by_task(train_records, rng)
    val_records = shuffle_by_task(val_records, rng)

    # 写出前 fail-fast 检查：每 group_id 必须 9 条且 none/mild/strong 各 3 条
    # 训练侧依赖 batch_size 为 9 的倍数 + 同 task 9 条相邻，分组缺失会让 estimator 静默错算
    _assert_e4_group_integrity(train_records, "train")
    _assert_e4_group_integrity(val_records, "val")

    write_parquet(train_records, Path(out_dir) / "train.parquet")
    write_parquet(val_records, Path(out_dir) / "val.parquet")
    logger.info(f"  E4: train_tasks={len(train_ids)}, val_tasks={len(val_ids)}, "
                f"train_rows={len(train_records)}, val_rows={len(val_records)}")


def _assert_e4_group_integrity(records: list[dict], split_name: str) -> None:
    """E4 数据完整性断言：每 group_id 必须正好 9 行 + 3:3:3 分布。"""
    expected = {LEVEL_NONE: 3, LEVEL_MILD: 3, LEVEL_STRONG: 3}
    grouped: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in records:
        grouped[r["group_id"]][r["perturbation_level"]] += 1
    bad = []
    for gid, dist in grouped.items():
        if dict(dist) != expected:
            bad.append((gid, dict(dist)))
    if bad:
        sample = "\n".join(f"  {gid}: {dist}" for gid, dist in bad[:5])
        raise AssertionError(
            f"E4 {split_name} 分组完整性检查失败：{len(bad)} 个 group_id 不满足 3:3:3。\n"
            f"前 5 个异常 group:\n{sample}"
        )
    logger.info(f"  E4 {split_name} 分组完整性 OK：{len(grouped)} groups × 9 records (3:3:3)")


def main():
    base_dir = Path(__file__).resolve().parent.parent / "data"
    out_dir = base_dir / "verl"
    out_dir.mkdir(parents=True, exist_ok=True)

    exp4_dir = out_dir / "exp4_schemashift"
    prepare_exp4(str(base_dir), str(exp4_dir))


if __name__ == "__main__":
    main()
