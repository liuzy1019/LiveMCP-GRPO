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

SFT_FILES = [
    "BFCL_v3_simple.json",
    "BFCL_v3_multiple.json",
    "BFCL_v3_parallel.json",
    "BFCL_v3_parallel_multiple.json",
]

LEVEL_DISTRIBUTION = [LEVEL_NONE, LEVEL_MILD, LEVEL_STRONG]
# E4 数据布局：1 task -> 9 records (3 none + 3 mild + 3 strong)，训练时 rollout.n=1、batch 是 9 的倍数
# E5 数据布局：1 task -> 3 records (none/mild/strong 各 1)，训练时 rollout.n=3、标准 GRPO 按 prompt 内 rollout 比较
E4_LEVEL_ASSIGNMENTS = [LEVEL_NONE] * 3 + [LEVEL_MILD] * 3 + [LEVEL_STRONG] * 3


def _canonical_sft_arg_value(value):
    """Normalize BFCL answer-list values into one SFT argument value."""
    if isinstance(value, list):
        if not value:
            return []
        for candidate in value:
            if candidate not in ("", None):
                return candidate
        return value[0]
    return value


def _bfcl_answer_dict_to_tool_calls(answer: dict) -> list[dict]:
    """Convert BFCL dict GT format into prompt-compatible tool-call dicts."""
    calls = []
    for name, arguments in answer.items():
        arguments = arguments if isinstance(arguments, dict) else {}
        calls.append({
            "name": name,
            "arguments": {
                k: _canonical_sft_arg_value(v)
                for k, v in arguments.items()
            },
        })
    return calls


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


def prepare_exp3(data_dir: str, out_dir: str, seed: int = 42, val_ratio: float = 0.1):
    logger.info("=== E3: GRPO 基线 ===")
    gt_all = _load_ground_truth(data_dir)
    records = []
    for fname in MULTI_TURN_FILES:
        loader = BFCLDataLoader(str(Path(data_dir) / fname))
        for task in loader.tasks:
            records.append({
                "prompt": build_messages(task.functions, task.user_turns),
                "data_source": "bfcl",
                "functions_json": json.dumps(task.functions),
                "initial_config": json.dumps(task.initial_config),
                "involved_classes": json.dumps(task.involved_classes),
                "user_turns_json": json.dumps(task.user_turns[1:]),
                "ground_truth_json": json.dumps(gt_all.get(task.id, [])),
                "perturbation_level": LEVEL_NONE,
                "name_map_json": "{}",
                "task_id": task.id,
                "max_turns": infer_max_turns(task.user_turns),
            })
    rng = random.Random(seed)
    rng.shuffle(records)
    split = int(len(records) * (1 - val_ratio))
    write_parquet(records[:split], Path(out_dir) / "train.parquet")
    write_parquet(records[split:], Path(out_dir) / "val.parquet")
    logger.info(f"  E3: train={split}, val={len(records)-split}")


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


# ═══════════════════════════════════════════
# E5: Aug-only 消融 — 混合扰动 + 标准 GRPO
# ═══════════════════════════════════════════

def _read_parquet_records(path: Path) -> list[dict]:
    try:
        import pyarrow.parquet as pq
    except ImportError:
        logger.error("需要 pyarrow: pip install pyarrow")
        sys.exit(1)
    return pq.read_table(str(path)).to_pylist()


def prepare_exp5(exp4_dir: str, out_dir: str) -> None:
    """E5 Aug-only 数据：从 E4 每个 task 抽 none/mild/strong 各 1 条。

    E5 使用标准 GRPO：每条 prompt 通过 rollout.n=3 产生 3 个 rollout，advantage
    在同一 (task, perturbation_level) prompt 内比较，不跨 level 做分层归一化。
    """
    logger.info("=== E5: Aug-only (3 records/task) ===")
    exp4_path = Path(exp4_dir)
    out_path = Path(out_dir)
    for split_name in ("train", "val"):
        records = _read_parquet_records(exp4_path / f"{split_name}.parquet")
        selected = _select_exp5_records(records, split_name)
        write_parquet(selected, out_path / f"{split_name}.parquet")
        logger.info(f"  E5 {split_name}: rows={len(selected)}, tasks={len(selected) // 3}")


def _select_exp5_records(exp4_records: list[dict], split_name: str) -> list[dict]:
    by_task: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    task_order: list[str] = []
    seen_tasks: set[str] = set()

    for record in exp4_records:
        tid = record.get("task_id") or record.get("group_id")
        level = record.get("perturbation_level", LEVEL_NONE)
        if not tid:
            raise AssertionError(f"E5 {split_name} 数据缺少 task_id/group_id")
        if tid not in seen_tasks:
            task_order.append(tid)
            seen_tasks.add(tid)
        by_task[tid][level].append(record)

    selected: list[dict] = []
    bad = []
    for tid in task_order:
        per_level = by_task[tid]
        dist = {level: len(per_level.get(level, [])) for level in LEVEL_DISTRIBUTION}
        if any(count < 1 for count in dist.values()):
            bad.append((tid, dist))
            continue
        for level in LEVEL_DISTRIBUTION:
            row = dict(per_level[level][0])
            row["uid"] = f"{tid}___{level}"
            row["group_id"] = tid
            selected.append(row)

    if bad:
        sample = "\n".join(f"  {tid}: {dist}" for tid, dist in bad[:5])
        raise AssertionError(
            f"E5 {split_name} 分组完整性失败：{len(bad)} 个 task 缺少 none/mild/strong。\n"
            f"前 5 个异常 task:\n{sample}"
        )

    _assert_exp5_group_integrity(selected, split_name)
    return selected


def _assert_exp5_group_integrity(records: list[dict], split_name: str) -> None:
    expected = {LEVEL_NONE: 1, LEVEL_MILD: 1, LEVEL_STRONG: 1}
    grouped: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for record in records:
        grouped[record["task_id"]][record["perturbation_level"]] += 1

    bad = []
    for tid, dist in grouped.items():
        if dict(dist) != expected:
            bad.append((tid, dict(dist)))
    if bad:
        sample = "\n".join(f"  {tid}: {dist}" for tid, dist in bad[:5])
        raise AssertionError(
            f"E5 {split_name} 分组完整性检查失败：{len(bad)} 个 task 不满足 1:1:1。\n"
            f"前 5 个异常 task:\n{sample}"
        )
    logger.info(f"  E5 {split_name} 分组完整性 OK：{len(grouped)} tasks x 3 records (1:1:1)")


# ═══════════════════════════════════════════
# E2: SFT 基线
# ═══════════════════════════════════════════

def prepare_exp2(data_dir: str, out_dir: str, val_ratio: float = 0.1):
    logger.info("=== E2: SFT ===")

    # 预先加载 ground truth
    gt_by_id: dict[str, list] = {}
    gt_dir = Path(data_dir) / "possible_answer"
    for fname in SFT_FILES:
        gt_path = gt_dir / fname
        if not gt_path.exists():
            continue
        with open(gt_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    tid = d.get("id", "")
                    gt_by_id[tid] = d.get("ground_truth", [])
                except json.JSONDecodeError:
                    continue

    records = []
    for fname in SFT_FILES:
        fpath = Path(data_dir) / fname
        if not fpath.exists():
            continue
        with open(fpath) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                    task_id = raw.get("id", "")
                    funcs = raw.get("function", [])
                    gt = gt_by_id.get(task_id, [])
                    question = raw.get("question", [])
                    if not funcs:
                        continue
                    messages = build_messages(funcs, question)
                    # 将 ground truth 转为系统 prompt 教的格式
                    # 系统 prompt: <tool_call>{"name": "...", "arguments": {...}}</tool_call>
                    # 每个 gt 条目是 BFCL 原生格式字符串 "func(args)"，需转为 tool_call XML
                    gt_calls = []
                    for gt_item in gt:
                        if isinstance(gt_item, str):
                            from src.agent_loop.bfcl_agent_loop import _parse_bfcl_native_args
                            name, args = _parse_bfcl_native_args(gt_item)
                            json_str = json.dumps({"name": name, "arguments": args}, ensure_ascii=False)
                            gt_calls.append(f"<tool_call>{json_str}</tool_call>")
                        elif isinstance(gt_item, dict):
                            for call in _bfcl_answer_dict_to_tool_calls(gt_item):
                                json_str = json.dumps(call, ensure_ascii=False)
                                gt_calls.append(f"<tool_call>{json_str}</tool_call>")
                        else:
                            gt_calls.append(str(gt_item))
                    target_str = "\n".join(gt_calls) if gt_calls else ""
                    records.append({
                        "prompt": json.dumps(messages),
                        "target": target_str,
                        "data_source": "bfcl_sft",
                        "functions_json": json.dumps(funcs),
                    })
                except json.JSONDecodeError:
                    continue
    rng = random.Random(42)
    rng.shuffle(records)
    split = int(len(records) * (1 - val_ratio))
    write_parquet(records[:split], Path(out_dir) / "train.parquet")
    write_parquet(records[split:], Path(out_dir) / "val.parquet")
    logger.info(f"  E2: train={split}, val={len(records)-split}")


def main():
    base_dir = Path(__file__).resolve().parent.parent / "data"
    out_dir = base_dir / "verl"
    out_dir.mkdir(parents=True, exist_ok=True)

    prepare_exp2(str(base_dir), str(out_dir / "exp2_sft"))
    prepare_exp3(str(base_dir), str(out_dir / "exp3_grpo"))
    exp4_dir = out_dir / "exp4_schemashift"
    prepare_exp4(str(base_dir), str(exp4_dir))
    prepare_exp5(str(exp4_dir), str(out_dir / "exp5_aug_only"))


if __name__ == "__main__":
    main()
