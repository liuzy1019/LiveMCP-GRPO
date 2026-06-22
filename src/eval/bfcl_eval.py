"""
BFCL 评估模块。

对训练 checkpoint 进行独立评估，计算：
- macro_overall_pass@1: 五个子类等权平均
- raw_pass@1: 全样本微平均
- strong_pass@1: 主要鲁棒性能指标
- robust_avg: mean(none, mild, strong)
- robustness_gap: pass@1(none) - pass@1(strong)，抗扰动衰减指标
- relative_gap: gap / pass@1(none)

评估方式说明：
- 本模块实现 AST 匹配（response-based），不做 API 执行。
- Online reward 和 offline eval 共享同一套 _args_match / _normalize_value 逻辑。
- 支持多轮推理（与训练 agent loop 一致）和多次采样（pass@k 估计）。
- 支持 seen/unseen 分组报告泛化能力。
- 本实验主要评估 BFCL-v3 multi-turn subset 上的 schema robustness，
  不评价 single-turn function calling general capability。

结论标准：在 pass@1(none) 不明显下降的前提下，
pass@1(strong) 越高、robustness_gap 越小，说明 schema robustness 越好。
"""
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

from loguru import logger

_PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))
sys.path.insert(0, str(_PROJECT_DIR / "verl"))

from src.eval.matching import _normalize_value, _args_match, map_enum_values  # noqa: F401

# BFCL V3 多轮子类
MULTI_TURN_CATEGORIES = ["base", "composite", "long_context", "miss_func", "miss_param"]


def _load_parquet(data_path: str) -> list[dict]:
    """加载 parquet 数据为 dict 列表。"""
    import pyarrow.parquet as pq
    table = pq.read_table(data_path)
    return table.to_pylist()


def _infer_category(task_id: str) -> str:
    """从 task_id 推断 BFCL 子类。"""
    for cat in MULTI_TURN_CATEGORIES:
        if f"multi_turn_{cat}" in task_id:
            return cat
    return "unknown"


def _evaluate_single_sample(
    model_outputs: list[dict],
    ground_truth: list[list[str]],
) -> bool:
    """判断单个样本的 agent 输出是否匹配 ground truth（AST 匹配）。

    按轮次匹配：
    - 非空 GT turn：模型必须输出对应的 tool calls（顺序、数量、内容精确匹配）
    - 空 GT turn（[]）：模型在该轮不应调用任何工具（应输出文本回复）

    当前实现为扁平匹配（因为推理是单轮收集所有调用），
    但保留按轮次结构以便后续扩展为真正的多轮推理。
    """
    from src.agent_loop.bfcl_agent_loop import _parse_bfcl_native_args

    gt_flat = []
    has_empty_turn = False
    for turn_gt in ground_truth:
        turn_calls = []
        items = turn_gt if isinstance(turn_gt, list) else [turn_gt]
        for gt_item in items:
            if isinstance(gt_item, str):
                turn_calls.append(_parse_bfcl_native_args(gt_item))
        if not turn_calls:
            has_empty_turn = True
        gt_flat.extend(turn_calls)

    if not gt_flat and not has_empty_turn:
        return len(model_outputs) == 0

    if not gt_flat and has_empty_turn:
        return len(model_outputs) == 0

    agent_dicts = [
        (fc.get("name", ""), fc.get("arguments", {}))
        for fc in model_outputs
    ]

    if len(agent_dicts) != len(gt_flat):
        return False

    for (a_name, a_args), (gt_name, gt_args) in zip(agent_dicts, gt_flat):
        if a_name != gt_name:
            return False
        if not _args_match(a_args, gt_args):
            return False

    return True


def _evaluate_per_turn(
    turn_outputs: list[list[dict]],
    ground_truth: list[list[str]],
) -> bool:
    """按轮次匹配（用于真正的多轮推理评估）。

    每个 turn 独立匹配：
    - GT turn 非空：模型该轮的 tool calls 必须精确匹配
    - GT turn 为空（[]）：模型该轮不应调用任何工具
    """
    from src.agent_loop.bfcl_agent_loop import _parse_bfcl_native_args

    # agent loop 在没有后续 user turn 时会多 append 一个空 turn 再 break，
    # 因此允许 GT 之后的尾部空 turn；但 GT 之后的任何非空 turn 都判错。
    if len(turn_outputs) < len(ground_truth):
        return False
    for extra in turn_outputs[len(ground_truth):]:
        if extra:
            return False

    for turn_idx, turn_gt in enumerate(ground_truth):
        gt_calls = []
        items = turn_gt if isinstance(turn_gt, list) else [turn_gt]
        for gt_item in items:
            if isinstance(gt_item, str):
                gt_calls.append(_parse_bfcl_native_args(gt_item))

        agent_calls = [
            (fc.get("name", ""), fc.get("arguments", {}))
            for fc in turn_outputs[turn_idx]
        ]

        if not gt_calls:
            # 空 GT turn：模型不应调用工具
            if agent_calls:
                return False
        else:
            if len(agent_calls) != len(gt_calls):
                return False
            for (a_name, a_args), (gt_name, gt_args) in zip(agent_calls, gt_calls):
                if a_name != gt_name:
                    return False
                if not _args_match(a_args, gt_args):
                    return False

    return True


def _classify_error(
    model_outputs: list[dict],
    ground_truth: list[list],
    stats: dict,
) -> None:
    """对错误样本进行分类统计。

    错误类型互斥，每个错误样本只归入一类。
    stats[key]["total"] 记录该错误类型出现的次数。
    """
    from src.agent_loop.bfcl_agent_loop import _parse_bfcl_native_args

    gt_flat = []
    for turn_gt in ground_truth:
        items = turn_gt if isinstance(turn_gt, list) else [turn_gt]
        for gt_item in items:
            if isinstance(gt_item, str):
                gt_flat.append(_parse_bfcl_native_args(gt_item))

    agent_names = [fc.get("name", "") for fc in model_outputs]
    gt_names = [name for name, _ in gt_flat]

    if len(model_outputs) > len(gt_flat):
        stats["error:extra_call"] += 1
    elif len(model_outputs) < len(gt_flat):
        stats["error:missing_call"] += 1
    elif agent_names != gt_names:
        if sorted(agent_names) == sorted(gt_names):
            stats["error:wrong_order"] += 1
        else:
            stats["error:wrong_function"] += 1
    else:
        stats["error:wrong_argument"] += 1


def _generate_single_turn(llm, tokenizer, messages: list[dict], sampling_params) -> str:
    """用 vLLM 对单条 prompt 做一次生成。"""
    prompt_text = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    outputs = llm.generate([prompt_text], sampling_params)
    return outputs[0].outputs[0].text if outputs else ""


def _run_inference_multi_turn(
    llm, tokenizer, record: dict, sampling_params, max_turns: int = 5
) -> tuple[list[dict], list[list[dict]], dict]:
    """对单条样本运行多轮推理。

    返回:
        flat_calls: 所有解析出的 tool call 扁平列表（用于错误分类与向后兼容）。
        turn_calls: 按"agent 输出轮次"分组的 tool calls。
                    每个元素是该轮内并行的 tool calls 列表；无 tool call 的 agent 轮记为 []。
                    主评估走 _evaluate_per_turn 时使用此字段。
        efficiency: 效率指标 {turns, tool_calls, tokens}。

    多轮逻辑：
    1. 生成 → 解析 tool calls
    2. 如果有 tool calls → 模拟 tool observation → 追加到 messages → 继续生成
    3. 如果无 tool calls → 检查是否有后续 user turn → 追加 → 继续
    4. 无 tool calls 且无后续 user turn → 结束

    这与训练时 agent loop 的行为一致，避免 train-eval mismatch。
    """
    from src.agent_loop.bfcl_agent_loop import BFCLAgentLoop, _parse_bfcl_native_args
    from src.envs.api_mapper import FunctionNameMapper

    prompt_raw = record.get("prompt", "[]")
    try:
        messages = json.loads(prompt_raw) if isinstance(prompt_raw, str) else prompt_raw
    except (json.JSONDecodeError, TypeError):
        messages = [{"role": "user", "content": str(prompt_raw)}]

    name_map = _json_field_to_dict(record.get("name_map_json", "{}"))
    enum_map = _json_field_to_dict(record.get("enum_map_json", "{}"))
    fn_mapper = FunctionNameMapper(name_map=name_map, enum_map=enum_map) if name_map or enum_map else None

    # 后续 user turns
    user_turns_raw = record.get("user_turns_json", "[]")
    try:
        remaining_turns = json.loads(user_turns_raw) if isinstance(user_turns_raw, str) else (user_turns_raw or [])
    except (json.JSONDecodeError, TypeError):
        remaining_turns = []
    user_turn_idx = 0

    all_calls = []
    turn_calls: list[list[dict]] = []
    total_tokens = 0
    num_turns_used = 0

    for turn in range(max_turns):
        response_text = _generate_single_turn(llm, tokenizer, messages, sampling_params)
        if not response_text:
            break
        total_tokens += len(tokenizer.encode(response_text))
        num_turns_used += 1

        # 解析 tool calls
        func_call_strs = BFCLAgentLoop._parse_tool_calls(response_text)
        if not func_call_strs:
            # 无 tool call：当前 agent 轮不调用工具，记为空列表
            turn_calls.append([])
            # 检查是否有后续 user turn
            if remaining_turns and user_turn_idx < len(remaining_turns):
                next_text = BFCLAgentLoop._extract_next_user_message(
                    remaining_turns, user_turn_idx
                )
                if next_text:
                    messages.append({"role": "assistant", "content": response_text})
                    messages.append({"role": "user", "content": next_text})
                    user_turn_idx += 1
                    continue
            break

        # 处理 tool calls
        messages.append({"role": "assistant", "content": response_text})
        this_turn_calls: list[dict] = []
        for fc_str in func_call_strs:
            try:
                call_data = json.loads(fc_str)
                name = call_data.get("name", "")
                args = call_data.get("arguments", {})
            except (json.JSONDecodeError, TypeError):
                name, args = _parse_bfcl_native_args(fc_str)
            if name:
                if fn_mapper:
                    original_name = fn_mapper.resolve(name)
                    if fn_mapper.enum_map:
                        args = map_enum_values(original_name, args, fn_mapper.enum_map)
                    name = original_name
                call_dict = {"name": name, "arguments": args}
                all_calls.append(call_dict)
                this_turn_calls.append(call_dict)
        turn_calls.append(this_turn_calls)

        # 模拟 tool observation（评估时不执行真实 API，返回占位结果）
        obs = json.dumps({"status": "ok", "data": "simulated"})
        messages.append({"role": "tool", "content": obs})

        # 注入后续 user turn（如果有）
        if remaining_turns and user_turn_idx < len(remaining_turns):
            next_text = BFCLAgentLoop._extract_next_user_message(
                remaining_turns, user_turn_idx
            )
            if next_text:
                messages.append({"role": "user", "content": next_text})
                user_turn_idx += 1

    efficiency = {
        "turns": num_turns_used,
        "tool_calls": len(all_calls),
        "tokens": total_tokens,
    }
    return all_calls, turn_calls, efficiency


def _json_field_to_dict(value) -> dict:
    """Parse parquet JSON-map fields; malformed or empty fields become {}."""
    if isinstance(value, dict):
        return value
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _pass_at_k(n: int, c: int, k: int) -> float:
    """估计 pass@k：n 次采样中 c 次成功，至少成功一次的概率。

    使用组合公式：1 - C(n-c, k) / C(n, k)
    """
    if n - c < k:
        return 1.0
    return 1.0 - math.prod(range(n - c - k + 1, n - c + 1)) / math.prod(range(n - k + 1, n + 1))


def evaluate_checkpoint(
    model_path: str,
    data_path: str,
    output_dir: Optional[str] = None,
    num_samples: Optional[int] = None,
    n_samples_per_task: int = 1,
    temperature: float = 0.0,
    train_task_ids: Optional[set[str]] = None,
    multi_turn: bool = True,
    max_turns: int = 5,
) -> dict:
    """评估一个 checkpoint 在 BFCL 数据上的鲁棒性。

    Args:
        model_path: checkpoint 路径。
        data_path: 评估数据 parquet 路径。
        output_dir: 结果输出目录。
        num_samples: 限制评估样本数（调试用）。
        n_samples_per_task: 每个 task 采样次数（>1 时用 temperature>0 估计方差）。
        temperature: 采样温度。0.0=贪心，>0 时多次采样估计 pass@k。
        train_task_ids: 训练集 task_id 集合，用于 seen/unseen 分组。
        multi_turn: 是否使用多轮推理（与训练一致）。False 时退化为单轮。
        max_turns: 多轮推理最大轮数。

    Returns:
        完整评估结果字典。
    """
    records = _load_parquet(data_path)

    # 评估侧去重：同一 (task_id, perturbation_level) 在 E4 train/val 中可能出现多次
    # （上游 build_parquet 为每个 task 生成 3 条用于 GRPO group sampling）
    # eval 阶段只需评一次，避免分母虚高 / 重复采样污染统计。
    seen_keys = set()
    deduped = []
    for r in records:
        key = (r.get("task_id", ""), r.get("perturbation_level", "none"))
        if key in seen_keys:
            continue
        seen_keys.add(key)
        deduped.append(r)
    if len(deduped) != len(records):
        logger.info(f"评估去重: {len(records)} -> {len(deduped)} 条 (按 task_id+level)")
    records = deduped

    if num_samples:
        records = records[:num_samples]

    # 加载模型
    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        raise RuntimeError(
            "evaluate_checkpoint 需要 vllm 进行推理，请安装: pip install vllm>=0.6.0"
        )
    from transformers import AutoTokenizer

    logger.info(f"加载模型: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    llm = LLM(model=model_path, trust_remote_code=True)

    # 采样参数：多次采样时用 temperature>0
    effective_temp = temperature if n_samples_per_task > 1 else 0.0
    sampling_params = SamplingParams(temperature=effective_temp, max_tokens=2048)

    # 统计容器
    # 按 task_id 聚合多次采样结果
    task_results: dict[str, dict] = defaultdict(lambda: {
        "successes": 0, "attempts": 0,
        "level": "none", "category": "unknown", "is_seen": True,
    })
    # 错误分类计数
    error_counts: dict[str, int] = defaultdict(int)
    # 效率指标累计
    efficiency_accum = {"turns": 0, "tool_calls": 0, "tokens": 0, "count": 0}

    total_evaluated = 0
    for idx, record in enumerate(records):
        level = record.get("perturbation_level", "none")
        task_id = record.get("task_id", "")
        category = _infer_category(task_id)
        is_seen = task_id in train_task_ids if train_task_ids else True

        gt_raw = record.get("ground_truth_json", "[]")
        try:
            gt = json.loads(gt_raw) if isinstance(gt_raw, str) else gt_raw
        except json.JSONDecodeError:
            gt = []

        # 多次采样
        for sample_idx in range(n_samples_per_task):
            if multi_turn:
                model_outputs, turn_outputs, eff = _run_inference_multi_turn(
                    llm, tokenizer, record, sampling_params, max_turns=max_turns
                )
                # 主评估走 per-turn 匹配，避免把"模型一轮内一次性输出全部调用"
                # 误判为正确（与训练 reward 一致）。
                correct = _evaluate_per_turn(turn_outputs, gt)
            else:
                # 单轮 fallback（向后兼容）
                model_outputs, eff = _run_inference_single_turn(
                    llm, tokenizer, record, sampling_params
                )
                correct = _evaluate_single_sample(model_outputs, gt)

            # 效率累计
            efficiency_accum["turns"] += eff["turns"]
            efficiency_accum["tool_calls"] += eff["tool_calls"]
            efficiency_accum["tokens"] += eff["tokens"]
            efficiency_accum["count"] += 1

            # 错误分类
            if not correct:
                if model_outputs:
                    _classify_error(model_outputs, gt, error_counts)
                else:
                    error_counts["error:missing_call"] += 1

            # 按 task+level 聚合（每条 record 是唯一的 task+level 组合）
            key = f"{task_id}__{level}__{sample_idx}"
            task_results[key]["successes"] += int(correct)
            task_results[key]["attempts"] += 1
            task_results[key]["level"] = level
            task_results[key]["category"] = category
            task_results[key]["is_seen"] = is_seen
            task_results[key]["task_id"] = task_id

        total_evaluated += 1
        if (idx + 1) % 50 == 0:
            logger.info(f"  进度: {idx + 1}/{len(records)}")

    # 汇总结果
    result = _compile_results(task_results, error_counts, efficiency_accum, n_samples_per_task)

    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        with open(output_path / "eval_results.json", "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        logger.info(f"结果已保存: {output_path / 'eval_results.json'}")

    return result


def _run_inference_single_turn(
    llm, tokenizer, record: dict, sampling_params
) -> tuple[list[dict], dict]:
    """单轮推理 fallback（向后兼容）。"""
    from src.agent_loop.bfcl_agent_loop import BFCLAgentLoop, _parse_bfcl_native_args
    from src.envs.api_mapper import FunctionNameMapper

    prompt_raw = record.get("prompt", "[]")
    try:
        messages = json.loads(prompt_raw) if isinstance(prompt_raw, str) else prompt_raw
    except (json.JSONDecodeError, TypeError):
        messages = [{"role": "user", "content": str(prompt_raw)}]

    name_map = _json_field_to_dict(record.get("name_map_json", "{}"))
    enum_map = _json_field_to_dict(record.get("enum_map_json", "{}"))
    fn_mapper = FunctionNameMapper(name_map=name_map, enum_map=enum_map) if name_map or enum_map else None

    response_text = _generate_single_turn(llm, tokenizer, messages, sampling_params)
    if not response_text:
        return [], {"turns": 1, "tool_calls": 0, "tokens": 0}

    total_tokens = len(tokenizer.encode(response_text))
    func_call_strs = BFCLAgentLoop._parse_tool_calls(response_text)
    all_calls = []
    for fc_str in func_call_strs:
        try:
            call_data = json.loads(fc_str)
            name = call_data.get("name", "")
            args = call_data.get("arguments", {})
        except (json.JSONDecodeError, TypeError):
            name, args = _parse_bfcl_native_args(fc_str)
        if name:
            if fn_mapper:
                original_name = fn_mapper.resolve(name)
                if fn_mapper.enum_map:
                    args = map_enum_values(original_name, args, fn_mapper.enum_map)
                name = original_name
            all_calls.append({"name": name, "arguments": args})

    return all_calls, {"turns": 1, "tool_calls": len(all_calls), "tokens": total_tokens}


def _compile_results(
    task_results: dict,
    error_counts: dict,
    efficiency_accum: dict,
    n_samples_per_task: int,
) -> dict:
    """从多维度统计中编译最终结果。"""

    # 按维度聚合 pass@1
    level_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    category_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    cat_level_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    overall_stats = {"correct": 0, "total": 0}
    seen_stats = {"correct": 0, "total": 0}
    unseen_stats = {"correct": 0, "total": 0}

    for key, info in task_results.items():
        # pass@1: 该 task 是否至少成功一次（多次采样时）
        success = info["successes"] > 0
        level = info["level"]
        category = info["category"]
        is_seen = info["is_seen"]

        level_stats[level]["total"] += 1
        level_stats[level]["correct"] += int(success)
        category_stats[category]["total"] += 1
        category_stats[category]["correct"] += int(success)
        cat_level_stats[f"{category}:{level}"]["total"] += 1
        cat_level_stats[f"{category}:{level}"]["correct"] += int(success)
        overall_stats["total"] += 1
        overall_stats["correct"] += int(success)

        if is_seen:
            seen_stats["total"] += 1
            seen_stats["correct"] += int(success)
        else:
            unseen_stats["total"] += 1
            unseen_stats["correct"] += int(success)

    def _rate(s):
        return s["correct"] / s["total"] if s["total"] > 0 else 0.0

    # 按扰动强度
    pass_by_level = {lv: _rate(level_stats[lv]) for lv in ["none", "mild", "strong"]}

    # 按子类
    pass_by_category = {cat: _rate(category_stats[cat]) for cat in MULTI_TURN_CATEGORIES}

    # 各子类的 robustness gap
    gap_by_category = {}
    for cat in MULTI_TURN_CATEGORIES:
        none_rate = _rate(cat_level_stats[f"{cat}:none"])
        strong_rate = _rate(cat_level_stats[f"{cat}:strong"])
        gap_by_category[cat] = none_rate - strong_rate

    # macro_overall: 按子类等权平均
    active_cats = [cat for cat in MULTI_TURN_CATEGORIES if category_stats[cat]["total"] > 0]
    macro_overall = (
        sum(pass_by_category[cat] for cat in active_cats) / len(active_cats)
        if active_cats else 0.0
    )

    raw_pass = _rate(overall_stats)

    # 核心鲁棒性指标
    none_pass = pass_by_level.get("none", 0.0)
    mild_pass = pass_by_level.get("mild", 0.0)
    strong_pass = pass_by_level.get("strong", 0.0)
    robustness_gap = none_pass - strong_pass
    robust_avg = (none_pass + mild_pass + strong_pass) / 3.0
    relative_gap = robustness_gap / none_pass if none_pass > 0 else 0.0

    # 错误类型分布（归一化为比率）
    total_errors = sum(error_counts.values())
    error_breakdown = {}
    for err_type in ["invalid_format", "wrong_function", "wrong_argument",
                     "extra_call", "missing_call", "wrong_order"]:
        count = error_counts.get(f"error:{err_type}", 0)
        error_breakdown[f"{err_type}_rate"] = count / total_errors if total_errors > 0 else 0.0
    error_breakdown["total_errors"] = total_errors

    # 效率指标
    n_eval = efficiency_accum["count"]
    efficiency = {
        "avg_turns": efficiency_accum["turns"] / n_eval if n_eval > 0 else 0.0,
        "avg_tool_calls": efficiency_accum["tool_calls"] / n_eval if n_eval > 0 else 0.0,
        "avg_tokens": efficiency_accum["tokens"] / n_eval if n_eval > 0 else 0.0,
    }

    # seen/unseen 分组
    generalization = {
        "seen_pass@1": _rate(seen_stats),
        "unseen_pass@1": _rate(unseen_stats),
        "seen_n": seen_stats["total"],
        "unseen_n": unseen_stats["total"],
    }

    return {
        # 主指标
        "macro_overall_pass@1": macro_overall,
        "raw_pass@1": raw_pass,
        # 鲁棒性指标
        "strong_pass@1": strong_pass,
        "robust_avg": robust_avg,
        "robustness_gap": robustness_gap,
        "relative_gap": relative_gap,
        # 按扰动强度
        "pass@1_by_level": pass_by_level,
        # 按子类
        "pass@1_by_category": pass_by_category,
        # 各子类 robustness gap
        "robustness_gap_by_category": gap_by_category,
        # 泛化能力
        "generalization": generalization,
        # 错误分布
        "error_breakdown": error_breakdown,
        # 效率
        "efficiency": efficiency,
        # 统计量
        "n_samples": overall_stats["total"],
        "n_correct": overall_stats["correct"],
        "n_samples_per_task": n_samples_per_task,
    }


# 向后兼容别名（旧测试引用此函数名）
def _run_inference_single(llm, tokenizer, record, sampling_params, max_turns=1):
    """向后兼容：单轮推理，返回 tool call 列表。"""
    calls, _ = _run_inference_single_turn(llm, tokenizer, record, sampling_params)
    return calls
