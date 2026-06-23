"""SchemaShift reward function — verl custom_reward_function 接口。

通过 verl config 的 custom_reward_function.path 指定本文件，
custom_reward_function.name 指定 "compute_score"。

接口签名：
    compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs) -> float | dict

verl 的 NaiveRewardManager 会对每个样本调用此函数，
将返回值放在 response 最后一个 token 的位置。

多步奖励设计（参考 COVERT 逐步平均 + PROVE coverage 信号）：
  - 单步 episode（call_only）：只评估第一步，和之前一致
  - 多步 episode（call_then_final / call_then_call）：
    1. 解析模型输出中的多个 action（支持 tool_call 后跟 final_answer）
    2. 逐步与 oracle_actions 对齐评估，取加权平均
    3. 加入 trajectory-level 信号：coverage bonus / trajectory penalty
"""

import json
import re
from typing import Any, Optional

from src.reward.action_parser import ActionParser, ParsedAction
from src.reward.component_reward import ComponentReward, OracleAction, SampleMetadata


# 模块级单例（避免每次调用都重新创建）
_parser = ActionParser(strict=False)
_reward_fn = ComponentReward()

# 多步解析：匹配所有 <tag>content</tag> 对
_MULTI_TAG_PATTERN = re.compile(
    r"<(tool_call|final_answer|ask_clarification|report_error)>(.*?)</\1>",
    re.DOTALL,
)

_COMPONENT_KEYS = (
    "format",
    "schema_valid",
    "tool_selection",
    "argument_keys",
    "argument_values",
    "no_extra_call",
    "final_answer_match",
    "error_type_match",
    "clarification_match",
)

# ============================================================
# 多步奖励权重配置
# ============================================================

# 多步 episode 中各步骤的权重分配策略
# 第一步（tool_call）权重较高，因为它决定了后续步骤能否正确执行
_STEP_WEIGHT_FIRST = 0.6   # 第一步权重
_STEP_WEIGHT_LATER = 0.4   # 后续步骤总权重（均分）

# Trajectory-level 信号
_COVERAGE_BONUS = 0.1       # 覆盖所有 oracle steps 的 bonus
_TRAJECTORY_PENALTY = -0.15  # 多步任务中模型直接给 final_answer 的惩罚
_ALL_EXACT_BONUS = 0.1      # 所有步骤都 exact match 的 bonus


def _reward_info(
    *,
    score: float,
    exact_success: bool = False,
    action_type_match: bool = False,
    oracle_action_type: str = "",
    model_action_type: str = "",
    components: Optional[dict[str, Any]] = None,
    error: str = "",
    n_model_steps: int = 1,
    n_oracle_steps: int = 1,
    coverage_ratio: float = 0.0,
) -> dict[str, Any]:
    """Build a fixed-shape reward dict for verl validation aggregation."""
    components = components or {}
    info: dict[str, Any] = {
        "score": float(score),
        "exact_success": float(exact_success),
        "action_type_match": float(action_type_match),
        "oracle_action_type": oracle_action_type,
        "model_action_type": model_action_type,
        "error": error,
        # 多步诊断指标
        "n_model_steps": float(n_model_steps),
        "n_oracle_steps": float(n_oracle_steps),
        "coverage_ratio": float(coverage_ratio),
    }
    for name in _COMPONENT_KEYS:
        info[f"component_{name}"] = float(components.get(name, 0.0))
    return info


def _parse_multi_step_actions(solution_str: str) -> list[ParsedAction]:
    """从模型输出中解析多个 action（多步输出）。

    模型可能在一次 response 中输出：
      <tool_call>...</tool_call>
      <final_answer>...</final_answer>

    或者只输出一个 action。

    Returns:
        解析出的 ParsedAction 列表（按出现顺序）。
    """
    matches = list(_MULTI_TAG_PATTERN.finditer(solution_str))

    if not matches:
        # 没有标签格式，用默认 parser 解析整体
        parsed = _parser.parse(solution_str)
        return [parsed]

    if len(matches) == 1:
        # 只有一个标签，用默认 parser 解析（它有更完善的 fallback）
        parsed = _parser.parse(solution_str)
        return [parsed]

    # 多个标签：逐个解析
    actions = []
    for match in matches:
        tag = match.group(1)
        content = match.group(2).strip()
        # 构造单标签文本让 parser 解析
        single_tagged = f"<{tag}>{content}</{tag}>"
        parsed = _parser.parse(single_tagged)
        if parsed.action_type != "unparseable":
            actions.append(parsed)

    # 如果多标签解析全部失败，fallback 到整体解析
    if not actions:
        parsed = _parser.parse(solution_str)
        return [parsed]

    return actions


def _build_oracle_action(oracle_action_data: dict) -> OracleAction:
    """从 oracle_action_data dict 构建 OracleAction 对象。"""
    return OracleAction(
        action_type=oracle_action_data.get("action_type", "tool_call"),
        tool_calls=oracle_action_data.get("tool_calls", []),
        match_mode=oracle_action_data.get("match_mode", "set"),
        final_answer=oracle_action_data.get("final_answer") or "",
        error_info=oracle_action_data.get("error_info") or "",
    )


def _compute_single_step_reward(
    solution_str: str,
    oracle_action: OracleAction,
    metadata: SampleMetadata,
) -> dict:
    """计算单步 reward（原有逻辑）。

    P0-1 修复：检测 extra tags。如果模型在正确的第一个 action 之后
    还输出了额外的 tagged actions，不授予 exact_success。
    """
    result = _reward_fn.compute(
        model_output=solution_str,
        oracle=oracle_action,
        metadata=metadata,
    )

    # P0-1: 检测 extra tags — 单步 oracle 不应有多个 tagged actions
    all_tags = list(_MULTI_TAG_PATTERN.finditer(solution_str))
    has_extra_tags = len(all_tags) > 1

    exact_success = result.exact_success
    score = result.total_reward

    if has_extra_tags:
        # 有 extra actions → 不授予 exact_success，施加惩罚
        exact_success = False
        # 重新计算 score：使用 partial reward 路径（0.3 * partial）
        # 并施加 extra action penalty
        partial_reward = result.diagnostics.get("partial_reward", 0.0)
        if result.exact_success:
            # 原本是 exact 但因 extra tags 降级
            score = 0.3 * partial_reward
        extra_penalty = -0.05 * (len(all_tags) - 1)
        score += max(extra_penalty, -0.1)

    # P2-5: coverage_ratio 基于 exact_success
    coverage_ratio = 1.0 if exact_success else 0.0

    return _reward_info(
        score=score,
        exact_success=exact_success,
        action_type_match=result.action_type_match,
        oracle_action_type=result.oracle_action_type,
        model_action_type=result.model_action_type,
        components=result.components,
        n_model_steps=len(all_tags) if all_tags else 1,
        n_oracle_steps=1,
        coverage_ratio=coverage_ratio,
    )


def _compute_multi_step_reward(
    solution_str: str,
    oracle_actions: list[dict],
    metadata: SampleMetadata,
    episode_type: str,
) -> dict:
    """计算多步 reward。

    策略（参考 COVERT 逐步平均 + PROVE coverage）：
    1. 从模型输出中解析多个 action
    2. 将模型 actions 与 oracle_actions 按顺序对齐
    3. 逐步计算 ComponentReward，加权平均
    4. 加入 trajectory-level 信号

    对于模型只输出一步的情况（最常见）：
    - 如果第一步正确（tool_call），给 coverage bonus（模型知道要先调工具）
    - 如果模型在多步任务中直接给 final_answer，给 trajectory penalty
    """
    model_actions = _parse_multi_step_actions(solution_str)
    n_oracle = len(oracle_actions)
    n_model = len(model_actions)

    # P0-3: 检测"同一 turn 多 action"违规
    # 在交互式 replay 中，正确的多步输出应该是：
    #   <tool_call>...</tool_call> [observation] <final_answer>...</final_answer>
    # 如果两个 action tag 之间没有 observation（非 tag 文本），则视为同一 turn 违规
    same_turn_violation = False
    if n_model > 1:
        # 检查 action tags 之间是否有 observation 分隔
        tag_positions = list(_MULTI_TAG_PATTERN.finditer(solution_str))
        if len(tag_positions) >= 2:
            for i in range(len(tag_positions) - 1):
                end_of_prev = tag_positions[i].end()
                start_of_next = tag_positions[i + 1].start()
                between_text = solution_str[end_of_prev:start_of_next].strip()
                # 如果两个 tag 之间没有实质性内容（observation），视为同一 turn
                if len(between_text) < 5:  # 少于 5 字符视为无 observation
                    same_turn_violation = True
                    break

    # 逐步评估：将 model actions 与 oracle actions 按顺序对齐
    step_rewards = []
    step_exact = []
    step_type_match = []
    step_results = []  # 保存每步的 RewardResult，避免重复计算

    for i in range(max(n_model, n_oracle)):
        if i < n_model and i < n_oracle:
            # 有对应的 model action 和 oracle action
            oracle_action = _build_oracle_action(oracle_actions[i])
            # 用单个 action 的 raw_output 计算 reward
            model_output = model_actions[i].raw_output
            result = _reward_fn.compute(
                model_output=model_output,
                oracle=oracle_action,
                metadata=metadata,
            )
            step_rewards.append(result.total_reward)
            step_exact.append(result.exact_success)
            step_type_match.append(result.action_type_match)
            step_results.append(result)
        elif i < n_oracle:
            # 模型步数不足（未覆盖的 oracle step）→ 0 分
            step_rewards.append(0.0)
            step_exact.append(False)
            step_type_match.append(False)
            step_results.append(None)
        else:
            # 模型多余步骤（超出 oracle）→ 不计入 reward，但影响 efficiency
            pass

    # 加权平均 step rewards
    if len(step_rewards) == 1:
        weighted_reward = step_rewards[0]
    else:
        # 第一步权重 _STEP_WEIGHT_FIRST，后续步骤均分 _STEP_WEIGHT_LATER
        first_weight = _STEP_WEIGHT_FIRST
        later_weight = _STEP_WEIGHT_LATER / max(len(step_rewards) - 1, 1)
        weighted_reward = first_weight * step_rewards[0]
        for r in step_rewards[1:]:
            weighted_reward += later_weight * r

    # Coverage ratio：基于 exact success 覆盖的 oracle steps
    # 只有 name + keys + values 全部正确才算覆盖（对齐 replay 环境的释放条件）
    exact_covered = sum(step_exact) if step_exact else 0
    coverage_ratio = exact_covered / n_oracle if n_oracle > 0 else 0.0

    # P0-1 修复：extra actions 检测
    has_extra_steps = n_model > n_oracle
    all_oracle_exact = len(step_exact) == n_oracle and all(step_exact)
    # trajectory_exact 要求：所有 oracle 步骤精确匹配 + 没有多余步骤 + 没有同 turn 违规
    trajectory_exact = all_oracle_exact and not has_extra_steps and not same_turn_violation

    # Trajectory-level 信号
    trajectory_bonus = 0.0

    # 1. Coverage bonus + All exact bonus：仅在 trajectory_exact 时授予
    if trajectory_exact:
        trajectory_bonus += _COVERAGE_BONUS
        trajectory_bonus += _ALL_EXACT_BONUS

    # 2. P0-3: 同一 turn 多 action 惩罚（模型试图绕过交互式约束）
    if same_turn_violation:
        trajectory_bonus += _TRAJECTORY_PENALTY  # 与跳过 tool_call 相同的惩罚

    # 3. Trajectory penalty：多步任务中模型直接给 final_answer（跳过 tool_call）
    if n_oracle > 1 and n_model >= 1:
        first_oracle_type = oracle_actions[0].get("action_type", "tool_call")
        first_model_type = model_actions[0].action_type
        if first_oracle_type == "tool_call" and first_model_type == "final_answer":
            trajectory_bonus += _TRAJECTORY_PENALTY

    # 4. Efficiency penalty：模型有任何多余步骤都惩罚（P0-1: 包括恰好多一步）
    if has_extra_steps:
        efficiency_penalty = -0.05 * (n_model - n_oracle)
        trajectory_bonus += max(efficiency_penalty, -0.1)

    total_reward = weighted_reward + trajectory_bonus

    # 汇总诊断信息
    # 复用循环中已计算的第一步结果，避免重复计算
    first_result = step_results[0] if step_results and step_results[0] is not None else None
    first_components = first_result.components if first_result else {}

    return _reward_info(
        score=total_reward,
        exact_success=trajectory_exact,
        action_type_match=step_type_match[0] if step_type_match else False,
        oracle_action_type=oracle_actions[0].get("action_type", "tool_call"),
        model_action_type=model_actions[0].action_type if model_actions else "unparseable",
        components=first_components,
        n_model_steps=n_model,
        n_oracle_steps=n_oracle,
        coverage_ratio=coverage_ratio,
    )


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: Optional[dict] = None,
    **kwargs,
) -> dict:
    """SchemaShift reward function（支持多步奖励）。

    Args:
        data_source: 数据源标识（"schemashift"）
        solution_str: 模型生成的完整 response 文本
        ground_truth: oracle 信息（JSON 序列化的 dict）
            {
                "oracle_actions": [...],  # 每步的 OracleAction（含 replay_observation）
                "episode_type": "call_only" | "call_then_final" | ...
            }
        extra_info: 额外信息
            {
                "perturbation_level": "none" | "light" | "medium" | "heavy",
                "name_map": {...},
                "enum_map": {...},
                "scenario_type": "...",
            }

    Returns:
        dict with "score" key (float) + scalar/string diagnostic keys.
        verl validation aggregates non-string diagnostics with np.mean(),
        so nested dict/list values must not be returned here.
    """
    extra_info = extra_info or {}

    # 统一 normalize（支持 JSON string、None、非 dict）
    from src.utils import normalize_extra_info, normalize_json_field
    extra_info = normalize_extra_info(extra_info)
    extra_info["name_map"] = normalize_json_field(extra_info.get("name_map"))
    extra_info["enum_map"] = normalize_json_field(extra_info.get("enum_map"))

    # 解析 ground_truth
    if isinstance(ground_truth, str):
        try:
            oracle_data = json.loads(ground_truth)
        except json.JSONDecodeError:
            return _reward_info(score=0.0, error="ground_truth not valid JSON")
    elif isinstance(ground_truth, dict):
        oracle_data = ground_truth
    else:
        return _reward_info(score=0.0, error=f"unexpected ground_truth type: {type(ground_truth)}")

    oracle_actions = oracle_data.get("oracle_actions", [])
    if not oracle_actions:
        return _reward_info(score=0.0, error="no oracle_actions in ground_truth")

    episode_type = oracle_data.get("episode_type", "call_only")

    # 构建 metadata
    metadata = SampleMetadata(
        name_map=extra_info.get("name_map", {}),
        enum_map=extra_info.get("enum_map", {}),
        perturbation_level=extra_info.get("perturbation_level", "none"),
        scenario_type=extra_info.get("scenario_type", episode_type),
    )

    # 判断是否为多步 episode
    is_multi_step = len(oracle_actions) > 1

    if not is_multi_step:
        # 单步 episode：使用原有逻辑
        oracle_action = _build_oracle_action(oracle_actions[0])
        return _compute_single_step_reward(solution_str, oracle_action, metadata)
    else:
        # 多步 episode：使用多步奖励计算
        return _compute_multi_step_reward(
            solution_str, oracle_actions, metadata, episode_type
        )
