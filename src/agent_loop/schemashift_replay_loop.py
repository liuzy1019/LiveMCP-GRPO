"""
SchemaShift 交互式静态 Replay Agent Loop。

核心思路（对齐 COVERT mock server）：
  - 模型生成 tool_call → 匹配 oracle → 返回预存的 replay_observation
  - 模型看到 tool output 后继续生成下一步（tool_call 或 final_answer）
  - 不需要真实 MCP server，纯字符串拼接

与 BFCLAgentLoop 的区别：
  - BFCLAgentLoop 调用 BFCL executor 真实执行工具
  - 本 loop 使用预存的 replay_observation，无需外部依赖

verl 集成方式：
  - 通过 configs/agent_loop.yaml 注册为 "schemashift_replay"
  - 数据中 extra_info.tools_kwargs.replay_data 传入 oracle_actions
"""

import json
import re
from typing import Any, Optional
from uuid import uuid4

from src.eval.matching import map_enum_values, strict_args_match

try:
    from verl.experimental.agent_loop.agent_loop import (
        AgentLoopBase,
        AgentLoopOutput,
        register,
    )
except ImportError:
    from abc import ABC, abstractmethod
    from dataclasses import dataclass, field

    class AgentLoopBase(ABC):
        @abstractmethod
        async def run(self, sampling_params, **kwargs) -> Any:
            ...

    @dataclass
    class AgentLoopOutput:
        prompt_ids: list[int] = field(default_factory=list)
        response_ids: list[int] = field(default_factory=list)
        response_mask: list[int] = field(default_factory=list)
        response_logprobs: Optional[list[float]] = None
        reward_score: Optional[float] = None
        num_turns: int = 0
        metrics: dict = field(default_factory=dict)
        extra_fields: dict = field(default_factory=dict)

    def register(name: str):
        def decorator(cls):
            return cls
        return decorator


from loguru import logger

logger = logger.opt(colors=True)


# ── 工具调用解析 ──

_TOOL_CALL_PATTERN = re.compile(
    r"<tool_call>(.*?)</tool_call>", re.DOTALL
)
_FINAL_ANSWER_PATTERN = re.compile(
    r"<final_answer>(.*?)</final_answer>", re.DOTALL
)
_REPORT_ERROR_PATTERN = re.compile(
    r"<report_error>(.*?)</report_error>", re.DOTALL
)
_ASK_CLARIFICATION_PATTERN = re.compile(
    r"<ask_clarification>(.*?)</ask_clarification>", re.DOTALL
)


def _is_terminal_response(text: str) -> bool:
    """判断模型输出是否为终止响应（final_answer / report_error / ask_clarification）。"""
    return bool(
        _FINAL_ANSWER_PATTERN.search(text)
        or _REPORT_ERROR_PATTERN.search(text)
        or _ASK_CLARIFICATION_PATTERN.search(text)
    )


def _parse_tool_calls_json(text: str) -> list[dict]:
    """从 <tool_call> 内容中解析工具调用（支持单个 dict 或 list）。"""
    text = text.strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "name" in obj:
            return [{"name": obj["name"], "arguments": obj.get("arguments", {})}]
        if isinstance(obj, list):
            calls = []
            for item in obj:
                if isinstance(item, dict) and "name" in item:
                    calls.append({"name": item["name"], "arguments": item.get("arguments", {})})
            return calls
    except json.JSONDecodeError:
        pass
    return []


def _call_exact_match(
    model_call: dict,
    oracle_call: dict,
    name_map: dict[str, str] | None = None,
    enum_map: dict | None = None,
) -> bool:
    """单个 tool_call 的精确匹配：name + arguments key+value。

    支持 SchemaShift 扰动场景：
    - name_map: 将 perturbed tool name 映射回 canonical name
    - enum_map: 将 perturbed enum value 映射回 canonical value

    匹配语义与 ComponentReward._check_exact_match_tool_call 一致。
    """
    name_map = name_map or {}
    enum_map = enum_map or {}

    # 通过 name_map 规范化工具名
    model_name = name_map.get(model_call.get("name", ""), model_call.get("name", ""))
    oracle_name = name_map.get(oracle_call.get("name", ""), oracle_call.get("name", ""))
    if model_name != oracle_name:
        return False

    model_args = model_call.get("arguments", {})
    oracle_args = oracle_call.get("arguments", {})
    if not isinstance(model_args, dict):
        return False

    # 通过 enum_map 规范化参数值（使用 canonical name 作为 func_name）
    if enum_map:
        model_args = map_enum_values(oracle_name, model_args, enum_map)

    return strict_args_match(model_args, oracle_args)


def _match_tool_call(
    model_calls: list[dict],
    oracle_action: dict,
    name_map: dict[str, str] | None = None,
    enum_map: dict | None = None,
) -> bool:
    """判断模型的 tool_call(s) 是否精确匹配 oracle action。

    匹配策略（对齐 SchemaShift 核心目标 + ComponentReward 语义）：
    - 数量必须一致
    - 每个 call 要求 name + argument keys + argument values 全匹配
    - 支持 match_mode="set"（无序）和 match_mode="ordered"（有序）
    - argument value 使用 strict_args_match（严格匹配：type_compatible + 值匹配）
    - name_map: perturbed tool name → canonical name
    - enum_map: perturbed enum value → canonical value

    这确保只有参数值正确时才释放真实 replay_observation。
    """
    oracle_calls = oracle_action.get("tool_calls", [])
    if not oracle_calls:
        return False

    # 数量必须一致
    if len(model_calls) != len(oracle_calls):
        return False

    match_mode = oracle_action.get("match_mode", "set")

    if match_mode == "ordered" or len(oracle_calls) <= 1:
        # 顺序匹配
        for model_call, oracle_call in zip(model_calls, oracle_calls):
            if not _call_exact_match(model_call, oracle_call, name_map, enum_map):
                return False
        return True

    # set matching（无序）：贪心匹配
    remaining_model = list(model_calls)
    for oracle_call in oracle_calls:
        found = False
        for i, model_call in enumerate(remaining_model):
            if _call_exact_match(model_call, oracle_call, name_map, enum_map):
                remaining_model.pop(i)
                found = True
                break
        if not found:
            return False
    return True


@register("schemashift_replay")
class SchemaShiftReplayLoop(AgentLoopBase):
    """SchemaShift 交互式静态 Replay Agent Loop。

    rollout 流程：
    1. 模型生成 response（可能包含 <tool_call>）
    2. 如果是 tool_call：匹配 oracle → 返回预存 replay_observation → 继续生成
    3. 如果是 terminal（final_answer/report_error/ask_clarification）：结束
    4. 重复直到 max_turns 或 response_length 耗尽
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # P1-4: 从 config 读取 max_turns，而非写死
        rollout_cfg = self.config.actor_rollout_ref.rollout
        multi_turn_cfg = rollout_cfg.get("multi_turn", {})
        self.max_turns = int(
            multi_turn_cfg.get("max_assistant_turns", None)
            or rollout_cfg.get("max_turns", 5)
            or 5
        )
        self.max_obs_length = 1024  # observation 最大长度
        self.max_consecutive_errors = 2  # P1-1: 同一 oracle step 最大连续错误次数
        self.response_length = int(rollout_cfg.response_length)
        # 读取 chat template kwargs（用于关闭 thinking mode 等）
        self.apply_chat_template_kwargs = self.config.data.get("apply_chat_template_kwargs", {})

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        """运行交互式静态 replay rollout。"""
        raw_prompt = kwargs.get("raw_prompt", [])
        extra_info = kwargs.get("extra_info", {})

        # ── Step 0: 统一 normalize extra_info（必须在读取任何字段之前） ──
        from src.utils import normalize_extra_info, normalize_json_field
        extra_info = normalize_extra_info(extra_info)

        # ── Step 1: 从 tools_kwargs 或 extra_info 中获取 replay 数据 ──
        tools_kwargs = kwargs.get("tools_kwargs", None)
        if tools_kwargs is None:
            tools_kwargs = extra_info.get("tools_kwargs", {})
        tools_kwargs = normalize_json_field(tools_kwargs)
        if not isinstance(tools_kwargs, dict):
            tools_kwargs = {}

        replay_tool_kwargs = tools_kwargs.get("replay_data", {})
        replay_tool_kwargs = normalize_json_field(replay_tool_kwargs)
        replay_data = replay_tool_kwargs.get("create_kwargs", replay_tool_kwargs)

        # fallback: 直接从 extra_info 中获取
        if not replay_data:
            rd = extra_info.get("replay_data", {})
            replay_data = normalize_json_field(rd)
            if not isinstance(replay_data, dict):
                replay_data = {}

        oracle_actions = replay_data.get("oracle_actions", [])
        episode_type = replay_data.get("episode_type", "call_only")

        # ── Step 2: 提取 name_map / enum_map 用于 perturbed schema 匹配 ──
        name_map: dict[str, str] = normalize_json_field(extra_info.get("name_map"))
        enum_map: dict = normalize_json_field(extra_info.get("enum_map"))

        if self.tokenizer is None:
            self.tokenizer = kwargs.get("tokenizer")
        if self.tokenizer is None:
            raise RuntimeError("SchemaShiftReplayLoop.tokenizer is None")

        # 解析 prompt
        if isinstance(raw_prompt, str):
            try:
                messages = json.loads(raw_prompt)
            except json.JSONDecodeError:
                messages = [{"role": "user", "content": raw_prompt}]
        else:
            messages = list(raw_prompt)

        # 编码初始 prompt
        prompt_ids = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
                **self.apply_chat_template_kwargs,
            ),
        )

        request_id = uuid4().hex
        rid_short = request_id[:8]

        all_response_ids: list[int] = []
        all_response_mask: list[int] = []
        oracle_step_idx = 0  # 当前匹配到的 oracle step
        consecutive_errors = 0  # P1-1: 同一 oracle step 的连续错误计数
        n_model_tool_calls = 0
        n_correct_calls = 0

        logger.info(
            f"[replay {rid_short}] start | episode_type={episode_type} "
            f"| oracle_steps={len(oracle_actions)} | prompt_len={len(prompt_ids)}"
        )

        for turn_idx in range(self.max_turns):
            # 1. 模型生成
            try:
                output = await self.server_manager.generate(
                    request_id=request_id,
                    prompt_ids=prompt_ids + all_response_ids,
                    sampling_params=sampling_params,
                    image_data=None,
                )
            except Exception as e:
                logger.error(f"[replay {rid_short}] turn={turn_idx} 生成失败: {e}")
                break

            response_ids = (
                output.token_ids.tolist()
                if hasattr(output.token_ids, "tolist")
                else list(output.token_ids)
            )
            response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)

            # LLM 生成的 token → mask=1
            all_response_ids.extend(response_ids)
            all_response_mask.extend([1] * len(response_ids))

            # 长度兜底
            if len(all_response_ids) >= self.response_length:
                logger.info(f"[replay {rid_short}] turn={turn_idx} response_length 耗尽")
                break

            # 2. 解析模型输出
            tool_call_matches = list(_TOOL_CALL_PATTERN.finditer(response_text))

            if not tool_call_matches:
                # 没有 tool_call → 终止（可能是 final_answer 或无标签输出）
                logger.info(
                    f"[replay {rid_short}] turn={turn_idx} 无 tool_call，终止 "
                    f"(is_terminal={_is_terminal_response(response_text)})"
                )
                break

            # P0-3: 同一 turn 同时输出 tool_call 和 terminal tag → 非法
            # 交互式 replay 的核心目的是模型在看到 observation 后才继续决策
            if _is_terminal_response(response_text):
                # 同一 turn 既有 tool_call 又有 terminal tag → 不释放 observation，终止
                logger.warning(
                    f"[replay {rid_short}] turn={turn_idx} 同一 turn 同时输出 "
                    f"tool_call 和 terminal tag，视为非法，终止 episode"
                )
                break

            # 3. 处理 tool_call → 匹配 oracle → 返回 replay_observation
            # P1-2: 支持多个 tool_call（parallel calls）
            # 收集本轮所有 tool_call
            all_parsed_calls: list[dict] = []
            for tc_match in tool_call_matches:
                tc_content = tc_match.group(1)
                parsed_list = _parse_tool_calls_json(tc_content)
                all_parsed_calls.extend(parsed_list)

            n_model_tool_calls += 1

            if not all_parsed_calls:
                # JSON 解析失败 → 返回错误 observation
                observation = "Error: Invalid tool call format. Please provide valid JSON."
                consecutive_errors += 1
                logger.warning(
                    f"[replay {rid_short}] turn={turn_idx} tool_call JSON 解析失败 "
                    f"errors={consecutive_errors}/{self.max_consecutive_errors}"
                )
                if consecutive_errors >= self.max_consecutive_errors:
                    logger.info(f"[replay {rid_short}] turn={turn_idx} 连续错误超限，终止 episode")
                    break
            elif oracle_step_idx < len(oracle_actions):
                current_oracle = oracle_actions[oracle_step_idx]
                oracle_type = current_oracle.get("action_type", "tool_call")

                if oracle_type != "tool_call":
                    # oracle 期望的不是 tool_call（比如 final_answer）
                    observation = (
                        "Error: No tool call expected at this step. "
                        "Please provide your final answer directly."
                    )
                    consecutive_errors += 1
                    logger.info(
                        f"[replay {rid_short}] turn={turn_idx} oracle 期望 {oracle_type}，"
                        f"模型给了 tool_call "
                        f"errors={consecutive_errors}/{self.max_consecutive_errors}"
                    )
                    if consecutive_errors >= self.max_consecutive_errors:
                        logger.info(f"[replay {rid_short}] turn={turn_idx} 连续错误超限，终止 episode")
                        break
                elif _match_tool_call(all_parsed_calls, current_oracle, name_map, enum_map):
                    # 匹配成功 → 返回预存 observation
                    observation = current_oracle.get("replay_observation", "")
                    if not observation:
                        observation = json.dumps({"status": "success"})
                    n_correct_calls += 1
                    oracle_step_idx += 1
                    consecutive_errors = 0  # 重置错误计数
                    logger.info(
                        f"[replay {rid_short}] turn={turn_idx} 匹配成功 "
                        f"tool={all_parsed_calls[0]['name']} step={oracle_step_idx}/{len(oracle_actions)}"
                    )
                else:
                    # 不匹配 → 返回错误，不释放真实 observation
                    expected_name = current_oracle.get("tool_calls", [{}])[0].get("name", "unknown")
                    model_name = all_parsed_calls[0].get("name", "") if all_parsed_calls else ""
                    observation = (
                        f"Error: Tool call failed. Expected tool '{expected_name}' "
                        f"with correct arguments. Please check and try again."
                    )
                    consecutive_errors += 1
                    logger.info(
                        f"[replay {rid_short}] turn={turn_idx} 不匹配 "
                        f"model={model_name} expected={expected_name} "
                        f"errors={consecutive_errors}/{self.max_consecutive_errors}"
                    )
                    # P2 修复：连续错误超限 → 终止 episode（而非推进 oracle）
                    # 避免跳过未完成的 oracle step 导致 coverage 语义混乱
                    if consecutive_errors >= self.max_consecutive_errors:
                        logger.info(
                            f"[replay {rid_short}] turn={turn_idx} 连续错误超限，终止 episode"
                        )
                        break
            else:
                # oracle 步骤已耗尽
                observation = "Error: No more actions expected. Please provide your final answer."
                consecutive_errors += 1
                logger.info(
                    f"[replay {rid_short}] turn={turn_idx} oracle 步骤已耗尽 "
                    f"errors={consecutive_errors}/{self.max_consecutive_errors}"
                )
                if consecutive_errors >= self.max_consecutive_errors:
                    logger.info(f"[replay {rid_short}] turn={turn_idx} 连续错误超限，终止 episode")
                    break

            # 4. 将 observation 编码为 tool role message，拼接到 response
            if len(observation) > self.max_obs_length:
                observation = observation[: self.max_obs_length] + "...(truncated)"

            tool_msg = [{"role": "tool", "content": observation}]
            tool_tokens = await self._encode_message_tokens(tool_msg)

            # 长度检查
            if len(all_response_ids) + len(tool_tokens) >= self.response_length:
                logger.info(f"[replay {rid_short}] turn={turn_idx} 加入 obs 后超长，终止")
                break

            # tool observation → mask=0（不参与 loss）
            all_response_ids.extend(tool_tokens)
            all_response_mask.extend([0] * len(tool_tokens))

        # 截断到 response_length
        all_response_ids = all_response_ids[: self.response_length]
        all_response_mask = all_response_mask[: self.response_length]

        # 计算 reward（不在 agent loop 中计算，交给 verl 的 reward function）
        # reward_score = None 表示由外部 reward function 计算
        logger.info(
            f"[replay {rid_short}] done | turns={turn_idx + 1} "
            f"| tool_calls={n_model_tool_calls} correct={n_correct_calls} "
            f"| oracle_covered={oracle_step_idx}/{len(oracle_actions)} "
            f"| response_len={len(all_response_ids)}"
        )

        # oracle_skipped: 因连续错误被跳过的步数（P2: 当前实现不跳过，终止 episode）
        oracle_skipped = 0  # 保留字段，当前逻辑不会产生 skip

        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=all_response_ids,
            response_mask=all_response_mask,
            reward_score=None,  # 由外部 reward function 计算
            num_turns=turn_idx + 1,
            metrics={},
            extra_fields={
                "n_model_tool_calls": n_model_tool_calls,
                "n_correct_calls": n_correct_calls,
                "oracle_covered": oracle_step_idx,
                "oracle_skipped": oracle_skipped,
                "oracle_total": len(oracle_actions),
                "episode_type": episode_type,
            },
        )

    async def _encode_message_tokens(self, add_messages: list[dict]) -> list[int]:
        """编码新消息（tool observation），追加到已有 conversation 末尾。

        使用 apply_chat_template 编码 standalone 消息。
        Qwen3 的 chat template 对纯 tool 消息不产生 system 前缀，
        因此不做截断，直接返回完整 token 序列。

        产出的 token 序列形如：
          <|im_start|>tool\n{observation}<|im_end|>\n<|im_start|>assistant\n
        其中末尾的 assistant prompt 告诉模型从下一 token 开始生成。
        """
        response_ids = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.apply_chat_template(
                add_messages, add_generation_prompt=True, tokenize=True,
                **self.apply_chat_template_kwargs,
            ),
        )
        return list(response_ids)
