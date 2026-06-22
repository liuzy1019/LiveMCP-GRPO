"""
BFCL Agent Loop（verl 集成）。

基于 verl 的 AgentLoopBase 实现 BFCL 多轮工具调用 agent loop。
与参考 repo (agentic-grpo-longhorizon) 使用的同一 verl fork 兼容。
"""

import ast
import asyncio
import json
import logging
import random
import re
from typing import Any, Optional
from uuid import uuid4

from src.envs.bfcl_env import ToolCallResult
from src.envs.api_mapper import FunctionNameMapper

try:
    from verl.experimental.agent_loop.agent_loop import (
        AgentLoopBase,
        AgentLoopOutput,
        register,
    )
    from verl.workers.rollout.replica import TokenOutput
    VERL_AVAILABLE = True
except ImportError:
    VERL_AVAILABLE = False
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

try:
    from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import (
        execute_multi_turn_func_call,
    )
    BFCL_EXECUTOR_AVAILABLE = True
except ImportError:
    BFCL_EXECUTOR_AVAILABLE = False

logger = logging.getLogger(__name__)
# rollout 阶段必须能看到 sample-level 进度，否则卡死无法定位
if logger.level == logging.NOTSET or logger.level > logging.INFO:
    logger.setLevel(logging.INFO)


def _format_bfcl_value(value: Any) -> str:
    """格式化参数值为 BFCL 原生调用字符串。"""
    if isinstance(value, str):
        return repr(str(value))
    elif isinstance(value, bool):
        return "True" if value else "False"
    elif value is None:
        return "None"
    else:
        return str(value)


# 解析器硬上限：超过这些值的输入直接放弃，避免 garbled LLM 输出拖垮 rollout
_PARSE_MAX_INPUT_LEN = 8192        # 整个 func_call_str 字符上限
_PARSE_MAX_ARGS = 64               # 段（参数）数量上限
_PARSE_MAX_KEY_LEN = 64            # 单个 key 长度上限（防 regex 病态回溯）
_PARSE_MAX_LITERAL_LEN = 4096      # 单个 value 字面量回填上限
_PARSE_NAME_MAX_SCAN = 256         # 函数名匹配只看前若干字符
_IDENT_RE = re.compile(r"^[a-zA-Z_]\w{0,63}$")


def _parse_bfcl_native_args(func_call_str: str) -> tuple[str, dict[str, Any]]:
    """解析 BFCL 原生格式的函数调用字符串，bounded linear parser。

    设计目标：永不卡死、无 O(N²) 路径、超长/畸形输入直接降级返回。
    任何分支失败都返回 ("", {}) 或 (name, {}), 由调用方走 fallback。
    """
    if not func_call_str or len(func_call_str) > _PARSE_MAX_INPUT_LEN:
        return "", {}

    # ---- Step 1: 函数名（仅扫描前 _PARSE_NAME_MAX_SCAN 字符）----
    head = func_call_str[:_PARSE_NAME_MAX_SCAN]
    name_match = re.match(r"([a-zA-Z_][\w.]{0,127})\s*\(", head)
    if not name_match:
        return "", {}
    name = name_match.group(1).rstrip(".")
    if not name:
        return "", {}

    # ---- Step 2: 单遍扫描定位匹配的 ')' ----
    args_start = func_call_str.find("(")
    if args_start < 0:
        return name, {}
    paren_depth = 0
    args_end = -1
    for j in range(args_start, len(func_call_str)):
        c = func_call_str[j]
        if c == "(":
            paren_depth += 1
        elif c == ")":
            paren_depth -= 1
            if paren_depth == 0:
                args_end = j
                break
    if args_end < 0:
        return name, {}
    args_part = func_call_str[args_start + 1:args_end]
    if not args_part or not args_part.strip():
        return name, {}

    # ---- Step 3: 单遍切段（top-level ',' 分隔），同时记录 top-level '=' 位置 ----
    # 用 i 严格递增保证不死循环；每个迭代必走 i += 1（除非显式 break）。
    segments: list[tuple[str, int]] = []  # (segment_text, top_level_eq_offset_or_-1)
    n = len(args_part)
    i = 0
    seg_start = 0
    eq_offset = -1
    quote = ""
    bracket = 0
    brace = 0
    paren = 0
    while i < n:
        if len(segments) >= _PARSE_MAX_ARGS:
            return name, {}
        c = args_part[i]
        if quote:
            # 字符串字面量内：遇到匹配引号关闭；'\\' 转义跳一字符
            if c == "\\" and i + 1 < n:
                i += 2
                continue
            if c == quote:
                quote = ""
            i += 1
            continue
        if c in ("'", '"'):
            quote = c
            i += 1
            continue
        if c == "[":
            bracket += 1
        elif c == "]" and bracket > 0:
            bracket -= 1
        elif c == "{":
            brace += 1
        elif c == "}" and brace > 0:
            brace -= 1
        elif c == "(":
            paren += 1
        elif c == ")" and paren > 0:
            paren -= 1
        elif c == "=" and bracket == 0 and brace == 0 and paren == 0 and eq_offset < 0:
            eq_offset = i - seg_start
        elif c == "," and bracket == 0 and brace == 0 and paren == 0:
            segments.append((args_part[seg_start:i], eq_offset))
            seg_start = i + 1
            eq_offset = -1
        i += 1
    # 收尾：最后一段
    tail = args_part[seg_start:n]
    if tail.strip():
        segments.append((tail, eq_offset))

    # ---- Step 4: 段内分 kwarg/positional ----
    args: dict[str, Any] = {}
    positional_idx = 0
    for seg_text, eq_off in segments:
        seg = seg_text.strip()
        if not seg:
            continue
        # eq_off 是相对 seg_text（含前导空白）的偏移；映射到 strip 后再判断
        if eq_off >= 0:
            raw_key = seg_text[:eq_off]
            raw_val = seg_text[eq_off + 1:]
            key_stripped = raw_key.strip()
            if (
                len(key_stripped) <= _PARSE_MAX_KEY_LEN
                and _IDENT_RE.match(key_stripped)
            ):
                args[key_stripped] = raw_val.strip()
                continue
        # 否则当成位置参数
        args[f"_pos_{positional_idx}"] = seg
        positional_idx += 1

    # ---- Step 5: 字面量回填 ----
    for k in list(args.keys()):
        v = args[k]
        if not isinstance(v, str):
            continue
        s = v.strip()
        if not s or len(s) > _PARSE_MAX_LITERAL_LEN:
            continue
        # 引号包裹的字符串：去掉外层引号即可
        if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
            try:
                args[k] = ast.literal_eval(s)
                continue
            except (ValueError, SyntaxError):
                args[k] = s[1:-1]
                continue
        if s[0] in "[{" or s in ("True", "False", "None") or _looks_like_number(s):
            try:
                args[k] = ast.literal_eval(s)
            except (ValueError, SyntaxError):
                pass
    return name, args


def _looks_like_number(s: str) -> bool:
    """判断字符串是否像数字字面量，用于 ast.literal_eval 前的快速过滤。"""
    if not s:
        return False
    try:
        float(s)
        return True
    except ValueError:
        return False


async def _retry_async(fn, max_retries=3, base_delay=1.0, max_delay=30.0):
    last_exc = None
    for attempt in range(max_retries):
        try:
            return await fn()
        except Exception as e:
            last_exc = e
            if attempt < max_retries - 1:
                delay = min(base_delay * (2 ** attempt), max_delay)
                jitter = random.uniform(0, delay * 0.1)
                logger.warning(f"重试 {attempt + 1}/{max_retries}: {e} ({delay:.1f}s)")
                await asyncio.sleep(delay + jitter)
    raise last_exc


@register("bfcl_agent")
class BFCLAgentLoop(AgentLoopBase):
    """BFCL 多轮工具调用 Agent Loop。"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.max_turns = int(kwargs.get("max_turns", 10))
        self.max_obs_length = int(kwargs.get("max_obs_length", 1024))
        # response_length 上限：与 verl 内部 _postprocess pad/cat 对齐，防止多轮累积溢出
        self.response_length = int(self.config.actor_rollout_ref.rollout.response_length)
        self.fn_mapper: Optional[FunctionNameMapper] = None
        # system prompt 前缀长度，用于 stripping（对齐 verl ToolAgentLoop 做法）
        self._system_prompt_len: int = 0

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        """运行多轮 agent 交互。

        verl 的 AgentLoopOutput 要求：
        - prompt_ids: 原始 prompt 的 token id（不含模型生成内容）
        - response_ids: 模型生成的 response token id（含 LLM 生成 + tool observation）
        - response_mask: 1=LLM 生成(参与loss)，0=tool observation(不参与loss)
        """
        raw_prompt = kwargs.get("raw_prompt", [])
        task_perturbed = kwargs.get("task_perturbed")
        _ic = kwargs.get("initial_config", kwargs.get("initial_config_json", "{}"))
        initial_config = json.loads(_ic) if isinstance(_ic, str) else (_ic or {})
        # 跨工具调用的环境状态（BFCL executor 返回的 instances）
        # 每次 _execute_tool 更新，下次调用时传入以保持状态连续
        running_instances = None
        _iv = kwargs.get("involved_classes", kwargs.get("involved_classes_json", "[]"))
        involved_classes = json.loads(_iv) if isinstance(_iv, str) else (_iv or [])
        llm_sampling_params = kwargs.get("llm_sampling_params", sampling_params)
        user_turns_json = kwargs.get("user_turns_json", "[]")
        ground_truth_json = kwargs.get("ground_truth_json", "[]")

        if self.tokenizer is None:
            self.tokenizer = kwargs.get("tokenizer")
        if self.tokenizer is None:
            raise RuntimeError(
                "BFCLAgentLoop.tokenizer is None — verl 未注入 tokenizer，"
                "请检查 agent loop 初始化配置"
            )

        # 计算 system prompt 前缀长度（对齐 verl ToolAgentLoop）
        # apply_chat_template 会将 system 格式 token 插入每次调用结果
        # 预先计算前缀长度以在后续增量追加中 stripping
        if self._system_prompt_len == 0:
            sp_tokens = self.tokenizer.apply_chat_template(
                [{}], add_generation_prompt=False, tokenize=True
            )
            self._system_prompt_len = len(sp_tokens)

        # 初始化 FunctionNameMapper
        self.fn_mapper = None
        if task_perturbed:
            self.fn_mapper = FunctionNameMapper(
                name_map=task_perturbed.name_map,
                enum_map=getattr(task_perturbed, 'enum_map', None) or {},
            )
        else:
            name_map_raw = kwargs.get("name_map_json", "{}")
            enum_map_raw = kwargs.get("enum_map_json", "{}")
            try:
                name_map = json.loads(name_map_raw) if isinstance(name_map_raw, str) else name_map_raw
                enum_map = json.loads(enum_map_raw) if isinstance(enum_map_raw, str) else enum_map_raw
            except (json.JSONDecodeError, TypeError):
                name_map, enum_map = {}, {}
            if name_map:
                self.fn_mapper = FunctionNameMapper(name_map=name_map, enum_map=enum_map)

        # 解析 prompt（verl 传 raw_prompt 可以是 list of messages 或 JSON string）
        if isinstance(raw_prompt, str):
            try:
                messages = json.loads(raw_prompt)
            except json.JSONDecodeError:
                messages = [{"role": "user", "content": raw_prompt}]
        else:
            messages = list(raw_prompt)

        # 后续用户轮次
        try:
            remaining_turns = json.loads(user_turns_json) if isinstance(user_turns_json, str) else (user_turns_json or [])
        except (json.JSONDecodeError, TypeError):
            remaining_turns = []
        num_user_turns_processed = 0

        try:
            gt_data = json.loads(ground_truth_json) if isinstance(ground_truth_json, str) else (ground_truth_json or [])
        except (json.JSONDecodeError, TypeError):
            gt_data = []

        # 优先使用样本级 max_turns，否则用 init 的默认值
        sample_max_turns = int(kwargs.get("max_turns", self.max_turns))
        request_id = uuid4().hex

        # 编码初始 prompt（不含 LLM 生成内容）
        prompt_ids = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True
            ),
        )

        all_response_ids: list[int] = []
        all_response_mask: list[int] = []
        self.func_calls = []
        self.tool_results = []
        # 每轮的工具调用列表，用于轮次级 reward 校验
        turn_func_calls: list[list[dict]] = []

        # BFCL execute_multi_turn_func_call 需要全局唯一的 test_entry_id
        # 作为 instance 缓存 key，避免不同 episode 间 状态污染。
        # request_id 已是 uuid4，直接复用。
        bfcl_test_entry_id = f"sshift_{request_id}"
        rid_short = request_id[:8]
        logger.info(f"[rollout {rid_short}] start max_turns={sample_max_turns} prompt_len={len(prompt_ids)}")

        for turn_idx in range(sample_max_turns):
            # LLM 生成：retry 耗尽即终止该 episode；不加调用层 timeout，由 vLLM/外层控制
            try:
                output = await _retry_async(
                    lambda: self.server_manager.generate(
                        request_id=request_id,
                        prompt_ids=prompt_ids + all_response_ids,
                        sampling_params=llm_sampling_params,
                        image_data=None,
                    )
                )
            except Exception as e:
                logger.error(f"[rollout {rid_short}] turn={turn_idx} LLM 生成失败（重试耗尽）: {e}")
                break
            response_ids = output.token_ids.tolist() if hasattr(output.token_ids, "tolist") else list(output.token_ids)
            response_text = self._decode_response(response_ids)

            # LLM 生成的 token → mask=1（参与 loss）
            all_response_ids.extend(response_ids)
            all_response_mask.extend([1] * len(response_ids))

            # 累积长度兜底：超过 response_length 立刻停，避免后续 _postprocess cat 失败
            if len(all_response_ids) >= self.response_length:
                break

            # 解析所有工具调用（支持 BFCL parallel / parallel_multiple 多调用）
            func_call_strs = self._parse_tool_calls(response_text)

            if not func_call_strs:
                # 本轮没有 tool call：要么是该轮本不该调（miss_func/miss_param 等），
                # 要么 agent 提前停。无论哪种，只要还有 user turn 要处理，就继续，
                # 并且必须把空列表 append 到 turn_func_calls，保证后续 reward 按轮对齐。
                if remaining_turns and num_user_turns_processed < len(remaining_turns):
                    next_text = self._extract_next_user_message(
                        remaining_turns, num_user_turns_processed
                    )
                    if next_text:
                        user_tokens = await self._encode_message_tokens(
                            [{"role": "user", "content": next_text}]
                        )
                        all_response_ids.extend(user_tokens)
                        all_response_mask.extend([0] * len(user_tokens))
                        num_user_turns_processed += 1
                        turn_func_calls.append([])
                        continue
                # 没有后续 user turn：把当前空轮也记一笔，避免末轮空 GT 被漏判
                turn_func_calls.append([])
                break

            # 依次执行每个工具调用（传递 instances 保持环境状态连续）
            turn_calls = []
            logger.info(
                f"[rollout {rid_short}] turn={turn_idx} parsed {len(func_call_strs)} tool call(s)"
            )
            for call_idx, func_call_str in enumerate(func_call_strs):
                result, running_instances = await self._execute_tool(
                    func_call_str=func_call_str,
                    fn_mapper=self.fn_mapper,
                    initial_config=initial_config,
                    involved_classes=involved_classes,
                    instances=running_instances,
                    test_entry_id=bfcl_test_entry_id,
                    rid_short=rid_short,
                    turn_idx=turn_idx,
                    call_idx=call_idx,
                )
                self.tool_results.append(result)
                # 记录本轮调用（用于轮次级 reward）
                if self.func_calls:
                    turn_calls.append(self.func_calls[-1])

                # 工具 observation token → mask=0（不参与 loss）
                # 每个调用的结果都要追加到 response 中
                obs_raw = result.result or ""
                if len(obs_raw) > self.max_obs_length:
                    obs_raw = obs_raw[:self.max_obs_length]
                tool_tokens = await self._encode_message_tokens(
                    [{"role": "tool", "content": obs_raw}]
                )
                all_response_ids.extend(tool_tokens)
                all_response_mask.extend([0] * len(tool_tokens))

            # 无论是否产出 turn_calls 都 append，保持轮次索引与对话轮次严格一致
            turn_func_calls.append(turn_calls)

            # 多轮用户指令注入（工具执行后）
            if remaining_turns and num_user_turns_processed < len(remaining_turns):
                next_text = self._extract_next_user_message(
                    remaining_turns, num_user_turns_processed
                )
                if next_text:
                    user_tokens = await self._encode_message_tokens(
                        [{"role": "user", "content": next_text}]
                    )
                    all_response_ids.extend(user_tokens)
                    all_response_mask.extend([0] * len(user_tokens))
                    messages.append({"role": "user", "content": next_text})
                    num_user_turns_processed += 1

        # 计算 response-based reward（按轮次校验）
        # 使用 AST 宽松匹配（与 offline eval 一致），避免 train-eval mismatch。
        from src.eval.matching import _args_match

        # 默认 0.0，避免 verl 在 reward_score=None 时回落到 reward_manager 的
        # naive reward_loop（要求样本带 reward_model 字段，本项目数据集不含）。
        reward_score = 0.0
        if gt_data and turn_func_calls:
            # agent loop 在没有后续 user turn 时会多 append 一个空 turn 再 break，
            # 因此允许 GT 之后的尾部空 turn；但 GT 之后的任何非空 turn 都判错。
            if len(turn_func_calls) < len(gt_data):
                reward_score = 0.0
            elif any(extra for extra in turn_func_calls[len(gt_data):]):
                reward_score = 0.0
            else:
                n_turns = len(gt_data)
                all_turns_matched = True
                for turn_idx in range(n_turns):
                    turn_gt = gt_data[turn_idx]
                    turn_agent = turn_func_calls[turn_idx]

                    # 解析 GT（BFCL 原生格式字符串）
                    gt_dicts = []
                    for gt_item in (turn_gt if isinstance(turn_gt, list) else [turn_gt]):
                        if isinstance(gt_item, str):
                            gt_dicts.append(_parse_bfcl_native_args(gt_item))

                    if not gt_dicts:
                        # 空 GT turn：模型在该轮不应调用任何工具
                        if len(turn_agent) != 0:
                            all_turns_matched = False
                            break
                        continue

                    agent_dicts = [
                        (
                            fc.get("name", ""),
                            fc.get("arguments", {}),
                        )
                        for fc in turn_agent
                    ]

                    # 按轮次精确检查：数量、顺序、内容必须匹配
                    if len(agent_dicts) != len(gt_dicts):
                        all_turns_matched = False
                        break
                    for (a_name, a_args), (gt_name, gt_args) in zip(agent_dicts, gt_dicts):
                        if a_name != gt_name or not _args_match(a_args, gt_args):
                            all_turns_matched = False
                            break
                    if not all_turns_matched:
                        break

                reward_score = 1.0 if all_turns_matched else 0.0

        # 使用 turn_idx（循环迭代次数）而非 len(tool_results)
        # tool_results 在并行调用时每条调用都 append，不能用于统计轮次
        # 边界：sample_max_turns == 0 或循环立即 break 时，turn_idx 未定义，回落到 0
        _num_turns = min(locals().get("turn_idx", -1) + 1, sample_max_turns)

        # 硬截断到 response_length 上限（对齐 verl tool_agent_loop / single_turn_agent_loop）
        all_response_ids = all_response_ids[: self.response_length]
        all_response_mask = all_response_mask[: self.response_length]

        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=all_response_ids,
            response_mask=all_response_mask,
            reward_score=reward_score,
            num_turns=_num_turns,
            metrics={},
            extra_fields={
                "func_calls": self.func_calls,
                "tool_results": [{"success": r.success, "error": r.error} for r in self.tool_results],
                "num_turns": _num_turns,
                "perturbation_level": kwargs.get("perturbation_level") or "none",
                "group_id": kwargs.get("group_id") or kwargs.get("task_id") or "",
            },
        )

    # ── 解码 / 辅助 ──

    def _decode_response(self, token_ids: list[int]) -> str:
        if self.tokenizer is None:
            return ""
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    async def _encode_message_tokens(self, add_messages: list[dict]) -> list[int]:
        """用 apply_chat_template 编码新消息，stripping system prompt 前缀。

        对齐 verl ToolAgentLoop 做法：apply_chat_template 产生的 token 包含
        system 格式化前缀，stripping 后只保留增量内容。
        """
        response_ids = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.apply_chat_template(
                add_messages, add_generation_prompt=True, tokenize=True
            ),
        )
        return response_ids[self._system_prompt_len:]

    @staticmethod
    def _extract_next_user_message(
        remaining_turns: list, idx: int
    ) -> str:
        """从 remaining_turns[idx] 提取下一条用户消息文本。

        支持两种格式：
        - turn 是 dict: {"role": "user", "content": "..."}
        - turn 是 list of dicts: 取所有 role="user" 的 content，用 "\\n" 拼接。
          注意：不再用赋值覆盖，防止多条用户消息只保留最后一条。
        """
        next_turn = remaining_turns[idx]
        if isinstance(next_turn, list):
            parts = []
            for msg in next_turn:
                if isinstance(msg, dict) and msg.get("role") == "user":
                    content = msg.get("content", "")
                    if content:
                        parts.append(content)
            return "\n".join(parts)
        elif isinstance(next_turn, dict) and next_turn.get("role") == "user":
            return next_turn.get("content", "")
        return ""

    # ── 解析工具调用 ──

    @staticmethod
    def _parse_tool_calls(text: str) -> list[str]:
        """解析响应文本中的所有工具调用。

        支持 BFCL 的 parallel / parallel_multiple 类别：
        同轮输出多个 <tool_call> 标签时，全部解析并返回。

        Returns:
            工具调用字符串列表。空列表 = 无工具调用（视为结束）。
        """
        if not text:
            return []
        calls = []

        # 1. <tool_call>...</tool_call> 模式（支持并行多调用）
        for m in re.finditer(r'<tool_call>(.*?)</tool_call>', text, re.DOTALL):
            calls.append(m.group(1).strip())

        if calls:
            return calls

        # 2. 裸 JSON（fallback，单匹配即可，malformed 输出不期望并行）
        json_call = BFCLAgentLoop._extract_first_tool_json(text)
        if json_call:
            return [json_call]

        # 3. BFCL 原生格式（fallback，单匹配，支持点号函数名）
        m = re.search(r'([a-zA-Z_][\w.]*\w)\s*\(', text, re.DOTALL)
        if m:
            lparen = m.end() - 1
            depth = 1
            i = lparen + 1
            while i < len(text) and depth > 0:
                if text[i] == '(':
                    depth += 1
                elif text[i] == ')':
                    depth -= 1
                i += 1
            if depth == 0:
                return [text[m.start():i].strip()]

        return []

    @staticmethod
    def _extract_first_tool_json(text: str) -> Optional[str]:
        """Extract the first complete JSON tool call object from free text."""
        decoder = json.JSONDecoder()
        start = text.find("{")
        while start != -1:
            try:
                obj, end = decoder.raw_decode(text[start:])
            except json.JSONDecodeError:
                start = text.find("{", start + 1)
                continue
            if (
                isinstance(obj, dict)
                and isinstance(obj.get("name"), str)
                and isinstance(obj.get("arguments"), dict)
            ):
                return text[start:start + end].strip()
            start = text.find("{", start + 1)
        return None

    # ── 执行工具 ──

    async def _execute_tool(
        self,
        func_call_str,
        fn_mapper,
        initial_config,
        involved_classes,
        instances=None,
        test_entry_id="",
        rid_short="",
        turn_idx=-1,
        call_idx=-1,
    ):
        prefix = f"[rollout {rid_short}] turn={turn_idx} call={call_idx}"
        try:
            original_name, arguments = self._normalize_func_call(func_call_str, fn_mapper)
            self.func_calls.append({"name": original_name, "arguments": arguments})
            logger.info(f"{prefix} tool={original_name} args_keys={list(arguments.keys())}")
            if BFCL_EXECUTOR_AVAILABLE:
                bfcl_call = self._format_for_bfcl(original_name, arguments)
                execution_results, new_instances = await asyncio.to_thread(
                    execute_multi_turn_func_call,
                    func_call_list=[bfcl_call],
                    initial_config=instances if instances is not None else (initial_config or {}),
                    involved_classes=involved_classes or [],
                    model_name="schemashift_agent",
                    test_entry_id=test_entry_id,
                )
                result_str = json.dumps(execution_results[0] if execution_results else {"status": "done"})
                return ToolCallResult(success=True, result=result_str), new_instances
            else:
                return ToolCallResult(
                    success=True,
                    result=json.dumps({"status": "ok", "function": original_name, "data": "simulated"}),
                ), instances
        except json.JSONDecodeError as e:
            logger.error(f"{prefix} 工具调用 JSON 解析失败: {e}")
            return ToolCallResult(success=False, result="", error=f"JSON 解析错误: {e}"), instances
        except Exception as e:
            logger.error(f"{prefix} 工具执行异常: {e}")
            return ToolCallResult(success=False, result="", error=str(e)), instances

    def _normalize_func_call(self, func_call_str, fn_mapper):
        from src.eval.matching import map_enum_values

        try:
            call_data = json.loads(func_call_str)
            name = call_data.get("name", "")
            args = call_data.get("arguments", {})
            is_json = True
        except json.JSONDecodeError:
            name, args = _parse_bfcl_native_args(func_call_str)
            is_json = False
        if not name:
            raise ValueError(f"无法解析函数名: {func_call_str[:80]}")
        if fn_mapper:
            original_name = fn_mapper.resolve(name)
            if fn_mapper.enum_map:
                args = map_enum_values(original_name, args, fn_mapper.enum_map)
        else:
            original_name = name
        return original_name, args

    @staticmethod
    def _format_for_bfcl(func_name, arguments):
        positional_args = []
        keyword_args = []
        for key, value in arguments.items():
            m = re.fullmatch(r"_pos_(\d+)", str(key))
            if m:
                positional_args.append((int(m.group(1)), _format_bfcl_value(value)))
            else:
                keyword_args.append(f"{key}={_format_bfcl_value(value)}")
        args_str = ", ".join(
            [value for _, value in sorted(positional_args)] + keyword_args
        )
        return f"{func_name}({args_str})"
