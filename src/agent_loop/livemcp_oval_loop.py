"""
LiveMCP Oval Agent Loop — live MCP execution with audit for verl GRPO rollout.

与 LiveMCPReplayLoop 的区别：
  - Replay：使用预存的 replay_observation，不调用真实 MCP
  - Oval：使用真实 MCP server subprocess 执行，产生真实 observation + 审计事件

rollout 流程：
  1. 模型生成 response（可能包含 <tool_call>）
  2. 解析 tool_call → 执行 LiveMCPExecutor → 获取真实 observation
  3. 通过 AuditWrapper 记录审计事件
  4. 返回 observation 给模型 → 继续生成下一步
  5. 终止或 max_turns → 将 audit_events 存入 extra_fields

verl 集成方式：
  - 通过 configs/agent_loop.yaml 注册为 "livemcp_oval"
  - 数据中 extra_info 需要包含 task 定义（target_servers, required_tools 等）
"""

import json
import os
import re
from typing import Any
from uuid import uuid4

from loguru import logger

from src.agent_loop.oval_mcp_worker import OvalMCPWorkerContext
from src.live_mcp.types import ToolCall

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
        response_logprobs: list[float] | None = None
        reward_score: float | None = None
        num_turns: int = 0
        metrics: dict = field(default_factory=dict)
        extra_fields: dict = field(default_factory=dict)

    def register(name: str):
        def decorator(cls):
            return cls
        return decorator


logger = logger.opt(colors=True)


# ── 工具调用解析（与 replay loop 相同） ──

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
    """判断模型输出是否为终止响应。"""
    return bool(
        _FINAL_ANSWER_PATTERN.search(text)
        or _REPORT_ERROR_PATTERN.search(text)
        or _ASK_CLARIFICATION_PATTERN.search(text)
    )


def _parse_tool_calls_json(text: str) -> list[dict]:
    """从 <tool_call> 内容中解析工具调用。"""
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


def _parse_terminal_type(text: str) -> str:
    """从模型输出中提取终止动作类型。"""
    if _FINAL_ANSWER_PATTERN.search(text):
        return "final_answer"
    if _REPORT_ERROR_PATTERN.search(text):
        return "report_error"
    if _ASK_CLARIFICATION_PATTERN.search(text):
        return "ask_clarification"
    return "unknown"


# ── 进程级 OvalMCPWorkerContext（单例，避免每个 rollout 重启 server） ──

import threading

_oval_ctx: OvalMCPWorkerContext | None = None
_oval_ctx_started: bool = False
_oval_ctx_lock = threading.Lock()


def _get_oval_ctx(
    suite_path: str = "configs/live_mcp/suite_mvp.yaml",
    domains: list[str] | None = None,
) -> OvalMCPWorkerContext:
    """获取或创建进程级 OvalMCPWorkerContext 单例（线程安全）。"""
    global _oval_ctx, _oval_ctx_started
    with _oval_ctx_lock:
        if _oval_ctx is None:
            _oval_ctx = OvalMCPWorkerContext(suite_path=suite_path, domains=domains)
        if not _oval_ctx_started:
            _oval_ctx.start()
            _oval_ctx_started = True
            logger.info("[oval] OvalMCPWorkerContext started (process-level singleton)")
    return _oval_ctx


def shutdown_oval_ctx() -> None:
    """关闭进程级 OvalMCPWorkerContext（用于测试清理，线程安全）。"""
    global _oval_ctx, _oval_ctx_started
    with _oval_ctx_lock:
        if _oval_ctx is not None and _oval_ctx_started:
            _oval_ctx.stop()
            _oval_ctx_started = False
            logger.info("[oval] OvalMCPWorkerContext stopped")


@register("livemcp_oval")
class LiveMCPOvalLoop(AgentLoopBase):
    """LiveMCP Oval Agent Loop — live MCP execution with audit。

    rollout 流程：
    1. 模型生成 response（可能包含 <tool_call>）
    2. 如果是 tool_call：通过 LiveMCPExecutor 执行 → 获取真实 observation → 记录审计事件
    3. 如果是 terminal：记录终止事件 → 结束
    4. 重复直到 max_turns 或 response_length 耗尽
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        rollout_cfg = self.config.actor_rollout_ref.rollout
        multi_turn_cfg = rollout_cfg.get("multi_turn", {})
        self.max_turns = int(
            multi_turn_cfg.get("max_assistant_turns", None)
            or rollout_cfg.get("max_turns", 5)
            or 5
        )
        self.max_obs_length = 1024
        self.response_length = int(rollout_cfg.response_length)
        self.apply_chat_template_kwargs = self.config.data.get("apply_chat_template_kwargs", {})

        # Oval 配置
        try:
            from src.training.livemcp_hyperparams import get_config
            cfg = get_config()
        except ImportError:
            cfg = None
        self.suite_path = (
            os.environ.get("OVAL_SUITE_PATH")
            or (cfg.suite_path if cfg else None)
            or "configs/live_mcp/suite_mvp.yaml"
        )
        domains_str = (
            os.environ.get("OVAL_DOMAINS")
            or (cfg.domains if cfg else None)
            or "calendar,shopping,banking,email,filesystem,payments,crm,issue_tracker,team_chat,food_delivery"
        )
        self.domains = [d.strip() for d in domains_str.split(",") if d.strip()]

        self._ctx: OvalMCPWorkerContext | None = None

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        """运行 live MCP Oval rollout。"""
        raw_prompt = kwargs.get("raw_prompt", [])
        extra_info = kwargs.get("extra_info", {})

        # ── normalize extra_info ──
        from src.utils import normalize_extra_info, normalize_json_field
        extra_info = normalize_extra_info(extra_info)

        # ── 获取 task 信息 ──
        task_domain = extra_info.get("target_servers", extra_info.get("domain", ""))
        if isinstance(task_domain, list):
            task_domain = task_domain[0] if task_domain else ""
        if not task_domain:
            task_domain = "calendar"  # fallback

        required_tools = extra_info.get("required_tools", [])
        if isinstance(required_tools, str):
            required_tools = [t.strip() for t in required_tools.split(",")]
        budget = extra_info.get("budget", self.max_turns)
        task_id = extra_info.get("task_id", str(uuid4().hex[:8]))
        request_id = uuid4().hex
        rid_short = request_id[:8]
        conversation_queries = normalize_json_field(
            extra_info.get("conversation_queries", []),
            default=[],
        )
        if not isinstance(conversation_queries, list):
            conversation_queries = []
        conversation_queries = [str(q) for q in conversation_queries if str(q).strip()]

        # ── 获取 OvalMCPWorkerContext ──
        if self._ctx is None:
            self._ctx = _get_oval_ctx(
                suite_path=self.suite_path,
                domains=self.domains,
            )

        ctx = self._ctx

        # ── 创建 session ──
        session_seed = extra_info.get("session_seed", 42)
        if isinstance(session_seed, str):
            session_seed = int(session_seed)
        session_id = ctx.create_session(seed=session_seed)

        # A row is valid only for the exact deterministic initial state used by
        # its teacher oracle.  Silent reset drift invalidates both reward and
        # safety diagnostics.
        expected_initial_hash = str(extra_info.get("initial_state_hash", ""))
        if expected_initial_hash:
            import hashlib
            actual_state = ctx.get_state(session_id, task_domain)
            canonical = json.dumps(actual_state, sort_keys=True, ensure_ascii=True, default=str)
            actual_initial_hash = hashlib.sha256(canonical.encode()).hexdigest()
            if actual_initial_hash != expected_initial_hash:
                ctx.close_session(session_id)
                raise RuntimeError(
                    f"initial state hash mismatch for task={task_id}: "
                    f"expected={expected_initial_hash[:12]} actual={actual_initial_hash[:12]}"
                )

        # ── missing_function: blocked tools ──
        hidden_tools = extra_info.get("hidden_tools", [])
        if isinstance(hidden_tools, str):
            hidden_tools = [t.strip() for t in hidden_tools.split(",") if t.strip()]
        blocked_tools: set[str] | None = set(hidden_tools) if hidden_tools else None

        # P1-3: 校验 visible_tools 与 hidden_tools 一致性。
        # missing_function 场景的核心机制是从 prompt 中移除 blocked 工具。
        # 如果 visible_tools 中仍包含 blocked 工具，模型会看到不可用的
        # 工具而产生困惑。
        visible_tool_names = extra_info.get("visible_tool_names", [])
        if isinstance(visible_tool_names, str):
            visible_tool_names = [t.strip() for t in visible_tool_names.split(",")]
        if blocked_tools and visible_tool_names:
            still_visible = blocked_tools & set(visible_tool_names)
            if still_visible:
                logger.warning(
                    f"[oval {rid_short}] hidden_tools 一致性警告: "
                    f"被标记为 hidden 的工具 {still_visible} 仍然出现在 "
                    f"visible_tool_names 中。模型可能会尝试调用不可用的工具。"
                )

        if self.tokenizer is None:
            self.tokenizer = kwargs.get("tokenizer")
        if self.tokenizer is None:
            ctx.close_session(session_id)
            raise RuntimeError("LiveMCPOvalLoop.tokenizer is None")

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

        all_response_ids: list[int] = []
        all_response_mask: list[int] = []
        audit_events: list[dict] = []
        n_model_tool_calls = 0
        n_exec_success = 0

        logger.debug(
            f"[oval {rid_short}] start | task={task_id} domain={task_domain} "
            f"| required_tools={required_tools} | budget={budget}"
        )

        # P2-9: respect per-task budget when smaller than max_turns.
        # budget comes from extra_info; max_turns is the loop hard cap.
        try:
            budget_int = int(budget)
        except (TypeError, ValueError):
            budget_int = self.max_turns
        effective_max_turns = min(self.max_turns, max(1, budget_int))
        turn_idx = -1  # so turn_idx+1 == 0 if loop never enters
        conversation_round_idx = 0

        for turn_idx in range(effective_max_turns):
            # 1. 模型生成
            try:
                output = await self.server_manager.generate(
                    request_id=request_id,
                    prompt_ids=prompt_ids + all_response_ids,
                    sampling_params=sampling_params,
                    image_data=None,
                )
            except Exception as e:
                logger.error(f"[oval {rid_short}] turn={turn_idx} 生成失败: {e}")
                break

            response_ids = (
                output.token_ids.tolist()
                if hasattr(output.token_ids, "tolist")
                else list(output.token_ids)
            )
            response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)

            all_response_ids.extend(response_ids)
            all_response_mask.extend([1] * len(response_ids))

            # 长度兜底
            if len(all_response_ids) >= self.response_length:
                logger.debug(f"[oval {rid_short}] turn={turn_idx} response_length 耗尽")
                break

            # 2. 解析模型输出
            tool_call_matches = list(_TOOL_CALL_PATTERN.finditer(response_text))

            if not tool_call_matches:
                # 无 tool_call → 终止动作
                terminal_type = _parse_terminal_type(response_text)
                logger.debug(
                    f"[oval {rid_short}] turn={turn_idx} terminal: {terminal_type}"
                )
                # 记录终止审计事件
                try:
                    event = ctx.execute_terminal_with_audit(
                        session_id=session_id,
                        domain=task_domain,
                        action_type=terminal_type,
                        model_output=response_text,
                    )
                    audit_events.append(event.to_dict())
                except Exception as e:
                    logger.warning(f"[oval {rid_short}] audit terminal 失败: {e}")
                next_round_idx = conversation_round_idx + 1
                if (
                    terminal_type in ("final_answer", "report_error", "ask_clarification")
                    and next_round_idx < len(conversation_queries)
                ):
                    followup = conversation_queries[next_round_idx]
                    user_tokens = await self._encode_message_tokens([
                        {"role": "user", "content": followup}
                    ])
                    if len(all_response_ids) + len(user_tokens) >= self.response_length:
                        logger.debug(
                            f"[oval {rid_short}] turn={turn_idx} follow-up 后超长，终止"
                        )
                        break
                    all_response_ids.extend(user_tokens)
                    all_response_mask.extend([0] * len(user_tokens))
                    conversation_round_idx = next_round_idx
                    logger.debug(
                        f"[oval {rid_short}] injected follow-up round "
                        f"{conversation_round_idx + 1}/{len(conversation_queries)}"
                    )
                    continue
                break

            # 同一 turn 同时输出 tool_call 和 terminal tag → 非法
            if _is_terminal_response(response_text):
                logger.debug(
                    f"[oval {rid_short}] turn={turn_idx} 同一 turn 同时输出 "
                    f"tool_call 和 terminal tag，视为非法，终止"
                )
                break

            # 3. 处理 tool_call → 真实 MCP 执行
            all_parsed_calls: list[dict] = []
            for tc_match in tool_call_matches:
                tc_content = tc_match.group(1)
                parsed_list = _parse_tool_calls_json(tc_content)
                all_parsed_calls.extend(parsed_list)

            n_model_tool_calls += 1

            if len(all_parsed_calls) != 1:
                # JSON 解析失败 → 返回错误 observation
                observation = (
                    "Error: Emit exactly one valid <tool_call> per assistant turn; "
                    f"received {len(all_parsed_calls)}."
                )
                logger.warning(
                    f"[oval {rid_short}] turn={turn_idx} expected one tool call, "
                    f"got {len(all_parsed_calls)}"
                )
            else:
                # 取第一个 tool_call 执行（串行模式）
                parsed_call = all_parsed_calls[0]
                tool_call = ToolCall(
                    name=parsed_call.get("name", ""),
                    arguments=parsed_call.get("arguments", {}),
                    call_id=uuid4().hex[:8],
                    raw_text=tc_content,
                )

                try:
                    event, exec_result = ctx.execute_with_audit(
                        session_id=session_id,
                        domain=task_domain,
                        tool_call=tool_call,
                        model_output=response_text,
                        blocked_tools=blocked_tools,
                    )
                    audit_events.append(event.to_dict())

                    if exec_result.success:
                        n_exec_success += 1
                        observation = (
                            json.dumps(exec_result.observation, ensure_ascii=False)
                            if isinstance(exec_result.observation, (dict, list))
                            else str(exec_result.observation or "")
                        )
                    else:
                        observation = (
                            f"Error: {exec_result.error_message}"
                            if exec_result.error_message
                            else "Error: tool execution failed."
                        )

                    logger.debug(
                        f"[oval {rid_short}] turn={turn_idx} exec: "
                        f"tool={tool_call.name} ok={exec_result.success}"
                    )
                except Exception as e:
                    observation = f"Error: tool execution failed: {e}"
                    logger.warning(f"[oval {rid_short}] turn={turn_idx} exec 异常: {e}")

            # 4. 拼接 observation 到 response
            if len(observation) > self.max_obs_length:
                observation = observation[: self.max_obs_length] + "...(truncated)"

            tool_msg = [{"role": "tool", "content": observation}]
            tool_tokens = await self._encode_message_tokens(tool_msg)

            if len(all_response_ids) + len(tool_tokens) >= self.response_length:
                logger.debug(f"[oval {rid_short}] turn={turn_idx} 加入 obs 后超长，终止")
                break

            all_response_ids.extend(tool_tokens)
            all_response_mask.extend([0] * len(tool_tokens))

        # 截断
        all_response_ids = all_response_ids[: self.response_length]
        all_response_mask = all_response_mask[: self.response_length]

        # Capture verifier evidence before closing the isolated session.
        final_state: dict[str, Any] = {}
        try:
            final_state = ctx.get_state(session_id, task_domain)
        except Exception as e:
            logger.warning(f"[oval {rid_short}] final state capture failed: {e}")

        # 清理 session
        try:
            ctx.close_session(session_id)
        except Exception:
            pass

        logger.debug(
            f"[oval {rid_short}] done | turns={turn_idx + 1} "
            f"| tool_calls={n_model_tool_calls} exec_ok={n_exec_success} "
            f"| audit_events={len(audit_events)} | response_len={len(all_response_ids)}"
        )

        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=all_response_ids,
            response_mask=all_response_mask,
            reward_score=None,  # 由外部 reward function 计算
            num_turns=turn_idx + 1,
            metrics={},
            extra_fields={
                "n_model_tool_calls": n_model_tool_calls,
                "n_exec_success": n_exec_success,
                "audit_events": audit_events,
                "task_id": task_id,
                "domain": task_domain,
                "required_tools": required_tools,
                "session_id": session_id,
                "final_state": final_state,
            },
        )

    async def _encode_message_tokens(self, add_messages: list[dict]) -> list[int]:
        """编码 tool observation 消息。"""
        response_ids = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.apply_chat_template(
                add_messages, add_generation_prompt=True, tokenize=True,
                **self.apply_chat_template_kwargs,
            ),
        )
        return list(response_ids)
