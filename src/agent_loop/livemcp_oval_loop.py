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
  5. 终止或 row action budget 耗尽 → 将 audit_events 存入 extra_fields

verl 集成方式：
  - 通过 configs/livemcp_rollout.yaml 注册为 "livemcp_oval"
  - 数据中 extra_info 需要包含 task 定义（target_servers, required_tools 等）
"""

import json
import hashlib
import os
import re
from collections import Counter
from collections.abc import Mapping
from typing import Any
from uuid import uuid4

from loguru import logger

from src.agent_loop.livemcp_oval_worker import OvalMCPWorkerContext
from src.live_mcp.protocol.observation import (
    TRAJECTORY_SCHEMA_VERSION,
    OBSERVATION_SCHEMA_VERSION,
    OBSERVATION_PROJECTION_VERSION,
    serialize_execution_error,
    serialize_tool_result,
)
from src.live_mcp.types import ToolCall
from src.utils import strip_think_tags

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


def _derive_sampling_seed(
    base_seed: int,
    trajectory_info: Mapping[str, Any],
    turn_idx: int,
) -> int:
    """Derive one stable vLLM request seed per rollout turn."""
    payload = {
        "base_seed": int(base_seed),
        "step": int(trajectory_info.get("step", -1)),
        "sample_index": str(trajectory_info.get("sample_index", "")),
        "rollout_n": int(trajectory_info.get("rollout_n", 0)),
        "validate": bool(trajectory_info.get("validate", False)),
        "turn_idx": int(turn_idx),
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def _single_conversation_token_ids(encoded: Any) -> list[int]:
    """Normalize one chat-template result to a flat token-id list."""
    if isinstance(encoded, Mapping):
        if "input_ids" not in encoded:
            raise TypeError("chat template encoding is missing input_ids")
        encoded = encoded["input_ids"]
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if not isinstance(encoded, (list, tuple)):
        raise TypeError(
            "unsupported chat template encoding type: "
            f"{type(encoded).__name__}"
        )
    if encoded and isinstance(encoded[0], (list, tuple)):
        if len(encoded) != 1:
            raise ValueError(
                "chat template must encode exactly one conversation; "
                f"got batch_size={len(encoded)}"
            )
        encoded = encoded[0]
    if not all(isinstance(token_id, int) for token_id in encoded):
        raise TypeError("chat template input_ids must contain integers")
    return list(encoded)


def _validate_environment_metadata(
    extra_info: dict[str, Any],
    current_tools: list[dict[str, Any]],
    reward_profile: str,
    *,
    current_tools_by_domain: dict[str, list[dict[str, Any]]] | None = None,
    required_owner_domains: set[str] | None = None,
    runtime_max_observation_chars: int | None = None,
) -> None:
    """Fail before rollout when data and live environment contracts drift."""
    from src.live_mcp.registry.environment_metadata import (
        validate_environment_metadata,
    )

    runtime_by_domain = dict(current_tools_by_domain or {})
    required_domains = set(required_owner_domains or set())
    if not runtime_by_domain and len(required_domains) == 1:
        runtime_by_domain[next(iter(required_domains))] = current_tools
    validate_environment_metadata(
        extra_info,
        current_tools_by_domain=runtime_by_domain,
        required_owner_domains=required_domains,
        reward_profile=reward_profile,
        runtime_max_observation_chars=(
            int(runtime_max_observation_chars)
            if runtime_max_observation_chars is not None
            else int(extra_info.get("max_observation_chars", 0))
        ),
    )


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
_MAX_IDENTICAL_INVALID_TOOL_CALLS = 2

# — P0-2: fuzzy terminal type fallback patterns ————————————————
# Qwen3-4B-Instruct-2507 frequently produces plain-text terminals without XML tags.
# These regexes provide content-based classification when XML parsing
# returns "unknown".  They are intentionally ordered:
#   clarification > report_error > answer (most-specific first).
_CLARIFICATION_MARKER_RE = re.compile(
    r"\b(?:"
    r"clarif(?:y|ication|y\s+which|y\s+what)"
    r"|which\s+(?:one|product|item|order|account|event|email)"
    r"|can\s+you\s+(?:specify|tell\s+me|provide|clarify|elaborate)"
    r"|could\s+you\s+(?:specify|tell\s+me|provide|clarify|elaborate|send|share)"
    r"|please\s+(?:specify|clarify|tell\s+me|provide|give\s+me)"
    r"|provide\s+(?:me|us)\s+with"
    r"|do\s+you\s+(?:have|know|mind)"
    r"|I\s+need\s+(?:more|additional)\s+(?:information|detail|context)"
    r"|not\s+sure\s+which"
    r")\b",
    re.IGNORECASE,
)
_ERROR_MARKER_RE = re.compile(
    r"\b(?:"
    # explicit inability + action verb (include assist/archive/help — the
    # abstention-domain verbs observed in real Qwen3-4B-Instruct-2507 output)
    r"(?:cannot|can't|unable\s+to|won't)\s+(?:complete|process|fulfil|do|find|"
    r"access|perform|assist|archive|help|provide|recommend|handle|support|suggest)"
    # capability absence (no-tool / abstention phrasing)
    r"|(?:no\s+tool|no\s+tools|none\s+of\s+the\s+available\s+tools)\s+"
    r"(?:is\s+|are\s+)?(?:available|support)"
    r"|(?:outside\s+the\s+scope|out\s+of\s+scope|not\s+within\s+the\s+scope)"
    r"|(?:does\s+not\s+support|do\s+not\s+support)"
    r"|(?:not\s+possible|not\s+available|not\s+supported|no\s+way)"
    r"|(?:sorry|unfortunately).*(?:cannot|can't|unable|not\s+(?:possible|able))"
    r"|I\s+(?:don't|do\s+not)\s+have\s+(?:enough|sufficient|the\s+required|access\s+to)"
    r"|(?:missing|absent|unavailable)\s+(?:tool|function|capability|feature)"
    r"|report_error"
    r")\b",
    re.IGNORECASE,
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


def _next_identical_action_count(
    previous_signature: str | None,
    current_signature: str,
    previous_count: int,
) -> int:
    """Count consecutive identical model actions after whitespace normalization."""
    if previous_signature == current_signature:
        return previous_count + 1
    return 1


def _unknown_tool_audit_event(
    *,
    session_id: str,
    turn_idx: int,
    round_idx: int,
    tool_name: str,
    tool_arguments: dict[str, Any],
) -> dict[str, Any]:
    """Record a model-selected tool outside the visible schema as a model error."""
    return {
        "event_id": f"unknown_tool_{uuid4().hex[:8]}",
        "session_id": session_id,
        "step": turn_idx,
        "round_idx": round_idx,
        "action_type": "tool_call",
        "tool_name": tool_name,
        "tool_arguments": dict(tool_arguments),
        "tool_name_known": False,
        "schema_valid": False,
        "execution_success": False,
        "error_type": "unknown_tool",
        "error_message": f"tool is not in the visible schema: {tool_name}",
        "state_changed": False,
    }


def _invalid_mixed_action_audit_event(
    *, session_id: str, turn_idx: int, round_idx: int,
) -> dict[str, Any]:
    """Represent a model-format error as scoreable evidence, not infra drift."""
    return {
        "event_id": f"invalid_mixed_action_{uuid4().hex[:8]}",
        "session_id": session_id,
        "step": turn_idx,
        "round_idx": round_idx,
        "action_type": "tool_call",
        "tool_name": "",
        "tool_arguments": {},
        "tool_name_known": False,
        "schema_valid": False,
        "execution_success": False,
        "error_type": "invalid_mixed_action",
        "error_message": "tool_call and terminal emitted together",
        "state_changed": False,
    }


def _parse_terminal_type(text: str) -> str:
    """Extract terminal action type from model output.

    Priority:
      1. XML tags (``<final_answer>``, ``<report_error>``,
         ``<ask_clarification>``) — highest confidence.
      2. Content-based keyword fallback — for Qwen3-4B-Instruct-2507 plain-text
         terminals that lack XML formatting.
      3. ``"unknown"`` when the text cannot be classified.
    """
    if _FINAL_ANSWER_PATTERN.search(text):
        return "final_answer"
    if _REPORT_ERROR_PATTERN.search(text):
        return "report_error"
    if _ASK_CLARIFICATION_PATTERN.search(text):
        return "ask_clarification"

    # — fuzzy content-based fallback (PROVE-aligned) —
    stripped = text.strip()
    if not stripped:
        return "unknown"

    # clarification markers: model is asking the user for more info
    if _CLARIFICATION_MARKER_RE.search(stripped):
        return "ask_clarification"

    # error markers: model reports it cannot complete the request
    if _ERROR_MARKER_RE.search(stripped):
        return "report_error"

    return "unknown"


def _resolve_terminal_type(
    text: str,
    *,
    allowed_terminal_actions: list[str] | None,
    required_tools: list[str],
    successful_tool_names: list[str],
    allow_plain_final: bool = True,
) -> tuple[str, bool]:
    """Resolve an explicit terminal or a justified plain-text fallback.

    PROVE baseline does not validate terminal semantics (§4, OVAL-MCP.md).
    The model's terminal classification should be content-based, not gated
    on XML tag presence.  Qwen3-4B-Instruct-2507 frequently emits plain text without
    ``<final_answer>`` tags.

    Returns ``(terminal_type, inferred_plain_answer)``.
    """
    explicit = _parse_terminal_type(text)
    if explicit != "unknown":
        return explicit, False

    # — PROVE-aligned plain-text fallback —
    # When the model produces non-empty text without <tool_call> tags
    # and without explicit terminal XML tags, classify by content.
    if not allow_plain_final:
        return "unknown", False

    stripped = text.strip()
    if not stripped:
        return "unknown", False

    # If all required tools in the round have already succeeded,
    # plain text is most likely a final answer.
    allowed = [str(value) for value in (allowed_terminal_actions or [])]
    missing = Counter(str(value) for value in required_tools) - Counter(
        str(value) for value in successful_tool_names
    )
    if not missing:
        # Plain text after all required work → final_answer.
        return "final_answer", True

    # Plain text before completing required tools, but
    # content-based parsing already tried above.  If the contract
    # only allows final_answer and the text doesn't match
    # clarification/error patterns, treat as final_answer anyway
    # (PROVE baseline doesn't penalize terminal choice).
    if allowed == ["final_answer"] or not allowed:
        return "final_answer", True

    # For other contracts (e.g., ["ask_clarification"] only),
    # fall back to the most permissive terminal the contract allows.
    if "final_answer" in allowed:
        return "final_answer", True
    if "ask_clarification" in allowed:
        return "ask_clarification", False
    if "report_error" in allowed:
        return "report_error", False

    return "unknown", False


# ── 进程级 OvalMCPWorkerContext（单例，避免每个 rollout 重启 server） ──

import threading

_oval_ctx: OvalMCPWorkerContext | None = None
_oval_ctx_started: bool = False
_oval_ctx_lock = threading.Lock()


def _get_oval_ctx(
    suite_path: str = "configs/live_mcp/ten_domain_suite.yaml",
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


@register("livemcp_oval")
class LiveMCPOvalLoop(AgentLoopBase):
    """LiveMCP Oval Agent Loop — live MCP execution with audit。

    rollout 流程：
    1. 模型生成 response（可能包含 <tool_call>）
    2. 如果是 tool_call：通过 LiveMCPExecutor 执行 → 获取真实 observation → 记录审计事件
    3. 如果是 terminal：记录终止事件 → 结束
    4. 重复直到逐行 action budget 或 response_length 耗尽
    """

    def __init__(self, **kwargs):
        max_action_tokens = kwargs.pop("max_action_tokens", 1024)
        super().__init__(**kwargs)
        rollout_cfg = self.config.actor_rollout_ref.rollout
        multi_turn_cfg = rollout_cfg.get("multi_turn", {})
        self.max_turns = int(
            multi_turn_cfg.get("max_assistant_turns", None)
            or rollout_cfg.get("max_turns", 5)
            or 5
        )
        self.response_length = int(rollout_cfg.response_length)
        self.max_action_tokens = int(max_action_tokens)
        if self.max_action_tokens <= 0:
            raise ValueError("max_action_tokens must be positive")
        self.max_action_tokens = min(
            self.max_action_tokens,
            self.response_length,
        )
        self.rollout_seed = int(rollout_cfg.get("seed", 0))
        self.apply_chat_template_kwargs = dict(
            self.config.data.get("apply_chat_template_kwargs", {}) or {}
        )
        self.apply_chat_template_kwargs.setdefault("enable_thinking", False)

        # Oval 配置
        from src.training.hyperparams import get_config
        cfg = get_config()
        self.allow_plain_final = bool(cfg.plain_final_compat)
        self.suite_path = (
            os.environ.get("OVAL_SUITE_PATH")
            or cfg.suite_path
            or "configs/live_mcp/ten_domain_suite.yaml"
        )
        try:
            from src.live_mcp.config import load_suite_config

            self.max_obs_length = cfg.resolve_max_observation_chars(
                load_suite_config(self.suite_path).rollout
            )
        except Exception as exc:
            raise RuntimeError(
                f"cannot resolve observation budget from {self.suite_path}: {exc}"
            ) from exc
        self.reward_profile = str(
            cfg.reward_profile
        )
        domains_str = (
            os.environ.get("OVAL_DOMAINS")
            or cfg.domains
            or "calendar,shopping,banking,email,filesystem,payments,crm,issue_tracker,team_chat,food_delivery"
        )
        self.domains = [d.strip() for d in domains_str.split(",") if d.strip()]

        self._ctx: OvalMCPWorkerContext | None = None

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        """Run one rollout and close every session even on unexpected errors."""
        cleanup: list[tuple[OvalMCPWorkerContext, str]] = []
        kwargs["_session_cleanup"] = cleanup
        try:
            return await self._run_impl(sampling_params, **kwargs)
        finally:
            for ctx, session_id in cleanup:
                try:
                    ctx.close_session(session_id)
                except Exception:
                    pass

    async def _run_impl(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        """运行 live MCP Oval rollout。"""
        raw_prompt = kwargs.get("raw_prompt", [])
        extra_info = kwargs.get("extra_info", {})
        trajectory_info = kwargs.get("_trajectory_info", {})
        if not isinstance(trajectory_info, Mapping):
            raise TypeError("_trajectory_info must be a mapping")

        # ── normalize extra_info ──
        from src.utils import normalize_extra_info, normalize_json_field
        extra_info = normalize_extra_info(extra_info)
        from src.live_mcp.registry.environment_metadata import (
            validate_prove_corpus_evidence,
            validate_semantic_gate_evidence,
            validate_teacher_generation_evidence,
            validate_training_artifact_evidence,
        )
        validate_prove_corpus_evidence(extra_info)
        validate_teacher_generation_evidence(extra_info)
        validate_semantic_gate_evidence(extra_info)
        validate_training_artifact_evidence(extra_info)
        from src.live_mcp.artifact.reward_task import build_reward_task
        build_reward_task(extra_info)

        # ── 获取 task 信息 ──
        task_domain = extra_info.get("target_servers", extra_info.get("domain", ""))
        if isinstance(task_domain, list):
            task_domain = task_domain[0] if task_domain else ""
        if not task_domain:
            raise RuntimeError("rollout row is missing target domain")

        required_tools = extra_info.get("required_tools", [])
        if isinstance(required_tools, str):
            required_tools = [t.strip() for t in required_tools.split(",")]
        if "budget" not in extra_info or "minimum_action_budget" not in extra_info:
            raise RuntimeError(
                "rollout row is missing canonical action budget metadata"
            )
        budget = extra_info["budget"]
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

        # P0-2: parse round contracts for rollout enforcement.
        # Each contract specifies required_tools and allowed_terminal_actions
        # for one conversation round.  The rollout loop MUST validate the
        # model's terminal against the contract before injecting follow-up.
        round_contracts_raw = normalize_json_field(
            extra_info.get("round_contracts", []),
            default=[],
        )
        if isinstance(round_contracts_raw, list):
            round_contracts = round_contracts_raw
        else:
            round_contracts = []

        # P0-2 Fix: multi-round data MUST have matching round_contracts.
        n_conversation_rounds = len(conversation_queries)
        if n_conversation_rounds < 1:
            raise RuntimeError(f"Task {task_id} has no conversation_queries")
        if not round_contracts:
            raise RuntimeError(f"Task {task_id} has no round_contracts")
        if len(round_contracts) != n_conversation_rounds:
                raise RuntimeError(
                    f"Task {task_id}: "
                    f"{len(round_contracts)} round_contracts vs "
                    f"{n_conversation_rounds} conversation queries. "
                    f"Counts must match."
                )
        for i, c in enumerate(round_contracts):
            actual_idx = c.get("round_idx", -1)
            if actual_idx != i:
                raise RuntimeError(
                    f"Task {task_id}: round_contracts[{i}] "
                    f"has round_idx={actual_idx}, expected {i}."
                )

        # A generated row must be structurally executable even before model
        # exploration: one action per reference tool call plus one terminal per
        # conversation round. This is enforced by the rollout contract.
        # quality gate.
        minimum_action_budget = len(required_tools) + max(1, n_conversation_rounds)
        try:
            row_action_budget = int(budget)
        except (TypeError, ValueError):
            row_action_budget = self.max_turns
        if row_action_budget < minimum_action_budget:
            raise RuntimeError(
                f"Task {task_id}: action budget {row_action_budget} cannot reproduce "
                f"{len(required_tools)} reference tool calls across "
                f"{max(1, n_conversation_rounds)} conversation round(s); "
                f"minimum is {minimum_action_budget}."
            )

        # ── 获取 OvalMCPWorkerContext ──
        if self._ctx is None:
            self._ctx = _get_oval_ctx(
                suite_path=self.suite_path,
                domains=self.domains,
            )

        ctx = self._ctx
        from src.live_mcp.registry.environment_metadata import (
            validate_tool_owner_contract,
        )
        tool_owner_domains = validate_tool_owner_contract(extra_info)
        raw_schema_hashes = normalize_json_field(
            extra_info.get("server_schema_hashes", {}), default={},
        )
        schema_owner_domains = (
            {str(domain) for domain in raw_schema_hashes}
            if isinstance(raw_schema_hashes, dict)
            else set()
        )
        required_owner_domains = {
            task_domain,
            *(str(domain) for domain in tool_owner_domains.values()),
        }
        _validate_environment_metadata(
            extra_info,
            ctx.get_tool_schemas(task_domain),
            self.reward_profile,
            current_tools_by_domain={
                domain: ctx.get_tool_schemas(domain)
                for domain in schema_owner_domains
            },
            required_owner_domains=required_owner_domains,
            runtime_max_observation_chars=self.max_obs_length,
        )

        # ── 创建 session ──
        session_seed = extra_info.get("session_seed", 42)
        if isinstance(session_seed, str):
            session_seed = int(session_seed)
        raw_state_profiles = normalize_json_field(
            extra_info.get("state_profiles", {}), default={},
        )
        if not isinstance(raw_state_profiles, dict):
            raise RuntimeError(
                f"Task {task_id}: state_profiles must be a mapping"
            )
        session_id = ctx.create_session(
            seed=session_seed,
            state_profiles={
                str(owner): str(profile)
                for owner, profile in raw_state_profiles.items()
            },
            server_names=sorted(required_owner_domains),
        )
        kwargs["_session_cleanup"].append((ctx, session_id))

        # Bind every executable owner, including cross-domain distractors, to
        # the exact deterministic state used during Teacher generation.
        import hashlib
        actual_initial_hashes: dict[str, str] = {}
        for owner in sorted(required_owner_domains):
            owner_state = ctx.get_state(session_id, owner)
            canonical = json.dumps(
                owner_state, sort_keys=True, ensure_ascii=True, default=str,
            )
            actual_initial_hashes[owner] = hashlib.sha256(
                canonical.encode()
            ).hexdigest()
        from src.live_mcp.registry.environment_metadata import (
            validate_environment_metadata,
        )
        validate_environment_metadata(
            extra_info,
            current_tools_by_domain={
                owner: ctx.get_tool_schemas(owner)
                for owner in required_owner_domains
            },
            required_owner_domains=required_owner_domains,
            reward_profile=self.reward_profile,
            runtime_max_observation_chars=self.max_obs_length,
            current_initial_state_hashes=actual_initial_hashes,
        )
        expected_primary_hash = str(extra_info.get("initial_state_hash", ""))
        if expected_primary_hash != actual_initial_hashes[task_domain]:
            ctx.close_session(session_id)
            raise RuntimeError(
                f"primary initial_state_hash mismatch for task={task_id}"
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
        visible_tool_names = normalize_json_field(
            extra_info.get("visible_tool_names", []), default=[],
        )
        if not isinstance(visible_tool_names, list):
            raise RuntimeError(
                f"visible_tool_names must be a list for task={task_id}"
            )
        visible_tool_names = [str(name) for name in visible_tool_names]
        visible_tool_name_set = set(visible_tool_names)
        if blocked_tools and visible_tool_names:
            still_visible = blocked_tools & visible_tool_name_set
            if still_visible:
                raise RuntimeError(
                    f"hidden tools remain visible for task={task_id}: "
                    f"{sorted(still_visible)}"
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

        prompt_text = json.dumps(messages, ensure_ascii=False, default=str)
        leak_markers = (
            "oracle_calls",
            "success_criteria",
            "ground_truth",
            "allowed_terminal_actions",
            "hidden_tools",
        )
        leaked_markers = [marker for marker in leak_markers if marker in prompt_text]
        if leaked_markers:
            ctx.close_session(session_id)
            raise RuntimeError(
                f"prompt leakage for task={task_id}: supervised field(s) "
                f"visible in model prompt: {leaked_markers}"
            )
        if blocked_tools:
            # P0: Check schema-level leakage only — verify hidden tools are NOT
            # in visible_tool_names (the candidate schema the model sees).
            # Do NOT do a raw string search over the full prompt_text because
            # the user query may legitimately contain the tool name as natural
            # language (e.g. "checkout" in a shopping query).
            still_visible = blocked_tools & visible_tool_name_set
            if still_visible:
                ctx.close_session(session_id)
                raise RuntimeError(
                    f"prompt leakage for task={task_id}: hidden tool(s) "
                    f"{still_visible} still present in visible_tool_names schema"
                )

        # 编码初始 prompt
        prompt_encoding = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
                **self.apply_chat_template_kwargs,
            ),
        )
        prompt_ids = _single_conversation_token_ids(prompt_encoding)

        all_response_ids: list[int] = []
        all_response_mask: list[int] = []
        audit_events: list[dict] = []
        trajectory_diagnostics: list[dict] = []
        trajectory_errors: list[dict] = []
        trajectory_integrity_ok = True
        n_model_tool_calls = 0
        n_exec_success = 0
        sampling_seeds: list[int] = []

        logger.debug(
            f"[oval {rid_short}] start | task={task_id} domain={task_domain} "
            f"| required_tools={required_tools} | budget={budget}"
        )

        # The row-level action budget is authoritative for generated data.
        # The 2--3 turn setting counts conversation rounds; this loop spends
        # one iteration on each tool call and each per-round terminal.  Using
        # the configured default as a smaller hard cap can make a verified
        # reference trajectory impossible to reproduce.
        try:
            budget_int = int(budget)
        except (TypeError, ValueError):
            budget_int = self.max_turns
        effective_max_turns = max(1, budget_int)
        turn_idx = -1  # so turn_idx+1 == 0 if loop never enters
        conversation_round_idx = 0
        round_successful_tool_names: list[str] = []  # P0-2: tools called in current round (preserves multiplicity)
        last_invalid_tool_call_signature: str | None = None
        identical_invalid_tool_call_count = 0

        for turn_idx in range(effective_max_turns):
            # 1. 模型生成
            try:
                turn_sampling_params = dict(sampling_params)
                turn_seed = _derive_sampling_seed(
                    self.rollout_seed,
                    trajectory_info,
                    turn_idx,
                )
                turn_sampling_params["seed"] = turn_seed
                remaining_response_tokens = max(
                    1,
                    self.response_length - len(all_response_ids),
                )
                turn_sampling_params["max_tokens"] = min(
                    self.max_action_tokens,
                    remaining_response_tokens,
                )
                sampling_seeds.append(turn_seed)
                output = await self.server_manager.generate(
                    request_id=request_id,
                    prompt_ids=prompt_ids + all_response_ids,
                    sampling_params=turn_sampling_params,
                    image_data=None,
                )
            except Exception as e:
                logger.error(f"[oval {rid_short}] turn={turn_idx} 生成失败: {e}")
                trajectory_integrity_ok = False
                trajectory_errors.append({
                    "stage": "model_generation",
                    "turn": turn_idx,
                    "error": f"{type(e).__name__}: {e}",
                })
                break

            response_ids = (
                output.token_ids.tolist()
                if hasattr(output.token_ids, "tolist")
                else list(output.token_ids)
            )
            response_text = strip_think_tags(
                self.tokenizer.decode(response_ids, skip_special_tokens=True)
            )

            all_response_ids.extend(response_ids)
            all_response_mask.extend([1] * len(response_ids))

            # 2. 解析模型输出
            tool_call_matches = list(_TOOL_CALL_PATTERN.finditer(response_text))
            stop_after_invalid_tool_call = False

            if not tool_call_matches:
                # 无 tool_call → 终止动作
                current_contract = (
                    round_contracts[conversation_round_idx]
                    if round_contracts
                    and conversation_round_idx < len(round_contracts)
                    else None
                )
                contract_allowed = (
                    current_contract.get("allowed_terminal_actions", [])
                    if current_contract else None
                )
                contract_required = (
                    current_contract.get("required_tools", [])
                    if current_contract else []
                )
                terminal_type, inferred_plain_final = _resolve_terminal_type(
                    response_text,
                    allowed_terminal_actions=contract_allowed,
                    required_tools=contract_required,
                    successful_tool_names=round_successful_tool_names,
                    allow_plain_final=self.allow_plain_final,
                )
                logger.debug(
                    f"[oval {rid_short}] turn={turn_idx} terminal: {terminal_type}"
                )
                if inferred_plain_final:
                    logger.debug(
                        f"[oval {rid_short}] turn={turn_idx} "
                        "inferred plain final_answer after satisfying the "
                        "current round contract"
                    )
                    trajectory_diagnostics.append({
                        "event_id": f"inferred_plain_final_{uuid4().hex[:8]}",
                        "session_id": session_id,
                        "step": turn_idx,
                        "action_type": "inferred_plain_final_answer",
                        "round_idx": conversation_round_idx,
                        "required_tools": list(contract_required),
                        "called_tools": list(round_successful_tool_names),
                    })
                # 记录终止审计事件
                try:
                    event = ctx.execute_terminal_with_audit(
                        session_id=session_id,
                        domain=task_domain,
                        action_type=terminal_type,
                        model_output=response_text,
                    )
                    event.round_idx = conversation_round_idx
                    audit_events.append(event.to_dict())
                except Exception as e:
                    logger.warning(f"[oval {rid_short}] audit terminal 失败: {e}")
                    trajectory_integrity_ok = False
                    trajectory_errors.append({
                        "stage": "terminal_audit",
                        "turn": turn_idx,
                        "error": f"{type(e).__name__}: {e}",
                    })
                    # Preserve one action event per model action.  Reward will
                    # reject the trajectory via the integrity flag, but the
                    # event count and action history remain factual.
                    audit_events.append({
                        "event_id": f"terminal_audit_error_{uuid4().hex[:8]}",
                        "session_id": session_id,
                        "step": turn_idx,
                        "round_idx": conversation_round_idx,
                        "action_type": terminal_type,
                        "terminal_action": response_text,
                        "operation": "terminal",
                        "execution_success": False,
                        "schema_valid": True,
                        "tool_name_known": True,
                        "state_changed": False,
                        "error_type": "audit_infrastructure_error",
                        "error_message": str(e),
                    })

                # P0-2: validate terminal against round contract.
                # PROVE baseline (§4, OVAL-MCP.md) does NOT validate terminal
                # semantics — the reward function scores tool-call quality
                # independently of terminal choice.  A contract mismatch is
                # recorded as a diagnostic so that _validate_round_contracts
                # can produce r_round_ok=False, but the Policy's chosen
                # terminal drives the actual conversation flow.
                # The model still receives full credit for successful tool
                # calls made before the terminal.
                if contract_allowed is not None and terminal_type not in contract_allowed:
                    logger.debug(
                        f"[oval {rid_short}] turn={turn_idx} round={conversation_round_idx} "
                        f"terminal {terminal_type} not in contract {contract_allowed} — "
                        f"diagnostic only (PROVE baseline), continuing with Policy terminal"
                    )
                    trajectory_diagnostics.append({
                        "event_id": f"contract_violation_{uuid4().hex[:8]}",
                        "session_id": session_id,
                        "step": turn_idx,
                        "action_type": "contract_violation",
                        "round_idx": conversation_round_idx,
                        "terminal_type": terminal_type,
                        "allowed": contract_allowed,
                    })
                    # DO NOT break — PROVE baseline scores the trajectory as-is.

                # report_error always terminates the episode (P0-2 rule 3).
                if terminal_type == "report_error":
                    logger.debug(
                        f"[oval {rid_short}] turn={turn_idx} report_error — "
                        f"episode terminated"
                    )
                    break

                # ask_clarification only continues with a paired clarification reply.
                next_round_idx = conversation_round_idx + 1
                if terminal_type == "ask_clarification":
                    if next_round_idx < len(conversation_queries):
                        logger.debug(
                            f"[oval {rid_short}] turn={turn_idx} "
                            f"ask_clarification with paired reply → advancing"
                        )
                    else:
                        logger.debug(
                            f"[oval {rid_short}] turn={turn_idx} "
                            f"ask_clarification without paired reply → stopping"
                        )
                        break

                # Per-round required_tools describe the reference trace.  They
                # are diagnostic only; equivalent executable tool paths remain valid.
                # an exact-name miss must not truncate the conversation.
                if contract_required:
                    required_counts = Counter(contract_required)
                    called_counts = Counter(round_successful_tool_names)
                    missing_counts = required_counts - called_counts
                    if missing_counts:
                        missing_names = list(missing_counts.elements())
                        logger.debug(
                            f"[oval {rid_short}] turn={turn_idx} round={conversation_round_idx} "
                            f"terminal {terminal_type} missing required tools: "
                            f"{sorted(missing_counts)} "
                            f"(called: {sorted(round_successful_tool_names)}, "
                            f"required: {contract_required})"
                        )
                        trajectory_diagnostics.append({
                            "event_id": f"round_tool_diagnostic_{uuid4().hex[:8]}",
                            "session_id": session_id,
                            "step": turn_idx,
                            "action_type": "round_tool_diagnostic",
                            "round_idx": conversation_round_idx,
                            "required_tools": contract_required,
                            "called_tools": sorted(round_successful_tool_names),
                            "missing_tools": sorted(missing_names),
                        })

                # Only final_answer (or paired ask_clarification) advances.
                if (
                    terminal_type in ("final_answer", "ask_clarification")
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
                    round_successful_tool_names = []  # reset for next round
                    logger.debug(
                        f"[oval {rid_short}] injected follow-up round "
                        f"{conversation_round_idx + 1}/{len(conversation_queries)}"
                    )
                    continue
                break

            # 同一 turn 同时输出 tool_call 和 terminal tag → 非法
            # Count every model action containing a tool tag, including a
            # scoreable mixed-action format violation.
            n_model_tool_calls += 1
            if _is_terminal_response(response_text):
                logger.debug(
                    f"[oval {rid_short}] turn={turn_idx} 同一 turn 同时输出 "
                    f"tool_call 和 terminal tag，视为非法，终止"
                )
                trajectory_diagnostics.append({
                    "stage": "model_action_format",
                    "turn": turn_idx,
                    "error": "assistant emitted tool_call and terminal in one action",
                })
                audit_events.append(_invalid_mixed_action_audit_event(
                    session_id=session_id,
                    turn_idx=turn_idx,
                    round_idx=conversation_round_idx,
                ))
                break

            # 3. 处理 tool_call → 真实 MCP 执行
            all_parsed_calls: list[dict] = []
            for tc_match in tool_call_matches:
                tc_content = tc_match.group(1)
                parsed_list = _parse_tool_calls_json(tc_content)
                all_parsed_calls.extend(parsed_list)

            if len(all_parsed_calls) != 1:
                invalid_signature = re.sub(r"\s+", " ", response_text).strip()
                identical_invalid_tool_call_count = _next_identical_action_count(
                    last_invalid_tool_call_signature,
                    invalid_signature,
                    identical_invalid_tool_call_count,
                )
                last_invalid_tool_call_signature = invalid_signature
                stop_after_invalid_tool_call = (
                    identical_invalid_tool_call_count
                    >= _MAX_IDENTICAL_INVALID_TOOL_CALLS
                )
                # JSON 解析失败 → 返回错误 observation
                error_message = (
                    "Emit exactly one valid <tool_call> per assistant turn; "
                    f"received {len(all_parsed_calls)}."
                )
                observation = serialize_execution_error(
                    "invalid_tool_call",
                    error_message,
                    self.max_obs_length,
                    observation={"parsed_tool_call_count": len(all_parsed_calls)},
                )
                logger.warning(
                    f"[oval {rid_short}] turn={turn_idx} expected one tool call, "
                    f"got {len(all_parsed_calls)}"
                )
                audit_events.append({
                    "event_id": f"invalid_tool_call_{uuid4().hex[:8]}",
                    "session_id": session_id,
                    "step": turn_idx,
                    "round_idx": conversation_round_idx,
                    "action_type": "tool_call",
                    "tool_name": "",
                    "tool_arguments": {},
                    "tool_name_known": False,
                    "schema_valid": False,
                    "execution_success": False,
                    "error_type": "invalid_tool_call",
                    "error_message": error_message,
                    "state_changed": False,
                })
                if stop_after_invalid_tool_call:
                    logger.warning(
                        f"[oval {rid_short}] turn={turn_idx} repeated identical "
                        "invalid tool call; stopping recovery"
                    )
                    trajectory_diagnostics.append({
                        "event_id": f"repeated_invalid_tool_call_{uuid4().hex[:8]}",
                        "session_id": session_id,
                        "step": turn_idx,
                        "round_idx": conversation_round_idx,
                        "action_type": "repeated_invalid_tool_call",
                        "repeat_count": identical_invalid_tool_call_count,
                    })
            else:
                last_invalid_tool_call_signature = None
                identical_invalid_tool_call_count = 0
                # 取第一个 tool_call 执行（串行模式）
                parsed_call = all_parsed_calls[0]
                tool_call = ToolCall(
                    name=parsed_call.get("name", ""),
                    arguments=parsed_call.get("arguments", {}),
                    call_id=uuid4().hex[:8],
                    raw_text=tc_content,
                )

                if tool_call.name not in visible_tool_name_set:
                    error_message = (
                        f"tool is not in the visible schema: {tool_call.name}"
                    )
                    observation = serialize_execution_error(
                        "unknown_tool",
                        error_message,
                        self.max_obs_length,
                    )
                    logger.warning(
                        f"[oval {rid_short}] turn={turn_idx} unknown tool: "
                        f"{tool_call.name}"
                    )
                    audit_events.append(_unknown_tool_audit_event(
                        session_id=session_id,
                        turn_idx=turn_idx,
                        round_idx=conversation_round_idx,
                        tool_name=tool_call.name,
                        tool_arguments=tool_call.arguments,
                    ))
                else:
                    try:
                        execution_domain = tool_owner_domains[tool_call.name]
                        event, exec_result = ctx.execute_with_audit(
                            session_id=session_id,
                            domain=execution_domain,
                            tool_call=tool_call,
                            model_output=response_text,
                            blocked_tools=blocked_tools,
                        )
                        event.round_idx = conversation_round_idx
                        event.tool_name_known = True
                        if execution_domain != task_domain:
                            event.forbidden_transition = "cross_domain_distractor_call"
                        audit_events.append(event.to_dict())

                        if event.state_evidence_errors:
                            trajectory_integrity_ok = False
                            trajectory_errors.append({
                                "stage": "tool_state_evidence",
                                "turn": turn_idx,
                                "tool_name": tool_call.name,
                                "errors": list(event.state_evidence_errors),
                            })

                        if exec_result.success:
                            n_exec_success += 1
                            round_successful_tool_names.append(tool_call.name)
                        observation = serialize_tool_result(
                            exec_result, self.max_obs_length,
                        )

                        logger.debug(
                            f"[oval {rid_short}] turn={turn_idx} exec: "
                            f"tool={tool_call.name} ok={exec_result.success}"
                        )
                    except Exception as e:
                        observation = serialize_execution_error(
                            "rollout_execution_exception",
                            f"tool execution failed: {e}",
                            self.max_obs_length,
                        )
                        logger.warning(f"[oval {rid_short}] turn={turn_idx} exec 异常: {e}")
                        trajectory_integrity_ok = False
                        trajectory_errors.append({
                            "stage": "tool_execution_audit",
                            "turn": turn_idx,
                            "tool_name": tool_call.name,
                            "error": f"{type(e).__name__}: {e}",
                        })
                        audit_events.append({
                            "event_id": f"execution_audit_error_{uuid4().hex[:8]}",
                            "session_id": session_id,
                            "step": turn_idx,
                            "round_idx": conversation_round_idx,
                            "action_type": "tool_call",
                            "tool_name": tool_call.name,
                            "tool_arguments": dict(tool_call.arguments),
                            "tool_name_known": True,
                            "schema_valid": False,
                            "execution_success": False,
                            "error_type": "rollout_execution_exception",
                            "error_message": str(e),
                            "state_changed": False,
                        })

            if stop_after_invalid_tool_call:
                break

            # 4. 拼接 observation 到 response
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
        state_evidence = {"status": "available", "error": ""}
        try:
            final_state = ctx.get_state(session_id, task_domain)
        except Exception as e:
            logger.warning(f"[oval {rid_short}] final state capture failed: {e}")
            trajectory_integrity_ok = False
            state_evidence = {
                "status": "error",
                "error": f"{type(e).__name__}: {e}",
            }
            trajectory_errors.append({
                "stage": "final_state_capture",
                "turn": turn_idx,
                "error": state_evidence["error"],
            })

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
                "trajectory_diagnostics": trajectory_diagnostics,
                "trajectory_integrity_ok": trajectory_integrity_ok,
                "trajectory_errors": trajectory_errors,
                "state_evidence": state_evidence,
                "task_id": task_id,
                "domain": task_domain,
                "required_tools": required_tools,
                "session_id": session_id,
                "final_state": final_state,
                "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
                "observation_projection_version": OBSERVATION_PROJECTION_VERSION,
                "trajectory_schema_version": TRAJECTORY_SCHEMA_VERSION,
                "max_observation_chars": self.max_obs_length,
                "reward_profile": self.reward_profile,
                "plain_final_compat": self.allow_plain_final,
                "sampling_seeds": sampling_seeds,
                "trajectory_info": dict(trajectory_info),
            },
        )

    async def _encode_message_tokens(self, add_messages: list[dict]) -> list[int]:
        """编码 tool observation 消息。"""
        response_encoding = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.apply_chat_template(
                add_messages, add_generation_prompt=True, tokenize=True,
                **self.apply_chat_template_kwargs,
            ),
        )
        return _single_conversation_token_ids(response_encoding)
