"""
共享工具函数：extra_info / metadata 的 JSON-string normalization、数据迭代。

verl/pyarrow 序列化后，extra_info 及其嵌套字段可能变为 JSON 字符串。
本模块提供统一的 normalize 函数，供 reward、replay loop、register_estimator 复用。
"""

import json
from typing import Any, Iterable


def normalize_extra_info(value: Any) -> dict:
    """将 extra_info 规范化为 dict。

    支持：
    - None → {}
    - JSON string → dict
    - dict → dict（原样返回）
    - 其他类型 → {}
    """
    if value is None:
        return {}
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (json.JSONDecodeError, TypeError, ValueError):
            return {}
    if not isinstance(value, dict):
        return {}
    return value


def normalize_json_field(value: Any, default: Any = None) -> Any:
    """将可能是 JSON 字符串的字段规范化。

    如果 value 是字符串，尝试 json.loads；否则原样返回。
    解析失败时返回 default。
    """
    if default is None:
        default = {}
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError, ValueError):
            return default
    return value if value is not None else default


def strip_think_tags(text: str) -> str:
    """Strip Qwen3 <think>...</think> blocks from model output.

    Handles both closed tags and unclosed (dangling) <think> tags.
    """
    import re
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    if "<think>" in text:
        after_think = re.sub(r"<think>[\s\S]+", "", text)
        text = after_think.strip() if after_think.strip() else ""
    return text.strip()


def extract_json(text: str) -> dict[str, Any]:
    """从 LLM 输出中提取 JSON 对象，处理 markdown fences 和 think 标签。

    供 task_planner 和 llm_client 复用。
    """
    import re

    text = strip_think_tags(text)

    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try markdown code fence
    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Multiple JSON blocks separated by blank lines.
    # Qwen3 occasionally outputs two or more JSON objects back-to-back
    # (e.g. {"action":"cd",...}\n\n{"action":"stat",...}).  The greedy
    # regex below would collapse them into one blob, producing a parse
    # error.  Split on blank-line boundaries and try each block.
    #
    # We prefer the FIRST valid JSON block because decide_action only
    # expects one action per turn.
    blocks = re.split(r"\n\s*\n", text)
    if len(blocks) > 1:
        for block in blocks:
            block = block.strip()
            if not block.startswith("{"):
                continue
            try:
                return json.loads(block)
            except json.JSONDecodeError:
                pass

    # Try to find JSON object boundaries (greedy) and progressively
    # peel off trailing junk so we tolerate common LLM mistakes such as a
    # spurious closing brace at the end (e.g.,
    #   {"action": "tool_call", "arguments": {"k": "v"}}}
    # ).
    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        candidate = brace_match.group(0)
        # Try as-is first
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
        # Iteratively trim trailing characters until parse succeeds or we
        # run out of '}' to drop. Cap iterations to avoid pathological inputs.
        for _ in range(8):
            if not candidate.endswith("}"):
                break
            candidate = candidate[:-1].rstrip()
            if not candidate.endswith("}"):
                break
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue

    # All parsing strategies failed — return empty dict so callers
    # don't crash on .get() with AttributeError: 'NoneType'
    return {}

def iter_prompt_messages(records: list[dict]) -> Iterable[list[dict]]:
    """从 parquet records 中迭代 prompt 消息。

    parquet 里 prompt 列实际是 list<struct{role, content}>。
    保留对 JSON 字符串形式的兜底（旧数据兼容）。
    """
    import json as _json
    for r in records:
        p = r.get("prompt")
        if p is None:
            yield [{"role": "user", "content": ""}]
            continue
        if isinstance(p, list):
            yield list(p)
            continue
        if isinstance(p, str):
            try:
                yield _json.loads(p)
            except (ValueError, TypeError):
                yield [{"role": "user", "content": p}]
            continue
        yield [{"role": "user", "content": str(p)}]
