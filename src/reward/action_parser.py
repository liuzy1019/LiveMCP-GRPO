"""Action Parser：解析模型输出为结构化 action。

支持 4 种 action type：
  - tool_call: <tool_call>{"name": ..., "arguments": ...}</tool_call>
  - final_answer: <final_answer>...</final_answer>
  - ask_clarification: <ask_clarification>...</ask_clarification>
  - report_error: <report_error>...</report_error>

模型输出 ask_clarification / report_error 时被判为 action-type mismatch，不是 format error。
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional


# 支持的 action types
ACTION_TYPES = ("tool_call", "final_answer", "ask_clarification", "report_error")

# 解析正则：匹配 <tag>content</tag> 格式
_TAG_PATTERN = re.compile(
    r"<(tool_call|final_answer|ask_clarification|report_error)>(.*?)</\1>",
    re.DOTALL,
)

# 备用：Qwen 风格的 tool_call 格式（无标签，直接 JSON）
_JSON_TOOL_CALL_PATTERN = re.compile(
    r'\{\s*"name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:',
    re.DOTALL,
)


@dataclass
class ParsedAction:
    """解析后的结构化 action。"""

    action_type: str  # "tool_call" / "final_answer" / "ask_clarification" / "report_error" / "unparseable"
    content: Any = None  # 解析后的内容
    raw_output: str = ""  # 原始模型输出
    parseable: bool = True  # 是否可解析（format reward 用）
    tool_calls: list[dict] = field(default_factory=list)  # 解析出的 tool_call 列表
    tool_name: Optional[str] = None  # 第一个 tool_call 的 name（快捷访问）
    arguments: Optional[dict] = None  # 第一个 tool_call 的 arguments（快捷访问）
    error_detail: str = ""  # 解析失败时的错误信息


class ActionParser:
    """解析模型输出为结构化 action。

    解析优先级：
    1. 标签格式：<tool_call>...</tool_call> 等
    2. Qwen 风格 JSON：直接输出 {"name": ..., "arguments": ...}
    3. 纯文本 final_answer：无标签但有实质内容
    4. unparseable：无法识别
    """

    def __init__(self, strict: bool = False):
        """
        Args:
            strict: 严格模式下只接受标签格式，不做 fallback 推断。
        """
        self.strict = strict

    @staticmethod
    def _normalize_tool_call(raw_call: dict) -> Optional[dict]:
        """统一归一化单个 tool call dict。

        规则：
        - name 必须存在且为字符串
        - arguments 非 dict 时强制置为空 dict 并标记 _args_was_invalid=True

        Returns:
            归一化后的 call dict，或 None（如果 name 缺失）。
        """
        if not isinstance(raw_call, dict) or "name" not in raw_call:
            return None
        name = raw_call["name"]
        if not isinstance(name, str) or not name.strip():
            return None
        raw_args = raw_call.get("arguments", {})
        args_invalid = not isinstance(raw_args, dict)
        return {
            "name": name,
            "arguments": {} if args_invalid else raw_args,
            "_args_was_invalid": args_invalid,
        }

    def parse(self, raw_output: str) -> ParsedAction:
        """解析模型输出。

        Args:
            raw_output: 模型生成的原始文本。

        Returns:
            ParsedAction 结构。
        """
        if not raw_output or not raw_output.strip():
            return ParsedAction(
                action_type="unparseable",
                raw_output=raw_output or "",
                parseable=False,
                error_detail="empty output",
            )

        # 1. 尝试标签格式
        result = self._parse_tagged(raw_output)
        if result is not None:
            return result

        # 严格模式下不做 fallback
        if self.strict:
            return ParsedAction(
                action_type="unparseable",
                raw_output=raw_output,
                parseable=False,
                error_detail="no recognized tag in strict mode",
            )

        # 2. 尝试 Qwen 风格 JSON tool_call
        result = self._parse_json_tool_call(raw_output)
        if result is not None:
            return result

        # 3. 尝试纯文本 final_answer（有实质内容且不像 tool_call）
        result = self._parse_plain_text_answer(raw_output)
        if result is not None:
            return result

        # 4. unparseable
        return ParsedAction(
            action_type="unparseable",
            raw_output=raw_output,
            parseable=False,
            error_detail="no recognized format",
        )

    def _parse_tagged(self, raw_output: str) -> Optional[ParsedAction]:
        """解析 <tag>content</tag> 格式。"""
        matches = list(_TAG_PATTERN.finditer(raw_output))
        if not matches:
            return None

        # 取第一个匹配的标签
        tag = matches[0].group(1)
        content_str = matches[0].group(2).strip()

        if tag == "tool_call":
            return self._build_tool_call_action(content_str, raw_output)
        elif tag == "final_answer":
            return ParsedAction(
                action_type="final_answer",
                content=content_str,
                raw_output=raw_output,
                parseable=True,
            )
        elif tag == "ask_clarification":
            return ParsedAction(
                action_type="ask_clarification",
                content=content_str,
                raw_output=raw_output,
                parseable=True,
            )
        elif tag == "report_error":
            return ParsedAction(
                action_type="report_error",
                content=content_str,
                raw_output=raw_output,
                parseable=True,
            )
        return None

    def _build_tool_call_action(self, content_str: str, raw_output: str) -> ParsedAction:
        """从 tool_call 标签内容构建 ParsedAction。

        支持单个 JSON 对象或 JSON 数组（并行调用）。
        """
        try:
            parsed = json.loads(content_str)
        except json.JSONDecodeError:
            # 尝试修复常见问题：多个 JSON 对象用换行分隔
            calls = self._try_parse_multiple_json(content_str)
            if calls:
                return ParsedAction(
                    action_type="tool_call",
                    content=calls,
                    raw_output=raw_output,
                    parseable=True,
                    tool_calls=calls,
                    tool_name=calls[0].get("name"),
                    arguments=calls[0].get("arguments", {}),
                )
            return ParsedAction(
                action_type="tool_call",
                raw_output=raw_output,
                parseable=False,
                error_detail=f"tool_call tag found but JSON parse failed",
            )

        # 单个 dict
        if isinstance(parsed, dict):
            calls = [parsed]
        # 数组
        elif isinstance(parsed, list):
            calls = parsed
        else:
            return ParsedAction(
                action_type="tool_call",
                raw_output=raw_output,
                parseable=False,
                error_detail="tool_call content is not dict or list",
            )

        # 验证每个 call 有 name，且 arguments 必须是 dict
        valid_calls = []
        for call in calls:
            normalized = self._normalize_tool_call(call)
            if normalized is not None:
                valid_calls.append(normalized)

        if not valid_calls:
            return ParsedAction(
                action_type="tool_call",
                raw_output=raw_output,
                parseable=False,
                error_detail="no valid tool_call with 'name' field",
            )

        return ParsedAction(
            action_type="tool_call",
            content=valid_calls,
            raw_output=raw_output,
            parseable=True,
            tool_calls=valid_calls,
            tool_name=valid_calls[0]["name"],
            arguments=valid_calls[0].get("arguments", {}),
        )

    def _parse_json_tool_call(self, raw_output: str) -> Optional[ParsedAction]:
        """尝试解析 Qwen 风格的直接 JSON tool_call。"""
        if not _JSON_TOOL_CALL_PATTERN.search(raw_output):
            return None

        # 尝试从输出中提取 JSON
        stripped = raw_output.strip()

        # 尝试整体解析
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, dict) and "name" in parsed:
                normalized = self._normalize_tool_call(parsed)
                if normalized:
                    calls = [normalized]
                    return ParsedAction(
                        action_type="tool_call",
                        content=calls,
                        raw_output=raw_output,
                        parseable=True,
                        tool_calls=calls,
                        tool_name=calls[0]["name"],
                        arguments=calls[0].get("arguments", {}),
                    )
            if isinstance(parsed, list):
                calls = []
                for c in parsed:
                    normalized = self._normalize_tool_call(c)
                    if normalized:
                        calls.append(normalized)
                if calls:
                    return ParsedAction(
                        action_type="tool_call",
                        content=calls,
                        raw_output=raw_output,
                        parseable=True,
                        tool_calls=calls,
                        tool_name=calls[0]["name"],
                        arguments=calls[0].get("arguments", {}),
                    )
        except json.JSONDecodeError:
            pass

        # 尝试提取第一个 JSON 对象
        calls = self._try_parse_multiple_json(stripped)
        if calls:
            return ParsedAction(
                action_type="tool_call",
                content=calls,
                raw_output=raw_output,
                parseable=True,
                tool_calls=calls,
                tool_name=calls[0]["name"],
                arguments=calls[0].get("arguments", {}),
            )

        return None

    def _parse_plain_text_answer(self, raw_output: str) -> Optional[ParsedAction]:
        """纯文本视为 final_answer（非严格模式下的 fallback）。"""
        stripped = raw_output.strip()
        # 至少有 10 个字符的实质内容
        if len(stripped) >= 10:
            return ParsedAction(
                action_type="final_answer",
                content=stripped,
                raw_output=raw_output,
                parseable=True,
            )
        return None

    def _try_parse_multiple_json(self, text: str) -> list[dict]:
        """尝试解析多个换行分隔的 JSON 对象。"""
        calls = []
        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                normalized = self._normalize_tool_call(obj)
                if normalized:
                    calls.append(normalized)
            except json.JSONDecodeError:
                continue
        return calls


# 模块级单例
_default_parser = ActionParser(strict=False)
_strict_parser = ActionParser(strict=True)


def parse_action(raw_output: str, strict: bool = False) -> ParsedAction:
    """便捷函数：解析模型输出。

    Args:
        raw_output: 模型生成的原始文本。
        strict: 是否使用严格模式（只接受标签格式）。

    Returns:
        ParsedAction 结构。
    """
    parser = _strict_parser if strict else _default_parser
    return parser.parse(raw_output)
