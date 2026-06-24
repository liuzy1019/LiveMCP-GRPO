"""ActionParser 回归测试。

覆盖 review 中指出的 fallback 路径对非 dict arguments 的处理。
"""

import pytest

from src.reward.action_parser import ActionParser, parse_action


class TestTaggedPathInvalidArguments:
    """标签路径下 arguments 非 dict 的处理。"""

    def test_string_arguments_tagged(self):
        """<tool_call> 内 arguments 为字符串 → parseable, _args_was_invalid=True。"""
        output = '<tool_call>{"name": "get_weather", "arguments": "city=Beijing"}</tool_call>'
        result = parse_action(output)
        assert result.action_type == "tool_call"
        assert result.parseable is True
        assert result.tool_calls[0]["arguments"] == {}
        assert result.tool_calls[0]["_args_was_invalid"] is True

    def test_list_arguments_tagged(self):
        """<tool_call> 内 arguments 为列表 → parseable, _args_was_invalid=True。"""
        output = '<tool_call>{"name": "get_weather", "arguments": ["city", "Beijing"]}</tool_call>'
        result = parse_action(output)
        assert result.action_type == "tool_call"
        assert result.parseable is True
        assert result.tool_calls[0]["arguments"] == {}
        assert result.tool_calls[0]["_args_was_invalid"] is True

    def test_null_arguments_tagged(self):
        """<tool_call> 内 arguments 为 null → parseable, _args_was_invalid=True。"""
        output = '<tool_call>{"name": "get_weather", "arguments": null}</tool_call>'
        result = parse_action(output)
        assert result.action_type == "tool_call"
        assert result.parseable is True
        assert result.tool_calls[0]["arguments"] == {}
        assert result.tool_calls[0]["_args_was_invalid"] is True

    def test_int_arguments_tagged(self):
        """<tool_call> 内 arguments 为数字 → parseable, _args_was_invalid=True。"""
        output = '<tool_call>{"name": "get_weather", "arguments": 42}</tool_call>'
        result = parse_action(output)
        assert result.action_type == "tool_call"
        assert result.parseable is True
        assert result.tool_calls[0]["arguments"] == {}
        assert result.tool_calls[0]["_args_was_invalid"] is True


class TestQwenDirectJsonInvalidArguments:
    """Qwen 风格直出 JSON 下 arguments 非 dict 的处理。"""

    def test_string_arguments_qwen_direct(self):
        """Qwen 直出 JSON，arguments 为字符串 → 不崩溃。"""
        output = '{"name": "get_weather", "arguments": "city=Beijing"}'
        result = parse_action(output)
        assert result.action_type == "tool_call"
        assert result.parseable is True
        assert result.tool_calls[0]["arguments"] == {}
        assert result.tool_calls[0]["_args_was_invalid"] is True

    def test_string_arguments_qwen_list(self):
        """Qwen 直出 JSON list，arguments 为字符串 → 不崩溃。"""
        output = '[{"name": "get_weather", "arguments": "city=Beijing"}, {"name": "get_time", "arguments": {"zone": "CST"}}]'
        result = parse_action(output)
        assert result.action_type == "tool_call"
        assert result.parseable is True
        assert len(result.tool_calls) == 2
        assert result.tool_calls[0]["_args_was_invalid"] is True
        assert result.tool_calls[1]["_args_was_invalid"] is False
        assert result.tool_calls[1]["arguments"] == {"zone": "CST"}


class TestMultiLineJsonInvalidArguments:
    """多行 JSON fallback 下 arguments 非 dict 的处理。"""

    def test_string_arguments_multiline(self):
        """tagged 内多行 JSON，arguments 为字符串 → 不崩溃。"""
        output = '\<tool_call\>\n{"name":"get_weather","arguments":"city=Beijing"}\n{"name":"get_time","arguments":{"zone":"CST"}}\n</tool_call>'
        result = parse_action(output)
        assert result.action_type == "tool_call"
        assert result.parseable is True
        assert len(result.tool_calls) == 2
        assert result.tool_calls[0]["_args_was_invalid"] is True
        assert result.tool_calls[1]["_args_was_invalid"] is False


class TestNormalizeToolCallHelper:
    """_normalize_tool_call 静态方法的边界测试。"""

    def test_missing_name(self):
        """无 name 字段 → None。"""
        assert ActionParser._normalize_tool_call({"arguments": {}}) is None

    def test_empty_name(self):
        """name 为空字符串 → None。"""
        assert ActionParser._normalize_tool_call({"name": "", "arguments": {}}) is None

    def test_non_dict_input(self):
        """输入不是 dict → None。"""
        assert ActionParser._normalize_tool_call("not a dict") is None
        assert ActionParser._normalize_tool_call(None) is None
        assert ActionParser._normalize_tool_call([]) is None

    def test_valid_call(self):
        """正常 call → 归一化后返回。"""
        result = ActionParser._normalize_tool_call({"name": "foo", "arguments": {"x": 1}})
        assert result == {"name": "foo", "arguments": {"x": 1}, "_args_was_invalid": False}

    def test_missing_arguments(self):
        """无 arguments 字段 → 默认空 dict。"""
        result = ActionParser._normalize_tool_call({"name": "foo"})
        assert result == {"name": "foo", "arguments": {}, "_args_was_invalid": False}


# End-to-end reward tests removed — component_reward module deleted as part of replay route cleanup.
# Oval reward tests live in tests/test_oval_mcp_components.py
