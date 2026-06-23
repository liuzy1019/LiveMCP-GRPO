"""
共享工具函数：extra_info / metadata 的 JSON-string normalization。

verl/pyarrow 序列化后，extra_info 及其嵌套字段可能变为 JSON 字符串。
本模块提供统一的 normalize 函数，供 reward、replay loop、register_estimator 复用。
"""

import json
from typing import Any


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
