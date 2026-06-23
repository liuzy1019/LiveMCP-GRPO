"""
API 函数名映射层。

在 BFCL 的工具调用 dispatch 中插入别名映射，将扰动后的函数名映射回原始 API 函数名。
"""

import re
from typing import Optional
from loguru import logger


class FunctionNameMapper:
    """函数名 + 枚举值映射器。

    在 BFCL 的 execute_multi_turn_func_call dispatch 前插入，
    将扰动后的函数名和枚举值替换为原始值，确保工具调用正常执行。

    enum_map 支持两种格式：
    1. 精确格式（推荐）: {func_name: {param_name: {perturbed_val: original_val}}}
    2. 扁平格式（兼容旧数据）: {perturbed_val: original_val}

    Example:
        >>> mapper = FunctionNameMapper(
        ...     name_map={"find_flights": "search_flights"},
        ...     enum_map={"search_flights": {"class": {"standard": "economy"}}},
        ... )
        >>> mapper.resolve("find_flights")
        "search_flights"
    """

    def __init__(
        self,
        name_map: Optional[dict[str, str]] = None,
        enum_map: Optional[dict] = None,
    ):
        """
        Args:
            name_map: {perturbed_name: original_name} 映射字典。
            enum_map: 支持两种格式：
                - 精确: {func_name: {param_name: {perturbed_val: original_val}}}
                - 扁平: {perturbed_val: original_val}
        """
        self.name_map: dict[str, str] = name_map or {}
        self.enum_map: dict = enum_map or {}

        if self.name_map:
            logger.info(
                f"FunctionNameMapper 初始化，{len(self.name_map)} 条函数名映射"
            )
        if self.enum_map:
            logger.info(
                f"FunctionNameMapper 初始化，{len(self.enum_map)} 条枚举值映射"
            )

    def resolve(self, perturbed_name: str) -> str:
        """将扰动名解析为原始名。

        Args:
            perturbed_name: 模型输出的函数名。

        Returns:
            原始 API 函数名。如果不在映射表中，返回原值。
        """
        return self.name_map.get(perturbed_name, perturbed_name)

    def map_func_call(self, func_call_str: str) -> str:
        """将函数调用字符串中的扰动名和枚举值替换为原始值。

        注意：此方法用于 BFCL executor dispatch，做文本级替换。
        对于精确的参数级映射，请使用 src.eval.matching.map_enum_values。
        """
        # Step 1: 替换函数名
        for pert_name, orig_name in self.name_map.items():
            pattern = re.compile(
                r'\b' + re.escape(pert_name) + r'\s*(?=\()'
            )
            func_call_str = pattern.sub(orig_name, func_call_str)

        # Step 2: 替换枚举值（只替换字符串字面量中的值）
        # 收集所有 perturbed->original 映射（从两种格式中提取）
        flat_enum = self._flatten_enum_map()
        for pert_val, orig_val in flat_enum.items():
            func_call_str = re.sub(
                r"(')" + re.escape(pert_val) + r"(')",
                r"\1" + orig_val + r"\2",
                func_call_str,
            )
            func_call_str = re.sub(
                r'(")' + re.escape(pert_val) + r'(")',
                r"\1" + orig_val + r"\2",
                func_call_str,
            )

        return func_call_str

    def _flatten_enum_map(self) -> dict[str, str]:
        """将嵌套格式的 enum_map 展平为 {perturbed: original}。"""
        if not self.enum_map:
            return {}
        has_nested = any(isinstance(v, dict) for v in self.enum_map.values())
        if has_nested:
            # 精确格式: {func: {param: {pert: orig}}}
            flat = {}
            for func_enums in self.enum_map.values():
                if not isinstance(func_enums, dict):
                    continue
                for param_enums in func_enums.values():
                    if not isinstance(param_enums, dict):
                        continue
                    flat.update(param_enums)
            return flat
        else:
            # 已经是扁平格式
            return self.enum_map


def build_perturbed_ground_truth(
    ground_truth_list: list[str],
    name_map: dict[str, str],
    enum_map: Optional[dict] = None,
) -> list[str]:
    """将 ground truth 函数调用列表中的函数名和枚举值同步扰动。

    Args:
        ground_truth_list: 原始 ground truth 函数调用列表。
        name_map: {perturbed_name: original_name} 映射。反向使用。
        enum_map: 支持嵌套或扁平格式。反向使用。

    Returns:
        扰动后的 ground truth 列表。
    """
    reverse_name_map = {v: k for k, v in name_map.items()}

    # 反转 enum_map（original → perturbed）
    if enum_map:
        has_nested = any(isinstance(v, dict) for v in enum_map.values())
        if has_nested:
            # 嵌套格式 → 展平后反转
            flat = {}
            for func_enums in enum_map.values():
                if not isinstance(func_enums, dict):
                    continue
                for param_enums in func_enums.values():
                    if not isinstance(param_enums, dict):
                        continue
                    flat.update(param_enums)
            reverse_enum_map = {}
            for k, v in flat.items():
                try:
                    hash(v)
                    reverse_enum_map[v] = k
                except TypeError:
                    pass
        else:
            reverse_enum_map = {}
            for k, v in enum_map.items():
                try:
                    hash(v)
                    reverse_enum_map[v] = k
                except TypeError:
                    pass
    else:
        reverse_enum_map = {}

    mapper = FunctionNameMapper(name_map=reverse_name_map, enum_map=reverse_enum_map)
    perturbed = [mapper.map_func_call(gt) for gt in ground_truth_list]
    return perturbed
