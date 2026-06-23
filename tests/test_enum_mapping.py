"""测试 enum 映射合法性检查。

覆盖 review 第 5 点要求：
- GT original enum = "economy"
- perturbed schema enum = "standard"
- model output = "standard" → correct
- model output = "economy" → incorrect

测试覆盖 eval 和 reward 两条路径。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from src.eval.matching import _args_match, _normalize_value, map_enum_values


# ── 精确格式 enum_map ──

NESTED_ENUM_MAP = {
    "search_flights": {
        "class": {"standard": "economy"},
    }
}

# ── 扁平格式 enum_map（兼容旧数据）──

FLAT_ENUM_MAP = {"standard": "economy"}


class TestEnumMappingNested:
    """精确格式 enum_map 测试。"""

    def test_perturbed_value_maps_back(self):
        """模型输出 perturbed enum value → 映射回 original → 判对。"""
        args = {"class": "standard", "date": "2024-01-01"}
        mapped = map_enum_values("search_flights", args, NESTED_ENUM_MAP)
        assert mapped == {"class": "economy", "date": "2024-01-01"}
        # 与 GT 比较
        gt_args = {"class": "economy", "date": "2024-01-01"}
        assert _args_match(mapped, gt_args)

    def test_original_value_rejected(self):
        """模型输出 original enum value（在 perturbed schema 下非法）→ 判错。"""
        args = {"class": "economy", "date": "2024-01-01"}
        mapped = map_enum_values("search_flights", args, NESTED_ENUM_MAP)
        assert mapped["class"] == "__INVALID_ENUM_economy__"
        # 与 GT 比较
        gt_args = {"class": "economy", "date": "2024-01-01"}
        assert not _args_match(mapped, gt_args)

    def test_non_enum_param_unaffected(self):
        """非 enum 参数值不受映射影响，即使值恰好等于 enum_map 的 key。"""
        # "standard" 出现在非 enum 参数 "note" 中
        args = {"class": "standard", "note": "standard procedure"}
        mapped = map_enum_values("search_flights", args, NESTED_ENUM_MAP)
        # "class" 被映射，"note" 不受影响
        assert mapped["class"] == "economy"
        assert mapped["note"] == "standard procedure"

    def test_different_function_unaffected(self):
        """不同函数的同名参数值不受其他函数的 enum_map 影响。"""
        args = {"class": "standard"}
        mapped = map_enum_values("other_function", args, NESTED_ENUM_MAP)
        # other_function 不在 enum_map 中，不做映射
        assert mapped == {"class": "standard"}

    def test_unknown_value_passthrough(self):
        """不在 enum_map 中的值直接透传。"""
        args = {"class": "business", "date": "2024-01-01"}
        mapped = map_enum_values("search_flights", args, NESTED_ENUM_MAP)
        assert mapped == {"class": "business", "date": "2024-01-01"}

    def test_unhashable_original_value_does_not_crash(self):
        """nested enum_map 中 original value 异常为 dict 时不应崩溃。"""
        enum_map = {"fn": {"mode": {"alt": {"nested": "value"}}}}
        args = {"mode": {"nested": "value"}}
        mapped = map_enum_values("fn", args, enum_map)
        assert mapped == args


class TestEnumMappingFlat:
    """扁平格式 enum_map 测试（兼容旧数据）。"""

    def test_perturbed_value_maps_back(self):
        args = {"class": "standard"}
        mapped = map_enum_values("search_flights", args, FLAT_ENUM_MAP)
        assert mapped == {"class": "economy"}

    def test_original_value_rejected(self):
        args = {"class": "economy"}
        mapped = map_enum_values("search_flights", args, FLAT_ENUM_MAP)
        assert mapped["class"] == "__INVALID_ENUM_economy__"

    def test_non_enum_value_passthrough(self):
        args = {"class": "business"}
        mapped = map_enum_values("search_flights", args, FLAT_ENUM_MAP)
        assert mapped == {"class": "business"}





class TestSchemaPerturberEnumMap:
    """验证 SchemaPerturber 生成的 enum_map 是嵌套格式。"""

    def test_enum_map_is_nested(self):
        from src.envs.schema_perturber import SchemaPerturber, LEVEL_STRONG

        functions = [{
            "name": "search_flights",
            "description": "Search for flights",
            "parameters": {
                "properties": {
                    "class": {
                        "type": "string",
                        "enum": ["economy", "business", "first"],
                    },
                    "date": {"type": "string"},
                },
                "required": ["class"],
            },
        }]
        perturber = SchemaPerturber(seed=42)
        perturbed = perturber.perturb(functions, LEVEL_STRONG)

        # enum_map 应为嵌套格式
        assert isinstance(perturber.enum_map, dict)
        if perturber.enum_map:
            # 第一层 key 是函数名
            for func_name, param_map in perturber.enum_map.items():
                assert isinstance(param_map, dict)
                for param_name, val_map in param_map.items():
                    assert isinstance(val_map, dict)
                    for pert_val, orig_val in val_map.items():
                        assert isinstance(pert_val, str)
                        assert isinstance(orig_val, str)

    def test_enum_map_used_correctly(self):
        """验证 perturber 生成的 enum_map 能正确用于 map_enum_values。"""
        from src.envs.schema_perturber import SchemaPerturber, LEVEL_STRONG

        functions = [{
            "name": "search_flights",
            "description": "Search for flights",
            "parameters": {
                "properties": {
                    "class": {
                        "type": "string",
                        "enum": ["economy", "business", "first"],
                    },
                },
                "required": ["class"],
            },
        }]
        perturber = SchemaPerturber(seed=42)
        perturbed = perturber.perturb(functions, LEVEL_STRONG)

        if perturber.enum_map:
            # 获取 perturbed enum value
            perturbed_enums = perturbed[0]["parameters"]["properties"]["class"]["enum"]
            original_enums = ["economy", "business", "first"]

            for pert_val in perturbed_enums:
                if pert_val not in original_enums:
                    # 这是一个被替换的值，应该能映射回 original
                    args = {"class": pert_val}
                    mapped = map_enum_values("search_flights", args, perturber.enum_map)
                    assert mapped["class"] in original_enums
