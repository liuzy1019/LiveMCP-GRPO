"""
APIMapper 单元测试。
"""
import pytest
from src.envs.api_mapper import FunctionNameMapper, build_perturbed_ground_truth


class TestFunctionNameMapper:
    """FunctionNameMapper 功能测试。"""

    def test_init_empty(self):
        """空映射表初始化。"""
        mapper = FunctionNameMapper()
        assert len(mapper.name_map) == 0

    def test_init_with_map(self):
        """初始化时提供映射表。"""
        mapper = FunctionNameMapper({"find_flights": "search_flights"})
        assert mapper.name_map["find_flights"] == "search_flights"

    def test_resolve_perturbed(self):
        """映射应返回原始名。"""
        mapper = FunctionNameMapper({"find_flights": "search_flights"})
        assert mapper.resolve("find_flights") == "search_flights"

    def test_resolve_unmapped(self):
        """未映射的名应原样返回。"""
        mapper = FunctionNameMapper({"find_flights": "search_flights"})
        assert mapper.resolve("unknown_function") == "unknown_function"

    def test_map_func_call_simple(self):
        """映射函数调用字符串。"""
        mapper = FunctionNameMapper({"find_flights": "search_flights"})
        result = mapper.map_func_call("find_flights(origin='JFK')")
        assert result == "search_flights(origin='JFK')"

    def test_map_func_call_no_change(self):
        """无映射时不应改变字符串。"""
        mapper = FunctionNameMapper()
        result = mapper.map_func_call("search_flights(origin='JFK')")
        assert result == "search_flights(origin='JFK')"

    def test_map_func_call_multiple(self):
        """多个映射。"""
        mapper = FunctionNameMapper({
            "find_flights": "search_flights",
            "book_reservation": "book_flight",
        })
        result = mapper.map_func_call(
            "find_flights(origin='JFK'); book_reservation(flight_id='ABC')"
        )
        assert "search_flights" in result
        assert "book_flight" in result
        assert "find_flights" not in result
        assert "book_reservation" not in result

    def test_mixed_enum_map_ignores_non_nested_entries(self):
        """混合 enum_map 不应因只看第一个 value 而把 nested dict 当替换值。"""
        mapper = FunctionNameMapper(
            enum_map={
                "legacy": "old",
                "search_flights": {"class": {"standard": "economy"}},
            }
        )
        result = mapper.map_func_call("search_flights(class='standard')")
        assert result == "search_flights(class='economy')"

class TestBuildPerturbedGroundTruth:
    """build_perturbed_ground_truth 测试。"""

    def test_basic(self):
        """基本的 ground truth 扰动。

        输入 name_map = {perturbed: original} = {"find_flights": "search_flights"}
        build_perturbed_ground_truth 内部构建反向映射 {original: perturbed}
        → "search_flights" 被替换为 "find_flights"
        """
        name_map = {"find_flights": "search_flights"}
        gt_list = ["search_flights(origin='JFK')"]

        result = build_perturbed_ground_truth(gt_list, name_map)
        assert "find_flights" in result[0]  # original → perturbed

    def test_reverse_mapping(self):
        """反向映射应正常工作。"""
        # name_map = {perturbed: original}
        # build_perturbed_ground_truth 需要 {original: perturbed}
        name_map = {"find_flights": "search_flights"}
        gt_list = ["search_flights(origin='JFK')"]

        # 反向映射后
        result = build_perturbed_ground_truth(gt_list, name_map)
        # search_flights 是 original，应该被映射为 find_flights
        assert "find_flights" in result[0]

    def test_unhashable_enum_original_does_not_crash(self):
        """异常 enum_map 值为 dict 时，构建扰动 GT 不应崩溃。"""
        result = build_perturbed_ground_truth(
            ["search_flights(class='economy')"],
            name_map={},
            enum_map={"search_flights": {"class": {"standard": {"raw": "economy"}}}},
        )
        assert result == ["search_flights(class='economy')"]
