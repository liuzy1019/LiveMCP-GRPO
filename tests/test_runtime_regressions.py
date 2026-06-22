"""Runtime regression coverage for data and BFCL integration paths.

These tests exercise project code paths that are easy to miss with pure formula
unit tests: parquet generation, BFCL call normalization, and reward matching.
"""

import json
import importlib.util
import sys
import tempfile
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

# scripts/ 不是 package，用 importlib 直接加载
_spec = importlib.util.spec_from_file_location(
    "build_parquet", _PROJECT_ROOT / "scripts" / "build_parquet.py"
)
_build_parquet = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_build_parquet)
prepare_exp2 = _build_parquet.prepare_exp2
prepare_exp5 = _build_parquet.prepare_exp5
from src.agent_loop.bfcl_agent_loop import BFCLAgentLoop, _parse_bfcl_native_args
from src.eval.bfcl_eval import _run_inference_single
from src.reward.bfcl_reward import compute_bfcl_reward
from src.eval.bfcl_eval import _evaluate_per_turn


def test_parser_supports_dotted_names_and_positional_arguments():
    name, args = _parse_bfcl_native_args("math.factorial(number=5)")
    assert name == "math.factorial"
    # 数字字面量应还原为 Python 原生类型（修复 BFCL native 解析与 JSON 输出类型不一致 bug）
    assert args == {"number": 5}

    name, args = _parse_bfcl_native_args("sort('final_report.pdf')")
    assert name == "sort"
    assert args == {"_pos_0": "final_report.pdf"}


def test_parser_recovers_native_types_for_list_dict_bool():
    """list/dict/bool 字面量应还原为 Python 原生类型，而非字符串。

    回归 P1 #4: 此前 'mean(numbers=[3,16,60])' 被解析成 {'numbers': '[3,16,60]'}，
    导致与模型 JSON 输出 {'numbers': [3,16,60]} 类型不匹配，reward 误判 0。
    """
    name, args = _parse_bfcl_native_args("mean(numbers=[3,16,60])")
    assert name == "mean"
    assert args == {"numbers": [3, 16, 60]}

    name, args = _parse_bfcl_native_args("config(opts={'a': 1})")
    assert name == "config"
    assert args == {"opts": {"a": 1}}

    name, args = _parse_bfcl_native_args("toggle(flag=True)")
    assert name == "toggle"
    assert args == {"flag": True}


def test_parser_rejects_oversized_input_within_milliseconds():
    """超长 garbled 输入必须立即降级为 ("", {})，不进入 O(N²) 慢路径。

    回归 P0 #2026-06-16 17:10 卡死事件：smoke train step 1 时 3 个 AgentLoopWorker
    的 active+gil 全停在 _parse_bfcl_native_args 反复调用 args_part.find("=", i)，
    GPU util 跌到 0% 但进程没死。修复要求：长度上限 + 单遍线性扫描 + 毫秒级返回。
    """
    import time

    # case 1: 长度超过 _PARSE_MAX_INPUT_LEN，必须立即返回空
    huge = "f(" + ("a" * 20000) + ")"
    t0 = time.perf_counter()
    name, args = _parse_bfcl_native_args(huge)
    elapsed = time.perf_counter() - t0
    assert (name, args) == ("", {})
    assert elapsed < 0.05, f"oversized input took {elapsed:.3f}s, expected <0.05s"

    # case 2: 长度刚好在上限内但塞满 '=' 触发原 O(N²) 路径
    # 旧实现这里会跑数十秒；新实现 < 50ms
    payload = "x=" * 4000
    crafted = f"f({payload[:6000]})"
    t0 = time.perf_counter()
    _parse_bfcl_native_args(crafted)
    elapsed = time.perf_counter() - t0
    assert elapsed < 0.05, f"dense '=' input took {elapsed:.3f}s, expected <0.05s"


def test_parser_handles_malformed_calls_gracefully():
    """各类畸形输入：未闭合括号、多余引号、空参数、纯垃圾，均不应抛异常或卡死。"""
    bad_inputs = [
        "",                                  # 空字符串
        "no_paren",                          # 没有括号
        "f(",                                # 未闭合
        "f(a=)",                             # 等号后空
        "f(=value)",                         # 等号前无 key
        "f('unclosed string",                # 未闭合字符串
        "f([1, 2, 3",                        # 未闭合 list
        "f({'a': 1",                         # 未闭合 dict
        "f(\\)",                             # 反斜杠
        "f(a==b)",                           # 双等号
        "f(),garbage trailing,))",           # 尾部垃圾
        "  f  (  ) ",                        # 大量空白
    ]
    for s in bad_inputs:
        # 只要不抛异常、不卡死，name 是 str、args 是 dict 即可
        name, args = _parse_bfcl_native_args(s)
        assert isinstance(name, str)
        assert isinstance(args, dict)


def test_parser_caps_argument_count():
    """段数超过上限应直接降级，不无限累积。"""
    # 80 个参数（上限 64）
    args_list = ",".join(f"k{i}={i}" for i in range(80))
    name, args = _parse_bfcl_native_args(f"f({args_list})")
    # 上限触发时返回 (name, {})；调用方会因 args 不完整产生 reward miss，但不会卡 rollout
    assert name == "f"
    assert isinstance(args, dict)


def test_parser_preserves_top_level_comma_inside_nested():
    """嵌套结构内的 ',' 不能被当作段分隔符。"""
    name, args = _parse_bfcl_native_args(
        "send(items=[1, 2, 3], opts={'k1': 'a, b', 'k2': 2}, flag=True)"
    )
    assert name == "send"
    assert args == {"items": [1, 2, 3], "opts": {"k1": "a, b", "k2": 2}, "flag": True}




def test_exp4_parquet_keeps_task_groups_adjacent():
    pq = pytest.importorskip("pyarrow.parquet")

    table = pq.read_table("data/verl/exp4_schemashift/train.parquet")
    rows = table.to_pylist()
    assert len(rows) == 8100

    for start in range(0, len(rows), 9):
        chunk = rows[start:start + 9]
        assert len({row["task_id"] for row in chunk}) == 1
        assert [row["perturbation_level"] for row in chunk] == (
            ["none"] * 3 + ["mild"] * 3 + ["strong"] * 3
        )


def test_prepare_exp5_rebuilds_aug_only_from_exp4_parquet():
    pq = pytest.importorskip("pyarrow.parquet")

    with tempfile.TemporaryDirectory() as tmpdir:
        prepare_exp5("data/verl/exp4_schemashift", tmpdir)
        train_rows = pq.read_table(f"{tmpdir}/train.parquet").to_pylist()
        val_rows = pq.read_table(f"{tmpdir}/val.parquet").to_pylist()

    assert len(train_rows) == 2700
    assert len(val_rows) == 300

    for rows in (train_rows, val_rows):
        grouped = {}
        for row in rows:
            grouped.setdefault(row["task_id"], []).append(row["perturbation_level"])
        for levels in grouped.values():
            assert sorted(levels) == ["mild", "none", "strong"]


def test_prepare_exp2_targets_match_prompt_tool_call_contract():
    pq = pytest.importorskip("pyarrow.parquet")

    with tempfile.TemporaryDirectory() as tmpdir:
        prepare_exp2("data", tmpdir)
        row = pq.read_table(f"{tmpdir}/train.parquet").slice(0, 1).to_pylist()[0]

    first_line = row["target"].splitlines()[0]
    assert first_line.startswith("<tool_call>")
    assert first_line.endswith("</tool_call>")
    payload = first_line.removeprefix("<tool_call>").removesuffix("</tool_call>")
    parsed = json.loads(payload)
    assert set(parsed) == {"name", "arguments"}
    assert isinstance(parsed["name"], str)
    assert isinstance(parsed["arguments"], dict)


def test_reward_rejects_extra_tool_calls_in_same_turn():
    reward = compute_bfcl_reward(
        func_calls=[],
        ground_truth_json=json.dumps([["a(x=1)"]]),
        turn_func_calls=[
            [
                {"name": "bad_call", "arguments": {}},
                {"name": "a", "arguments": {"x": "1"}},
            ]
        ],
    )
    assert reward == 0.0


def test_reward_accepts_empty_agent_turn_when_gt_turn_is_empty():
    """GT 含空轮次（miss_func/miss_param）：agent 在该轮不调用工具应判对。

    回归 P0: agent loop 之前不把空轮 append 进 turn_func_calls，
    导致 GT [[a()], [], [b()]] 与 agent 实际 [[a()], [], [b()]] 长度对不上
    或错位匹配，让"正确不调用"的轨迹被 reward 判 0。
    """
    reward = compute_bfcl_reward(
        func_calls=[],
        ground_truth_json=json.dumps([["a(x=1)"], [], ["b(y=2)"]]),
        turn_func_calls=[
            [{"name": "a", "arguments": {"x": 1}}],
            [],
            [{"name": "b", "arguments": {"y": 2}}],
        ],
    )
    assert reward == 1.0


def test_reward_rejects_call_when_gt_turn_is_empty():
    """GT 该轮要求不调用工具，但 agent 调了工具：reward 必须 0。"""
    reward = compute_bfcl_reward(
        func_calls=[],
        ground_truth_json=json.dumps([["a(x=1)"], [], ["b(y=2)"]]),
        turn_func_calls=[
            [{"name": "a", "arguments": {"x": 1}}],
            [{"name": "unexpected", "arguments": {}}],
            [{"name": "b", "arguments": {"y": 2}}],
        ],
    )
    assert reward == 0.0


def test_reward_rejects_extra_nonempty_turn_after_gt_ends():
    """GT 已结束、agent 又发起一轮非空工具调用：reward 必须 0。

    回归 codex review (2026-06-15) 指出的 P0：旧实现只校验
    len(turn_func_calls) < len(gt) 导致尾部多余 tool call 被忽略。
    """
    reward = compute_bfcl_reward(
        func_calls=[],
        ground_truth_json=json.dumps([["a(x=1)"]]),
        turn_func_calls=[
            [{"name": "a", "arguments": {"x": 1}}],
            [{"name": "extra", "arguments": {}}],
        ],
    )
    assert reward == 0.0


def test_reward_allows_trailing_empty_turn_after_gt_ends():
    """GT 已结束、agent 多走一轮但未调用工具：reward 应判 1。

    agent loop 在没有后续 user turn 时会先 append 一个空 turn 再 break，
    因此尾部空 turn 是 agent loop 的正常行为，不能被误判为错。
    """
    reward = compute_bfcl_reward(
        func_calls=[],
        ground_truth_json=json.dumps([["a(x=1)"]]),
        turn_func_calls=[
            [{"name": "a", "arguments": {"x": 1}}],
            [],
        ],
    )
    assert reward == 1.0


def test_eval_per_turn_rejects_extra_nonempty_turn_after_gt_ends():
    """offline eval 与 reward 必须保持一致：GT 之后多调用一轮 tool 应判 False。"""
    gt = [["a(x=1)"]]
    turn_outputs = [
        [{"name": "a", "arguments": {"x": 1}}],
        [{"name": "extra", "arguments": {}}],
    ]
    assert _evaluate_per_turn(turn_outputs, gt) is False


def test_eval_per_turn_allows_trailing_empty_turn_after_gt_ends():
    """offline eval 与 reward 一致：GT 之后的尾部空 turn 应判 True。"""
    gt = [["a(x=1)"]]
    turn_outputs = [
        [{"name": "a", "arguments": {"x": 1}}],
        [],
    ]
    assert _evaluate_per_turn(turn_outputs, gt) is True


def test_format_for_bfcl_does_not_emit_internal_positional_keys():
    formatted = BFCLAgentLoop._format_for_bfcl("sort", {"_pos_0": "final_report.pdf"})
    assert "_pos_0" not in formatted


def test_bare_json_tool_call_parser_keeps_nested_arguments_intact():
    call = '{"name":"a","arguments":{"x":{"nested":1}}}'
    assert BFCLAgentLoop._parse_tool_calls(call) == [call]


def test_eval_inference_maps_perturbed_calls_before_scoring():
    class FakeTokenizer:
        def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False):
            return "prompt"

        def encode(self, text):
            return list(range(len(text)))

    class FakeLLM:
        def generate(self, prompts, sampling_params):
            output = type(
                "Output",
                (),
                {
                    "outputs": [
                        type(
                            "Candidate",
                            (),
                            {
                                "text": (
                                    '\<tool_call\>{"name": "find_flights", '
                                    '"arguments": {"class": "standard"}}</tool_call>'
                                )
                            },
                        )()
                    ]
                },
            )
            return [output()]

    record = {
        "prompt": '[{"role": "user", "content": "x"}]',
        "name_map_json": json.dumps({"find_flights": "search_flights"}),
        "enum_map_json": json.dumps({"standard": "economy"}),
    }

    assert _run_inference_single(FakeLLM(), FakeTokenizer(), record, object(), max_turns=1) == [
        {"name": "search_flights", "arguments": {"class": "economy"}}
    ]
