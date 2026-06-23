"""Phase 1 数据构建模块测试。

覆盖：
  - DistractorSampler: 工具池构建、采样逻辑、领域匹配、同强度扰动
  - ConditionedDecisionBuilder: 从真实数据加载 conditioned decision
"""

import json
from pathlib import Path

import pytest

from src.data.distractor_sampler import DistractorSampler, DistractorConfig, ToolEntry
from src.data.conditioned_builder import ConditionedDecisionBuilder, ConditionedBuilderConfig, NoToolSample
from src.data.episode_seed_builder import BuilderConfig, EpisodeSeedBuilder
from scripts.prepare_grpo_data import _stable_seed


# ============================================================
# 测试数据 fixtures
# ============================================================

def _make_tool(name: str, description: str = "", params: dict = None) -> dict:
    """构造一个简单的 tool schema。"""
    return {
        "name": name,
        "description": description or f"Tool {name}",
        "parameters": {
            "type": "object",
            "properties": params or {"query": {"type": "string", "description": "input"}},
            "required": list((params or {"query": {}}).keys()),
        },
    }


def _make_decision_step(
    task_id: str,
    tools: list[dict],
    gt_calls: list[dict],
    messages: list[dict] = None,
    step_index: int = 0,
    total_steps: int = 1,
) -> dict:
    """构造一条 decision step。"""
    return {
        "task_id": task_id,
        "tools": tools,
        "messages": messages or [{"role": "user", "content": "Help me"}],
        "ground_truth_calls": gt_calls,
        "num_tools": len(tools),
        "num_prior_turns": len(messages or []),
        "step_index": step_index,
        "total_steps_in_sample": total_steps,
    }


def _make_conditioned_step(
    task_id: str,
    tools: list[dict],
    messages: list[dict],
    action_type: str = "final_answer",
    ground_truth_action: dict = None,
) -> dict:
    """构造一条 conditioned step。"""
    if ground_truth_action is None:
        if action_type == "final_answer":
            ground_truth_action = {"type": "final_answer", "content": "The result is 42."}
        else:
            ground_truth_action = {"type": "tool_call", "tool_calls": [{"name": tools[0]["name"], "arguments": {}}]}
    return {
        "task_id": task_id,
        "source": "toolace",
        "tools": tools,
        "messages": messages,
        "ground_truth_action": ground_truth_action,
        "action_type": action_type,
        "provenance": "real",
        "scenario_type": "final_answer" if action_type == "final_answer" else "conditioned_tool_call",
        "step_index": 1,
    }


def _make_no_tool_step(
    task_id: str,
    tools: list[dict],
    messages: list[dict],
    action_type: str = "ask_clarification",
    content: str = "Please provide the API key.",
) -> dict:
    """构造一条 no-tool step。"""
    return {
        "task_id": task_id,
        "source": "toolace",
        "tools": tools,
        "messages": messages,
        "ground_truth_action": {
            "type": "ask_clarification" if action_type == "ask_clarification" else "final_answer",
            "content": content,
            "no_tool_subtype": action_type,
        },
        "action_type": "ask_clarification" if action_type == "ask_clarification" else "final_answer",
        "no_tool_subtype": action_type,
        "provenance": "real",
        "scenario_type": "no_tool",
        "step_index": 0,
    }


def test_prepare_grpo_stable_seed_is_process_independent():
    """prepare_grpo_data 不应依赖 Python 内置 hash() 的进程随机盐。"""
    assert _stable_seed("episode-1:none") == _stable_seed("episode-1:none")
    assert _stable_seed("episode-1:none") != _stable_seed("episode-1:mild")


def test_episode_seed_builder_empty_input_does_not_divide_by_zero(tmp_path):
    """空 Toucan 文件应返回空列表，而不是在成功率日志处除零。"""
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")

    builder = EpisodeSeedBuilder(
        BuilderConfig(toucan_data_path=str(empty), output_path=str(tmp_path / "out.jsonl"))
    )
    assert builder.build() == []


@pytest.fixture
def sample_data_file(tmp_path):
    """创建一个临时的 decision_steps.jsonl 文件。"""
    data_path = tmp_path / "decision_steps.jsonl"

    tools_finance = [
        _make_tool("get_stock_price", "Get current stock price", {"symbol": {"type": "string"}}),
        _make_tool("get_market_index", "Get market index data", {"index": {"type": "string"}}),
        _make_tool("calculate_returns", "Calculate investment returns", {"amount": {"type": "number"}, "rate": {"type": "number"}}),
    ]

    tools_weather = [
        _make_tool("get_weather", "Get weather forecast", {"city": {"type": "string"}}),
        _make_tool("get_temperature", "Get temperature", {"location": {"type": "string"}}),
    ]

    tools_search = [
        _make_tool("search_web", "Search the web", {"query": {"type": "string"}}),
        _make_tool("find_documents", "Find documents", {"keywords": {"type": "string"}}),
        _make_tool("lookup_info", "Lookup information", {"topic": {"type": "string"}}),
    ]

    steps = [
        # 单步 finance 样本
        _make_decision_step(
            "toolace_00001_step_0", tools_finance,
            [{"name": "get_stock_price", "arguments": {"symbol": "AAPL"}}],
        ),
        # 单步 weather 样本
        _make_decision_step(
            "toolace_00002_step_0", tools_weather,
            [{"name": "get_weather", "arguments": {"city": "Beijing"}}],
        ),
        # 多步 search 样本 - step 0
        _make_decision_step(
            "toolace_00003_step_0", tools_search,
            [{"name": "search_web", "arguments": {"query": "AI papers"}}],
            step_index=0, total_steps=2,
        ),
        # 多步 search 样本 - step 1（有 tool_output history）
        _make_decision_step(
            "toolace_00003_step_1", tools_search,
            [{"name": "find_documents", "arguments": {"keywords": "transformer"}}],
            messages=[
                {"role": "user", "content": "Find AI papers about transformers"},
                {"role": "assistant", "content": "[search_web(query=\"AI papers\")]"},
                {"role": "tool", "content": '{"results": [{"title": "Attention is All You Need"}]}'},
            ],
            step_index=1, total_steps=2,
        ),
        # 更多单步样本用于 distractor 测试
        _make_decision_step(
            "toolace_00004_step_0", tools_finance,
            [{"name": "calculate_returns", "arguments": {"amount": 1000, "rate": 0.05}}],
        ),
        _make_decision_step(
            "toolace_00005_step_0", tools_search,
            [{"name": "lookup_info", "arguments": {"topic": "machine learning"}}],
        ),
    ]

    with open(data_path, "w") as f:
        for step in steps:
            f.write(json.dumps(step, ensure_ascii=False) + "\n")

    return data_path


@pytest.fixture
def conditioned_data_file(tmp_path):
    """创建临时的 conditioned_steps.jsonl 文件。"""
    data_path = tmp_path / "conditioned_steps.jsonl"

    tools = [
        _make_tool("search_web", "Search the web", {"query": {"type": "string"}}),
        _make_tool("find_documents", "Find documents", {"keywords": {"type": "string"}}),
    ]

    steps = [
        # 真实 final_answer（tool_output 后给出最终回答）
        _make_conditioned_step(
            "toolace_00010_final_3",
            tools=tools,
            messages=[
                {"role": "user", "content": "Find papers about transformers"},
                {"role": "assistant", "content": "[search_web(query=\"transformers\")]"},
                {"role": "tool", "content": '{"results": [{"title": "Attention is All You Need", "year": 2017}]}'},
            ],
            action_type="final_answer",
            ground_truth_action={
                "type": "final_answer",
                "content": "I found a paper: 'Attention is All You Need' (2017).",
            },
        ),
        # 真实 conditioned_tool_call（tool_output 后继续调用）
        _make_conditioned_step(
            "toolace_00011_conditioned",
            tools=tools,
            messages=[
                {"role": "user", "content": "Find and download transformer papers"},
                {"role": "assistant", "content": "[search_web(query=\"transformer papers\")]"},
                {"role": "tool", "content": '{"results": [{"title": "BERT", "id": "123"}]}'},
            ],
            action_type="conditioned_tool_call",
            ground_truth_action={
                "type": "tool_call",
                "tool_calls": [{"name": "find_documents", "arguments": {"keywords": "BERT"}}],
            },
        ),
    ]

    with open(data_path, "w") as f:
        for step in steps:
            f.write(json.dumps(step, ensure_ascii=False) + "\n")

    return data_path


@pytest.fixture
def no_tool_data_file(tmp_path):
    """创建临时的 no_tool_steps.jsonl 文件。"""
    data_path = tmp_path / "no_tool_steps.jsonl"

    tools = [
        _make_tool("get_stock_price", "Get stock price", {"symbol": {"type": "string"}}),
    ]

    steps = [
        _make_no_tool_step(
            "toolace_00020_notool_1",
            tools=tools,
            messages=[{"role": "user", "content": "What is the stock price?"}],
            action_type="ask_clarification",
            content="Please provide the stock symbol you'd like to look up.",
        ),
        _make_no_tool_step(
            "toolace_00021_notool_1",
            tools=tools,
            messages=[{"role": "user", "content": "What is a stock?"}],
            action_type="no_tool_needed",
            content="A stock represents ownership in a company.",
        ),
    ]

    with open(data_path, "w") as f:
        for step in steps:
            f.write(json.dumps(step, ensure_ascii=False) + "\n")

    return data_path


# ============================================================
# DistractorSampler 测试
# ============================================================


class TestDistractorSampler:
    """Distractor Sampler 测试。"""

    def test_build_pool(self, sample_data_file):
        """工具池构建。"""
        sampler = DistractorSampler.from_decision_steps(sample_data_file)
        stats = sampler.get_pool_stats()
        assert stats["total_tools"] > 0
        assert len(stats["domains"]) > 0

    def test_sample_basic(self, sample_data_file):
        """基本采样功能。"""
        sampler = DistractorSampler.from_decision_steps(
            sample_data_file,
            config=DistractorConfig(min_distractors=2, max_distractors=3, seed=42),
        )
        distractors = sampler.sample(gt_tool_names=["get_weather"], num_distractors=2)
        assert len(distractors) <= 2
        # 不应包含 GT 工具
        distractor_names = [d["name"] for d in distractors]
        assert "get_weather" not in distractor_names

    def test_sample_excludes_gt(self, sample_data_file):
        """采样排除 GT 工具。"""
        sampler = DistractorSampler.from_decision_steps(sample_data_file)
        distractors = sampler.sample(
            gt_tool_names=["get_stock_price", "get_market_index"],
            num_distractors=5,
        )
        distractor_names = {d["name"] for d in distractors}
        assert "get_stock_price" not in distractor_names
        assert "get_market_index" not in distractor_names

    def test_sample_with_metadata(self, sample_data_file):
        """采样带元数据。"""
        sampler = DistractorSampler.from_decision_steps(sample_data_file)
        schemas, metadata = sampler.sample_with_metadata(
            gt_tool_names=["get_weather"],
            num_distractors=2,
        )
        assert "num_candidates" in metadata
        assert "distractor_names" in metadata
        assert "domain_overlap_ratio" in metadata
        assert len(schemas) == len(metadata["distractor_names"])

    def test_domain_preference(self, sample_data_file):
        """同领域工具优先被采样。"""
        config = DistractorConfig(domain_weight=0.9, param_weight=0.05, name_weight=0.05, seed=42)
        sampler = DistractorSampler.from_decision_steps(sample_data_file, config=config)

        # finance 工具应优先采样 finance 领域的 distractor
        _, metadata = sampler.sample_with_metadata(
            gt_tool_names=["get_stock_price"],
            num_distractors=2,
        )
        # 至少有一些候选
        assert metadata["num_candidates"] > 0

    def test_empty_pool_handling(self):
        """空工具池处理。"""
        sampler = DistractorSampler()
        sampler._built = True  # 模拟已构建但为空
        result = sampler.sample(gt_tool_names=["nonexistent"], num_distractors=3)
        assert result == []

    def test_sample_with_perturbation_none(self, sample_data_file):
        """同强度扰动 - none 级别（不扰动）。"""
        sampler = DistractorSampler.from_decision_steps(sample_data_file)
        distractors, info = sampler.sample_with_perturbation(
            gt_tool_names=["get_weather"],
            perturbation_level="none",
            num_distractors=2,
        )
        assert len(distractors) <= 2
        assert info["level"] == "none"
        assert info["name_map"] == {}

    def test_sample_with_perturbation_mild(self, sample_data_file):
        """同强度扰动 - mild 级别。"""
        sampler = DistractorSampler.from_decision_steps(sample_data_file)
        distractors, info = sampler.sample_with_perturbation(
            gt_tool_names=["get_weather"],
            perturbation_level="mild",
            num_distractors=2,
        )
        # 应该有结果（可能扰动成功也可能回退）
        assert isinstance(distractors, list)
        assert "level" in info


# ============================================================
# ConditionedDecisionBuilder 测试
# ============================================================


class TestConditionedDecisionBuilder:
    """Conditioned Decision Builder 测试。"""

    def test_load_real_conditioned(self, conditioned_data_file, no_tool_data_file):
        """从真实数据加载 conditioned decisions。"""
        config = ConditionedBuilderConfig(
            conditioned_steps_path=str(conditioned_data_file),
            no_tool_steps_path=str(no_tool_data_file),
            include_no_tool=False,
        )
        builder = ConditionedDecisionBuilder(config=config)
        conditioned, no_tool = builder.build()

        # 应该有 2 条 conditioned samples
        assert len(conditioned) == 2
        assert len(no_tool) == 0  # include_no_tool=False

        # 检查类型
        final_answers = [s for s in conditioned if s.action_type == "final_answer"]
        tool_calls = [
            s for s in conditioned
            if s.action_type == "tool_call" and s.scenario_type == "conditioned_tool_call"
        ]
        assert len(final_answers) == 1
        assert len(tool_calls) == 1

    def test_load_real_no_tool(self, conditioned_data_file, no_tool_data_file):
        """从真实数据加载 no-tool samples。"""
        config = ConditionedBuilderConfig(
            conditioned_steps_path=str(conditioned_data_file),
            no_tool_steps_path=str(no_tool_data_file),
            include_no_tool=True,
        )
        builder = ConditionedDecisionBuilder(config=config)
        conditioned, no_tool = builder.build()

        assert len(conditioned) == 2
        assert len(no_tool) == 2

        # 检查 no_tool 类型
        ask_clar = [s for s in no_tool if s.action_type == "ask_clarification"]
        no_needed = [s for s in no_tool if s.no_tool_subtype == "no_tool_needed"]
        assert len(ask_clar) == 1
        assert len(no_needed) == 1
        assert no_needed[0].action_type == "final_answer"
        assert no_needed[0].ground_truth_action["type"] == "final_answer"

    def test_conditioned_messages_have_tool_output(self, conditioned_data_file, no_tool_data_file):
        """Conditioned samples 的 messages 包含 tool_output。"""
        config = ConditionedBuilderConfig(
            conditioned_steps_path=str(conditioned_data_file),
            no_tool_steps_path=str(no_tool_data_file),
        )
        builder = ConditionedDecisionBuilder(config=config)
        conditioned, _ = builder.build()

        for s in conditioned:
            has_tool = any(m["role"] == "tool" for m in s.messages)
            assert has_tool, f"conditioned sample missing tool_output: {s.task_id}"

    def test_final_answer_has_real_content(self, conditioned_data_file, no_tool_data_file):
        """Final answer oracle 包含真实内容（非模板）。"""
        config = ConditionedBuilderConfig(
            conditioned_steps_path=str(conditioned_data_file),
            no_tool_steps_path=str(no_tool_data_file),
        )
        builder = ConditionedDecisionBuilder(config=config)
        conditioned, _ = builder.build()

        final_answers = [s for s in conditioned if s.action_type == "final_answer"]
        for s in final_answers:
            content = s.ground_truth_action.get("content", "")
            # 真实 final_answer 应引用 tool_output 中的实体
            assert "Attention is All You Need" in content or len(content) > 20

    def test_backward_compat_build_from_decision_steps(self, sample_data_file):
        """兼容旧接口 build_from_decision_steps。"""
        builder = ConditionedDecisionBuilder()
        # 当 conditioned_steps.jsonl 不存在时，从 decision_steps 提取
        samples = builder.build_from_decision_steps(sample_data_file)
        # toolace_00003 有 2 步，step_1 有 tool_output → 应提取 1 条
        real_samples = [s for s in samples if s.provenance == "real"]
        assert len(real_samples) >= 1

    def test_stats_tracking(self, conditioned_data_file, no_tool_data_file):
        """统计追踪。"""
        config = ConditionedBuilderConfig(
            conditioned_steps_path=str(conditioned_data_file),
            no_tool_steps_path=str(no_tool_data_file),
            include_no_tool=True,
        )
        builder = ConditionedDecisionBuilder(config=config)
        builder.build()
        stats = builder.get_stats()
        assert stats["conditioned_tool_call"] == 1
        assert stats["conditioned_final_answer"] == 1
        assert stats["no_tool_ask_clarification"] == 1
        assert stats["no_tool_no_needed"] == 1
