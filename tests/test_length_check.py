"""length_check 模块单元测试 + 真实 parquet 集成测试。"""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.training.length_check import (
    SKIP_ENV,
    LengthStats,
    _iter_prompt_messages,
    _percentile,
    check_split_length,
    parse_data_args_from_argv,
)


########## 纯逻辑测试（不依赖 tokenizer）


def test_percentile_basic():
    xs = sorted([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    assert _percentile(xs, 0.5) == 5
    assert _percentile(xs, 0.95) == 9
    assert _percentile(xs, 0.99) == 9
    assert _percentile([], 0.5) == 0


def test_iter_prompt_messages_handles_all_shapes():
    records = [
        {"prompt": [{"role": "user", "content": "hi"}]},
        {"prompt": json.dumps([{"role": "user", "content": "hi"}])},
        {"prompt": "raw string fallback"},
        {"prompt": None},
        {},  # 完全缺 prompt
    ]
    out = list(_iter_prompt_messages(records))
    assert len(out) == 5
    assert out[0][0]["content"] == "hi"
    assert out[1][0]["content"] == "hi"
    assert out[2][0]["role"] == "user"
    assert out[3][0]["content"] == ""
    assert out[4][0]["content"] == ""


def test_parse_data_args_from_argv():
    argv = [
        "data.train_files=[data/exp3/train.parquet]",
        "data.val_files=data/exp3/val.parquet",
        "actor_rollout_ref.model.path=/models/qwen",
        "data.max_prompt_length=10240",
        "data.foo=ignored",
    ]
    args = parse_data_args_from_argv(argv)
    assert args["train_files"] == "data/exp3/train.parquet"
    assert args["val_files"] == "data/exp3/val.parquet"
    assert args["model_path"] == "/models/qwen"
    assert args["max_prompt_length"] == 10240


def test_parse_data_args_partial():
    args = parse_data_args_from_argv(["unrelated=1"])
    assert args == {}


########## fail-fast / warn 行为测试（用 fake tokenizer 避免 HF 下载）


class _FakeTokenizer:
    """简易 tokenizer：返回 chars 数等长 token。"""

    def __init__(self, scale: int = 1):
        self._scale = scale
        self.pad_token_id = 0
        self.pad_token = "<pad>"

    def apply_chat_template(self, messages, add_generation_prompt=False):
        text = "".join(m.get("content", "") for m in messages)
        return [0] * (len(text) * self._scale)


@pytest.fixture
def tiny_parquet(tmp_path: Path):
    """造一个 5 行的 parquet，prompt 是 list<struct>。"""
    prompts = [
        [{"role": "user", "content": "x" * 100}],
        [{"role": "user", "content": "x" * 200}],
        [{"role": "user", "content": "x" * 500}],
        [{"role": "user", "content": "x" * 800}],
        [{"role": "user", "content": "x" * 1500}],  # 超长
    ]
    table = pa.table({"prompt": prompts})
    path = tmp_path / "tiny.parquet"
    pq.write_table(table, path)
    return path


def test_check_split_length_fails_when_overflow(tiny_parquet, monkeypatch):
    """max_prompt_length=1000，最长 1500 → 必须抛 RuntimeError。"""
    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained",
        lambda *_a, **_kw: _FakeTokenizer(),
    )
    with pytest.raises(RuntimeError, match="超过 max_prompt_length"):
        check_split_length(tiny_parquet, "fake/path", max_prompt_length=1000, split="train")


def test_check_split_length_passes_with_buffer(tiny_parquet, monkeypatch, caplog):
    """max_prompt_length=4096，全部安全 → 不抛。"""
    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained",
        lambda *_a, **_kw: _FakeTokenizer(),
    )
    stats = check_split_length(
        tiny_parquet, "fake/path", max_prompt_length=4096, split="train"
    )
    assert stats.n_overflow == 0
    assert stats.max_len == 1500


def test_check_split_length_warns_near_limit(tiny_parquet, monkeypatch, capsys):
    """max_prompt_length=1600，p99=1500 / 1600 = 93.75% 接近上限 → 不抛但应 warn。"""
    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained",
        lambda *_a, **_kw: _FakeTokenizer(),
    )
    # 这里 p99 在 5 行小样本里是 max（1500），1500 > 1600 * 0.95 = 1520 不成立
    # 调成更紧的 limit 让它落到 warn 区间
    stats = check_split_length(
        tiny_parquet, "fake/path", max_prompt_length=1550, split="train"
    )
    assert stats.n_overflow == 0
    assert stats.n_near_limit > 0


def test_skip_env_disables_check(monkeypatch):
    """SKIP_ENV=1 时跳过整个流程。"""
    from src.training.length_check import maybe_run_length_check

    monkeypatch.setenv(SKIP_ENV, "1")
    # 即使 argv 给的是不存在的 parquet 也不会报错
    maybe_run_length_check([
        "data.train_files=/nonexistent.parquet",
        "actor_rollout_ref.model.path=/nonexistent",
        "data.max_prompt_length=10240",
    ])


def test_lengthstats_format():
    stats = LengthStats(
        n_rows=100, max_len=500, p99=480, p95=400, p50=200,
        n_overflow=0, n_near_limit=2,
    )
    s = stats.format("train", limit=512)
    assert "rows=100" in s
    assert "max=500" in s
    assert "limit=512" in s
