import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.training.length_check import _encoded_token_length, check_split_length


def test_encoded_token_length_supports_common_single_conversation_shapes() -> None:
    assert _encoded_token_length([1, 2, 3]) == 3
    assert _encoded_token_length([[1, 2, 3, 4]]) == 4
    assert _encoded_token_length(
        {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}
    ) == 3


@pytest.mark.parametrize(
    "encoded",
    [
        {"attention_mask": [1, 1]},
        [[1, 2], [3, 4]],
        "not-tokenized",
    ],
)
def test_encoded_token_length_rejects_ambiguous_outputs(encoded: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        _encoded_token_length(encoded)


def test_check_split_length_counts_batch_encoding_input_ids(
    tmp_path, monkeypatch
) -> None:
    class FakeTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert messages[0]["role"] == "user"
            assert kwargs == {"add_generation_prompt": True, "tokenize": True}
            return {
                "input_ids": list(range(17)),
                "attention_mask": [1] * 17,
            }

    monkeypatch.setattr(
        "verl.utils.tokenizer.hf_tokenizer", lambda _path: FakeTokenizer()
    )
    parquet_path = tmp_path / "one-row.parquet"
    prompt = json.dumps([{"role": "user", "content": "hello"}])
    pq.write_table(pa.table({"prompt": [prompt]}), parquet_path)

    stats = check_split_length(parquet_path, "unused", 32, "test")

    assert stats.n_rows == 1
    assert stats.max_len == 17
    assert stats.p50 == 17
    assert stats.n_overflow == 0


def test_check_split_length_fails_closed_on_real_token_overflow(
    tmp_path, monkeypatch
) -> None:
    class FakeTokenizer:
        def apply_chat_template(self, _messages, **_kwargs):
            return {"input_ids": list(range(17)), "attention_mask": [1] * 17}

    monkeypatch.setattr(
        "verl.utils.tokenizer.hf_tokenizer", lambda _path: FakeTokenizer()
    )
    parquet_path = tmp_path / "overflow.parquet"
    prompt = json.dumps([{"role": "user", "content": "hello"}])
    pq.write_table(pa.table({"prompt": [prompt]}), parquet_path)

    with pytest.raises(RuntimeError, match="1/1.*max_prompt_length=16"):
        check_split_length(parquet_path, "unused", 16, "test")
