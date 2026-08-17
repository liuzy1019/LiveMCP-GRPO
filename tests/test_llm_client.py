from src.live_mcp.llm_client import _remaining_context_output_budget


def _context_error(context: int, output: int, prompt: int) -> RuntimeError:
    return RuntimeError(
        f"This model's maximum context length is {context} tokens. "
        f"However, you requested {output} output tokens and your prompt "
        f"contains at least {prompt} input tokens."
    )


def test_context_retry_reserves_server_framing_margin() -> None:
    assert _remaining_context_output_budget(
        _context_error(8192, 255, 7938), 256,
    ) == 246


def test_context_retry_fails_closed_when_json_budget_is_too_small() -> None:
    assert _remaining_context_output_budget(
        _context_error(8192, 255, 8121), 256,
    ) is None


def test_context_retry_ignores_unrelated_bad_request() -> None:
    assert _remaining_context_output_budget(
        RuntimeError("schema invalid"), 256,
    ) is None
