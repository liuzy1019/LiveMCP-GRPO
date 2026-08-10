"""Regression tests for the local stdio process boundary."""

from __future__ import annotations

import io
import json
import sys
from copy import deepcopy
from types import SimpleNamespace

import pytest

try:
    from mcp.types import CallToolResult, ImageContent, TextContent
except ModuleNotFoundError:
    _MCP_AVAILABLE = False
else:
    _MCP_AVAILABLE = True

pytestmark = pytest.mark.skipif(
    not _MCP_AVAILABLE,
    reason="native MCP transport is optional in the Policy/GRPO environment",
)

from src.live_mcp.config import load_suite_config, project_root
from src.live_mcp.protocol.manager import LiveMCPManager
from src.live_mcp.server_base import StatefulToolServer, serve
from src.live_mcp.protocol.transport import (
    MCPStdioTransport,
    TransportError,
    mcp_error_message,
    normalize_mcp_call_result,
    normalize_mcp_tool_response,
)


def test_stdio_servers_use_manager_python_interpreter() -> None:
    suite = load_suite_config("configs/live_mcp/ten_domain_suite.yaml")
    manager = LiveMCPManager(suite)
    config = next(server for server in suite.servers if server.enabled)

    transport = manager._build_transport(config, project_root())

    assert transport.argv[0] == sys.executable


def test_native_mcp_transport_requires_explicit_kind() -> None:
    suite = load_suite_config("configs/live_mcp/ten_domain_suite.yaml")
    manager = LiveMCPManager(suite)
    config = deepcopy(next(server for server in suite.servers if server.enabled))
    config.transport["kind"] = "mcp_stdio"

    transport = manager._build_transport(config, project_root())

    assert isinstance(transport, MCPStdioTransport)


def test_tool_discovery_queries_each_server_once() -> None:
    suite = load_suite_config("configs/live_mcp/ten_domain_suite.yaml")
    manager = LiveMCPManager(suite)
    manager._sessions["audit"] = SimpleNamespace(server_names=manager.server_names)
    calls: list[str] = []

    def request(server_name, method, params):
        assert method == "tools/list"
        calls.append(server_name)
        return {"tools": []}

    manager._request = request
    manager.discover_tools("audit")
    manager.discover_tools("audit")

    # Schemas are process-level contracts: later seeded sessions reuse the
    # atomic bootstrap snapshot instead of issuing N more tools/list RPCs.
    assert calls == manager.server_names


def test_single_json_text_result_preserves_project_envelope() -> None:
    result = CallToolResult(content=[TextContent(
        type="text", text='{"success": true, "observation": {"id": "x"}}',
    )])

    assert normalize_mcp_call_result(result) == {
        "success": True,
        "observation": {"id": "x"},
    }


def test_native_mcp_result_preserves_structured_and_all_content_blocks() -> None:
    result = CallToolResult(
        structuredContent={"success": True, "observation": {"id": "x"}},
        content=[
            TextContent(type="text", text="human summary"),
            TextContent(type="text", text='{"detail": "tail"}'),
            ImageContent(type="image", data="aGVsbG8=", mimeType="image/png"),
        ],
    )

    normalized = normalize_mcp_call_result(result)

    assert normalized["success"] is True
    assert normalized["observation"] == {"id": "x"}
    assert [item["type"] for item in normalized["_mcp_content"]] == [
        "text", "text", "image",
    ]
    assert normalized["_mcp_content"][1]["text"] == '{"detail": "tail"}'


def test_native_mcp_project_envelope_keeps_blocks_in_executor_observation() -> None:
    result = CallToolResult(
        structuredContent={"success": True, "observation": {"id": "x"}},
        content=[
            TextContent(type="text", text="human summary"),
            ImageContent(type="image", data="aGVsbG8=", mimeType="image/png"),
        ],
    )

    response = normalize_mcp_tool_response(result)

    assert response["success"] is True
    assert response["observation"]["id"] == "x"
    assert [item["type"] for item in response["observation"]["_mcp_content"]] == [
        "text", "image",
    ]
    assert "_mcp_content" not in response


def test_native_mcp_generic_readonly_payload_is_wrapped_as_observation() -> None:
    result = CallToolResult(
        structuredContent={"id": "x", "status": "ready"},
        content=[TextContent(type="text", text="ready")],
    )

    response = normalize_mcp_tool_response(result, allow_generic_readonly=True)

    assert response["success"] is True
    assert response["state_changed"] is False
    assert response["observation"]["id"] == "x"
    assert response["observation"]["_mcp_content"][0]["text"] == "ready"


def test_native_mcp_generic_mutation_without_envelope_is_rejected() -> None:
    result = CallToolResult(
        structuredContent={"id": "x", "status": "updated"},
        content=[TextContent(type="text", text="updated")],
    )

    with pytest.raises(TransportError, match="no auditable LiveMCP envelope"):
        normalize_mcp_tool_response(result)


def test_native_mcp_error_uses_every_block() -> None:
    result = CallToolResult(
        isError=True,
        content=[
            TextContent(type="text", text="first"),
            TextContent(type="text", text="second"),
        ],
    )

    assert mcp_error_message(result) == "first\nsecond"


def test_server_boundary_preserves_request_id_on_handler_error(monkeypatch) -> None:
    server = StatefulToolServer("broken", [])

    def fail(_method: str, _params: dict[str, object]) -> dict[str, object]:
        raise RuntimeError("visible failure")

    server.handle_request = fail  # type: ignore[method-assign]
    stdin = io.StringIO(json.dumps({"id": "req-7", "method": "boom"}) + "\n")
    stdout = io.StringIO()
    monkeypatch.setattr(sys, "stdin", stdin)
    monkeypatch.setattr(sys, "stdout", stdout)

    serve(server)

    response = json.loads(stdout.getvalue())
    assert response["id"] == "req-7"
    assert response["error"]["message"] == "visible failure"
