"""Subprocess stdio transport for local Live MCP servers."""

from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Protocol

from src.live_mcp import errors


class TransportError(RuntimeError):
    def __init__(self, error_type: str, message: str):
        super().__init__(message)
        self.error_type = error_type


def _dump_mcp_content_block(block: Any) -> dict[str, Any]:
    """Convert one MCP content block to a JSON-safe, lossless mapping."""
    if hasattr(block, "model_dump"):
        dumped = block.model_dump(by_alias=True, exclude_none=True)
        if isinstance(dumped, dict):
            return dumped
    if isinstance(block, dict):
        return dict(block)
    return {"type": "unknown", "value": str(block)}


def normalize_mcp_call_result(result: Any) -> dict[str, Any]:
    """Normalize MCP structuredContent and every content[] block.

    A single JSON text block keeps the historical plain-dict contract. For a
    structured or multi-block result, the primary dict remains at top level
    and the complete block envelope is retained under ``_mcp_content``.
    """
    blocks = [
        _dump_mcp_content_block(block)
        for block in (getattr(result, "content", None) or [])
    ]
    structured = getattr(result, "structuredContent", None)
    if structured is None:
        structured = getattr(result, "structured_content", None)

    parsed_text: list[Any] = []
    for block in blocks:
        if block.get("type") != "text":
            continue
        try:
            parsed_text.append(json.loads(str(block.get("text", ""))))
        except json.JSONDecodeError:
            continue

    if isinstance(structured, dict):
        normalized = dict(structured)
    else:
        primary = next(
            (item for item in parsed_text if isinstance(item, dict)),
            None,
        )
        if primary is not None:
            normalized = dict(primary)
        elif parsed_text:
            normalized = {"data": parsed_text[0]}
        elif blocks:
            normalized = {}
        else:
            return {}

    # Preserve the exact envelope whenever a single JSON text block is not a
    # complete representation of the result.
    single_json_text = (
        structured is None
        and len(blocks) == 1
        and blocks[0].get("type") == "text"
        and len(parsed_text) == 1
        and isinstance(parsed_text[0], dict)
    )
    if blocks and not single_json_text:
        normalized["_mcp_content"] = blocks
    return normalized


def normalize_mcp_tool_response(
    result: Any, *, allow_generic_readonly: bool = False,
) -> dict[str, Any]:
    """Adapt a successful native MCP result to the executor response contract.

    Project-owned MCP servers may already return the LiveMCP envelope
    (``success``/``observation``/``state_changed``). Generic MCP servers
    instead return their domain payload directly via ``structuredContent`` or
    ``content[]``. Such payloads are accepted only for tools explicitly marked
    readonly during discovery: without an auditable envelope, a mutation cannot
    be represented truthfully at the executor boundary.
    """
    normalized = normalize_mcp_call_result(result)
    if "success" not in normalized:
        if not allow_generic_readonly:
            raise TransportError(
                errors.EXECUTION_ERROR,
                "native MCP tool returned no auditable LiveMCP envelope",
            )
        return {
            "success": True,
            "observation": normalized,
            "state_changed": False,
        }

    content = normalized.pop("_mcp_content", None)
    if content:
        observation = normalized.get("observation")
        if isinstance(observation, dict):
            observation = dict(observation)
            observation["_mcp_content"] = content
        else:
            observation = {
                "data": observation,
                "_mcp_content": content,
            }
        normalized["observation"] = observation
    return normalized


def mcp_error_message(result: Any) -> str:
    """Render every MCP error content block instead of dropping the tail."""
    parts: list[str] = []
    structured = getattr(result, "structuredContent", None)
    if structured is None:
        structured = getattr(result, "structured_content", None)
    if structured is not None:
        parts.append(json.dumps(structured, ensure_ascii=False, default=str))
    for block in getattr(result, "content", None) or []:
        dumped = _dump_mcp_content_block(block)
        if dumped.get("type") == "text":
            parts.append(str(dumped.get("text", "")))
        else:
            parts.append(json.dumps(dumped, ensure_ascii=False, default=str))
    return "\n".join(part for part in parts if part).strip()


class MCPTransport(Protocol):
    def start(self) -> None: ...

    def stop(self) -> None: ...

    def request(
        self,
        method: str,
        params: dict[str, Any],
        timeout_s: float,
    ) -> dict[str, Any]: ...


class SubprocessStdioTransport:
    """Line-delimited JSON RPC over a local subprocess stdio pair."""

    def __init__(
        self,
        argv: list[str],
        cwd: Path,
        env: dict[str, str] | None = None,
        startup_timeout_s: float = 20.0,
    ):
        self.argv = argv
        self.cwd = cwd
        self.env = env or {}
        self.startup_timeout_s = startup_timeout_s
        self.process: subprocess.Popen[str] | None = None
        self._responses: dict[str, queue.Queue[dict[str, Any]]] = {}
        self._response_lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._stderr_lines: list[str] = []
        self._reader_thread: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None

    @property
    def stderr_text(self) -> str:
        return "\n".join(self._stderr_lines[-50:])

    def start(self) -> None:
        if self.process and self.process.poll() is None:
            return
        child_env = os.environ.copy()
        child_env.update(self.env)
        self.process = subprocess.Popen(
            self.argv,
            cwd=str(self.cwd),
            env=child_env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._reader_thread = threading.Thread(target=self._read_stdout, daemon=True)
        self._stderr_thread = threading.Thread(target=self._read_stderr, daemon=True)
        self._reader_thread.start()
        self._stderr_thread.start()
        deadline = time.monotonic() + self.startup_timeout_s
        while time.monotonic() < deadline:
            try:
                resp = self.request("healthcheck", {}, timeout_s=0.5)
            except TransportError:
                if self.process.poll() is not None:
                    raise
                continue
            if resp.get("ok"):
                return
        raise TransportError(errors.TIMEOUT, "server startup timed out")

    def stop(self) -> None:
        if not self.process:
            return
        if self.process.poll() is None:
            try:
                self.request("shutdown", {}, timeout_s=0.5)
            except TransportError:
                pass
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self.process.kill()
        self.process = None

    def request(
        self,
        method: str,
        params: dict[str, Any],
        timeout_s: float,
    ) -> dict[str, Any]:
        if not self.process or self.process.poll() is not None:
            raise TransportError(errors.SERVER_UNAVAILABLE, self.stderr_text or "server not running")
        if not self.process.stdin:
            raise TransportError(errors.SERVER_UNAVAILABLE, "server stdin unavailable")
        req_id = uuid.uuid4().hex
        q: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=1)
        with self._response_lock:
            self._responses[req_id] = q
        request = {"id": req_id, "method": method, "params": params}
        try:
            # One request is one line-delimited frame.  Protect serialization,
            # write and flush as a unit when generation threads share transport.
            with self._write_lock:
                self.process.stdin.write(json.dumps(request, ensure_ascii=True) + "\n")
                self.process.stdin.flush()
            response = q.get(timeout=timeout_s)
        except queue.Empty as exc:
            raise TransportError(errors.TIMEOUT, f"request timed out: {method}") from exc
        except (BrokenPipeError, OSError) as exc:
            raise TransportError(errors.SERVER_UNAVAILABLE, str(exc)) from exc
        finally:
            with self._response_lock:
                self._responses.pop(req_id, None)
        if "error" in response:
            error = response.get("error") or {}
            raise TransportError(
                str(error.get("type") or errors.EXECUTION_ERROR),
                str(error.get("message") or "request failed"),
            )
        return response.get("result", {})

    def _read_stdout(self) -> None:
        if not self.process or not self.process.stdout:
            return
        for line in self.process.stdout:
            try:
                response = json.loads(line)
            except json.JSONDecodeError:
                continue
            req_id = response.get("id")
            if not req_id:
                continue
            with self._response_lock:
                q = self._responses.get(req_id)
            if q is not None:
                q.put(response)

    def _read_stderr(self) -> None:
        if not self.process or not self.process.stderr:
            return
        for line in self.process.stderr:
            self._stderr_lines.append(line.rstrip())


class MCPStdioTransport:
    """MCP protocol transport using mcp.ClientSession + stdio_client.

    Spawns a FastMCP server subprocess and communicates via the standard MCP
    JSON-RPC 2.0 protocol.  Maintains an async event loop in a background
    thread so the rest of the system (which is sync) stays unchanged.
    """

    def __init__(
        self,
        argv: list[str],
        cwd: Path,
        env: dict[str, str] | None = None,
        startup_timeout_s: float = 20.0,
    ):
        self.argv = argv
        self.cwd = cwd
        self.env = env or {}
        self.startup_timeout_s = startup_timeout_s
        self._loop: object | None = None  # asyncio.AbstractEventLoop
        self._thread: threading.Thread | None = None
        self._session: object | None = None  # mcp.ClientSession
        self._stdio_ctx: object | None = None  # async context manager
        self._server_name: str = ""
        self._stderr_lines: list[str] = []
        self._readonly_tools: set[str] = set()

    @property
    def stderr_text(self) -> str:
        return "\n".join(self._stderr_lines[-50:])

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._session is not None:
            return

        import asyncio

        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        ready_event = threading.Event()
        start_error: Exception | None = None

        async def _async_start() -> None:
            nonlocal start_error
            try:
                server_params = StdioServerParameters(
                    command=self.argv[0],
                    args=self.argv[1:] if len(self.argv) > 1 else [],
                    env={**os.environ, **self.env} if self.env else None,
                    cwd=str(self.cwd),
                )
                ctx = stdio_client(server_params)
                read_stream, write_stream = await ctx.__aenter__()
                self._stdio_ctx = ctx
                session = ClientSession(read_stream, write_stream)
                await session.initialize()
                self._session = session
            except Exception as exc:
                start_error = exc
            finally:
                ready_event.set()

        def _run_loop() -> None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._loop = loop
            loop.run_until_complete(_async_start())
            # Keep the loop running for subsequent requests
            loop.run_forever()

        self._thread = threading.Thread(target=_run_loop, daemon=True)
        self._thread.start()
        ready_event.wait(timeout=self.startup_timeout_s)

        if start_error is not None:
            raise TransportError(errors.SERVER_UNAVAILABLE, str(start_error)) from start_error
        if self._session is None:
            raise TransportError(errors.TIMEOUT, "server startup timed out")

    def stop(self) -> None:
        import asyncio

        session = self._session
        ctx = self._stdio_ctx
        loop = self._loop  # type: asyncio.AbstractEventLoop | None
        self._session = None
        self._stdio_ctx = None

        if session is not None and loop is not None:

            async def _cleanup() -> None:
                if ctx is not None:
                    await ctx.__aexit__(None, None, None)  # type: ignore[func-returns-value]

            try:
                future = asyncio.run_coroutine_threadsafe(_cleanup(), loop)
                future.result(timeout=5)
            except Exception:
                pass

        if loop is not None:
            loop.call_soon_threadsafe(loop.stop)

        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=3)
        self._thread = None
        self._loop = None

    # ------------------------------------------------------------------
    # Request dispatch
    # ------------------------------------------------------------------

    def request(
        self,
        method: str,
        params: dict[str, Any],
        timeout_s: float,
    ) -> dict[str, Any]:
        import asyncio

        session = self._session
        loop = self._loop  # type: asyncio.AbstractEventLoop | None
        if session is None or loop is None:
            raise TransportError(errors.SERVER_UNAVAILABLE, "server not running")

        async def _async_request() -> dict[str, Any]:
            from mcp import ClientSession as CS

            s: CS = session  # type: ignore[assignment]

            # ----- MCP native operations -----
            if method == "healthcheck":
                await s.send_ping()
                return {"ok": True, "server_name": self._server_name or "unknown"}

            if method == "shutdown":
                return {"ok": True}

            # ----- tools/list -----
            if method == "tools/list":
                result = await s.list_tools()
                tools: list[dict[str, Any]] = []
                readonly_tools: set[str] = set()
                for t in result.tools:
                    td = t.model_dump() if hasattr(t, "model_dump") else dict(t)  # type: ignore[arg-type]
                    # Convert camelCase MCP keys → snake_case our system expects
                    if "inputSchema" in td:
                        td["input_schema"] = td.pop("inputSchema")
                    if "outputSchema" in td:
                        td["output_schema"] = td.pop("outputSchema")
                    # Strip session_id from input_schema — it is injected
                    # by the transport layer and must NEVER be part of the
                    # schema that executor validates LLM arguments against.
                    ischema = td.get("input_schema", {})
                    if isinstance(ischema, dict):
                        if "session_id" in ischema.get("properties", {}):
                            del ischema["properties"]["session_id"]
                        if "required" in ischema:
                            ischema["required"] = [
                                r for r in ischema["required"]
                                if r != "session_id"
                            ]
                    annotations = td.get("annotations") or {}
                    if isinstance(annotations, dict) and bool(
                        annotations.get("readOnlyHint")
                        or annotations.get("read_only_hint")
                        or annotations.get("readonly")
                    ):
                        readonly_tools.add(str(td.get("name", "")))
                    tools.append(td)
                # Filter out internal lifecycle tools (prefixed with _)
                tools = [td for td in tools if not str(td.get("name", "")).startswith("_")]
                self._readonly_tools = {
                    name for name in readonly_tools if name and not name.startswith("_")
                }
                return {"tools": tools}

            # ----- tools/call -----
            if method == "tools/call":
                session_id = str(params["session_id"])
                name = str(params["name"])
                llm_arguments = dict(params.get("arguments", {}))
                # Nest LLM arguments under "arguments" key to match
                # the FastMCP wrapper: _tool_fn(session_id, arguments)
                mcp_args = {
                    "session_id": session_id,
                    "arguments": llm_arguments,
                }
                result = await s.call_tool(name, mcp_args)
                if result.isError:
                    error_text = mcp_error_message(result)
                    raise TransportError(errors.EXECUTION_ERROR, error_text or "tool call failed")
                return normalize_mcp_tool_response(
                    result,
                    allow_generic_readonly=name in self._readonly_tools,
                )

            # ----- session/reset -----
            if method == "session/reset":
                session_id = str(params["session_id"])
                seed = int(params.get("seed", 42))
                result = await s.call_tool("_session_reset", {"session_id": session_id, "seed": seed})
                return normalize_mcp_call_result(result) or {"ok": True}

            # ----- session/close -----
            if method == "session/close":
                session_id = str(params["session_id"])
                result = await s.call_tool("_session_close", {"session_id": session_id})
                return normalize_mcp_call_result(result) or {"ok": True}

            # ----- debug/get_state -----
            if method == "debug/get_state":
                session_id = str(params["session_id"])
                result = await s.call_tool("_debug_get_state", {"session_id": session_id})
                return normalize_mcp_call_result(result) or {"state": {}}

            raise TransportError(errors.UNKNOWN_TOOL, f"unknown method: {method}")

        future = asyncio.run_coroutine_threadsafe(_async_request(), loop)
        try:
            return future.result(timeout=timeout_s)
        except TimeoutError as exc:
            raise TransportError(errors.TIMEOUT, f"request timed out: {method}") from exc
        except TransportError:
            raise
        except Exception as exc:
            raise TransportError(errors.EXECUTION_ERROR, str(exc)) from exc
