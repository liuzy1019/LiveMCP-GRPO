"""Live MCP server and session lifecycle manager."""

from __future__ import annotations

import itertools
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.live_mcp import errors
from src.live_mcp.config import ServerConfig, SuiteConfig, project_root
from src.live_mcp.schema_registry import SchemaRegistry
from src.live_mcp.transport import MCPTransport, SubprocessStdioTransport, TransportError
from src.live_mcp.types import SessionSpec


class LiveMCPManager:
    def __init__(self, suite_config: SuiteConfig):
        self.suite_config = suite_config
        self.registry = SchemaRegistry()
        self._transports: dict[str, MCPTransport] = {}
        self._sessions: dict[str, SessionSpec] = {}
        self._session_counter = itertools.count(1)

    @property
    def server_names(self) -> list[str]:
        return [cfg.name for cfg in self.suite_config.servers if cfg.enabled]

    @property
    def subprocess_stdio_used(self) -> bool:
        return bool(self._transports) and all(
            isinstance(transport, SubprocessStdioTransport)
            for transport in self._transports.values()
        )

    def start_suite(self) -> None:
        root = project_root()
        try:
            for cfg in self.suite_config.servers:
                if not cfg.enabled:
                    continue
                transport = self._build_transport(cfg, root)
                transport.start()
                self._transports[cfg.name] = transport
            bootstrap = self.create_session(seed=self._default_seed())
            try:
                self.discover_tools(bootstrap.session_id)
            finally:
                self.close_session(bootstrap.session_id)
        except Exception:
            self.stop_suite()
            raise

    def stop_suite(self) -> None:
        for transport in self._transports.values():
            transport.stop()
        self._transports.clear()
        self._sessions.clear()

    def create_session(self, seed: int | None = None) -> SessionSpec:
        seed = self._default_seed() if seed is None else seed
        session_id = f"{self.suite_config.suite_name}_{next(self._session_counter):06d}"
        spec = SessionSpec(
            session_id=session_id,
            suite_name=self.suite_config.suite_name,
            server_names=self.server_names,
            seed=seed,
            created_at=datetime.now(timezone.utc).isoformat(),
            max_turns=int(self.suite_config.rollout.get("max_turns", 8)),
        )
        self._sessions[session_id] = spec
        for server_name in spec.server_names:
            self._request(server_name, "session/reset", {"session_id": session_id, "seed": seed})
        return spec

    def reset_session(self, session_id: str, seed: int | None = None) -> None:
        spec = self._sessions[session_id]
        seed = spec.seed if seed is None else seed
        spec.seed = seed
        for server_name in spec.server_names:
            self._request(server_name, "session/reset", {"session_id": session_id, "seed": seed})

    def close_session(self, session_id: str) -> None:
        spec = self._sessions.pop(session_id, None)
        if spec is None:
            return
        for server_name in spec.server_names:
            try:
                self._request(server_name, "session/close", {"session_id": session_id})
            except TransportError:
                pass

    def discover_tools(self, session_id: str) -> list[dict[str, Any]]:
        tools: list[dict[str, Any]] = []
        spec = self._sessions[session_id]
        for server_name in spec.server_names:
            result = self._request(server_name, "tools/list", {"session_id": session_id})
            server_tools = list(result.get("tools", []))
            self.registry.register_tools(server_name, server_tools)
            tools.extend(server_tools)
        return tools

    def healthcheck(self) -> dict[str, bool]:
        status: dict[str, bool] = {}
        for name in self.server_names:
            try:
                status[name] = bool(self._request(name, "healthcheck", {}).get("ok"))
            except TransportError:
                status[name] = False
        return status

    def call_tool(
        self,
        server_name: str,
        session_id: str,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        return self._request(
            server_name,
            "tools/call",
            {"session_id": session_id, "name": tool_name, "arguments": arguments},
        )

    def get_state(self, session_id: str, server_name: str | None = None) -> dict[str, Any]:
        names = [server_name] if server_name else self._sessions[session_id].server_names
        state: dict[str, Any] = {}
        for name in names:
            state[name] = self._request(name, "debug/get_state", {"session_id": session_id}).get(
                "state", {}
            )
        return state

    def _request(self, server_name: str, method: str, params: dict[str, Any]) -> dict[str, Any]:
        transport = self._transports.get(server_name)
        if transport is None:
            raise TransportError(errors.SERVER_UNAVAILABLE, f"server not started: {server_name}")
        timeout_s = self._timeout_for(server_name)
        return transport.request(method, params, timeout_s=timeout_s)

    def _build_transport(self, cfg: ServerConfig, root: Path) -> MCPTransport:
        if cfg.transport.get("kind") != "stdio":
            raise ValueError(f"unsupported live transport: {cfg.transport.get('kind')}")
        command = cfg.command
        cwd = Path(command.get("cwd", "."))
        if not cwd.is_absolute():
            cwd = root / cwd
        return SubprocessStdioTransport(
            argv=list(command["argv"]),
            cwd=cwd,
            env={str(k): str(v) for k, v in command.get("env", {}).items()},
            startup_timeout_s=float(cfg.transport.get("startup_timeout_s", 20)),
        )

    def _timeout_for(self, server_name: str) -> float:
        for cfg in self.suite_config.servers:
            if cfg.name == server_name:
                return float(cfg.transport.get("request_timeout_s", 10))
        return 10.0

    def _default_seed(self) -> int:
        for cfg in self.suite_config.servers:
            if cfg.enabled:
                return int(cfg.session.get("default_seed", 42))
        return 42
