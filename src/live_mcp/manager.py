"""Live MCP server and session lifecycle manager."""

from __future__ import annotations

import itertools
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.live_mcp import errors
from src.live_mcp.config import ServerConfig, SuiteConfig, project_root
from src.live_mcp.schema_registry import SchemaRegistry
from src.live_mcp.transport import (
    MCPStdioTransport,
    MCPTransport,
    SubprocessStdioTransport,
    TransportError,
)
from src.live_mcp.types import SessionSpec


class LiveMCPManager:
    def __init__(self, suite_config: SuiteConfig):
        self.suite_config = suite_config
        self.registry = SchemaRegistry()
        self._transports: dict[str, MCPTransport] = {}
        self._sessions: dict[str, SessionSpec] = {}
        self._session_locks: dict[str, threading.RLock] = {}
        self._poisoned_sessions: dict[str, str] = {}
        self._lifecycle_lock = threading.RLock()
        self._discovery_lock = threading.RLock()
        self._tools_discovered = False
        self._session_counter = itertools.count(1)

    @property
    def server_names(self) -> list[str]:
        return [cfg.name for cfg in self.suite_config.servers if cfg.enabled]

    @property
    def subprocess_stdio_used(self) -> bool:
        return bool(self._transports) and all(
            isinstance(transport, SubprocessStdioTransport) for transport in self._transports.values()
        )

    def start_suite(self) -> None:
        root = project_root()
        try:
            for cfg in self.suite_config.servers:
                if not cfg.enabled:
                    continue
                transport = self._build_transport(cfg, root)
                try:
                    transport.start()
                except Exception:
                    # ``start`` may already have created a process or reader
                    # thread before reporting failure.  It is not registered
                    # yet, so suite-level cleanup cannot reach it.
                    transport.stop()
                    raise
                with self._lifecycle_lock:
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
        with self._lifecycle_lock:
            transports = list(self._transports.values())
            self._transports.clear()
            self._sessions.clear()
            self._session_locks.clear()
            self._poisoned_sessions.clear()
            self._tools_discovered = False
            self.registry = SchemaRegistry()
        for transport in transports:
            transport.stop()

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
        with self._lifecycle_lock:
            self._sessions[session_id] = spec
            operation_lock = self._session_locks.setdefault(
                session_id, threading.RLock(),
            )
        try:
            with operation_lock:
                for server_name in spec.server_names:
                    self._request(
                        server_name,
                        "session/reset",
                        {"session_id": session_id, "seed": seed},
                    )
        except Exception:
            with self._lifecycle_lock:
                self._sessions.pop(session_id, None)
                self._session_locks.pop(session_id, None)
                self._poisoned_sessions[session_id] = "session creation failed"
            # The failing request may have reached the server before a timeout,
            # so close on every owner rather than only acknowledged resets.
            for server_name in spec.server_names:
                try:
                    self._request(
                        server_name,
                        "session/close",
                        {"session_id": session_id},
                    )
                except TransportError:
                    pass
            raise
        return spec

    def reset_session(self, session_id: str, seed: int | None = None) -> None:
        with self._operation_lock(session_id):
            spec = self._active_session(session_id)
            seed = spec.seed if seed is None else seed
            try:
                for server_name in spec.server_names:
                    self._request(
                        server_name,
                        "session/reset",
                        {"session_id": session_id, "seed": seed},
                    )
            except Exception:
                # A partially reset multi-server session is not safe to reuse.
                self.quarantine_session(session_id, "session reset failed")
                raise
            spec.seed = seed

    def close_session(self, session_id: str) -> None:
        with self._operation_lock(session_id, allow_missing=True):
            with self._lifecycle_lock:
                spec = self._sessions.pop(session_id, None)
            if spec is None:
                with self._lifecycle_lock:
                    self._session_locks.pop(session_id, None)
                return
            for server_name in spec.server_names:
                try:
                    self._request(server_name, "session/close", {"session_id": session_id})
                except TransportError:
                    pass
        with self._lifecycle_lock:
            self._session_locks.pop(session_id, None)

    def quarantine_session(self, session_id: str, reason: str) -> None:
        """Make an unknown-commit session permanently unavailable.

        A transport timeout cannot prove that a stateful request was not
        committed by the server.  Remove the session from the active registry
        before any best-effort close I/O so concurrent callers fail closed.
        Session IDs are monotonic and never reused.
        """
        with self._operation_lock(session_id, allow_missing=True):
            with self._lifecycle_lock:
                spec = self._sessions.pop(session_id, None)
                self._poisoned_sessions[session_id] = str(reason)
            if spec is not None:
                for server_name in spec.server_names:
                    try:
                        self._request(
                            server_name,
                            "session/close",
                            {"session_id": session_id},
                            timeout_s=0.5,
                        )
                    except TransportError:
                        pass
        with self._lifecycle_lock:
            self._session_locks.pop(session_id, None)

    def discover_tools(self, session_id: str) -> list[dict[str, Any]]:
        spec = self._active_session(session_id)
        # Tool schemas are process-level contracts in this environment; they
        # do not depend on seeded session state.  The suite bootstrap performs
        # one full discovery.  Reusing that immutable snapshot removes ten
        # redundant RPCs from every generation/replay/rollout session.
        with self._discovery_lock:
            if self._tools_discovered:
                return self.registry.all_tools()

            discovered: dict[str, list[dict[str, Any]]] = {}
            for server_name in spec.server_names:
                result = self._request(
                    server_name, "tools/list", {"session_id": session_id},
                )
                discovered[server_name] = list(result.get("tools", []))

            # Publish only after all owners answered, so a partial suite
            # discovery can never become the runtime schema snapshot.
            registry = SchemaRegistry()
            for server_name, server_tools in discovered.items():
                registry.register_tools(server_name, server_tools)
            self.registry = registry
            self._tools_discovered = True
            return self.registry.all_tools()

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
        with self._operation_lock(session_id):
            spec = self._active_session(session_id)
            if server_name not in spec.server_names:
                raise KeyError(f"server {server_name!r} is not bound to session {session_id}")
            return self._request(
                server_name,
                "tools/call",
                {"session_id": session_id, "name": tool_name, "arguments": arguments},
            )

    def get_state(self, session_id: str, server_name: str | None = None) -> dict[str, Any]:
        with self._operation_lock(session_id):
            spec = self._active_session(session_id)
            if server_name is not None and server_name not in spec.server_names:
                raise KeyError(f"server {server_name!r} is not bound to session {session_id}")
            names = [server_name] if server_name else spec.server_names
            state: dict[str, Any] = {}
            for name in names:
                state[name] = self._request(name, "debug/get_state", {"session_id": session_id}).get(
                    "state", {}
                )
            return state

    def _operation_lock(
        self, session_id: str, *, allow_missing: bool = False,
    ) -> threading.RLock:
        with self._lifecycle_lock:
            lock = self._session_locks.get(session_id)
            if lock is None and allow_missing:
                lock = threading.RLock()
            if lock is None:
                self._active_session(session_id)
                raise AssertionError("active session missing operation lock")
            return lock

    def _active_session(self, session_id: str) -> SessionSpec:
        with self._lifecycle_lock:
            spec = self._sessions.get(session_id)
            poisoned_reason = self._poisoned_sessions.get(session_id)
        if spec is not None:
            return spec
        if poisoned_reason:
            raise KeyError(
                f"poisoned session cannot be reused: {session_id}: {poisoned_reason}"
            )
        raise KeyError(f"unknown or closed session: {session_id}")

    def _request(
        self,
        server_name: str,
        method: str,
        params: dict[str, Any],
        *,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        with self._lifecycle_lock:
            transport = self._transports.get(server_name)
        if transport is None:
            raise TransportError(errors.SERVER_UNAVAILABLE, f"server not started: {server_name}")
        request_timeout = self._timeout_for(server_name) if timeout_s is None else timeout_s
        return transport.request(method, params, timeout_s=request_timeout)

    def _build_transport(self, cfg: ServerConfig, root: Path) -> MCPTransport:
        transport_kind = cfg.transport.get("kind")
        if transport_kind not in {"stdio", "mcp_stdio"}:
            raise ValueError(f"unsupported live transport: {transport_kind}")
        command = cfg.command
        cwd = Path(command.get("cwd", "."))
        if not cwd.is_absolute():
            cwd = root / cwd
        argv = list(command["argv"])
        # Suite files stay portable (`python -m ...`), while subprocesses must
        # inherit the exact environment selected for the manager process.
        if argv and argv[0] in {"python", "python3"}:
            argv[0] = sys.executable
        transport_type = (
            MCPStdioTransport
            if transport_kind == "mcp_stdio"
            else SubprocessStdioTransport
        )
        return transport_type(
            argv=argv,
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
