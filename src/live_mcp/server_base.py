"""Helpers for simple line-delimited JSON stdio servers."""

from __future__ import annotations

import copy
import json
import sys
from typing import Any, Callable

from src.live_mcp import errors
from src.live_mcp.state_seeder import StateSeeder
from src.live_mcp.tool_semantics import build_tool_semantics


class StatefulToolServer:
    def __init__(
        self,
        server_name: str,
        tools: list[dict[str, Any]],
        *,
        enforce_tool_semantics: bool = True,
    ):
        self.server_name = server_name
        self.tools = tools
        self.seeder = StateSeeder()
        self.sessions: dict[str, dict[str, Any]] = {}
        self.handlers: dict[str, Callable[[str, dict[str, Any]], dict[str, Any]]] = {}
        self.tool_semantics = (
            build_tool_semantics(server_name, tools)
            if enforce_tool_semantics else {}
        )

    def handle_request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        if method == "healthcheck":
            return {"result": {"ok": True, "server_name": self.server_name}}
        if method == "shutdown":
            return {"result": {"ok": True}}
        if method == "tools/list":
            return {"result": {"tools": copy.deepcopy(self.tools)}}
        if method == "session/reset":
            session_id = str(params["session_id"])
            seed = int(params.get("seed", 42))
            if session_id not in self.sessions and len(self.sessions) >= self.MAX_SESSIONS:
                raise RuntimeError(
                    f"session capacity exceeded for {self.server_name}: {self.MAX_SESSIONS}"
                )
            self.sessions[session_id] = self.seeder.reset_state(self.server_name, session_id, seed)
            return {"result": {"ok": True}}
        if method == "session/close":
            self.sessions.pop(str(params["session_id"]), None)
            return {"result": {"ok": True}}
        if method == "debug/get_state":
            state = self._state(str(params["session_id"]))
            return {"result": {"state": copy.deepcopy(state)}}
        if method == "tools/call":
            return {"result": self._call_tool(params)}
        return {
            "error": {
                "type": errors.UNKNOWN_TOOL,
                "message": f"unknown method: {method}",
            }
        }

    MAX_SESSIONS = 500

    def _state(self, session_id: str) -> dict[str, Any]:
        if session_id not in self.sessions:
            # Unknown and closed sessions must never silently revive with a
            # default seed: that would break rollout/replay state isolation.
            raise KeyError(f"unknown or closed session: {session_id}")
        return self.sessions[session_id]

    def _call_tool(self, params: dict[str, Any]) -> dict[str, Any]:
        session_id = str(params["session_id"])
        name = str(params["name"])
        arguments = params.get("arguments") or {}
        handler = self.handlers.get(name)
        if handler is None:
            return _result(False, None, errors.UNKNOWN_TOOL, f"unknown tool: {name}", False)
        # Treat every handler invocation as a small transaction.  A handler can
        # fail after touching more than one collection, and trusting only its
        # self-reported ``state_changed`` bit would let partial mutations leak
        # into the next Teacher decision and into success criteria.
        try:
            before = copy.deepcopy(self._state(session_id))
        except KeyError as exc:
            return _result(False, None, errors.PRECONDITION_FAILED, str(exc), False)
        try:
            result = handler(session_id, arguments)
            if not isinstance(result, dict):
                raise TypeError(
                    f"handler {name} returned {type(result).__name__}, expected dict"
                )
            required_fields = {"success", "state_changed"}
            missing_fields = sorted(required_fields - set(result))
            if missing_fields:
                raise ValueError(
                    f"handler {name} response missing fields: {missing_fields}"
                )
            after = self._state(session_id)
            delta_paths = _state_delta_paths(before, after)
            actual_changed = bool(delta_paths)
            declared_changed = bool(result.get("state_changed"))
            if not bool(result.get("success")) and actual_changed:
                raise ValueError(
                    f"failed handler {name} mutated state: {delta_paths}"
                )
            if actual_changed != declared_changed:
                raise ValueError(
                    f"handler state_changed mismatch for {name}: "
                    f"declared={declared_changed}, actual={actual_changed}"
                )
            semantics = self.tool_semantics.get(name)
            if semantics is not None:
                changed_roots = {
                    path.split(".", 1)[0] for path in delta_paths if path
                }
                unexpected_roots = sorted(
                    changed_roots - set(semantics.allowed_state_roots)
                )
                if unexpected_roots:
                    raise ValueError(
                        f"handler {name} changed disallowed state roots: "
                        f"{unexpected_roots}"
                    )
            result["state_delta_paths"] = delta_paths
            return result
        except KeyError as exc:
            _rollback_session(self.sessions, session_id, before)
            return _result(False, None, errors.PRECONDITION_FAILED, str(exc), False)
        except Exception as exc:  # defensive transaction boundary
            _rollback_session(self.sessions, session_id, before)
            return _result(False, None, errors.EXECUTION_ERROR, str(exc), False)


def _restore_state(target: dict[str, Any], snapshot: dict[str, Any]) -> None:
    """Rollback in place so callers holding the session dict see the reset."""
    target.clear()
    target.update(copy.deepcopy(snapshot))


def _rollback_session(
    sessions: dict[str, dict[str, Any]],
    session_id: str,
    snapshot: dict[str, Any],
) -> None:
    """Restore even when a faulty handler replaced/deleted its session slot."""
    current = sessions.get(session_id)
    if isinstance(current, dict):
        _restore_state(current, snapshot)
    else:
        sessions[session_id] = copy.deepcopy(snapshot)


def _state_delta_paths(before: Any, after: Any, path: tuple[str, ...] = ()) -> list[str]:
    """Return factual leaf/container paths changed by one handler call.

    Lists are kept as one container path because list indices are unstable
    under insert/remove operations.  Dict changes recurse to the affected
    entity/field, which is the useful granularity for mutation-footprint audit.
    """
    if type(before) is not type(after):
        return [".".join(path)] if path else ["<root>"]
    if isinstance(before, dict):
        changed: list[str] = []
        for key in sorted(before.keys() | after.keys(), key=str):
            child_path = (*path, str(key))
            if key not in before or key not in after:
                changed.append(".".join(child_path))
            else:
                changed.extend(_state_delta_paths(before[key], after[key], child_path))
        return changed
    if isinstance(before, (list, set)):
        return ([".".join(path)] if before != after else [])
    return ([".".join(path)] if before != after else [])


def _result(
    success: bool,
    observation: dict[str, Any] | None,
    error_type: str | None,
    error_message: str,
    state_changed: bool,
) -> dict[str, Any]:
    return {
        "success": success,
        "observation": observation,
        "error_type": error_type,
        "error_message": error_message,
        "state_changed": state_changed,
    }


def serve(server: StatefulToolServer) -> None:
    for line in sys.stdin:
        req_id: Any = None
        try:
            request = json.loads(line)
            req_id = request.get("id")
            response = server.handle_request(request.get("method", ""), request.get("params") or {})
            response["id"] = req_id
        except Exception as exc:  # pragma: no cover - server safety net
            response = {
                # Preserve correlation for parseable requests so the client
                # receives the real server error instead of timing out.
                "id": req_id,
                "error": {"type": errors.EXECUTION_ERROR, "message": str(exc)},
            }
        sys.stdout.write(json.dumps(response, ensure_ascii=True) + "\n")
        sys.stdout.flush()
