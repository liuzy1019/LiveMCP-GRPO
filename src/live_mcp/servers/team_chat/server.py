"""Stateful team chat server — 11 tools (PROVE-aligned).
Append-only: channels, messages, threads, reactions, DMs, search, user status.
"""

from __future__ import annotations
from typing import Any
from src.live_mcp.server_base import StatefulToolServer, _result, serve

TOOLS = [
    {"name": "list_channels", "description": "List channels with member count.", "input_schema": {"type": "object", "properties": {"archived": {"type": "boolean"}}, "required": []}, "annotations": {"readonly": True, "mutating": False}},
    {"name": "create_channel", "description": "Create a new channel.", "input_schema": {"type": "object", "properties": {"name": {"type": "string"}, "members": {"type": "array"}, "description": {"type": "string"}}, "required": ["name"]}, "annotations": {"mutating": True}},
    {"name": "archive_channel", "description": "Archive a channel.", "input_schema": {"type": "object", "properties": {"channel_id": {"type": "string"}}, "required": ["channel_id"]}, "annotations": {"mutating": True}},
    {"name": "get_channel", "description": "Get channel details and recent messages.", "input_schema": {"type": "object", "properties": {"channel_id": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["channel_id"]}, "annotations": {"readonly": True, "mutating": False}},
    {"name": "send_message", "description": "Send message to a channel.", "input_schema": {"type": "object", "properties": {"channel_id": {"type": "string"}, "content": {"type": "string"}, "thread_id": {"type": "string"}}, "required": ["channel_id", "content"]}, "annotations": {"mutating": True}},
    {"name": "send_dm", "description": "Send a direct message to a user.", "input_schema": {"type": "object", "properties": {"recipient": {"type": "string"}, "content": {"type": "string"}}, "required": ["recipient", "content"]}, "annotations": {"mutating": True}},
    {"name": "create_thread", "description": "Create a thread from a message.", "input_schema": {"type": "object", "properties": {"message_id": {"type": "string"}, "channel_id": {"type": "string"}}, "required": ["message_id", "channel_id"]}, "annotations": {"mutating": True}},
    {"name": "get_thread", "description": "Get thread messages.", "input_schema": {"type": "object", "properties": {"thread_id": {"type": "string"}}, "required": ["thread_id"]}, "annotations": {"readonly": True, "mutating": False}},
    {"name": "react_message", "description": "Add a reaction to a message.", "input_schema": {"type": "object", "properties": {"message_id": {"type": "string"}, "channel_id": {"type": "string"}, "reaction": {"type": "string"}}, "required": ["message_id", "channel_id", "reaction"]}, "annotations": {"mutating": True}},
    {"name": "search_messages", "description": "Search messages across channels by keyword.", "input_schema": {"type": "object", "properties": {"query": {"type": "string"}, "channel_id": {"type": "string"}, "from_user": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["query"]}, "annotations": {"readonly": True, "mutating": False}},
    {"name": "get_user_status", "description": "Get online/offline status for users.", "input_schema": {"type": "object", "properties": {"user_ids": {"type": "array"}}, "required": []}, "annotations": {"readonly": True, "mutating": False}},
]

class TeamChatServer(StatefulToolServer):
    def __init__(self) -> None:
        super().__init__("team_chat", TOOLS)
        self.handlers = {t["name"]: getattr(self, t["name"]) for t in TOOLS}

    def list_channels(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id); show_archived = arguments.get("archived", False)
        channels = [{"channel_id": c["channel_id"], "name": c["name"], "member_count": len(c["members"]), "archived": c.get("archived", False), "description": c.get("description", "")} for c in state["channels"].values() if show_archived or not c.get("archived")]
        return _result(True, {"channels": channels, "count": len(channels)}, None, "", False)

    def create_channel(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id); cid = f"ch_{arguments['name'].replace(' ', '_').lower()}"
        if cid in state["channels"]: raise KeyError(f"channel already exists: {cid}")
        ch = {"channel_id": cid, "name": arguments["name"], "members": arguments.get("members", ["current_user"]), "description": arguments.get("description", ""), "archived": False, "messages": []}
        state["channels"][cid] = ch
        return _result(True, {"channel": ch}, None, "", True)

    def archive_channel(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id); cid = arguments["channel_id"]
        if cid not in state["channels"]: raise KeyError(f"channel not found: {cid}")
        state["channels"][cid]["archived"] = True
        return _result(True, {"channel_id": cid, "archived": True}, None, "", True)

    def get_channel(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id); cid = arguments["channel_id"]
        if cid not in state["channels"]: raise KeyError(f"channel not found: {cid}")
        limit = int(arguments.get("limit", 20)); ch = state["channels"][cid]
        return _result(True, {"channel": {**ch, "messages": ch["messages"][-limit:]}, "member_count": len(ch["members"])}, None, "", False)

    def send_message(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id); cid = arguments["channel_id"]
        if cid not in state["channels"]: raise KeyError(f"channel not found: {cid}")
        if state["channels"][cid].get("archived"): raise KeyError("channel is archived")
        tid = arguments.get("thread_id")
        if tid and tid not in state["threads"]: raise KeyError(f"thread not found: {tid}")
        mid = f"msg_{state['next_msg_num']:04d}"; state["next_msg_num"] += 1
        msg = {"message_id": mid, "channel_id": cid, "content": arguments["content"], "author": "current_user", "thread_id": tid, "reactions": [], "timestamp": "2026-06-24T21:40:00"}
        state["channels"][cid]["messages"].append(msg)
        return _result(True, {"message": msg}, None, "", True)

    def send_dm(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id); recipient = arguments["recipient"]
        members = {m for ch in state["channels"].values() for m in ch["members"]}
        if recipient not in members: raise KeyError(f"user not found: {recipient}")
        did = f"dm_{state['next_msg_num']:04d}"; state["next_msg_num"] += 1
        dm = {"dm_id": did, "sender": "current_user", "recipient": recipient, "content": arguments["content"], "timestamp": "2026-06-24T21:40:00"}
        state.setdefault("dms", []).append(dm)
        return _result(True, {"direct_message": dm}, None, "", True)

    def create_thread(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id); mid, cid = arguments["message_id"], arguments["channel_id"]
        if cid not in state["channels"]: raise KeyError(f"channel not found: {cid}")
        msg = next((m for m in state["channels"][cid]["messages"] if m["message_id"] == mid), None)
        if not msg: raise KeyError(f"message not found: {mid}")
        if msg.get("thread_id"): raise KeyError("message already has thread")
        tid = f"thd_{state['next_thread_num']:04d}"; state["next_thread_num"] += 1
        state["threads"][tid] = {"thread_id": tid, "root_message_id": mid, "channel_id": cid, "messages": []}
        msg["thread_id"] = tid
        return _result(True, {"thread": state["threads"][tid]}, None, "", True)

    def get_thread(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id); tid = arguments["thread_id"]
        if tid not in state["threads"]: raise KeyError(f"thread not found: {tid}")
        return _result(True, {"thread": state["threads"][tid]}, None, "", False)

    def react_message(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id); mid, cid, reaction = arguments["message_id"], arguments["channel_id"], arguments["reaction"]
        if cid not in state["channels"]: raise KeyError(f"channel not found: {cid}")
        msg = next((m for m in state["channels"][cid]["messages"] if m["message_id"] == mid), None)
        if not msg: raise KeyError(f"message not found: {mid}")
        if reaction not in msg["reactions"]: msg["reactions"].append(reaction)
        return _result(True, {"message_id": mid, "reactions": msg["reactions"]}, None, "", True)

    def search_messages(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id); query = arguments["query"].lower(); limit = int(arguments.get("limit", 20))
        results = []
        for ch in state["channels"].values():
            if arguments.get("channel_id") and ch["channel_id"] != arguments["channel_id"]: continue
            for m in ch["messages"]:
                if arguments.get("from_user") and m["author"] != arguments["from_user"]: continue
                if query in m["content"].lower(): results.append({**m, "channel_name": ch["name"]})
        return _result(True, {"messages": results[:limit], "total": len(results)}, None, "", False)

    def get_user_status(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id)
        user_ids = arguments.get("user_ids", [])
        all_members = {m for ch in state["channels"].values() for m in ch["members"]}
        if not user_ids: user_ids = list(all_members)
        status = {uid: "online" if uid in ("current_user", "alice", "bob") else "offline" for uid in user_ids if uid in all_members}
        return _result(True, {"statuses": status}, None, "", False)

if __name__ == "__main__":
    serve(TeamChatServer())
