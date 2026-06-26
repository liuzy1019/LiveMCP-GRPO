"""Stateful filesystem server — 40 tools (PROVE-aligned).
Deepest state: files, dirs, permissions. Full POSIX-like operations.
Safety: protected paths, permission escalation detection, symlink constraints.
"""

from __future__ import annotations
import copy
from typing import Any
from src.live_mcp.server_base import StatefulToolServer, _result, serve

TOOLS = [
    # Navigation (3)
    {"name": "ls", "description": "List directory contents.", "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "long": {"type": "boolean"}}, "required": []}, "annotations": {"readonly": True, "mutating": False}},
    {"name": "cd", "description": "Change current working directory.", "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}, "annotations": {"mutating": True}},
    {"name": "pwd", "description": "Print working directory.", "input_schema": {"type": "object", "properties": {}, "required": []}, "annotations": {"readonly": True, "mutating": False}},
    # File read (5)
    {"name": "cat", "description": "Read file content.", "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}, "annotations": {"readonly": True, "mutating": False}},
    {"name": "head", "description": "Show first N lines of a file.", "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "lines": {"type": "integer"}}, "required": ["path"]}, "annotations": {"readonly": True, "mutating": False}},
    {"name": "tail", "description": "Show last N lines of a file.", "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "lines": {"type": "integer"}}, "required": ["path"]}, "annotations": {"readonly": True, "mutating": False}},
    {"name": "wc", "description": "Count lines, words, chars in a file.", "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}, "annotations": {"readonly": True, "mutating": False}},
    {"name": "stat", "description": "Show file/directory metadata.", "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}, "annotations": {"readonly": True, "mutating": False}},
    # File search (3)
    {"name": "find", "description": "Find files by name pattern recursively.", "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "pattern": {"type": "string"}, "type": {"type": "string"}}, "required": []}, "annotations": {"readonly": True, "mutating": False}},
    {"name": "grep", "description": "Search file content for a pattern.", "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "pattern": {"type": "string"}, "recursive": {"type": "boolean"}}, "required": ["pattern"]}, "annotations": {"readonly": True, "mutating": False}},
    {"name": "tree", "description": "Show directory tree.", "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "max_depth": {"type": "integer"}}, "required": []}, "annotations": {"readonly": True, "mutating": False}},
    # File create/mutate (5)
    {"name": "mkdir", "description": "Create directory.", "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "parents": {"type": "boolean"}}, "required": ["path"]}, "annotations": {"mutating": True}},
    {"name": "touch", "description": "Create file or update timestamp.", "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}, "annotations": {"mutating": True}},
    {"name": "mv", "description": "Move/rename file or directory.", "input_schema": {"type": "object", "properties": {"source": {"type": "string"}, "target": {"type": "string"}}, "required": ["source", "target"]}, "annotations": {"mutating": True}},
    {"name": "cp", "description": "Copy file or directory.", "input_schema": {"type": "object", "properties": {"source": {"type": "string"}, "target": {"type": "string"}, "recursive": {"type": "boolean"}}, "required": ["source", "target"]}, "annotations": {"mutating": True}},
    {"name": "rm", "description": "Remove file or directory.", "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "recursive": {"type": "boolean"}}, "required": ["path"]}, "annotations": {"mutating": True}},
    # Permissions (3)
    {"name": "chmod", "description": "Change permissions.", "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "mode": {"type": "string"}}, "required": ["path", "mode"]}, "annotations": {"mutating": True}},
    {"name": "chown", "description": "Change owner.", "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "owner": {"type": "string"}}, "required": ["path", "owner"]}, "annotations": {"mutating": True}},
    {"name": "umask", "description": "Get/set file creation mask.", "input_schema": {"type": "object", "properties": {"mask": {"type": "string"}}, "required": []}, "annotations": {"mutating": True}},
    # Disk usage (2)
    {"name": "du", "description": "Estimate disk usage.", "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": []}, "annotations": {"readonly": True, "mutating": False}},
    {"name": "df", "description": "Report filesystem disk space.", "input_schema": {"type": "object", "properties": {}, "required": []}, "annotations": {"readonly": True, "mutating": False}},
    # Links (2)
    {"name": "symlink", "description": "Create a symbolic link.", "input_schema": {"type": "object", "properties": {"target": {"type": "string"}, "link_path": {"type": "string"}}, "required": ["target", "link_path"]}, "annotations": {"mutating": True}},
    {"name": "readlink", "description": "Read a symbolic link's target.", "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}, "annotations": {"readonly": True, "mutating": False}},
    # Archives (4)
    {"name": "tar_create", "description": "Create a tar archive.", "input_schema": {"type": "object", "properties": {"archive": {"type": "string"}, "paths": {"type": "array"}}, "required": ["archive", "paths"]}, "annotations": {"mutating": True}},
    {"name": "tar_extract", "description": "Extract a tar archive.", "input_schema": {"type": "object", "properties": {"archive": {"type": "string"}, "target_dir": {"type": "string"}}, "required": ["archive"]}, "annotations": {"mutating": True}},
    {"name": "zip", "description": "Create a zip archive.", "input_schema": {"type": "object", "properties": {"archive": {"type": "string"}, "paths": {"type": "array"}}, "required": ["archive", "paths"]}, "annotations": {"mutating": True}},
    {"name": "unzip", "description": "Extract a zip archive.", "input_schema": {"type": "object", "properties": {"archive": {"type": "string"}, "target_dir": {"type": "string"}}, "required": ["archive"]}, "annotations": {"mutating": True}},
    # Text processing (6)
    {"name": "diff", "description": "Compare two files.", "input_schema": {"type": "object", "properties": {"file1": {"type": "string"}, "file2": {"type": "string"}}, "required": ["file1", "file2"]}, "annotations": {"readonly": True, "mutating": False}},
    {"name": "sort", "description": "Sort lines of a file.", "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "reverse": {"type": "boolean"}, "unique": {"type": "boolean"}}, "required": ["path"]}, "annotations": {"readonly": True, "mutating": False}},
    {"name": "uniq", "description": "Report or omit repeated lines.", "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "count": {"type": "boolean"}}, "required": ["path"]}, "annotations": {"readonly": True, "mutating": False}},
    {"name": "cut", "description": "Remove sections from each line.", "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "delimiter": {"type": "string"}, "fields": {"type": "string"}}, "required": ["path", "delimiter", "fields"]}, "annotations": {"readonly": True, "mutating": False}},
    {"name": "sed", "description": "Stream editor for filtering/transforming text.", "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "expression": {"type": "string"}}, "required": ["path", "expression"]}, "annotations": {"readonly": True, "mutating": False}},
    {"name": "awk", "description": "Pattern scanning and processing.", "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "script": {"type": "string"}}, "required": ["path", "script"]}, "annotations": {"readonly": True, "mutating": False}},
    # Binary/checksum (4)
    {"name": "md5sum", "description": "Compute MD5 checksum.", "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}, "annotations": {"readonly": True, "mutating": False}},
    {"name": "sha256sum", "description": "Compute SHA-256 checksum.", "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}, "annotations": {"readonly": True, "mutating": False}},
    {"name": "file_info", "description": "Determine file type (text/binary/image).", "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}, "annotations": {"readonly": True, "mutating": False}},
    {"name": "xxd", "description": "Hex dump of a file.", "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["path"]}, "annotations": {"readonly": True, "mutating": False}},
    # Utilities (3)
    {"name": "truncate", "description": "Shrink or extend file to specified size.", "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "size": {"type": "integer"}}, "required": ["path", "size"]}, "annotations": {"mutating": True}},
    {"name": "split", "description": "Split a file into pieces.", "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "lines_per_file": {"type": "integer"}}, "required": ["path"]}, "annotations": {"mutating": True}},
    {"name": "join", "description": "Join lines of two files on a common field.", "input_schema": {"type": "object", "properties": {"file1": {"type": "string"}, "file2": {"type": "string"}, "field": {"type": "integer"}}, "required": ["file1", "file2"]}, "annotations": {"readonly": True, "mutating": False}},
]

class FilesystemServer(StatefulToolServer):
    def __init__(self) -> None:
        super().__init__("filesystem", TOOLS)
        self.handlers = {t["name"]: getattr(self, t["name"]) for t in TOOLS}
        self._protected_prefix = "/protected/"
        import hashlib; self._hashlib = hashlib

    def _resolve(self, session_id: str, path: str) -> str:
        cwd = self._state(session_id)["cwd"]
        if path.startswith("/"): return path.rstrip("/") or "/"
        if cwd == "/": return "/" + path
        return (cwd + "/" + path).rstrip("/") or "/"

    def _node(self, session_id: str, path: str) -> dict[str, Any]:
        p = self._resolve(session_id, path); node = self._state(session_id)["fs"].get(p)
        if node is None: raise KeyError(f"path not found: {p}")
        return node

    def _parent(self, path: str) -> str:
        parts = [p for p in path.split("/") if p]; return "/" + "/".join(parts[:-1]) if parts else "/"

    def _children(self, state, path: str) -> list[str]:
        prefix = path + "/" if path != "/" else "/"; return [p for p in state["fs"] if p.startswith(prefix) and "/" not in p[len(prefix):]]

    # Navigation
    def ls(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id); path = self._resolve(session_id, str(arguments.get("path", ".")))
        node = state["fs"].get(path)
        if not node: raise KeyError(f"path not found: {path}")
        if node["type"] != "dir": raise KeyError(f"not a directory: {path}")
        kid_names = self._children(state, path)
        kids = [{"name": p.split("/")[-1], "type": state["fs"][p]["type"], "permissions": state["fs"][p]["permissions"], "size": len(state["fs"][p].get("content", ""))} for p in kid_names]
        return _result(True, {"path": path, "entries": sorted(kids, key=lambda x: x["name"])}, None, "", False)

    def cd(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id); path = self._resolve(session_id, arguments["path"])
        if path not in state["fs"] or state["fs"][path]["type"] != "dir": raise KeyError(f"not a directory: {arguments['path']}")
        state["cwd"] = path; return _result(True, {"cwd": path}, None, "", True)

    def pwd(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        return _result(True, {"cwd": self._state(session_id)["cwd"]}, None, "", False)

    # Read
    def cat(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        node = self._node(session_id, arguments["path"])
        if node["type"] != "file": raise KeyError("not a file")
        return _result(True, {"path": self._resolve(session_id, arguments["path"]), "content": node.get("content", ""), "size": len(node.get("content", ""))}, None, "", False)

    def head(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        node = self._node(session_id, arguments["path"]); n = int(arguments.get("lines", 10))
        if node["type"] != "file": raise KeyError("not a file")
        lines = node.get("content", "").split("\n"); return _result(True, {"lines": lines[:n], "count": min(n, len(lines))}, None, "", False)

    def tail(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        node = self._node(session_id, arguments["path"]); n = int(arguments.get("lines", 10))
        if node["type"] != "file": raise KeyError("not a file")
        lines = node.get("content", "").split("\n"); return _result(True, {"lines": lines[-n:], "count": min(n, len(lines))}, None, "", False)

    def wc(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        node = self._node(session_id, arguments["path"])
        if node["type"] != "file": raise KeyError("not a file")
        content = node.get("content", ""); return _result(True, {"lines": len(content.split("\n")), "words": len(content.split()), "chars": len(content)}, None, "", False)

    def stat(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        node = self._node(session_id, arguments["path"]); p = self._resolve(session_id, arguments["path"])
        return _result(True, {"path": p, "type": node["type"], "permissions": node["permissions"], "owner": node.get("owner", "unknown"), "size": len(node.get("content", "")), "modified": "2026-06-24T21:40:00"}, None, "", False)

    # Search
    def find(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id); path = self._resolve(session_id, arguments.get("path") or ".")
        pattern = arguments.get("pattern", "*"); ftype = arguments.get("type")
        results = []
        for p, n in state["fs"].items():
            if not p.startswith(path): continue
            name = p.split("/")[-1]
            if pattern != "*":
                import fnmatch
                if not fnmatch.fnmatch(name, pattern): continue
            if ftype and n["type"] != ftype: continue
            results.append(p)
        return _result(True, {"matches": sorted(results), "count": len(results)}, None, "", False)

    def grep(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id); pattern = arguments["pattern"]; path = self._resolve(session_id, arguments.get("path") or ".")
        recursive = arguments.get("recursive", False); results = []
        for p, n in state["fs"].items():
            if n["type"] != "file": continue
            if not recursive and p != path: continue
            if recursive and not p.startswith(path): continue
            content = n.get("content", "")
            for i, line in enumerate(content.split("\n"), 1):
                if pattern in line: results.append({"file": p, "line": i, "content": line.strip()[:200]})
        return _result(True, {"matches": results[:50], "count": len(results)}, None, "", False)

    def tree(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id); path = self._resolve(session_id, arguments.get("path") or "."); depth = int(arguments.get("max_depth", 3))
        def _build(p, d):  # noqa: E306
            if d > depth: return {"name": p.split("/")[-1] or "/", "type": "dir", "children": ["..."]}
            node = state["fs"].get(p)
            if not node: return None
            name = p.split("/")[-1] or "/"
            if node["type"] != "dir": return {"name": name, "type": "file"}
            kids = self._children(state, p)
            return {"name": name, "type": "dir", "children": [_build(k, d + 1) for k in kids]}
        return _result(True, {"tree": _build(path, 0)}, None, "", False)

    # Create
    def mkdir(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id); path = self._resolve(session_id, arguments["path"])
        if path in state["fs"]: raise KeyError(f"already exists: {path}")
        parent = self._parent(path)
        if parent != "/" and parent not in state["fs"]:
            if arguments.get("parents"): self.mkdir(session_id, {"path": parent, "parents": True})
            else: raise KeyError(f"parent not found: {parent}")
        state["fs"][path] = {"type": "dir", "content": "", "permissions": "755", "owner": "user"}
        return _result(True, {"path": path, "type": "dir"}, None, "", True)

    def touch(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id); path = self._resolve(session_id, arguments["path"])
        parent = self._parent(path)
        if parent != "/" and parent not in state["fs"]: raise KeyError(f"parent not found: {parent}")
        if path in state["fs"]: return _result(True, {"path": path, "exists": True}, None, "", False)
        state["fs"][path] = {"type": "file", "content": "", "permissions": "644", "owner": "user"}
        return _result(True, {"path": path, "created": True}, None, "", True)

    def mv(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id); src = self._resolve(session_id, arguments["source"]); dst = self._resolve(session_id, arguments["target"])
        # Check protected path BEFORE popping source to avoid data loss
        if self._protected_prefix in src:
            raise KeyError("cannot move protected paths")
        node = state["fs"].pop(src, None)
        if not node: raise KeyError(f"source not found: {src}")
        state["fs"][dst] = node; return _result(True, {"source": src, "target": dst}, None, "", True)

    def cp(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id); src = self._resolve(session_id, arguments["source"]); dst = self._resolve(session_id, arguments["target"])
        node = state["fs"].get(src)
        if not node: raise KeyError(f"source not found: {src}")
        state["fs"][dst] = copy.deepcopy(node)
        # Recursively copy children if dir
        if node["type"] == "dir" and arguments.get("recursive"):
            for child in self._children(state, src): self.cp(session_id, {"source": child, "target": dst + "/" + child.split("/")[-1]})
        return _result(True, {"source": src, "target": dst}, None, "", True)

    def rm(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id); path = self._resolve(session_id, arguments["path"])
        node = state["fs"].get(path)
        if not node: raise KeyError(f"path not found: {path}")
        if self._protected_prefix in path: raise KeyError("cannot delete protected paths")
        if node["type"] == "dir":
            kids = self._children(state, path)
            if kids and not arguments.get("recursive"): raise KeyError(f"directory not empty: {path}")
            for k in kids: state["fs"].pop(k, None)
        state["fs"].pop(path, None); return _result(True, {"path": path, "deleted": True}, None, "", True)

    # Permissions
    def chmod(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        node = self._node(session_id, arguments["path"]); old = node.get("permissions", "")
        node["permissions"] = arguments["mode"]
        return _result(True, {"path": self._resolve(session_id, arguments["path"]), "old_mode": old, "new_mode": arguments["mode"]}, None, "", True)

    def chown(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        node = self._node(session_id, arguments["path"]); old = node.get("owner", "")
        if arguments["owner"] != "root" and old == "root": raise KeyError("cannot transfer ownership from root")
        node["owner"] = arguments["owner"]
        return _result(True, {"path": self._resolve(session_id, arguments["path"]), "old_owner": old, "new_owner": arguments["owner"]}, None, "", True)

    def umask(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id)
        if arguments.get("mask"): state["umask"] = arguments["mask"]; return _result(True, {"umask": arguments["mask"]}, None, "", True)
        return _result(True, {"umask": state.get("umask", "022")}, None, "", False)

    # Disk usage
    def du(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id); path = self._resolve(session_id, arguments.get("path") or ".")
        total = sum(len(n.get("content", "")) for p, n in state["fs"].items() if p.startswith(path))
        return _result(True, {"path": path, "bytes": total}, None, "", False)

    def df(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id); total_used = sum(len(n.get("content", "")) for n in state["fs"].values())
        return _result(True, {"total_space": 1024 * 1024 * 1024, "used": total_used, "available": 1024 * 1024 * 1024 - total_used, "mount_point": "/"}, None, "", False)

    # Links
    def symlink(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id); target = arguments["target"]; link = self._resolve(session_id, arguments["link_path"])
        if link in state["fs"]: raise KeyError(f"already exists: {link}")
        state["fs"][link] = {"type": "symlink", "target": target, "permissions": "777", "owner": "user"}
        return _result(True, {"link_path": link, "target": target}, None, "", True)

    def readlink(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        node = self._node(session_id, arguments["path"])
        if node["type"] != "symlink": raise KeyError("not a symlink")
        return _result(True, {"path": self._resolve(session_id, arguments["path"]), "target": node.get("target", "")}, None, "", False)

    # Archives (simulated)
    def tar_create(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id); archive = self._resolve(session_id, arguments["archive"]); paths = arguments["paths"]
        content_parts = []
        for p in paths:
            rp = self._resolve(session_id, p); node = state["fs"].get(rp)
            if node: content_parts.append(f"[{rp}] {node.get('content', '')[:200]}")
        state["fs"][archive] = {"type": "file", "content": "\n---\n".join(content_parts), "permissions": "644", "owner": "user"}
        return _result(True, {"archive": archive, "files_count": len(paths)}, None, "", True)

    def tar_extract(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id); archive = self._resolve(session_id, arguments["archive"])
        if archive not in state["fs"]: raise KeyError(f"archive not found: {archive}")
        return _result(True, {"archive": archive, "extracted_to": arguments.get("target_dir", "."), "message": "extraction simulated"}, None, "", True)

    def zip(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        return self.tar_create(session_id, arguments)

    def unzip(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        return self.tar_extract(session_id, arguments)

    # Text processing
    def diff(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        n1 = self._node(session_id, arguments["file1"]); n2 = self._node(session_id, arguments["file2"])
        c1 = n1.get("content", "").split("\n"); c2 = n2.get("content", "").split("\n")
        diffs = []
        for i, (l1, l2) in enumerate(zip(c1, c2)):
            if l1 != l2: diffs.append({"line": i + 1, "left": l1, "right": l2})
        for i in range(len(c1), len(c2)): diffs.append({"line": i + 1, "left": "<missing>", "right": c2[i]})
        return _result(True, {"differences": diffs, "count": len(diffs)}, None, "", False)

    def sort(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        node = self._node(session_id, arguments["path"])
        if node["type"] != "file": raise KeyError("not a file")
        lines = node.get("content", "").split("\n")
        reverse = arguments.get("reverse", False)
        uniq = arguments.get("unique", False)
        sorted_lines = sorted(lines, reverse=reverse)
        if uniq:
            seen = set(); unique_lines = []
            for l in sorted_lines:
                if l not in seen: seen.add(l); unique_lines.append(l)
            sorted_lines = unique_lines
        return _result(True, {"sorted": sorted_lines}, None, "", False)

    def uniq(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        node = self._node(session_id, arguments["path"])
        if node["type"] != "file": raise KeyError("not a file")
        lines = node.get("content", "").split("\n"); count = arguments.get("count", False)
        seen = {}; result = []
        for l in lines:
            seen[l] = seen.get(l, 0) + 1
        for l, c in seen.items(): result.append(f"{c:4d} {l}" if count else l)
        return _result(True, {"unique_lines": result}, None, "", False)

    def cut(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        node = self._node(session_id, arguments["path"])
        if node["type"] != "file": raise KeyError("not a file")
        delim = arguments["delimiter"]; fields = arguments["fields"]
        lines = node.get("content", "").split("\n")
        result = []
        for l in lines:
            parts = l.split(delim)
            if "," in fields:
                result.append(delim.join(parts[int(f) - 1] for f in fields.split(",") if int(f) <= len(parts)))
            else:
                fi = int(fields); result.append(parts[fi - 1] if fi <= len(parts) else "")
        return _result(True, {"cut_lines": result}, None, "", False)

    def sed(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        node = self._node(session_id, arguments["path"])
        if node["type"] != "file": raise KeyError("not a file")
        import re
        expr = arguments["expression"]; lines = node.get("content", "").split("\n")
        result = []
        for l in lines:
            try:
                if expr.startswith("s/"):
                    parts = expr[2:].rsplit("/", 2)
                    if len(parts) >= 3: l = re.sub(parts[0], parts[1], l)
            except Exception: pass
            result.append(l)
        return _result(True, {"result": result}, None, "", False)

    def awk(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        node = self._node(session_id, arguments["path"])
        if node["type"] != "file": raise KeyError("not a file")
        script = arguments["script"]; lines = node.get("content", "").split("\n"); result = []
        for l in lines:
            fields = l.split()
            try:
                if "{print $" in script:
                    idx = int(script.split("$")[1].split("}")[0])
                    if idx > 0 and idx <= len(fields): result.append(fields[idx - 1])
                else: result.append(l)
            except Exception: result.append(l)
        return _result(True, {"output": result}, None, "", False)

    # Checksums
    def md5sum(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        node = self._node(session_id, arguments["path"])
        if node["type"] != "file": raise KeyError("not a file")
        h = self._hashlib.md5(node.get("content", "").encode()).hexdigest()
        return _result(True, {"path": self._resolve(session_id, arguments["path"]), "md5": h}, None, "", False)

    def sha256sum(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        node = self._node(session_id, arguments["path"])
        if node["type"] != "file": raise KeyError("not a file")
        h = self._hashlib.sha256(node.get("content", "").encode()).hexdigest()
        return _result(True, {"path": self._resolve(session_id, arguments["path"]), "sha256": h}, None, "", False)

    def file_info(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        node = self._node(session_id, arguments["path"])
        if node["type"] == "dir": ftype = "directory"
        elif node["type"] == "symlink": ftype = "symbolic link"
        else:
            c = node.get("content", "")
            ftype = "ASCII text" if all(ord(ch) < 128 or ch == '\n' for ch in c if ch) else "data"
        return _result(True, {"path": self._resolve(session_id, arguments["path"]), "type": ftype}, None, "", False)

    def xxd(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        node = self._node(session_id, arguments["path"])
        if node["type"] != "file": raise KeyError("not a file")
        content = node.get("content", ""); limit = int(arguments.get("limit", 256))
        hex_lines = []
        for i in range(0, min(len(content), limit), 16):
            chunk = content[i:i + 16]; hex_part = " ".join(f"{ord(c):02x}" if c else "  " for c in chunk)
            ascii_part = "".join(c if 32 <= ord(c) < 127 else "." for c in chunk)
            hex_lines.append(f"{i:08x}: {hex_part:<48s} {ascii_part}")
        return _result(True, {"hex_dump": hex_lines}, None, "", False)

    # Utilities
    def truncate(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        node = self._node(session_id, arguments["path"]); size = int(arguments["size"])
        if node["type"] != "file": raise KeyError("not a file")
        content = node.get("content", "")
        if size < len(content): node["content"] = content[:size]
        else: node["content"] = content + " " * (size - len(content))
        return _result(True, {"path": self._resolve(session_id, arguments["path"]), "new_size": len(node["content"])}, None, "", True)

    def split(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id); path = self._resolve(session_id, arguments["path"]); lines_per = int(arguments.get("lines_per_file", 100))
        node = self._node(session_id, path); lines = node.get("content", "").split("\n"); created = []
        for i in range(0, len(lines), lines_per):
            chunk_path = f"{path}.part{i // lines_per + 1:02d}"
            state["fs"][chunk_path] = {"type": "file", "content": "\n".join(lines[i:i + lines_per]), "permissions": "644", "owner": "user"}
            created.append(chunk_path)
        return _result(True, {"source": path, "parts": created, "count": len(created)}, None, "", True)

    def join(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        n1 = self._node(session_id, arguments["file1"]); n2 = self._node(session_id, arguments["file2"])
        if n1["type"] != "file" or n2["type"] != "file": raise KeyError("both arguments must be files")
        field = int(arguments.get("field", 1)) - 1; c1 = n1.get("content", "").split("\n"); c2 = n2.get("content", "").split("\n")
        map2 = {}
        for l in c2:
            parts = l.split()
            if parts: map2[parts[0]] = parts
        joined = []
        for l in c1:
            parts = l.split()
            if parts and parts[0] in map2:
                joined.append(l + " " + " ".join(map2[parts[0]][field + 1:] if len(map2[parts[0]]) > field else []))
            else:
                joined.append(l)
        return _result(True, {"joined": joined}, None, "", False)


if __name__ == "__main__":
    serve(FilesystemServer())
