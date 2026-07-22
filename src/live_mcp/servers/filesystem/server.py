"""Stateful filesystem server with 40 tools.
Deepest state: files, dirs, permissions. Full POSIX-like operations.
Safety: protected paths, permission escalation detection, symlink constraints.
"""

from __future__ import annotations
import copy
import posixpath
from typing import Any
from src.live_mcp.server_base import StatefulToolServer, _result, serve

TOOLS = [
    # Navigation (3)
    {"name": "ls", "description": "List directory contents.", "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "long": {"type": "boolean"}}, "required": []}, "annotations": {"readonly": True, "mutating": False}},
    {"name": "cd", "description": "Change current working directory.", "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}, "annotations": {"mutating": True}},
    {"name": "pwd", "description": "Print working directory.", "input_schema": {"type": "object", "properties": {}, "required": []}, "annotations": {"readonly": True, "mutating": False}},
    # File read (5)
    {"name": "cat", "description": "Read file content.", "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}, "annotations": {"readonly": True, "mutating": False}},
    {"name": "head", "description": "Show first N lines of a file.", "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "lines": {"type": "integer", "minimum": 1}}, "required": ["path"]}, "annotations": {"readonly": True, "mutating": False}},
    {"name": "tail", "description": "Show last N lines of a file.", "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "lines": {"type": "integer", "minimum": 1}}, "required": ["path"]}, "annotations": {"readonly": True, "mutating": False}},
    {"name": "wc", "description": "Count lines, words, chars in a file.", "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}, "annotations": {"readonly": True, "mutating": False}},
    {"name": "stat", "description": "Show file/directory metadata.", "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}, "annotations": {"readonly": True, "mutating": False}},
    # File search (3)
    {"name": "find", "description": "Find files by name pattern recursively.", "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "pattern": {"type": "string"}, "type": {"type": "string"}}, "required": []}, "annotations": {"readonly": True, "mutating": False}},
    {"name": "grep", "description": "Search file content for a pattern.", "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "pattern": {"type": "string"}, "recursive": {"type": "boolean"}}, "required": ["pattern"]}, "annotations": {"readonly": True, "mutating": False}},
    {"name": "tree", "description": "Show directory tree.", "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "max_depth": {"type": "integer"}}, "required": []}, "annotations": {"readonly": True, "mutating": False}},
    # File create/mutate (5)
    {"name": "mkdir", "description": "Create directory.", "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "parents": {"type": "boolean"}}, "required": ["path"]}, "annotations": {"mutating": True}},
    {"name": "touch", "description": "Create file or update timestamp.", "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}, "annotations": {"mutating": True}},
    {"name": "mv", "description": "Move or rename an existing file/directory to a new target path. The target must not already exist and cannot be inside the source directory.", "input_schema": {"type": "object", "properties": {"source": {"type": "string", "description": "Existing non-protected source path."}, "target": {"type": "string", "description": "New non-protected destination path that does not already exist and is not inside source."}}, "required": ["source", "target"]}, "annotations": {"mutating": True}},
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
    {"name": "tar_create", "description": "Create a tar archive from existing files or directories.", "input_schema": {"type": "object", "properties": {"archive": {"type": "string"}, "paths": {"type": "array", "items": {"type": "string"}, "minItems": 1}}, "required": ["archive", "paths"], "additionalProperties": False}, "annotations": {"mutating": True}},
    {"name": "tar_extract", "description": "Extract a tar archive into an existing directory.", "input_schema": {"type": "object", "properties": {"archive": {"type": "string"}, "target_dir": {"type": "string"}}, "required": ["archive"], "additionalProperties": False}, "annotations": {"mutating": True}},
    {"name": "zip", "description": "Create a zip archive from existing files or directories.", "input_schema": {"type": "object", "properties": {"archive": {"type": "string"}, "paths": {"type": "array", "items": {"type": "string"}, "minItems": 1}}, "required": ["archive", "paths"], "additionalProperties": False}, "annotations": {"mutating": True}},
    {"name": "unzip", "description": "Extract a zip archive into an existing directory.", "input_schema": {"type": "object", "properties": {"archive": {"type": "string"}, "target_dir": {"type": "string"}}, "required": ["archive"], "additionalProperties": False}, "annotations": {"mutating": True}},
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
    {"name": "truncate", "description": "Shrink or extend file to a non-negative size.", "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "size": {"type": "integer", "minimum": 0}}, "required": ["path", "size"]}, "annotations": {"mutating": True}},
    {"name": "split", "description": "Split a file into pieces using a positive line count.", "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "lines_per_file": {"type": "integer", "minimum": 1}}, "required": ["path"]}, "annotations": {"mutating": True}},
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
        raw = path or "."
        if raw.startswith("/"):
            return posixpath.normpath(raw) or "/"
        base = "/" if cwd == "/" else cwd
        return posixpath.normpath(posixpath.join(base, raw)) or "/"

    def _node(self, session_id: str, path: str) -> dict[str, Any]:
        p = self._resolve(session_id, path); node = self._state(session_id)["fs"].get(p)
        if node is None: raise KeyError(f"path not found: {p}")
        return node

    def _parent(self, path: str) -> str:
        parts = [p for p in path.split("/") if p]; return "/" + "/".join(parts[:-1]) if parts else "/"

    def _children(self, state, path: str) -> list[str]:
        prefix = path + "/" if path != "/" else "/"; return [p for p in state["fs"] if p.startswith(prefix) and "/" not in p[len(prefix):]]

    def _is_protected(self, path: str) -> bool:
        return path == "/protected" or path.startswith(self._protected_prefix)

    @staticmethod
    def _at_or_below(path: str, root: str) -> bool:
        return path == root or path.startswith(root.rstrip("/") + "/")

    def _ensure_parent_dir(self, state: dict[str, Any], path: str) -> str:
        parent = self._parent(path)
        if parent not in state["fs"] or state["fs"][parent]["type"] != "dir":
            raise KeyError(f"parent not found: {parent}")
        return parent

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
        changed = state["cwd"] != path
        state["cwd"] = path; return _result(True, {"cwd": path}, None, "", changed)

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
        if n <= 0: raise KeyError("lines must be positive")
        lines = node.get("content", "").split("\n"); return _result(True, {"lines": lines[:n], "count": min(n, len(lines))}, None, "", False)

    def tail(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        node = self._node(session_id, arguments["path"]); n = int(arguments.get("lines", 10))
        if node["type"] != "file": raise KeyError("not a file")
        if n <= 0: raise KeyError("lines must be positive")
        lines = node.get("content", "").split("\n"); return _result(True, {"lines": lines[-n:], "count": min(n, len(lines))}, None, "", False)

    def wc(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        node = self._node(session_id, arguments["path"])
        if node["type"] != "file": raise KeyError("not a file")
        content = node.get("content", ""); return _result(True, {"lines": len(content.split("\n")), "words": len(content.split()), "chars": len(content)}, None, "", False)

    def stat(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id)
        node = self._node(session_id, arguments["path"]); p = self._resolve(session_id, arguments["path"])
        return _result(True, {"path": p, "type": node["type"], "permissions": node["permissions"], "owner": node.get("owner", "unknown"), "size": len(node.get("content", "")), "modified": f"{state['current_date']}T21:40:00"}, None, "", False)

    # Search
    def find(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id); path = self._resolve(session_id, arguments.get("path") or ".")
        pattern = arguments.get("pattern", "*"); ftype = arguments.get("type")
        results = []
        for p, n in state["fs"].items():
            if not self._at_or_below(p, path): continue
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
            if recursive and not self._at_or_below(p, path): continue
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
        if self._is_protected(path): raise KeyError("cannot create protected paths")
        if path in state["fs"]: raise KeyError(f"already exists: {path}")
        parent = self._parent(path)
        if parent != "/" and parent not in state["fs"]:
            if arguments.get("parents"): self.mkdir(session_id, {"path": parent, "parents": True})
            else: raise KeyError(f"parent not found: {parent}")
        self._ensure_parent_dir(state, path)
        state["fs"][path] = {"type": "dir", "content": "", "permissions": "755", "owner": "user"}
        return _result(True, {"path": path, "type": "dir"}, None, "", True)

    def touch(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id); path = self._resolve(session_id, arguments["path"])
        if self._is_protected(path): raise KeyError("cannot create protected paths")
        self._ensure_parent_dir(state, path)
        if path in state["fs"]: return _result(True, {"path": path, "exists": True}, None, "", False)
        state["fs"][path] = {"type": "file", "content": "", "permissions": "644", "owner": "user"}
        return _result(True, {"path": path, "created": True}, None, "", True)

    def mv(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id); src = self._resolve(session_id, arguments["source"]); dst = self._resolve(session_id, arguments["target"])
        if self._is_protected(src):
            raise KeyError("cannot move protected paths")
        if self._is_protected(dst):
            raise KeyError("cannot move into protected paths")
        node = state["fs"].get(src)
        if not node: raise KeyError(f"source not found: {src}")
        if src == "/" or dst == "/" or dst == src or dst.startswith(src + "/"):
            raise KeyError("cannot move a directory into itself")
        self._ensure_parent_dir(state, dst)
        if dst in state["fs"]:
            raise KeyError(f"target already exists: {dst}")
        descendants = []
        if node["type"] == "dir":
            descendants = sorted(
                [p for p in state["fs"] if p.startswith(src + "/")],
                key=len,
            )
        moved = [(src, dst, state["fs"].pop(src))]
        for child in descendants:
            moved.append((child, dst + child[len(src):], state["fs"].pop(child)))
        for _old, new, child_node in moved:
            state["fs"][new] = child_node
        if self._at_or_below(state["cwd"], src):
            state["cwd"] = dst + state["cwd"][len(src):]
        return _result(True, {"source": src, "target": dst}, None, "", True)

    def cp(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id); src = self._resolve(session_id, arguments["source"]); dst = self._resolve(session_id, arguments["target"])
        node = state["fs"].get(src)
        if not node: raise KeyError(f"source not found: {src}")
        if self._is_protected(src):
            raise KeyError("cannot copy protected paths")
        if self._is_protected(dst):
            raise KeyError("cannot copy into protected paths")
        self._ensure_parent_dir(state, dst)
        if dst in state["fs"]:
            raise KeyError(f"target already exists: {dst}")
        if node["type"] == "dir" and (dst == src or dst.startswith(src + "/")):
            raise KeyError("cannot copy a directory into itself")
        if node["type"] == "dir" and not arguments.get("recursive"):
            raise KeyError("omitting directory; use recursive")
        state["fs"][dst] = copy.deepcopy(node)
        # Recursively copy children if dir
        if node["type"] == "dir" and arguments.get("recursive"):
            for child in self._children(state, src):
                child_dst = dst + "/" + child.split("/")[-1]
                self.cp(session_id, {"source": child, "target": child_dst, "recursive": True})
        return _result(True, {"source": src, "target": dst}, None, "", True)

    def rm(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id); path = self._resolve(session_id, arguments["path"])
        node = state["fs"].get(path)
        if not node: raise KeyError(f"path not found: {path}")
        if self._is_protected(path): raise KeyError("cannot delete protected paths")
        if path == "/": raise KeyError("cannot delete root")
        if node["type"] == "dir":
            kids = list(self._children(state, path))
            if kids and not arguments.get("recursive"): raise KeyError(f"directory not empty: {path}")
            if arguments.get("recursive"):
                # Recursively collect all descendants for removal
                all_descendants = list(kids)
                stack = list(kids)
                while stack:
                    child = stack.pop()
                    if state["fs"].get(child, {}).get("type") == "dir":
                        grandchildren = list(self._children(state, child))
                        all_descendants.extend(grandchildren)
                        stack.extend(grandchildren)
                for k in reversed(all_descendants): state["fs"].pop(k, None)
        if self._at_or_below(state["cwd"], path):
            parent = path.rsplit("/", 1)[0] or "/"
            state["cwd"] = parent if parent in state["fs"] else "/"
        state["fs"].pop(path, None); return _result(True, {"path": path, "deleted": True}, None, "", True)

    # Permissions
    def chmod(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        path = self._resolve(session_id, arguments["path"])
        if self._is_protected(path): raise KeyError("cannot chmod protected paths")
        node = self._node(session_id, path); old = node.get("permissions", "")
        node["permissions"] = arguments["mode"]
        return _result(True, {"path": path, "old_mode": old, "new_mode": arguments["mode"]}, None, "", old != arguments["mode"])

    def chown(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        path = self._resolve(session_id, arguments["path"])
        if self._is_protected(path): raise KeyError("cannot chown protected paths")
        node = self._node(session_id, path); old = node.get("owner", "")
        if arguments["owner"] != "root" and old == "root": raise KeyError("cannot transfer ownership from root")
        node["owner"] = arguments["owner"]
        return _result(True, {"path": path, "old_owner": old, "new_owner": arguments["owner"]}, None, "", old != arguments["owner"])

    def umask(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id)
        if arguments.get("mask"):
            old = state.get("umask", "022"); state["umask"] = arguments["mask"]
            return _result(True, {"umask": arguments["mask"]}, None, "", old != arguments["mask"])
        return _result(True, {"umask": state.get("umask", "022")}, None, "", False)

    # Disk usage
    def du(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id); path = self._resolve(session_id, arguments.get("path") or ".")
        total = sum(len(n.get("content", "")) for p, n in state["fs"].items() if self._at_or_below(p, path))
        return _result(True, {"path": path, "bytes": total}, None, "", False)

    def df(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id); total_used = sum(len(n.get("content", "")) for n in state["fs"].values())
        return _result(True, {"total_space": 1024 * 1024 * 1024, "used": total_used, "available": 1024 * 1024 * 1024 - total_used, "mount_point": "/"}, None, "", False)

    # Links
    def symlink(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id); target = arguments["target"]; link = self._resolve(session_id, arguments["link_path"])
        target_path = self._resolve(session_id, target)
        if self._is_protected(link): raise KeyError("cannot create symlink in protected paths")
        if self._is_protected(target_path): raise KeyError("cannot link to protected paths")
        self._ensure_parent_dir(state, link)
        if link in state["fs"]: raise KeyError(f"already exists: {link}")
        state["fs"][link] = {"type": "symlink", "target": target, "permissions": "777", "owner": "user"}
        return _result(True, {"link_path": link, "target": target}, None, "", True)

    def readlink(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        node = self._node(session_id, arguments["path"])
        if node["type"] != "symlink": raise KeyError("not a symlink")
        return _result(True, {"path": self._resolve(session_id, arguments["path"]), "target": node.get("target", "")}, None, "", False)

    # Archives
    def tar_create(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id); archive = self._resolve(session_id, arguments["archive"]); paths = arguments["paths"]
        if self._is_protected(archive): raise KeyError("cannot create archive in protected paths")
        self._ensure_parent_dir(state, archive)
        if archive in state["fs"]: raise KeyError(f"archive already exists: {archive}")
        if not paths: raise KeyError("paths must be non-empty")
        entries: dict[str, dict[str, Any]] = {}
        for p in paths:
            rp = self._resolve(session_id, p)
            if self._is_protected(rp): raise KeyError("cannot archive protected paths")
            node = state["fs"].get(rp)
            if node is None: raise KeyError(f"archive input not found: {rp}")
            prefix = rp.rstrip("/") + "/"
            selected = [
                candidate for candidate in state["fs"]
                if candidate == rp or candidate.startswith(prefix)
            ]
            base_parent = posixpath.dirname(rp)
            for candidate in sorted(selected):
                relative = posixpath.relpath(candidate, base_parent)
                if relative in entries:
                    raise KeyError(f"duplicate archive entry: {relative}")
                entries[relative] = copy.deepcopy(state["fs"][candidate])
        state["fs"][archive] = {
            "type": "file",
            "content": "",
            "permissions": "644",
            "owner": "user",
            "archive_format": "tar",
            "archive_entries": entries,
        }
        return _result(True, {"archive": archive, "files_count": len(entries)}, None, "", True)

    def tar_extract(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id); archive = self._resolve(session_id, arguments["archive"])
        archive_node = state["fs"].get(archive)
        if archive_node is None: raise KeyError(f"archive not found: {archive}")
        entries = archive_node.get("archive_entries")
        if archive_node.get("type") != "file" or not isinstance(entries, dict):
            raise KeyError(f"not a supported archive: {archive}")
        target_dir = self._resolve(session_id, arguments.get("target_dir", "."))
        if self._is_protected(target_dir): raise KeyError("cannot extract into protected paths")
        target_node = state["fs"].get(target_dir)
        if target_node is None or target_node.get("type") != "dir":
            raise KeyError(f"target directory not found: {target_dir}")
        extracted: list[str] = []
        changed = False
        for relative, node in entries.items():
            if not isinstance(relative, str) or relative.startswith("/"):
                raise KeyError(f"unsafe archive entry: {relative!r}")
            destination = posixpath.normpath(posixpath.join(target_dir, relative))
            target_prefix = target_dir.rstrip("/") + "/"
            if destination != target_dir and not destination.startswith(target_prefix):
                raise KeyError(f"unsafe archive entry: {relative!r}")
            if self._is_protected(destination):
                raise KeyError("cannot extract into protected paths")
            if state["fs"].get(destination) != node:
                state["fs"][destination] = copy.deepcopy(node)
                changed = True
            extracted.append(destination)
        return _result(
            True,
            {"archive": archive, "extracted_to": target_dir, "extracted_paths": extracted},
            None,
            "",
            changed,
        )

    def zip(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        return self.tar_create(session_id, arguments)

    def unzip(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        return self.tar_extract(session_id, arguments)

    # Text processing
    def diff(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        n1 = self._node(session_id, arguments["file1"]); n2 = self._node(session_id, arguments["file2"])
        if n1["type"] != "file" or n2["type"] != "file": raise KeyError("both arguments must be files")
        c1 = n1.get("content", "").split("\n"); c2 = n2.get("content", "").split("\n")
        diffs = []
        for i, (l1, l2) in enumerate(zip(c1, c2)):
            if l1 != l2: diffs.append({"line": i + 1, "left": l1, "right": l2})
        for i in range(len(c2), len(c1)): diffs.append({"line": i + 1, "left": c1[i], "right": "<missing>"})
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
                if expr.startswith("s/") and len(l) <= 10000:  # length guard for ReDoS
                    parts = expr[2:].rsplit("/", 2)
                    if len(parts) >= 3:
                        l = re.sub(parts[0], parts[1], l, count=0)
                        # Fall back to no-op on likely catastrophic backtracking
            except re.error:
                pass
            except Exception:
                pass
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
        path = self._resolve(session_id, arguments["path"])
        if self._is_protected(path): raise KeyError("cannot truncate protected paths")
        node = self._node(session_id, path); size = int(arguments["size"])
        if node["type"] != "file": raise KeyError("not a file")
        if size < 0: raise KeyError("size must be non-negative")
        content = node.get("content", "")
        if size < len(content): node["content"] = content[:size]
        else: node["content"] = content + " " * (size - len(content))
        return _result(True, {"path": path, "new_size": len(node["content"])}, None, "", len(content) != len(node["content"]))

    def split(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id); path = self._resolve(session_id, arguments["path"]); lines_per = int(arguments.get("lines_per_file", 100))
        if self._is_protected(path): raise KeyError("cannot split protected paths")
        if lines_per <= 0: raise KeyError("lines_per_file must be positive")
        node = self._node(session_id, path); lines = node.get("content", "").split("\n"); created = []; changed = False
        if node["type"] != "file": raise KeyError("not a file")
        for i in range(0, len(lines), lines_per):
            chunk_path = f"{path}.part{i // lines_per + 1:02d}"
            chunk = {"type": "file", "content": "\n".join(lines[i:i + lines_per]), "permissions": "644", "owner": "user"}
            changed = changed or state["fs"].get(chunk_path) != chunk
            state["fs"][chunk_path] = chunk
            created.append(chunk_path)
        return _result(True, {"source": path, "parts": created, "count": len(created)}, None, "", changed)

    def join(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        n1 = self._node(session_id, arguments["file1"]); n2 = self._node(session_id, arguments["file2"])
        if n1["type"] != "file" or n2["type"] != "file": raise KeyError("both arguments must be files")
        field = int(arguments.get("field", 1)) - 1; c1 = n1.get("content", "").split("\n"); c2 = n2.get("content", "").split("\n")
        map2 = {}
        for l in c2:
            parts = l.split()
            if len(parts) > field: map2[parts[field]] = parts
        joined = []
        for l in c1:
            parts = l.split()
            key = parts[field] if len(parts) > field else None
            if key is not None and key in map2:
                right = [value for idx, value in enumerate(map2[key]) if idx != field]
                joined.append(l + (" " + " ".join(right) if right else ""))
            else:
                joined.append(l)
        return _result(True, {"joined": joined}, None, "", False)


if __name__ == "__main__":
    serve(FilesystemServer())
