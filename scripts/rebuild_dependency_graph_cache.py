"""Rebuild dependency graph cache from an existing raw/cache snapshot."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.live_mcp.api import LiveMCPBranch
from src.live_mcp.orchestrator import TaskOrchestrator


class _NoClient:
    pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--dest", default=Path("data/dependency_graphs"), type=Path)
    parser.add_argument("--backup", type=Path)
    args = parser.parse_args()

    if args.backup and args.dest.exists():
        if args.backup.exists():
            raise SystemExit(f"backup already exists: {args.backup}")
        shutil.move(str(args.dest), str(args.backup))
    args.dest.mkdir(parents=True, exist_ok=True)

    branch = LiveMCPBranch.from_suite("configs/live_mcp/suite_mvp.yaml")
    branch.start()
    try:
        assert branch.executor is not None
        orchestrator = TaskOrchestrator(branch.suite_config, branch.manager, branch.executor, _NoClient())
        rebuilt: list[tuple[str, int, int]] = []
        for src_path in sorted(args.source.glob("*.json")):
            raw = json.loads(src_path.read_text(encoding="utf-8"))
            server = raw.get("server_name")
            tool_names = raw.get("tool_names")
            schema_hash = raw.get("schema_hash")
            if not server or not tool_names:
                raise SystemExit(f"missing server_name/tool_names in {src_path}")
            if not schema_hash:
                raise SystemExit(f"missing schema_hash in {src_path}")
            server_tools = branch.manager.registry.server_tools(server)
            graph = TaskOrchestrator._normalize_cached_graph(raw.get("graph", {}), tool_names)
            TaskOrchestrator._apply_prove_dependency_definition_filter(graph, server_tools, server)
            out_path = orchestrator._graph_cache_path(server, schema_hash)
            out_path = args.dest / out_path.name
            out = {
                "server_name": server,
                "schema_hash": schema_hash,
                "tool_names": tool_names,
                "graph": graph,
            }
            out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
            edges = sum(len(v.get("implicit", [])) + len(v.get("explicit", [])) for v in graph.values())
            rebuilt.append((server, len(graph), edges))
    finally:
        branch.stop()

    for server, nodes, edges in rebuilt:
        print(f"{server}: nodes={nodes} edges={edges}")


if __name__ == "__main__":
    main()
