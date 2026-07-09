#!/usr/bin/env python3
"""PROVE dependency-graph cache management for Live MCP domains.

Two modes:
  live    — Precompute dependency graphs via LLM probing (requires model).
  rebuild — Rebuild graphs from raw cache snapshots (offline, no model).

Usage:
  # Live: probe servers with LLM to build dependency graphs
  python scripts/dependency_graph.py live --domain all --model gemini-2.5-flash --api-base https://proxy/v1

  # Rebuild: reconstruct filtered graphs from raw cache snapshots
  python scripts/dependency_graph.py rebuild --source data/dependency_graphs_raw --dest data/dependency_graphs
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from loguru import logger

from src.live_mcp.api import LiveMCPBranch
from src.live_mcp.llm_client import LLMClient
from src.live_mcp.orchestrator import TaskOrchestrator


# ═══════════════════════════════════════════════════════════════════
# shared helpers
# ═══════════════════════════════════════════════════════════════════

def _select_domains(branch: LiveMCPBranch, domain_arg: str) -> list[str]:
    if domain_arg == "all":
        return list(branch.manager.server_names)
    domains = [item.strip() for item in domain_arg.split(",") if item.strip()]
    unknown = [d for d in domains if d not in branch.manager.server_names]
    if unknown:
        raise ValueError(f"unknown domains: {unknown}")
    return domains


# ═══════════════════════════════════════════════════════════════════
# live mode
# ═══════════════════════════════════════════════════════════════════

def cmd_live(args: argparse.Namespace) -> None:
    branch = LiveMCPBranch.from_suite(args.suite)
    branch.start()
    try:
        client = (
            LLMClient(mode="openai", model_path=args.model, api_base=args.api_base)
            if args.api_base
            else LLMClient(mode="local", model_path=args.model, device=args.device)
        )
        assert branch.executor is not None
        orchestrator = TaskOrchestrator(
            branch.suite_config, branch.manager, branch.executor, client,
        )
        domains = _select_domains(branch, args.domain)
        for domain in domains:
            logger.info(f"Precomputing dependency graph cache for {domain}")
            graph = orchestrator._probe_dependency_graph(domain)
            chains = orchestrator._extract_dependency_chains(domain)
            logger.info(f"{domain}: graph_nodes={len(graph)} chains={len(chains)}")
    finally:
        branch.stop()


# ═══════════════════════════════════════════════════════════════════
# rebuild mode
# ═══════════════════════════════════════════════════════════════════

class _NoClient:
    pass


def cmd_rebuild(args: argparse.Namespace) -> None:
    source = Path(args.source)
    dest = Path(args.dest) if args.dest else Path("data/dependency_graphs")

    if args.backup and dest.exists():
        backup = Path(args.backup)
        if backup.exists():
            raise SystemExit(f"backup already exists: {backup}")
        shutil.move(str(dest), str(backup))
    dest.mkdir(parents=True, exist_ok=True)

    branch = LiveMCPBranch.from_suite(args.suite)
    branch.start()
    try:
        assert branch.executor is not None
        orchestrator = TaskOrchestrator(
            branch.suite_config, branch.manager, branch.executor, _NoClient(),
        )
        rebuilt: list[tuple[str, int, int]] = []
        for src_path in sorted(source.glob("*.json")):
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
            out_path = dest / out_path.name
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


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="PROVE dependency-graph cache management.",
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    # live
    p_live = sub.add_parser("live", help="Precompute dependency graphs via LLM probing")
    p_live.add_argument("--domain", default="all",
                        help="Domain name, all, or comma-separated list.")
    p_live.add_argument("--model", required=True,
                        help="Teacher model name/path.")
    p_live.add_argument("--api-base", default=None,
                        help="OpenAI-compatible API base URL. Uses local transformers if unset.")
    p_live.add_argument("--suite", default="configs/live_mcp/suite_mvp.yaml",
                        help="Suite config path.")
    p_live.add_argument("--device", type=int, default=None,
                        help="GPU device ID for local inference.")

    # rebuild
    p_rebuild = sub.add_parser("rebuild", help="Rebuild graphs from raw cache snapshots")
    p_rebuild.add_argument("--source", required=True, type=Path,
                           help="Directory containing raw cache JSON files.")
    p_rebuild.add_argument("--dest", default=Path("data/dependency_graphs"), type=Path,
                           help="Output directory (default: data/dependency_graphs).")
    p_rebuild.add_argument("--backup", type=Path,
                           help="Backup existing dest directory before overwriting.")
    p_rebuild.add_argument("--suite", default="configs/live_mcp/suite_mvp.yaml",
                           help="Suite config path.")

    args = parser.parse_args()

    if args.mode == "live":
        cmd_live(args)
    elif args.mode == "rebuild":
        cmd_rebuild(args)


if __name__ == "__main__":
    main()
