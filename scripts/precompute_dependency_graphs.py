#!/usr/bin/env python3
"""Precompute PROVE dependency-graph caches for Live MCP domains."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from loguru import logger

from src.live_mcp.api import LiveMCPBranch
from src.live_mcp.llm_client import LLMClient
from src.live_mcp.orchestrator import TaskOrchestrator


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build schema-hash PROVE dependency graph caches."
    )
    parser.add_argument(
        "--domain",
        default="all",
        help="Domain name, all, or comma-separated list.",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Teacher model name for vLLM/OpenAI-compatible mode or local path.",
    )
    parser.add_argument(
        "--api-base",
        default=None,
        help="OpenAI-compatible API base URL. Uses local transformers if unset.",
    )
    parser.add_argument(
        "--suite",
        default="configs/live_mcp/suite_mvp.yaml",
        help="Suite config path.",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="GPU device ID for local inference.",
    )
    return parser


def _select_domains(branch: LiveMCPBranch, domain_arg: str) -> list[str]:
    if domain_arg == "all":
        return list(branch.manager.server_names)
    domains = [item.strip() for item in domain_arg.split(",") if item.strip()]
    unknown = [domain for domain in domains if domain not in branch.manager.server_names]
    if unknown:
        raise ValueError(f"unknown domains: {unknown}")
    return domains


def main() -> None:
    args = build_arg_parser().parse_args()
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
            branch.suite_config,
            branch.manager,
            branch.executor,
            client,
        )
        domains = _select_domains(branch, args.domain)
        for domain in domains:
            logger.info(f"Precomputing dependency graph cache for {domain}")
            graph = orchestrator._probe_dependency_graph(domain)
            chains = orchestrator._extract_dependency_chains(domain)
            logger.info(
                f"{domain}: graph_nodes={len(graph)} chains={len(chains)}"
            )
    finally:
        branch.stop()


if __name__ == "__main__":
    main()
