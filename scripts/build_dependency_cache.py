#!/usr/bin/env python3
"""Build PROVE dependency-graph caches via pairwise LLM classification.

Usage:
  # Live: probe servers with LLM to build dependency graphs
  python scripts/build_dependency_cache.py --domain all --model gemini-2.5-flash --api-base https://proxy/v1

"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from loguru import logger

from src.live_mcp.generation_runtime import TeacherGenerationRuntime
from src.live_mcp.llm_client import LLMClient
from src.live_mcp.orchestrator import TaskOrchestrator


# ═══════════════════════════════════════════════════════════════════
# shared helpers
# ═══════════════════════════════════════════════════════════════════

def _select_domains(
    runtime: TeacherGenerationRuntime, domain_arg: str,
) -> list[str]:
    if domain_arg == "all":
        return list(runtime.manager.server_names)
    domains = [item.strip() for item in domain_arg.split(",") if item.strip()]
    unknown = [d for d in domains if d not in runtime.manager.server_names]
    if unknown:
        raise ValueError(f"unknown domains: {unknown}")
    return domains


# ═══════════════════════════════════════════════════════════════════
# live mode
# ═══════════════════════════════════════════════════════════════════

def build_dependency_caches(args: argparse.Namespace) -> None:
    runtime = TeacherGenerationRuntime.from_suite(args.suite)
    runtime.start()
    try:
        client = (
            LLMClient(mode="openai", model_path=args.model, api_base=args.api_base)
            if args.api_base
            else LLMClient(mode="local", model_path=args.model, device=args.device)
        )
        assert runtime.executor is not None
        orchestrator = TaskOrchestrator(
            runtime.suite_config, runtime.manager, runtime.executor, client,
        )
        domains = _select_domains(runtime, args.domain)
        def build_domain_cache(domain: str) -> tuple[str, int, int]:
            logger.info(f"Building dependency cache for {domain}")
            graph = orchestrator._get_or_build_dependency_graph(domain)
            chains = orchestrator._extract_dependency_chains(domain)
            return domain, len(graph), len(chains)

        workers = min(max(1, args.workers), len(domains))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(build_domain_cache, domain): domain
                for domain in domains
            }
            for future in as_completed(futures):
                domain, graph_nodes, chain_count = future.result()
                logger.info(
                    f"{domain}: graph_nodes={graph_nodes} chains={chain_count}"
                )
    finally:
        runtime.stop()


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build PROVE dependency-graph caches.",
    )
    parser.add_argument("--domain", default="all",
                        help="Domain name, all, or comma-separated list.")
    parser.add_argument("--model", required=True,
                        help="Teacher model name/path.")
    parser.add_argument("--api-base", default=None,
                        help="OpenAI-compatible API base URL. Uses local transformers if unset.")
    parser.add_argument("--suite", default="configs/live_mcp/ten_domain_suite.yaml",
                        help="Suite config path.")
    parser.add_argument("--device", type=int, default=None,
                        help="GPU device ID for local inference.")
    parser.add_argument("--workers", type=int, default=4,
                        help="Domains classified concurrently (default: 4).")

    args = parser.parse_args()
    build_dependency_caches(args)


if __name__ == "__main__":
    main()
