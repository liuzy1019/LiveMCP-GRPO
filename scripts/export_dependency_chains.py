"""Export current dependency chains for manual review."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.live_mcp.api import LiveMCPBranch
from src.live_mcp.orchestrator import TaskOrchestrator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="configs/live_mcp/suite_mvp.yaml")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    branch = LiveMCPBranch.from_suite(args.suite)
    branch.start()
    lines: list[str] = []
    try:
        assert branch.executor is not None
        orchestrator = TaskOrchestrator(
            branch.suite_config,
            branch.manager,
            branch.executor,
            None,
        )
        for domain in branch.manager.server_names:
            graph = orchestrator._probe_dependency_graph(domain)
            orchestrator._domain_graphs[domain] = graph
            chains = orchestrator._extract_dependency_chains(domain)
            lines.append(f"## {domain} ({len(chains)} chains)")
            for idx, chain in enumerate(chains, 1):
                lines.append(f"{idx:03d}. {' -> '.join(chain)}")
            lines.append("")
    finally:
        branch.stop()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
