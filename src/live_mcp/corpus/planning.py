#!/usr/bin/env python3
"""Plan and optionally launch one provenance-isolated gap-fill generation run."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.live_mcp.corpus.profile import (
    PROVE_PUBLISHED_COUNTS,
    _largest_remainder_ratio_quotas,
    _prove_proxy_bucket,
)
from src.live_mcp.corpus.merge import (
    _as_extra,
    _current_tools,
    _domain_unique_chain_capacity,
    _oracle_calls,
)
from src.live_mcp.domain_allocation import jaccard_unique_sequence_count
from src.live_mcp.contracts.chain_simulator import simulate_symbolic_chain
from src.live_mcp.contracts.factory import build_contract_registry
from src.live_mcp.orchestrator import TaskOrchestrator


PROVE_TARGET_ROWS = 6_761
MAX_DEFAULT_NET_NEW = 700
DEFAULT_CANDIDATE_NUMERATOR = 30
DEFAULT_CANDIDATE_DENOMINATOR = 7
LAUNCHABLE_BUCKETS = ("mcp_conversation", "missing_function")
DOMAINS = (
    "banking",
    "calendar",
    "crm",
    "email",
    "filesystem",
    "food_delivery",
    "issue_tracker",
    "payments",
    "shopping",
    "team_chat",
)
CHAIN_BINS = ("1-2", "3-5", "6+")


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _bucket_capacities(train: pd.DataFrame) -> tuple[dict[str, int], int]:
    counts: Counter[str] = Counter()
    excluded = 0
    for value in train["extra_info"]:
        bucket, _ = _prove_proxy_bucket(value)
        if bucket is None:
            excluded += 1
        else:
            counts[bucket] += 1
    return {
        bucket: int(counts.get(bucket, 0))
        for bucket in PROVE_PUBLISHED_COUNTS
    }, excluded


def _chain_bin(required_count: int) -> str:
    if required_count <= 2:
        return "1-2"
    if required_count <= 5:
        return "3-5"
    return "6+"


def _domain_snapshot(
    train: pd.DataFrame,
) -> tuple[dict[str, dict[str, int]], dict[str, dict[str, int]]]:
    bucket_counts = {
        domain: {bucket: 0 for bucket in PROVE_PUBLISHED_COUNTS}
        for domain in DOMAINS
    }
    chain_counts = {
        domain: {chain_bin: 0 for chain_bin in CHAIN_BINS}
        for domain in DOMAINS
    }
    unknown_domains: Counter[str] = Counter()
    for value in train["extra_info"]:
        bucket, required_count = _prove_proxy_bucket(value)
        extra = value if isinstance(value, dict) else json.loads(value)
        domain = str(extra.get("domain") or "unknown")
        if domain not in bucket_counts:
            unknown_domains[domain] += 1
            continue
        if bucket is not None:
            bucket_counts[domain][bucket] += 1
        if bucket == "mcp_conversation":
            chain_counts[domain][_chain_bin(required_count)] += 1
    if unknown_domains:
        raise RuntimeError(
            f"corpus contains domains outside the ten-domain suite: "
            f"{dict(unknown_domains)}"
        )
    return bucket_counts, chain_counts


def _as_string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple, set)):
        return []
    return [str(item) for item in value if str(item)]


def _strict_cache_unique_chain_capacity(
    domain: str,
    tools: list[dict[str, Any]],
) -> int | None:
    schema_hash = TaskOrchestrator._tool_schema_hash(tools, domain)
    path = (
        PROJECT_ROOT / "data" / "dependency_graphs"
        / f"{domain}_{schema_hash}.json"
    )
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    names = sorted(str(tool.get("name") or "") for tool in tools)
    expected_pairs = len(names) * (len(names) - 1) // 2
    if (
        data.get("server_name") != domain
        or data.get("schema_hash") != schema_hash
        or data.get("cache_version")
        != TaskOrchestrator.DEPENDENCY_CACHE_VERSION
        or data.get("dependency_semantics_version")
        != TaskOrchestrator.DEPENDENCY_SEMANTICS_VERSION
        or data.get("tool_names") != names
        or data.get("classified_pair_count") != expected_pairs
        or data.get("classification_complete") is not True
    ):
        return None
    graph = data.get("graph")
    if not TaskOrchestrator._valid_cached_graph(graph, names):
        return None

    chains: list[list[str]] = []
    contract_registry = build_contract_registry({domain: tools})

    def visit(current: str, path_items: list[str], seen: set[str]) -> None:
        if len(path_items) >= 5:
            return
        node = graph.get(current, {})
        neighbors = list(node.get("explicit", [])) + list(
            node.get("implicit", [])
        )
        for neighbor in neighbors:
            if neighbor in seen:
                continue
            candidate = path_items + [neighbor]
            if (
                len(candidate) >= 2
                and not simulate_symbolic_chain(
                    contract_registry, domain, candidate,
                )[1]
            ):
                chains.append(candidate)
            visit(neighbor, candidate, seen | {neighbor})

    for start in graph:
        visit(start, [start], {start})
    return jaccard_unique_sequence_count(chains, threshold=0.70)


def _capacity_evidence(train: pd.DataFrame) -> dict[str, dict[str, Any]]:
    tools = _current_tools()
    mcp_mask = train["extra_info"].map(
        lambda value: _prove_proxy_bucket(value)[0] == "mcp_conversation"
    )
    unique_chain_capacity = _domain_unique_chain_capacity(
        train.loc[mcp_mask],
        list(DOMAINS),
        threshold=0.70,
    )
    called_tools: dict[str, set[str]] = defaultdict(set)
    represented_hidden_tools: dict[str, set[str]] = defaultdict(set)
    for _, row in train.iterrows():
        extra = _as_extra(row["extra_info"])
        domain = str(extra.get("domain") or "")
        if domain not in DOMAINS:
            continue
        bucket, _ = _prove_proxy_bucket(extra)
        if bucket == "mcp_conversation":
            called_tools[domain].update(
                str(call.get("tool_name") or "")
                for call in _oracle_calls(extra)
                if str(call.get("action") or "tool_call") == "tool_call"
                and str(call.get("tool_name") or "")
            )
        elif bucket == "missing_function":
            represented_hidden_tools[domain].update(
                _as_string_list(extra.get("hidden_tools", []))
            )

    evidence: dict[str, dict[str, Any]] = {}
    for domain in DOMAINS:
        observed_unique_chains = unique_chain_capacity[domain]
        certified_unique_chains = _strict_cache_unique_chain_capacity(
            domain,
            tools[domain],
        )
        unique_chains = (
            certified_unique_chains
            if certified_unique_chains is not None
            else observed_unique_chains
        )
        tool_count = len(tools[domain])
        uncovered_hidden = sorted(
            called_tools[domain] - represented_hidden_tools[domain]
        )
        evidence[domain] = {
            "tool_count": tool_count,
            "observed_jaccard_unique_mcp_chains": observed_unique_chains,
            "certified_live_feasible_jaccard_unique_seed_chains": (
                certified_unique_chains
            ),
            "mcp_structural_weight": math.sqrt(tool_count * unique_chains),
            "called_tools_in_mcp": len(called_tools[domain]),
            "represented_hidden_tools": len(
                represented_hidden_tools[domain]
            ),
            "uncovered_hideable_tools": uncovered_hidden,
            "missing_function_weight": len(uncovered_hidden),
            "capacity_certification": (
                "current_strict_cache"
                if certified_unique_chains is not None
                else "observed_corpus_provisional"
            ),
        }
    return evidence


def _increment_quotas(
    *,
    gaps: dict[str, int],
    evidence: dict[str, dict[str, Any]],
) -> dict[str, dict[str, int]]:
    mcp_weights = {
        domain: float(evidence[domain]["mcp_structural_weight"])
        for domain in DOMAINS
    }
    missing_weights = {
        domain: int(evidence[domain]["missing_function_weight"])
        for domain in DOMAINS
    }
    if sum(missing_weights.values()) <= 0:
        missing_weights = {
            domain: int(evidence[domain]["tool_count"])
            for domain in DOMAINS
        }
    mcp = _largest_remainder_ratio_quotas(
        mcp_weights,
        gaps["mcp_conversation"],
    )
    missing = _largest_remainder_ratio_quotas(
        missing_weights,
        gaps["missing_function"],
    )
    return {
        domain: {
            "mcp_conversation": mcp[domain],
            "missing_function": missing[domain],
            # No paper-defined per-domain quota and no strict external source.
            "internal_abstention_proxy": 0,
        }
        for domain in DOMAINS
    }


def _project_chain_targets(
    chain_counts: dict[str, dict[str, int]],
    *,
    mcp_target_per_domain: dict[str, int],
) -> dict[str, dict[str, dict[str, int]]]:
    global_counts: Counter[str] = Counter()
    for counts in chain_counts.values():
        global_counts.update(counts)
    if sum(global_counts.values()) <= 0:
        raise RuntimeError("current corpus contains no MCP chain-length evidence")

    result: dict[str, dict[str, dict[str, int]]] = {}
    for domain in DOMAINS:
        current = chain_counts[domain]
        target_total = mcp_target_per_domain[domain]
        remaining_bins = set(CHAIN_BINS)
        fixed: dict[str, int] = {}
        while remaining_bins:
            remaining_total = target_total - sum(fixed.values())
            remaining_weight = sum(global_counts[key] for key in remaining_bins)
            projected = {
                key: remaining_total * global_counts[key] / remaining_weight
                for key in remaining_bins
            }
            below_current = {
                key for key in remaining_bins
                if projected[key] < current[key]
            }
            if not below_current:
                break
            for key in below_current:
                fixed[key] = current[key]
            remaining_bins -= below_current

        exact = {
            key: float(fixed.get(
                key,
                (
                    target_total - sum(fixed.values())
                ) * global_counts[key]
                / sum(global_counts[item] for item in remaining_bins),
            ))
            for key in CHAIN_BINS
        }
        final = {key: max(current[key], int(exact[key])) for key in CHAIN_BINS}
        remaining = target_total - sum(final.values())
        order = sorted(
            CHAIN_BINS,
            key=lambda key: (-(exact[key] - int(exact[key])), key),
        )
        for key in order[:remaining]:
            final[key] += 1
        result[domain] = {
            "current": dict(current),
            "target": final,
            "gap": {
                key: max(0, final[key] - current[key])
                for key in CHAIN_BINS
            },
        }
    return result


def _select_domain(
    domain_plans: dict[str, dict[str, Any]],
    bucket: str,
) -> str | None:
    candidates = [
        domain for domain, plan in domain_plans.items()
        if plan["gaps"].get(bucket, 0) > 0
    ]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda domain: (
            domain_plans[domain]["gaps"][bucket],
            domain_plans[domain]["capacity_evidence"].get(
                "mcp_structural_weight", 0,
            ),
            domain,
        ),
    )


def _batch_chain_quotas(
    chain_gaps: dict[str, int],
    target: int,
) -> dict[str, int]:
    weights = {
        key: max(0, int(chain_gaps.get(key, 0)))
        for key in CHAIN_BINS
    }
    if target <= 0:
        return {key: 0 for key in CHAIN_BINS}
    if sum(weights.values()) < target:
        raise ValueError(
            f"chain gaps {weights} cannot cover batch target {target}"
        )
    return _largest_remainder_ratio_quotas(weights, target)


def _select_bucket(gaps: dict[str, int]) -> str | None:
    for bucket in LAUNCHABLE_BUCKETS:
        if gaps.get(bucket, 0) > 0:
            return bucket
    return None


def _generation_args(bucket: str) -> list[str]:
    common = ["--irrelevance-ratio", "0"]
    if bucket == "mcp_conversation":
        return [
            "--tool-required-only",
            "--missing-function-rate",
            "0",
            *common,
        ]
    if bucket == "missing_function":
        return ["--missing-function-rate", "1", *common]
    raise ValueError(f"unsupported generation bucket: {bucket}")


def build_plan(
    *,
    input_dir: Path,
    max_net_new: int,
    candidate_numerator: int,
    candidate_denominator: int,
    seed: int,
    run_id: str | None,
) -> dict[str, Any]:
    train_path = input_dir / "train.parquet"
    val_path = input_dir / "val.parquet"
    for path in (train_path, val_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not any(
        (input_dir / name).is_file()
        for name in ("merge_report.json", "recertification_report.json")
    ):
        raise FileNotFoundError(
            input_dir / "merge_report.json|recertification_report.json"
        )

    train = pd.read_parquet(train_path)
    capacities, excluded = _bucket_capacities(train)
    targets = _largest_remainder_ratio_quotas(
        PROVE_PUBLISHED_COUNTS,
        PROVE_TARGET_ROWS,
    )
    gaps = {
        bucket: max(0, targets[bucket] - capacities[bucket])
        for bucket in targets
    }
    domain_capacities, chain_counts = _domain_snapshot(train)
    capacity_evidence = _capacity_evidence(train)
    increment_quotas = _increment_quotas(
        gaps=gaps,
        evidence=capacity_evidence,
    )
    domain_plans: dict[str, dict[str, Any]] = {}
    for domain in DOMAINS:
        domain_gaps = increment_quotas[domain]
        domain_targets = {
            bucket: domain_capacities[domain][bucket] + domain_gaps[bucket]
            for bucket in targets
        }
        domain_plans[domain] = {
            "capacities": domain_capacities[domain],
            "targets": domain_targets,
            "gaps": domain_gaps,
            "capacity_evidence": capacity_evidence[domain],
            "mapped_capacity": sum(domain_capacities[domain].values()),
            "mapped_target": sum(domain_targets.values()),
            "total_gap": sum(domain_gaps.values()),
            "launchable_gap": sum(
                domain_gaps[bucket] for bucket in LAUNCHABLE_BUCKETS
            ),
            "blocked_gap": domain_gaps["internal_abstention_proxy"],
        }
    chain_plan = _project_chain_targets(
        chain_counts,
        mcp_target_per_domain={
            domain: domain_plans[domain]["targets"]["mcp_conversation"]
            for domain in DOMAINS
        },
    )
    for domain in DOMAINS:
        domain_plans[domain]["mcp_chain_bins"] = chain_plan[domain]

    selected_bucket = _select_bucket(gaps)
    selected_domain = (
        _select_domain(domain_plans, selected_bucket)
        if selected_bucket else None
    )

    plan: dict[str, Any] = {
        "created_at": datetime.now().astimezone().isoformat(),
        "input_dir": str(input_dir),
        "train_rows": len(train),
        "excluded_unmapped_rows": excluded,
        "target_rows": PROVE_TARGET_ROWS,
        "targets": targets,
        "capacities": capacities,
        "gaps": gaps,
        "domain_policy": {
            "kind": "capacity_weighted_incremental_gap",
            "paper_published": False,
            "domains": list(DOMAINS),
            "chain_bins": list(CHAIN_BINS),
            "serial_one_domain_per_run": True,
            "mcp_weight": "sqrt(tool_count * observed_jaccard_unique_chains)",
            "missing_function_weight": "uncovered_hideable_tools",
            "raw_entity_count_is_not_a_direct_weight": True,
            "allocation_recomputed_after_each_merge": True,
        },
        "domains": domain_plans,
        "strict_external_abstention_available": False,
        "internal_abstention_proxy_note": (
            "The current generator cannot guarantee an abstention-only shard "
            "across recovery/top-up rounds, so this planner does not launch "
            "that bucket automatically."
        ),
        "selected_domain": selected_domain,
        "selected_bucket": selected_bucket,
        "status": "planned" if selected_bucket else "no_launchable_gap",
    }
    if selected_bucket is None:
        return plan

    net_new = min(max_net_new, gaps[selected_bucket])
    net_new = min(
        net_new,
        domain_plans[selected_domain]["gaps"][selected_bucket],
    )
    candidate_budget = max(
        net_new,
        (
            net_new * candidate_numerator
            + candidate_denominator
            - 1
        ) // candidate_denominator,
    )
    resolved_run_id = run_id or (
        f"{datetime.now():%Y%m%d}_gapfill_"
        f"{selected_domain}_{selected_bucket}_net{net_new}_cand{candidate_budget}"
    )
    relative_input_dir = input_dir.relative_to(PROJECT_ROOT)
    output_dir = Path("data") / "runs" / resolved_run_id
    command = [
        sys.executable,
        "-m",
        "src.live_mcp.corpus.cli",
        "run",
        "--mode",
        "supplement",
        "--net-new",
        str(net_new),
        "--candidate-budget",
        str(candidate_budget),
        "--base",
        str(relative_input_dir),
        "--seed",
        str(seed),
        "--run-id",
        resolved_run_id,
        "--bucket",
        selected_bucket,
        "--domain",
        selected_domain,
    ]
    batch_chain_quotas = None
    if selected_bucket == "mcp_conversation":
        batch_chain_quotas = _batch_chain_quotas(
            domain_plans[selected_domain]["mcp_chain_bins"]["gap"],
            net_new,
        )
    plan["launch"] = {
        "domain": selected_domain,
        "bucket": selected_bucket,
        "net_new_target": net_new,
        "candidate_budget": candidate_budget,
        "candidate_multiplier": (
            f"{candidate_numerator}/{candidate_denominator}"
        ),
        "chain_bin_diagnostic_target": batch_chain_quotas,
        "chain_bin_policy": "monitor_only",
        "run_id": resolved_run_id,
        "output_dir": str(output_dir),
        "command": command,
        "command_display": shlex.join(command),
    }
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--max-net-new", type=int, default=MAX_DEFAULT_NET_NEW)
    parser.add_argument(
        "--candidate-numerator",
        type=int,
        default=DEFAULT_CANDIDATE_NUMERATOR,
    )
    parser.add_argument(
        "--candidate-denominator",
        type=int,
        default=DEFAULT_CANDIDATE_DENOMINATOR,
    )
    parser.add_argument("--seed", type=int, default=2026072801)
    parser.add_argument("--run-id")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    if args.max_net_new < 1:
        parser.error("--max-net-new must be >= 1")
    if args.candidate_numerator < 1 or args.candidate_denominator < 1:
        parser.error("candidate multiplier terms must be >= 1")

    input_dir = args.input_dir.resolve()
    report_path = args.report.resolve()
    plan = build_plan(
        input_dir=input_dir,
        max_net_new=args.max_net_new,
        candidate_numerator=args.candidate_numerator,
        candidate_denominator=args.candidate_denominator,
        seed=args.seed,
        run_id=args.run_id,
    )
    _atomic_write_json(report_path, plan)
    print(json.dumps(plan, indent=2, sort_keys=True), flush=True)

    if not args.execute or "launch" not in plan:
        return

    output_dir = Path(plan["launch"]["output_dir"])
    if output_dir.exists():
        raise FileExistsError(
            f"refusing duplicate launch; output already exists: {output_dir}"
        )
    plan["status"] = "running"
    _atomic_write_json(report_path, plan)
    env = os.environ.copy()
    env.setdefault("GPU_COUNT", "4")
    env.setdefault("GEN_OVERSAMPLE_PCT", "0")
    completed = subprocess.run(
        plan["launch"]["command"],
        cwd=PROJECT_ROOT,
        env=env,
        check=False,
    )
    plan["status"] = "completed" if completed.returncode == 0 else "failed"
    plan["returncode"] = completed.returncode
    plan["finished_at"] = datetime.now().astimezone().isoformat()
    _atomic_write_json(report_path, plan)
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
