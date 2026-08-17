"""Merge Validation."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.live_mcp.protocol.observation import (
    TRAJECTORY_SCHEMA_VERSION, OBSERVATION_SCHEMA_VERSION,
    OBSERVATION_PROJECTION_VERSION, compute_server_schema_hash,
)
from src.live_mcp.registry.tool_semantics import resolve_tool_operation
from src.live_mcp.corpus.local_quality import (
    evaluate_persisted_candidate_quality,
)
from src.live_mcp.artifact.validation import validate_artifact_contract

DOMAINS_ALL = [
    "banking", "calendar", "crm", "email", "filesystem",
    "food_delivery", "issue_tracker", "payments", "shopping", "team_chat",
]

_LEAK_MARKERS = (
    "oracle_calls",
    "success_criteria",
    "ground_truth",
    "allowed_terminal_actions",
    "hidden_tools",
)

class FatalShardIntegrityError(RuntimeError):
    """A whole non-empty shard split is incompatible with this runtime."""

    def __init__(self, pattern: str, row_count: int, issue: str) -> None:
        self.pattern = pattern
        self.row_count = row_count
        self.issue = issue
        super().__init__(
            f"all {row_count} rows from {pattern} failed the same "
            f"environment contract: {issue}"
        )

def _as_extra(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        return json.loads(value)
    return {}

def _as_json_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if hasattr(value, "tolist"):
        return list(value.tolist())
    if isinstance(value, str):
        return json.loads(value)
    return list(value) if isinstance(value, tuple) else []

def _oracle_calls(extra: dict[str, Any]) -> list[dict[str, Any]]:
    calls = _as_json_list(extra.get("oracle_calls", []))
    return [call for call in calls if isinstance(call, dict)]

def _current_tools() -> dict[str, list[dict[str, Any]]]:
    import importlib

    return {
        domain: list(
            importlib.import_module(
                f"src.live_mcp.servers.{domain}.server"
            ).TOOLS
        )
        for domain in DOMAINS_ALL
    }

def _current_schema_hashes() -> dict[str, str]:
    return {
        domain: compute_server_schema_hash(tools)
        for domain, tools in _current_tools().items()
    }

def _runtime_observation_budget() -> int:
    from src.live_mcp.config import load_suite_config

    suite_path = os.environ.get(
        "OVAL_SUITE_PATH", "configs/live_mcp/ten_domain_suite.yaml",
    )
    suite = load_suite_config(suite_path)
    return int(suite.rollout.get("observation_max_chars", 4096))

def _row_fingerprint(row: pd.Series) -> str:
    extra = _as_extra(row["extra_info"])
    domain = extra.get("domain", "")
    query = " ".join((extra.get("user_query", "") or "").lower().split())
    calls = _oracle_calls(extra)
    sig = json.dumps(
        {"d": domain, "q": query, "c": calls},
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    )
    return hashlib.sha256(sig.encode()).hexdigest()

def _quality_issue(row: pd.Series) -> str:
    extra = _as_extra(row["extra_info"])
    try:
        reward_model = row.get("reward_model") or {}
        ground_truth = (
            reward_model.get("ground_truth")
            if isinstance(reward_model, dict) else None
        )
        validate_artifact_contract(
            extra,
            require_training=False,
            ground_truth=ground_truth,
        )
    except Exception as exc:
        if str(exc).startswith((
            "ground_truth ",
            "canonical task/ground_truth ",
        )):
            return f"ground_truth_contract_invalid:{exc}"
        return f"artifact_contract_invalid:{exc}"
    recorded_schema = str(extra.get("trajectory_schema_version") or "")
    if not recorded_schema:
        return "missing_environment_metadata:trajectory_schema_version"
    if recorded_schema != TRAJECTORY_SCHEMA_VERSION:
        return f"stale_trajectory_schema:{recorded_schema}"
    owner_domains = extra.get("tool_owner_domains", {})
    if isinstance(owner_domains, str):
        try:
            owner_domains = json.loads(owner_domains)
        except json.JSONDecodeError:
            return "tool_owner_domains_invalid"
    primary_domain = str(extra.get("domain") or "")
    required_owners = {
        primary_domain,
        *(str(owner) for owner in (
            owner_domains.values() if isinstance(owner_domains, dict) else []
        )),
    }
    required_owners.discard("")
    try:
        from src.live_mcp.registry.environment_metadata import (
            compute_initial_state_hashes,
            normalize_state_profiles,
            validate_environment_metadata,
        )
        validate_environment_metadata(
            extra,
            current_tools_by_domain={
                owner: _current_tools()[owner]
                for owner in required_owners if owner in _current_tools()
            },
            required_owner_domains=required_owners,
            reward_profile="prove_baseline",
            runtime_max_observation_chars=_runtime_observation_budget(),
            current_initial_state_hashes=compute_initial_state_hashes(
                required_owners,
                int(extra["session_seed"]),
                normalize_state_profiles(
                    extra.get("state_profiles"), required_owners
                ),
            ),
        )
    except Exception as exc:
        return f"environment_metadata_invalid:{exc}"
    required_fields = (
        "observation_schema_version",
        "observation_projection_version",
        "server_schema_hash",
        "server_schema_hashes",
        "initial_state_hash",
        "max_observation_chars",
    )
    for field in required_fields:
        value = extra.get(field)
        if value is None or value == "" or value == {}:
            return f"missing_environment_metadata:{field}"
    recorded_observation_schema = str(
        extra.get("observation_schema_version") or ""
    )
    if recorded_observation_schema != OBSERVATION_SCHEMA_VERSION:
        return (
            "stale_observation_schema:"
            f"{recorded_observation_schema}"
        )
    recorded_projection = str(extra.get("observation_projection_version") or "")
    if recorded_projection != OBSERVATION_PROJECTION_VERSION:
        return f"stale_projection:{recorded_projection}"
    recorded_hashes = extra.get("server_schema_hashes", {})
    if isinstance(recorded_hashes, str):
        try:
            recorded_hashes = json.loads(recorded_hashes)
        except json.JSONDecodeError:
            return "server_schema_hashes_invalid"
    if not isinstance(recorded_hashes, dict) or not recorded_hashes:
        return "missing_environment_metadata:server_schema_hashes"
    if isinstance(recorded_hashes, dict):
        current_hashes = _current_schema_hashes()
        stale_domains = sorted(
            domain for domain, recorded in recorded_hashes.items()
            if domain in current_hashes and str(recorded) != current_hashes[domain]
        )
        if stale_domains:
            return f"stale_server_schema:{stale_domains}"
    scenario = str(row.get("scenario_type") or extra.get("scenario_type") or "")
    calls = _oracle_calls(extra)
    # Filter completed traces by replay error rate, provenance, and
    # sequence similarity. Scenario labels are metadata, not a rejection gate;
    # recovery may legitimately end in final_answer or graceful report_error.

    hidden = set(_as_json_list(extra.get("hidden_tools", [])))
    visible = set(_as_json_list(extra.get("visible_tool_names", [])))
    overlap = hidden & visible
    if overlap:
        return f"hidden_tool_visible:{sorted(overlap)}"

    quality_finding = evaluate_persisted_candidate_quality(extra)
    if quality_finding is not None:
        return quality_finding.quality_issue

    try:
        prompt = json.loads(row["prompt"])
    except Exception:
        return "prompt_json_invalid"
    prompt_text = "\n".join(str(message.get("content", "")) for message in prompt)
    if any(marker in prompt_text for marker in _LEAK_MARKERS):
        return "prompt_leaks_training_target"

    return ""


def _row_tool_sequence(row: pd.Series, mode: str = "prove") -> list[str]:
    """Tool-call sequence used for Jaccard dedup.

    mode="prove": plain tool-name sequence — the PROVE hard-gate definition.
    mode="local": enriched signature ``server::cl<N>::tool::op`` — a LOCAL
        diagnostic only.  Hidden source-chain metadata must not alter the
        paper gate.
    """
    extra = _as_extra(row["extra_info"])
    primary_domain = str(extra.get("domain") or "")
    canonical_calls = [
        call for call in _oracle_calls(extra)
        if call.get("action", "tool_call") == "tool_call"
    ]
    if mode == "prove":
        names = [
            str(call.get("tool_name", ""))
            for call in canonical_calls if call.get("tool_name")
        ]
        if names:
            return names
        raw_seq = extra.get("teacher_trace_tool_sequence", [])
        if isinstance(raw_seq, str):
            try:
                raw_seq = json.loads(raw_seq)
            except (json.JSONDecodeError, TypeError):
                raw_seq = []
        elif hasattr(raw_seq, "tolist"):
            raw_seq = raw_seq.tolist()
        if isinstance(raw_seq, (list, tuple)) and raw_seq:
            return [str(name) for name in raw_seq if str(name)]
        return []
    # Include source-chain length in the Jaccard token so that two candidates
    # from chains of different length cannot collide even when the Teacher adds
    # identical auxiliary discovery calls.  Also annotate each call with its
    # handler-level operation type (read / create / update / delete / execute)
    # for additional position-level discrimination.
    source_chain = _as_json_list(extra.get("source_chain_seed"))
    chain_len_tag = f"cl{len(source_chain)}" if source_chain else "cl0"
    if canonical_calls:
        result: list[str] = []
        for call in canonical_calls:
            tool_name = str(call.get("tool_name", ""))
            try:
                op = resolve_tool_operation(tool_name, primary_domain)
            except (ValueError, KeyError):
                op = "call"
            server = str(call.get("server_name") or primary_domain)
            result.append(f"{server}::{chain_len_tag}::{tool_name}::{op}")
        return result
    raw_sequence = extra.get("teacher_trace_tool_sequence", [])
    if isinstance(raw_sequence, str):
        try:
            raw_sequence = json.loads(raw_sequence)
        except (json.JSONDecodeError, TypeError):
            raw_sequence = []
    elif hasattr(raw_sequence, "tolist"):
        raw_sequence = raw_sequence.tolist()
    if isinstance(raw_sequence, (list, tuple)) and raw_sequence:
        return [f"{primary_domain}::{chain_len_tag}::{name}::call" for name in raw_sequence]
    return []
