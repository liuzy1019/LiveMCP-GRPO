"""Merge Validation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from collections import defaultdict
from collections.abc import Hashable
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.live_mcp.protocol.observation import (
    TRAJECTORY_SCHEMA_VERSION, OBSERVATION_SCHEMA_VERSION,
    OBSERVATION_PROJECTION_VERSION, compute_server_schema_hash,
)
from src.live_mcp.domain_allocation import (
    capacity_weighted_domain_quotas, position_aware_jaccard,
)
from src.live_mcp.registry.tool_semantics import (
    is_mutating_tool, resolve_tool_operation, unresolved_failed_tool_names,
)
from src.live_mcp.corpus.semantic_quarantine import evaluate_semantic_quarantine
from src.live_mcp.corpus.semantic_core import resolve_semantic_gate_profile
from src.live_mcp.domain_contracts.semantic_policies import evaluate_domain_label_issue

DOMAINS_ALL = [
    "banking", "calendar", "crm", "email", "filesystem",
    "food_delivery", "issue_tracker", "payments", "shopping", "team_chat",
]

_ACTION_DATE_FIELDS = frozenset({
    "date", "due_date", "execute_date", "scheduled_date", "start_time",
})

_ANY_WEEKDAY_RE = re.compile(
    r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    re.IGNORECASE,
)

_ISO_DATE_RE = re.compile(r"\b(20\d{2}-\d{2}-\d{2})(?=T|\b)")

_LEAK_MARKERS = (
    "oracle_calls",
    "success_criteria",
    "ground_truth",
    "allowed_terminal_actions",
    "hidden_tools",
)

_QUESTION_VALUE_RE = re.compile(
    r"^(?:what|which|who|where|when|how|can you|could you|would you|"
    r"please (?:provide|tell|confirm|specify))\b|\?$",
    re.IGNORECASE,
)

_TARGET_WEEKDAY_RE = re.compile(
    r"\b(?:next|this|on|for|every|by|due)\s+"
    r"(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b"
    r"|\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)"
    r"\s*,",
    re.IGNORECASE,
)

_WEEKDAY_INDEX = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}

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

def _unresolved_failure_issue(extra: dict[str, Any]) -> str:
    """Reject a round that claims success with an unresolved failed action.

    A failed attempt may remain in an accepted replay trace. The structural
    defect is a ``final_answer``
    after the failed capability was never successfully retried on the same
    explicit mutation target: that terminal presents an unresolved action as
    completed.  ``report_error`` and ``ask_clarification`` remain valid
    graceful recovery terminals.
    """
    rounds = _as_json_list(extra.get("teacher_round_trace", []))
    for round_pos, round_trace in enumerate(rounds):
        if not isinstance(round_trace, dict):
            continue
        history = [
            event
            for event in _as_json_list(
                round_trace.get("execution_history", [])
            )
            if isinstance(event, dict)
        ]
        unresolved = unresolved_failed_tool_names(history)
        terminal = next(
            (
                str(call.get("action") or "")
                for call in reversed(
                    _as_json_list(round_trace.get("oracle_calls", []))
                )
                if isinstance(call, dict)
                and str(call.get("action") or "tool_call") != "tool_call"
            ),
            "",
        )
        if unresolved and terminal == "final_answer":
            round_idx = int(round_trace.get("round_idx", round_pos))
            return (
                f"unresolved_failed_action_final_answer:round={round_idx}:"
                f"tools={sorted(unresolved)}"
            )
    return ""

def _nested_strings(value: Any) -> list[str]:
    if isinstance(value, dict):
        return [item for child in value.values() for item in _nested_strings(child)]
    if isinstance(value, (list, tuple)) or hasattr(value, "tolist"):
        raw = value.tolist() if hasattr(value, "tolist") else value
        return [item for child in raw for item in _nested_strings(child)]
    return [str(value)] if isinstance(value, str) else []

def _action_dates(arguments: dict[str, Any]) -> list[str]:
    found: list[str] = []

    def visit(value: Any, field: str = "") -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                visit(child, str(key))
            return
        if isinstance(value, (list, tuple)) or hasattr(value, "tolist"):
            raw = value.tolist() if hasattr(value, "tolist") else value
            for child in raw:
                visit(child, field)
            return
        if field in _ACTION_DATE_FIELDS:
            found.extend(_ISO_DATE_RE.findall(str(value)))

    visit(arguments)
    return found

def _round_terminal(round_trace: dict[str, Any]) -> str:
    return next(
        (
            str(call.get("action") or "")
            for call in reversed(_as_json_list(round_trace.get("oracle_calls", [])))
            if isinstance(call, dict)
            and str(call.get("action") or "tool_call") != "tool_call"
        ),
        "",
    )

def _deterministic_label_issue(extra: dict[str, Any]) -> str:
    """Return a local, fact-provable GT label defect.

    These checks quarantine contradictions directly visible in the persisted
    query/execution trace that would make tool names or arguments incorrect GT.
    """
    domain = str(extra.get("domain") or "")
    rounds = _as_json_list(extra.get("teacher_round_trace", []))
    for round_pos, round_trace in enumerate(rounds):
        if not isinstance(round_trace, dict):
            continue
        round_idx = int(round_trace.get("round_idx", round_pos))
        query = str(
            round_trace.get("user_query")
            or round_trace.get("query")
            or ""
        )
        query_normalized = " ".join(query.lower().split())
        oracle_calls = [
            call for call in _as_json_list(round_trace.get("oracle_calls", []))
            if isinstance(call, dict)
        ]
        history = [
            event for event in _as_json_list(
                round_trace.get("execution_history", [])
            )
            if isinstance(event, dict)
        ]
        terminal = _round_terminal(round_trace)

        mentioned_weekdays = {
            (match.group(1) or match.group(2)).lower()
            for match in _TARGET_WEEKDAY_RE.finditer(query)
        }
        all_weekdays = {
            match.group(1).lower() for match in _ANY_WEEKDAY_RE.finditer(query)
        }
        if len(mentioned_weekdays) == 1 and len(all_weekdays) == 1:
            expected_weekday = _WEEKDAY_INDEX[next(iter(mentioned_weekdays))]
            for call in oracle_calls:
                if str(call.get("action") or "tool_call") != "tool_call":
                    continue
                tool_name = str(call.get("tool_name") or "")
                try:
                    mutating = is_mutating_tool(tool_name, domain)
                except ValueError:
                    continue
                if not mutating:
                    continue
                arguments = call.get("arguments") or {}
                if not isinstance(arguments, dict):
                    continue
                for iso_date in _action_dates(arguments):
                    try:
                        actual_weekday = date.fromisoformat(iso_date).weekday()
                    except ValueError:
                        continue
                    if actual_weekday != expected_weekday:
                        return (
                            "deterministic_label_quarantine:weekday_mismatch:"
                            f"round={round_idx}:tool={tool_name}:date={iso_date}"
                        )

        if terminal == "ask_clarification":
            for event in history:
                if event.get("success") is not True or event.get("state_changed") is not True:
                    continue
                tool_name = str(event.get("tool_name") or "")
                try:
                    mutating = is_mutating_tool(tool_name, domain)
                except ValueError:
                    continue
                if not mutating:
                    continue
                for value in _nested_strings(event.get("arguments") or {}):
                    normalized_value = " ".join(value.lower().split())
                    if (
                        _QUESTION_VALUE_RE.search(value.strip())
                        and normalized_value not in query_normalized
                    ):
                        return (
                            "deterministic_label_quarantine:question_placeholder_"
                            f"mutation:round={round_idx}:tool={tool_name}"
                        )

        domain_issue = evaluate_domain_label_issue(
            domain, query, terminal, history, round_idx,
        )
        if domain_issue:
            return domain_issue
    return ""

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
    recorded_schema = str(extra.get("trajectory_schema_version") or "")
    if not recorded_schema:
        return "missing_environment_metadata:trajectory_schema_version"
    if recorded_schema != TRAJECTORY_SCHEMA_VERSION:
        return f"stale_trajectory_schema:{recorded_schema}"
    try:
        from src.live_mcp.registry.environment_metadata import (
            validate_prove_corpus_evidence,
            validate_teacher_generation_evidence,
        )
        validate_prove_corpus_evidence(extra)
    except Exception as exc:
        return f"prove_corpus_evidence_invalid:{exc}"
    try:
        validate_teacher_generation_evidence(extra)
    except Exception as exc:
        if str(exc).startswith("semantic_quarantine:"):
            return str(exc)
        return f"teacher_generation_evidence_invalid:{exc}"
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

    failure_issue = _unresolved_failure_issue(extra)
    if failure_issue:
        return failure_issue

    deterministic_issue = _deterministic_label_issue(extra)
    if deterministic_issue:
        return deterministic_issue

    try:
        semantic_gate_profile = resolve_semantic_gate_profile(extra)
    except ValueError as exc:
        return f"invalid_semantic_gate_profile:{exc}"
    semantic_issue = evaluate_semantic_quarantine(extra)
    if (
        semantic_issue is not None
        and semantic_issue.hard_gate
        and semantic_gate_profile == "deterministic_v1"
    ):
        return semantic_issue.quality_issue

    # Rows generated before mutable email label templates were isolated can
    # contain state criteria for unrelated emails: add_label mutated every
    # entity sharing the same Python list.  Such criteria are impossible under
    # the corrected environment and must not become RL labels.
    if str(extra.get("domain", "")) == "email":
        label_targets = {
            str(call.get("arguments", {}).get("email_id", ""))
            for call in calls
            if call.get("action", "tool_call") == "tool_call"
            and call.get("tool_name") in {"add_label", "remove_label"}
        }
        criteria = _as_json_list(extra.get("success_criteria", []))
        criteria_label_ids = {
            str(parts[1])
            for criterion in criteria
            if isinstance(criterion, dict)
            and isinstance((parts := criterion.get("path_parts")), list)
            and len(parts) >= 3
            and parts[0] == "emails"
            and parts[2] == "labels"
        }
        # A reply/send/forward creates a new email whose complete state
        # legitimately includes a ``labels`` field even when no label tool was
        # called.  The historical aliasing defect only polluted *seeded*
        # emails, so compare non-target criteria against the deterministic
        # initial state instead of rejecting every created email label field.
        initial_email_ids: set[str] | None = None
        if extra.get("session_seed") is not None:
            from src.live_mcp.state_seeder import StateSeeder

            initial = StateSeeder().seed_state(
                "email", "merge-audit", int(extra["session_seed"]),
            )
            initial_email_ids = {
                str(email_id) for email_id in initial.get("emails", {})
            }
        unrelated = criteria_label_ids - label_targets
        if initial_email_ids is not None:
            unrelated &= initial_email_ids
        if unrelated:
            return f"unrelated_email_label_criteria:{sorted(unrelated)}"

    # The same historical aliasing defect existed in team-chat reaction
    # templates.  Its criteria are stored at channels.<id>.messages because
    # messages are a list, so compare those records with the deterministic
    # seeded state and reject changes to messages not targeted by react_message.
    if str(extra.get("domain", "")) == "team_chat":
        reaction_calls = [
            call for call in calls
            if call.get("action", "tool_call") == "tool_call"
            and call.get("tool_name") == "react_message"
        ]
        if reaction_calls:
            from src.live_mcp.state_seeder import StateSeeder

            seed = int(extra.get("session_seed", 0))
            initial = StateSeeder().seed_state("team_chat", "merge-audit", seed)
            initial_messages = {
                str(message.get("message_id")): message
                for channel in initial.get("channels", {}).values()
                for message in channel.get("messages", [])
                if isinstance(message, dict)
            }
            target_ids = {
                str(call.get("arguments", {}).get("message_id", ""))
                for call in reaction_calls
            }
            unrelated_reactions: set[str] = set()
            criteria = _as_json_list(extra.get("success_criteria", []))
            for criterion in criteria:
                if not isinstance(criterion, dict):
                    continue
                parts = criterion.get("path_parts")
                messages = criterion.get("value")
                if not (
                    isinstance(parts, list)
                    and len(parts) >= 3
                    and parts[0] == "channels"
                    and parts[2] == "messages"
                    and isinstance(messages, list)
                ):
                    continue
                for message in messages:
                    if not isinstance(message, dict):
                        continue
                    message_id = str(message.get("message_id", ""))
                    seeded = initial_messages.get(message_id)
                    if (
                        seeded is not None
                        and message_id not in target_ids
                        and message.get("reactions") != seeded.get("reactions")
                    ):
                        unrelated_reactions.add(message_id)
            if unrelated_reactions:
                return (
                    "unrelated_team_chat_reaction_criteria:"
                    f"{sorted(unrelated_reactions)}"
                )

    try:
        from src.live_mcp.artifact.reward_task import build_reward_task
        build_reward_task(extra)
    except Exception as exc:
        return f"training_contract_invalid:{exc}"

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
