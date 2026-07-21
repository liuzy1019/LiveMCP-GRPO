#!/usr/bin/env python3
"""Merge Teacher-generation shards with corpus and environment quality gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from collections.abc import Hashable
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.live_mcp.observation import (
    TRAJECTORY_SCHEMA_VERSION,
    OBSERVATION_SCHEMA_VERSION,
    OBSERVATION_PROJECTION_VERSION,
    compute_server_schema_hash,
)
from src.live_mcp.tool_semantics import (
    is_mutating_tool,
    unresolved_failed_tool_names,
)

DOMAINS_ALL = [
    "banking", "calendar", "crm", "email", "filesystem",
    "food_delivery", "issue_tracker", "payments", "shopping", "team_chat",
]


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

_LEAK_MARKERS = (
    "oracle_calls",
    "success_criteria",
    "ground_truth",
    "allowed_terminal_actions",
    "hidden_tools",
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

    PROVE permits execution errors up to its replay threshold, so the failure
    itself is not a corpus defect.  The structural defect is a ``final_answer``
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


_WEEKDAY_INDEX = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}
_TARGET_WEEKDAY_RE = re.compile(
    r"\b(?:next|this|on|for|every|by|due)\s+"
    r"(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b"
    r"|\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)"
    r"\s*,",
    re.IGNORECASE,
)
_ANY_WEEKDAY_RE = re.compile(
    r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    re.IGNORECASE,
)
_ISO_DATE_RE = re.compile(r"\b(20\d{2}-\d{2}-\d{2})(?=T|\b)")
_ACTION_DATE_FIELDS = frozenset({
    "date", "due_date", "execute_date", "scheduled_date", "start_time",
})
_QUESTION_VALUE_RE = re.compile(
    r"^(?:what|which|who|where|when|how|can you|could you|would you|"
    r"please (?:provide|tell|confirm|specify))\b|\?$",
    re.IGNORECASE,
)
_FILESYSTEM_PERSISTENCE_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bremove\b.+\bfrom\s+(?:/|the file\b|this file\b)",
        r"\breplace\b.+\b(?:in|inside)\s+(?:/|it\b|them\b|the file\b)",
        r"\breverse\s+the\s+sort\s+order\s+of\b",
        r"\bchange\s+the\s+line\s+in\b",
        r"\badd\s+a\s+line\s+to\b",
    )
)
_FOOD_SIZE_WORDS = frozenset({
    "small", "medium", "large", "xl", "xxl",
})


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

    PROVE's public corpus gates remain replay, sensitive provenance, and
    Jaccard deduplication.  These narrow checks quarantine only contradictions
    that are directly visible in the persisted query/execution trace and would
    otherwise make tool names or arguments incorrect RL ground truth.
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

        if domain == "crm" and terminal in {"ask_clarification", "report_error"}:
            existing_task = re.search(r"\btask[_-][a-z0-9]+\b", query, re.IGNORECASE)
            update_intent = re.search(
                r"\b(?:update|change|set|make|mark|assign|priority|due)\b",
                query,
                re.IGNORECASE,
            )
            create_intent = re.search(
                r"\b(?:create|add|new)\b.{0,20}\btask\b",
                query,
                re.IGNORECASE,
            )
            created_task = any(
                event.get("tool_name") == "create_task"
                and event.get("success") is True
                and event.get("state_changed") is True
                for event in history
            )
            if existing_task and update_intent and not create_intent and created_task:
                return (
                    "deterministic_label_quarantine:create_as_update:"
                    f"round={round_idx}:tool=create_task"
                )

        if domain == "filesystem" and terminal == "final_answer" and any(
            pattern.search(query) for pattern in _FILESYSTEM_PERSISTENCE_PATTERNS
        ):
            successful = [event for event in history if event.get("success") is True]
            if successful:
                mutation_seen = False
                semantics_known = True
                for event in successful:
                    try:
                        mutation_seen = mutation_seen or is_mutating_tool(
                            str(event.get("tool_name") or ""), domain,
                        )
                    except ValueError:
                        semantics_known = False
                        break
                if semantics_known and not mutation_seen:
                    return (
                        "deterministic_label_quarantine:readonly_persistence:"
                        f"round={round_idx}"
                    )

        if domain == "food_delivery" and terminal == "final_answer":
            failed_orders = [
                event for event in history
                if event.get("tool_name") == "create_order"
                and event.get("success") is False
            ]
            successful_orders = [
                event for event in history
                if event.get("tool_name") == "create_order"
                and event.get("success") is True
            ]
            for failed in failed_orders:
                failed_names = {
                    " ".join(str(item.get("name") or "").lower().split())
                    for item in (failed.get("arguments") or {}).get("items", [])
                    if isinstance(item, dict)
                }
                for successful in successful_orders:
                    successful_names = {
                        " ".join(str(item.get("name") or "").lower().split())
                        for item in (successful.get("arguments") or {}).get("items", [])
                        if isinstance(item, dict)
                    }
                    for failed_name in failed_names:
                        failed_tokens = set(re.findall(r"[a-z0-9]+", failed_name))
                        explicit_sizes = failed_tokens & _FOOD_SIZE_WORDS
                        if not explicit_sizes:
                            continue
                        if all(
                            not explicit_sizes.issubset(
                                set(re.findall(r"[a-z0-9]+", successful_name))
                            )
                            for successful_name in successful_names
                        ):
                            return (
                                "deterministic_label_quarantine:food_size_"
                                f"downgrade:round={round_idx}:tool=create_order"
                            )
    return ""


@lru_cache(maxsize=1)
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


@lru_cache(maxsize=1)
def _current_schema_hashes() -> dict[str, str]:
    return {
        domain: compute_server_schema_hash(tools)
        for domain, tools in _current_tools().items()
    }


@lru_cache(maxsize=1)
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
        from src.live_mcp.environment_metadata import (
            validate_prove_corpus_evidence,
            validate_teacher_generation_evidence,
        )
        validate_prove_corpus_evidence(extra)
    except Exception as exc:
        return f"prove_corpus_evidence_invalid:{exc}"
    try:
        validate_teacher_generation_evidence(extra)
    except Exception as exc:
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
        from src.live_mcp.environment_metadata import (
            compute_initial_state_hashes,
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
                required_owners, int(extra["session_seed"]),
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
    # PROVE filters completed traces by replay error rate, provenance, and
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
        from src.reward.oval_reward_fn import _build_task_dict
        _build_task_dict(extra)
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


def _stratified_head(df: pd.DataFrame, target: int) -> pd.DataFrame:
    if target <= 0 or len(df) <= target:
        return df.reset_index(drop=True)
    buckets: dict[tuple[str, str], list[Hashable]] = {}
    for idx, row in df.iterrows():
        extra = _as_extra(row["extra_info"])
        domain = str(extra.get("domain", ""))
        scenario = str(row.get("scenario_type") or extra.get("scenario_type") or "")
        key = (domain, scenario)
        lst = buckets.get(key)
        if lst is None:
            lst = []
            buckets[key] = lst
        lst.append(idx)

    ordered_keys = sorted(buckets)
    selected: list[Hashable] = []
    while len(selected) < target and ordered_keys:
        next_keys: list[tuple[str, str]] = []
        for key in ordered_keys:
            if buckets[key] and len(selected) < target:
                selected.append(buckets[key].pop(0))
            if buckets[key]:
                next_keys.append(key)
        ordered_keys = next_keys
    return df.loc[selected].reset_index(drop=True)


def _domain_quotas(target: int, domains: list[str]) -> dict[str, int]:
    base, remainder = divmod(target, len(domains))
    return {
        domain: base + (1 if index < remainder else 0)
        for index, domain in enumerate(domains)
    }


def _capacity_weighted_domain_quotas(
    target: int,
    domains: list[str],
    capacities: dict[str, int],
    *,
    minimum_per_domain: int | None = None,
) -> dict[str, int]:
    """Apportion a split by unique feasible-chain capacity plus a floor."""
    if not domains:
        return {}
    if target < 0:
        raise ValueError("target must be non-negative")
    if minimum_per_domain is None:
        minimum_per_domain = (
            max(1, target // (2 * len(domains)))
            if target >= len(domains)
            else 0
        )
    floor = max(0, int(minimum_per_domain))
    if floor * len(domains) > target:
        raise ValueError(
            f"minimum domain coverage {floor} x {len(domains)} exceeds "
            f"split target {target}"
        )
    quotas = {domain: floor for domain in domains}
    remainder = target - floor * len(domains)
    if remainder == 0:
        return quotas

    weights = {
        domain: max(0, int(capacities.get(domain, 0)))
        for domain in domains
    }
    total_weight = sum(weights.values())
    if total_weight == 0:
        weights = {domain: 1 for domain in domains}
        total_weight = len(domains)

    raw = {
        domain: remainder * weights[domain] / total_weight
        for domain in domains
    }
    assigned = 0
    for domain in domains:
        whole = math.floor(raw[domain])
        quotas[domain] += whole
        assigned += whole
    leftover = remainder - assigned
    order = sorted(
        domains,
        key=lambda domain: (-(raw[domain] - math.floor(raw[domain])), domain),
    )
    for domain in order[:leftover]:
        quotas[domain] += 1
    return quotas


def _minimum_domain_coverage(
    target: int,
    domain_count: int,
    configured: int | None,
) -> int:
    if configured is not None:
        return max(0, int(configured))
    return max(1, target // (2 * domain_count)) if target >= domain_count else 0


def _add_weighted_with_caps(
    quotas: dict[str, int],
    maximums: dict[str, int],
    weights: dict[str, int],
    amount: int,
) -> bool:
    """Add units proportionally to frozen weights without exceeding caps."""
    added = {domain: 0 for domain in quotas}
    for _ in range(amount):
        eligible = [
            domain for domain in quotas
            if quotas[domain] < maximums.get(domain, 0)
        ]
        if not eligible:
            return False
        domain = min(
            eligible,
            key=lambda item: (
                (added[item] + 1) / max(1, int(weights.get(item, 0))),
                item,
            ),
        )
        quotas[domain] += 1
        added[domain] += 1
    return True


def _availability_constrained_split_quotas(
    count: int,
    val_count: int,
    domains: list[str],
    capacities: dict[str, int],
    available: dict[str, int],
    train_quotas: dict[str, int],
    val_quotas: dict[str, int],
    *,
    min_domain_train: int | None = None,
    min_domain_val: int | None = None,
) -> tuple[dict[str, int], dict[str, int], bool]:
    """Fit frozen-weight quotas to the eligible pool while preserving floors.

    Capacity weights remain frozen across top-up rounds, but their exact
    apportionment is not a corpus hard gate.  If total eligible supply is
    sufficient, a domain-local shortage is reassigned to domains with spare
    Jaccard-unique rows.  Separate train/validation floors remain mandatory.
    """
    if not domains:
        return train_quotas, val_quotas, False
    train_floor = _minimum_domain_coverage(
        count, len(domains), min_domain_train,
    )
    val_floor = _minimum_domain_coverage(
        val_count, len(domains), min_domain_val,
    )
    combined_floor = train_floor + val_floor
    if any(available.get(domain, 0) < combined_floor for domain in domains):
        return train_quotas, val_quotas, False
    if sum(available.get(domain, 0) for domain in domains) < count + val_count:
        return train_quotas, val_quotas, False

    desired_total = {
        domain: train_quotas[domain] + val_quotas[domain]
        for domain in domains
    }
    total_quotas = {
        domain: min(desired_total[domain], available[domain])
        for domain in domains
    }
    missing_total = count + val_count - sum(total_quotas.values())
    if missing_total and not _add_weighted_with_caps(
        total_quotas, available, capacities, missing_total,
    ):
        return train_quotas, val_quotas, False

    adjusted_val = {
        domain: min(
            max(val_floor, val_quotas[domain]),
            total_quotas[domain] - train_floor,
        )
        for domain in domains
    }
    missing_val = val_count - sum(adjusted_val.values())
    if missing_val < 0:
        return train_quotas, val_quotas, False
    val_maximums = {
        domain: total_quotas[domain] - train_floor
        for domain in domains
    }
    if missing_val and not _add_weighted_with_caps(
        adjusted_val, val_maximums, capacities, missing_val,
    ):
        return train_quotas, val_quotas, False
    adjusted_train = {
        domain: total_quotas[domain] - adjusted_val[domain]
        for domain in domains
    }
    changed = adjusted_train != train_quotas or adjusted_val != val_quotas
    return adjusted_train, adjusted_val, changed


def _dependency_chain_sequence(row: pd.Series) -> list[str]:
    """Return the live-feasible dependency seed, falling back for old rows."""
    extra = _as_extra(row["extra_info"])
    raw = extra.get("source_chain_seed", [])
    chain = _as_json_list(raw)
    sequence = [str(name) for name in chain if str(name)]
    if not sequence:
        sequence = [item.split("::", 1)[-1] for item in _row_tool_sequence(row)]
    return sequence


def _sequence_jaccard(a: list[str], b: list[str]) -> float:
    if not a or not b:
        return 0.0
    pos_a = set(enumerate(a))
    pos_b = set(enumerate(b))
    return len(pos_a & pos_b) / len(pos_a | pos_b)


def _domain_unique_chain_capacity(
    pool: pd.DataFrame,
    domains: list[str],
    threshold: float = 0.70,
) -> dict[str, int]:
    """Count position-aware Jaccard-unique feasible chains per domain."""
    capacities: dict[str, int] = {}
    for domain in domains:
        kept: list[list[str]] = []
        domain_rows = pool.loc[
            pool["extra_info"].map(
                lambda value: str(_as_extra(value).get("domain", "")) == domain
            )
        ]
        for _, row in domain_rows.iterrows():
            sequence = _dependency_chain_sequence(row)
            if not sequence:
                continue
            if any(
                _sequence_jaccard(sequence, prior) >= threshold
                for prior in kept
            ):
                continue
            kept.append(sequence)
        capacities[domain] = len(kept)
    return capacities


def _domain_deficit_report(
    pool: pd.DataFrame,
    count: int,
    val_count: int,
    domains: list[str],
    candidate_by_domain: dict[str, int] | None = None,
    train_quotas: dict[str, int] | None = None,
    val_quotas: dict[str, int] | None = None,
    unique_chain_capacity: dict[str, int] | None = None,
) -> dict[str, Any]:
    train_quotas = train_quotas or _domain_quotas(count, domains)
    val_quotas = val_quotas or _domain_quotas(val_count, domains)
    available = {
        domain: int(
            pool["extra_info"].map(
                lambda value: str(_as_extra(value).get("domain", "")) == domain
            ).sum()
        )
        for domain in domains
    }
    required = {
        domain: train_quotas[domain] + val_quotas[domain]
        for domain in domains
    }
    candidates = candidate_by_domain or dict(available)
    retention = {
        domain: (
            available[domain] / candidates[domain]
            if candidates.get(domain, 0) > 0
            else 0.0
        )
        for domain in domains
    }
    deficits = {
        domain: required[domain] - available[domain]
        for domain in domains
        if available[domain] < required[domain]
    }
    return {
        "pool_size": len(pool),
        "required_total": count + val_count,
        "available_by_domain": available,
        "candidate_by_domain": candidates,
        "jaccard_retention_by_domain": retention,
        "unique_chain_capacity_by_domain": unique_chain_capacity or {},
        "train_quota_by_domain": train_quotas,
        "val_quota_by_domain": val_quotas,
        "required_by_domain": required,
        "deficits": deficits,
        "suggested_topup_by_domain": {
            domain: _suggest_topup_count(
                missing=missing,
                available=available[domain],
                candidates=candidates.get(domain, 0),
            )
            for domain, missing in deficits.items()
        },
    }


def _suggest_topup_count(
    *,
    missing: int,
    available: int,
    candidates: int,
) -> int:
    """Estimate candidate top-up from the observed global retention.

    The previous fixed ``missing + 50%`` estimate assumes at least 2/3 of new
    rows survive global Jaccard. Real domains can retain much less, causing
    repeated undersized rounds even though every generated shard is valid. This
    estimate changes only how many candidates are requested; all candidates
    still pass the unchanged replay/provenance/Jaccard gates.
    """
    missing = max(0, int(missing))
    if missing == 0:
        return 0
    observed = available / candidates if candidates > 0 else 0.0
    # Avoid division by zero for a domain with no survivor, while keeping the
    # estimate bounded.  The 20% margin absorbs ordinary sampling variance.
    effective = min(1.0, max(0.10, observed))
    base = math.ceil(missing / effective)
    margin = max(2, math.ceil(base * 0.20))
    return base + margin


def _write_deficit_report(path: Path | None, report: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _frozen_allocation_capacity(
    path: Path | None,
    domains: list[str],
) -> dict[str, int] | None:
    """Reuse the first merge's capacity weights across top-up rounds."""
    if path is None or not path.exists():
        return None
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return None
    raw = report.get("allocation_capacity_by_domain")
    if not isinstance(raw, dict) or set(raw) != set(domains):
        return None
    try:
        return {domain: max(0, int(raw[domain])) for domain in domains}
    except (TypeError, ValueError):
        return None


def _initial_query_key(row: pd.Series) -> str:
    """Return the normalized policy-visible first user query for split isolation."""
    extra = _as_extra(row["extra_info"])
    return " ".join(str(extra.get("user_query", "")).lower().split())


def _isolate_initial_queries(
    train: pd.DataFrame,
    val: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """Keep identical initial policy inputs wholly within one split.

    This is an evaluation isolation contract, not a corpus filter: rows are
    swapped between equally sized train/validation partitions and none are
    removed.  Moving the overlapping validation group into train is preferred;
    an equally sized set of complete train-side query groups moves to val.
    """
    train = train.reset_index(drop=True)
    val = val.reset_index(drop=True)
    while True:
        train_keys = train.apply(_initial_query_key, axis=1)
        val_keys = val.apply(_initial_query_key, axis=1)
        overlap = (set(train_keys) & set(val_keys)) - {""}
        if not overlap:
            return train, val

        key = sorted(overlap)[0]
        val_overlap_idx = list(val.index[val_keys == key])
        val_key_set = set(val_keys)
        domains = sorted({
            str(_as_extra(value).get("domain", "")) or "__all__"
            for value in pd.concat([train["extra_info"], val["extra_info"]])
        })

        def domain_counts(frame: pd.DataFrame, indices: list[Hashable]) -> tuple[int, ...]:
            counts = {domain: 0 for domain in domains}
            for idx in indices:
                domain = str(
                    _as_extra(frame.loc[idx, "extra_info"]).get("domain", "")
                ) or "__all__"
                counts[domain] += 1
            return tuple(counts[domain] for domain in domains)

        target_counts = domain_counts(val, val_overlap_idx)
        candidate_groups: list[tuple[str, list[Hashable], tuple[int, ...]]] = []
        for candidate_key in sorted(set(train_keys) - val_key_set - {""}):
            indices = list(train.index[train_keys == candidate_key])
            candidate_groups.append(
                (candidate_key, indices, domain_counts(train, indices))
            )

        # Multi-dimensional subset-sum over complete query groups.  Matching
        # the val-overlap domain vector keeps every per-domain quota unchanged
        # while moving the duplicate query wholly into train.
        zero = tuple(0 for _ in domains)
        choices: dict[tuple[int, ...], list[Hashable]] = {zero: []}
        for _, indices, group_counts in candidate_groups:
            for current, selected in list(choices.items())[::-1]:
                new_counts = tuple(
                    current[i] + group_counts[i] for i in range(len(domains))
                )
                if any(
                    new_counts[i] > target_counts[i]
                    for i in range(len(domains))
                ):
                    continue
                choices.setdefault(new_counts, selected + indices)
        train_to_val_idx = choices.get(target_counts)
        if train_to_val_idx is None:
            return None

        incoming_train = val.loc[val_overlap_idx].copy()
        incoming_val = train.loc[train_to_val_idx].copy()
        train = pd.concat(
            [train.drop(index=train_to_val_idx), incoming_train],
            ignore_index=True,
        )
        val = pd.concat(
            [val.drop(index=val_overlap_idx), incoming_val],
            ignore_index=True,
        )


def _balanced_domain_split(
    pool: pd.DataFrame,
    count: int,
    val_count: int,
    domains: list[str],
    train_quotas: dict[str, int] | None = None,
    val_quotas: dict[str, int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    train_quotas = train_quotas or _domain_quotas(count, domains)
    val_quotas = val_quotas or _domain_quotas(val_count, domains)
    train_parts: list[pd.DataFrame] = []
    val_parts: list[pd.DataFrame] = []
    for domain in domains:
        domain_rows = pool.loc[
            pool["extra_info"].map(
                lambda value: str(_as_extra(value).get("domain", "")) == domain
            )
        ].reset_index(drop=True)
        needed = train_quotas[domain] + val_quotas[domain]
        if len(domain_rows) < needed:
            print(
                f"  FATAL: domain {domain} has {len(domain_rows)} global unique "
                f"candidates, need {needed} "
                f"(train={train_quotas[domain]}, val={val_quotas[domain]})"
            )
            return None
        selected = _stratified_head(domain_rows, needed)
        domain_train = selected.iloc[:train_quotas[domain]].reset_index(drop=True)
        domain_val = selected.iloc[train_quotas[domain]:needed].reset_index(drop=True)
        train_parts.append(domain_train)
        val_parts.append(domain_val)
    train_df = pd.concat(train_parts, ignore_index=True) if train_parts else pool.iloc[:0].copy()
    val_df = pd.concat(val_parts, ignore_index=True) if val_parts else pool.iloc[:0].copy()
    isolated = _isolate_initial_queries(train_df, val_df)
    if isolated is None:
        print(
            "  FATAL: cannot isolate identical initial queries globally "
            "without changing per-domain quotas"
        )
        return None
    return isolated


def merge_split(
    tmpdir: Path,
    pattern: str,
    outpath: Path,
    target: int,
    *,
    write_output: bool = True,
) -> tuple[bool, pd.DataFrame]:
    dfs = [pd.read_parquet(path) for path in sorted(tmpdir.glob(pattern))]
    if not dfs:
        print(f"WARNING: no {pattern} data")
        return False, pd.DataFrame()

    merged = pd.concat(dfs, ignore_index=True)
    if len(merged) == 0 or len(merged.columns) == 0:
        print(f"  {outpath}: 0 rows (empty parquet files)")
        return True, pd.DataFrame()
    merged["_quality_issue"] = merged.apply(_quality_issue, axis=1)
    bad_mask = merged["_quality_issue"].astype(bool)
    dropped_quality = int(bad_mask.sum())
    if dropped_quality:
        quality_counts = merged.loc[bad_mask, "_quality_issue"].value_counts().to_dict()
        print(f"  quality: dropped {dropped_quality} rows: {quality_counts}")
        if dropped_quality == len(merged) and len(quality_counts) == 1:
            only_issue = str(next(iter(quality_counts)))
            if only_issue.startswith("environment_metadata_invalid:"):
                raise FatalShardIntegrityError(pattern, len(merged), only_issue)
    merged = merged.loc[~bad_mask].drop(columns=["_quality_issue"]).reset_index(drop=True)

    before_dedup = len(merged)
    seen: set[str] = set()
    keep_rows: list[dict[str, Any]] = []
    for _, row in merged.iterrows():
        fingerprint = _row_fingerprint(row)
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        keep_rows.append(cast(dict[str, Any], row.to_dict()))
    merged = pd.DataFrame(keep_rows)
    dropped_dedup = before_dedup - len(merged)
    if dropped_dedup:
        print(f"  dedup: dropped {dropped_dedup} cross-shard duplicates, {len(merged)} remaining")

    if target > 0 and len(merged) > target:
        merged = _stratified_head(merged, target)
    if target > 0 and len(merged) < target:
        print(f"  FATAL: {outpath} has {len(merged)} rows, below target {target}")
        return False, merged
    if write_output:
        outpath.parent.mkdir(parents=True, exist_ok=True)
        merged.to_parquet(outpath, index=False)
        print(f"  {outpath}: {len(merged)} rows (target={target})")
    return True, merged


def _dedup_task_ids(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    seen: set[str] = set()
    keep: list[Hashable] = []
    removed = 0
    for idx, row in df.iterrows():
        task_id = str(_as_extra(row["extra_info"]).get("task_id", ""))
        if task_id and task_id in seen:
            removed += 1
            continue
        if task_id:
            seen.add(task_id)
        keep.append(idx)
    return df.loc[keep].reset_index(drop=True), removed


def _load_quarantined_task_ids(path: Path | None) -> set[str]:
    if path is None:
        return set()
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_samples: Any
    if isinstance(payload, list):
        raw_samples = payload
    elif isinstance(payload, dict):
        raw_samples = payload.get("samples", [])
    else:
        raise ValueError(f"invalid quarantine manifest root: {type(payload).__name__}")
    if not isinstance(raw_samples, list):
        raise ValueError("quarantine manifest samples must be a list")
    task_ids: set[str] = set()
    for index, sample in enumerate(raw_samples):
        if isinstance(sample, str):
            task_id = sample.strip()
        elif isinstance(sample, dict):
            task_id = str(sample.get("task_id") or "").strip()
        else:
            raise ValueError(
                f"quarantine manifest samples[{index}] must be string or object"
            )
        if not task_id:
            raise ValueError(f"quarantine manifest samples[{index}] has empty task_id")
        if task_id in task_ids:
            raise ValueError(f"duplicate quarantine task_id: {task_id}")
        task_ids.add(task_id)
    return task_ids


def _drop_quarantined_tasks(
    df: pd.DataFrame,
    task_ids: set[str],
) -> tuple[pd.DataFrame, int]:
    if not task_ids or df.empty:
        return df.reset_index(drop=True), 0
    quarantined = df["extra_info"].map(
        lambda value: str(_as_extra(value).get("task_id") or "") in task_ids
    )
    return df.loc[~quarantined].reset_index(drop=True), int(quarantined.sum())


def _row_tool_sequence(row: pd.Series) -> list[str]:
    extra = _as_extra(row["extra_info"])
    primary_domain = str(extra.get("domain") or "")
    raw_sequence = extra.get("teacher_trace_tool_sequence", [])
    if isinstance(raw_sequence, str):
        try:
            raw_sequence = json.loads(raw_sequence)
        except (json.JSONDecodeError, TypeError):
            raw_sequence = []
    elif hasattr(raw_sequence, "tolist"):
        raw_sequence = raw_sequence.tolist()
    if isinstance(raw_sequence, (list, tuple)) and raw_sequence:
        return [f"{primary_domain}::{name}" for name in raw_sequence]
    return [
        f"{primary_domain}::{call.get('tool_name', '')}"
        for call in _oracle_calls(extra)
        if call.get("action", "tool_call") == "tool_call"
    ]


def _row_jaccard(a: pd.Series, b: pd.Series) -> float:
    seq_a, seq_b = _row_tool_sequence(a), _row_tool_sequence(b)
    if not seq_a and not seq_b:
        return 0.0
    if not seq_a or not seq_b:
        return 0.0
    pos_a = set(enumerate(seq_a))
    pos_b = set(enumerate(seq_b))
    return len(pos_a & pos_b) / len(pos_a | pos_b)


def _dedup_jaccard(df: pd.DataFrame, threshold: float = 0.70) -> tuple[pd.DataFrame, int]:
    kept: list[Hashable] = []
    removed = 0
    for idx, row in df.iterrows():
        if any(_row_jaccard(row, df.loc[prior]) >= threshold for prior in kept):
            removed += 1
            continue
        kept.append(idx)
    return df.loc[kept].reset_index(drop=True), removed


def merge_shards(
    tmpdir: Path,
    output_dir: Path,
    count: int,
    val_count: int,
    domains: list[str] | None = None,
    deficits_output: Path | None = None,
    min_domain_train: int | None = None,
    min_domain_val: int | None = None,
    quarantine_task_ids: set[str] | None = None,
) -> int:
    """Globally deduplicate candidates before final train/val truncation."""
    train_path = output_dir / "train.parquet"
    val_path = output_dir / "val.parquet"
    try:
        ok_train, train_candidates = merge_split(
            tmpdir, "shard_*_train.parquet", train_path, 0, write_output=False,
        )
        ok_val, val_candidates = merge_split(
            tmpdir, "shard_*_val.parquet", val_path, 0, write_output=False,
        )
    except FatalShardIntegrityError as exc:
        report = {
            "pool_size": 0,
            "required_total": count + val_count,
            "available_by_domain": {},
            "required_by_domain": {},
            "deficits": {},
            "suggested_topup_by_domain": {},
            "fatal_integrity_errors": {
                exc.issue: exc.row_count,
            },
        }
        _write_deficit_report(deficits_output, report)
        print(f"  FATAL: {exc}")
        return 2
    if not ok_train or (val_count > 0 and not ok_val):
        return 1

    pool = pd.concat([train_candidates, val_candidates], ignore_index=True)
    pool, quarantine_removed = _drop_quarantined_tasks(
        pool, quarantine_task_ids or set(),
    )
    pool, tid_removed = _dedup_task_ids(pool)
    if pool.empty:
        if domains:
            deficit_report = _domain_deficit_report(
                pd.DataFrame({"extra_info": []}),
                count,
                val_count,
                domains,
                candidate_by_domain={domain: 0 for domain in domains},
            )
        else:
            deficit_report = {
                "pool_size": 0,
                "required_total": count + val_count,
                "available_by_domain": {},
                "required_by_domain": {},
                "deficits": (
                    {"__all__": count + val_count}
                    if count + val_count > 0 else {}
                ),
            }
        deficit_report.update({
            "quarantine_removed": quarantine_removed,
            "task_id_removed": tid_removed,
            "exact_removed": 0,
            "jaccard_removed": 0,
        })
        _write_deficit_report(deficits_output, deficit_report)
        print(f"  FATAL: global unique candidates=0, need {count + val_count}")
        return 1
    before_exact = len(pool)
    pool["_semantic_fp"] = pool.apply(_row_fingerprint, axis=1)
    pool = pool.drop_duplicates(subset=["_semantic_fp"], keep="first").drop(
        columns=["_semantic_fp"],
    ).reset_index(drop=True)
    exact_removed = before_exact - len(pool)
    candidate_by_domain = (
        {
            domain: int(
                pool["extra_info"].map(
                    lambda value: str(_as_extra(value).get("domain", "")) == domain
                ).sum()
            )
            for domain in domains
        }
        if domains else {}
    )
    observed_chain_capacity = {}
    allocation_capacity = {}
    if domains:
        observed_chain_capacity = _domain_unique_chain_capacity(pool, domains)
        allocation_capacity = _frozen_allocation_capacity(
            deficits_output, domains,
        ) or observed_chain_capacity
    pool, jaccard_removed = _dedup_jaccard(pool, threshold=0.70)

    if domains:
        train_quotas = _capacity_weighted_domain_quotas(
            count, domains, allocation_capacity,
            minimum_per_domain=min_domain_train,
        )
        val_quotas = _capacity_weighted_domain_quotas(
            val_count, domains, allocation_capacity,
            minimum_per_domain=min_domain_val,
        )
        frozen_train_quotas = dict(train_quotas)
        frozen_val_quotas = dict(val_quotas)
        available_by_domain = {
            domain: int(
                pool["extra_info"].map(
                    lambda value: str(_as_extra(value).get("domain", "")) == domain
                ).sum()
            )
            for domain in domains
        }
        train_quotas, val_quotas, quota_rebalanced = (
            _availability_constrained_split_quotas(
                count,
                val_count,
                domains,
                allocation_capacity,
                available_by_domain,
                train_quotas,
                val_quotas,
                min_domain_train=min_domain_train,
                min_domain_val=min_domain_val,
            )
        )
        deficit_report = _domain_deficit_report(
            pool, count, val_count, domains,
            candidate_by_domain=candidate_by_domain,
            train_quotas=train_quotas,
            val_quotas=val_quotas,
            unique_chain_capacity=observed_chain_capacity,
        )
        deficit_report["allocation_capacity_by_domain"] = allocation_capacity
        deficit_report["frozen_train_quota_by_domain"] = frozen_train_quotas
        deficit_report["frozen_val_quota_by_domain"] = frozen_val_quotas
        deficit_report["quota_rebalanced"] = quota_rebalanced
    else:
        required = count + val_count
        deficit_report = {
            "pool_size": len(pool),
            "required_total": required,
            "available_by_domain": {},
            "required_by_domain": {},
            "deficits": {"__all__": required - len(pool)} if len(pool) < required else {},
        }
    deficit_report.update({
        "quarantine_removed": quarantine_removed,
        "task_id_removed": tid_removed,
        "exact_removed": exact_removed,
        "jaccard_removed": jaccard_removed,
    })
    _write_deficit_report(deficits_output, deficit_report)

    required = count + val_count
    if len(pool) < required:
        print(
            f"  FATAL: global unique candidates={len(pool)}, need {required} "
            f"(quarantine_removed={quarantine_removed}, "
            f"task_id_removed={tid_removed}, exact_removed={exact_removed}, "
            f"jaccard_removed={jaccard_removed})"
        )
        return 1

    if domains:
        split = _balanced_domain_split(
            pool, count, val_count, domains,
            train_quotas=train_quotas, val_quotas=val_quotas,
        )
        if split is None:
            return 1
        train_df, val_df = split
    else:
        selected = _stratified_head(pool, required)
        train_df = selected.iloc[:count].reset_index(drop=True)
        val_df = selected.iloc[count:count + val_count].reset_index(drop=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)
    print(f"  {train_path}: {len(train_df)} rows (target={count})")
    print(f"  {val_path}: {len(val_df)} rows (target={val_count})")
    print(
        f"  merge ok: {len(train_df)} train + {len(val_df)} val, "
        f"quarantine_removed={quarantine_removed}, "
        f"task_id_removed={tid_removed}, exact_removed={exact_removed}, "
        f"jaccard_removed={jaccard_removed}"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tmpdir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--val-count", type=int, required=True)
    parser.add_argument("--domain", default="all")
    parser.add_argument("--deficits-output")
    parser.add_argument("--min-domain-train", type=int)
    parser.add_argument("--min-domain-val", type=int)
    parser.add_argument("--quarantine-task-ids", type=Path)
    args = parser.parse_args()

    domains = (
        list(DOMAINS_ALL)
        if args.domain == "all"
        else [item.strip() for item in args.domain.split(",") if item.strip()]
    )
    unknown = sorted(set(domains) - set(DOMAINS_ALL))
    if not domains or unknown:
        parser.error(f"invalid --domain value: {args.domain!r}; unknown={unknown}")
    if len(domains) != len(set(domains)):
        parser.error(f"duplicate --domain entries are not allowed: {domains}")
    return merge_shards(
        Path(args.tmpdir), Path(args.output_dir), args.count, args.val_count,
        domains=domains,
        deficits_output=Path(args.deficits_output) if args.deficits_output else None,
        min_domain_train=args.min_domain_train,
        min_domain_val=args.min_domain_val,
        quarantine_task_ids=_load_quarantined_task_ids(args.quarantine_task_ids),
    )


if __name__ == "__main__":
    raise SystemExit(main())
