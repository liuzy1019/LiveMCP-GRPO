"""Shared dependency-chain alignment for generation and corpus serialization.

The sampled dependency chain, the realized oracle workflow, and auxiliary
discovery/detail calls are different objects.  This module keeps their index
semantics identical on both sides of the Parquet boundary.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from typing import Any

from src.live_mcp.types import ToolCall


def _call_action(call: Any) -> str:
    if isinstance(call, dict):
        return str(call.get("action", "tool_call"))
    return str(getattr(call, "action", "tool_call"))


def _call_tool_name(call: Any) -> str:
    if isinstance(call, dict):
        return str(call.get("tool_name") or "")
    return str(getattr(call, "tool_name", "") or "")


def align_sampled_chain(
    oracle_calls: list[Any],
    sampled_chain: list[str] | tuple[str, ...],
    *,
    verified_dependency_evidence: Any | None = None,
    prefer_latest: bool = False,
) -> list[int] | None:
    """Align every sampled chain step to an ordered oracle tool-call index.

    Extra oracle calls are allowed between dependency steps.  ``None`` means
    the sampled chain was not fully realized; an empty sampled chain aligns to
    an empty index list.
    """
    chain = [str(name) for name in sampled_chain if str(name)]
    if not chain:
        return []

    if verified_dependency_evidence is not None and len(chain) > 1:
        evidence = verified_dependency_evidence
        if hasattr(evidence, "as_py"):
            evidence = evidence.as_py()
        if hasattr(evidence, "tolist"):
            evidence = evidence.tolist()
        if isinstance(evidence, str):
            try:
                evidence = json.loads(evidence)
            except json.JSONDecodeError:
                return None
        if not isinstance(evidence, list):
            return None

        candidates: list[list[tuple[int, int]]] = []
        for source_name, target_name in zip(chain, chain[1:]):
            edge_candidates: list[tuple[int, int]] = []
            for item in evidence:
                if not isinstance(item, dict):
                    continue
                if (
                    str(item.get("source_capability") or "") != source_name
                    or str(item.get("target_capability") or "") != target_name
                ):
                    continue
                try:
                    source_index = int(item.get("source_call_index", -1))
                    target_index = int(item.get("target_call_index", -1))
                except (TypeError, ValueError):
                    continue
                if not (0 <= source_index < target_index < len(oracle_calls)):
                    continue
                if (
                    _call_action(oracle_calls[source_index]) != "tool_call"
                    or _call_action(oracle_calls[target_index]) != "tool_call"
                    or _call_tool_name(oracle_calls[source_index]) != source_name
                    or _call_tool_name(oracle_calls[target_index]) != target_name
                ):
                    continue
                edge_candidates.append((source_index, target_index))
            if not edge_candidates:
                return None
            candidates.append(edge_candidates)

        # One call must represent each intermediate chain node.  Splicing two
        # same-name recovery calls would incorrectly turn auxiliary work into
        # a reward dependency.
        paths = [[source, target] for source, target in candidates[0]]
        for edge_candidates in candidates[1:]:
            paths = [
                path + [target]
                for path in paths
                for source, target in edge_candidates
                if path[-1] == source
            ]
            if not paths:
                return None
        return min(paths)

    if prefer_latest:
        aligned_reversed: list[int] = []
        cursor = len(oracle_calls)
        for expected_name in reversed(chain):
            match = next(
                (
                    index
                    for index in range(cursor - 1, -1, -1)
                    if _call_action(oracle_calls[index]) == "tool_call"
                    and _call_tool_name(oracle_calls[index]) == expected_name
                ),
                None,
            )
            if match is None:
                return None
            aligned_reversed.append(match)
            cursor = match
        return list(reversed(aligned_reversed))

    aligned: list[int] = []
    cursor = -1
    for expected_name in chain:
        match = next(
            (
                index
                for index in range(cursor + 1, len(oracle_calls))
                if _call_action(oracle_calls[index]) == "tool_call"
                and _call_tool_name(oracle_calls[index]) == expected_name
            ),
            None,
        )
        if match is None:
            return None
        aligned.append(match)
        cursor = match
    return aligned


def dependency_edges_from_alignment(aligned_indices: list[int]) -> list[list[int]]:
    """Return adjacent dependency edges for a fully aligned sampled chain."""
    return [
        [aligned_indices[index], aligned_indices[index + 1]]
        for index in range(len(aligned_indices) - 1)
    ]


def auxiliary_tool_call_indices(
    oracle_calls: list[Any],
    aligned_indices: list[int],
) -> list[int]:
    """Return tool-call indices not used by the sampled dependency chain."""
    dependency_indices = set(aligned_indices)
    return [
        index
        for index, call in enumerate(oracle_calls)
        if _call_action(call) == "tool_call" and index not in dependency_indices
    ]


def unauthorized_mutating_tool_names(
    oracle_calls: list[Any],
    authorized_capabilities: Iterable[str],
    *,
    is_mutating: Callable[[str], bool],
) -> list[str]:
    """Return state-changing calls absent from immutable query provenance."""
    authorized = {str(name) for name in authorized_capabilities if str(name)}
    return sorted({
        name
        for call in oracle_calls
        if _call_action(call) == "tool_call"
        and (name := _call_tool_name(call))
        and is_mutating(name)
        and name not in authorized
    })


def select_realized_dependency_chain(
    oracle_calls: list[Any],
    graph: dict[str, dict[str, list[str]]],
    *,
    max_length: int = 5,
) -> tuple[list[str], list[int]]:
    """Select the longest graph-valid path actually executed by Teacher.

    ``source_chain_seed`` is query provenance, not an execution template.  A
    successful alternative route receives reward dependency edges only when
    its own ordered calls form a path in the audited dependency graph.
    """
    tool_indices = [
        index
        for index, call in enumerate(oracle_calls)
        if _call_action(call) == "tool_call" and _call_tool_name(call)
    ]
    paths: list[list[int]] = []

    def extend(path: list[int], start: int) -> None:
        if len(path) >= 2:
            paths.append(list(path))
        if len(path) >= max_length:
            return
        for position in range(start, len(tool_indices)):
            call_index = tool_indices[position]
            if path:
                source_name = _call_tool_name(oracle_calls[path[-1]])
                target_name = _call_tool_name(oracle_calls[call_index])
                relations = [
                    relation
                    for relation in ("explicit", "implicit")
                    if target_name in graph.get(source_name, {}).get(
                        relation, []
                    )
                ]
                if len(relations) != 1:
                    continue
            extend(path + [call_index], position + 1)

    extend([], 0)
    if not paths:
        return [], []
    selected = min(
        paths,
        key=lambda path: (
            -len(path),
            tuple(path),
            tuple(_call_tool_name(oracle_calls[index]) for index in path),
        ),
    )
    return [
        _call_tool_name(oracle_calls[index]) for index in selected
    ], selected


def verify_implicit_edges_counterfactually(
    *,
    manager: Any,
    executor: Any,
    server_name: str,
    seed: int,
    oracle_calls: list[Any],
    sampled_chain: list[str],
    explicitly_verified_edges: set[tuple[str, str]],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Prove implicit edges by removing one source from a fresh replay.

    For each edge without an explicit observation-to-argument binding, replay
    the already-required dependency prefix.  The exact target call must fail
    without the source and succeed after the source changes state.
    """
    aligned = align_sampled_chain(
        oracle_calls, sampled_chain, prefer_latest=True,
    )
    if aligned is None:
        return [], ["sampled chain is not aligned for counterfactual replay"]
    evidence: list[dict[str, Any]] = []
    issues: list[str] = []

    def _arguments(call: Any) -> dict[str, Any]:
        if isinstance(call, dict):
            return dict(call.get("arguments") or {})
        return dict(getattr(call, "arguments", {}) or {})

    def _execute(session_id: str, call: Any, suffix: str) -> Any:
        return executor.execute(
            session_id,
            ToolCall(
                _call_tool_name(call),
                _arguments(call),
                call_id=f"dependency_cf_{suffix}",
            ),
            domain=server_name,
        )

    for edge_index, (source_name, target_name) in enumerate(
        zip(sampled_chain, sampled_chain[1:])
    ):
        if (source_name, target_name) in explicitly_verified_edges:
            continue
        source_index = aligned[edge_index]
        target_index = aligned[edge_index + 1]
        source_call = oracle_calls[source_index]
        target_call = oracle_calls[target_index]
        # Hold every other successful action before the target fixed.  Using
        # only sampled-chain nodes silently removes auxiliary setup calls
        # (for example mkdir between cp and mv), so the present replay can
        # fail for a reason unrelated to the edge under test.
        prior_tool_calls = [
            (index, call)
            for index, call in enumerate(oracle_calls[:target_index])
            if _call_action(call) == "tool_call"
        ]

        absent = manager.create_session(seed=seed, server_names=[server_name])
        try:
            prefix_ok = all(
                _execute(
                    absent.session_id, call, f"absent_prefix_{call_index}",
                ).success
                for call_index, call in prior_tool_calls
                if call_index != source_index
            )
            absent_result = (
                _execute(absent.session_id, target_call, "absent_target")
                if prefix_ok else None
            )
        finally:
            manager.close_session(absent.session_id)
        if not prefix_ok:
            issues.append(
                f"{source_name}->{target_name}: counterfactual prefix failed"
            )
            continue
        if absent_result is not None and absent_result.success:
            issues.append(
                f"{source_name}->{target_name}: target succeeds without source"
            )
            continue

        present = manager.create_session(seed=seed, server_names=[server_name])
        try:
            present_prefix_ok = True
            source_result = None
            for call_index, call in prior_tool_calls:
                result = _execute(
                    present.session_id, call, f"present_prefix_{call_index}",
                )
                if call_index == source_index:
                    source_result = result
                if not result.success:
                    present_prefix_ok = False
                    break
            target_result = (
                _execute(present.session_id, target_call, "present_target")
                if present_prefix_ok
                and source_result is not None
                and source_result.success
                else None
            )
        finally:
            manager.close_session(present.session_id)
        if (
            not present_prefix_ok
            or source_result is None
            or not source_result.success
            or not bool(source_result.state_changed)
            or target_result is None
            or not target_result.success
        ):
            issues.append(
                f"{source_name}->{target_name}: source does not establish target state"
            )
            continue
        evidence.append({
            "source_capability": source_name,
            "target_capability": target_name,
            "evidence_type": "implicit_counterfactual_replay",
            "source_call_index": aligned[edge_index],
            "target_call_index": aligned[edge_index + 1],
            "target_without_source_error_type": str(
                getattr(absent_result, "error_type", "") or ""
            ),
            "source_state_changed": True,
            "target_with_source_succeeded": True,
        })
    return evidence, issues
