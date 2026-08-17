"""Canonical PROVE artifact invariants shared by every downstream consumer."""

from __future__ import annotations

import json
from typing import Any

from src.live_mcp.dependency_trace import (
    align_sampled_chain,
    auxiliary_tool_call_indices,
    dependency_edges_from_alignment,
)


def _list(value: Any, field: str) -> list[Any]:
    if hasattr(value, "as_py"):
        value = value.as_py()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"invalid {field} JSON") from exc
    if not isinstance(value, list):
        raise RuntimeError(f"missing or invalid {field}")
    return value


def _validate_relation_edges(
    *, chain: list[str], edges: list[Any], field: str,
) -> None:
    if len(edges) != max(0, len(chain) - 1):
        raise RuntimeError(f"{field} count mismatch")
    for edge_index, (source, target) in enumerate(zip(chain, chain[1:])):
        relation_entry = edges[edge_index]
        if (
            not isinstance(relation_entry, dict)
            or relation_entry.get("source_capability") != source
            or relation_entry.get("target_capability") != target
            or relation_entry.get("relation") not in {"explicit", "implicit"}
        ):
            raise RuntimeError(
                f"invalid {field}[{edge_index}] relation metadata"
            )


def validate_dependency_artifact(extra_info: dict[str, Any]) -> None:
    """Validate the sampled/realized/auxiliary partition and reward edges."""
    oracle_calls = _list(extra_info.get("oracle_calls"), "oracle_calls")
    chain = [str(item) for item in _list(
        extra_info.get("chain_seed", []), "chain_seed",
    )]
    name_alignment = align_sampled_chain(oracle_calls, chain)
    if name_alignment is None:
        raise RuntimeError("chain_seed does not align to canonical oracle")
    scenario = str(extra_info.get("scenario_type") or "")
    success_scenario = scenario in {
        "normal_safe_success", "tool_error_recovery",
    }
    source_chain = [str(item) for item in _list(
        extra_info.get("source_chain_seed"), "source_chain_seed",
    )]
    if success_scenario and not chain:
        raise RuntimeError("successful artifact has no realized dependency chain")
    source_edges = _list(
        extra_info.get("source_chain_edges"), "source_chain_edges",
    )
    _validate_relation_edges(
        chain=source_chain, edges=source_edges, field="source_chain_edges",
    )
    raw_realized_edges = extra_info.get("realized_chain_edges")
    if raw_realized_edges is None and not chain:
        raw_realized_edges = []
    elif raw_realized_edges is None and chain == source_chain:
        # Backward-compatible read of rows written before realized relations
        # became a first-class Parquet field.  Alternative paths never use
        # this fallback because their source and realized chains differ.
        raw_realized_edges = source_edges
    realized_edges = _list(raw_realized_edges, "realized_chain_edges")
    _validate_relation_edges(
        chain=chain, edges=realized_edges, field="realized_chain_edges",
    )
    evidence = _list(
        extra_info.get("verified_dependency_evidence", []),
        "verified_dependency_evidence",
    )
    aligned = (
        align_sampled_chain(
            oracle_calls,
            chain,
            verified_dependency_evidence=evidence if len(chain) > 1 else None,
        )
        if success_scenario else name_alignment
    )
    if aligned is None:
        raise RuntimeError("chain_seed does not align to canonical oracle")
    auxiliary = auxiliary_tool_call_indices(oracle_calls, aligned)
    edges = dependency_edges_from_alignment(aligned)
    recorded_edges = _list(
        extra_info.get("dependency_edges", []), "dependency_edges",
    )
    if recorded_edges != edges:
        raise RuntimeError(
            f"dependency_edges mismatch: data={recorded_edges}, expected={edges}"
        )
    recorded_aligned = _list(
        extra_info.get("dependency_call_indices"),
        "dependency_call_indices",
    )
    recorded_auxiliary = _list(
        extra_info.get("auxiliary_call_indices"),
        "auxiliary_call_indices",
    )
    if recorded_aligned != aligned or recorded_auxiliary != auxiliary:
        raise RuntimeError("canonical dependency/auxiliary call partition mismatch")
    realized = [
        str(call.get("tool_name") or "")
        for call in oracle_calls
        if isinstance(call, dict)
        and call.get("action", "tool_call") == "tool_call"
    ]
    if _list(
        extra_info.get("realized_tool_sequence"), "realized_tool_sequence",
    ) != realized:
        raise RuntimeError("realized_tool_sequence mismatch")

    if not success_scenario:
        if chain or edges:
            raise RuntimeError(
                f"non-success scenario {scenario!r} must not carry reward dependencies"
            )
        return
    for edge_index, (source, target) in enumerate(zip(chain, chain[1:])):
        expected_indices = (aligned[edge_index], aligned[edge_index + 1])
        relation_entry = realized_edges[edge_index]
        expected_evidence_type = (
            "explicit_value_binding"
            if relation_entry["relation"] == "explicit"
            else "implicit_counterfactual_replay"
        )
        if not any(
            isinstance(item, dict)
            and str(item.get("source_capability") or "") == source
            and str(item.get("target_capability") or "") == target
            and (
                int(item.get("source_call_index", -1)),
                int(item.get("target_call_index", -1)),
            ) == expected_indices
            and str(item.get("evidence_type") or "") == expected_evidence_type
            for item in evidence
        ):
            raise RuntimeError(
                f"missing verified dependency evidence for {source}->{target}"
            )
