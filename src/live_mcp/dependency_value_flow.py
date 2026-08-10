"""Dependency value-flow contracts and realized-edge evidence."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from decimal import Decimal, InvalidOperation
from typing import Any

from src.live_mcp.contracts.factory import build_contract_registry
from src.live_mcp.contracts.value_flow import novel_output_fields
from src.live_mcp.domain_contracts.dependency import (
    _DEPENDENCY_TOOL_OUTPUT_FIELDS,
)
from src.live_mcp.domain_contracts.value_bindings import OUTPUT_ARGUMENT_ALIASES
from src.live_mcp.domain_contracts.entities import _CREATED_ENTITY_BY_TOOL
from src.live_mcp.domain_contracts.states import DOMAIN_STATE_FACTS
from src.live_mcp.registry.tool_semantics import resolve_tool_execution_semantics
from src.live_mcp.task_spec import DifficultyVector
from src.live_mcp.types import OracleCall

def _dependency_value_key(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, default=str,
        separators=(",", ":"),
    )


def _aggregate_probe_results_by_tool(
    probe_results: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Union usable values from repeated readonly probes of the same tool.

    Some samplers call a detail tool once per discovered entity. Keeping only
    the last call silently drops identities observed by earlier calls and makes
    chain feasibility depend on probe order.
    """
    aggregated: dict[str, dict[str, Any]] = {}
    value_maps: dict[str, dict[str, dict[str, Any]]] = {}
    for item in probe_results:
        if not isinstance(item, dict):
            continue
        tool_name = str(item.get("tool") or "").lower()
        if not tool_name:
            continue
        entry = aggregated.setdefault(tool_name, {
            "tool": tool_name,
            "success": False,
            "state_changed": False,
            "output_field_counts": {},
            "output_field_values": {},
        })
        entry["state_changed"] = bool(
            entry["state_changed"] or item.get("state_changed")
        )
        usable = bool(item.get("success")) and not bool(
            item.get("state_changed")
        )
        if not usable:
            continue
        entry["success"] = True
        tool_values = value_maps.setdefault(tool_name, {})
        for field_name, values in (
            item.get("output_field_values") or {}
        ).items():
            field_values = tool_values.setdefault(str(field_name), {})
            for value in values or []:
                field_values.setdefault(_dependency_value_key(value), value)
    for tool_name, field_maps in value_maps.items():
        entry = aggregated[tool_name]
        rendered = {
            field_name: list(values.values())
            for field_name, values in field_maps.items()
        }
        entry["output_field_values"] = rendered
        entry["output_field_counts"] = {
            field_name: len(values)
            for field_name, values in rendered.items()
        }
    return aggregated


def _field_values(value: Any, field_name: str) -> list[Any]:
    """Collect exact values stored under one field in a nested observation."""
    found: list[Any] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key) == field_name:
                found.append(item)
            found.extend(_field_values(item, field_name))
    elif isinstance(value, list):
        for item in value:
            found.extend(_field_values(item, field_name))
    return found


def _value_is_explicit_in_query(value: Any, user_query: str) -> bool:
    """Detect exact scalar values even when numbers use human formatting."""
    normalized_query = user_query.casefold()
    if isinstance(value, bool):
        return str(value).casefold() in normalized_query
    if isinstance(value, (int, float, Decimal)):
        try:
            expected = Decimal(str(value))
        except InvalidOperation:
            return False
        for match in re.finditer(
            r"(?<![\w.])[-+]?(?:\d[\d,]*(?:\.\d+)?|\.\d+)(?![\w.])",
            user_query,
        ):
            try:
                observed = Decimal(match.group(0).replace(",", ""))
            except InvalidOperation:
                continue
            if observed == expected:
                return True
        return False
    if isinstance(value, str):
        rendered = value.strip().casefold()
        return bool(rendered and rendered in normalized_query)
    rendered = str(value).strip().casefold()
    return bool(rendered and rendered in normalized_query)


def _required_arguments_by_tool(
    tool_schemas: list[dict[str, Any]],
) -> dict[str, set[str]]:
    required_by_tool: dict[str, set[str]] = {}
    for schema in tool_schemas:
        name = str(schema.get("name") or "")
        input_schema = schema.get("input_schema") or schema.get("inputSchema") or {}
        required_by_tool[name] = {
            str(item) for item in (input_schema.get("required") or [])
        }
    return required_by_tool


def _novel_dependency_output_fields(
    server_name: str,
    source_name: str,
    tool_schemas_by_name: dict[str, dict[str, Any]],
) -> set[str]:
    """Return source fields that are not merely echoed required inputs."""
    if server_name in DOMAIN_STATE_FACTS:
        source_schema = tool_schemas_by_name.get(source_name)
        if source_schema is None:
            return set()
        registry = build_contract_registry({server_name: [source_schema]})
        return set(novel_output_fields(registry.get(server_name, source_name)))
    known_fields = set(
        _DEPENDENCY_TOOL_OUTPUT_FIELDS.get(server_name, {}).get(source_name, ())
    )
    source_schema = tool_schemas_by_name.get(source_name, {})
    source_input = (
        source_schema.get("input_schema")
        or source_schema.get("inputSchema")
        or {}
    )
    source_required = {
        str(field) for field in (source_input.get("required") or [])
    }
    # A returned scalar can be an echo of a required source argument even when
    # the handler exposes the singular/plural alias used by another tool.  For
    # example compare_products(product_ids=[...]) returns product.product_id;
    # those IDs were supplied by the caller and are not newly discovered.
    echoed_fields = {
        output_field
        for output_field in known_fields
        if _dependency_argument_bindings(
            server_name, {output_field}, source_required,
        )
    }
    return known_fields - echoed_fields


def _dependency_argument_bindings(
    server_name: str,
    source_fields: set[str],
    target_required: set[str],
) -> list[tuple[str, str]]:
    """Map factual source fields to semantically equivalent target arguments."""
    bindings: list[tuple[str, str]] = []
    aliases = OUTPUT_ARGUMENT_ALIASES.get(server_name, {})
    for source_field in sorted(source_fields):
        target_candidates = {
            source_field,
            *aliases.get(source_field, ()),
        }
        for target_argument in sorted(target_candidates & target_required):
            bindings.append((source_field, target_argument))
    return bindings


def _sampled_chain_edges(
    chain: list[str],
    graph: dict[str, dict[str, list[str]]],
) -> list[dict[str, str]]:
    """Preserve the classifier relation for every sampled adjacent edge."""
    edges: list[dict[str, str]] = []
    for source, target in zip(chain, chain[1:]):
        relations = [
            relation
            for relation in ("explicit", "implicit")
            if target in graph.get(source, {}).get(relation, [])
        ]
        if len(relations) != 1:
            raise RuntimeError(
                f"sampled chain edge {source}->{target} has relations={relations}"
            )
        edges.append({
            "source_capability": source,
            "target_capability": target,
            "relation": relations[0],
        })
    return edges


def _operational_dependency_contracts(
    chain: list[str],
    server_name: str,
    tool_schemas: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """Return deterministic source-output -> required-target contracts.

    This is a local trainability precheck for dependency-focused prompt
    profiles. It does not change the cached PROVE-style pair classifications.
    """
    required_by_tool = _required_arguments_by_tool(tool_schemas)
    tools_by_name = {
        str(tool.get("name") or ""): tool for tool in tool_schemas
    }
    contract_registry = build_contract_registry({server_name: tool_schemas})
    contracts: list[dict[str, str]] = []
    for source_index, source_name in enumerate(chain[:-1]):
        source_fields = _novel_dependency_output_fields(
            server_name, source_name, tools_by_name,
        )
        if not source_fields:
            continue
        for target_name in chain[source_index + 1:]:
            for source_field, target_argument in _dependency_argument_bindings(
                server_name,
                source_fields,
                required_by_tool.get(target_name, set()),
            ):
                contracts.append({
                    "source_capability": source_name,
                    "target_capability": target_name,
                    "target_argument": target_argument,
                    "source_output_field": source_field,
                })
    return contracts


def _difficulty_vector_for_chain(
    *,
    chain: list[str],
    server_name: str,
    server_tools: list[dict[str, Any]],
    chain_context: dict[str, Any],
    feasible_chains: list[list[str]],
    distractor_count: int,
) -> DifficultyVector:
    """Measure decision structure before Query/Action Teacher generation."""
    contracts = _operational_dependency_contracts(
        chain, server_name, server_tools,
    )
    contract_registry = build_contract_registry({server_name: server_tools})
    hidden_types = {
        str(value)
        for value in chain_context.get("opaque_id_hidden_types", [])
        if str(value)
    }
    selector_candidates = {
        (str(record.get("type") or ""), str(record.get("id") or ""))
        for record in chain_context.get("entity_records", [])
        if isinstance(record, dict)
        and str(record.get("type") or "") in hidden_types
        and str(record.get("id") or "")
    }
    observation_arguments = {
        (
            str(item.get("target_capability") or ""),
            str(item.get("target_argument") or ""),
        )
        for item in contracts
        if item.get("target_capability") and item.get("target_argument")
    }
    tools_by_name = {
        str(tool.get("name") or ""): tool
        for tool in server_tools
        if isinstance(tool, dict)
    }
    post_mutation_rechecks = 0
    mutated_entity_types: set[str] = set()
    for tool_name in chain:
        semantics = resolve_tool_execution_semantics(tool_name, server_name)
        contract = contract_registry.get(server_name, tool_name)
        entity_types = set(contract.required_entity_types) | {
            binding.entity_type for binding in contract.output_entities
        }
        if semantics == "readonly" and mutated_entity_types & entity_types:
            post_mutation_rechecks += 1
        if semantics == "state_transition":
            mutated_entity_types.update(entity_types)
        # Fail closed on schemas that claim mutation even when the local
        # semantic registry has not classified the capability yet.
        annotations = tools_by_name.get(tool_name, {}).get("annotations") or {}
        if annotations.get("mutating") is True:
            mutated_entity_types.update(entity_types)
    final_capability = chain[-1] if chain else ""
    viable_same_outcome = sum(
        1 for candidate in feasible_chains
        if candidate and candidate[-1] == final_capability
    )
    return DifficultyVector(
        selector_candidate_count=len(selector_candidates),
        viable_chain_count=max(1, viable_same_outcome),
        operational_dependency_count=len(contracts),
        observation_derived_argument_count=len(observation_arguments),
        post_mutation_recheck_count=post_mutation_rechecks,
        distractor_count=max(0, int(distractor_count)),
        oracle_tool_count=len(chain),
    )


def _decision_stratum(vector: DifficultyVector) -> str:
    """Map a static vector to one mutually exclusive gray-test stratum."""
    if vector.post_mutation_recheck_count > 0:
        return "stateful"
    if vector.observation_derived_argument_count >= 2:
        return "dependent"
    if vector.selector_candidate_count >= 2:
        return "discovery"
    return "direct"


def _verify_dependency_evidence(
    evidence: tuple[dict[str, str], ...] | list[dict[str, str]],
    oracle_calls: list[OracleCall],
    oracle_observations: list[Any],
    user_query: str,
    tool_schemas: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Verify a downstream argument was produced by an earlier observation.

    This is a profile-specific trainability contract, not a PROVE corpus gate.
    Calls and observations are 1:1 aligned, including empty terminal entries.
    """
    verified: list[dict[str, Any]] = []
    required_by_tool = _required_arguments_by_tool(tool_schemas)
    for item in evidence:
        source_name = str(item.get("source_capability") or "")
        target_name = str(item.get("target_capability") or "")
        target_argument = str(item.get("target_argument") or "")
        source_output_field = str(item.get("source_output_field") or "")
        if not all((
            source_name, target_name, target_argument, source_output_field,
        )):
            continue
        if target_argument not in required_by_tool.get(target_name, set()):
            continue
        for source_index, (source_call, observation) in enumerate(zip(
            oracle_calls, oracle_observations, strict=True,
        )):
            if (
                getattr(source_call, "action", "tool_call") != "tool_call"
                or source_call.tool_name != source_name
            ):
                continue
            observed_values = _field_values(
                observation, source_output_field,
            )
            if not observed_values:
                continue
            observed_keys = {
                _dependency_value_key(value): value for value in observed_values
            }
            for target_index in range(source_index + 1, len(oracle_calls)):
                target_call = oracle_calls[target_index]
                if (
                    getattr(target_call, "action", "tool_call") != "tool_call"
                    or target_call.tool_name != target_name
                    or target_argument not in (target_call.arguments or {})
                ):
                    continue
                target_value = target_call.arguments[target_argument]
                target_values = (
                    list(target_value)
                    if isinstance(target_value, (list, tuple, set))
                    else [target_value]
                )
                target_keys = {
                    _dependency_value_key(value) for value in target_values
                }
                if not target_keys or not target_keys.issubset(observed_keys):
                    continue
                if _value_is_explicit_in_query(target_value, user_query):
                    continue
                verified.append({
                    **item,
                    "evidence_type": "explicit_value_binding",
                    "source_call_index": source_index,
                    "target_call_index": target_index,
                    "value_sha256": hashlib.sha256(
                        "\n".join(sorted(target_keys)).encode("utf-8")
                    ).hexdigest(),
                })
                break
            else:
                continue
            break
    return verified


def _verify_realized_chain_dependencies(
    chain: list[str],
    contracts: list[dict[str, str]],
    oracle_calls: list[OracleCall],
    oracle_observations: list[Any],
    user_query: str,
    tool_schemas: list[dict[str, Any]],
    server_name: str,
    sampled_edges: list[dict[str, str]],
    counterfactual_evidence: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Verify every adjacent chain edge against executed value flow."""
    verified = _verify_dependency_evidence(
        contracts,
        oracle_calls,
        oracle_observations,
        user_query,
        tool_schemas,
    )
    verified.extend(counterfactual_evidence or [])
    issues: list[str] = []
    if len(sampled_edges) != max(0, len(chain) - 1):
        return verified, ["sampled edge relation count mismatch"]
    for edge_index, (source_name, target_name) in enumerate(zip(chain, chain[1:])):
        relation_entry = sampled_edges[edge_index]
        relation = str(relation_entry.get("relation") or "")
        if (
            relation_entry.get("source_capability") != source_name
            or relation_entry.get("target_capability") != target_name
            or relation not in {"explicit", "implicit"}
        ):
            issues.append(
                f"{source_name}->{target_name}: sampled relation metadata mismatch"
            )
            continue
        edge_contracts = [
            item for item in contracts
            if item.get("source_capability") == source_name
            and item.get("target_capability") == target_name
        ]
        if relation == "explicit":
            if not any(
                item.get("source_capability") == source_name
                and item.get("target_capability") == target_name
                and item.get("evidence_type") == "explicit_value_binding"
                for item in verified
            ):
                issues.append(
                    f"{source_name}->{target_name}: no realized value binding"
                )
            continue
        implicit_match = next((
            item for item in (counterfactual_evidence or [])
            if item.get("source_capability") == source_name
            and item.get("target_capability") == target_name
            and item.get("evidence_type") == "implicit_counterfactual_replay"
        ), None)
        if implicit_match is None:
            issues.append(
                f"{source_name}->{target_name}: no explicit binding or "
                "counterfactual implicit evidence"
            )
    return verified, issues
