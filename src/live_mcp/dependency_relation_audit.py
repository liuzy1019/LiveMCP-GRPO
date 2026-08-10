"""Deterministic relation audit and graph projection for PROVE dependencies."""

from __future__ import annotations

from typing import Any

from src.live_mcp.contracts.factory import build_contract_registry
from src.live_mcp.contracts.models import ToolContract
from src.live_mcp.contracts.chain_simulator import simulate_symbolic_chain
from src.live_mcp.contracts.state_relations import implicit_directions
from src.live_mcp.contracts.value_flow import value_bindings
from src.live_mcp.dependency_value_flow import (
    _dependency_argument_bindings,
    _novel_dependency_output_fields,
)
from src.live_mcp.domain_contracts.dependency import (
    _DEPENDENCY_ENTITY_TYPE_COMPATIBILITY,
    _DEPENDENCY_TOOL_OUTPUT_FIELDS,
    _DEPENDENCY_TOOL_STATE_POSTCONDITIONS,
    _DEPENDENCY_TOOL_STATE_PRECONDITIONS,
    _DOMAIN_TOOL_OUTPUT_ENTITY_TYPES,
)
from src.live_mcp.domain_contracts.entities import _CREATED_ENTITY_BY_TOOL
from src.live_mcp.domain_contracts.requirements import _DOMAIN_TOOL_REQUIREMENTS
from src.live_mcp.domain_contracts.states import DOMAIN_STATE_FACTS


def _canonical_contracts(
    domain: str,
    tools_by_name: dict[str, dict],
) -> dict[str, ToolContract] | None:
    if domain not in DOMAIN_STATE_FACTS:
        return None
    registry = build_contract_registry({domain: tools_by_name.values()})
    return {
        contract.name: contract for contract in registry.domain(domain)
    }


def _typed_pair_contradiction(
    source: ToolContract,
    target: ToolContract,
) -> str | None:
    """Describe a direct typed transition/precondition contradiction."""
    for postcondition in source.postconditions:
        for precondition in target.preconditions:
            if (
                postcondition.slot == precondition.slot
                and postcondition.subject.entity_type
                == precondition.subject.entity_type
                and postcondition.value != precondition.value
            ):
                return (
                    f"{postcondition.slot} becomes {postcondition.value!r}, "
                    f"but target requires {precondition.value!r}"
                )
    return None


class DependencyRelationAuditMixin:
    @classmethod
    def _pair_classification_contract_issue(
        cls,
        entry: dict[str, Any],
        tools_by_name: dict[str, dict],
        server_name: str = "",
    ) -> str | None:
        """Return a deterministic contradiction with the classifier contract."""
        relation = entry.get("relation")
        domain = server_name or str(
            next(
                (
                    tool.get("_server_name")
                    for tool in tools_by_name.values()
                    if tool.get("_server_name")
                ),
                "",
            )
        )
        pair = tuple(sorted(str(name) for name in entry.get("pair", [])))
        contracts_by_name = _canonical_contracts(domain, tools_by_name)
        certified_directions = cls._dependency_pair_certified_explicit_directions(
            pair, tools_by_name, domain,
        )
        if relation == "none":
            if certified_directions:
                rendered = ", ".join(
                    f"{source} -> {target}"
                    for source, target in certified_directions
                )
                return (
                    "none contradicts certified novel-output to required-input "
                    f"direction(s): {rendered}"
                )
            return None
        source = tools_by_name.get(str(entry.get("source") or ""), {})
        target = tools_by_name.get(str(entry.get("target") or ""), {})
        if relation == "explicit":
            source_name = str(entry.get("source") or "")
            target_name = str(entry.get("target") or "")
            typed_contradiction = (
                _typed_pair_contradiction(
                    contracts_by_name[source_name],
                    contracts_by_name[target_name],
                )
                if contracts_by_name is not None else None
            )
            if typed_contradiction:
                return (
                    "explicit source-produced state contradicts the typed "
                    f"target precondition: {typed_contradiction}"
                )
            required = set(
                (
                    target.get("input_schema")
                    or target.get("inputSchema")
                    or {}
                ).get("required") or []
            )
            if not required:
                return "explicit target has no required input"
            source_contract = (
                contracts_by_name[source_name]
                if contracts_by_name is not None else None
            )
            target_contract = (
                contracts_by_name[str(entry.get("target") or "")]
                if contracts_by_name is not None else None
            )
            output_contract = (
                set(source_contract.output_fields)
                if source_contract is not None
                else set(_DEPENDENCY_TOOL_OUTPUT_FIELDS.get(domain, {}).get(
                    source_name, (),
                ))
            )
            if output_contract:
                bindings = (
                    value_bindings(domain, source_contract, target_contract)
                    if source_contract is not None and target_contract is not None
                    else _dependency_argument_bindings(
                        domain,
                        _novel_dependency_output_fields(
                            domain, source_name, tools_by_name,
                        ),
                        required,
                    )
                )
                if not bindings:
                    return (
                        "explicit source has no novel output field matching a "
                        "target required input, including declared semantic "
                        "aliases; echoed source-required fields do not establish "
                        "a dependency"
                    )
            # An explicit edge supplies a required target input. Reject a
            # direct entity-type contradiction and retry the pair.
            domain = domain or str(
                source.get("_server_name") or target.get("_server_name") or ""
            )
            binding_source_fields = {source for source, _ in bindings}
            created_types = (
                {
                    binding.entity_type
                    for binding in source_contract.output_entities
                    if binding.name in binding_source_fields
                }
                if source_contract is not None
                else set(_DOMAIN_TOOL_OUTPUT_ENTITY_TYPES.get(domain, {}).get(
                    source_name, _CREATED_ENTITY_BY_TOOL.get(source_name, set()),
                ))
            )
            required_types = (
                set(target_contract.required_entity_types)
                if target_contract is not None
                else set(_DOMAIN_TOOL_REQUIREMENTS.get(domain, {}).get(
                    str(entry.get("target") or ""), set(),
                ))
            )
            compatible_created_types = set(created_types)
            if source_contract is None or target_contract is None:
                compatibility = _DEPENDENCY_ENTITY_TYPE_COMPATIBILITY.get(
                    domain, {}
                )
                for entity_type in created_types:
                    compatible_created_types.update(
                        compatibility.get(entity_type, set())
                    )
            if (
                created_types
                and required_types
                and compatible_created_types.isdisjoint(required_types)
            ):
                return (
                    "explicit source output entity types "
                    f"{sorted(created_types)} cannot satisfy target entity types "
                    f"{sorted(required_types)}"
                )
        if relation == "implicit":
            typed_contradiction = (
                _typed_pair_contradiction(
                    contracts_by_name[str(entry.get("source") or "")],
                    contracts_by_name[str(entry.get("target") or "")],
                )
                if contracts_by_name is not None else None
            )
            if typed_contradiction:
                return (
                    "implicit source-produced state contradicts the typed "
                    f"target precondition: {typed_contradiction}"
                )
            if certified_directions:
                rendered = ", ".join(
                    f"{source} -> {target}"
                    for source, target in certified_directions
                )
                return (
                    "implicit contradicts the explicit-precedence rule because "
                    "the certified output contract supports direction(s): "
                    f"{rendered}"
                )
            annotations = source.get("annotations") or {}
            if annotations.get("readonly") is True or annotations.get("mutating") is False:
                return "implicit source is explicitly readonly/non-mutating"
        return None

    @staticmethod
    def _dependency_pair_certified_implicit_directions(
        pair: tuple[str, str],
        tools_by_name: dict[str, dict],
        server_name: str = "",
    ) -> list[tuple[str, str]]:
        """Return directions proven by structured postcondition/precondition facts."""
        contracts_by_name = _canonical_contracts(server_name, tools_by_name)
        if contracts_by_name is None:
            preconditions = _DEPENDENCY_TOOL_STATE_PRECONDITIONS.get(
                server_name, {},
            )
            postconditions = _DEPENDENCY_TOOL_STATE_POSTCONDITIONS.get(
                server_name, {},
            )
            return [
                (source_name, target_name)
                for source_name, target_name in (pair, tuple(reversed(pair)))
                if source_name in postconditions
                and target_name in preconditions
                and postconditions[source_name] & preconditions[target_name]
            ]
        registry = build_contract_registry({
            server_name: tools_by_name.values(),
        })
        return [
            direction for direction in implicit_directions(pair, contracts_by_name)
            if not simulate_symbolic_chain(
                registry, server_name, list(direction),
            )[1]
        ]

    @classmethod
    def _local_dependency_relation_audit(
        cls,
        entry: dict[str, Any],
        tools_by_name: dict[str, dict],
        server_name: str = "",
        contracts_by_name: dict[str, ToolContract] | None = None,
    ) -> dict[str, Any]:
        """Audit one immutable raw label against structured local facts.

        This function never changes the raw relation.  It only decides whether
        that raw edge is eligible for chain extraction.
        """
        pair = tuple(str(name) for name in entry.get("pair", []))
        explicit_directions = cls._dependency_pair_certified_explicit_directions(
            pair, tools_by_name, server_name,
        )
        implicit_directions = cls._dependency_pair_certified_implicit_directions(
            pair, tools_by_name, server_name,
        )
        relation = str(entry.get("relation") or "")
        source = str(entry.get("source") or "")
        target = str(entry.get("target") or "")
        raw_direction = (source, target)

        if contracts_by_name is None:
            contracts_by_name = _canonical_contracts(server_name, tools_by_name)
        if contracts_by_name is None:
            output_contract = _DEPENDENCY_TOOL_OUTPUT_FIELDS.get(server_name, {})
            state_preconditions = _DEPENDENCY_TOOL_STATE_PRECONDITIONS.get(
                server_name, {},
            )
            state_postconditions = _DEPENDENCY_TOOL_STATE_POSTCONDITIONS.get(
                server_name, {},
            )
            output_coverage = all(name in output_contract for name in pair)
            state_coverage = all(
                name in state_preconditions and name in state_postconditions
                for name in pair
            )
            covered_contract_names = set(state_preconditions) & set(state_postconditions)
        else:
            output_contract = contracts_by_name
            output_coverage = all(name in contracts_by_name for name in pair)
            state_coverage = output_coverage
            covered_contract_names = set(contracts_by_name)

        verdict = "insufficient_evidence"
        reason = "local output/state contracts do not cover both tools"
        if relation == "explicit":
            if raw_direction in explicit_directions:
                verdict = "supported"
                reason = "novel output binds a required target input"
            elif source in output_contract:
                verdict = "contradicted"
                reason = (
                    "raw explicit direction has no certified novel-output "
                    "binding or violates a fixed target precondition"
                )
        elif relation == "implicit":
            if explicit_directions:
                verdict = "contradicted"
                reason = (
                    "raw implicit conflicts with a certified explicit binding "
                    "under the explicit-precedence definition"
                )
            elif raw_direction in implicit_directions:
                verdict = "supported"
                reason = (
                    "source postcondition establishes a target precondition"
                )
            elif source in covered_contract_names and target in covered_contract_names:
                verdict = "contradicted"
                reason = (
                    "source postconditions do not establish any target "
                    "precondition"
                )
        elif relation == "none":
            if explicit_directions or implicit_directions:
                verdict = "contradicted"
                reason = "raw none misses a locally certified dependency"
            elif output_coverage and state_coverage:
                verdict = "supported"
                reason = "both directions are covered and no dependency is certified"

        return {
            "pair": list(pair),
            "raw": {
                "source": source,
                "target": target,
                "relation": relation,
            },
            "verdict": verdict,
            "eligible": verdict == "supported" and relation != "none",
            "reason": reason,
            "certified_explicit_directions": [
                [src, dst]
                for src, dst in explicit_directions
            ],
            "certified_implicit_directions": [
                [src, dst]
                for src, dst in implicit_directions
            ],
            "output_contract_complete": output_coverage,
            "state_contract_complete": state_coverage,
        }

    @classmethod
    def _build_local_relation_audits(
        cls,
        pair_classifications: list[dict[str, Any]],
        server_tools: list[dict],
        server_name: str = "",
    ) -> list[dict[str, Any]]:
        tools_by_name = {
            str(tool.get("name") or ""): tool for tool in server_tools
        }
        contracts_by_name = _canonical_contracts(server_name, tools_by_name)
        return [
            cls._local_dependency_relation_audit(
                entry, tools_by_name, server_name, contracts_by_name,
            )
            for entry in pair_classifications
        ]

    @classmethod
    def _validate_local_relation_audits(
        cls,
        relation_audits: Any,
        pair_classifications: list[dict[str, Any]],
        server_tools: list[dict],
        server_name: str = "",
    ) -> list[dict[str, Any]] | None:
        expected = cls._build_local_relation_audits(
            pair_classifications, server_tools, server_name,
        )
        return expected if relation_audits == expected else None

    @classmethod
    def _eligible_graph_from_relation_audits(
        cls,
        pair_classifications: list[dict[str, Any]],
        relation_audits: list[dict[str, Any]],
        expected_tool_names: list[str],
    ) -> dict[str, dict[str, list[str]]]:
        eligible_pairs = {
            tuple(audit["pair"])
            for audit in relation_audits
            if audit.get("eligible") is True
        }
        eligible_ledger = [
            entry
            for entry in pair_classifications
            if tuple(entry["pair"]) in eligible_pairs
        ]
        return cls._graph_from_pair_classifications(
            eligible_ledger, expected_tool_names,
        )

    @classmethod
    def _pair_classification_retry_constraint(
        cls,
        pair: tuple[str, str],
        tools_by_name: dict[str, dict],
        server_name: str = "",
    ) -> str:
        """Describe a deterministic relation constraint for an invalid pair."""
        explicit_directions = cls._dependency_pair_certified_explicit_directions(
            pair, tools_by_name, server_name,
        )
        implicit_source_possible = False
        for source_name, target_name in (pair, tuple(reversed(pair))):
            source = tools_by_name.get(source_name, {})
            annotations = source.get("annotations") or {}
            if not (
                annotations.get("readonly") is True
                or annotations.get("mutating") is False
            ):
                implicit_source_possible = True

        if not explicit_directions and not implicit_source_possible:
            return (
                'Deterministic contract constraint: both tools are readonly and '
                'neither direction has a novel source output matching a target '
                'required input. The only valid answer is relation="none" with '
                'empty source and target.'
            )
        if explicit_directions:
            rendered = ", ".join(
                f"{source} -> {target}"
                for source, target in explicit_directions
            )
            return (
                "Deterministic contract constraint: the schema/output ledger "
                f"supports explicit direction(s) {rendered}."
            )
        return ""

    @staticmethod
    def _dependency_pair_certified_explicit_directions(
        pair: tuple[str, str],
        tools_by_name: dict[str, dict],
        server_name: str = "",
    ) -> list[tuple[str, str]]:
        """Return audited output bindings not blocked by fixed state facts.

        This is a local cache-certification contract. It does not infer edges
        from lexical similarity: every direction must come from the
        domain-specific factual output ledger and a required target argument.
        """
        if len(pair) != 2:
            return []
        contracts_by_name = _canonical_contracts(server_name, tools_by_name)
        directions: list[tuple[str, str]] = []
        registry = (
            build_contract_registry({server_name: tools_by_name.values()})
            if contracts_by_name is not None else None
        )
        for source_name, target_name in (pair, tuple(reversed(pair))):
            bindings = (
                value_bindings(
                    server_name,
                    contracts_by_name[source_name],
                    contracts_by_name[target_name],
                )
                if contracts_by_name is not None
                else _dependency_argument_bindings(
                    server_name,
                    _novel_dependency_output_fields(
                        server_name, source_name, tools_by_name,
                    ),
                    set(
                        (
                            tools_by_name[target_name].get("input_schema")
                            or tools_by_name[target_name].get("inputSchema")
                            or {}
                        ).get("required") or []
                    ),
                )
            )
            if not bindings:
                continue
            if registry is not None and simulate_symbolic_chain(
                registry, server_name, [source_name, target_name],
            )[1]:
                continue
            binding_source_fields = {source for source, _ in bindings}
            created_types = (
                {
                    binding.entity_type
                    for binding in contracts_by_name[source_name].output_entities
                    if binding.name in binding_source_fields
                }
                if contracts_by_name is not None
                else set(_DOMAIN_TOOL_OUTPUT_ENTITY_TYPES.get(
                    server_name, {},
                ).get(source_name, _CREATED_ENTITY_BY_TOOL.get(source_name, set())))
            )
            required_types = (
                set(contracts_by_name[target_name].required_entity_types)
                if contracts_by_name is not None
                else set(_DOMAIN_TOOL_REQUIREMENTS.get(
                    server_name, {},
                ).get(target_name, set()))
            )
            compatible_created_types = set(created_types)
            if contracts_by_name is None:
                compatibility = _DEPENDENCY_ENTITY_TYPE_COMPATIBILITY.get(
                    server_name, {}
                )
                for entity_type in created_types:
                    compatible_created_types.update(
                        compatibility.get(entity_type, set())
                    )
            if (
                created_types
                and required_types
                and compatible_created_types.isdisjoint(required_types)
            ):
                continue
            directions.append((source_name, target_name))
        return directions

    @staticmethod
    def _dependency_pair_binding_candidates(
        pair: tuple[str, str],
        tools_by_name: dict[str, dict],
        server_name: str = "",
    ) -> list[str]:
        """Render factual novel-output to required-input candidates for one pair."""
        contracts_by_name = _canonical_contracts(server_name, tools_by_name)
        candidates: list[str] = []
        for source_name, target_name in (pair, tuple(reversed(pair))):
            bindings = (
                value_bindings(
                    server_name,
                    contracts_by_name[source_name],
                    contracts_by_name[target_name],
                )
                if contracts_by_name is not None
                else _dependency_argument_bindings(
                    server_name,
                    _novel_dependency_output_fields(
                        server_name, source_name, tools_by_name,
                    ),
                    set(
                        (
                            tools_by_name[target_name].get("input_schema")
                            or tools_by_name[target_name].get("inputSchema")
                            or {}
                        ).get("required") or []
                    ),
                )
            )
            for source_field, target_argument in bindings:
                rendered = (
                    f"{source_name}.{source_field} -> "
                    f"{target_name}.{target_argument}"
                )
                if contracts_by_name is not None and simulate_symbolic_chain(
                    build_contract_registry({
                        server_name: tools_by_name.values(),
                    }),
                    server_name,
                    [source_name, target_name],
                )[1]:
                    rendered += " [blocked by typed state contradiction]"
                candidates.append(rendered)
        return sorted(set(candidates))

    @classmethod
    def _pair_classification_contract_issues(
        cls,
        pair_classifications: list[dict[str, Any]],
        server_tools: list[dict],
        server_name: str = "",
    ) -> list[dict[str, Any]]:
        tools_by_name = {
            str(tool.get("name") or ""): tool for tool in server_tools
        }
        issues: list[dict[str, Any]] = []
        for entry in pair_classifications:
            reason = cls._pair_classification_contract_issue(
                entry, tools_by_name, server_name,
            )
            if reason is not None:
                issues.append({"pair": list(entry["pair"]), "reason": reason})
        return issues

    @classmethod
    def _graph_from_pair_classifications(
        cls,
        pair_classifications: list[dict[str, Any]],
        expected_tool_names: list[str],
    ) -> dict[str, dict[str, list[str]]]:
        graph = {
            name: {"explicit": [], "implicit": []}
            for name in sorted(expected_tool_names)
        }
        for entry in pair_classifications:
            relation = entry["relation"]
            if relation == "none":
                continue
            graph[entry["source"]][relation].append(entry["target"])
        return cls._normalize_cached_graph(graph, sorted(expected_tool_names))

    @classmethod
    def _pair_classifications_from_graph(
        cls,
        graph: dict,
        expected_tool_names: list[str],
    ) -> list[dict[str, Any]]:
        """Reconstruct a v5 ledger from a trusted complete v4 graph."""
        normalized = cls._normalize_cached_graph(graph, sorted(expected_tool_names))
        entries: list[dict[str, Any]] = []
        for pair in sorted(cls._expected_dependency_pairs(expected_tool_names)):
            directed: list[tuple[str, str, str]] = []
            for source, target in (pair, (pair[1], pair[0])):
                for relation in ("explicit", "implicit"):
                    if target in normalized[source][relation]:
                        directed.append((source, target, relation))
            if len(directed) > 1:
                raise ValueError(
                    f"Dependency graph has multiple directed relations for pair {pair}: {directed}"
                )
            if directed:
                source, target, relation = directed[0]
            else:
                source = target = ""
                relation = "none"
            entries.append({
                "pair": list(pair),
                "source": source,
                "target": target,
                "relation": relation,
            })
        return entries
