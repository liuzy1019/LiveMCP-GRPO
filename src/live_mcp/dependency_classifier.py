"""Teacher pair classification with complete-ledger provenance."""

from __future__ import annotations

import re
from typing import Any

from loguru import logger

from src.live_mcp.contracts.factory import build_contract_registry
from src.live_mcp.contracts.state_relations import render_predicate
from src.live_mcp.contracts.value_flow import novel_output_fields
from src.live_mcp.domain_contracts.value_bindings import OUTPUT_ARGUMENT_ALIASES
from src.utils import extract_json as _extract_json


class DependencyClassifierMixin:
    def _classify_edges_llm(
        self,
        server_tools: list[dict],
        server_name: str,
    ) -> tuple[
        dict,
        list[dict[str, Any]],
        list[dict[str, Any]],
    ] | None:
        """Classify pairwise tool relationships with the teacher LLM.

        Sends each unordered C(n,2) tool pair to the LLM once. The LLM selects
        source and target when the relationship is directed, then classifies it
        as explicit, implicit, or none.

        Returns the graph, the complete pair ledger, and independent-review
        provenance, or None if any required classification fails.
        """
        tool_names = sorted(t["name"] for t in server_tools)
        n = len(tool_names)
        if n < 2:
            return None
        contract_registry = build_contract_registry({server_name: server_tools})
        contracts_by_name = {
            contract.name: contract
            for contract in contract_registry.domain(server_name)
        }

        # Build compact tool descriptions for the LLM
        tool_descs: list[str] = []
        for t in server_tools:
            name = t["name"]
            desc = t.get("description", "")
            contract = contracts_by_name[name]
            output_fields = contract.output_fields
            props = t.get("input_schema", {}).get("properties", {})
            required = t.get("input_schema", {}).get("required", [])
            novel_fields = sorted(novel_output_fields(contract))
            echoed_output_fields = sorted(
                set(output_fields) - set(novel_fields)
            )
            param_lines = []
            for pk, pv in props.items():
                req_mark = "*" if pk in required else ""
                ptype = pv.get("type", "?")
                pdesc = pv.get("description", "")
                param_lines.append(f"    {pk}{req_mark} ({ptype}){': ' + pdesc if pdesc else ''}")
            params_str = "\n".join(param_lines) if param_lines else "    (none)"
            output_fields_str = (
                f"\n  Novel/discovered output fields: "
                f"{', '.join(novel_fields) if novel_fields else '(none)'}"
                f"\n  Echoed caller-provided fields (not new outputs): "
                f"{', '.join(echoed_output_fields) if echoed_output_fields else '(none)'}"
            )
            output_bindings = [
                f"{source_field} -> {target_argument}"
                for source_field in novel_fields
                for target_argument in OUTPUT_ARGUMENT_ALIASES.get(
                    server_name, {},
                ).get(source_field, ())
            ]
            output_bindings_str = (
                f"\n  Known target-argument aliases: {', '.join(output_bindings)}"
                if output_bindings else ""
            )
            preconditions = [
                render_predicate(predicate)
                for predicate in contract.preconditions
            ] + [
                "any(" + " | ".join(
                    render_predicate(predicate) for predicate in group
                ) + ")"
                for group in contract.precondition_groups
            ]
            postconditions = [
                render_predicate(predicate)
                for predicate in contract.postconditions
            ]
            state_contract_str = (
                f"\n  Required state predicates: "
                f"{', '.join(preconditions) if preconditions else '(none)'}"
                f"\n  Established state predicates: "
                f"{', '.join(postconditions) if postconditions else '(none)'}"
            )
            tool_descs.append(
                f"Tool: {name}\n"
                f"  Description: {desc}\n"
                f"  Parameters:\n{params_str}"
                f"{output_fields_str}"
                f"{output_bindings_str}"
                f"{state_contract_str}"
            )

        tool_desc_by_name = {
            str(tool.get("name") or ""): desc
            for tool, desc in zip(server_tools, tool_descs)
        }

        pairs = [
            (tool_names[i], tool_names[j])
            for i in range(len(tool_names))
            for j in range(i + 1, len(tool_names))
        ]
        logger.debug(
            f"_classify_edges_llm: {server_name} classifying {len(pairs)} "
            f"unordered tool pairs"
        )

        # Batch pairs to fit LLM context and bound single-request decode time.
        # Classify all C(n,2) pairs; each request carries one schema batch.
        BATCH_SIZE = self.DEPENDENCY_PAIR_BATCH_SIZE
        directed_classifications: dict[
            tuple[str, str], tuple[str, str, str]
        ] = {}
        classified_pairs: set = set()

        all_pair_keys: set[tuple[str, str]] = set(pairs)
        expected_pair_count = len(all_pair_keys)
        BATCH_RETRIES = 2
        tool_order = {name: index for index, name in enumerate(tool_names)}

        def _canonical_pair(a_name: str, b_name: str) -> tuple[str, str]:
            return (
                (a_name, b_name)
                if tool_order[a_name] < tool_order[b_name]
                else (b_name, a_name)
            )

        tools_by_name = {
            str(tool.get("name") or ""): tool for tool in server_tools
        }

        def _consume_classifications(
            data: Any,
            valid_pairs: set[tuple[str, str]],
        ) -> None:
            if not isinstance(data, dict):
                return
            entries = data.get("classifications", [])
            if not isinstance(entries, list):
                return
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                relation = str(entry.get("relation", "none")).strip().lower()
                if relation not in ("explicit", "implicit", "none"):
                    continue
                source = str(entry.get("source") or "").strip()
                target = str(entry.get("target") or "").strip()
                pair_text = str(entry.get("pair") or "")
                pair_parts = [
                    part.strip()
                    for part in re.split(r"\s*→\s*", pair_text, maxsplit=1)
                ]
                pair_members = (
                    pair_parts
                    if len(pair_parts) == 2
                    and all(part in tool_desc_by_name for part in pair_parts)
                    and pair_parts[0] != pair_parts[1]
                    else []
                )
                if not pair_members and (
                    source in tool_desc_by_name
                    and target in tool_desc_by_name
                    and source != target
                ):
                    pair_members = [source, target]
                if not pair_members:
                    continue
                pair_key = _canonical_pair(pair_members[0], pair_members[1])
                if pair_key not in valid_pairs or pair_key in classified_pairs:
                    continue

                if source not in tool_desc_by_name or target not in tool_desc_by_name:
                    if relation != "none":
                        continue
                if relation != "none" and (
                    source == target or _canonical_pair(source, target) != pair_key
                ):
                    continue
                classified_pairs.add(pair_key)
                if relation != "none":
                    directed_classifications[pair_key] = (
                        source, target, relation,
                    )

        for batch_start in range(0, len(pairs), BATCH_SIZE):
            batch_pairs = pairs[batch_start:batch_start + BATCH_SIZE]
            valid_batch_pairs = set(batch_pairs)
            if valid_batch_pairs <= classified_pairs:
                continue

            system = self._dependency_classifier_system_prompt()
            batch_complete = False
            for batch_attempt in range(BATCH_RETRIES + 1):
                # Ask only for pairs that are still missing. Re-sending the full
                # batch makes deterministic teachers repeat the same prefix and
                # never fill truncated/omitted tail entries.
                pending_pairs = [
                    pair for pair in batch_pairs
                    if pair not in classified_pairs
                ]
                pending_tool_names = sorted({
                    name for pair in pending_pairs for name in pair
                })
                pending_tools_text = "\n\n".join(
                    tool_desc_by_name[name]
                    for name in pending_tool_names
                    if name in tool_desc_by_name
                )
                def _pair_binding_hint(pair: tuple[str, str]) -> str:
                    candidates = self._dependency_pair_binding_candidates(
                        pair, tools_by_name, server_name,
                    )
                    if not candidates:
                        return "no verified direct required-input binding"
                    return (
                        "verified explicit binding candidate(s): "
                        + ", ".join(candidates)
                    )
                pending_pairs_text = "\n".join(
                    f"{i + 1}. {pair[0]} → {pair[1]} "
                    f"[{_pair_binding_hint(pair)}]"
                    for i, pair in enumerate(pending_pairs)
                )
                user = (
                    f"## Server: {server_name}\n\n"
                    f"## Tools\n{pending_tools_text}\n\n"
                    f"## Pairs to Classify\n{pending_pairs_text}\n\n"
                    f"Classify every listed pair exactly once. The displayed A → B "
                    f"only identifies the pair; for explicit/implicit you may return "
                    f"either A as source and B as target or the reverse direction. "
                    f"Do not omit any pair.\n\n"
                    f"## Output Format\n"
                    f'{{"classifications": [\n'
                    f'  {{"pair": "tool_a → tool_b", "source": "tool_a", "target": "tool_b", "relation": "explicit"}},\n'
                    f'  {{"pair": "tool_c → tool_d", "source": "tool_c", "target": "tool_d", "relation": "implicit"}},\n'
                    f'  {{"pair": "tool_e → tool_f", "source": "tool_e", "target": "tool_f", "relation": "none"}}\n'
                    f']}}\n\n'
                    f"Output ONLY the JSON, nothing else:"
                )
                try:
                    raw = self.client.generate_chat(
                        [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
                        temperature=0.1 + 0.05 * batch_attempt,
                        max_tokens=self.DEPENDENCY_CLASSIFICATION_MAX_TOKENS,
                    )
                    _consume_classifications(_extract_json(raw), valid_batch_pairs)
                except Exception as e:
                    logger.debug(
                        f"_classify_edges_llm batch {batch_start // BATCH_SIZE + 1} "
                        f"attempt {batch_attempt + 1}/{BATCH_RETRIES + 1} "
                        f"failed for {server_name}: {e}"
                    )
                if valid_batch_pairs <= classified_pairs:
                    batch_complete = True
                    break
                missing_count = len(valid_batch_pairs - classified_pairs)
                logger.debug(
                    f"_classify_edges_llm batch {batch_start // BATCH_SIZE + 1} "
                    f"missing {missing_count}/{len(valid_batch_pairs)} pair(s) after "
                    f"attempt {batch_attempt + 1}"
                )
            if not batch_complete:
                logger.warning(
                    f"_classify_edges_llm batch {batch_start // BATCH_SIZE + 1} "
                    f"incomplete after {BATCH_RETRIES + 1} attempts for {server_name}"
                )
                return None
            logger.info(
                f"Dependency cache {server_name}: classified "
                f"{len(classified_pairs)}/{expected_pair_count} unordered pairs"
            )

        # P1-1: completeness gate — refuse to return an incomplete graph.
        # Expected: C(n,2) unordered pairs, each classified exactly once.
        # classification (explicit, implicit, or none).
        if len(classified_pairs) != expected_pair_count:
            logger.warning(
                f"_classify_edges_llm completeness check FAILED for {server_name}: "
                f"classified {len(classified_pairs)}/{expected_pair_count} pairs. "
                f"Discarding partial graph — it will NOT be cached."
            )
            return None

        pair_classifications: list[dict[str, Any]] = []
        for pair_key in sorted(all_pair_keys):
            directed = directed_classifications.get(pair_key)
            if directed is None:
                source = target = ""
                relation = "none"
            else:
                source, target, relation = directed
            pair_classifications.append({
                "pair": list(pair_key),
                "source": source,
                "target": target,
                "relation": relation,
            })

        if self._uses_paper_baseline():
            # PROVE publishes one LLM classification for every unordered pair;
            # it does not publish a second per-pair reviewer or a handcrafted
            # relation-audit gate. Persist the complete raw ledger immediately.
            return (
                self._graph_from_pair_classifications(
                    pair_classifications, tool_names,
                ),
                pair_classifications,
                [],
            )

        initial_by_pair = {
            tuple(entry["pair"]): {
                "source": entry["source"],
                "target": entry["target"],
                "relation": entry["relation"],
            }
            for entry in pair_classifications
        }

        def _parse_single_pair_decision(
            raw: str,
            pair: tuple[str, str],
        ) -> tuple[dict[str, str] | None, str]:
            data = _extract_json(raw)
            if not isinstance(data, dict):
                return None, "response is not a JSON object"
            entries = data.get("classifications")
            if not isinstance(entries, list) or len(entries) != 1:
                return None, "response must contain exactly one classification"
            entry = entries[0]
            if not isinstance(entry, dict):
                return None, "classification is not a JSON object"
            relation = str(entry.get("relation") or "").strip().lower()
            source = str(entry.get("source") or "").strip()
            target = str(entry.get("target") or "").strip()
            pair_text = str(entry.get("pair") or "")
            pair_parts = [
                part.strip()
                for part in re.split(r"\s*(?:→|↔)\s*", pair_text, maxsplit=1)
            ]
            if (
                len(pair_parts) != 2
                or set(pair_parts) != set(pair)
            ):
                return None, "classification pair does not match the requested pair"
            if relation == "none":
                # Direction has no meaning for a none relation. Batch parsing
                # already canonicalizes these endpoints, so the independent
                # review must not reject a correct semantic label merely
                # because the model repeated the displayed pair names.
                source = target = ""
            elif (
                relation not in ("explicit", "implicit")
                or source == target
                or {source, target} != set(pair)
            ):
                return None, "directed classification has invalid endpoints or relation"
            return {
                "source": source,
                "target": target,
                "relation": relation,
            }, ""

        def _request_pair_review(
            pair: tuple[str, str],
            bindings: list[str],
        ) -> dict[str, str] | None:
            tools_text = "\n\n".join(
                tool_desc_by_name[name] for name in pair
            )
            binding_text = (
                "\n".join(f"- {binding}" for binding in bindings)
                if bindings else "- none"
            )
            user_prefix = (
                f"## Server: {server_name}\n\n"
                f"## Tools\n{tools_text}\n\n"
                f"## Unordered Pair\n{pair[0]} ↔ {pair[1]}\n\n"
                f"## Verified Binding Candidates\n{binding_text}\n\n"
                "Audit this pair against the explicit/implicit/none definitions. "
                "For none, source and target must be empty strings."
            )
            output_format = (
                "## Output Format\n"
                '{"classifications": [{"pair": "'
                f'{pair[0]} ↔ {pair[1]}", '
                '"source": "<directed source or empty>", '
                '"target": "<directed target or empty>", '
                '"relation": "<explicit, implicit, or none>"}]}\n\n'
                "Output ONLY the JSON, nothing else:"
            )
            rejection_reason = ""
            for attempt in range(3):
                feedback = (
                    "\n\n## Contract Feedback\n"
                    f"The previous response was rejected because "
                    f"{rejection_reason}.\n"
                    "Reclassify the pair without repeating the rejected "
                    "relation or direction."
                    if rejection_reason else ""
                )
                user = f"{user_prefix}{feedback}\n\n{output_format}"
                try:
                    raw = self.client.generate_chat(
                        [{
                            "role": "system",
                            "content": self._dependency_pair_audit_system_prompt(),
                        }, {
                            "role": "user",
                            "content": user,
                        }],
                        temperature=0.05 + 0.05 * attempt,
                        max_tokens=self.DEPENDENCY_CLASSIFICATION_MAX_TOKENS,
                    )
                    decision, rejection_reason = _parse_single_pair_decision(
                        raw, pair,
                    )
                    if decision is not None:
                        return decision
                    logger.debug(
                        f"Dependency pair audit rejected {pair} for "
                        f"{server_name} on attempt {attempt + 1}/3: "
                        f"{rejection_reason}"
                    )
                except Exception as exc:
                    logger.debug(
                        f"Dependency pair audit {pair} attempt "
                        f"{attempt + 1}/3 failed for {server_name}: {exc}"
                    )
            return None

        pair_audits: list[dict[str, Any]] = []
        audit_pairs = sorted(all_pair_keys)
        for audit_index, pair in enumerate(audit_pairs, start=1):
            bindings = self._dependency_pair_binding_candidates(
                pair, tools_by_name, server_name,
            )
            initial = initial_by_pair[pair]
            review = _request_pair_review(pair, bindings)
            if review is None:
                logger.warning(
                    f"Dependency pair audit failed for {server_name}: {pair}"
                )
                return None
            disagrees_with_raw = review != initial
            pair_audits.append({
                "pair": list(pair),
                "binding_candidates": bindings,
                "initial": initial,
                "review": review,
                "tie_break": None,
                "final": initial,
                "disagrees_with_raw": disagrees_with_raw,
            })
            logger.info(
                f"Dependency cache {server_name}: adversarially audited "
                f"{audit_index}/{len(audit_pairs)} unordered pairs"
            )

        validated_audits = self._validate_dependency_pair_audits(
            pair_audits, pair_classifications, server_tools, server_name,
        )
        if validated_audits is None:
            logger.warning(
                f"Dependency pair audit provenance invalid for {server_name}"
            )
            return None
        relation_audits = self._build_local_relation_audits(
            pair_classifications, server_tools, server_name,
        )
        graph = self._eligible_graph_from_relation_audits(
            pair_classifications, relation_audits, tool_names,
        )

        # The raw ledger remains immutable PROVE provenance. Only locally
        # supported raw edges enter the graph consumed by chain extraction.
        return graph, pair_classifications, validated_audits
