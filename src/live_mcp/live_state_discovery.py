"""Readonly discovery pipeline for canonical PROVE live-state facts."""

from __future__ import annotations

from typing import Any

from src.live_mcp.contracts.factory import build_contract_registry
from src.live_mcp.dependency_value_flow import _dependency_value_key, _field_values
from src.live_mcp.domain_contracts.probes import _READONLY_REQUIRED_PROBE_ARGS
from src.live_mcp.live_state_availability import build_live_state_availability
from src.live_mcp.live_state_enrichment import (
    enrich_domain_entity_records,
    enrich_readonly_entity_records,
)
from src.live_mcp.live_state_projection import (
    extract_probe_entities,
    format_entity_summary,
)


def _is_readonly_discovery_tool(tool_schema: dict[str, Any]) -> bool:
    annotations = tool_schema.get("annotations") or {}
    if not bool(annotations.get("readonly")) or bool(annotations.get("mutating")):
        return False
    name = str(tool_schema.get("name") or "")
    return name.startswith(("list_", "search_")) or name in {
        "get_cart", "get_wishlist", "get_coupons", "get_user_status",
        "get_working_hours", "get_exchange_rate", "pwd", "ls", "tree", "df",
    }


def _readonly_probe_args(
    tool_schema: dict[str, Any], domain: str,
) -> dict[str, Any] | None:
    name = str(tool_schema.get("name") or "")
    required = list(tool_schema.get("input_schema", {}).get("required", []) or [])
    if not required:
        return {}
    args = _READONLY_REQUIRED_PROBE_ARGS.get(domain, {}).get(name)
    if args is None or not set(required) <= set(args):
        return None
    return dict(args)


def probe_live_sampling_context(
    executor: Any,
    session_id: str,
    server_name: str,
    server_tools: list[dict],
) -> dict[str, Any]:
    """Enumerate real entities through read-only tools.

    This is intentionally separate from ``debug/get_state``.  The sampler
    should only expose entities that a policy could discover through the
    live MCP interface.  Tools with required entity IDs are skipped here;
    they become usable after a list/search/get_cart style probe has surfaced
    concrete IDs.
    """
    from src.live_mcp.types import ToolCall

    entity_ids: list[dict[str, str]] = []
    entity_summaries: list[str] = []
    entity_records: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    entity_index: dict[tuple[str, str], int] = {}
    probe_results: list[dict[str, Any]] = []
    contract_registry = build_contract_registry({
        server_name: server_tools,
    })

    def add_entity(eid: str, etype: str, edata: dict[str, Any] | None = None) -> None:
        if not eid or not etype:
            return
        key = (etype, eid)
        if key in seen:
            # A broad discovery tool may reveal an opaque ID before a
            # dedicated lookup returns its descriptive fields. Preserve
            # identity deduplication but enrich the existing record so the
            # Query Teacher receives the most complete MCP-observed facts.
            if isinstance(edata, dict):
                idx = entity_index[key]
                existing = entity_records[idx].get("data")
                merged = {
                    **(existing if isinstance(existing, dict) else {}),
                    **edata,
                }
                entity_records[idx]["data"] = merged
                entity_summaries[idx] = format_entity_summary(
                    eid, etype, merged, domain=server_name,
                )
            return
        seen.add(key)
        entity_index[key] = len(entity_ids)
        entity_ids.append({"id": eid, "type": etype})
        entity_summaries.append(
            format_entity_summary(eid, etype, edata, domain=server_name)
        )
        entity_records.append({
            "id": eid,
            "type": etype,
            "data": dict(edata) if isinstance(edata, dict) else {},
        })

    for tool in sorted(server_tools, key=lambda t: str(t.get("name", ""))):
        tool_name = str(tool.get("name") or "")
        if not tool_name or not _is_readonly_discovery_tool(tool):
            continue
        args = _readonly_probe_args(tool, server_name)
        if args is None:
            continue
        result = executor.execute(
            session_id,
            ToolCall(tool_name, args, call_id=f"live_probe_{tool_name}"),
            domain=server_name,
        )
        output_field_values: dict[str, list[Any]] = {}
        for field_name in contract_registry.get(
            server_name, tool_name,
        ).output_fields:
            unique_values: dict[str, Any] = {}
            for value in _field_values(result.observation, field_name):
                if not isinstance(value, (str, int, float, bool)):
                    continue
                unique_values.setdefault(_dependency_value_key(value), value)
            output_field_values[field_name] = list(unique_values.values())[:100]
        probe_results.append({
            "tool": tool_name,
            "arguments": args,
            "success": bool(result.success),
            "state_changed": bool(result.state_changed),
            "error_type": result.error_type,
            "output_field_counts": {
                field_name: len(values)
                for field_name, values in output_field_values.items()
            },
            # Step-2 feasibility needs identity-level joins. Counts alone
            # cannot distinguish "a product exists in wishlist" from "the
            # same product is currently removable from cart".
            "output_field_values": output_field_values,
        })
        if not result.success or result.state_changed:
            continue
        extract_probe_entities(
            result.observation,
            add_entity,
            domain=server_name,
            tool_name=tool_name,
        )

    probe_results.extend(enrich_domain_entity_records(
        executor=executor,
        session_id=session_id,
        domain=server_name,
        tool_schemas=server_tools,
        entity_records=entity_records,
        probe_results=probe_results,
        add_entity=add_entity,
    ))

    # Detail probes are declared centrally and executed through the same
    # readonly MCP boundary as primary discovery.  This prevents missing
    # fields from being treated as valid target state.
    detail_probe_audits = enrich_readonly_entity_records(
        executor=executor,
        session_id=session_id,
        server_name=server_name,
        entity_records=entity_records,
        add_entity=add_entity,
    )
    probe_results.extend(detail_probe_audits)
    entity_summaries = [
        format_entity_summary(
            str(record.get("id") or ""),
            str(record.get("type") or "entity"),
            record.get("data") if isinstance(record.get("data"), dict) else {},
            domain=server_name,
        )
        for record in entity_records
    ]

    availability_audit = build_live_state_availability(
        server_name=server_name,
        tool_schemas=server_tools,
        entity_ids=entity_ids,
        entity_records=entity_records,
        contract_registry=contract_registry,
        probe_results=probe_results,
    )

    return {
        "source": "live_readonly_probe",
        "entity_ids": entity_ids,
        "entity_summaries": entity_summaries,
        "entity_records": entity_records,
        "entity_types": sorted({item["type"] for item in entity_ids}),
        "probe_results": probe_results,
        "probed_entity_count": len(entity_ids),
        **availability_audit,
    }

