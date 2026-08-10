"""Formatting helpers for LiveMCP task planning prompts."""

from __future__ import annotations

import json as _json
from typing import Any

from src.live_mcp.protocol.observation import (
    DEFAULT_TEACHER_OBSERVATION_CHARS,
    project_observation,
)


def format_tools(tool_schemas: list[dict[str, Any]], strip_enums: bool = False) -> str:
    """Format tool schemas as human-readable text, optionally hiding enum values."""
    lines: list[str] = []
    for tool in tool_schemas:
        name = tool["name"]
        desc = tool.get("description", "")
        annotations = tool.get("annotations") or {}
        if annotations.get("readonly") is True and annotations.get("mutating") is False:
            desc = (
                f"{desc.rstrip()} Read-only: this tool does not modify server state. "
                "Any transformed content is display-only output and is never persisted."
            ).strip()
        props = tool.get("input_schema", {}).get("properties", {})
        required = tool.get("input_schema", {}).get("required", [])
        args_parts = []
        for k, info in props.items():
            if strip_enums and "enum" in info:
                info = {kk: vv for kk, vv in info.items() if kk != "enum"}
            req = "*" if k in required else ""
            ptype = _schema_type_hint(info)
            enum_str = f": {', '.join(info['enum'])}" if "enum" in info else ""
            desc_part = f" ({ptype}{enum_str})" if ptype else ""
            param_desc = str(info.get("description") or "").strip()
            desc_suffix = f" — {param_desc}" if param_desc else ""
            args_parts.append(f"{k}{req}{desc_part}{desc_suffix}")
        args_str = ", ".join(args_parts)
        lines.append(f"  - {name}({args_str}): {desc}")
    return "\n".join(lines)


def _schema_type_hint(schema: dict[str, Any]) -> str:
    """Render the argument structure the Teacher must actually produce."""
    ptype = str(schema.get("type") or "")
    if ptype == "array" and isinstance(schema.get("items"), dict):
        return f"array<{_schema_type_hint(schema['items'])}>"
    if ptype == "object":
        properties = schema.get("properties") or {}
        required = set(schema.get("required") or [])
        fields = []
        for name, child in properties.items():
            marker = "*" if name in required else ""
            child_hint = _schema_type_hint(child) if isinstance(child, dict) else ""
            child_desc = str(child.get("description") or "").strip() if isinstance(child, dict) else ""
            fields.append(
                f"{name}{marker}: {child_hint or 'any'}"
                + (f" [{child_desc}]" if child_desc else "")
            )
        return "object{" + ", ".join(fields) + "}"
    constraints = []
    if "enum" in schema:
        constraints.append("enum=" + "|".join(str(item) for item in schema["enum"]))
    if "minimum" in schema:
        constraints.append(f"minimum={schema['minimum']}")
    if "exclusiveMinimum" in schema:
        constraints.append(f"exclusiveMinimum={schema['exclusiveMinimum']}")
    if "maximum" in schema:
        constraints.append(f"maximum={schema['maximum']}")
    if constraints:
        return f"{ptype}({', '.join(constraints)})"
    return ptype


def format_state_compact(state: dict[str, Any], max_entities: int = 20) -> str:
    """Format grounded state as compact entity summaries.

    Instead of dumping full JSON (which can exceed teacher attention window),
    output one line per entity with key fields only.
    """
    if not isinstance(state, dict) or not state:
        return "(empty state)"

    lines: list[str] = []
    groups: list[tuple[str, list[tuple[str, Any]]]] = []
    for entity_type, entities in sorted(state.items()):
        if isinstance(entities, dict) and entities:
            groups.append((entity_type, sorted(entities.items())))

    # Round-robin over resource types.  A global first-N truncation made later
    # types disappear completely (for example payments after invoices).
    selected: list[tuple[str, str, Any]] = []
    index = 0
    while len(selected) < max_entities:
        added = False
        for entity_type, entities in groups:
            if index < len(entities):
                entity_id, entity_data = entities[index]
                selected.append((entity_type, entity_id, entity_data))
                added = True
                if len(selected) >= max_entities:
                    break
        if not added:
            break
        index += 1

    for entity_type, entity_id, entity_data in selected:
        if isinstance(entity_data, dict):
                # Extract key identity fields (expanded for all domains)
            id_fields: list[str] = []
            for fk in (
                    "name", "title", "subject", "status", "type",
                    "balance", "amount", "price", "quantity",
                    "date", "start_time", "end_time", "due_date",
                    "priority", "stage", "label", "category",
                    "sender", "recipient",
            ):
                if fk in entity_data:
                    val = entity_data[fk]
                    if isinstance(val, str) and len(val) > 60:
                        val = val[:57] + "..."
                    id_fields.append(f"{fk}={val}")
                # Also capture id-like fields
            for fk, fv in entity_data.items():
                if fk.endswith("_id") or fk.endswith("_name"):
                    id_fields.append(f"{fk}={fv}")
            if entity_data.get("summary"):
                id_fields.append(f"facts={entity_data['summary']}")
            summary = ", ".join(id_fields[:5])
            lines.append(f"  {entity_type}/{entity_id}: {summary}" if summary else f"  {entity_type}/{entity_id}")
        else:
            lines.append(f"  {entity_type}/{entity_id}: {entity_data}")
    total_entities = sum(len(entities) for _, entities in groups)
    if total_entities > len(selected):
        shown_by_type: dict[str, int] = {}
        for entity_type, _, _ in selected:
            shown_by_type[entity_type] = shown_by_type.get(entity_type, 0) + 1
        distribution = ", ".join(
            f"{entity_type}={shown_by_type.get(entity_type, 0)}/{len(entities)}"
            for entity_type, entities in groups
        )
        lines.append(
            f"... ({total_entities} total entities; stratified view: {distribution})"
        )
    if not lines:
        return str(state)[:2000]
    return "\n".join(lines)


def format_conversation_context(context: list[dict[str, Any]] | None) -> str:
    if not context:
        return "(this is the first conversation round)"
    lines: list[str] = []
    for item in context:
        round_idx = item.get("round_idx", "?")
        query = str(item.get("user_query") or "").strip()
        response = str(item.get("assistant_response") or "").strip()
        terminal = str(item.get("terminal_action") or "").strip()
        lines.append(f"Round {round_idx} user: {query or '(missing)'}")
        lines.append(
            f"Round {round_idx} assistant ({terminal or 'response'}): "
            f"{response or '(no visible text)'}"
        )
    return "\n".join(lines)


def format_history(
    history: list[dict[str, Any]],
    *,
    max_chars: int = DEFAULT_TEACHER_OBSERVATION_CHARS,
) -> str:
    """Format loss-aware MCP execution events for the next Agent decision."""
    if not history:
        return "(no actions yet — this is the first turn)"
    lines = []
    for i, entry in enumerate(history, 1):
        tool = entry.get("tool_name", "?")
        args = _json.dumps(entry.get("arguments", {}), ensure_ascii=False)
        success = entry.get("success", True)
        outcome = str(entry.get("execution_status") or ("SUCCESS" if success else "FAILURE"))
        state_changed = entry.get("state_changed")
        envelope = {
            "success": bool(success),
            "execution_status": outcome,
            "error_type": entry.get("error_type"),
            "error_message": str(entry.get("error_message") or ""),
            "state_changed": bool(state_changed),
            "schema_valid": bool(entry.get("schema_valid", False)),
            "observation": entry.get("observation"),
        }
        lines.append(
            f"Step {i}: {tool}({args}) → "
            f"{outcome}"
            + (f"; state_changed={bool(state_changed)}" if state_changed is not None else "")
        )
        error_type = str(envelope["error_type"] or "").strip()
        error_message = envelope["error_message"].strip()
        if error_type or error_message:
            lines.append(
                f"  Error: {error_type or 'execution_error'}"
                + (f": {error_message}" if error_message else "")
            )
        loop_warning = str(entry.get("no_progress_warning") or "").strip()
        if loop_warning:
            lines.append(f"  No-progress warning: {loop_warning}")
        lines.append(
            "  Result envelope: "
            f"{project_observation(envelope, max_chars=max_chars)}"
        )
    return "\n".join(lines)