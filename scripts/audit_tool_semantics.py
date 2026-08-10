#!/usr/bin/env python3
"""Audit all public tools against the canonical PROVE semantic contracts."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.live_mcp.contracts.catalog import domain_contract_registry
from src.live_mcp.contracts.factory import build_contract_registry
from src.live_mcp.generation.scenario import (
    classify_scenario,
    detect_duplicate_side_effect,
    detect_missing_dependency,
)
from src.live_mcp.registry.tool_semantics import (
    build_tool_semantics,
    is_mutating_tool,
)
from src.live_mcp.types import OracleCall


def load_domain_schemas() -> dict[str, list[dict[str, Any]]]:
    schemas: dict[str, list[dict[str, Any]]] = {}
    servers_root = PROJECT_ROOT / "src" / "live_mcp" / "servers"
    for server_path in sorted(servers_root.glob("*/server.py")):
        domain = server_path.parent.name
        module = importlib.import_module(
            f"src.live_mcp.servers.{domain}.server"
        )
        schemas[domain] = list(module.TOOLS)
    return schemas


def call(name: str, **arguments: Any) -> OracleCall:
    return OracleCall(
        action="tool_call", tool_name=name, arguments=arguments,
    )


def main() -> int:
    schemas = load_domain_schemas()
    registry = build_contract_registry(schemas)
    failures: list[str] = []
    total_tools = sum(len(items) for items in schemas.values())
    mutating_count = 0

    for domain, tool_schemas in sorted(schemas.items()):
        names = {str(schema["name"]) for schema in tool_schemas}
        registered = {contract.name for contract in registry.domain(domain)}
        if registered != names:
            failures.append(
                f"{domain}: registry mismatch missing={sorted(names - registered)} "
                f"extra={sorted(registered - names)}"
            )
        if {
            contract.name
            for contract in domain_contract_registry(domain).domain(domain)
        } != names:
            failures.append(f"{domain}: catalog registry does not match public schema")

        semantics = build_tool_semantics(domain, tool_schemas)
        if set(semantics) != names:
            failures.append(f"{domain}: execution semantics coverage mismatch")

        for schema in tool_schemas:
            name = str(schema["name"])
            annotations = schema.get("annotations") or {}
            schema_mutating = bool(annotations.get("mutating")) and not bool(
                annotations.get("readonly")
            )
            mutating_count += int(schema_mutating)
            if is_mutating_tool(name, domain) != schema_mutating:
                failures.append(
                    f"{domain}.{name}: schema/ToolSemantics mutation mismatch"
                )

            contract = registry.get(domain, name)
            required = frozenset(
                schema.get("input_schema", {}).get("required", []) or []
            )
            if contract.required_arguments != required:
                failures.append(
                    f"{domain}.{name}: required arguments "
                    f"{contract.required_arguments!r} != schema {required!r}"
                )
            properties = set(
                (schema.get("input_schema", {}).get("properties") or {}).keys()
            )
            for binding in contract.input_entities:
                if binding.name not in properties:
                    failures.append(
                        f"{domain}.{name}: entity binding {binding.name!r} "
                        "is absent from input schema"
                    )
            for group in contract.precondition_groups:
                for predicate in group:
                    subject = predicate.subject
                    if (
                        subject.source == "argument"
                        and subject.name not in properties
                    ):
                        failures.append(
                            f"{domain}.{name}: predicate argument "
                            f"{subject.name!r} is absent from input schema"
                        )

    dependency_cases = [
        (
            "calendar grounded update",
            [call("update_event", event_id="evt_001", title="x")],
            "calendar",
            False,
        ),
        (
            "calendar ungrounded update",
            [call("update_event")],
            "calendar",
            True,
        ),
        (
            "calendar creator",
            [call("create_event", title="x")],
            "calendar",
            False,
        ),
        (
            "shopping grounded chain",
            [
                call("get_product", product_id="prod_001"),
                call("add_to_cart", product_id="prod_001", quantity=1),
                call("checkout"),
            ],
            "shopping",
            False,
        ),
    ]
    for label, calls, domain, expected in dependency_cases:
        actual = detect_missing_dependency(calls, domain)
        if actual != expected:
            failures.append(
                f"{label}: missing_dependency={actual}, expected={expected}"
            )

    if not detect_duplicate_side_effect(
        [call("delete_event", event_id="evt_001"), call("create_event")],
        "calendar",
    ):
        failures.append("calendar delete/create shortcut was not detected")

    scenario_cases = [
        (
            "clarification",
            [],
            [],
            "ask_clarification",
            "clarification_required",
        ),
        (
            "recovery",
            [call("get_product", product_id="prod_001")],
            [{"tool_name": "get_product", "success": False}],
            "final_answer",
            "tool_error_recovery",
        ),
        (
            "normal",
            [call("get_product", product_id="prod_001")],
            [{"tool_name": "get_product", "success": True}],
            "final_answer",
            "normal_safe_success",
        ),
    ]
    for label, calls, history, terminal, expected in scenario_cases:
        actual = classify_scenario("shopping", calls, history, terminal)
        if actual != expected:
            failures.append(f"{label}: scenario={actual}, expected={expected}")

    print(f"domains={len(schemas)} tools={total_tools} mutating={mutating_count}")
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        print(f"failures={len(failures)}")
        return 1
    print("failures=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
