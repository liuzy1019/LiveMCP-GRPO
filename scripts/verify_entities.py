#!/usr/bin/env python3
"""对抗性审查脚本 — 验证 livemcp 所有 entity mapping 和关键逻辑"""
import sys
sys.path.insert(0, ".")
from src.live_mcp.orchestrator import (
    _tool_entity, _UNSAFE_SHORTCUT_TOOLS, _TOOL_ENTITY_OVERRIDE,
    _detect_missing_dependency, _validate_query_chain,
)
from src.live_mcp.types import OracleCall
from src.live_mcp.task_planner import _is_mutating_tool

# 1. Entity override 覆盖所有 executor tools
print("=" * 60)
print("1. Entity override coverage for _UNSAFE_SHORTCUT_TOOLS")
print("=" * 60)
for domain, tools in sorted(_UNSAFE_SHORTCUT_TOOLS.items()):
    for tool in sorted(tools):
        entity = _tool_entity(tool)
        print(f"  {domain:15s} {tool:25s} → entity='{entity}'")

# 2. 关键配对验证
print()
print("=" * 60)
print("2. Key pair verification (should all be False = no missing dep)")
print("=" * 60)
test_pairs = [
    ("shopping", "get_cart → checkout",
     [("get_cart", {}), ("checkout", {})]),
    ("shopping", "get_cart → clear_cart",
     [("get_cart", {}), ("clear_cart", {})]),
    ("shopping", "search_products → add_to_cart → checkout",
     [("search_products", {}), ("add_to_cart", {}), ("checkout", {})]),
    ("banking", "get_balance → transfer",
     [("get_balance", {}), ("transfer", {})]),
    ("banking", "get_balance → wire_transfer",
     [("get_balance", {}), ("wire_transfer", {})]),
    ("banking", "get_balance → bill_pay",
     [("get_balance", {}), ("bill_pay", {})]),
    ("payments", "get_invoice → pay_invoice",
     [("get_invoice", {}), ("pay_invoice", {})]),
    ("payments", "get_invoice → refund_invoice",
     [("get_invoice", {}), ("refund_invoice", {})]),
    ("payments", "get_payment → cancel_payment",
     [("get_payment", {}), ("cancel_payment", {})]),
    ("calendar", "list_events → update_event",
     [("list_events", {}), ("update_event", {})]),
    ("email", "get_email → move_to_thread",
     [("get_email", {}), ("move_to_thread", {})]),
    ("crm", "get_lead → convert_lead",
     [("get_lead", {}), ("convert_lead", {})]),
    ("issue_tracker", "get_issue → assign_issue",
     [("get_issue", {}), ("assign_issue", {})]),
]
all_ok = True
for domain, desc, calls_plain in test_pairs:
    calls = [OracleCall(action='tool_call', tool_name=n, arguments=a) 
             for n, a in calls_plain]
    result = _detect_missing_dependency(calls, domain)
    status = "OK" if not result else "FAIL"
    if result:
        all_ok = False
    print(f"  [{status}] {domain}: {desc} → missing_dep={result}")

# 3. _validate_query_chain
print()
print("=" * 60)
print("3. _validate_query_chain verification")
print("=" * 60)
qc_tests = [
    ("check order status", ["get_order", "checkout"], False),
    ("buy items in my cart please", ["add_to_cart", "checkout"], True),
    ("Please place the order after checking my cart", ["get_cart", "checkout"], True),
    ("Can you check my cart before ordering?", ["get_cart", "checkout"], True),
    ("move file to tmp", ["stat", "mv"], True),
    ("just list the files", ["ls"], True),
]
qc_ok = True
for query, chain, expect in qc_tests:
    ok, _ = _validate_query_chain(query, chain)
    status = "OK" if ok == expect else "FAIL"
    if ok != expect:
        qc_ok = False
    print(f"  [{status}] '{query[:55]}' → {ok} (expect {expect})")

# 4. _is_mutating_tool 覆盖
print()
print("=" * 60)
print("4. _is_mutating_tool coverage")
print("=" * 60)
key_tools = [
    "move_to_thread", "add_to_wishlist", "mark_read", "assign_issue",
    "transition_issue", "bill_pay", "wire_transfer", "clear_cart",
    "rate_order", "return_order", "reorder", "apply_coupon",
    "checkout", "transfer", "create_event", "update_event",
]
mt_ok = True
for tool in key_tools:
    result = _is_mutating_tool(tool)
    status = "OK" if result else "FAIL"
    if not result:
        mt_ok = False
    print(f"  [{status}] {tool} → {result}")

# 5. 边界验证：单 tool checkout 无前置读 → 应为 True
print()
print("=" * 60)
print("5. Edge cases")
print("=" * 60)
calls_single = [OracleCall(action='tool_call', tool_name='checkout', arguments={})]
r = _detect_missing_dependency(calls_single, 'shopping')
print(f"  [{'OK' if r else 'FAIL'}] checkout alone → {r} (expect True)")

calls_order_only = [OracleCall(action='tool_call', tool_name='get_order', arguments={})]
r = _detect_missing_dependency(calls_order_only, 'shopping')
print(f"  [{'OK' if not r else 'FAIL'}] get_order alone (read-only, not executor) → {r} (expect False)")

print()
if all_ok and qc_ok and mt_ok:
    print("✅ ALL VERIFICATIONS PASSED")
else:
    print("❌ SOME VERIFICATIONS FAILED")
    sys.exit(1)
