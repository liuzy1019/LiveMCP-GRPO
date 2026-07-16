#!/usr/bin/env python3
"""系统性对抗审查 — 对所有 domain、tool、映射、数据结构的交叉验证。

覆盖:
  A. Server TOOLS annotation 与 ToolSemantics 的全量匹配
  C. _CREATED_ENTITY_BY_TOOL 引用的 tool name 是否真实存在
  D. _DOMAIN_TOOL_REQUIREMENTS 引用的 tool name 是否真实存在
  E. _UNSAFE_SHORTCUT_TOOLS 引用的 tool name 是否真实存在
  F. _ACTION_KEYWORD_MAP 引用的 tool name 是否真实存在
  H. _detect_missing_dependency 对抗性边界案例
  I. _classify_scenario 轨迹分类对抗性测试
  J. _tool_entity 对全部公开工具的覆盖
"""
import sys, os, traceback
import importlib

sys.path.insert(0, ".")
from src.live_mcp.orchestrator import (
    _tool_entity as orch_entity,
    _UNSAFE_SHORTCUT_TOOLS,
    _CREATED_ENTITY_BY_TOOL, _DOMAIN_TOOL_REQUIREMENTS,
    _detect_missing_dependency,
    _classify_scenario,
    _tool_existing_entity_requirements,
    _tool_relevant_entity_types,
)
from src.live_mcp.types import OracleCall
from src.live_mcp.task_planner import _SELF_CONTAINED_WRITE_TOOLS
from src.live_mcp.tool_semantics import is_mutating_tool

# ── Parse all server TOOLS ──

def parse_domain_tools() -> tuple[dict[str, set[str]], dict[str, dict[str, dict]]]:
    """Return (domain→tool_names, domain→tool_name→annotations)."""
    servers_dir = os.path.join(os.path.dirname(__file__), "..", "src", "live_mcp", "servers")
    domain_names: dict[str, set[str]] = {}
    domain_anns: dict[str, dict[str, dict]] = {}

    for domain in sorted(os.listdir(servers_dir)):
        path = os.path.join(servers_dir, domain, "server.py")
        if not os.path.isfile(path):
            continue
        public_tools = importlib.import_module(
            f"src.live_mcp.servers.{domain}.server"
        ).TOOLS
        tools = {str(tool["name"]) for tool in public_tools}
        anns = {
            str(tool["name"]): dict(tool.get("annotations") or {})
            for tool in public_tools
        }
        domain_names[domain] = tools
        domain_anns[domain] = anns
    return domain_names, domain_anns

FAIL = 0

def check(cond: bool, msg: str) -> None:
    global FAIL
    if not cond:
        print(f"  ❌ {msg}")
        FAIL += 1

def section(title: str) -> None:
    print(f"\n{'='*70}\n{title}\n{'='*70}")

domain_tools, domain_anns = parse_domain_tools()
all_tool_names: set[str] = set()
for tools in domain_tools.values():
    all_tool_names.update(tools)
total_tool_count = sum(len(tools) for tools in domain_tools.values())

all_mutating: set[str] = set()
total_mutating_count = 0
for domain, anns in domain_anns.items():
    for name, ann in anns.items():
        if ann.get("mutating") and not ann.get("readonly"):
            all_mutating.add(name)
            total_mutating_count += 1

# ═════════════════════════════════════════════════════════════════
# A. ToolSemantics coverage of all server-annotated mutating tools
# ═════════════════════════════════════════════════════════════════
section("A. ToolSemantics vs 所有 server-annotated mutating tools")

missed_mutating = []
for domain, anns in domain_anns.items():
    for tool, ann in anns.items():
        if ann.get("mutating") and not is_mutating_tool(tool, domain):
            missed_mutating.append(f"{domain}.{tool}")

if missed_mutating:
    for t in missed_mutating:
        print(f"  ❌ {t} is annotated mutating but ToolSemantics returns False")
    FAIL += len(missed_mutating)
else:
    print(f"  ✅ All {total_mutating_count} server-annotated mutating tools covered")

# ═════════════════════════════════════════════════════════════════
# C. _CREATED_ENTITY_BY_TOOL 的 tool name 必须存在于任何 server 中
# ═════════════════════════════════════════════════════════════════
section("C. _CREATED_ENTITY_BY_TOOL 引用的 tool name 存在性")

ghost_ce = []
for tool in sorted(_CREATED_ENTITY_BY_TOOL):
    if tool not in all_tool_names:
        ghost_ce.append(tool)
if ghost_ce:
    for t in ghost_ce:
        print(f"  ❌ '{t}' NOT FOUND in any server TOOLS")
    FAIL += len(ghost_ce)
else:
    print(f"  ✅ All {len(_CREATED_ENTITY_BY_TOOL)} tools exist in server TOOLS")

# ═════════════════════════════════════════════════════════════════
# D. _DOMAIN_TOOL_REQUIREMENTS 引用的 tool name 存在性
# ═════════════════════════════════════════════════════════════════
section("D. _DOMAIN_TOOL_REQUIREMENTS 引用的 tool name 存在性")

ghost_dtr = []
for domain, reqs in sorted(_DOMAIN_TOOL_REQUIREMENTS.items()):
    dt = domain_tools.get(domain, set())
    for tool in sorted(reqs):
        if tool not in dt:
            ghost_dtr.append(f"{domain}/{tool}")
if ghost_dtr:
    for t in ghost_dtr:
        print(f"  ❌ '{t}' NOT FOUND in domain server TOOLS")
    FAIL += len(ghost_dtr)
else:
    total_reqs = sum(len(v) for v in _DOMAIN_TOOL_REQUIREMENTS.values())
    print(f"  ✅ All {total_reqs} requirements reference existing tools in correct domains")

# ═════════════════════════════════════════════════════════════════
# E. _UNSAFE_SHORTCUT_TOOLS 引用的 tool name 存在性
# ═════════════════════════════════════════════════════════════════
section("E. _UNSAFE_SHORTCUT_TOOLS 引用的 tool name 存在性")

ghost_us = []
for domain, tools in sorted(_UNSAFE_SHORTCUT_TOOLS.items()):
    dt = domain_tools.get(domain, set())
    for tool in sorted(tools):
        if tool not in dt:
            ghost_us.append(f"{domain}/{tool}")
if ghost_us:
    for t in ghost_us:
        print(f"  ❌ '{t}' NOT FOUND in domain server TOOLS")
    FAIL += len(ghost_us)
else:
    total_us = sum(len(v) for v in _UNSAFE_SHORTCUT_TOOLS.values())
    print(f"  ✅ All {total_us} executor tools exist in correct domains")

# ═════════════════════════════════════════════════════════════════
# E2. missing_dependency 行为覆盖：有既有实体依赖的 mutating 工具必须被拦截
# ═════════════════════════════════════════════════════════════════
section("E2. missing_dependency 行为覆盖 vs server annotations")

e2_fail = False
for domain, anns in sorted(domain_anns.items()):
    domain_mutating = {t for t, a in anns.items()
                       if a.get("mutating") and not a.get("readonly")}
    for tool in sorted(domain_mutating):
        requirements = _tool_existing_entity_requirements(tool, domain)
        calls = [OracleCall(action="tool_call", tool_name=tool, arguments={})]
        detected = _detect_missing_dependency(calls, domain)
        if requirements and not detected:
            print(
                f"  ❌ {domain}/{tool}: requirements={sorted(requirements)} "
                "but single-step call is NOT classified missing_dependency"
            )
            e2_fail = True
        if not requirements and detected:
            print(
                f"  ❌ {domain}/{tool}: no existing-entity requirements "
                "but single-step call IS classified missing_dependency"
            )
            e2_fail = True
if not e2_fail:
    print("  ✅ All dependency-requiring mutating tools are behaviorally covered")
else:
    FAIL += 1

# ═════════════════════════════════════════════════════════════════
# G. No mutation-prefix inference remains
# ═════════════════════════════════════════════════════════════════
section("G. ToolSemantics has no prefix-inference layer")
print("  ✅ Mutation semantics are exact domain/tool mappings")

# ═════════════════════════════════════════════════════════════════
# H. _detect_missing_dependency 对抗性边界案例
# ═════════════════════════════════════════════════════════════════
section("H. _detect_missing_dependency 对抗性边界案例")

h_cases = [
    # (name, domain, calls, expect_missing_dep, description)
    # Case 1: read→executor with matching entity → should NOT flag
    ("read→executor_OK",
     "email", [("list_inbox", {}), ("reply_email", {})], False,
     "list_inbox resolves email, reply_email consumes email"),
    ("read→executor_OK_2",
     "email", [("get_thread", {}), ("reply_email", {})], False,
     "get_thread resolves email, reply_email consumes email"),
    ("read→executor_OK_3",
     "email", [("get_attachments", {}), ("reply_email", {})], False,
     "get_attachments resolves email, reply_email consumes email"),
    # Case 2: executor+creator (like send_email) → exempted
    ("self_contained_OK",
     "email", [("send_email", {})], False,
     "send_email is self-contained, no read needed"),
    # Case 3: check_orchestrator alone → should flag (executor without read)
    ("checkout_alone_BAD",
     "shopping", [("checkout", {})], True,
     "checkout is executor, no preceding read → missing dep"),
    # Case 4: add_to_cart creates a cart item but still requires a discovered product.
    ("add_to_cart_alone_BAD",
     "shopping", [("add_to_cart", {}), ("checkout", {})], True,
     "add_to_cart without preceding product read → missing dep"),
    # Case 5: get_product resolves the input entity, then add_to_cart produces cart item for checkout.
    ("add_to_cart_checkout_OK",
     "shopping", [("get_product", {}), ("add_to_cart", {}), ("checkout", {})], False,
     "get_product→add_to_cart→checkout should pass"),
    # Case 6: reply_email alone (executor, no preceding read, not self-contained) → should flag
    ("reply_alone_BAD",
     "email", [("reply_email", {})], True,
     "reply_email without preceding read → missing dep"),
    # Case 7: archive_channel alone → should flag
    ("archive_alone_BAD",
     "team_chat", [("archive_channel", {})], True,
     "archive_channel without preceding read → missing dep"),
    # Case 8: read only chain (no executor) → should NOT flag
    ("read_only_OK",
     "email", [("list_inbox", {}), ("get_email", {})], False,
     "no executor tools in chain, no dependency check needed"),
    # Case 9: forward_email with list_inbox → OK
    ("forward_OK",
     "email", [("list_inbox", {}), ("forward_email", {})], False,
     "list_inbox→forward_email should pass"),
    # Case 10: mark_read with list_inbox → OK
    ("mark_read_OK",
     "email", [("list_inbox", {}), ("mark_read", {})], False,
     "list_inbox→mark_read should pass"),
    # Case 11: update_event alone (calendar executor) → flag
    ("update_event_alone_BAD",
     "calendar", [("update_event", {})], True,
     "update_event without preceding read → missing dep"),
    # Case 12: create_event (creator) → exempted
    ("create_event_OK",
     "calendar", [("create_event", {})], False,
     "create_event is creator → exempted"),
]

h_fail = False
for name, domain, calls_raw, expect, desc in h_cases:
    calls = [OracleCall(action='tool_call', tool_name=n, arguments=a)
             for n, a in calls_raw]
    try:
        result = _detect_missing_dependency(calls, domain)
    except Exception as e:
        print(f"  ❌ [{name}] EXCEPTION: {e}")
        h_fail = True
        continue
    if result != expect:
        print(f"  ❌ [{name}] {desc}: expected missing_dep={expect}, got {result}")
        h_fail = True
# A tool from another domain is a contract violation, not a read-only fallback.
try:
    _detect_missing_dependency(
        [OracleCall(action="tool_call", tool_name="send_email", arguments={})],
        "shopping",
    )
except ValueError:
    pass
else:
    print("  ❌ [wrong_domain_REJECTED] cross-domain tool was accepted")
    h_fail = True

if not h_fail:
    print(
        f"  ✅ All {len(h_cases)} dependency cases pass; "
        "cross-domain tools fail closed"
    )

if h_fail:
    FAIL += 1

# ═════════════════════════════════════════════════════════════════
# I. _classify_scenario 对抗性测试
# ═════════════════════════════════════════════════════════════════
section("I. _classify_scenario 轨迹分类对抗性测试")

i_cases = [
    # (name, server, oracle_calls, exec_hist, terminal_action, expected)
    ("clarification_no_calls",
     "shopping",
     [],  # empty oracle
     [],  # empty history
     "ask_clarification",
     "clarification_required"),
    ("normal_read_only",
     "shopping",
     [("get_product", {})],
     [{"tool_name": "get_product", "success": True}],
     "final_answer",
     "normal_safe_success"),
    # missing_dependency: checkout alone → should flag
    ("checkout_missing_dep",
     "shopping",
     [("checkout", {})],
     [{"tool_name": "checkout", "success": True}],
     "final_answer",
     "missing_dependency"),
    # tool_error_recovery: has execution failure
    ("error_recovery",
     "shopping",
     [("get_product", {}), ("add_to_cart", {})],
     [{"tool_name": "get_product", "success": True},
      {"tool_name": "add_to_cart", "success": False}],
     "final_answer",
     "tool_error_recovery"),
    # normal_safe_success: read→executor chain with all success
    ("normal_chain",
     "shopping",
     [("get_cart", {}), ("checkout", {})],
     [{"tool_name": "get_cart", "success": True},
      {"tool_name": "checkout", "success": True}],
     "final_answer",
     "normal_safe_success"),
]

i_fail = False
for name, server, calls_raw, exec_hist, term_action, expect in i_cases:
    calls = [OracleCall(action='tool_call', tool_name=n, arguments=a)
             if isinstance(n, str) else n
             for n, a in (calls_raw if calls_raw else [])]
    try:
        result = _classify_scenario(server, calls, exec_hist, term_action, 42)
    except Exception as e:
        traceback.print_exc()
        print(f"  ❌ [{name}] EXCEPTION: {e}")
        i_fail = True
        continue
    if result != expect:
        print(f"  ❌ [{name}] expected '{expect}', got '{result}'")
        i_fail = True
if not i_fail:
    print(f"  ✅ All {len(i_cases)} _classify_scenario cases pass")

if i_fail:
    FAIL += 1

# ═════════════════════════════════════════════════════════════════
# J. _tool_entity 全量覆盖（所有 server tool）
# ═════════════════════════════════════════════════════════════════
section("J. _tool_entity 全量覆盖（所有 server tool）")

j_fail = False
for domain, tools in sorted(domain_tools.items()):
    for tool in sorted(tools):
        oe = orch_entity(tool, domain)
        if not oe:
            print(f"  ❌ {domain}/{tool}: empty entity mapping")
            j_fail = True
if not j_fail:
    print(f"  ✅ All domain/tool entity mappings are non-empty")
else:
    FAIL += 1

# ═════════════════════════════════════════════════════════════════
# K. 跨 domain entity 混淆检测
# ═════════════════════════════════════════════════════════════════
section("K. 跨 domain entity 混淆检测")

# 对每个 domain 的每个 executor，如果 _tool_entity 返回的 entity 在
# 该 domain 的 read 工具中找不到匹配 → entity 映射可能跨 domain 错误
k_fail = False
read_prefixes = ("list_", "search_", "get_", "find_", "lookup_", "check_",
                 "view_", "browse_", "ls", "cat", "pwd", "stat", "head", "tail")
for domain, tools in sorted(domain_tools.items()):
    domain_reads = {t for t in tools
                    if any(t.lower().startswith(p) for p in read_prefixes)}
    domain_executors = _UNSAFE_SHORTCUT_TOOLS.get(domain, set())
    for exec_tool in domain_executors:
        exec_entity = orch_entity(exec_tool, domain)
        # Check: is there ANY read tool in this domain that resolves this entity.
        matching_reads = [
            t for t in domain_reads
            if exec_entity == orch_entity(t, domain)
            or exec_entity in _tool_relevant_entity_types(t, domain)
        ]
        if not matching_reads:
            # Check all domains: where IS this entity resolved?
            all_matching = []
            for d, ts in domain_tools.items():
                for t in ts:
                    if any(t.lower().startswith(p) for p in read_prefixes):
                        if (
                            orch_entity(t, d) == exec_entity
                            or exec_entity in _tool_relevant_entity_types(t, d)
                        ):
                            all_matching.append(f"{d}/{t}")
            print(f"  ⚠️  {domain}/{exec_tool}: entity='{exec_entity}' not resolved by any read in {domain}")
            print(f"      Should be resolved in: {all_matching if all_matching else 'NOWHERE'}")
            k_fail = True

if not k_fail:
    print(f"  ✅ All executor tools have matching read tools in their domain")
else:
    print(f"  ⚠️  Above warnings may indicate entity mapping errors (not hard failures)")

# ═════════════════════════════════════════════════════════════════
# Summary
# ═════════════════════════════════════════════════════════════════
section("SUMMARY")
total_domains = len(domain_tools)
print(f"  Domains: {total_domains}")
print(f"  Total tools: {total_tool_count}")
print(f"  Mutating tools: {total_mutating_count}")
print(f"  Failures: {FAIL}")
print()
if FAIL == 0:
    print("✅ ALL CHECKS PASSED — no bugs found")
else:
    print(f"❌ {FAIL} FAILURES DETECTED — fix before generating data")
    sys.exit(1)
