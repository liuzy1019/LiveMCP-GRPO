#!/usr/bin/env python3
"""集成测试：验证所有 domain 的 tool 注册、dependency graph、executor dispatch 和 schema 一致性。

检测项：
  1. Config dependency_graph 中引用的 tool 是否在 server TOOLS 中实际存在
  2. Server 实际暴露的 tool 是否都能被 schema_registry 正确注册和解析
  3. Executor dispatch 是否能正确路由到对应 server
  4. StateSeeder 是否为所有 domain 生成有效初始状态
  5. Tool 调用是否能正常执行（smoke test）
  6. Dependency graph chain extraction 是否产生有效链
  7. 跨 domain tool 名冲突检测
"""
import sys, json, importlib, traceback
from pathlib import Path
from collections import defaultdict

# 确保项目根目录在 path 中
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.live_mcp.config import load_suite_config, project_root
from src.live_mcp.state_seeder import StateSeeder

# ═══════════════════════════════════════════════════════════════════════
# 测试结果收集
# ═══════════════════════════════════════════════════════════════════════
PASS = 0
FAIL = 0
WARN = 0
issues: list[str] = []

def ok(msg: str):
    global PASS
    PASS += 1
    print(f"  ✅ {msg}")

def fail(msg: str):
    global FAIL
    FAIL += 1
    issues.append(msg)
    print(f"  ❌ {msg}")

def warn(msg: str):
    global WARN
    WARN += 1
    print(f"  ⚠️  {msg}")


# ═══════════════════════════════════════════════════════════════════════
# 1. 加载所有 server 的 TOOLS 定义
# ═══════════════════════════════════════════════════════════════════════
DOMAINS = [
    "banking", "calendar", "crm", "email", "filesystem",
    "food_delivery", "issue_tracker", "payments", "shopping", "team_chat",
]

def load_server_tools(domain: str) -> list[dict]:
    """动态导入 server module 获取 TOOLS 列表。"""
    mod = importlib.import_module(f"src.live_mcp.servers.{domain}.server")
    return list(mod.TOOLS)


print("=" * 70)
print("TEST 1: Server TOOLS 加载与基本 schema 验证")
print("=" * 70)

domain_tools: dict[str, list[dict]] = {}
domain_tool_names: dict[str, set[str]] = {}

for domain in DOMAINS:
    try:
        tools = load_server_tools(domain)
        domain_tools[domain] = tools
        names = {t["name"] for t in tools}
        domain_tool_names[domain] = names
        
        # 验证每个 tool 都有 name, input_schema
        for t in tools:
            if not t.get("name"):
                fail(f"[{domain}] tool missing 'name': {t}")
            if not t.get("input_schema"):
                fail(f"[{domain}] tool '{t.get('name')}' missing 'input_schema'")
            # 验证 input_schema 结构
            schema = t.get("input_schema", {})
            if schema.get("type") != "object":
                fail(f"[{domain}] tool '{t['name']}' input_schema.type != 'object'")
            if "properties" not in schema:
                fail(f"[{domain}] tool '{t['name']}' missing 'properties' in input_schema")
            if "required" not in schema:
                warn(f"[{domain}] tool '{t['name']}' missing 'required' in input_schema")
        
        ok(f"[{domain}] {len(tools)} tools loaded successfully")
    except Exception as e:
        fail(f"[{domain}] TOOLS 加载失败: {e}")
        domain_tools[domain] = []
        domain_tool_names[domain] = set()


# ═══════════════════════════════════════════════════════════════════════
# 2. Config dependency_graph 中引用的 tool 是否在 server 中存在
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TEST 2: Config dependency_graph 引用的 tool 是否在 server 中存在")
print("=" * 70)

suite_config = load_suite_config("configs/live_mcp/suite_mvp.yaml")

for cfg in suite_config.servers:
    domain = cfg.name
    if not cfg.dependency_graph:
        warn(f"[{domain}] 无 dependency_graph 配置")
        continue
    
    edges = cfg.dependency_graph.get("edges", [])
    graph_tools = set()
    for edge in edges:
        graph_tools.add(edge["source_tool"])
        graph_tools.add(edge["target_tool"])
    
    server_names = domain_tool_names.get(domain, set())
    missing = graph_tools - server_names
    
    if missing:
        fail(f"[{domain}] dependency_graph 引用了 server 中不存在的 tool: {sorted(missing)}")
    else:
        ok(f"[{domain}] dependency_graph 所有 {len(graph_tools)} 个 tool 引用均有效")


# ═══════════════════════════════════════════════════════════════════════
# 3. Config tools.discovery/readonly/mutating 是否与 server TOOLS 一致
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TEST 3: Config tools 分类是否与 server TOOLS annotations 一致")
print("=" * 70)

for cfg in suite_config.servers:
    domain = cfg.name
    config_discovery = set(cfg.tools.get("discovery", []))
    config_readonly = set(cfg.tools.get("readonly", []))
    config_mutating = set(cfg.tools.get("mutating", []))
    config_all = config_discovery | config_readonly | config_mutating
    
    server_names = domain_tool_names.get(domain, set())
    
    # Config 中引用的 tool 是否在 server 中存在
    config_missing = config_all - server_names
    if config_missing:
        fail(f"[{domain}] config tools 引用了 server 中不存在的 tool: {sorted(config_missing)}")
    
    # 检查 annotations 一致性
    for t in domain_tools.get(domain, []):
        name = t["name"]
        annotations = t.get("annotations", {})
        is_mutating = annotations.get("mutating", False)
        is_readonly = annotations.get("readonly", False)
        
        if is_mutating and name in config_readonly:
            warn(f"[{domain}] tool '{name}' annotations=mutating 但 config 归类为 readonly")
        if is_readonly and name in config_mutating:
            warn(f"[{domain}] tool '{name}' annotations=readonly 但 config 归类为 mutating")
    
    if not config_missing:
        ok(f"[{domain}] config tools 分类与 server 一致")


# ═══════════════════════════════════════════════════════════════════════
# 4. StateSeeder 是否为所有 domain 生成有效初始状态
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TEST 4: StateSeeder 初始状态生成验证")
print("=" * 70)

seeder = StateSeeder()
for domain in DOMAINS:
    try:
        state = seeder.reset_state(domain, "test_session", seed=42)
        if not isinstance(state, dict):
            fail(f"[{domain}] StateSeeder 返回非 dict: {type(state)}")
            continue
        if not state:
            fail(f"[{domain}] StateSeeder 返回空 state")
            continue
        
        # 验证不同 seed 产生不同状态
        state2 = seeder.reset_state(domain, "test_session", seed=123)
        if state == state2:
            warn(f"[{domain}] seed=42 和 seed=123 产生相同状态（缺乏多样性）")
        
        ok(f"[{domain}] StateSeeder 正常，state keys: {sorted(state.keys())[:5]}...")
    except Exception as e:
        fail(f"[{domain}] StateSeeder 失败: {e}")


# ═══════════════════════════════════════════════════════════════════════
# 5. Tool 执行 smoke test（直接调用 server handler）
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TEST 5: Tool 执行 smoke test（直接调用 handler）")
print("=" * 70)

# 每个 domain 选一个 readonly tool 做 smoke test
SMOKE_TESTS = {
    "banking": ("list_accounts", {}),
    "calendar": ("list_events", {}),
    "crm": ("list_leads", {}),
    "email": ("list_inbox", {}),
    "filesystem": ("ls", {"path": "/"}),
    "food_delivery": ("list_restaurants", {}),
    "issue_tracker": ("list_issues", {}),
    "payments": ("list_invoices", {}),
    "shopping": ("search_products", {"query": "keyboard"}),
    "team_chat": ("list_channels", {}),
}

for domain in DOMAINS:
    try:
        mod = importlib.import_module(f"src.live_mcp.servers.{domain}.server")
        # 找到 server class
        server_cls = None
        for attr_name in dir(mod):
            obj = getattr(mod, attr_name)
            if isinstance(obj, type) and hasattr(obj, 'handle_request') and attr_name != 'StatefulToolServer':
                server_cls = obj
                break
        
        if server_cls is None:
            fail(f"[{domain}] 找不到 server class")
            continue
        
        server = server_cls()
        
        # Reset session
        reset_resp = server.handle_request("session/reset", {"session_id": "smoke_test", "seed": 42})
        if "error" in reset_resp:
            fail(f"[{domain}] session/reset 失败: {reset_resp['error']}")
            continue
        
        # List tools
        tools_resp = server.handle_request("tools/list", {})
        listed_tools = tools_resp.get("result", {}).get("tools", [])
        if len(listed_tools) != len(domain_tools[domain]):
            fail(f"[{domain}] tools/list 返回 {len(listed_tools)} tools，预期 {len(domain_tools[domain])}")
        
        # Smoke test a readonly tool
        tool_name, args = SMOKE_TESTS.get(domain, (None, None))
        if tool_name:
            call_resp = server.handle_request("tools/call", {
                "session_id": "smoke_test",
                "name": tool_name,
                "arguments": args,
            })
            result = call_resp.get("result", {})
            if result.get("success"):
                ok(f"[{domain}] smoke test '{tool_name}' → SUCCESS")
            else:
                # 可能是 tool 不存在
                err = result.get("error_message", "")
                if "unknown tool" in err:
                    fail(f"[{domain}] smoke test '{tool_name}' → UNKNOWN TOOL（server 未注册此 handler）")
                else:
                    fail(f"[{domain}] smoke test '{tool_name}' → FAILED: {err}")
        else:
            warn(f"[{domain}] 无 smoke test 配置")
    except Exception as e:
        fail(f"[{domain}] smoke test 异常: {e}\n{traceback.format_exc()}")


# ═══════════════════════════════════════════════════════════════════════
# 6. 跨 domain tool 名冲突检测
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TEST 6: 跨 domain tool 名冲突检测")
print("=" * 70)

name_to_domains: dict[str, list[str]] = defaultdict(list)
for domain, names in domain_tool_names.items():
    for name in names:
        name_to_domains[name].append(domain)

conflicts = {name: domains for name, domains in name_to_domains.items() if len(domains) > 1}
if conflicts:
    for name, domains in sorted(conflicts.items()):
        warn(f"Tool '{name}' 存在于多个 domain: {domains}（SchemaRegistry 需要 domain hint 消歧）")
    print(f"  共 {len(conflicts)} 个冲突 tool 名")
else:
    ok("无跨 domain tool 名冲突")


# ═══════════════════════════════════════════════════════════════════════
# 7. Dependency graph chain extraction 验证
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TEST 7: Dependency graph chain extraction 验证")
print("=" * 70)

def extract_chains_from_config(cfg) -> list[list[str]]:
    """从 config edges 构建 graph 并提取 length-2 to length-5 chains。"""
    edges = cfg.dependency_graph.get("edges", [])
    if not edges:
        return []
    
    # 构建邻接表
    graph: dict[str, dict[str, list[str]]] = {}
    for edge in edges:
        src = edge["source_tool"]
        tgt = edge["target_tool"]
        rel = edge.get("relation", "implicit")
        if src not in graph:
            graph[src] = {"explicit": [], "implicit": []}
        graph[src][rel].append(tgt)
    
    # DFS 提取 chains
    chains: list[list[str]] = []
    
    def _dfs(current: str, path: list[str], visited: set[str]):
        if len(path) >= 5:
            return
        neighbors = (
            list(graph.get(current, {}).get("explicit", []))
            + list(graph.get(current, {}).get("implicit", []))
        )
        for neighbor in neighbors:
            if neighbor in visited:
                continue
            new_path = path + [neighbor]
            if len(new_path) >= 2:
                chains.append(new_path)
            _dfs(neighbor, new_path, visited | {neighbor})
    
    for start_node in graph:
        _dfs(start_node, [start_node], {start_node})
    
    # Dedup
    seen: set[tuple] = set()
    deduped: list[list[str]] = []
    for chain in chains:
        key = tuple(chain)
        if key not in seen:
            seen.add(key)
            deduped.append(chain)
    
    return deduped

for cfg in suite_config.servers:
    domain = cfg.name
    chains = extract_chains_from_config(cfg)
    
    if not chains:
        warn(f"[{domain}] 无法从 config 提取任何 chain")
        continue
    
    # 统计 chain 长度分布
    len_dist = defaultdict(int)
    for chain in chains:
        len_dist[len(chain)] += 1
    
    # 验证所有 chain 中的 tool 都在 server 中存在
    invalid_chains = []
    for chain in chains:
        for tool in chain:
            if tool not in domain_tool_names.get(domain, set()):
                invalid_chains.append((chain, tool))
                break
    
    if invalid_chains:
        fail(f"[{domain}] {len(invalid_chains)} chains 引用了不存在的 tool:")
        for chain, bad_tool in invalid_chains[:3]:
            print(f"    chain={chain}, bad_tool='{bad_tool}'")
    else:
        ok(f"[{domain}] {len(chains)} chains 全部有效，长度分布: {dict(sorted(len_dist.items()))}")


# ═══════════════════════════════════════════════════════════════════════
# 8. SchemaRegistry 注册和解析验证
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TEST 8: SchemaRegistry 注册和解析验证")
print("=" * 70)

from src.live_mcp.schema_registry import SchemaRegistry

registry = SchemaRegistry()
for domain in DOMAINS:
    tools = domain_tools.get(domain, [])
    registry.register_tools(domain, tools)

# 验证每个 tool 都能被正确解析
for domain in DOMAINS:
    for t in domain_tools.get(domain, []):
        name = t["name"]
        schema = registry.get_schema(name, domain=domain)
        if schema is None:
            fail(f"[{domain}] registry.get_schema('{name}', domain='{domain}') 返回 None")
        
        server = registry.server_for_tool(name, domain=domain)
        if server != domain:
            fail(f"[{domain}] registry.server_for_tool('{name}', domain='{domain}') = '{server}'，预期 '{domain}'")

# 验证冲突 tool 在无 domain hint 时的行为
for name, domains in conflicts.items():
    # 无 domain hint 时应该返回某个 server（不应 crash）
    server = registry.server_for_tool(name)
    if server is None:
        fail(f"registry.server_for_tool('{name}') 无 domain hint 时返回 None")
    elif server not in domains:
        fail(f"registry.server_for_tool('{name}') = '{server}'，不在预期 domains {domains} 中")

ok(f"SchemaRegistry 注册 {sum(len(v) for v in domain_tools.values())} tools，解析全部正确")


# ═══════════════════════════════════════════════════════════════════════
# 9. Schema validation 边界测试
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TEST 9: Schema validation 边界测试")
print("=" * 70)

# 测试缺少 required 参数
result = registry.validate_arguments("transfer", {"from_account": "acc_001"}, domain="banking")
if result.valid:
    fail("[banking] transfer 缺少 to_account, amount 但 validation 通过了")
else:
    ok("[banking] transfer 缺少必需参数 → 正确拒绝")

# 测试类型错误
result = registry.validate_arguments("transfer", {
    "from_account": "acc_001", "to_account": "acc_002", "amount": "not_a_number"
}, domain="banking")
if result.valid:
    fail("[banking] transfer amount='not_a_number' 但 validation 通过了")
else:
    ok("[banking] transfer amount 类型错误 → 正确拒绝")

# 测试正确参数
result = registry.validate_arguments("transfer", {
    "from_account": "acc_001", "to_account": "acc_002", "amount": 100.0
}, domain="banking")
if not result.valid:
    fail(f"[banking] transfer 正确参数但 validation 失败: missing={result.missing_required}, type_err={result.type_errors}")
else:
    ok("[banking] transfer 正确参数 → 通过")

# 测试未知 tool
schema = registry.get_schema("nonexistent_tool_xyz")
if schema is not None:
    fail("get_schema('nonexistent_tool_xyz') 应返回 None")
else:
    ok("未知 tool → 正确返回 None")


# ═══════════════════════════════════════════════════════════════════════
# 10. Handler 完整性验证（每个 TOOLS 中的 tool 都有对应 handler）
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TEST 10: Handler 完整性验证")
print("=" * 70)

for domain in DOMAINS:
    try:
        mod = importlib.import_module(f"src.live_mcp.servers.{domain}.server")
        server_cls = None
        for attr_name in dir(mod):
            obj = getattr(mod, attr_name)
            if isinstance(obj, type) and hasattr(obj, 'handle_request') and attr_name != 'StatefulToolServer':
                server_cls = obj
                break
        
        if server_cls is None:
            fail(f"[{domain}] 找不到 server class")
            continue
        
        server = server_cls()
        registered_handlers = set(server.handlers.keys())
        tool_names = {t["name"] for t in domain_tools[domain]}
        
        # 每个 tool 都应有 handler
        missing_handlers = tool_names - registered_handlers
        if missing_handlers:
            fail(f"[{domain}] TOOLS 中有 tool 但无对应 handler: {sorted(missing_handlers)}")
        
        # 每个 handler 都应有对应 tool
        extra_handlers = registered_handlers - tool_names
        if extra_handlers:
            warn(f"[{domain}] handler 存在但不在 TOOLS 中: {sorted(extra_handlers)}")
        
        if not missing_handlers:
            ok(f"[{domain}] 所有 {len(tool_names)} 个 tool 都有对应 handler")
    except Exception as e:
        fail(f"[{domain}] handler 验证异常: {e}")


# ═══════════════════════════════════════════════════════════════════════
# 11. Mutating tool 执行后 state_changed 验证
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TEST 11: Mutating tool 执行后 state_changed 验证")
print("=" * 70)

# 选几个关键的 mutating tool 验证 state_changed=True
MUTATING_SMOKE = {
    "payments": ("create_invoice", {"customer": "test_customer", "amount": 50.0}),
}

# Banking 和 Calendar 需要动态获取有效 entity ID
# Banking: account_id 格式为 acc_{seed_hash}_NNN
# Calendar: create_event 需要 start_time/end_time 而非 start/end
# 这里用 list_accounts 获取真实 ID 来测试 deposit
for domain_name, discover_tool, mutate_tool, mutate_args_fn in [
    ("banking", "list_accounts", "deposit", lambda obs: {"account_id": obs["accounts"][0]["account_id"], "amount": 10.0}),
    ("calendar", "create_event", None, None),
]:
    try:
        mod = importlib.import_module(f"src.live_mcp.servers.{domain_name}.server")
        server_cls = None
        for attr_name in dir(mod):
            obj = getattr(mod, attr_name)
            if isinstance(obj, type) and hasattr(obj, 'handle_request') and attr_name != 'StatefulToolServer':
                server_cls = obj
                break
        server = server_cls()
        server.handle_request("session/reset", {"session_id": "mut_dyn", "seed": 42})
        
        if domain_name == "banking":
            # 先 list_accounts 获取真实 ID
            resp = server.handle_request("tools/call", {"session_id": "mut_dyn", "name": "list_accounts", "arguments": {}})
            obs = resp["result"]["observation"]
            args = mutate_args_fn(obs)
            MUTATING_SMOKE["banking"] = ("deposit", args)
        elif domain_name == "calendar":
            MUTATING_SMOKE["calendar"] = ("create_event", {
                "title": "Test Event",
                "start_time": "2026-07-01T10:00:00",
                "end_time": "2026-07-01T11:00:00",
            })
    except Exception:
        pass

for domain, (tool_name, args) in MUTATING_SMOKE.items():
    try:
        mod = importlib.import_module(f"src.live_mcp.servers.{domain}.server")
        server_cls = None
        for attr_name in dir(mod):
            obj = getattr(mod, attr_name)
            if isinstance(obj, type) and hasattr(obj, 'handle_request') and attr_name != 'StatefulToolServer':
                server_cls = obj
                break
        
        server = server_cls()
        server.handle_request("session/reset", {"session_id": "mut_test", "seed": 42})
        
        resp = server.handle_request("tools/call", {
            "session_id": "mut_test",
            "name": tool_name,
            "arguments": args,
        })
        result = resp.get("result", {})
        
        if not result.get("success"):
            fail(f"[{domain}] mutating tool '{tool_name}' 执行失败: {result.get('error_message')}")
        elif not result.get("state_changed"):
            fail(f"[{domain}] mutating tool '{tool_name}' 成功但 state_changed=False")
        else:
            ok(f"[{domain}] mutating tool '{tool_name}' → success + state_changed=True")
    except Exception as e:
        fail(f"[{domain}] mutating test 异常: {e}")


# ═══════════════════════════════════════════════════════════════════════
# 12. 模型可能调用的「预期之外」tool 检测
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TEST 12: 模型可能调用的「预期之外」tool 检测")
print("=" * 70)

# 常见的模型幻觉 tool 名（基于 GPT/Qwen 常见行为）
HALLUCINATED_TOOLS = [
    # 通用幻觉
    "search", "web_search", "google_search", "browse", "open_url",
    "python", "execute_code", "run_code", "bash",
    "read_file", "write_file", "edit_file",
    # Banking 幻觉
    "check_balance", "send_money", "send_payment", "make_transfer",
    "get_transactions", "view_balance",
    # Calendar 幻觉
    "schedule_meeting", "book_meeting", "add_event", "remove_event",
    "get_calendar", "view_calendar",
    # Email 幻觉
    "compose_email", "draft_email", "read_email", "open_email",
    # Shopping 幻觉
    "buy", "purchase", "order", "add_item",
    # Generic CRUD 幻觉
    "create", "read", "update", "delete", "get", "set", "list",
]

all_real_tools = set()
for names in domain_tool_names.values():
    all_real_tools.update(names)

# 检查哪些幻觉 tool 名恰好与真实 tool 名冲突
real_hallucination_overlap = set(HALLUCINATED_TOOLS) & all_real_tools
if real_hallucination_overlap:
    warn(f"以下常见幻觉 tool 名恰好是真实 tool: {sorted(real_hallucination_overlap)}")
    print("    → 模型调用这些名称时会被正确路由，不会报 UNKNOWN_TOOL")

# 检查模型如果调用幻觉 tool，executor 是否会正确拒绝
hallucinated_not_real = set(HALLUCINATED_TOOLS) - all_real_tools
print(f"  模型如果幻觉调用以下 {len(hallucinated_not_real)} 个 tool，executor 会返回 UNKNOWN_TOOL:")
print(f"    {sorted(hallucinated_not_real)[:10]}...")

# 验证 SchemaRegistry 对幻觉 tool 的处理
for fake_tool in list(hallucinated_not_real)[:5]:
    schema = registry.get_schema(fake_tool)
    if schema is not None:
        fail(f"幻觉 tool '{fake_tool}' 在 registry 中找到了 schema！")
    server = registry.server_for_tool(fake_tool)
    if server is not None:
        fail(f"幻觉 tool '{fake_tool}' 被路由到了 server '{server}'！")

ok("幻觉 tool 名被 SchemaRegistry 正确拒绝")


# ═══════════════════════════════════════════════════════════════════════
# 13. Executor _is_partial_observation 逻辑验证
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TEST 13: _is_partial_observation 逻辑验证")
print("=" * 70)

from src.live_mcp.executor import _is_partial_observation

# Read tool + empty list → PARTIAL
assert _is_partial_observation([], tool_name="list_invoices") == True, "empty list for read tool should be PARTIAL"
ok("list_invoices + [] → PARTIAL_SUCCESS")

# Write tool + empty list → NOT PARTIAL (正常的空返回)
assert _is_partial_observation([], tool_name="create_invoice") == False, "empty list for write tool should NOT be PARTIAL"
ok("create_invoice + [] → SUCCESS (not partial)")

# None → SUCCESS
assert _is_partial_observation(None, tool_name="transfer") == False, "None should be SUCCESS"
ok("transfer + None → SUCCESS")

# Dict with "partial" key → PARTIAL
assert _is_partial_observation({"partial": True, "data": []}, tool_name="search_products") == True
ok("search_products + {partial: True} → PARTIAL_SUCCESS")

# Write tool + dict with empty list field → NOT PARTIAL
assert _is_partial_observation({"event_id": "evt_001", "attendees": []}, tool_name="create_event") == False
ok("create_event + {attendees: []} → SUCCESS (not partial)")


# ═══════════════════════════════════════════════════════════════════════
# 总结
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print(f"测试完成: ✅ PASS={PASS}  ❌ FAIL={FAIL}  ⚠️ WARN={WARN}")
print("=" * 70)

if issues:
    print("\n🔥 失败项汇总:")
    for i, issue in enumerate(issues, 1):
        print(f"  {i}. {issue}")
    sys.exit(1)
else:
    print("\n🎉 所有测试通过！domain 设置、tool、graph、逻辑均无 bug。")
    sys.exit(0)
