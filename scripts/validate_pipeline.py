#!/usr/bin/env python3
"""Domain Validation Pipeline —— 三阶段统一验证入口。

Stage 1 — 拓扑级：验证 dependency graph JSON 结构
    • 边引用的 tool 是否存在于 tool_names
    • 孤立节点检测（无入边也无出边）
    • 链长分布统计 + 环路检测
    • 确定性 schema 边 vs LLM 分类图对比表

Stage 2 — 逻辑级：Server tool schema 交叉验证
    • Config dependency_graph / tools 分类 vs Server TOOLS 一致性
    • explicit 边参数覆盖检查（source output 是否包含 target required）
    • SchemaRegistry 注册与解析正确性
    • Handler 完整性（每个 TOOLS 中的 tool 都有对应 handler）
    • 跨 domain tool 名冲突检测

Stage 3 — 生成冒烟：逐个 domain 小规模 data generation 测试
    • 调用 generate_data.py --domain X --count N
    • 检查运行时错误、parquet 产出

用法：
    python scripts/validate_pipeline.py                     # 全部三步
    python scripts/validate_pipeline.py --stages 1,2        # 仅拓扑+逻辑（无需 LLM）
    python scripts/validate_pipeline.py --stages 3 --domain banking --model gemini-2.5-flash --api-base https://your-proxy/v1
    python scripts/validate_pipeline.py --model gemini-2.5-flash --api-base https://your-proxy/v1 --count 5
"""

from __future__ import annotations

import argparse
import datetime
import importlib
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ── 项目内部依赖 ────────────────────────────────────────────────
from src.live_mcp.config import load_suite_config
from src.live_mcp.schema_registry import SchemaRegistry
from src.live_mcp.orchestrator import _deterministic_schema_edges

DOMAINS_ALL = [
    "banking", "calendar", "crm", "email", "filesystem",
    "food_delivery", "issue_tracker", "payments", "shopping", "team_chat",
]

CACHE_DIR = ROOT / "data" / "dependency_graphs"

# ═══════════════════════════════════════════════════════════════════
# 结果收集
# ═══════════════════════════════════════════════════════════════════
_results: dict[str, list[str]] = {"pass": [], "fail": [], "warn": []}

def _ok(stage: str, msg: str) -> None:
    _results["pass"].append(f"[{stage}] {msg}")
    print(f"  ✅ {msg}")

def _fail(stage: str, msg: str) -> None:
    _results["fail"].append(f"[{stage}] {msg}")
    print(f"  ❌ {msg}")

def _warn(stage: str, msg: str) -> None:
    _results["warn"].append(f"[{stage}] {msg}")
    print(f"  ⚠️  {msg}")


# ═══════════════════════════════════════════════════════════════════
# 共享工具函数
# ═══════════════════════════════════════════════════════════════════

def _resolve_domains(domain_arg: str) -> list[str]:
    if domain_arg == "all":
        return list(DOMAINS_ALL)
    domains = [d.strip() for d in domain_arg.split(",")]
    unknown = set(domains) - set(DOMAINS_ALL)
    if unknown:
        raise ValueError(f"Unknown domains: {sorted(unknown)}")
    return domains


def _load_cached_graphs() -> dict[str, dict]:
    """加载所有缓存的 dependency graph JSON。"""
    graphs: dict[str, dict] = {}
    if not CACHE_DIR.exists():
        return graphs
    for f in sorted(CACHE_DIR.glob("*.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            name = data.get("server_name", f.stem.split("_")[0])
            graphs[name] = data
        except Exception:
            pass
    return graphs


def _load_server_tools(domain: str) -> list[dict]:
    mod = importlib.import_module(f"src.live_mcp.servers.{domain}.server")
    return list(mod.TOOLS)


# ═══════════════════════════════════════════════════════════════════
# Stage 1 — 拓扑级验证
# ═══════════════════════════════════════════════════════════════════

def _has_cycle(graph: dict[str, dict]) -> bool:
    """DFS 环路检测。"""
    WHITE, GRAY, BLACK = 0, 1, 2
    color: dict[str, int] = {n: WHITE for n in graph}

    def dfs(node: str) -> bool:
        color[node] = GRAY
        for nb in graph.get(node, {}).get("explicit", []) + graph.get(node, {}).get("implicit", []):
            if color.get(nb, BLACK) == GRAY:
                return True
            if color.get(nb, BLACK) == WHITE and dfs(nb):
                return True
        color[node] = BLACK
        return False

    for n in graph:
        if color.get(n, BLACK) == WHITE and dfs(n):
            return True
    return False


def _extract_chains(graph: dict[str, dict], max_len: int = 5) -> list[list[str]]:
    """DFS 提取所有 length-2 到 max_len 的链。"""
    chains: list[list[str]] = []

    def dfs(cur: str, path: list[str], visited: set[str]) -> None:
        if len(path) >= max_len:
            return
        for nb in graph.get(cur, {}).get("explicit", []) + graph.get(cur, {}).get("implicit", []):
            if nb in visited:
                continue
            new = path + [nb]
            if len(new) >= 2:
                chains.append(list(new))
            dfs(nb, new, visited | {nb})

    for s in graph:
        dfs(s, [s], {s})

    seen: set[tuple] = set()
    deduped: list[list[str]] = []
    for c in chains:
        key = tuple(c)
        if key not in seen:
            seen.add(key)
            deduped.append(c)
    return deduped


def stage1_topology(domains: list[str]) -> None:
    """Stage 1: 拓扑级验证。"""
    print("\n" + "=" * 70)
    print("STAGE 1 — 拓扑级：Dependency Graph JSON 结构验证")
    print("=" * 70)

    cached = _load_cached_graphs()

    suite_config = load_suite_config("configs/live_mcp/suite_mvp.yaml")
    config_graphs: dict[str, dict] = {}
    for cfg in suite_config.servers:
        if cfg.dependency_graph:
            edges = cfg.dependency_graph.get("edges", [])
            g: dict[str, dict] = {}
            for e in edges:
                src, tgt = e["source_tool"], e["target_tool"]
                rel = e.get("relation", "implicit")
                g.setdefault(src, {"explicit": [], "implicit": []})
                g.setdefault(tgt, {"explicit": [], "implicit": []})
                g[src][rel].append(tgt)
            config_graphs[cfg.name] = g

    for domain in domains:
        print(f"\n── {domain} ──")

        # ── 1a. 检查是否有缓存 ──
        if domain not in cached:
            _warn("S1", f"{domain}: 无缓存 JSON（precompute 未完成或跳过）")
            continue

        data = cached[domain]
        graph: dict[str, dict] = data.get("graph", {})
        tool_names: list[str] = data.get("tool_names", [])

        # ── 1b. 边引用的 tool 是否在 tool_names 中 ──
        graph_tools: set[str] = set()
        for src, node in graph.items():
            graph_tools.add(src)
            for rel in ("explicit", "implicit"):
                for tgt in node.get(rel, []):
                    graph_tools.add(tgt)

        missing = graph_tools - set(tool_names)
        if missing:
            _fail("S1", f"{domain}: graph 引用了 tool_names 中不存在的 tool: {sorted(missing)}")
        else:
            _ok("S1", f"{domain}: 所有 {len(graph_tools)} 个 graph tool 引用有效")

        # ── 1c. 孤立节点 ──
        in_edges: set[str] = set()
        for src, node in graph.items():
            for rel in ("explicit", "implicit"):
                for tgt in node.get(rel, []):
                    in_edges.add(tgt)
        isolated = [
            n for n in tool_names
            if not (graph.get(n, {}).get("explicit") or graph.get(n, {}).get("implicit"))
            and n not in in_edges
        ]
        if isolated:
            _warn("S1", f"{domain}: {len(isolated)} 个孤立节点: {isolated}")
        else:
            _ok("S1", f"{domain}: 无孤立节点")

        # ── 1d. 环路检测 ──
        if _has_cycle(graph):
            _fail("S1", f"{domain}: 检测到环路！")
        else:
            _ok("S1", f"{domain}: 无环路")

        # ── 1e. 链提取与分布 ──
        chains = _extract_chains(graph)
        dist = Counter(len(c) for c in chains)
        dist_str = " ".join(f"{k}:{v}" for k, v in sorted(dist.items()))
        print(f"     chains={len(chains)}  dist=[{dist_str}]")

        # ── 1f. 边统计 ──
        ex = sum(len(g.get("explicit", [])) for g in graph.values())
        im = sum(len(g.get("implicit", [])) for g in graph.values())
        print(f"     nodes={len(graph)}  edges: explicit={ex}  implicit={im}")

        # ── 1g. 确定性图 vs 缓存图对比 ──
        if domain not in config_graphs:
            _warn("S1", f"{domain}: 无 config dependency_graph，跳过确定性对比")
            continue

        det = config_graphs[domain]
        det_edges: set[tuple[str, str]] = set()
        for src, node in det.items():
            for tgt in node.get("explicit", []):
                det_edges.add((src, tgt))

        cached_ex_edges: set[tuple[str, str]] = set()
        for src, node in graph.items():
            for tgt in node.get("explicit", []):
                cached_ex_edges.add((src, tgt))

        only_det = det_edges - cached_ex_edges
        only_cached = cached_ex_edges - det_edges
        if only_det:
            _warn("S1", f"{domain}: {len(only_det)} explicit 边仅存在于确定性图中: {sorted(only_det)[:5]}...")
        if only_cached and det_edges:  # only warn if det graph exists
            _warn("S1", f"{domain}: {len(only_cached)} explicit 边仅存在于 LLM 分类图中: {sorted(only_cached)[:5]}...")


# ═══════════════════════════════════════════════════════════════════
# Stage 2 — 逻辑级验证
# ═══════════════════════════════════════════════════════════════════

def stage2_logic(domains: list[str]) -> None:
    """Stage 2: 逻辑级验证。"""
    print("\n" + "=" * 70)
    print("STAGE 2 — 逻辑级：Server Tool Schema 交叉验证")
    print("=" * 70)

    # ── 加载所有 server TOOLS ──
    domain_tools: dict[str, list[dict]] = {}
    domain_tool_names: dict[str, set[str]] = {}

    for domain in domains:
        try:
            tools = _load_server_tools(domain)
            domain_tools[domain] = tools
            domain_tool_names[domain] = {t["name"] for t in tools}
        except Exception as e:
            _fail("S2", f"{domain}: TOOLS 加载失败: {e}")
            domain_tools[domain] = []
            domain_tool_names[domain] = set()

    suite_config = load_suite_config("configs/live_mcp/suite_mvp.yaml")

    for cfg in suite_config.servers:
        domain = cfg.name
        if domain not in domains:
            continue
        print(f"\n── {domain} ──")

        server_names = domain_tool_names.get(domain, set())

        # ── 2a. Config dependency_graph 引用的 tool 是否存在 ──
        if cfg.dependency_graph:
            edges = cfg.dependency_graph.get("edges", [])
            graph_tools: set[str] = set()
            for e in edges:
                graph_tools.add(e["source_tool"])
                graph_tools.add(e["target_tool"])
            missing = graph_tools - server_names
            if missing:
                _fail("S2", f"{domain}: config dep_graph 引用了不存在的 tool: {sorted(missing)}")
            else:
                _ok("S2", f"{domain}: config dep_graph {len(graph_tools)} tool 引用有效")
        else:
            _warn("S2", f"{domain}: 无 dependency_graph 配置")

        # ── 2b. Config tools 分类 vs Server TOOLS 一致性 ──
        config_all = set(cfg.tools.get("discovery", [])) | set(cfg.tools.get("readonly", [])) | set(cfg.tools.get("mutating", []))
        config_missing = config_all - server_names
        if config_missing:
            _fail("S2", f"{domain}: config tools 引用了不存在的 tool: {sorted(config_missing)}")
        else:
            _ok("S2", f"{domain}: config tools 分类与实际一致 ({len(config_all)} tools)")

        # ── 2c. annotations 一致性 ──
        config_readonly = set(cfg.tools.get("readonly", []))
        config_mutating = set(cfg.tools.get("mutating", []))
        for t in domain_tools.get(domain, []):
            name = t["name"]
            ann = t.get("annotations", {})
            if ann.get("mutating") and name in config_readonly:
                _warn("S2", f"{domain}: '{name}' annotations=mutating 但 config=readonly")
            if ann.get("readonly") and name in config_mutating:
                _warn("S2", f"{domain}: '{name}' annotations=readonly 但 config=mutating")

        # ── 2d. Handler 完整性 ──
        try:
            mod = importlib.import_module(f"src.live_mcp.servers.{domain}.server")
            server_cls = next(
                v for k, v in vars(mod).items()
                if isinstance(v, type) and hasattr(v, "handle_request") and k != "StatefulToolServer"
            )
            server = server_cls()
            registered = set(server.handlers.keys())
            missing_h = server_names - registered
            extra_h = registered - server_names
            if missing_h:
                _fail("S2", f"{domain}: {len(missing_h)} tool 无 handler: {sorted(missing_h)}")
            else:
                _ok("S2", f"{domain}: 所有 {len(server_names)} tool 都有 handler")
            if extra_h:
                _warn("S2", f"{domain}: {len(extra_h)} handler 不在 TOOLS 中: {sorted(extra_h)}")
        except Exception as e:
            _fail("S2", f"{domain}: handler 验证异常: {e}")

    # ── 2e. SchemaRegistry 正确性 ──
    print(f"\n── SchemaRegistry ──")
    registry = SchemaRegistry()
    for domain in domains:
        registry.register_tools(domain, domain_tools.get(domain, []))

    total = sum(len(v) for v in domain_tools.values())
    errors = 0
    for domain in domains:
        for t in domain_tools.get(domain, []):
            name = t["name"]
            schema = registry.get_schema(name, domain=domain)
            if schema is None:
                _fail("S2", f"registry.get_schema('{name}', domain='{domain}') → None")
                errors += 1
            srv = registry.server_for_tool(name, domain=domain)
            if srv != domain:
                _fail("S2", f"registry.server_for_tool('{name}', domain='{domain}') → '{srv}'")
                errors += 1
    if errors == 0:
        _ok("S2", f"SchemaRegistry 注册 {total} tools，解析全部正确")

    # ── 2f. 跨 domain tool 名冲突 ──
    name_to_domains: dict[str, list[str]] = defaultdict(list)
    for domain, names in domain_tool_names.items():
        for name in names:
            name_to_domains[name].append(domain)
    conflicts = {n: d for n, d in name_to_domains.items() if len(d) > 1}
    if conflicts:
        for name, dms in sorted(conflicts.items()):
            _warn("S2", f"tool '{name}' 存在于多个 domain: {dms}")
    else:
        _ok("S2", "无跨 domain tool 名冲突")


# ═══════════════════════════════════════════════════════════════════
# Stage 3 — 生成冒烟
# ═══════════════════════════════════════════════════════════════════

def stage3_smoke(domains: list[str], model: str, count: int, api_base: str | None, device: int | None) -> None:
    """Stage 3: 一次性 data generation smoke test（避免重复加载模型）。"""
    total = count * len(domains)
    print("\n" + "=" * 70)
    print(f"STAGE 3 — 生成冒烟：{len(domains)} domains × {count} = {total} 条，单次调用（避免重复加载模型）")
    print("=" * 70)

    script = ROOT / "scripts" / "generate_data.py"
    domains_str = ",".join(domains)

    cmd = [
        sys.executable, str(script),
        "--domain", domains_str,
        "--count", str(total),
        "--val-count", "0",
        "--model", model,
        "--seed", "42",
        "--output", "/tmp/validate_pipeline_train.parquet",
    ]
    if api_base:
        cmd += ["--api-base", api_base]
    if device is not None:
        cmd += ["--device", str(device)]

    try:
        result = subprocess.run(
            cmd,
            timeout=3600,  # 60 min（本地 transformers 推理 50 条）
            cwd=str(ROOT),
        )
        if result.returncode != 0:
            _fail("S3", f"generate_data 返回码 {result.returncode}")
        else:
            _ok("S3", f"generate_data 成功 ({total} 条)")
    except subprocess.TimeoutExpired:
        _fail("S3", f"generate_data 超时 (>{3600}s)")
    except Exception as e:
        _fail("S3", f"generate_data 异常: {e}")

    # 归档临时文件到 logs/ 供后续分析
    import shutil
    tmp = Path("/tmp/validate_pipeline_train.parquet")
    if tmp.exists():
        archive_dir = ROOT / "logs"
        archive_dir.mkdir(parents=True, exist_ok=True)
        archive_name = f"validate_pipeline_train_{datetime.datetime.now():%Y%m%d_%H%M%S}.parquet"
        shutil.copy2(tmp, archive_dir / archive_name)
        tmp.unlink()


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Domain Validation Pipeline — 三阶段统一验证",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  python scripts/validate_pipeline.py --model gemini-2.5-flash --api-base https://your-proxy/v1
  python scripts/validate_pipeline.py --stages 1,2
  python scripts/validate_pipeline.py --stages 3 --domain banking --model gemini-2.5-flash --api-base https://your-proxy/v1
        """,
    )
    p.add_argument("--stages", default="1,2,3",
                   help="执行的阶段，逗号分隔 (默认: 1,2,3)")
    p.add_argument("--domain", default="all",
                   help="目标 domain，all 或逗号分隔列表 (默认: all)")
    p.add_argument("--model", default=None,
                   help="Stage 3 使用的模型路径 (Stage 3 必需)")
    p.add_argument("--count", type=int, default=5,
                   help="Stage 3 每个 domain 生成的任务数 (默认: 5)")
    p.add_argument("--api-base", default=None,
                   help="Stage 3 OpenAI-compatible API base URL")
    p.add_argument("--device", type=int, default=None,
                   help="Stage 3 GPU device ID")
    return p


def main() -> int:
    args = build_parser().parse_args()

    stages = {int(s.strip()) for s in args.stages.split(",")}
    domains = _resolve_domains(args.domain)

    if 3 in stages and not args.model:
        print("❌ Stage 3 需要 --model 参数")
        return 1

    print(f"Validate Pipeline: stages={sorted(stages)}  domains={domains}")
    print(f"  model={args.model or '(n/a)'}  count={args.count}")

    if 1 in stages:
        stage1_topology(domains)

    if 2 in stages:
        stage2_logic(domains)

    if 3 in stages:
        stage3_smoke(domains, args.model, args.count, args.api_base, args.device)

    # ── 汇总 ──
    print("\n" + "=" * 70)
    print(f"结果汇总: ✅ {len(_results['pass'])}  "
          f"❌ {len(_results['fail'])}  "
          f"⚠️  {len(_results['warn'])}")
    print("=" * 70)

    if _results["fail"]:
        print("\n🔥 失败项:")
        for item in _results["fail"]:
            print(f"  {item}")

    if _results["warn"]:
        print("\n⚡ 警告项:")
        for item in _results["warn"]:
            print(f"  {item}")

    return 1 if _results["fail"] else 0


if __name__ == "__main__":
    sys.exit(main())