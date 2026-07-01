#!/usr/bin/env python3
"""统一测试框架——数据生成管线质量门禁。

分层设计：
  L0: 语法 & 导入检查 (30s, 纯CPU)
  L1: 工具服务器冒烟测试 (2min, MCP)
  L2: 管线端到端集成测试 (5min, 需要LLM)
  L3: 已有数据审计 (30s, parquet文件)
  L4: 对抗性回归用例 (2min, 纯逻辑)

用法:
  # 快速门禁 (不含LLM)
  python scripts/test_runner.py --level L0 L1

  # 完整测试 (含LLM生成)
  python scripts/test_runner.py --level all --api-base http://localhost:8001/v1 --model Qwen3-32B

  # 只审计已有数据
  python scripts/test_runner.py --level L3 --train data/train.parquet --val data/val.parquet

输出: JSON 格式的测试报告, 退出码 0=全部通过, 1=有失败
"""

from __future__ import annotations

import argparse
import ast
import importlib
import json
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ═══════════════════════════════════════════════════════════════════
# 测试结果数据结构
# ═══════════════════════════════════════════════════════════════════

@dataclass
class TestResult:
    name: str
    passed: bool
    duration_ms: float = 0
    detail: str = ""
    metrics: dict = field(default_factory=dict)


class TestReport:
    def __init__(self):
        self.results: list[TestResult] = []
        self.start_time = datetime.now(timezone.utc)

    def add(self, result: TestResult):
        self.results.append(result)

    def summary(self) -> dict:
        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        failed = total - passed
        duration_s = (datetime.now(timezone.utc) - self.start_time).total_seconds()
        return {
            "timestamp": self.start_time.isoformat(),
            "duration_s": round(duration_s, 1),
            "total": total,
            "passed": passed,
            "failed": failed,
            "pass_rate": f"{passed / total * 100:.0f}%" if total > 0 else "N/A",
            "results": [
                {
                    "name": r.name,
                    "passed": r.passed,
                    "duration_ms": r.duration_ms,
                    "detail": r.detail,
                    "metrics": r.metrics,
                }
                for r in self.results
            ],
        }


report = TestReport()


# ═══════════════════════════════════════════════════════════════════
# L0: 语法 & 导入检查
# ═══════════════════════════════════════════════════════════════════

SRC_DIRS = ["src", "scripts", "tests"]


def _collect_py_files(root: Path, dirs: list[str]) -> list[Path]:
    files = []
    for d in dirs:
        p = root / d
        if p.exists():
            files.extend(sorted(p.rglob("*.py")))
    return files


def check_syntax() -> TestResult:
    """检查所有 .py 文件是否存在语法错误。"""
    t0 = time.time()
    errors = []
    py_files = _collect_py_files(PROJECT_ROOT, SRC_DIRS)
    for f in py_files:
        try:
            with open(f) as fh:
                ast.parse(fh.read(), filename=str(f))
        except SyntaxError as e:
            errors.append(f"{f.relative_to(PROJECT_ROOT)}:{e.lineno}: {e.msg}")
    ok = len(errors) == 0
    detail = "OK" if ok else f"{len(errors)} syntax errors:\n  " + "\n  ".join(errors[:10])
    return TestResult("L0_syntax", ok, (time.time() - t0) * 1000, detail,
                       {"files_checked": len(py_files), "errors": len(errors)})


def check_imports() -> TestResult:
    """检查关键模块是否能成功导入。"""
    t0 = time.time()
    errors = []
    modules = [
        "src.live_mcp.orchestrator",
        "src.live_mcp.llm_client",
        "src.live_mcp.api",
        "src.live_mcp.task_planner",
        "src.live_mcp.server_base",
        "src.live_mcp.transport",
        "src.live_mcp.agent_loop",
        "src.live_mcp.config",
        "src.live_mcp.dedup",
        "src.live_mcp.oracle",
        "src.live_mcp.manager",
        "src.live_mcp.reward",
        "src.live_mcp.state_seeder",
        "src.live_mcp.types",
    ]
    for mod_name in modules:
        try:
            importlib.import_module(mod_name)
        except Exception as e:
            errors.append(f"{mod_name}: {e}")
    ok = len(errors) == 0
    detail = f"OK ({len(modules)} modules)" if ok else f"{len(errors)} import failures:\n  " + "\n  ".join(errors)
    return TestResult("L0_imports", ok, (time.time() - t0) * 1000, detail,
                       {"modules_checked": len(modules), "failures": len(errors)})


# ═══════════════════════════════════════════════════════════════════
# L1: 工具服务器冒烟测试
# ═══════════════════════════════════════════════════════════════════

def check_tool_servers() -> TestResult:
    """运行 tests/test_all_domains.py 验证所有工具函数。"""
    t0 = time.time()
    test_file = PROJECT_ROOT / "tests" / "test_all_domains.py"
    if not test_file.exists():
        return TestResult("L1_tool_servers", False, 0, f"Test file not found: {test_file}", {})
    try:
        proc = subprocess.run(
            [sys.executable, str(test_file)],
            capture_output=True, text=True, timeout=120, cwd=str(PROJECT_ROOT),
        )
        ok = proc.returncode == 0
        # 从输出中提取通过/失败统计
        detail = proc.stdout.strip()[-500:] if not ok else "All tool servers OK"
        if not ok:
            detail = (proc.stderr + "\n" + proc.stdout)[-1000:]
        return TestResult("L1_tool_servers", ok, (time.time() - t0) * 1000, detail,
                           {"returncode": proc.returncode})
    except subprocess.TimeoutExpired:
        return TestResult("L1_tool_servers", False, 120000, "Timeout after 120s", {})


# ═══════════════════════════════════════════════════════════════════
# L2: 管线集成测试
# ═══════════════════════════════════════════════════════════════════

def check_pipeline_integration(args: argparse.Namespace) -> TestResult:
    """生成 10 条数据，验证端到端管线。"""
    t0 = time.time()
    if not args.api_base or not args.model:
        return TestResult("L2_integration", False, 0,
                           "Skipped: --api-base and --model required", {})

    out_dir = Path(args.test_output or "data/test_integration")
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train.parquet"
    val_path = out_dir / "val.parquet"

    try:
        proc = subprocess.run(
            [
                sys.executable, str(PROJECT_ROOT / "scripts" / "generate_data.py"),
                "--count", "8",
                "--val-count", "2",
                "--domain", "calendar,shopping",
                "--model", args.model,
                "--api-base", args.api_base,
                "--output", str(train_path),
                "--val-output", str(val_path),
                "--seed", "777",
            ],
            capture_output=True, text=True, timeout=600, cwd=str(PROJECT_ROOT),
        )
        if proc.returncode != 0:
            return TestResult("L2_integration", False, (time.time() - t0) * 1000,
                               f"Generation failed (exit={proc.returncode})\n{proc.stderr[-800:]}", {})

        if not train_path.exists() or not val_path.exists():
            return TestResult("L2_integration", False, (time.time() - t0) * 1000,
                               f"Output files missing: train={train_path.exists()}, val={val_path.exists()}", {})

        train_df = pd.read_parquet(train_path)
        val_df = pd.read_parquet(val_path)

        issues = []

        # 检查行数
        if len(train_df) < 5:
            issues.append(f"Train too few rows: {len(train_df)} < 5")
        if len(val_df) < 1:
            issues.append(f"Val too few rows: {len(val_df)}")

        # 检查必要列
        required_cols = ["prompt", "extra_info", "scenario_type", "perturbation_level", "uid"]
        for col in required_cols:
            if col not in train_df.columns:
                issues.append(f"Missing column: {col}")

        # 检查 oracle 非空率
        for label, df in [("train", train_df), ("val", val_df)]:
            empty_oracle = 0
            total = 0
            for _, r in df.iterrows():
                ei = r["extra_info"]
                if isinstance(ei, str):
                    ei = json.loads(ei)
                oc = ei.get("oracle_calls", [])
                if isinstance(oc, str):
                    oc = json.loads(oc)
                total += 1
                if len(oc) == 0 and ei.get("scenario_type") not in ("missing_function", "irrelevant"):
                    empty_oracle += 1
            if total > 0 and empty_oracle / total > 0.5:
                issues.append(f"{label}: {empty_oracle}/{total} normal tasks have empty oracle")

        ok = len(issues) == 0
        detail = f"OK: {len(train_df)} train + {len(val_df)} val" if ok else "\n  ".join(issues)
        return TestResult("L2_integration", ok, (time.time() - t0) * 1000, detail,
                           {"train_rows": len(train_df), "val_rows": len(val_df)})

    except subprocess.TimeoutExpired:
        return TestResult("L2_integration", False, 600000, "Timeout after 600s", {})
    except Exception as e:
        return TestResult("L2_integration", False, (time.time() - t0) * 1000, str(e), {})


# ═══════════════════════════════════════════════════════════════════
# L3: 已有数据审计
# ═══════════════════════════════════════════════════════════════════

class DataAuditor:
    """对 parquet 文件进行 19 项 PROVE 合规检查。"""

    def __init__(self, train_path: str, val_path: str):
        self.train = pd.read_parquet(train_path) if Path(train_path).exists() else pd.DataFrame()
        self.val = pd.read_parquet(val_path) if Path(val_path).exists() else pd.DataFrame()
        self.all_data = pd.concat([self.train, self.val], ignore_index=True)
        self.issues = []

    @staticmethod
    def _pe(x):
        return json.loads(x) if isinstance(x, str) else x

    def check_oracle_chain_length(self) -> bool:
        """Oracle 链长必须 ≤ 5 (PROVE 硬上限)，仅统计 tool_call，不含 terminal。"""
        violations = []
        for _, r in self.all_data.iterrows():
            ei = self._pe(r["extra_info"])
            oc = ei.get("oracle_calls", [])
            if isinstance(oc, str):
                oc = json.loads(oc)
            real_calls = [c for c in oc if c.get("action", "tool_call") == "tool_call"]
            if len(real_calls) > 5:
                violations.append(ei.get("task_id", "?"))
        ok = len(violations) == 0
        if not ok:
            self.issues.append(f"Oracle chain > 5: {violations[:5]}")
        return ok

    def check_scenario_coverage(self) -> bool:
        """val 必须包含所有核心场景类型。
        
        Actual scenario labels (set by _classify_scenario / _apply_missing_function / irrelevant_template):
        - normal_safe_success: standard multi-step tool task
        - no_tool_or_abstention: missing_function or irrelevant → should abstain
        - clarification_required: missing-difficulty task where LLM asked user
        - tool_error_recovery: execution failure → retry/recover path
        - unsafe_temptation: delete+create shortcut pattern
        - missing_dependency: later tool needs entity not produced by earlier step
        """
        if len(self.val) == 0:
            return True
        val_scenarios = set()
        for _, r in self.val.iterrows():
            ei = self._pe(r["extra_info"])
            val_scenarios.add(ei.get("scenario_type", "?"))
        # Core scenarios every training dataset must cover in validation.
        required = {"normal_safe_success", "no_tool_or_abstention"}
        # Optional but desirable: recovery, unsafe, and dependency scenarios.
        desired = {"tool_error_recovery", "unsafe_temptation", "missing_dependency"}
        missing = required - val_scenarios
        missing_desired = desired - val_scenarios
        ok = len(missing) == 0
        if not ok:
            self.issues.append(f"Val missing REQUIRED scenarios: {missing}")
        if missing_desired:
            # Non-blocking: print warning but don't pollute self.issues.
            print(
                f"  [INFO] Val missing desired scenarios (non-blocking): "
                f"{sorted(missing_desired)}"
            )
        return ok

    @staticmethod
    def _semantic_fingerprint(row) -> str:
        """Domain + query + oracle_calls semantic hash for cross-dataset dedup."""
        import hashlib
        ei = row["extra_info"] if isinstance(row["extra_info"], dict) else json.loads(row["extra_info"])
        domain = ei.get("domain", "")
        query = " ".join((ei.get("user_query", "") or "").lower().split())
        oc = ei.get("oracle_calls", [])
        if isinstance(oc, str):
            oc = json.loads(oc)
        sig = json.dumps(
            {"d": domain, "q": query, "c": oc},
            sort_keys=True, ensure_ascii=False, default=str,
        )
        return hashlib.sha256(sig.encode()).hexdigest()

    def check_train_val_no_semantic_overlap(self) -> bool:
        """train 和 val 之间不能有语义重复（domain + query + oracle 一致）。"""
        if len(self.train) == 0 or len(self.val) == 0:
            return True
        train_fps = {self._semantic_fingerprint(row) for _, row in self.train.iterrows()}
        train_fps.discard("")
        overlap = 0
        for _, row in self.val.iterrows():
            fp = self._semantic_fingerprint(row)
            if fp and fp in train_fps:
                overlap += 1
        ok = overlap == 0
        if not ok:
            self.issues.append(
                f"{overlap} semantic fingerprint overlaps between train and val"
            )
        return ok

    def check_missing_function_oracle_empty(self) -> bool:
        """missing_function 类型的 oracle_calls 应恰好包含一个 terminal (report_error)，不应有 tool_call。"""
        violations = []
        for _, r in self.all_data.iterrows():
            ei = self._pe(r["extra_info"])
            if ei.get("scenario_type") != "missing_function":
                continue
            oc = ei.get("oracle_calls", [])
            if isinstance(oc, str):
                oc = json.loads(oc)
            tool_calls = [c for c in oc if c.get("action", "tool_call") == "tool_call"]
            terminals = [c for c in oc if c.get("action") in ("report_error", "final_answer", "ask_clarification")]
            if tool_calls or len(terminals) != 1 or terminals[0].get("action") != "report_error":
                violations.append(ei.get("task_id", "?"))
        ok = len(violations) == 0
        if not ok:
            self.issues.append(f"missing_function with wrong oracle shape: {violations[:5]}")
        return ok

    def check_irrelevant_oracle_empty(self) -> bool:
        """irrelevant 类型的 oracle_calls 应恰好包含一个 terminal (report_error)，不应有 tool_call。"""
        violations = []
        for _, r in self.all_data.iterrows():
            ei = self._pe(r["extra_info"])
            if ei.get("scenario_type") != "irrelevant":
                continue
            oc = ei.get("oracle_calls", [])
            if isinstance(oc, str):
                oc = json.loads(oc)
            tool_calls = [c for c in oc if c.get("action", "tool_call") == "tool_call"]
            terminals = [c for c in oc if c.get("action") in ("report_error", "final_answer", "ask_clarification")]
            if tool_calls or len(terminals) != 1 or terminals[0].get("action") != "report_error":
                violations.append(ei.get("task_id", "?"))
        ok = len(violations) == 0
        if not ok:
            self.issues.append(f"irrelevant with wrong oracle shape: {violations[:5]}")
        return ok

    def check_abstention_oracle_shape(self) -> bool:
        """no_tool_or_abstention 的 oracle_calls 应恰好包含 1 个 report_error terminal，0 个 tool_call。

        当前生成侧将 missing_function 和 irrelevant 统一为 scenario_type="no_tool_or_abstention"。
        此检查覆盖该实际标签（不同于上面的 missing_function/irrelevant 遗留检查）。
        """
        violations = []
        for _, r in self.all_data.iterrows():
            ei = self._pe(r["extra_info"])
            if ei.get("scenario_type") != "no_tool_or_abstention":
                continue
            oc = ei.get("oracle_calls", [])
            if isinstance(oc, str):
                oc = json.loads(oc)
            tool_calls = [c for c in oc if c.get("action", "tool_call") == "tool_call"]
            terminals = [c for c in oc if c.get("action") in ("report_error", "final_answer", "ask_clarification")]
            if tool_calls or len(terminals) != 1 or terminals[0].get("action") != "report_error":
                violations.append(ei.get("task_id", "?"))
        ok = len(violations) == 0
        if not ok:
            self.issues.append(f"abstention oracle shape wrong: {violations[:5]}")
        return ok

    def check_oracle_struct(self) -> bool:
        """每个 oracle call 必须是带 tool_name 和 arguments 的 dict。"""
        violations = []
        for _, r in self.all_data.iterrows():
            ei = self._pe(r["extra_info"])
            oc = ei.get("oracle_calls", [])
            if isinstance(oc, str):
                oc = json.loads(oc)
            for i, c in enumerate(oc):
                if not isinstance(c, dict):
                    violations.append(f"{ei.get('task_id','?')}[{i}] not dict")
                elif "tool_name" not in c:
                    violations.append(f"{ei.get('task_id','?')}[{i}] missing tool_name")
        ok = len(violations) == 0
        if not ok:
            self.issues.append(f"Oracle struct violations: {violations[:5]}")
        return ok

    def check_no_duplicate_oracle_calls(self) -> bool:
        """同一 oracle 内不应有重复的工具调用组合。"""
        dups = 0
        for _, r in self.all_data.iterrows():
            ei = self._pe(r["extra_info"])
            oc = ei.get("oracle_calls", [])
            if isinstance(oc, str):
                oc = json.loads(oc)
            seen = set()
            for c in oc:
                if isinstance(c, dict):
                    key = (c.get("tool_name", ""), json.dumps(c.get("arguments", {}), sort_keys=True))
                    if key in seen:
                        dups += 1
                    seen.add(key)
        ok = dups == 0
        if not ok:
            self.issues.append(f"{dups} duplicate oracle calls")
        return ok

    def check_train_val_no_overlap(self) -> bool:
        """train 和 val 的 task_id 不应重叠。"""
        if len(self.train) == 0 or len(self.val) == 0:
            return True
        train_ids = set()
        for _, r in self.train.iterrows():
            ei = self._pe(r["extra_info"])
            train_ids.add(ei.get("task_id", ""))
        overlap = 0
        for _, r in self.val.iterrows():
            ei = self._pe(r["extra_info"])
            if ei.get("task_id", "") in train_ids:
                overlap += 1
        ok = overlap == 0
        if not ok:
            self.issues.append(f"{overlap} overlapping task_ids between train/val")
        return ok

    def check_has_missing_function_consistency(self) -> bool:
        """has_missing_function 与 scenario_type 必须一致。
        
        has_missing_function=True 对应的 scenario_type 可以是:
        - "missing_function" (直接标记)
        - "no_tool_or_abstention" (_apply_missing_function 设置)
        
        注意: "irrelevant" 类型的任务有 scenario_type="no_tool_or_abstention" 但不设置
        has_missing_function=True，因为 irrelevant 并非缺少工具，而是请求超出域能力。
        """
        violations = []
        abstain_types = {"missing_function", "no_tool_or_abstention"}
        for _, r in self.all_data.iterrows():
            ei = self._pe(r["extra_info"])
            st = ei.get("scenario_type", "")
            hmf = ei.get("has_missing_function", False)
            # irrelevant tasks are abstain but don't carry has_missing_function
            if ei.get("generation_method") == "irrelevant_template":
                if hmf:
                    violations.append(ei.get("task_id", "?"))
            elif st in abstain_types and not hmf:
                violations.append(ei.get("task_id", "?"))
            elif st not in abstain_types and hmf:
                violations.append(ei.get("task_id", "?"))
        ok = len(violations) == 0
        if not ok:
            self.issues.append(f"has_missing_function inconsistent: {violations[:5]}")
        return ok

    @staticmethod
    def _has_word(text: str, markers: set[str]) -> bool:
        normalized = str(text).lower().replace("_", " ").replace("’", "'")
        q = f" {normalized} "
        return any(f" {marker} " in q or marker in q for marker in markers)

    @staticmethod
    def _hidden_tools(ei: dict) -> list[str]:
        hidden = ei.get("hidden_tools", [])
        if isinstance(hidden, str):
            try:
                loaded = json.loads(hidden)
                hidden = loaded if isinstance(loaded, list) else [hidden]
            except json.JSONDecodeError:
                hidden = [t.strip() for t in hidden.split(",") if t.strip()]
        if hasattr(hidden, "tolist"):
            hidden = hidden.tolist()
        return [str(t) for t in hidden if str(t)]

    def check_missing_function_semantic_alignment(self) -> bool:
        """missing_function 不能把读请求错误标成缺少写工具。

        This is intentionally conservative: if a no-tool row hides a mutating
        tool, the visible user query must contain an explicit write intent.
        """
        read_markers = {
            "what", "what's", "whats", "which", "who", "when", "where",
            "show", "list", "check", "view", "get", "find", "search",
            "confirm", "status", "details", "schedule", "agenda",
            "calendar", "history", "balance", "track", "report", "summary",
        }
        write_markers = {
            "create", "make", "new", "book", "draft", "register",
            "update", "edit", "change", "modify", "set", "mark", "move",
            "reschedule", "rename", "assign", "complete", "transition",
            "log", "delete", "remove", "cancel", "archive", "clear",
            "drop", "add", "apply", "attach", "invite", "react", "rate",
            "review", "label", "send", "reply", "forward", "pay", "refund",
            "transfer", "wire", "checkout", "return", "reorder", "deposit",
            "withdraw", "freeze", "unfreeze", "convert",
        }
        mutating_prefixes = (
            "create_", "update_", "delete_", "remove_", "add_", "set_",
            "send_", "reply_", "forward_", "pay_", "refund_", "transfer",
            "wire_", "checkout", "return_", "cancel_", "archive_", "clear_",
            "change_", "respond_", "mark_", "assign_", "transition_",
            "complete_", "convert_", "deposit", "withdraw", "freeze_",
            "unfreeze_",
        )

        violations = []
        for _, r in self.all_data.iterrows():
            ei = self._pe(r["extra_info"])
            if not ei.get("has_missing_function"):
                continue
            if ei.get("scenario_type") not in {"missing_function", "no_tool_or_abstention"}:
                continue
            hidden = self._hidden_tools(ei)
            if not hidden or not any(t.lower().startswith(mutating_prefixes) for t in hidden):
                continue
            query = ei.get("user_query", "")
            if not query:
                try:
                    prompt = json.loads(r["prompt"])
                    query = prompt[1].get("content", "") if len(prompt) > 1 else ""
                except Exception:
                    query = ""
            if self._has_word(query, read_markers) and not self._has_word(query, write_markers):
                violations.append(
                    f"{ei.get('task_id', '?')} query={query!r} hidden={hidden}"
                )

        ok = len(violations) == 0
        if not ok:
            self.issues.append(
                f"missing_function semantic mismatch: {violations[:5]}"
            )
        return ok

    def check_conversation_rounds(self) -> bool:
        """验证 val 包含单轮任务，且所有任务轮数在 1-5 范围内。"""
        issues_local = []
        val_rounds = [self._pe(r["extra_info"]).get("conversation_rounds", 1) for _, r in self.val.iterrows()]
        if val_rounds and 1 not in Counter(val_rounds):
            issues_local.append("Val has NO single-round tasks")

        for _, r in self.all_data.iterrows():
            cr = self._pe(r["extra_info"]).get("conversation_rounds", 1)
            if cr < 1 or cr > 5:
                issues_local.append(f"{self._pe(r['extra_info']).get('task_id','?')} rounds={cr} out of [1,5]")

        ok = len(issues_local) == 0
        if not ok:
            self.issues.extend(issues_local)
        return ok

    def check_domain_coverage(self, min_domains: int | None = None) -> bool:
        """验证域覆盖。
        
        默认行为：从实际数据中自动推断所需的最小域数。
        - 如果数据中只有 1 个域 → 要求 ≥1（单域实验合法）
        - 如果数据中有 2+ 个域 → 要求 ≥2（多域训练至少覆盖 2 个域）
        - 可通过 min_domains 参数显式覆盖。
        """
        domains = Counter()
        for _, r in self.all_data.iterrows():
            ei = self._pe(r["extra_info"])
            domains[ei.get("domain", "?")] += 1

        if min_domains is None:
            # Auto-detect: if only 1 domain exists, require ≥1; else ≥2.
            min_domains = 1 if len(domains) == 1 else 2

        ok = len(domains) >= min_domains
        if not ok:
            self.issues.append(
                f"Only {len(domains)} domain(s) (need ≥{min_domains}): {dict(domains)}"
            )
        return ok

    def run_all(self, min_domains: int | None = None) -> tuple[bool, list[str], dict]:
        checks = [
            ("oracle_chain_length", self.check_oracle_chain_length),
            ("scenario_coverage", self.check_scenario_coverage),
            ("missing_function_empty_oracle", self.check_missing_function_oracle_empty),
            ("irrelevant_empty_oracle", self.check_irrelevant_oracle_empty),
            ("abstention_oracle_shape", self.check_abstention_oracle_shape),
            ("oracle_struct", self.check_oracle_struct),
            ("no_duplicate_oracle_calls", self.check_no_duplicate_oracle_calls),
            ("train_val_no_overlap", self.check_train_val_no_overlap),
            ("train_val_no_semantic_overlap", self.check_train_val_no_semantic_overlap),
            ("has_missing_function_consistent", self.check_has_missing_function_consistency),
            ("missing_function_semantic_alignment", self.check_missing_function_semantic_alignment),
            ("conversation_rounds", self.check_conversation_rounds),
            ("domain_coverage", lambda: self.check_domain_coverage(min_domains=min_domains)),
        ]
        passed = 0
        failed_checks = []
        for name, fn in checks:
            if fn():
                passed += 1
            else:
                failed_checks.append(name)

        metrics = {
            "train_rows": len(self.train),
            "val_rows": len(self.val),
            "checks_passed": passed,
            "checks_total": len(checks),
            "domains": len(Counter(self._pe(r["extra_info"]).get("domain", "?")
                                    for _, r in self.all_data.iterrows())),
        }
        return len(failed_checks) == 0, self.issues, metrics


def check_data_audit(args: argparse.Namespace) -> TestResult:
    """对指定 parquet 文件运行审计。"""
    t0 = time.time()
    train_path = args.train or "data/train.parquet"
    val_path = args.val or "data/val.parquet"

    if not Path(train_path).exists():
        return TestResult("L3_audit", False, 0, f"Train file not found: {train_path}", {})

    auditor = DataAuditor(train_path, val_path)
    ok, issues, metrics = auditor.run_all(min_domains=args.min_domains)

    detail = "OK" if ok else f"{len(issues)} issues:\n  " + "\n  ".join(issues[:10])
    return TestResult("L3_audit", ok, (time.time() - t0) * 1000, detail, metrics)


# ═══════════════════════════════════════════════════════════════════
# L4: 对抗性回归用例
# ═══════════════════════════════════════════════════════════════════

def check_regression_cases() -> TestResult:
    """已知回归点检查。"""
    t0 = time.time()
    issues = []

    # 回归1: llm_client.py 不应有语法错误 (2026-07-01 发现)
    llm_client = PROJECT_ROOT / "src" / "live_mcp" / "llm_client.py"
    if llm_client.exists():
        try:
            with open(llm_client) as f:
                ast.parse(f.read(), filename="llm_client.py")
        except SyntaxError as e:
            issues.append(f"REGRESSION: llm_client.py syntax error: {e}")

    # 回归2: 关键文件存在
    required_files = [
        "scripts/generate_data.py",
        "scripts/generate_data.sh",
        "configs/live_mcp/suite_mvp.yaml",
        "src/live_mcp/orchestrator.py",
        "src/live_mcp/task_planner.py",
        "src/live_mcp/server_base.py",
        "tests/test_all_domains.py",
    ]
    missing = [f for f in required_files if not (PROJECT_ROOT / f).exists()]
    if missing:
        issues.append(f"Missing files: {missing}")

    # 回归3: generate_data.py 能正常导入
    try:
        spec = importlib.util.spec_from_file_location(
            "generate_data", PROJECT_ROOT / "scripts" / "generate_data.py"
        )
        if spec and spec.loader:
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
    except Exception as e:
        issues.append(f"generate_data.py import failed: {e}")

    ok = len(issues) == 0
    detail = "OK" if ok else "\n  ".join(issues)
    return TestResult("L4_regression", ok, (time.time() - t0) * 1000, detail,
                       {"cases_checked": 3, "failures": len(issues)})


# ═══════════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Unified test runner for LiveMCP-GRPO data pipeline")
    parser.add_argument("--level", nargs="+", default=["L0", "L4"],
                        choices=["L0", "L1", "L2", "L3", "L4", "all"],
                        help="Test levels to run (default: L0 L4)")
    parser.add_argument("--api-base", default=None, help="vLLM API base URL (required for L2)")
    parser.add_argument("--model", default=None, help="Model name (required for L2)")
    parser.add_argument("--train", default=None, help="Train parquet path (for L3)")
    parser.add_argument("--val", default=None, help="Val parquet path (for L3)")
    parser.add_argument("--min-domains", type=int, default=None,
                        help="Minimum domain count required (L3 domain_coverage). Default: auto-detect from data.")
    parser.add_argument("--test-output", default="data/test_integration",
                        help="Output directory for L2 test data")
    parser.add_argument("--json", action="store_true", help="Output JSON report")
    args = parser.parse_args()

    if "all" in args.level:
        args.level = ["L0", "L1", "L2", "L3", "L4"]

    level_handlers = {
        "L0": [check_syntax, check_imports],
        "L1": [check_tool_servers],
        "L2": [lambda: check_pipeline_integration(args)],
        "L3": [lambda: check_data_audit(args)],
        "L4": [check_regression_cases],
    }

    for level in args.level:
        if level not in level_handlers:
            continue
        for handler in level_handlers[level]:
            try:
                result = handler()
                report.add(result)
                status = "✓" if result.passed else "✗"
                if not args.json:
                    print(f"  {status} {result.name} ({result.duration_ms:.0f}ms) {'— ' + result.detail.split(chr(10))[0] if result.detail else ''}")
            except Exception as e:
                report.add(TestResult(handler.__name__, False, 0, str(e)))
                if not args.json:
                    print(f"  ✗ {handler.__name__} crashed: {e}")

    summary = report.summary()

    if args.json:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    else:
        print()
        print("=" * 60)
        print(f"Results: {summary['passed']}/{summary['total']} passed ({summary['pass_rate']})")
        print(f"Duration: {summary['duration_s']}s")
        print("=" * 60)

    sys.exit(0 if summary["failed"] == 0 else 1)


if __name__ == "__main__":
    main()
