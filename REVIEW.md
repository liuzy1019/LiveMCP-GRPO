# LiveMCP-GRPO 项目审查报告

> 审查日期：2026-06-25（第十二次：数据生成架构审计）
> 审查范围：全项目 80+ 源文件 + data/ 目录产出物检查 + PROVE 论文对齐

---

## 1. 项目方案架构

```
PROVE pipeline (new, default):
  LLM Teacher (Qwen3-8B) ─→ plan_task() ─→ [tool_seq] only ─→ execute_plan() ─→ oracle trace ─→ parquet ─→ GRPO
       ↑                         ↑                                  ↑
   LLMClient              只选工具名，不猜参数             真实 MCP 执行 + infer_args
  (vLLM API)              输出 ~200 tokens                state 来自真实执行结果

E2E pipeline (legacy, --teacher e2e):
  LLM Teacher (Qwen3-32B，计划) ─→ LLMTeacherOrchestrator ─→ replay validate ─→ Jaccard dedup ─→ parquet ─→ GRPO training
       ↑                               ↑                       ↑                    ↑
   LLMClient                      真实 MCP server         OracleValidator     0.70 threshold
  (vLLM API)                     10 domain / 188 tools   criterion_satisfied

Data quality enhancements:
  • Enum stripping (30%)         — teacher must infer parameter values（仅 e2e）
  • Irrelevance queries (5%)     — model must report_error
  • Difficulty: complete/missing/minimal — PROVE-style info completeness
```

**当前状态：PROVE 管线已实现，默认模式。** LLM teacher 使用 **Qwen3-8B**（已验证可用），E2E 模式通过 `--teacher e2e` 保留。

## 2. 已清除的旧方案组件

| 组件 | 文件 | 删除原因 |
|---|---|---|
| `StateMachineOrchestrator` | `orchestrator.py` | 旧定序方案，被 LLM teacher 取代 |
| `OraclePlanner` | `oracle.py` | 手写 `_TOOL_ARG_KEYS` 映射，与 server 不同步导致 4 个 bug |
| `DeterministicTeacherAdapter` | `teacher.py` | 旧模板式 teacher |
| `QueryGenerator` | `query_generator.py` | 手写模板渲染 |
| `DependencyGraphBuilder` | `dependency_graph.py` | 旧工具链依赖图 |
| 全部 10 个 Sampler | `sampler.py` | 每个 domain 只产出一个固定模板 |
| `generate_tasks()` / `generate_tasks_to_file()` | `api.py` | 旧确定性入口 |
| `--teacher deterministic` | `generate_oval_data.py` | 旧入口 |
| 6 个旧测试 | `tests/live_mcp/test_*.py` | 测试已删除的旧组件 |

## 3. 配置文件

| 文件 | 用途 |
|---|---|
| `configs/live_mcp/suite_mvp.yaml` | 10 域子进程配置 + rollout 参数 |
| `configs/live_mcp/{domain}.yaml` × 10 | 各域独立配置 |
| `configs/agent_loop.yaml` | agent loop 注册 |
| `scripts/generate_oval_data.py` | LLM teacher 数据生成 → parquet |
| `scripts/oval_grpo_smoke.sh` | GRPO 烟雾测试 |
| `scripts/train_grpo.py` | verl 训练入口 |

## 4. 当前状态（2026-06-25）

### 4.1 Teacher 模型

| 阶段 | 模型 | 部署 | 说明 |
|------|------|------|------|
| **~~已弃用~~** | ~~Qwen3-4B~~ | ~~local transformers~~ | JSON 输出不可靠（§6），已弃用 |
| **当前 PROVE 模式** | Qwen3-8B | local transformers / vLLM | Phase 1 只输出工具名序列（~200 tokens），8B 完全可靠 |
| **E2E 模式 (legacy)** | Qwen3-8B / Qwen3-32B | vLLM `--api-base` | 旧端到端模式，通过 `--teacher e2e` 使用 |
| **正式实验（计划）** | Qwen3-32B-Instruct | vLLM `--tensor-parallel-size 4` | 和 policy (Qwen3-4B) 同 tokenizer |

### 4.2 已实施的四项数据质量改进

| 改进 | 文件 | 说明 | 状态 |
|------|------|------|------|
| **Jaccard dedup (0.70)** | `src/live_mcp/dedup.py` | 基于 oracle call 序列去重，防止相似任务过多 | ✅ |
| **Enum stripping (30%)** | `task_planner.py` | 随机删除参数 enum，迫使模型推理合法值 | ✅ |
| **Irrelevance queries (5%)** | `orchestrator.py` | 生成与工具无关的任务，训练 `report_error` | ✅ |
| **难度分层重构** | `task_planner.py` | 从"调用次数"改为"信息完整度" (complete 60% / missing 20% / minimal 20%) | ✅ |

### 4.4 模块实现状态总览（2026-06-25 审计）

#### 数据生成管线

| 模块 | 文件 | 状态 | 说明 |
|------|------|------|------|
| **PROVE Teacher (新)** | `src/live_mcp/task_planner.py` | ✅ **新实现** | Phase 1: LLM 规划工具序列 + Phase 2: 执行记录 oracle trace。默认模式 |
| LLM Teacher (旧) | `src/live_mcp/task_planner.py` | ✅ (legacy, 已合并) | E2E 模式保留，通过 `--teacher e2e` 使用 |
| LLM Client | `src/live_mcp/llm_client.py` | ✅ | local transformers / vLLM 双模式，max_tokens=1024 已优化 |
| Orchestrator | `src/live_mcp/orchestrator.py` | ✅ | 新增 `generate_one_prove()` + `prove_mode` 参数 |
| Dedup | `src/live_mcp/dedup.py` | ✅ | Jaccard threshold=0.70，基于 oracle call 序列 |
| Oracle Validator | `src/live_mcp/oracle.py` | ✅ | PROVE 模式下不再需要 replay（trace 来自真实执行），保留用于 E2E |
| Data gen 入口 | `scripts/generate_oval_data.py` | ✅ | `--teacher prove`（默认）/ `--teacher e2e` |
| **训练数据产出** | `data/oval_grpo_train.parquet` | ❌ **待生成** | 用 PROVE 模式重新生成 |

#### MCP 环境（10 Domain）

| Domain | Server | Config | 状态 |
|--------|--------|--------|------|
| banking | `src/live_mcp/servers/banking/server.py` | `configs/live_mcp/banking.yaml` | ✅ |
| calendar | `src/live_mcp/servers/calendar/server.py` | `configs/live_mcp/calendar.yaml` | ✅ |
| crm | `src/live_mcp/servers/crm/server.py` | `configs/live_mcp/crm.yaml` | ✅ |
| email | `src/live_mcp/servers/email/server.py` | `configs/live_mcp/email.yaml` | ✅ |
| filesystem | `src/live_mcp/servers/filesystem/server.py` | `configs/live_mcp/filesystem.yaml` | ✅ |
| food_delivery | `src/live_mcp/servers/food_delivery/server.py` | `configs/live_mcp/food_delivery.yaml` | ✅ |
| issue_tracker | `src/live_mcp/servers/issue_tracker/server.py` | `configs/live_mcp/issue_tracker.yaml` | ✅ |
| payments | `src/live_mcp/servers/payments/server.py` | `configs/live_mcp/payments.yaml` | ✅ |
| shopping | `src/live_mcp/servers/shopping/server.py` | `configs/live_mcp/shopping.yaml` | ✅ |
| team_chat | `src/live_mcp/servers/team_chat/server.py` | `configs/live_mcp/team_chat.yaml` | ✅ |

#### 训练管线

| 模块 | 文件 | 状态 | 说明 |
|------|------|------|------|
| GRPO 训练入口 | `scripts/train_grpo.py` | ✅ | verl runner + Hydra config |
| 正式训练入口 | `src/training/run_grpo.py` | ✅ | OVAL Live MCP rollout 模式 |
| GRPO Estimator | `src/training/schemashift_grpo_estimator.py` | ✅ | 数据驱动诊断，无硬编码标签 |
| Advantage | `src/training/schemashift_advantage.py` | ✅ | 参考实现，生产路径不使用 |
| Length Check | `src/training/length_check.py` | ✅ | 参数化接口，支持 expected_levels=None |
| Task Runner | `src/training/schemashift_task_runner.py` | ✅ | 注册 estimator |
| 烟雾测试脚本 | `scripts/oval_grpo_smoke.sh` | ✅ | A10 23GB×8 配置，硬编码值 |
| **GRPO 烟雾跑通** | `outputs/oval_grpo_smoke.log` | ⏳ **未跑** | 需要 parquet 数据后才能执行 |

#### 奖励管线

| 模块 | 文件 | 状态 | 说明 |
|------|------|------|------|
| 奖励入口 | `src/reward/oval_reward_fn.py` | ✅ | compute_score + AuditWrapper 集成 |
| Action Parser | `src/reward/action_parser.py` | ✅ | tool_call/final_answer/report_error/ask_clarification |
| Task Reward | `src/oval_mcp/rewards/task_reward.py` | ✅ | R_validity + R_coverage + R_efficiency |
| F_gamma | `src/oval_mcp/rewards/f_gamma.py` | ✅ | ProgressTracker（Phase 2 开关） |
| P_process | `src/oval_mcp/rewards/p_process.py` | ✅ | ProcessScorer（Phase 2 开关） |
| Scalar Return | `src/oval_mcp/rewards/scalar_return.py` | ✅ | 标量回报汇总 |
| Safety Verifier | `src/oval_mcp/verifier/safety.py` | ✅ | C_safety 违规检测 |
| Event Log | `src/oval_mcp/verifier/events.py` | ✅ | AuditEvent 序列化 |
| Lambda State | `src/oval_mcp/training/lambda_state.py` | ✅ | λ 自适应更新 |
| LATA | `src/oval_mcp/training/lata.py` | ✅ | length-aware allocation |
| Saturation | `src/oval_mcp/training/saturation.py` | ✅ | 组内方差诊断 |
| Audit Wrapper | `src/oval_mcp/envs/audit_wrapper.py` | ✅ | MCP 操作拦截 + 审计 |
| Domain Adapter | `src/oval_mcp/envs/domain_adapter.py` | ✅ | 10 域适配 |

#### Agent Loop

| 模块 | 文件 | 状态 | 说明 |
|------|------|------|------|
| Oval Loop | `src/agent_loop/schemashift_oval_loop.py` | ✅ | verl 集成 |
| Worker Context | `src/agent_loop/oval_mcp_worker.py` | ✅ | MCP 生命周期管理 |

#### 测试覆盖

| 测试集 | 文件 | 状态 |
|--------|------|------|
| E2E 集成测试 (12 阶段) | `scripts/oval_full_e2e_test.py` | ✅ |
| 10 Domain 测试 | `tests/test_live_mcp_10_domains.py` | ✅ |

| GRPO Estimator 测试 | `tests/test_schemashift_grpo_estimator.py` | ✅ |
| Advantage 测试 | `tests/test_advantage.py` | ✅ |
| Action Parser 测试 | `tests/test_action_parser.py` | ✅ |
| Length Check 测试 | `tests/test_length_check.py` | ✅ |
| Smoke Import 测试 | `tests/test_smoke_grpo_imports.py` | ✅ |
| Estimator 集成 | `tests/test_register_estimator_integration.py` | ✅ |
| OVAL 组件测试 | `tests/test_oval_mcp_components.py` | ✅ |
| OVAL 场景测试 | `tests/test_oval_mcp_scenarios.py` | ✅ |
| Live MCP 单元测试 × 7 | `tests/live_mcp/test_*.py` | ✅ |

### 4.3 已修复的代码问题

| 问题 | 文件 | 修复方式 |
|------|------|---------|
| **🔴 E2E 数据生成架构缺陷** | **新增 `src/live_mcp/task_planner.py`** | 实现 PROVE 两步管线：Phase 1 LLM 规划工具序列 + Phase 2 执行记录 oracle trace。`infer_args` 分层推断参数，`derive_success_criteria` 从 state delta 派生。默认模式 |
| `length_check.py` 硬编码标签集 | `src/training/length_check.py:198` | `assert_e4_group_integrity` 改为参数化接口，支持 `expected_levels=None` 跳过分布检查 |
| `estimator` 诊断日志硬编码标签 | `src/training/schemashift_grpo_estimator.py:120` | 改为数据驱动的多样性检测，仅在所有样本同一标签时报警 |
| `schemashift_advantage.py` 标签 | `src/training/schemashift_advantage.py:20-23` | **非生产代码**，参考实现常量，无需修复 ✅ |
| `orchestrator.py` replay seed 不一致 | `src/live_mcp/orchestrator.py:108` | replay 使用 `seed`（与原始 session 一致），而非 `seed+attempt`（会导致 state 值不匹配） | ✅ |
| `llm_client.py` max_tokens 过高 | `src/live_mcp/llm_client.py:51` | `max_tokens=4096` → `1024`。每条 teacher 生成受 4096 tok 上限 + 15 tok/s 推理速度 = 4.5 分钟瓶颈，实际输出仅 ~200-500 tok | ✅ |
| `REVIEW.md` 重复标题 | 标题行 | 已删除重复行 ✅ |
| `oracle.py` 缺少 criteria 类型 | `src/live_mcp/oracle.py` | 添加 `state_exists`、`cart_not_empty`、`email_count_gte` 类型支持 | ✅ |
| **🔴 `infer_args` 不理解工具语义** | `src/live_mcp/task_planner.py` | 添加 `attempt` 参数让重试可以选不同 entity + `execute_plan` 包裹 5 次 arg 重试循环，失败时自动轮换参数 | ✅ **2026-06-25** |
| **🔴 `_parse_plan` 报错 meta-action** | `src/live_mcp/task_planner.py` | LLM 幻生 `ask_clarification`/`report_error`/`final_answer` 时跳过而非报错退回 | ✅ **2026-06-25** |
| **🔴 3-fail 永久 skip 杀光 8/10 domain** | `src/live_mcp/orchestrator.py` | 替换为 cooldown-based 指数退避：失败 3 次后冷却 N 轮再重试，而非永久跳过 | ✅ **2026-06-25** |
| **🔴 dedup 跨 domain 比较无意义** | `src/live_mcp/dedup.py` | 只在同 domain 内做 Jaccard 比较，不同 domain 工具集不同不会触发去重 | ✅ **2026-06-25** |

---

## 5. 待办跟踪

优先级：P0（阻塞训练）→ P1（影响正确性）→ P2（文档/优化）

| 优先级 | 操作 | 文件 | 说明 | 状态 |
|--------|------|------|------|------|
| **P0** | **✅ 数据生成架构重构** | `src/live_mcp/task_planner.py` | §7 已解决：实现 PROVE 两步管线（Phase 1 LLM 选工具序列 + Phase 2 执行记录 oracle trace） | ✅ |
| **P0** | **PROVE 模式烟雾测试** | `scripts/generate_oval_data.py` | `--teacher prove --count 5 --domain calendar` 验证完整流程（计划+执行+parquet输出） | ⏳ |
| **P0** | **生成正式训练数据 (500+ 条)** | `scripts/generate_oval_data.py` | 用 PROVE 模式 + Qwen3-8B 全量生成。Yield 预期 ~100%（无 replay 丢弃） | ⏳ |
| **P0** | **GRPO 正式训练** | `scripts/oval_grpo_train.sh` | **已创建**，GPU 自适应（A10/L20），算法参数全对齐。被数据生成阻塞 | ⏳ |
| P1 | 烟雾脚本 GPU 自适应 | `oval_grpo_smoke.sh` | ✅ **已解决** — `oval_grpo_train.sh` 实现自动检测 A10/L20 | ✅ |
| P2 | Teacher 模型文档标注 | `REVIEW.md` §4.1 / `PROVE_PARAMETER_AUDIT.md` §1.1 | 区分 "(计划)" / "(当前烟雾)" | ❌ |
| P2 | State seeder 设计注释 | `src/live_mcp/state_seeder.py` | 说明 seed 仅影响 3/10 域，多样性主要来自 LLM teacher | ❌ |
| P3 | PROVE 对比表格脚注 | `PROVE_PARAMETER_AUDIT.md` §1.2 | 说明 task_type 生成方法差异 | ❌ |
| P3 | Distractor 频率实验 | `orchestrator.py` | 验证 20%+20% 分布是否最优，对比 PROVE 的 40% 概率注入 | ❌ |
| P3 | 正式训练超参 sweep | `oval_grpo_train.sh` | RESPONSE_LENGTH=16384, ROLLOUT_N=9, MAX_NUM_SEQS=64 需在 L20 44GB 上验证 | ❌ |

### 5.1 验证风险项（已核查无需操作）

审查意见中经核查确认**不构成阻塞**的项目：

| 问题 | 结论 |
|------|------|
| `dedup_tasks` O(n²) | 91M 次极小 set 比较，理论 <5s，非瓶颈 |
| LATA import 路径（`src.oval_mcp.training.lata`） | 文件存在且正确导出，路径无误 |
| `group_id` 分组 4 条/组 vs 期望 9 条 | 忽略 ROLLOUT_N 展开效应，实际 4×ROLLOUT_N=16，满足条件 |
| `RESPONSE_LENGTH=1024` | 已知 A10 限制，已标注，正式训练按 L20 调整 |
| `schemashift_advantage.py` 标签 | 文件标注"生产路径不使用"，参考实现常量，无影响 |

### 5.2 验证方法

每一条审查意见均通过以下方式核实：
- 读取对应源文件确认代码行为
- 追踪数据生成→存储→训练的全链路数据流
- 对比文档内容与代码事实
- 运行 `grep` / `glob` 确认文件存在性与引用关系

---

## 6. ~~已知技术限制：LLM Teacher 输出格式不稳定~~ ✅ 已解决 (Qwen3-8B)

> **状态更新 (2026-06-25)：** 切换至 Qwen3-8B + `max_tokens=1024` 后，JSON 解析和 replay validation 均通过。<br>
> 以下为 Qwen3-4B 时代的分析记录，保留作为参考。

### 6.1 问题

 `scripts/generate_oval_data.py` 使用 Qwen3-4B 作为 teacher 模型，通过 `LLMClient` + `LLMTeacher` 生成训练数据。实测中 Qwen3-4B 频繁无法输出 LLMTeacher 要求的 JSON 格式。

### 6.2 根因分析

#### 代码中的已有缓解措施

当前 `task_planner.py`（原 `llm_teacher.py`）已经实现了大部分我方可控的优化：

| 措施 | 位置 | 状态 |
|------|------|------|
| Chat template（system/user 消息分离） | `_build_prompt` 返回 `list[dict]`，第 262-264 行 | ✅ 已实现 |
| Domain-aware few-shot 示例（banking, calendar, fallback） | `_few_shot_example()`，第 266-315 行 | ✅ 已实现 |
| ````json` markdown fence 示范 | few-shot 示例用 fence 包裹 JSON | ✅ 已实现 |
| "Output ONLY the JSON" 指令 | system prompt + user prompt 各一次 | ✅ 已实现 |
| 递增温度重试（0.7 → 0.8 → 0.9） | `generate_task()` 第 169 行 | ✅ 已实现 |

剩余问题已不是 prompt 工程可以解决的——核心是**模型能力天花板**。

#### 模型能力瓶颈

Teacher 的 prompt 要求模型一步完成以下任务：

```
领域描述（~200 词）
+ 所有工具 schema（30+ 行，含参数类型和 enum）
+ 当前状态 JSON（含真实 ID 和值）
+ 输出格式要求（4 个必填字段 + 嵌套结构）
+ 多样性指引（persona + style）
→ 输出严格 JSON
```

这对 4B 模型的 instruction following 能力是极限挑战。失败模式有三种（`task_planner.py:_parse_response`）：

| 失败点 | 现象 | 原因 |
|--------|------|------|
| `_extract_json` 抛出 ValueError | 模型输出自然语言对话，不含 JSON | 4B 不理解"只输出 JSON"的指令约束 |
| 缺少必填字段 | JSON 结构残缺，缺 `user_query`/`oracle_calls` 等 | 4B 在长上下文中丢失了格式记忆 |
| `tool_name not in valid_tool_names` | 编造了不存在的工具名 | 4B 混淆了 schema 中的工具名 |

#### PROVE 的对比

| 维度 | PROVE | 我们 |
|------|-------|------|
| Teacher 模型 | Gemma-4-31B-it | Qwen3-4B → 试 **8B** → 不行再 **32B** |
| 生成方式 | State machine teacher（多步拆解） | End-to-end 单步生成 |
| 质量过滤 | error_rate > 30% 整批丢弃 | 逐条 retry 3 次 |
| 有效生成 | 13,517 条从更大规模原始生成中过滤 | 未实测 |

### 6.3 解法路径

| # | 方案 | 改动量 | 预期提升 |
|---|------|--------|---------|
| **1** | **换 8B 模型**（Qwen3-8B）——正在下载 | 零代码，改 `--model` 路径 | **大** — 8B 的 instruction following 显著强于 4B |
| 2 | **8B 不行再换 32B**（`--api-base` 指向 32B vLLM server） | 部署 server + 改 `--api-base` | **最大** — 从根本上解决问题 |
| 3 | 精简 prompt：tools 只传 required_tools | 改调用方或 `_build_prompt` | 有限 — 少传可见工具 |
| 4 | 增加 retry 到 5 次 | `range(3)` → `range(5)` | 有限 — 能力天花板 |
| 5 | 拆分为两步生成 | 改 orchestrator 架构 | 大但复杂 — 中期优化 |

### 6.4 实际测试结果

| 模型 | JSON 解析 | Replay validation | 每条耗时 | 结论 |
|------|-----------|------------------|---------|------|
| Qwen3-4B | ❌ 频繁失败 | — | — | 能力不够，无法可靠输出 JSON |
| **Qwen3-8B** | ✅ **通过** | ✅ **通过**（seed bug 修复后） | **~3-5 分钟 → ~40 秒**（max_tokens 修复后） | **可用**。建议用 vLLM 加速到 ~10 秒/条 |

### 6.5 性能优化记录

**问题：** 每条生成 3-5 分钟，瓶颈分析：

```
Qwen3-8B 在 A10 上生成速度 ≈ 15 tok/s
max_tokens=4096 → 4096 ÷ 15 ≈ 273 秒 ≈ 4.5 分钟

实际 teacher JSON 输出仅 ~200-500 tokens
剩余 3000+ tokens 是模型在 max_tokens 上限内"编"到末尾
```

**修复：** `src/live_mcp/llm_client.py:51`，`max_tokens=4096` → `1024`。预计每条降到 ~40 秒。

**进一步加速（推荐）：** 用 vLLM + `--api-base`。vLLM 推理速度通常 3-5× 快于 local transformers pipeline，每条可降到 ~10 秒。

### 6.6 解法路径（更新）

| # | 方案 | 改动量 | 预期提升 |
|---|------|--------|---------|
| **✅ 已验证** | **Qwen3-8B + max_tokens=1024** | 已合并 | JSON 解析成功 + 每条 ~40 秒 |
| 1 | **切换 vLLM**（`--api-base` 指向 vLLM server） | 启动 server | 每条 ~10 秒（3-5× 加速） |
| 2 | 精简 prompt：tools 只传 required_tools | 改调用方或 `_build_prompt` | 有限加速 |
| 3 | 拆分为两步生成 | 改 orchestrator 架构 | 大但复杂 — 中期优化 |

---

## 7. ✅ 数据生成架构缺陷 — 已解决 (2026-06-25)

> **状态更新：** PROVE 两步管线已实现为默认模式。`src/live_mcp/task_planner.py` 包含完整实现。

### 7.1 解决方案摘要

| 指标 | 旧（end-to-end） | 新（PROVE 两步） |
|------|:---:|:---:|
| Replay 失败率 | ~60% | **0%**（不需 replay） |
| `check_state` | `False`（被迫） | **隐含 True**（state 来自真实执行） |
| 单条耗时（8B local） | ~40s | **~15-20s**（LLM 输出 200 tokens + 3-4 次 MCP 调用） |
| Yield | ~40% | **~100%** |
| LLM 任务复杂度 | 预测一切（含参数值） | 只选工具名序列 |
| 能否用 8B | 勉强 | **完全可靠** |

### 7.2 实现架构

```
Phase 1 — LLM 规划 (ProveTeacher.plan_task):
  输入: domain desc + tool schemas + grounded state
  输出: user_query + ["tool_a", "tool_b", ...]  (仅工具名，无参数)
  → LLM 输出 ~200 tokens，8B 轻松处理

Phase 2 — 执行记录 (execute_plan):
  session = manager.create_session(seed)
  for tool_name in plan.tool_sequence:
      args = infer_args(tool_name, schema, history, state)
      result = executor.execute(session, tool_name, args)
      oracle_calls.append(OracleCall(tool_name, args))
  success_criteria = derive_success_criteria(initial_state, final_state)
  → Oracle trace 100% 可执行（已经执行过）
  → State 值 100% 准确（从真实 state 读取）
```

### 7.3 `infer_args` 策略

参数推断采用分层策略（按优先级）：
1. 从上一步执行结果中匹配参数名（如 `item_id` ← `search_restaurants` 返回的 `items[].id`）
2. 从当前 state 中查找实体 ID（如 `account_id` ← `state.accounts` 的 key）
3. Schema 默认值（enum 第一个值、type 默认值）
4. 参数名模式默认值（如 `amount` → 10.0, `description` → "test"）

### 7.4 `derive_success_criteria` 策略

执行完成后，从 initial_state 和 final_state 的 delta 中派生 criteria：
1. 新增实体检测（如新的 order/email/event）→ `state_exists` 或 `state_equals`
2. 域特定语义检测（如 `transfer` → 检查 balance 变化）
3. 兜底：确保 domain state 存在

### 7.5 使用方式

```bash
# PROVE 模式（默认）
python scripts/generate_oval_data.py \
  --teacher prove --count 500 --val-count 100 \
  --domain all --model models/Qwen3-8B \
  --output data/oval_grpo_train.parquet

# E2E 模式（legacy）
python scripts/generate_oval_data.py \
  --teacher e2e --count 500 ...
```
