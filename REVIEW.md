# REVIEW.md — LiveMCP-GRPO 项目审查报告

> 最后更新：2026-06-29（第十六轮：P0/P1 修复 + 对抗式全管线审查）
> 当前状态：核心管线可用，500+100 全量数据生成进行中

---

## 1. 第十六轮修复记录（2026-06-28）

| # | 问题 | 严重度 | 修复方式 | 文件 |
|---|------|--------|---------|------|
| 1 | 混合类型 `success_criteria` 无法写入 Parquet | P0 | `success_criteria` 序列化为 JSON 字符串存入 `reward_model.ground_truth`，reward 端 `json.loads` 恢复 | `generate_data.py` + `oval_reward_fn.py` |
| 2 | `success_criteria` 未参与 reward 判断 | P0 | 新增 `_count_completed_state_criteria()` — 验证 `state_equals`/`state_exists`/`file_exists`/`cart_not_empty`/`email_count_gte`，与 operation 断言合并计算 coverage | `task_reward.py` |
| 3 | 澄清任务得 0 分 | P0 | `OracleCall(action="clarification")` 保留到 parquet，`_build_task_dict` 识别后设 `allowed_terminal=["ask_clarification"]`，排除出 `required_tool_calls` | `generate_data.py` + `oval_reward_fn.py` |
| 4 | terminal 审计事件 `execution_success/schema_valid=False` | P1 | `_make_terminal_event` 显式设 `execution_success=True, schema_valid=True` | `audit_wrapper.py` |
| 5 | irrelevant 任务不区分终止方式 | P1 | `scenario_type="irrelevant"` → `allowed_terminal=["report_error"]` | `oval_reward_fn.py` |
| 6 | 生成失败和去重后数量不足不报错 | P1 | `generate_many`：0 产量抛 `RuntimeError`（count>0 时），<50% 产量打 error 日志 | `orchestrator.py` |
| 7 | 去重丢失调用顺序 | P1 | Jaccard 改用 `(position, tool_name, frozenset(k=v))` 位置感知 multiset | `dedup.py` |
| 8 | provenance 检查允许"未来信息" | P1 | 严格按时间线检查——step i 只能看到 step i-1 的 observation | `task_planner.py` |
| 9 | 任务预算被读取但未使用 | P2 | `effective_max_turns = min(self.max_turns, budget_int)` | `livemcp_oval_loop.py` |
| 10 | 并行分片超过请求数量 | P2 | merge 时 `merged.head(target)` 裁剪 | `generate_data.sh` |

---

## 2. 对抗式审查结果（2026-06-29）

全链路逐路径验证，28/28 PASS：

| 验证项 | 方法 | 结果 |
|--------|------|------|
| oracle_calls 混合类型 arguments → Parquet | int+str 两种 task，to_parquet → round-trip | ✅ |
| success_criteria JSON 字符串 round-trip | float 1500.5 + "paid" + state_exists | ✅ |
| `_build_task_dict` 6 种场景全覆盖 | normal / missing / irrelevant / clarification / legacy / 无 criteria | ✅ |
| reward_model 全量 round-trip 后 reward_fn 可消费 | 3 rows → to_parquet → read → `_build_task_dict` | ✅ |
| terminal event `execution_success=True` | 源码确认 `_make_terminal_event` | ✅ |
| `generate_many` 缺量守卫 | 源码确认 `count>0` guard | ✅ |
| `generate_data.sh` shard 裁剪 | 源码确认 `merged.head(target)` | ✅ |
| 非法 tool JSON 分支 | 源码确认：有 error observation，无 audit | ✅ 已确认（设计限制） |

---

## 3. 已知设计限制（非 bug，不阻塞管线）

| # | 问题 | 为什么不修 | 影响 |
|---|------|----------|------|
| A | 非法 tool JSON 不产生 AuditEvent | 需要跨 5+ 文件扩展 AuditEvent 类型，action_type 语义不明确 | 丢失一个 turn 的审计信号，模型仍收到 error observation |
| B | `_count_completed_state_criteria` 用最后 observation 近似 final state | 需要改 agent loop → reward 的 post_state 传递链路 | 连续 tool_call 时精确值验证可能漏检（小概率） |
| C | perturbation 仅 teacher 阶段出现 | 设计选择 — PROVE 扰动用于测试 teacher 鲁棒性，不应注入训练环境 | oracle 可能包含对幻影错误的恢复步骤 |

---

## 4. 当前数据生成状态

```
模型: Qwen3-32B-Instruct (vLLM TP=4, GPU 4-7, 4×L20 44GB)
目标: 500 train + 100 val
最新产出: data/train_v5.parquet + data/val_v5.parquet (200+30, 实验 gen_v5)
状态: P0/P1 bug 修复已完成，全量生成待重跑
```

---

## 5. 模块清单

| 模块 | 文件 | 状态 |
|------|------|------|
| PROVE Teacher | `src/live_mcp/task_planner.py` | ✅ 已验证（calendar domain 100% yield） |
| LLM Client | `src/live_mcp/llm_client.py` | ✅ local/vLLM 双模式 |
| Orchestrator | `src/live_mcp/orchestrator.py` | ✅ 缺量守卫 + retry |
| Dedup | `src/live_mcp/dedup.py` | ✅ 位置感知 Jaccard 0.70 |
| 10 Domain servers | `src/live_mcp/servers/{domain}/` | ✅ |
| 数据生成入口 | `scripts/generate_data.py` | ✅ Parquet 序列化修复 |
| OVAL rollout | `src/agent_loop/livemcp_oval_loop.py` | ✅ budget 修复 |
| Reward fn | `src/reward/oval_reward_fn.py` | ✅ 全部场景覆盖 |
| Task Reward | `src/oval_mcp/rewards/task_reward.py` | ✅ state criteria 验证 |
| Safety Verifier | `src/oval_mcp/verifier/safety.py` | ✅ |
| Audit Wrapper | `src/oval_mcp/envs/audit_wrapper.py` | ✅ terminal event 修复 |
| Lambda State | `src/oval_mcp/training/lambda_state.py` | ✅ |
| GRPO Estimator | `src/training/livemcp_grpo_estimator.py` | ✅ 饱和检测+降级 |

---

## 6. 待办

| 优先级 | 操作 | 说明 |
|--------|------|------|
| P0 | 完成 500+100 数据生成 | 进行中，ETA ~3h |
| P0 | GRPO 训练 smoke test | 需数据生成完成后执行 |
| P3 | 非法 JSON AuditEvent 盲区 | 低优先级，需跨模块类型扩展 |
| P3 | final state 精确 snapshot | 需改 agent loop → reward 接口 |
