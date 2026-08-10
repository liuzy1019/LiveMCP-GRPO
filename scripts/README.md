# scripts/

本目录只放项目级命令入口。可复用的数据质量、fingerprint、Jaccard 和 Parquet 逻辑应放入
`src/`，脚本只负责参数解析、编排和报告。

## 唯一数据生成入口

所有正式生成、缺口规划、恢复、合并和 re-certification 都从以下入口调用：

```bash
livemcp-gen <command> [options]
# 或
python -m src.live_mcp.corpus.cli <command> [options]
```

支持的 command：

| Command | 职责 |
|---|---|
| `plan` | 基于不可变 corpus 计算逐 domain、三桶和 MCP 链长缺口；默认不启动 GPU |
| `run` | 启动 full 或 provenance-isolated supplement 生成 |
| `resume` | 从保留的 candidate/checkpoint 目录恢复 |
| `finalize` | base-priority 合并增量数据并完成全局审计 |
| `profile` | 从完整 corpus 构建训练 composition view |
| `recertify` | 在当前 runtime/reward 合同下重新认证旧 artifact |
| `build-cache` | 构建/重建依赖图缓存 |

`src/live_mcp/corpus/launcher.sh` 和 corpus Python 模块属于内部实现，不是第二套用户入口。
不得直接调用内部入口启动正式补产。

`resume` 会先执行无 Teacher 的全局 merge：候选已足够时直接完成；只有仍存在可恢复缺口时
才启动 vLLM 做 top-up。

## 其他项目脚本

| 脚本 | 职责 | 是否写数据 |
|---|---|---|
| `train_grpo.sh` | 唯一正式 GRPO 启动入口 | 写实验目录 |
| `smoke_rollout_reward.sh` | 多 seed 真实 rollout/reward smoke | 写实验目录 |
| `analyze_rollout_rewards.py` | 汇总 rollout/reward 诊断 | 否 |
| `setup_training_env.sh` | 创建或核验 Policy 训练环境 | 会修改指定环境 |
| `validate_generation_pipeline.py` | dependency cache、schema 和生成管线静态验证 | 否 |
| `audit_prove_domains.py` | 十域 PROVE 准入认证审计 | 否 |
| `audit_tool_semantics.py` | 十域 tool schema/handler 语义审计 | 否 |
| `gpu_config.sh` | 共享 GPU tier 和 vLLM 参数计算 | 否 |
| `wait_for_gpu_quiescence.py` | GPU 显存释放等待 | 否 |

`smoke_rollout_reward.sh` 必须显式传入 `--reward-profile`。PROVE 控制组使用
`prove_baseline`；不得依赖默认值把 `oval_full` 混入 baseline。

## 生成模式

普通全域生成：

```bash
python -m src.live_mcp.corpus.cli run \
  --mode full \
  --count 500 \
  --val-count 100
```

MCP-only 补产：

```bash
python -m src.live_mcp.corpus.cli run \
  --mode supplement \
  --base data/runs/<base> \
  --bucket mcp_conversation \
  --net-new <net-new-target> \
  --candidate-budget <candidate-count> \
  --domain <domain-or-all>
```

Missing-function-only 补产：

```bash
python -m src.live_mcp.corpus.cli run \
  --mode supplement \
  --base data/runs/<base> \
  --bucket missing_function \
  --net-new <net-new-target> \
  --candidate-budget <candidate-count> \
  --domain <domain-or-all>
```

CLI 负责固定 bucket 参数；调用者不能自行组合出
`missing-function + tool-required-only`。`internal_abstention_proxy` 当前只可规划和选择，
不能被伪装成 strict external abstention 生成。

正式补产的 `plan` 每次只选择一个 domain 和一个 bucket，并在 launch command 中显式传递
`--domain`。不得把 `plan` 输出改回 `--domain all`。完成一次增量 merge 后，必须用新的不可变
base 再次执行 `plan`；旧 plan 只代表其输入 artifact 的快照，不可跨批复用。

MCP plan 同时报告 `1-2`、`3-5`、`6+` required-call 链长软缺口。链长目标是本地选择约束，
不是 PROVE hard gate；不能通过放宽 replay、provenance 或 Jaccard 来满足链长配额。

## 约束

- 长时间生成必须传递 `--checkpoint-path`，默认每 25 条 accepted task 原子保存一次；
- 候选预算、最终净新增、桶分布和失败率必须分别报告；
- 不得用内存 accepted 数量代替 Parquet 或 merge report；
- 不得降低 fresh replay、provenance、Jaccard 0.70 或训练结构合同来填补配额；
- 运行中任务不做实现重构；先封存产物，再修改内部模块；
- full run 默认不发布；supplement 不允许直接发布，必须先 `finalize`。

## 后续重构边界

生成实现统一放在 `src/live_mcp/corpus/`。脚本层不得互相导入私有 helper；CLI contract、
Parquet round-trip 和各 bucket 参数隔离必须有回归测试。
