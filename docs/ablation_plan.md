# SchemaShift-GRPO 消融实验设计

> 配套 `docs/technical_report.md`，本文件聚焦**消融逻辑**与**实验矩阵**。
> 状态：2026-06-16 — 算法/数据/脚本已就绪，等待 GPU smoke run 与正式跑批。

---

## 1. 实验矩阵

| 实验 | 训练 Schema | Advantage | 测试 Schema | 核心测量 |
|------|-----------|-----------|------------|----------|
| E1 | 无训练 | — | 原版 + 变体 | 零样本下限 |
| E2 | 原版（SFT） | — | 原版 + 变体 | SFT 泛化 baseline |
| E3 | 原版 | 标准 GRPO（per-prompt 内 z-score） | 原版 + 变体 | **GRPO 是否放大 schema 过拟合** |
| E5 | 混合 1:1:1 prompts，rollout.n=3 | 标准 GRPO（同 `(task, level)` 内 z-score） | 原版 + 变体 | **Schema Aug 单独贡献**（ablation） |
| **E4 (Ours)** | **混合 3:3:3** | **StratAdv（同 task 跨 none/mild/strong 9 rollout 分层 z-score + 全局残差，β=0.25）** | 原版 + 变体 | **联合方案** |
| E6 | 原版 + weight decay/dropout | 标准 GRPO | 原版 + 变体 | 正则化 baseline |

> 消融链：E3 → 加 Schema Aug → E5 → 加 StratAdv → E4。每步只换一个变量，对标 longhorizon 的 Vanilla → PRM-Lite → LATA → Joint。

---

## 2. 数据组织

### 2.1 各实验的 parquet 形态

| 实验 | parquet 路径 | records/task | rollout.n | uid 格式 | 训练侧分组语义 |
|------|------------|-------------|-----------|---------|--------------|
| E2 (SFT) | `data/verl/exp2_sft/` | 1 | — | — | — |
| E3 | `data/verl/exp3_grpo/` | 1 | 9 | 随机（verl 自动生成） | 同 prompt 内 9 条 rollout 做 z-score |
| E4 | `data/verl/exp4_schemashift/` | 9 (3:3:3) | 1 | `{task_id}___{level}` | 同 task 跨 level 的 9 条记录走 SchemaShift estimator 分层 |
| E5 | `data/verl/exp5_aug_only/` | 3 (none/mild/strong 各 1) | 3 | `{task_id}___{level}` | 同 `(task, level)` 内 3 条 rollout 做 z-score |

> E4 训练时要求 `batch_size` 是 9 的倍数 + `data.shuffle=False`，使同 task 的 9 条记录始终落在同一 batch。
> E5 复用 "Schema Aug 信号"但不跨 level 比较，用以分离 Aug 与 StratAdv 的贡献。
> 当前 `scripts/build_parquet.py` 会重建 E2/E3/E4/E5；E5 从 E4 parquet 中每个 task 抽 none/mild/strong 各 1 条。

### 2.2 完整性检查

`scripts/build_parquet.py` 在写出 E4 parquet 前对 train/val 做 fail-fast 断言：每个 `group_id` 必须 9 行，且 `none/mild/strong` 严格 3:3:3。当前 parquet：

```
train: 8100 rows = 900 groups × 9
val:    900 rows = 100 groups × 9
分布：none/mild/strong = 2700/2700/2700 (train)，300/300/300 (val)
0 train/val task overlap
```

---

## 3. 假设验证

| 假设 | 比较 | 判据 |
|------|------|------|
| H1: GRPO 放大 schema 过拟合 | E3 vs E2 | E3 robustness gap > E2 gap |
| H2: Schema Aug 降低过拟合 | E5 vs E3 | E5 robustness gap < E3 gap |
| H3: StratAdv 进一步提升 | E4 vs E5 | E4 robustness gap < E5 gap |
| H4: 联合方案 > 最优基线 | E4 vs max(E3, E6) | E4 综合指标最优（pass@1(none) 不退化前提下，pass@1(strong) ↑、gap ↓） |

> **Gap 不能单独作为"越小越好"的指标**：如果模型整体能力下降，none 和 strong 都低，gap 也可能变小。结论必须以 `pass@1(none)` 不显著退化为前提。

---

## 4. 主要指标

| 指标 | 计算 | 用途 |
|------|------|------|
| `pass@1(none)` | 原版 schema 下 pass@1 | 能力 guardrail |
| `pass@1(mild/strong)` | 各扰动强度下 pass@1 | 鲁棒性主指标 |
| `robust_avg` | mean(none, mild, strong) | 综合鲁棒性 |
| `robustness_gap` | `pass@1(none) − pass@1(strong)` | 抗扰动衰减 |
| `relative_gap` | `gap / pass@1(none)` | 归一化衰减 |
| `macro_overall_pass@1` | 五个 BFCL 多轮子类等权平均 | 总体能力 |
| `seen_pass@1 / unseen_pass@1` | 训练集可见性分组 | 泛化能力 |
| `error:wrong_function / wrong_argument / extra_call / missing_call / wrong_order` | offline eval 错误归因 | 行为诊断 |

> 当前评估为 **response-based AST 匹配**（轨迹层），不是 BFCL 官方 state-based。后者需接入真实 BFCL executor，列为后续工作。

---

## 5. Pilot 设计与 Go/No-Go

| 项目 | 值 |
|------|-----|
| 任务数 | 30（从 900 个训练集中随机抽样） |
| 扰动强度 | strong（最大灵敏度） |
| Seeds | 2 |
| 预算 | ~20 A100-hours（30 task × 2 seeds × 5 experiments） |

| 结果 | 判定 | 行动 |
|------|------|------|
| GRPO gap ≥ SFT gap + 3% | ✅ Go | RL 确实放大 overfit，按原方案执行 |
| \|GRPO gap − SFT gap\| ≤ 2% | ⚠️ 弱 Go | 转向"负结果"叙事：RL 对 schema 鲁棒性无显著影响 |
| GRPO gap < SFT gap − 3% | 🔄 调整 | RL 反而提升鲁棒性！检查扰动强度，加强后重试 |
| 所有模型 gap < 2% | ❌ No-go | 扰动太弱或 schema 格式不影响该模型，换扰动方式 |

Pilot 直接跑完整 300 步（1.5B 模型，30 task × 300 step ≈ 3h），每 50 步评估一次，不做缩减版本。

---

## 6. 计算量预算

| 阶段 | 配置 | GPU-hours |
|------|------|-----------|
| Pilot | 30 task × 2 seeds × 5 experiments | ~20 |
| 全量 | 900 task × 3 seeds × 6 experiments | ~146.5 |
| **总计** | | **~166.5** |

硬件：1×A100-80GB，Qwen2.5-1.5B-Instruct，300 steps/run。

---

## 7. 现状（2026-06-16）

| 维度 | 状态 |
|------|------|
| 算法/公式 | ✅ 收敛，代码-文档一致 |
| 数据 | ✅ E2/E3/E4/E5 均由 `build_parquet.py` 重建；E5 从 E4 parquet 抽样生成 |
| 训练入口 | ✅ E2/E3/E4/E5 脚本就绪，`run_exp4.py` 语法已修复 |
| 单元/集成测试 | ✅ 76/76 pass（含 `register_estimator` patch 集成测试 + reward/eval extra-turn 回归）；`test_core.py` 5/5 smoke pass |
| GPU smoke run | ⏳ 待执行（E2/E3/E4/E5 各 1 step）；E4 启动前可设 `SCHEMASHIFT_PRECHECK=1` 跳过训练仅检查加载期 group 完整性 |
| BFCL state-based eval | ⏳ 仅 response-based AST，state-based 待接入 executor |
