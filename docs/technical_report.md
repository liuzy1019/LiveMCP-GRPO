# SchemaShift-GRPO: 技术方案报告（定稿）

> 面向 MCP Tool Schema 鲁棒性的 GRPO 优化
> 定稿版本：V2 | 2026-06-11 | Codex Review 通过

---

## 📋 解决方案：4 个 Codex 阻塞点修复

### 修复 1：BFCL 评估兼容性（R1.3）

**背景**：BFCL 官方多轮评估分为 state-based（执行后检查系统状态）和 response-based（检查函数调用轨迹）两种。本项目本地评估**当前不接入 BFCL executor**（环境中不可用），所以仅运行 **response-based AST 匹配**：模型输出的函数调用 vs ground truth 调用轨迹，不验证最终执行状态。schema 扰动改了工具名与枚举值，需要在评估时将模型输出反映射回原始调用以与 GT 匹配。

> 未来如需复现 BFCL 官方结果，需要接入真实的 BFCL executor 并仅使用官方 state-based eval pipeline，这是后续工作。

**解决方案**：

- **参数名不扰动**：只扰动工具名（T 类）、描述（D 类）和枚举值（E 类），避开参数名匹配问题。注意：E 类扰动需要额外验证 enum value reverse map 完整性（见 ablation_plan.md §8）
- **评估时反映射**：offline eval 在模型输出后使用 `name_map` / `enum_map` 反向映射回原始 API 名，再与原始 GT 做 AST 匹配。训练时同样使用轻量映射层将扰动名转为原始 API 名，以确保环境调用正确。
- **训练 reward 与 offline eval 共享 AST 匹配逻辑**：避免 train-eval mismatch（`src/eval/matching.py`）。

```
训练流程（response-based reward）：
  扰动 schema (name="find_flights") → 模型输出 "find_flights"
  → 映射层 "find_flights" → "search_flights"
  → BFCL_EXECUTOR 可用时执行环境调用，不可用时走 stub（包含在 environment_check 中）
  → 轨迹 vs GT 做 AST 匹配 → reward ∈ {0.0, 1.0}

评估流程（response-based AST eval）：
  扰动 schema (name="find_flights") → 模型输出 "find_flights"
  → 映射层 反映射 → 原始 GT (name="search_flights") 做 AST 匹配
```

**依据**：UnifiedToolHub 代码确认 BFCL response-based eval 是纯文本比较。BFCL executor 是可选模块，本仓库 fallback 到 stub 使理论流程可跑。

---

### 修复 2：Group Size 分配（R1.1）

**问题**：原方案 2:2:2:2（N=8）每层只有 2 个 rollout，σ 统计量不稳定（2 个样本的标准差不可靠）。

**解决方案**：

- 扰动强度从 4 档降为 3 档：**无 (none)、轻度 (mild)、重度 (strong)**
- Group size 从 N=8 调整为 N=9：**3:3:3**（每层 3 个 rollout）
- 去掉"moderate"档是为了保证每层有足够样本量

```
Group 构成（N=9）：
  none:    3 rollouts (原版 schema)
  mild:    3 rollouts (工具名同义替换)
  strong:  3 rollouts (工具名替换 + 描述改写 + 枚举值替换)
```

**好处**：每层 3 个 rollout → σ 估计更稳定；3 档强度提供足够区分度。

---

### 修复 3：SFT 数据源（R3.3）

**问题**：BFCL V3 只有测试集（1000 个多轮任务），没有官方 SFT 训练分割。

**解决方案**：使用 BFCL V2（单轮）和 V3 的少量示例做 SFT，确保与 V3 评测集不重叠。

| 数据源 | 用途 | 条数 | 是否与评测重叠 |
|--------|------|------|--------------|
| BFCL V2 (simple) | SFT 训练 | ~1,500 | ❌ 单轮 vs 多轮，无重叠 |
| BFCL V2 (multiple) | SFT 训练 | ~1,000 | ❌ 同上 |
| BFCL V3 10% (held-out split) | SFT 验证 | ~100 | 随机的 10% 从多轮训练集中抽出，只用于验证 |
| **BFCL V3 90%** | **GRPO 训练 + 评测** | **~900** | **✅ 不出现在 SFT 中** |

**为什么需要 SFT 基线**：SFT 基线（E2）用于区分"预训练模型的固有能力"和"RL 训练导致的过拟合"。如果 GRPO 的 robustness gap 大于 SFT 的 gap，说明 RL 放大 schema 过拟合。

**关于 BFCL V4（2025-07-17 发布）**：V4 新增 Agentic 域（Web Search / Memory Management）并调整了 leaderboard 权重（Agentic 40%、Multi-Turn 30%），但其多轮测试数据与 V3 完全相同（HuggingFace 数据集 repo 中无 `v4_multi_turn_*` 文件，V4 复用 V3 数据）。我们不迁移到 V4 的原因：(1) 多轮底层数据没变，迁移无收益；(2) V4 新增的 Agentic 域与 schema perturbation robustness 研究问题无关；(3) 如需提交 leaderboard，只需按 V4 目录结构组织 result 文件即可，不影响训练和评估代码。

---

### 修复 4：Pilot 实验 go/no-go 标准（R5.1）

**Pilot 设计**：

| 项目 | 值 |
|------|-----|
| 任务数 | 30（从 900 个训练集中随机抽样） |
| 扰动强度 | strong（保证最大检测灵敏度） |
| Seeds | 2 |
| 总计算量 | ~20 A100-hours（30 task × 2 seeds × 5 experiments，详见 ablation_plan.md§7） |
| 评估指标 | robustness gap（原版 pass@1 - 变体 pass@1） |

**Go/No-Go 标准**：

| 结果 | 判定 | 行动 |
|------|------|------|
| GRPO gap ≥ SFT gap + 3% | ✅ **Go** | RL 确实放大 overfit，按原方案执行 |
| GRPO gap ≈ SFT gap (±2%) | ⚠️ **弱 Go** | RL 不放大也不缩小，转向"负结果"叙事：RL 对 schema 鲁棒性无显著影响 |
| GRPO gap < SFT gap - 3% | 🔄 **调整** | RL 反而提升鲁棒性！检查扰动强度是否足够，加强后重试 |
| 所有模型 gap < 2% | ❌ **No-go** | 扰动太弱或 schema 格式不影响该模型。换用 persuasive text 扰动（参考 Gaming Tool Preferences）或放弃该方向 |

**Pilot 执行**：直接跑完整 300 步训练（模型小，1.5B，30 task × 300 step ≈ 3h），每 50 步评估一次。不浪费算力在缩减版本上。

---

## 🔥 最终实验矩阵（与 ablation_plan.md V4 同步）

| 实验 | 训练 Schema | Advantage | 测试 Schema | 核心测量 |
|------|-----------|-----------|------------|----------|
| E1 | 无训练 | — | 原版 + 变体 | 零样本下限 |
| E2 | 原版 (SFT) | — | 原版 + 变体 | SFT 泛化能力 |
| E3 | 原版 | 标准 GRPO（per-prompt 内 z-score） | 原版 + 变体 | **GRPO 是否放大 overfit** |
| E5 | 混合（none/mild/strong 各 1 prompt，rollout.n=3） | 标准 GRPO（同 (task, level) 内 z-score） | 原版 + 变体 | **Schema Aug 单独贡献**（ablation）|
| **E4 (Ours)** | **混合（3:3:3）** | **StratAdv（同 task 跨 level 9 rollout 分层：A = strat_z + 0.25 × global_z）** | 原版 + 变体 | **联合方案** |
| E6 | 原版 + weight decay/dropout | 标准 GRPO | 原版 + 变体 | 正则化基线 vs 本方法 |

> **V4 消融叙事**：E3（基线）→ E5（加 Schema Aug）→ E4（加 StratAdv）。三步两个对比，对标 longhorizon 的 Vanilla → PRM-Lite → LATA → Joint。
>
> **贡献归因**：E5 是必须的 ablation，用于区分 Schema Augmentation 和 Stratified Advantage 的独立贡献。
>
> E5 与 E4 使用同一批 task 和同样的 none/mild/strong 扰动 schema；E5 每 task/level 保留 1 条 prompt 并用 rollout.n=3，E4 每 task 保留 9 条记录并用 rollout.n=1。
>
> - **E5**：标准 GRPO，在**同一 (task, level)** prompt 的 3 条 rollout 间做 z-score（不跨 level 比较）。
> - **E4**：StratAdv，**在同一 task 跨 none/mild/strong** 的 9 条 rollout 上做分层 z-score + 跨层全局残差。
>
> 因此 E4 vs E5 的差异分离的是“**SchemaShift 分层 advantage vs 标准 GRPO 分组方式**”；两者共享扰动 schema，但训练 batch 组织不同。
> 如果 E4 > E5 > E3，说明两个组件各有贡献；如果 E5 ≈ E4，说明 StratAdv 无额外收益。
### 主要关注指标

| 指标 | 计算方式 | 用于 |
|------|---------|------|
| **pass@1(none)** | 原版 schema 下的 pass@1 | 能力不退化的 guardrail |
| **pass@1(strong)** | 强扰动 schema 下的 pass@1 | **主要鲁棒性能指标** |
| **robust_avg** | mean(none, mild, strong) | 综合鲁棒性 |
| **robustness_gap** | pass@1(none) - pass@1(strong) | 抗扰动衰减指标 |
| **relative_gap** | gap / pass@1(none) | 归一化衰减率 |
| **macro_overall_pass@1** | 五个子类等权平均 | 总体能力 |
| **seen_pass@1 / unseen_pass@1** | 按训练集可见性分组 | 泛化能力 |

> **结论标准**：在 pass@1(none) 不明显下降的前提下，pass@1(strong) 越高、robustness_gap 越小，说明 schema robustness 越好。
> gap 不能单独作为"越小越好"的核心指标——如果模型整体能力下降，none 和 strong 都很低，gap 也可能变小，但这不是鲁棒性变好。

### 评估运行方式

| 维度 | 设计 | 对齐参考 repo |
|------|------|-------------|
| **推理模式** | 多轮交互（与训练 agent loop 一致） | ✅ 真实多轮 rollout |
| **采样次数** | 每 task 4 次采样（temperature=0.6） | ✅ 4 samples/task |
| **数据分组** | seen（训练集任务）/ unseen（holdout 任务） | ✅ covered_seen / unseen |
| **匹配逻辑** | AST 宽松匹配（与 online reward 共享） | ✅ train-eval 一致 |
| **行为诊断** | avg_turns / avg_tool_calls / avg_tokens | ✅ 效率指标 |
| **错误分类** | wrong_function / wrong_argument / extra_call / missing_call / wrong_order | ✅ 错误归因 |

> **关键设计**：评估时使用多轮推理（模拟 tool observation），与训练时 agent loop 行为一致，避免 train-eval mismatch。
> 训练 reward 只用于诊断，不作为最终效果依据。独立 eval 才是主结论。

---

## 🏗️ 最终项目结构

```
/Users/liuzhanyi/Desktop/liuzy/llm_repo/schemashift-grpo/
├── README.md                    # 项目概述 + 结果 + 快速开始
├── setup.sh                     # 一键环境配置
├── requirements.txt
├── configs/
│   ├── exp1_zeroshot.yaml
│   ├── exp2_sft.yaml
│   ├── exp3_grpo_baseline.yaml
│   └── exp4_schemashift.yaml
├── src/
│   ├── envs/
│   │   ├── schema_perturber.py  # 扰动生成器（T+D+E）
│   │   ├── bfcl_env.py          # BFCL 多轮环境
│   │   └── api_mapper.py        # 工具名映射层（~30 行）
│   ├── training/
│   │   ├── schemashift_advantage.py
│   │   ├── schemashift_grpo_estimator.py  # 自定义 verl estimator
│   │   └── register_estimator.py         # verl 集成入口
│   ├── agent_loop/
│   │   └── bfcl_agent_loop.py
│   ├── reward/
│   │   └── bfcl_reward.py
│   └── eval/
│       └── bfcl_eval.py
├── scripts/
│   ├── train/grpo/
│   │   ├── run_exp3_grpo_baseline.sh
│   │   └── run_exp4_schemashift.sh
│   ├── train/sft/
│   │   └── run_exp2_sft.sh
│   └── eval/
│       ├── eval_all.sh
│       └── eval_zeroshot.sh
├── docs/
│   ├── technical_report.md      # 本文件
│   ├── ablation_plan.md         # 消融设计
│   └── novelty_check.md         # 查重报告
├── data/                        # 数据目录（.gitignore）
└── experiments/results/         # 实验结果
```

---

## 🚀 实施时间线（更新版）

| 阶段 | 周次 | 关键产出 |
|------|------|---------|
| **P1: 环境搭建** | 1-2 | verl + BFCL 集成通过。BFCL agent loop 可执行一个完整多轮任务 |
| **P2: 工具开发** | 2-3 | SchemaPerturber + ADV 计算 + 映射层。单元测试覆盖 |
| **P3: Pilot** | 3-4 | 30 task × E1/E2/E3/E4 × 2 seeds。确认 robustness gap 方向 |
| **P4: 全量实验** | 4-6 | E1-E7 × 3 seeds（~150 GPU-hours） |
| **P5: 分析+文档** | 6-8 | 数据清洗 -> 可视化 -> README -> 发布 |

---

## ✅ 和参考 repo 的对比（给老板看）

| 维度 | 参考 repo (agentic-grpo-longhorizon) | 我们 (SchemaShift-GRPO) |
|------|--------------------------------------|----------------------|
| 核心发现 | 标准 GRPO 在长任务中训练坍塌 | **标准 GRPO 放大 schema 过拟合** |
| 方法 1 | PRM-Lite：15 条过程奖励规则 | **Schema Augmentation：扰动训练** |
| 方法 2 | LATA：1/√L 归一化 | **分层 Advantage：按扰动强度分组** |
| 基线对比 | Vanilla / Turn-Discount / PRM-Lite / LATA / Joint | 零样本/SFT/GRPO/本方法/增强消融/正则化消融 |
| 核心代码 | ~200 行 | ~330 行 |
| Benchmark | τ-bench（50 任务） | BFCL V3（1000 任务） |
| 硬件需求 | 2×A800 | 1×A100-80GB |
| 总 GPU | ~100 A800-hours | ~200 A100-hours |
| 查重结果 | N/A | 15+ 论文无重叠 ✅ |
| Codex 审核 | N/A | 7/10，4 个问题已修复 ✅ |

---

## 📌 为什么不采纳其他候选方案（决策理由）

| 方案 | 不采纳的原因 |
|------|------------|
| MCP-Strata GRPO | 和 P-GRPO 边界太模糊。虽然机制不同（concurrent vs history），但论文 reviewer 可能认为"就是 P-GRPO 换个应用场景"。简历价值不如 SchemaShift |
| MVAD-GRPO | 依赖 gold path 的 server 级标注。BFCL 数据中没有显式 server 字段，需要推断，增加风险和工程量 |
| RecoveryAdvantage | 需要在 BFCL 上先分析 recovery 频率，"赌"recovery 行为足够普遍。风险高 |
| MCP-EquiTool | TAPO（June 2026）已经覆盖工具等价方向。我们的语义相似度版本面临阈值调参的脆弱性 |
| 其他 6 个 | 均在查重中被覆盖或淘汰（见 `novelty_check.md`） |

**SchemaShift-GRPO 胜出的三点理由**：

1. **独特性最干净**：没有任何现有工作覆盖 RL 训练对 schema 过拟合的研究。Gaming Tool Preferences 只做零样本推理，Trace-Free+ 做描述工程，我们都是不同的。
2. **MCP 绑定最深**：schema 扰动问题是 MCP 协议的特有挑战——不同 MCP server 实现同一功能时 schema 描述必然不同。这个研究直接服务实际部署场景。
3. **故事线完整**：发现问题（RL 放大 overfit）→ 提出方法（扰动训练 + 分层 advantage）→ 验证效果（消融实验），和参考 repo 的结构一致。
