# SchemaShift-GRPO：面向 MCP Tool Schema 鲁棒性的 GRPO 训练优化

> 对标 [agentic-grpo-longhorizon](https://github.com/qiqihezh/agentic-grpo-longhorizon) 的问题-方法-验证结构
> 状态（2026-06-16 收尾）：`arl` 已降级至 verl 0.6.1 兼容窗口（torch 2.8 / vllm 0.11 / flash_attn 2.7.3）。4 卡 A10 上 **E3 / E4 / E5 step 1 smoke 全部跑通**：loss 有限、grad_norm finite、ckpt 落盘。正式训练（>1 step）尚未跑。
> 审查：7 轮 Codex 对抗式审查 + 1 轮完整代码 review（R7: 0 P0；2026-06-15 review P0 全部修复；2026-06-16：GPU 验证期间连续修复 5 个运行期 bug 与 1 个 yaml 配置错）

---

## 1. 问题

标准 GRPO 在多轮 function calling 任务上训练时，策略模型过拟合到训练时见过的具体 schema 表面形式（工具名、描述措辞、枚举值命名），遇到 MCP 场景下语义等价但表面形式不同的 schema 变体时性能急剧下降。

```
训练：search_flights(origin="JFK", destination="LHR")  → 正确
测试：find_flights(departure_location="JFK", to="LHR") → 失败
```

### 问题诊断（对标 longhorizon 的三大致命原因）

| 原因 | 说明 |
|------|------|
| **Schema 表面形式过拟合** | GRPO 会放大模型对训练 schema 具体工具名 / 描述措辞 / 枚举值的依赖，换个同义写法就脚 |
| **群组奖励饱和** | 二元 reward（0/1），group_size 较大时容易全 0 或全 1 → advantage 方差→0 → 梯度消失 |
| **扰动强度信号淹没** | 强扰动 schema 下模型更难答对，reward 偏低。标准 GRPO 在 prompt 内做 z-score 会惩罚所有低分 rollout，包括"在困难条件下表现合理"的 |

---

## 2. 方法

### 2.1 两个组件

| 组件 | 机制 | 对应 longhorizon |
|------|------|-----------------|
| **Schema Augmentation** | 训练时混合 3 种扰动强度的 schema 变体（none/mild/strong），让模型见过不同的表面形式 | PRM-Lite（信号源） |
| **Stratified Advantage** | 按扰动强度分层归一化 advantage + 全局残差，A = strat_z + beta × global_z | LATA（信号通路） |

### 2.2 Schema Augmentation

扰动是确定性的（给定 seed），基于内置 DEFAULT_SYNONYM_MAP 的 ~100 条同义规则（82 key）。

```
mild:  tool_name 同义替换
      search_flights → find_flights

strong: tool_name + description + enum 全部替换
      search_flights → find_flights
      "Search for available flights" → "Lookup for accessible flights"
      "economy" → "standard"
```

**代码**：`src/envs/schema_perturber.py:229-234`
```python
LEVEL_MILD:  ["tool_name"],
LEVEL_STRONG: ["tool_name", "description", "enum"],
```

扰动器同时维护 `name_map`（工具名映射）和 `enum_map`（枚举值映射），双向可查，用于训练时的 API dispatch 反向映射和评估时的 ground truth 同步。

**安全性检查**（`_validate_perturbation`）：
- 工具名不冲突（扰动后无同名工具）
- 必需参数字段不变
- 参数类型不变

### 2.3 Stratified Advantage

**公式**：
```
A = strat_z + beta × global_z

strat_z:  同一 task 内，同 level 内独立 z-score（跨 none/mild/strong 分层）
global_z: 同一 task 内，跨 level 全局 z-score（作为跨层信号残差）
beta:     跨层信号强度，默认 0.25
```

beta=0 → 纯分层归一化，beta 很大 → 趋近标准 GRPO。

**代码**：`src/training/schemashift_grpo_estimator.py:84-105`
```python
# Step 1: 同 task 内跨 level 分层 z-score
strat_advs = (level_scores - level_mean) / level_std

# Step 2: 同 task 内跨 level 的全局 z-score 作为残差
global_z = (group_scores - group_mean) / group_std

# Step 3: 加法融合
advantages = strat_advs + beta * global_z
```

**单样本安全**：单层或单组只有 1 个样本时，std 设为 1.0（不做归一化），不产生 NaN。

### 2.4 与 verl 框架的集成

基于对 verl 源码的阅读（`ray_trainer.py:237-265`, `core_algos.py:113-150`）：

- verl 的非 GRPO 路径通过 `adv_kwargs` 调用自定义 estimator，传入 `index(uid)` + `token_level_rewards` + `response_mask` + `config`
- `ADV_ESTIMATOR_REGISTRY` 是普通 dict，`register_adv_est` 装饰器在 import 时注册，`get_adv_estimator_fn` 做 dict 查找
- 我们将 `task_id` 和 `perturbation_level` 编码进 uid：`{task_id}___{level}`
- 当前实现通过 `register_estimator.py` 注册 `schemashift_grpo`，并 monkey-patch `verl.trainer.ppo.core_algos.compute_advantage` 以把 `non_tensor_batch` 传入 estimator

```
parquet                       verl trainer                  estimator
───────                       ────────────                  ─────────
uid = "task_92___none"   →    index = uid array        →    parse uid
uid = "task_92___mild"   →    adv_estimator =           →    group by task_id
uid = "task_92___strong" →      "schemashift_grpo"     →    strat by level
                                                                  ↓
                                                           A = strat_z + beta*global_z
```

### 2.5 函数名 + 枚举值双向映射

`FunctionNameMapper`（`src/envs/api_mapper.py`）同时处理函数名和枚举值的反向映射：
- `map_func_call("find_flights(class='standard')")` → `"search_flights(class='economy')"`
- 训练时：扰动名 → 原始名 → API dispatch 正确
- 评估时：模型输出中的扰动名/枚举值会映射回原始形式，再与 ground truth 匹配

---

## 3. 实验设计

### 3.1 实验矩阵

| 实验 | 训练 Schema | Advantage | 测量 |
|------|-----------|-----------|------|
| E1 | 无训练 | — | 零样本下限 |
| E2 | 原版 (SFT) | — | SFT 泛化 baseline |
| E3 | 原版 | 标准 GRPO（per-prompt 内 z-score） | **GRPO 是否放大 overfit** |
| E5 | 混合 1:1:1 prompts，rollout.n=3 | 标准 GRPO（同 (task, level) 内 z-score） | Aug 单独贡献 |
| **E4** | **混合 3:3:3** | **StratAdv (beta=0.25，同 task 跨 level)** | **联合方案** |
| E6 | 原版 + weight decay/dropout | 标准 GRPO | 正则化 baseline |

### 3.2 消融逻辑

```
E3 (Vanilla GRPO) ──加 Schema Aug──▶ E5 ──加 StratAdv──▶ E4 (Full)
                                          │                    │
                                          └─ Aug 贡献         └─ StratAdv 贡献
                                          (E5 - E3)           (E4 - E5)
```

每步只变一个变量。对标 longhorizon 的 Vanilla → PRM-Lite → LATA → Joint。

### 3.3 假设验证

| 假设 | 验证 | 判据 |
|------|------|------|
| H1: GRPO 放大 schema 过拟合 | E3 vs E1/E2 | E3 robustness gap > E2 gap |
| H2: Schema Aug 降低过拟合 | E5 vs E3 | E5 robustness gap < E3 gap |
| H3: StratAdv 进一步提升 | E4 vs E5 | E4 robustness gap < E5 gap |
| H4: 联合方案 > 最优基线 | E4 vs max(E3,E6) | E4 综合指标最优 |

### 3.4 预期结果表

| 实验 | Robustness Gap ↓ | 原版 pass@1 ↑ | mild pass@1 ↑ | strong pass@1 ↑ |
|------|-----------------|-------------|-------------|----------------|
| E1 (Zero-shot) | ? | ? | ? | ? |
| E2 (SFT) | ? | ? | ? | ? |
| E3 (Vanilla GRPO) | ? | ? | ? | ? |
| E5 (+Aug) | ? | ? | ? | ? |
| **E4 (Full)** | **?** | **?** | **?** | **?** |
| E6 (RegBaseline) | ? | ? | ? | ? |

---

## 4. 数据与工程

### 4.1 数据组织

```
E3: 1 record/task, rollout.n=9, 标准 GRPO 分组
E4: 9 records/task (3 none + 3 mild + 3 strong)
       rollout.n=1, batch_size=18 (9的倍数), shuffle=False
       E4 使用 StratAdv estimator (adv_estimator=schemashift_grpo)
       uid 格式: {task_id}___{level}
E5: 3 records/task (none/mild/strong 各1)
       rollout.n=3, 标准 GRPO 分组（同 (task, level) 内 z-score）
       uid 格式: {task_id}___{level}
```

**当前 parquet 状态**：
- E4 (`exp4_schemashift/`)：train=8100 (900 tasks × 9)、val=900 (100 tasks × 9)，精确 3:3:3 分布（2700/2700/2700），0 train/val overlap，按 task 级别 shuffle（同 task 9 条记录保持相邻）
- E5 (`exp5_aug_only/`)：train=2700、val=300，none/mild/strong 各 1/3。由 `scripts/build_parquet.py` 从 E4 parquet 每 task/level 抽 1 条重建

### 4.2 可运行性

| 实验 | 训练脚本 | 数据 | 状态 |
|------|---------|------|------|
| E1 | eval 脚本 | — | ✅ |
| E2: SFT | `run_exp2_sft.sh` | `exp2_sft/` | ✅ |
| E3 | `run_vanilla_grpo.sh` | `exp3_grpo/` | ✅ step 1 smoke pass（55s/step, max_mem 22.4 GB） |
| E4 | `run_schemashift.sh` → `run_exp4.py` | `exp4_schemashift/` (8100/900, 9 records/task) | ✅ step 1 smoke pass（~13min/step, ~21GB offload） |
| E5 | `run_aug_only.sh` | `exp5_aug_only/` (2700/300, 3 records/task) | ✅ step 1 smoke pass（145s/step, max_mem 18.0 GB） |
| E6 | — | — | ❌ 仅文档 |
### 4.3 测试覆盖

```
105/105 tests pass
  test_advantage.py: 10  — 公式正确性 + 边界条件（单样本、缺层、极端 beta）
  test_api_mapper.py: 9  — 函数名映射 + 枚举值映射 + ground truth 同步
  test_schema_perturber.py: 13 — 扰动生成 + 确定性 + 安全性校验
  test_schemashift_grpo_estimator.py: 11 — estimator 注册 + 分层 advantage + uid fallback
  test_runtime_regressions.py: 19 — parquet、BFCL 解析、reward/eval 运行级回归（含 4 个解析器灾难性输入防卡死测试）
  test_enum_mapping.py: 14 — enum 映射合法性与 reward/eval 路径
  test_register_estimator_integration.py: 4 — register_estimator monkey-patch 真实注入 non_tensor_batch 的集成测试
  test_smoke_grpo_imports.py: 16 — 依赖契约（vllm/torch/trl/flash_attn 版本兜底）
  test_length_check.py: 9 — 数据长度预检（fail-fast / warn / skip env / argv 解析）
```

### 4.4 计算量

| 阶段 | 配置 | GPU-hours |
|------|------|-----------|
| Pilot | 30 task, 2 seeds, 5 experiments | ~20 |
| 全量 | 900 task, 3 seeds, 6 experiments | ~146.5 |
| **总计** | | **~166.5** |

硬件：1×A100-80GB, Qwen2.5-1.5B-Instruct, 300 steps per run.

---

## 5. 与 longhorizon 的结构对照

| 维度 | longhorizon | SchemaShift-GRPO |
|------|-----------|-----------------|
| **问题** | GRPO 在长 horizon 任务训练崩溃（奖励饱和/泄漏偏差/长度惩罚） | GRPO 放大 schema 过拟合（奖励饱和/schema 依赖/扰动信号淹没） |
| **信号源** | PRM-Lite：15 条手工过程奖励规则 | Schema Augmentation：扰动数据增强（~100 条同义规则） |
| **信号通路** | LATA：1/√L 长度感知归一化 | StratAdv：分层归一化 + 全局残差（beta=0.25） |
| **联合方案** | PRM-Lite + LATA | Aug + StratAdv (E4) |
| **对比基线** | Turn-Discounted Advantage | 标准正则化 (E6) |
| **消融结构** | Vanilla → PRM-Lite → LATA → Joint | E3 → E5 → E4 |
| **假设验证** | H1-H5（5 条） | H1-H4（4 条） |
| **Benchmark** | τ-bench（50 任务） | BFCL V3（1000 任务） |
| **模型** | 7B + 72B simulator | Qwen2.5-1.5B-Instruct |
| **框架集成** | verl GRPO trainer patch | verl 自定义 estimator（`@register_adv_est`） |

---

## 6. 项目结构

```
schemashift-grpo/
├── src/
│   ├── envs/
│   │   ├── schema_perturber.py    # 扰动生成器（3 强度 × 3 类型，~380 行）
│   │   ├── api_mapper.py          # 函数名 + 枚举值双向映射（~130 行）
│   │   └── bfcl_env.py            # BFCL 多轮环境
│   ├── training/
│   │   ├── schemashift_grpo_estimator.py  # 自定义 verl estimator
│   │   ├── schemashift_advantage.py       # 独立 advantage 函数（测试用）
│   │   └── register_estimator.py          # verl 集成入口
│   ├── agent_loop/
│   │   └── bfcl_agent_loop.py    # BFCL 多轮 agent loop
│   ├── reward/
│   │   └── bfcl_reward.py        # BFCL 正确性 reward
│   └── eval/
│       └── bfcl_eval.py           # 鲁棒性评估
├── scripts/
│   ├── build_parquet.py           # 数据 → parquet（含 train/val split + uid 编码）
│   ├── generate_perturbations.py  # 预生成扰动
│   └── train/
│       ├── sft/
│       │   ├── run_exp2_sft.py
│       │   └── run_exp2_sft.sh
│       └── grpo/
│           ├── run_vanilla_grpo.sh    # E3 训练脚本
│           ├── run_schemashift.sh     # E4 训练脚本
│           └── run_aug_only.sh        # E5 训练脚本
├── tests/
│   ├── test_advantage.py          # 10 tests
│   ├── test_api_mapper.py         # 9 tests
│   ├── test_schema_perturber.py   # 13 tests
│   ├── test_schemashift_grpo_estimator.py # 11 tests
│   ├── test_runtime_regressions.py # 15 tests
│   ├── test_enum_mapping.py       # 14 tests
│   └── test_register_estimator_integration.py # 4 tests
├── docs/
│   ├── technical_report.md        # 技术方案
│   └── ablation_plan.md           # 消融实验方案
└── data/verl/
    ├── exp4_schemashift/          # E4 数据（8100/900 rows，9 records/task）
    ├── exp5_aug_only/             # E5 数据（2700/300 rows，3 records/task）
    ├── exp3_grpo/                 # E3 数据（900/100 rows，1 record/task）
    └── exp2_sft/                  # E2 数据
```

---

## 7. 当前状态

### 算法层

| 维度 | 状态 | 依据 |
|------|------|------|
| 公式 | ✅ 收敛 | `A = strat_z + beta × global_z`，代码-文档一致 |
| 扰动机制 | ✅ 完整 | name_map + enum_map 双向映射 |
| verl 集成 | ✅ step 1 已跳通 | 设计基于 verl 源码（自定义 estimator + uid 编码 + compute_advantage monkey-patch）。`test_register_estimator_integration.py` 验证 patch 后 `non_tensor_batch` 真实注入 estimator；E4 step 1 smoke 实跑验证 ray actor 跨进程重注册 |
| 边界安全 | ✅ | 单样本 std→1.0 防 NaN，缺层回退 |
| 映射管线 | ✅ | agent loop 从 parquet 读 name_map_json/enum_map_json 初始化 mapper；GT 同步支持 enum_map |
| Reward/Eval extra-turn | ✅ | reward/eval/agent loop 三处统一拒绝 GT 结束后的非空 tool-call turn，允许尾部空轮次 |
| BFCL 原生解析器 | ✅ | bounded linear parser（输入≤8192 、段数≤64），防 17:10 卡死重现；4 个 regression 测试覆盖 |
| 数据长度预检 | ✅ | `src/training/length_check.py` 三入口默认开启；实测 max=8809、p99=8732 vs limit=10240 buffer=1.4k |
| 单元测试 | ✅ | 105/105 pass |

### 项目层

| 维度 | 状态 | 依据 |
|------|------|------|
| 消融设计 | ✅ | E3→E5→E4 两步链，每步单变量 |
| 文档一致性 | ✅ | README / docs/project_status.md / docs/archive/review.md / CLAUDE.md 已按 2026-06-16 收尾事实同步 |
| E3/E4/E5 可运行 | ✅ | 4×A10 step 1 smoke 全部跑通，loss finite、ckpt 落盘 |
| E6 可运行 | ❌ | 仅文档，无脚本 |
| Parquet 数据 | ✅ | E4 train=8100/val=900 (3:3:3 完整性断言通过)，E5 train=2700/val=300 (1:1:1 完整性断言通过) |
| 代码测试 | ✅ | 105/105 pass |

### 待 GPU 验证

- ✅ E2 SFT 1-step smoke 已跑通（train_loss=0.2412, ~13s/step, 4×A10）
- ✅ E3 1-step smoke pass（55s/step, max_mem 22.4 GB, grad_norm finite, ckpt saved）
- ✅ E5 1-step smoke pass（145s/step, max_mem 18.0 GB, grad_norm finite, ckpt saved）
- ✅ E4 1-step smoke pass（~13min/step 含 final val rollout, ~21GB offload 后, schemashift estimator monkey-patch 验证 batch=36, has_perturbation_level=True）
- ✅ verl Hydra config `+algorithm.beta` 在 E4 smoke 中正常传入 estimator
- ✅ BFCL `execute_multi_turn_func_call` 透传 `test_entry_id` 已验证不再报参数错
- ✅ `_parse_bfcl_native_args` bounded linear parser 上线后 cluster GIL 卡死未再复现
- ⏳ 全量训练收敛性（>1 step）尚未验证
- ⏳ BFCL 官方 state-based eval 集成（当前仅 response-based AST）
- E4 全量训练前可跑 `SCHEMASHIFT_PRECHECK=1 python -m src.training.run_exp4 …` 进一步在 verl 启动前验证 group 完整性

### 本轮（2026-06-16 收尾）发现并修复的项

| 问题 | 根原因 | 修复 |
|---|---|---|
| arl 环境 verl import 全面崩 | torch 2.10 / vllm 0.17 / trl 1.1 都超出 verl 0.6.1 兼容窗口 | 按方案 A 一次性降级；torch 2.8 / vllm 0.11 / trl 0.11.4 / tensordict 0.10 / flash_attn 2.7.3 / xformers 0.0.32.post1；备份 `docs/archive/arl.requirements.bak.txt` |
| flashinfer 0.6.4 JIT 报 `<cuda/functional>` not found | vllm 0.11 默认走 flashinfer，但它需 CCCL 2.x（CUDA 12.x 带），本机是 CUDA 11.8 + CCCL 1.x | 3 个 GRPO sh 默认 export `VLLM_USE_FLASHINFER_SAMPLER=0` + `VLLM_ATTENTION_BACKEND=FLASH_ATTN` |
| dataloader 报 `sequence_length=5458 > max_length=2048` | yaml 里 `max_prompt_length=2048` 从未根据实际数据验证；BFCL prompt 带工具 schema 后 p99=8800 | exp3/4/5 yaml 统一调为 10240 |
| AgentLoopWorker 报 `Got size of DataProto 324 and chunk 8` | `train_batch_size × group_size = 324`，verl 默认 `agent.num_workers=8`，324 不被 8 整除 | 3 个 sh 加 `actor_rollout_ref.rollout.agent.num_workers=${N_GPUS}` |
| BFCL `execute_multi_turn_func_call` 报 `missing 1 required positional argument: 'test_entry_id'` | 项目代码 bug：BFCL 当前版本该参数为必传，但 `bfcl_agent_loop.py:606` 未传 | 用 `request_id`（uuid4）生成唯一 `test_entry_id` 透传下去，同时避免 BFCL `globals()` instance 缓存跨 episode 状态污染 |
| **AgentLoopWorker GIL 卡死 15 min**（本轮重大发现） | py-spy 定位到 3 个 worker 全停在 `_parse_bfcl_native_args:180`，active+gil。原实现是 O(N²)（`args_part.find("=", i)` × outer while），garbled 超长 LLM 输出 → 几万字符 → 单次解析几十分钟 | 重写为 bounded linear parser：输入≤8192/段数≤64/key≤64字符，单遍扫描，毫秒级返回。增 4 个 regression 测试覆盖超长/畑形/超限/嵌套 |
| **E4 estimator 未注册到 ray actor 进程**（本轮重大发现） | `run_exp4.py` 在主进程注册 estimator + monkey-patch，但 verl trainer 跑在独立 ray actor，主进程状态不跨进程传递 | 引入 `SchemaShiftTaskRunner(TaskRunner)` 子类，在 `run()` 重新注册，通过 `run_ppo(config, task_runner_class=...)` hook 注入 |
| 数据长度默认不验证 | 之前指望看 yaml 注释，可靠不足 | 新增 `src/training/length_check.py`，三个入口默认开启：超 limit → RuntimeError，p99 > limit×0.95 → warn；9 个单元测试覆盖 |
| 依赖契约无 CI 兜底 | 之后升 vllm/torch/trl 会静默裂 | 新增 `tests/test_smoke_grpo_imports.py` 16 个测试，包括 vllm ≤0.11.0 上限、flash_attn import、trl ValueHead 符号、flashinfer JIT 能启动检查 |

---

## 8. 审查趋势

### 7 轮 Codex 对抗式审查

```
R1: P0=6  ████████████  消融设计漏洞（E7/E8 无 Aug 退化）
R2: P0=3  ██████        公式冲突 + test split
R3: P0=5  ██████████    工程链路全断
R4: P0=4  ████████      parquet 旧数据 + 分组前提
R5: P0=5  ██████████    shuffle 打散 + metadata 不匹配
R6: P0=4  ████████      patch 是 no-op（读了 verl 源码才发现）
R7: P0=0  ▏             ✅ 通过
```

关键转折点：R6 读了 verl 源码后，将"monkey-patch verl"改为"编码 uid + 注册自定义 estimator"，消除了前面 6 轮关于 advantage 数据链路的不确定性。

### 2026-06-15 完整代码 review + 2026-06-16 收尾

P0:
- ✅ `src/training/run_exp4.py:105` 未闭合字符串已修复，E4 入口可拉起
- ✅ reward / eval / agent loop 三处统一拒绝 GT 结束后的非空 tool-call turn，允许尾部空轮次；新增 4 个回归测试覆盖
- ✅ BFCL `test_entry_id` 透传 + `_parse_bfcl_native_args` bounded linear parser 上线并跳通 step 1 smoke
- ✅ E4 estimator 跨 ray actor 进程注册问题以 `SchemaShiftTaskRunner` 修复

P1/P2:
- ✅ 测试覆盖 65 → 76 → **105**（本轮新增 16 个 smoke imports + 9 个 length_check + 4 个 parser regression）
- ✅ 训练入口加载期 group_id 完整性检查（`run_exp4.py`，`SCHEMASHIFT_PRECHECK=1` 触发，添加后重构为 `length_check.assert_e4_group_integrity`）
- ✅ 数据长度预检该项默认开启：E3/E4/E5 入口启动时会跳 `length_check.maybe_run_length_check`
- ✅ 依赖契约 CI 兜底：`tests/test_smoke_grpo_imports.py` 16 个测试覆盖 vllm/torch/trl/flash_attn/flashinfer
- ✅ GPU smoke run（E3/E4/E5）已 step 1 跳通
- ⏳ BFCL 官方 state-based eval 集成待接入（当前仅 response-based AST）
- ⏳ 全量训练收敛性（>1 step）尚未跑
- ⏳ 可选重构：`src/agent_loop/bfcl_agent_loop.py` 分层（`_parse_bfcl_native_args` 从 agent_loop 抽到 `src/envs/bfcl_parser.py`）——仅在要接第二个环境时再做，当前 ROI 低
