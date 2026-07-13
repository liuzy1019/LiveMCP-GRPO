# data/

本目录存放训练数据产出和实验记录。原始 parquet 数据不入库（见 `.gitignore`），实验配置与统计结果跟踪入库。

---

## 目录结构

```
data/
├── dependency_graphs/              # 各 domain 工具依赖图（precompute 产出）
│   └── {domain}_{hash}.json        # 单个 domain 的拓扑依赖图
├── experiments/                    # 实验记录（配置+结果摘要，跟踪入库）
│   ├── .gitkeep
│   └── {YYYY-MM-DD}_{tag}/         # 单次实验目录
│       ├── config.json             # 完整运行参数
│       └── result.json             # 产出统计
├── external/                       # 外部数据集（when2call / xlam，gitignored）
├── runs/                           # 每次生成的带版本子目录（gitignored）
│   └── {MMDD_HHMM}/                # 单次生成目录，命名与日志一致
│       ├── train.parquet
│       └── val.parquet
├── train.parquet -> runs/{MMDD_HHMM}/train.parquet   # 完整 run 验收后建立
├── val.parquet   -> runs/{MMDD_HHMM}/val.parquet     # 完整 run 验收后建立
└── README.md
```

### 命名约定

- `runs/{MMDD_HHMM}/` — 与 `logs/{MMDD_HHMM}_gen_{N}.log` 时间戳对应，方便追溯
- `data/train.parquet` 和 `data/val.parquet` 只在 train/val 均完整产出并通过生成门禁后更新；失败或零验证集 smoke 不得覆盖入口
- 旧 run 目录保留，手动清理时删除对应 `runs/{MMDD_HHMM}/` 即可
- vLLM 运行日志写到 `logs/`，生成成功后自动删除；失败时保留用于排查

---

## 数据生成管线

> 重新生成后建议运行 `python scripts/validate_pipeline.py --stages 1,2`，并用 Stage 3 或正式生成 artifact 核对运行链路。
> 完整设计参考 PROVE 论文 §3.2。

### 当前验收状态

当前未通过修复后的十域端到端验收。有效问题、证据路径和下一步门禁统一记录在 `docs/KNOWN_ISSUES.md`；本文件不保留已失效 smoke 或旧 cache 数量。

### Step 1 — 自动发现依赖图（Auto-Discovered Dependency Graph）

```
对每个 domain，对所有有序 `n(n-1)` 工具对（不含 self-pair）调用 LLM 分类器，判断关系：
  explicit（工具 A 的输出是工具 B 的必需输入）
  implicit（A 必须先执行以建立状态）
  none
→ 缓存为有向依赖图（按 tool schema 和 dependency semantics 版本索引）
→ 提取 length-2 到 length-5 的工具链，作为多步 query 的种子
```

每个 domain 的依赖图缓存在 `data/dependency_graphs/{domain}_{hash}.json`。
严格 cache 必须同时记录 `expected_pair_count=n(n-1)`、`classified_pair_count` 和 `classification_complete=true`；旧的无序 `nC2` cache 或缺少这些字段的 artifact 均不能作为“全部有向 pair 已由 LLM 分类”的证据。
论文写作“all n² tool pairs”，但没有公开是否包含 self-pair，也没有公开 chain traversal 是否允许重复节点。当前工程选择全部不同工具的有序 pair，即 `n(n-1)`，并提取不重复节点的 simple paths；这是可审计的本地实现选择，不能宣称为论文逐实现事实。pairwise classifier 默认每批 8 个 pair、最多 512 输出 tokens；若返回缺项，只重问未分类 pair。任一 batch 重试耗尽仍不完整时，整张 graph fail-closed，不缓存部分结果。
并行生成时，同一 `{domain}_{hash}` 的冷缓存只允许一个进程执行 LLM 分类；等待进程获得文件锁后必须再次读取缓存。缓存通过同目录临时文件和 `os.replace` 原子发布，生成 shard 启动前默认由单进程完成所选 domain 的预热。

缓存保留 LLM 对全部有向 pair 的分类结果，不再经过本地 handler denylist、domain 黑名单、entity heuristic 或强制破环改写。候选 chain 的可行性由后续 live-state、execution 与 replay 验证；handler 事实只能用于 feasibility 或运行验证，不能回写 pairwise classifier 的 cache。

### Step 2 — 实时状态采样（Live-State Sampling / Grounded Query Generation）

```
在 query 生成前, 对每个 domain 调用只读探测工具（search / list / get_unique_values）
→ 枚举真实存在的实体（账户、联系人、事件、商品等）
→ 按已选 chain 的真实 handler 前置条件筛选可用实体；当前不维护论文示例中的全局 supporting-data filter 表
→ 每个 seed/session 首次使用时重建 compact sampling context；同一未变化 session 内最多复用 k 次，写入后的 continuation 强制刷新
→ 注入 query 生成 prompt，约束 LLM 只能使用 context 中存在的实体
```

这一步约束 query 使用真实实体，并在生成前排除已知不可行 chain；最终是否可执行仍以 live execution 和 fresh replay 为准。

### Step 3 — 状态机编排器（State-Machine Orchestrator）

```
每个对话由 5 组状态驱动（论文 §3.2 Step 3）：
  1. QUERY GENERATION — 从依赖图链中采样，用实时状态 grounding，
     按信息完整度分层（complete 60% / missing 20% / minimal 20%），
     叠加 persona + reference date 条件
  2. LLM PROCESSING — Teacher LLM 被 prompt 提供 query + tool schemas，
     输出 tool call / 澄清 / 终止；validator 检查 well-formed JSON
  3. TOOL EXECUTION — tool call 分派到 live MCP server，
     结果分类为 SUCCESS / PARTIAL_SUCCESS / FAILURE
  4. RECOVERY — 失败时状态机在 retry-with-corrected-params /
     retry-with-alternative-tool / give-up 中选择
  5. CONTINUATION — 采样是否结束对话/追问/澄清，turn-decay 调度，
     min_turns=2, max_turns=3
```

首轮 user query 由一条 live-state feasible dependency chain 生成。该 chain 及其 graph hints 不传给 Teacher action prompt；Teacher 只根据 query、candidate tool schemas 和真实 execution history 决策。前置节点不拆成后续 user turn；continuation 只根据刷新后的 live state、此前 query 与真实 execution history继续同一 conversation，不为后续 round 重新构造 dependency chain。Replay 可执行不等于语义正确，正式 smoke 还必须核对 query、oracle 和 terminal。

可行链在通过 schema/handler/live-state feasibility 后均匀采样；不再按 mutation 数量或单一工具名设置论文外优先池。失败由真实 execution、state-machine recovery 与 fresh replay 处理。

运行链路不执行状态反转、capability 同义词或实体字符串匹配过滤；PROVE 未发布这些 gate。是否保留样本由用户目标的实际执行、fresh replay、provenance 和去重决定。`chain_seed`/`dependency_edges` 的完整性仅是本项目 OVAL coverage reward 的本地结构合约。

`source_chain_seed` 表示生成 query 时使用的工作流种子；只有 oracle 实际完整覆盖该序列时才写入 `chain_seed` 并构造 OVAL dependency edges。PROVE 没有发布“必须逐节点覆盖 seed chain”的 corpus hard gate。每个 round 的用户目标仍须由成功执行完成，或以 clarification / abstention 合法终止。Missing-function 在 chain 选择后隐藏必要工具。

Recovery 的 alternative tool 不要求与 chain 节点名称相同；成功轨迹由 live execution 和 fresh replay 判定。Missing-function 可以先调用仍可见的前置工具，再因隐藏能力缺失而 clarification / abstention；不得以 `missing_function=True` 拒绝所有 tool call。硬约束仅包括 hidden tool 不可见、不可执行、不可进入 oracle，以及最终 clarification / abstention。

过滤边界：fresh replay、sensitive-parameter provenance 和 Jaccard 0.70 是论文公开 corpus gates。Parquet schema、显式 terminal、round contracts、hidden-tool 不泄漏与 dependency-edge 索引是本项目 rollout/reward 的结构合同。位置感知 Jaccard 和按公开 corpus 数量反推的 missing-function 采样率属于本地工程选择，必须在实验 metadata 中记录，不能写成论文公布的实现细节。

Recovery 边界：论文明确包含 graceful give-up with fallback explanation。Teacher 判断当前 candidate tools 无法完成用户结果时可以直接 `report_error`，不要求先执行一个必然失败的工具调用；“无 execution failure 的 report_error”只记录诊断，不作为导出拒绝条件。

预算边界：论文 continuation 的 `min_turns=2, max_turns=3` 指 conversation rounds；一次 round 内可以包含多个顺序 tool calls。Parquet `budget` 是本项目 rollout 的 action-turn 工程合同，必须满足 `budget >= ground-truth tool-call 数 + conversation round 数`，从而至少容纳每轮一个 terminal。论文 §3.3 的 adaptive efficiency budget 按 tool-call 数计算奖励惩罚，不负责截断 episode。

真实十域 smoke 的补充约束：成功 normal conversation 必须有 2--3 个真实 user turns；follow-up 必须依赖当前 live state 并继续此前 conversation，不得由内部 chain 节点机械改写。filesystem archive 操作只使用 live-state 中的 file 实体。

每一步 LLM 决策都看到完整 domain context（tools, live state, execution history），形成 teacher 轨迹。每条轨迹记录为 oracle trace，包含全部 tool call 参数和 server 响应。

### Step 4 — 鲁棒性注入（Robustness Knobs）

```
生成过程中以概率注入扰动，扩展训练分布（论文 §3.2 Step 4）：
  - Distractor injection（40%）：混入 3-8 个来自其他 domain 的无关工具
  - Enum stripping（30%）：从参数 schema 中移除枚举值列表，迫使模型从描述推理约束
  - Irrelevance queries（5%）：生成没有可用工具能处理的 query，训练拒绝能力
  - Missing function（目标约 12.1%）：在链选择后隐藏一个必需工具，产生 abstention/clarification 样本；比例由论文公开的 1,500 / (10,895 + 1,500) corpus count 推导
```

强制执行契约：

```text
missing-function:
  完整 query_chain 先生成确实需要 hidden tool 的 query
  Teacher 只接收移除 hidden tool 后的 candidate schemas，不接收 graph hints
  可先执行仍可见的前置工具，最终以 clarification/abstention 终止
  hidden_tools 在生成、Parquet、rollout executor 和 reward 中保持一致

Replay / provenance:
  验证完整 teacher_attempt_trace（失败、blocked、retry、alternative 均保留）
  ground-truth oracle 仅保留最终有效步骤

distractor:
  在 Teacher 生成前把 3-8 个跨 domain 无关工具加入 candidate set，并对同一候选集 replay
  schema 携带 owner domain 并路由到真实 MCP server
  ground-truth oracle 不应调用无关 distractor
```

### Step 5 — 重放验证与去重（Replay Validation and Deduplication）

```
每条完成的对话在 freshly reset 的 MCP 环境中重放：
  - 只计入 schema 级和执行级错误（不含空结果响应）
  - 错误率 > 30% 的对话丢弃
  - Provenance check：敏感参数（passwords, tokens）只能出现在可追溯到前序 user
    turn 或 tool output 的位置
  - Jaccard 去重：全部 replay/provenance 存活样本进入同一 tool-call sequence 候选池，阈值 0.70；不按 domain 豁免
  - 零工具 clarification/abstention 不回退到 query-text Jaccard，避免加入论文未发布的文本过滤
```

### 训练数据构成

```
训练集由三部分来源组成（对齐 PROVE 论文）：
  1. Multi-turn MCP conversations — 状态机生成轨迹
  2. Clarification trajectories — missing-function 变体产生的 ask_clarification 样本
  3. External abstention — When2Call（806 条，G=∅ 场景）
                       + xLAM-Irrelevance（316 条，不相关 query）
```

**生成策略**：全局只计算一次候选预算；shard 子进程不重复增加固定 oversample floor。`generate_many` 按 domain 增量提交任务，达到 quota 后停止为该 domain 创建新 future。不足时 recovery 只补充缺口，经 replay、Jaccard 去重和训练合约过滤后取精确 N 条。recovery 默认最多 3 轮；正式十域任务可通过 `--max-recovery-rounds` 提高上限，直到每域满足最终配额，不能用其他域的富余样本代替缺口。长任务应指定 `--checkpoint-path`：每轮完成后原子保存候选 `LiveTask`、下一轮缺口请求和配置指纹；用相同配置重启时只补剩余 domain，不重复首轮。

多 shard 边界：`--shard-mode` 子进程只保证本 shard 的总 Jaccard-unique 行数和可消费结构，不要求每个 shard 独立满足十域比例，也不按单 shard domain 缺口 recovery；所有 shard 汇总后，global merge 才执行跨 shard Jaccard、逐域 train/val quota 与 split 隔离。只有全局 domain 候选不足才 fail-closed。

### 难度分布

| 类型 | 比例 | 说明 |
|------|------|------|
| **complete** | 60% | user query 包含全部所需信息 |
| **missing** | 20% | user query 省略一个关键参数 |
| **minimal** | 20% | user query 极其简略，需模型自行推断 |

---

## Parquet Schema

| 字段 | 类型 | 说明 |
|------|------|------|
| `prompt` | str | 仅包含初始 `system + user`；不得包含 teacher tool history |
| `data_source` | str | `"live_mcp_state_machine"` |
| `reward_model` | dict | `{"style":"rule", "ground_truth": {"task_id","oracle_calls","success_criteria","required_tools"}}` |
| `extra_info` | dict | domain, target_servers, required_tools, scenario_type, oracle_calls, hidden_tools 等 |
| `uid` | str | 等于 task_id |
| `group_id` | str | 等于 task_id（每个 task 独立一组） |
| `perturbation_level` | str | `complete` / `missing` / `minimal` |
| `scenario_type` | str | `task_planner` / `distractor` / `missing_function` / `irrelevant` |

### P0 Baseline Integrity Fields（extra_info）

以下字段在 `extra_info` 中以 JSON string 存储（Parquet round-trip 保障）：

| 字段 | 类型 | 说明 |
|------|------|------|
| `round_contracts` | JSON `[{"round_idx":int, "required_tools":[str], "allowed_terminal_actions":[str]}]` | 每轮对话的预期工具和合法终止动作。多轮数据必须存在且数量与 `conversation_queries` 一致。rollout 和 reward 均强制校验 |
| `dependency_edges` | JSON `[[src_idx, dst_idx], ...]` | 索引边列表，通过全链对齐（full-chain alignment）生成。src_idx/dst_idx 是 `oracle_calls` 中 tool_call 的位置（0-based）。所有边必须满足 `src_idx < dst_idx`。中间节点可同时作为前边 dst 和后边 src。chain_seeded 任务必须恰好 `len(chain_seed)-1` 条边，不完整者在 split 前被拒绝 |
| `dependency_graph_complete` | bool | **诊断字段**，不替代导出前门禁。chain_seeded 任务由 `_validate_task_training_contract` 强制校验边完整性
| `conversation_queries` | JSON `[str, ...]` | 多轮对话的每轮 user query。长度即轮数 |
| `paper_replay_valid` | bool | schema/execution error rate ≤ 30%（PROVE §3.2 Step 5） |
| `project_outcome_valid` | bool | 所有 success_criteria 在 fresh session 上满足 |
| `criteria_failed_count` | int | 实际失败 success_criteria 数量（来自 replay 统计） |
| `replay_error_rate` | float | replay 错误率 |
| `chain_seed` | JSON `[str]` | 任务使用的工具链种子 |
| `generation_mode` | str | `chain_seeded` 或 `unseeded_fallback` |

### Terminal Progression 规则（P0-2）

多轮 rollout 遵循以下 fail-closed 规则：

| Terminal | 行为 |
|----------|------|
| `final_answer` | 仅在本轮 `required_tools` **全部成功执行**且 `allowed_terminal_actions` 包含时才推进下一轮 |
| `report_error` | **始终终止** episode，不推进 |
| `ask_clarification` | 仅在有配对 user reply 时推进，否则终止 |
| 非法 terminal | 记录 `contract_violation` 审计事件并停止 |
| 多余 terminal | reward 拒绝（`n_actual != n_expected`） |

Rollout 缺 contract 或 contract 数量与对话轮数不一致 → `RuntimeError`（fail-closed）。

### Entity Quality Filtering（P0-1）

数据生成时先构造 probe record，再按候选 chain 做实体可行性判断：

1. **首轮 probe**：枚举所有实体（list_/search_ 只读工具）
2. **阶段 enrichment**（仅 food_delivery）：对 restaurant 实体逐次调用 `get_menu`，合并菜单数据到 record
3. **状态过滤**：全局 `DOMAIN_ENTITY_QUALITY_FILTERS` 当前为空；按已选 chain 的 handler 前置条件判定，只有字段明确冲突才排除，字段缺失交给 live execution/replay

chain-specific 条件包括账户基数/状态、商品库存、订单 lifecycle、filesystem 类型等；它们只在对应 chain 需要时生效。`_extract_chain_context` 按字段**存在性**判断是否使用 qualified list，空 qualified 不会回退到原始实体。

### 关键约束

- `reward_model.ground_truth.success_criteria` 是 **JSON 字符串**（非 list[dict]），避免 pyarrow 混合类型崩溃
- `reward_model.ground_truth.oracle_calls` 保存 Teacher 实际成功 oracle calls 和一个显式终止动作；seed chain 仅在完整实现时写入 `chain_seed/dependency_edges`，不能假定每条 oracle 都严格等于 2–5 步 seed chain
- `prompt` 是 JSON 字符串，OVAL loop 端自动 `json.loads` 恢复
- `oracle_calls` 在 `extra_info` 中也是 JSON 字符串序列化，避免 pyarrow struct 统一化导致的字段丢失

---

## 数据生成命令

### vLLM 参数自适应

`generate_data.sh` 根据 GPU 显存自动缩放 vLLM 参数，无需手动调参：

| GPU 等级 | KV Cache 余量 | max_model_len | max_num_seqs | clients | enforce_eager | gpu_memory_utilization |
|----------|--------------|--------------|-------------|---------|---------------|------------------------|
| 紧 (< 1.8 GiB) | A10 + Gemma-4-31B TP=4 典型 | 7168 | 8 | 2 | 0 | 0.88 |
| 充裕 (≥ 1.8 GiB) | 大显存卡或小模型 | 8192 | 32 | 4 | 0 | 0.88 |

判断公式：`kv_budget = GPU_MEM × 0.88 − model/TP − 1.5(overhead)`，< 1.8 GiB 即为"紧"。
`enforce_eager` 始终为 0（CUDA Graph 开），A10 sm_86 支持且性能更优。

所有参数均可通过环境变量覆盖：
```bash
VLLM_MAX_NUM_SEQS=16 VLLM_CLIENTS_PER_INSTANCE=4 \
  bash scripts/generate_data.sh --count 500
```

### 运行命令

```bash
# 统一生成脚本（推荐，自动检测并行策略，自动管理 vLLM）
# 默认 Teacher: 本地 Gemma-4-31B-it
# 输出到 data/runs/{MMDD_HHMM}/，并自动更新 data/train.parquet 符号链接
bash scripts/generate_data.sh --count 500 --val-count 100

# 指定 run-id（默认自动取当前时间 MMDD_HHMM）
bash scripts/generate_data.sh --count 500 --run-id 0709_1500

# 单 domain 快速测试
bash scripts/generate_data.sh --domain calendar --count 200
```

生成完成后：
- 数据写入 `data/runs/{MMDD_HHMM}/train.parquet` 和 `val.parquet`
- `data/train.parquet` / `data/val.parquet` 符号链接自动更新
- 主日志写入 `logs/{MMDD_HHMM}_gen_{N}.log`
- vLLM 日志（如有）成功后自动删除，失败时保留在 `logs/`

---

## 实验记录规范

每次正式数据生成运行，在 `data/experiments/{YYYY-MM-DD}_{tag}/` 下记录：

- **`config.json`** — 完整 CLI 参数 + 环境信息（模型版本、GPU、commit hash）
- **`result.json`** — 产出统计（总行数、各 domain 分布、scenario_type 分布、难度分布）

示例 `config.json`：

```json
{
  "run_id": "2026-06-29_prove_v2",
  "command": "python scripts/generate_data.py --count 500 --val-count 100 --experiment-tag prove_v2",
  "model": "models/Google/Gemma-4-31B-it",
  "domain": "all",
  "count": 500,
  "val_count": 100,
  "seed": 42,
  "distractor_rate": 0.40,
  "missing_function_rate": 0.121,
  "irrelevance_ratio": 0.05,
  "difficulty_mix": {"complete": 0.6, "missing": 0.2, "minimal": 0.2},
  "git_commit": "abc1234",
  "gpu_model": "L20",
  "timestamp": "2026-06-29T10:38:48+08:00"
}
```

示例 `result.json`：

```json
{
  "train_rows": 475,
  "val_rows": 96,
  "yield": 0.95,
  "duration_seconds": 12345.6,
  "domain_distribution": {"calendar": 50, "banking": 48, "email": 50},
  "scenario_distribution": {"task_planner": 239, "distractor": 143, "missing_function": 93},
  "difficulty_distribution": {"complete": 285, "missing": 95, "minimal": 95}
}
```

## 训练数据读取

训练时通过环境变量指定数据路径：
```bash
export OVAL_TRAIN_FILE=data/train.parquet
export OVAL_VAL_FILE=data/val.parquet
bash scripts/train_grpo.sh
```
