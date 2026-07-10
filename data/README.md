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
├── train.parquet -> runs/{MMDD_HHMM}/train.parquet   # 符号链接，指向最新 run
├── val.parquet   -> runs/{MMDD_HHMM}/val.parquet     # 符号链接，指向最新 run
└── README.md
```

### 命名约定

- `runs/{MMDD_HHMM}/` — 与 `logs/{MMDD_HHMM}_gen_{N}.log` 时间戳对应，方便追溯
- `data/train.parquet` 和 `data/val.parquet` 始终是符号链接，指向最新 run
- 旧 run 目录保留，手动清理时删除对应 `runs/{MMDD_HHMM}/` 即可
- vLLM 运行日志写到 `logs/`，生成成功后自动删除；失败时保留用于排查

---

## 数据生成管线

> 重新生成后建议运行 `python scripts/validate_pipeline.py --live` 验证管线完整性。
> 完整设计参考 PROVE 论文 §3.2。

### Step 1 — 自动发现依赖图（Auto-Discovered Dependency Graph）

```
对每个 domain，对所有 n² 个工具对调用 LLM 分类器，判断关系：
  explicit（工具 A 的输出是工具 B 的必需输入）
  implicit（A 必须先执行以建立状态）
  none
→ 缓存为有向依赖图（按 tool schema hash 索引）
→ 提取 length-2 到 length-5 的工具链，作为多步 query 的种子
```

每个 domain 的依赖图缓存在 `data/dependency_graphs/{domain}_{hash}.json`。

### Step 2 — 实时状态采样（Live-State Sampling / Grounded Query Generation）

```
在 query 生成前, 对每个 domain 调用只读探测工具（search / list / get_unique_values）
→ 枚举真实存在的实体（账户、联系人、事件、商品等）
→ 过滤到有足够支撑数据的子集（如至少有 20 个商家的城市、有非零余额的账户）
→ 缓存 compact "sampling context"（每 k 个对话刷新一次）
→ 注入 query 生成 prompt，约束 LLM 只能使用 context 中存在的实体
```

这一步解决了 naive 生成器产生不存在的实体 ID / 名称的问题，保证生成的 tool chain 端到端可执行。

### Step 3 — 状态机编排器（State-Machine Orchestrator）

```
每个对话由 5 组状态驱动（论文 §3.2 Step 3）：
  1. QUERY GENERATION — 从依赖图链中采样，用实时状态 grounding，
     按信息完整度分层（complete 70% / missing 10% / minimal 20%），
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

每一步 LLM 决策都看到完整 domain context（tools, live state, execution history），形成 teacher 轨迹。每条轨迹记录为 oracle trace，包含全部 tool call 参数和 server 响应。

### Step 4 — 鲁棒性注入（Robustness Knobs）

```
生成过程中以概率注入扰动，扩展训练分布（论文 §3.2 Step 4）：
  - Distractor injection（40%）：混入 3-8 个来自其他 domain 的无关工具
  - Enum stripping（30%）：从参数 schema 中移除枚举值列表，迫使模型从描述推理约束
  - Irrelevance queries（5%）：生成没有可用工具能处理的 query，训练拒绝能力
  - Missing function（20%）：在链选择后隐藏一个必需工具，产生 abstention/clarification 样本
```

### Step 5 — 重放验证与去重（Replay Validation and Deduplication）

```
每条完成的对话在 freshly reset 的 MCP 环境中重放：
  - 只计入 schema 级和执行级错误（不含空结果响应）
  - 错误率 > 30% 的对话丢弃
  - Provenance check：敏感参数（passwords, tokens）只能出现在可追溯到前序 user
    turn 或 tool output 的位置
  - Jaccard 去重：基于 tool-call 序列，阈值 0.70（位置感知）
```

### 训练数据构成

```
训练集由三部分来源组成（对齐 PROVE 论文）：
  1. Multi-turn MCP conversations — 20 个 domain 的状态机生成轨迹
  2. Clarification trajectories — missing-function 变体产生的 ask_clarification 样本
  3. External abstention — When2Call（806 条，G=∅ 场景）
                       + xLAM-Irrelevance（316 条，不相关 query）
```

**生成策略**：目标数量 N，实际生成 N + max(10, N/2) 条（50% oversample），经 replay 验证、Jaccard 去重、训练合约过滤后取前 N 条。不足时自动 recovery（最多 3 轮），用不同 seed 偏移补充。

### 难度分布

| 类型 | 比例 | 说明 |
|------|------|------|
| **complete** | 70% | user query 包含全部所需信息 |
| **missing** | 10% | user query 省略一个关键参数 |
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

### 关键约束

- `reward_model.ground_truth.success_criteria` 是 **JSON 字符串**（非 list[dict]），避免 pyarrow 混合类型崩溃
- `reward_model.ground_truth.oracle_calls` 保存完整 2-5 步工具链和一个显式终止动作；`action` 为 `tool_call` / `ask_clarification` / `final_answer` / `report_error`
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
  "missing_function_rate": 0.20,
  "irrelevance_ratio": 0.05,
  "difficulty_mix": {"complete": 0.7, "missing": 0.1, "minimal": 0.2},
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
