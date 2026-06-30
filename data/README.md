# data/

本目录存放训练数据产出和实验记录。原始 parquet 数据不入库（见 `.gitignore`），实验配置与统计结果跟踪入库。

---

## 目录结构

```
data/
├── experiments/                    # 实验记录（配置+结果摘要，跟踪入库）
│   ├── .gitkeep
│   └── {YYYY-MM-DD}_{tag}/         # 单次实验目录
│       ├── config.json             # 完整运行参数
│       └── result.json             # 产出统计
├── train.parquet                   # GRPO 训练数据（gitignored）
├── val.parquet                     # GRPO 验证数据（gitignored）
└── README.md
```

---

## 数据生成管线

```
PROVE Teacher（LLM-in-the-loop，每轮决策）
  ┌──────────────────────────────────────────────────┐
  │ 1. LLM 决策 (task_planner.py)                     │
  │   输入: domain schemas + live state + history      │
  │   输出: 下一步 action (tool_call / terminal)       │
  └──────────────────────────────────────────────────┘
                        ↓ 真实 MCP 执行
  ┌──────────────────────────────────────────────────┐
  │ 2. 执行记录                                       │
  │   真实 MCP session 执行 → 记录 oracle trace        │
  │   derive_success_criteria: 从 state delta 派生     │
  │   PROVE 扰动: intermittent/paginated/partial_batch │
  └──────────────────────────────────────────────────┘
                        ↓ replay validate
  ┌──────────────────────────────────────────────────┐
  │ 3. 鲁棒性注入 (orchestrator.py)                    │
  │   distractor tools:  30% (默认)                   │
  │   missing function:  20%                          │
  │   irrelevance query:  5%                          │
  └──────────────────────────────────────────────────┘
                        ↓ Jaccard dedup (位置感知, 0.70)
  ┌──────────────────────────────────────────────────┐
  │ 4. 导出 parquet (generate_data.py)                 │
  │   verl 格式: prompt (JSON string) + reward_model   │
  │     + extra_info + scenario_type                   │
  │   success_criteria: JSON 字符串 (类型安全)          │
  │   oracle_calls: 保留 action 字段 (澄清任务)         │
  └──────────────────────────────────────────────────┘
```

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
| `prompt` | str | JSON 序列化的消息列表 `[{"role":"system",...},{"role":"user",...}]` |
| `data_source` | str | `"live_mcp_state_machine"` |
| `reward_model` | dict | `{"style":"rule", "ground_truth": {"task_id","oracle_calls","success_criteria","required_tools"}}` |
| `extra_info` | dict | domain, target_servers, required_tools, scenario_type, oracle_calls, hidden_tools 等 |
| `uid` | str | 等于 task_id |
| `group_id` | str | 等于 task_id（每个 task 独立一组） |
| `perturbation_level` | str | `complete` / `missing` / `minimal` |
| `scenario_type` | str | `task_planner` / `distractor` / `missing_function` / `irrelevant` |

### 关键约束

- `reward_model.ground_truth.success_criteria` 是 **JSON 字符串**（非 list[dict]），避免 pyarrow 混合类型崩溃
- `reward_model.ground_truth.oracle_calls` 中每个 call 包含 `action` 字段，区分 `tool_call` / `clarification`
- `prompt` 是 JSON 字符串，OVAL loop 端自动 `json.loads` 恢复
- `oracle_calls` 在 `extra_info` 中也是 JSON 字符串序列化，避免 pyarrow struct 统一化导致的字段丢失

---

## 数据生成命令

```bash
# 统一生成脚本（推荐，自动检测并行策略）
bash scripts/generate_data.sh --model models/Qwen/Qwen3-8B --count 500
bash scripts/generate_data.sh --model models/Qwen/Qwen3-32B --count 500 --val-count 100

# vLLM 模式（32B 必须用此模式）
python scripts/generate_data.py \
  --count 500 --val-count 100 \
  --domain all \
  --model Qwen3-32B-Instruct \
  --api-base http://localhost:8000/v1 \
  --seed 42 \
  --output data/train.parquet \
  --val-output data/val.parquet

# Local transformers 模式（8B 可用）
python scripts/generate_data.py \
  --count 500 --val-count 100 \
  --domain all \
  --model models/Qwen/Qwen3-8B \
  --seed 42

# 单 domain 快速测试
bash scripts/generate_data.sh --model models/Qwen/Qwen3-8B --domain calendar --count 200

# 记录实验配置与结果（自动写入 data/experiments/）
python scripts/generate_data.py \
  --experiment-tag prove_v1 \
  --count 500 --val-count 100 \
  --model Qwen3-32B-Instruct \
  --api-base http://localhost:8000/v1
```

---

## 实验记录规范

每次正式数据生成运行，在 `data/experiments/{YYYY-MM-DD}_{tag}/` 下记录：

- **`config.json`** — 完整 CLI 参数 + 环境信息（模型版本、GPU、commit hash）
- **`result.json`** — 产出统计（总行数、各 domain 分布、scenario_type 分布、难度分布）

示例 `config.json`：

```json
{
  "run_id": "2026-06-29_prove_v2",
  "command": "python scripts/generate_data.py --count 500 --val-count 100 --model Qwen3-32B-Instruct --api-base http://localhost:8000/v1 --experiment-tag prove_v2",
  "model": "Qwen3-32B-Instruct",
  "domain": "all",
  "count": 500,
  "val_count": 100,
  "seed": 42,
  "distractor_rate": 0.30,
  "missing_function_rate": 0.20,
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
