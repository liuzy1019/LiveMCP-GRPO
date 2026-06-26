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
├── oval_grpo_train.parquet         # GRPO 训练数据（gitignored）
├── oval_grpo_val.parquet           # GRPO 验证数据（gitignored）
└── README.md
```

---

## 数据生成管线

```
PROVE Teacher（两步管线，默认模式）
  ┌──────────────────────────────────────────────────┐
  │ Phase 1: LLM 规划 (task_planner.py)               │
  │   输入: domain schemas + grounded state            │
  │   输出: user_query + [tool_a, tool_b, ...]         │
  │   LLM 仅输出工具名序列，~200 tokens，8B 可靠       │
  └──────────────────────────────────────────────────┘
                        ↓
  ┌──────────────────────────────────────────────────┐
  │ Phase 2: 执行记录 (task_planner.py)               │
  │   真实 MCP session 执行 → 记录 oracle trace        │
  │   infer_args: 从 state/schema/history 推断参数     │
  │   derive_success_criteria: 从 state delta 派生     │
  └──────────────────────────────────────────────────┘
                        ↓
  ┌──────────────────────────────────────────────────┐
  │ 鲁棒性注入 (orchestrator.py)                       │
  │   distractor tools:  40%                          │
  │   missing function:  20%                          │
  │   irrelevance query:  5%                          │
  └──────────────────────────────────────────────────┘
                        ↓
  ┌──────────────────────────────────────────────────┐
  │ 导出 parquet (generate_oval_data.py)               │
  │   verl 格式: prompt + reward_model + extra_info    │
  │   group_id 按 (domain × batch) 分组               │
  └──────────────────────────────────────────────────┘
```

### 难度分布

| 类型 | 比例 | 说明 |
|------|------|------|
| **complete** | 60% | user query 包含全部所需信息 |
| **missing** | 20% | user query 省略一个关键参数 |
| **minimal** | 20% | user query 极其简略，需模型自行推断 |

---

## 数据生成命令

```bash
# PROVE 模式（默认，推荐）
# vLLM 模式：
python scripts/generate_oval_data.py \
  --count 500 --val-count 100 \
  --domain all \
  --model Qwen3-8B \
  --api-base http://localhost:8001/v1 \
  --seed 42 \
  --output data/oval_grpo_train.parquet \
  --val-output data/oval_grpo_val.parquet

# Local transformers 模式：
python scripts/generate_oval_data.py \
  --count 500 --val-count 100 \
  --domain all \
  --model models/Qwen/Qwen3-8B \
  --seed 42 \
  ...

# E2E 模式（legacy）
python scripts/generate_oval_data.py \
  --teacher e2e --count 500 ...

# 记录实验配置与结果（自动写入 data/experiments/）
python scripts/generate_oval_data.py \
  --experiment-tag prove_v1 \
  ...
```

---

## 实验记录规范

每次正式数据生成运行，在 `data/experiments/{YYYY-MM-DD}_{tag}/` 下记录：

- **`config.json`** — 完整 CLI 参数 + 环境信息（模型版本、GPU、commit hash）
- **`result.json`** — 产出统计（总行数、各 domain 分布、scenario_type 分布、难度分布）

示例 `config.json`：

```json
{
  "run_id": "2026-06-26_prove_v1",
  "command": "python scripts/generate_oval_data.py --count 500 --val-count 100 --domain all --model models/Qwen3-8B --seed 42 --experiment-tag prove_v1",
  "model": "Qwen3-8B",
  "domain": "all",
  "count": 500,
  "val_count": 100,
  "seed": 42,
  "teacher_mode": "prove",
  "distractor_rate": 0.40,
  "missing_function_rate": 0.20,
  "irrelevance_ratio": 0.05,
  "difficulty_mix": {"complete": 0.6, "missing": 0.2, "minimal": 0.2},
  "git_commit": "abc1234",
  "gpu_model": "A10",
  "timestamp": "2026-06-26T14:38:21+08:00"
}
```

示例 `result.json`：

```json
{
  "train_rows": 478,
  "val_rows": 96,
  "yield": 0.956,
  "duration_seconds": 1234.5,
  "domain_distribution": {"calendar": 50, "banking": 48, "email": 50},
  "scenario_distribution": {"normal": 239, "distractor": 191, "missing_function": 48},
  "difficulty_distribution": {"complete": 287, "missing": 96, "minimal": 95}
}
```
