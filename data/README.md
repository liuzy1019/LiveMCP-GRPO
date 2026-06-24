# data/

本目录下的数据文件不入库（详见根目录 `.gitignore`），clone 后请按以下步骤复现。

当前本地快照已包含：

- `data/toucan/episode_seeds.jsonl`：4962 条有效 EpisodeSeed（见 `data/toucan/builder_stats.json`）
- `data/sft/sft_train.jsonl`：9146 条 SFT 样本（见 `data/sft/export_stats.json`）
- `data/grpo_train_replay.parquet` 与 `data/grpo_val_replay.parquet`：GRPO replay 数据。当前本地快照为 train 33003 行 / 3667 groups，val 1728 行 / 192 groups。

## 目录结构

```
data/
├── toucan/                     # Toucan 主数据源（EpisodeSeed JSONL）
├── grpo_train_replay.parquet   # verl GRPO 训练数据（由 prepare_grpo_data.py 生成）
├── grpo_val_replay.parquet     # verl GRPO 验证数据（由 prepare_grpo_data.py 生成）
└── live_mcp/                   # Live MCP 生成任务与 trace（由 live smoke 脚本生成）
```

## 数据复现

```bash
# 1. 获取 Toucan EpisodeSeed 数据
python scripts/download_toucan.py
python scripts/inspect_toucan.py

# 2. SFT 样本导出（依赖 episode seeds）
# 由 src/data/sft_step_exporter.py 从 episode_seed 导出

# 3. GRPO parquet 导出
python scripts/prepare_grpo_data.py \
  --episode_seeds data/toucan/episode_seeds.jsonl \
  --output data/grpo_train_replay.parquet \
  --val_output data/grpo_val_replay.parquet

# 4. Live MCP 任务与 smoke trace
python scripts/generate_live_mcp_tasks.py \
  --suite configs/live_mcp/suite_mvp.yaml \
  --output data/live_mcp/tasks/live_mcp_mvp.jsonl \
  --num-tasks 20 \
  --seed 42
```

当前默认 GRPO 导出策略：

- 每个 group 展开为 `none/mild/strong × 3 copies = 9` 行。
- `no_tool` 默认按最终 group 约 15% 采样进入 GRPO。
- 每条展开记录默认以 40% 概率注入 3-8 个 distractor tools。
- `error_output` 暂不进入 GRPO，避免未验证错误恢复 oracle 引入噪声。

## 设计原则

- **训练主源 Toucan**：所有 GRPO 数据从 Toucan EpisodeSeed 构建
- **Oracle-Preserving**：Schema 扰动后 ground truth 通过 name_map/enum_map 可还原
- **行为分布补齐**：GRPO 数据包含 `call_then_final`、`call_then_call` 和采样后的 `no_tool`
- **Distractor 增强**：部分行额外注入相似/跨域 distractor tools，并写入 `extra_info`
- **SFT 仅对齐格式**：SFT 样本从 episode_seed 可见上下文导出，不暴露 oracle_trace
- **GRPO parquet 含 replay_observation**：离线 replay 执行器无需真实 MCP server
