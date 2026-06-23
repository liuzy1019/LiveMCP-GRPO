# SchemaShift-GRPO

> MCP-Tools-Aware Tool-Use RL — Schema Robustness via Online Rollout + Verifiable Reward

通过 Schema Perturbation + Multi-Step Reward + Stratified Advantage，让模型在陌生 schema、多工具干扰下学会稳定的 tool-use 决策。

当前方案：
- **数据**: Toucan EpisodeSeed → `prepare_grpo_data.py` → parquet（含 `replay_observation`）
- **SFT Cold-Start**: Qwen3-4B, 格式对齐, 9146 条样本
- **GRPO**: verl + SchemaShiftReplayLoop (交互式静态 replay) + 五组件 Reward + 多步奖励 + StratAdv
- **硬件**: 8×L20 44GB（colocated 模式）

---

## 当前进度

| 阶段 | 状态 | 说明 |
|------|------|------|
| 数据准备 | ✅ 完成 | Toucan inspection + EpisodeSeed 构建 + SFT 样本导出 (9146条) + GRPO parquet |
| SFT Cold-Start | ✅ 完成 | Qwen3-4B, 8×L20 44GB, DeepSpeed ZeRO-2 |
| RL 训练 | ✅ 正式训练 | 300 step, group_size=1, 9 records/task, max_turns=5, StratAdv beta=0.25 |
| Live MCP MVP | ✅ smoke 可运行 | calendar/shopping subprocess stdio server |
| 评测 | ⏳ 待构建 | Self MCP Robustness Set |

---

## 项目结构

```
schemashift-grpo/
├── src/
│   ├── envs/                       # schema_perturber + api_mapper + replay_mcp_executor + mcp_tool_environment
│   ├── data/                       # episode_seed_builder + conditioned_builder + distractor_sampler + sft_step_exporter
│   ├── reward/                     # action_parser + component_reward + schemashift_reward_fn (多步奖励)
│   ├── eval/                       # bfcl_eval + matching
│   ├── training/                   # schemashift_advantage + grpo_estimator + register_estimator + length_check
│   ├── agent_loop/                 # schemashift_replay_loop (正式) + bfcl_agent_loop (legacy)
│   └── live_mcp/                   # live MCP MVP: subprocess stdio servers + offline rollout/reward
├── scripts/                        # 训练/数据/环境脚本
│   ├── run_grpo.sh                 # 路由入口 → run_schemashift.sh
│   ├── run_grpo_smoke.sh           # smoke test shell
│   ├── sft_cold_start.py           # SFT cold-start
│   ├── prepare_grpo_data.py        # EpisodeSeed → verl parquet
│   └── train/grpo/run_schemashift.sh  # E4 正式训练脚本
├── configs/
│   ├── exp4_schemashift.yaml       # E4 直接 GRPO 配置
│   ├── exp4_schemashift_cold.yaml   # E4 SFT冷启动→GRPO 配置
│   ├── grpo_smoke.yaml             # smoke test 配置
│   ├── sft_cold_start_4b.yaml      # SFT 配置
│   └── live_mcp/                   # Live MCP 配置
├── data/                           # 训练/评测数据 (gitignored)
├── tests/                          # 单元测试
├── requirements.txt
└── pyproject.toml
```

---

## 环境搭建

```bash
conda activate arl
python -m pip install -e .
python scripts/check_dependency_conflicts.py
```

**硬件要求**：8×L20 (44GB) 用于 GRPO；SFT 可在 4×A10 (23GB) 上运行（batch=1 + ZeRO-2）。

---

## 测试

```bash
pytest tests/  # 100+ passed
```

## SFT Cold-Start

```bash
torchrun --nproc_per_node=8 scripts/sft_cold_start.py \
    --config configs/sft_cold_start_4b.yaml
```

输出：`outputs/sft_cold_start_4b/final`

## GRPO 数据准备

```bash
python scripts/prepare_grpo_data.py \
    --episode_seeds data/toucan/episode_seeds.jsonl \
    --output data/grpo_train_replay.parquet \
    --val_output data/grpo_val_replay.parquet
```

## GRPO 训练

```bash
# 直接 GRPO（默认）
bash scripts/run_grpo.sh

# SFT 冷启动 → GRPO
MODE=cold bash scripts/run_grpo.sh

# 可通过环境变量覆盖参数
N_GPUS=4 BETA=0.3 TOTAL_STEPS=300 bash scripts/run_grpo.sh
```

当前正式训练配置（[exp4_schemashift.yaml](configs/exp4_schemashift.yaml)）：

| 参数 | 值 |
|------|-----|
| 模型 | Qwen3-4B |
| 学习率 | 由 verl 默认 |
| 训练步数 | 300 |
| rollout.n | 1（数据侧 9 条/task） |
| max_turns | 5（交互式 replay） |
| StratAdv beta | 0.25 |
| KL coef | 0.04 |
| max prompt length | 10240 |
| max response length | 4096 |

## GRPO Smoke Test

```bash
bash scripts/run_grpo_smoke.sh --config configs/grpo_smoke.yaml
```

---

## Live MCP Smoke

Live MCP 默认不接入 GRPO，只通过显式脚本运行：

```bash
python scripts/generate_live_mcp_tasks.py \
    --suite configs/live_mcp/suite_mvp.yaml \
    --output data/live_mcp/tasks/live_mcp_mvp.jsonl \
    --num-tasks 20 \
    --seed 42

python scripts/run_live_mcp_smoke.py \
    --suite configs/live_mcp/suite_mvp.yaml \
    --tasks data/live_mcp/tasks/live_mcp_mvp.jsonl \
    --server calendar \
    --num-tasks 10 \
    --seed 42
```

---

## 文档

| 文档 | 内容 |
|------|------|
| [docs/project_status.md](docs/project_status.md) | 项目设计文档：算法方案、训练路线、参数配置、组件状态 |
| [docs/live_mcp_branch.md](docs/live_mcp_branch.md) | Live MCP 并行分支（未接入主训练路线） |
| [configs/README.md](configs/README.md) | 配置文件清单与参数说明 |
| [data/README.md](data/README.md) | 数据目录结构与复现 |

---

## 许可

MIT License.
