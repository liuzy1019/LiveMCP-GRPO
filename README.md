# SchemaShift-GRPO

> MCP-Tools-Aware Tool-Use RL — Schema Robustness via Online Rollout + Verifiable Reward

通过 Schema Perturbation + Multi-Step Reward + Stratified Advantage，让模型在陌生 schema、多工具干扰下学会稳定的 tool-use 决策。

当前方案：
- **数据**: Toucan EpisodeSeed → `prepare_grpo_data.py` → parquet（含 `replay_observation`）
- **SFT Cold-Start**: Qwen3-4B，本地产物位于 `outputs/sft_cold_start_4b/final`
- **GRPO**: verl + SchemaShiftReplayLoop（交互式静态 replay）+ 五组件 Reward + 多步奖励 + StratAdv
- **目标硬件**: 8×L20 44GB（colocated 模式）

---

## 当前进度

| 阶段 | 状态 | 说明 |
|------|------|------|
| 数据准备 | ✅ 本地就绪 | 4962 条 EpisodeSeed、9146 条 SFT 样本、GRPO train/val parquet 已重建：train 33003 行 / val 1728 行，含 no_tool 与 distractor |
| SFT Cold-Start | ✅ 本地产物存在 | `training_report.json` 与 `final/` 权重存在；当前 shell 无可用 GPU，未重新复训 |
| RL 训练 | ⏳ 入口/配置就绪 | direct/cold 配置与脚本已就绪；本地未发现正式 GRPO checkpoint |
| Live MCP MVP | ✅ 可选分支 | calendar/shopping subprocess stdio server；默认不接入 GRPO |
| 评测 | ⏳ 待构建 | Self MCP Robustness Set |

---

## 项目结构

```
schemashift-grpo/
├── src/
│   ├── envs/                       # schema_perturber + api_mapper + replay_mcp_executor + mcp_tool_environment
│   ├── data/                       # episode_seed_builder + conditioned_builder + distractor_sampler + sft_step_exporter
│   ├── reward/                     # action_parser + component_reward + schemashift_reward_fn (多步奖励)
│   ├── eval/                       # enum matching
│   ├── training/                   # schemashift_advantage + grpo_estimator + register_estimator + length_check
│   ├── agent_loop/                 # schemashift_replay_loop (正式) + schemashift_oval_loop (live MCP)
│   └── live_mcp/                   # live MCP MVP: subprocess stdio servers + offline rollout/reward
├── scripts/                        # 训练/数据/环境脚本
│   ├── train_grpo.py                # GRPO 统一训练入口
│   ├── run_grpo_smoke.sh           # smoke test shell
│   ├── oval_grpo_smoke.sh          # OVAL smoke test shell
│   ├── sft_cold_start.py           # SFT cold-start
│   ├── prepare_grpo_data.py        # EpisodeSeed → verl parquet
│   └── generate_oval_data.py       # OVAL 训练数据生成
├── configs/
│   ├── grpo_direct.yaml           # 直接 GRPO 配置
│   ├── grpo_cold.yaml             # SFT冷启动→GRPO 配置
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

**硬件要求**：GRPO 默认按 8×L20 44GB 设计；SFT/GRPO 的 GPU 数、batch、micro batch 与 TP size 应通过配置、环境变量或 Hydra override 调整。

---

## 测试

```bash
conda run -n arl python -m pytest tests/

# 依赖或 GPU 不完整时至少运行：
conda run -n arl python -m compileall src scripts tests
git diff --check
```

当前本地复核结果：`arl` 环境下全量测试 297 passed；当前 shell 的 CUDA 不可用，GRPO smoke/正式训练仍需目标 GPU 环境复验。

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

当前默认导出策略：

- 每个 group 仍按 `none/mild/strong × 3 copies = 9` 行展开。
- `no_tool` 默认按最终 group 约 15% 采样进入 GRPO。
- 每条展开记录默认以 40% 概率注入 3-8 个 distractor tools。
- 本地当前快照：train `33003` 行 / `3667` groups；val `1728` 行 / `192` groups。

## GRPO 训练

```bash
# 直接 GRPO（默认）
bash scripts/run_grpo.sh

# SFT 冷启动 → GRPO
MODE=cold bash scripts/run_grpo.sh

# 显式指定配置
bash scripts/run_grpo.sh --config configs/grpo_direct.yaml

# 可通过环境变量覆盖参数
N_GPUS=4 BETA=0.3 TOTAL_STEPS=300 bash scripts/run_grpo.sh
```

当前正式训练配置（[grpo_direct.yaml](configs/grpo_direct.yaml)）；本地尚无正式 GRPO checkpoint 可证明跑批完成：

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
| [docs/project_plan.md](docs/project_plan.md) | 权威方案文档与工程约束 |
| [configs/README.md](configs/README.md) | 配置文件清单与参数说明 |
| [data/README.md](data/README.md) | 数据目录结构与复现 |

---

## 许可

MIT License.
