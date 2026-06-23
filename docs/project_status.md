# SchemaShift-GRPO：项目设计文档

> 状态：2026-06-22 | 当前阶段：GRPO 训练配置就绪，待正式跑批

---

## 1. 项目目标

**问题**：MCP 场景下，不同 MCP server 实现同一功能时，工具名、描述措辞、枚举值命名可能完全不同（schema perturbation）。标准 GRPO 训练会放大模型对训练时见过的具体 schema 表面形式的依赖，面对陌生 schema 变体时性能急剧下降。

**方案**：在 GRPO 训练中引入 Schema Perturbation + Stratified Advantage + 多步 Reward，让模型学会透过 schema 表面差异识别工具语义。

| 组件 | 机制 |
|------|------|
| Schema Perturbation | 训练时对工具 schema 扰动（工具名/描述/枚举值同义替换），三档强度：none / mild / strong |
| Stratified Advantage | 按扰动强度分层归一化：A = strat_z + β × global_z（β=0.25） |
| 交互式静态 Replay | 模型输出 tool_call → 匹配 oracle → 返回预存 observation，模拟真实多轮交互 |
| 五组件 Reward | format / schema_valid / tool_selection / argument_keys / argument_values |
| 多步 Reward | 逐步对齐 oracle_actions，含 trajectory-level 信号（coverage bonus + penalty） |

---

## 2. 训练路线

入口：`scripts/run_grpo.sh`，通过 `MODE` 环境变量选择起点。

| 模式 | 命令 | 初始模型 | 配置文件 |
|------|------|---------|---------|
| **direct**（默认） | `bash scripts/run_grpo.sh` | `models/Qwen3-4B` | `exp4_schemashift.yaml` |
| **cold** | `MODE=cold bash scripts/run_grpo.sh` | `outputs/sft_cold_start_4b/final` | `exp4_schemashift_cold.yaml` |

两条路线共用相同的训练算法和参数，仅模型起点不同。cold 模式需要先跑 SFT 冷启动：

```bash
torchrun --nproc_per_node=8 scripts/sft_cold_start.py --config configs/sft_cold_start_4b.yaml
```

---

## 3. 训练配置（exp4_schemashift.yaml）

| 参数 | 值 | 说明 |
|------|-----|------|
| 模型 | Qwen3-4B | direct 模式：原始权重；cold 模式：SFT 产物 |
| 数据 | Toucan EpisodeSeed | `data/grpo_train_replay.parquet` / `data/grpo_val_replay.parquet` |
| 训练步数 | 300 | |
| rollout.n | 1 | 数据侧 9 条/task（none/mild/strong 各 3），shuffle=False |
| max_turns | 5 | 交互式 replay 最大轮次 |
| agent_loop | `schemashift_replay` | SchemaShiftReplayLoop |
| estimator | `schemashift_grpo` | Stratified Advantage，β=0.25 |
| KL coef | 0.04 | |
| ppo_epochs | 1 | |
| micro_batch_per_gpu | 3 | |
| max_prompt_length | 10240 | |
| max_response_length | 4096 | |

可通过环境变量覆盖：`N_GPUS`、`BETA`、`TOTAL_STEPS`、`SAVE_FREQ`、`TEST_FREQ`。

---

## 4. 数据管线

```
Toucan EpisodeSeed (JSONL)
  └─ scripts/prepare_grpo_data.py
       ├─ SchemaPerturber：按 perturbation_level 扰动 schema
       ├─ name_map / enum_map：双向映射注入 extra_info
       ├─ ground_truth：oracle_actions（含 replay_observation）
       └─ 输出：data/grpo_train_replay.parquet + data/grpo_val_replay.parquet
```

parquet 的 `extra_info` 必须含：`perturbation_level`、`name_map`、`enum_map`、`scenario_type`、`group_id`、`uid`。

---

## 5. 奖励计算

### 单步 Episode（call_only）
ComponentReward 对模型输出的 action 与 oracle_action 做五组件评估，返回 0-1 分数。

### 多步 Episode（call_then_final 等）
1. 解析模型输出中的多个 `<tag>` 块
2. 逐步与 oracle_actions 对齐评估
3. 加权平均：第一步 0.6，后续步骤均分 0.4
4. trajectory-level 信号：coverage_bonus(0.1) + all_exact_bonus(0.1) - early_final_penalty(0.15)

---

## 6. 组件状态

| 组件 | 状态 | 说明 |
|------|------|------|
| SFT Cold-Start | ✅ 完成 | Qwen3-4B, 9146 条样本, DeepSpeed ZeRO-2 |
| GRPO 数据制备 | ✅ 完成 | `prepare_grpo_data.py` |
| SchemaPerturber | ✅ 可用 | none/mild/strong, name_map + enum_map |
| ReplayMCPExecutor | ✅ 可用 | 离线 replay，无需真实 MCP server |
| SchemaShiftReplayLoop | ✅ 可用 | COVERT 风格交互式静态 replay |
| ComponentReward | ✅ 可用 | 五组件评估 |
| 多步 Reward | ✅ 可用 | 逐步对齐 + trajectory 信号 |
| Stratified Advantage | ✅ 可用 | β=0.25 |
| GRPO Smoke Test | ✅ 通过 | step 1 跑通 |
| GRPO 正式训练 | ⏳ 配置就绪 | 待启动 |
| Live MCP | ✅ smoke 可用 | 并行分支，未接入 GRPO，见 [live_mcp_branch.md](live_mcp_branch.md) |
| 评测 | ⏳ 待构建 | Self MCP Robustness Set |

---

## 7. 关键文件索引

| 文件 | 用途 |
|------|------|
| [configs/exp4_schemashift.yaml](../configs/exp4_schemashift.yaml) | 正式训练配置（direct 模式） |
| [configs/exp4_schemashift_cold.yaml](../configs/exp4_schemashift_cold.yaml) | 正式训练配置（cold 模式） |
| [configs/sft_cold_start_4b.yaml](../configs/sft_cold_start_4b.yaml) | SFT 冷启动配置 |
| [scripts/train/grpo/run_schemashift.sh](../scripts/train/grpo/run_schemashift.sh) | GRPO 训练启动脚本 |
| [scripts/sft_cold_start.py](../scripts/sft_cold_start.py) | SFT 冷启动脚本 |
| [scripts/prepare_grpo_data.py](../scripts/prepare_grpo_data.py) | 数据制备 |
| [src/training/run_exp4.py](../src/training/run_exp4.py) | verl GRPO 入口 |
| [src/agent_loop/schemashift_replay_loop.py](../src/agent_loop/schemashift_replay_loop.py) | SchemaShiftReplayLoop |
| [src/reward/schemashift_reward_fn.py](../src/reward/schemashift_reward_fn.py) | 五组件多步 Reward |
| [src/reward/component_reward.py](../src/reward/component_reward.py) | 组件级 reward |
| [src/envs/schema_perturber.py](../src/envs/schema_perturber.py) | Schema 扰动 |
| [src/envs/replay_mcp_executor.py](../src/envs/replay_mcp_executor.py) | 离线 replay 执行器 |
| [src/training/schemashift_advantage.py](../src/training/schemashift_advantage.py) | Stratified Advantage |
| [src/training/register_estimator.py](../src/training/register_estimator.py) | verl estimator 注册 |

---

## 8. 硬件

8×L20 44GB，colocated 模式（actor + ref + rollout 同卡）。4B BF16 约 8GB，显存充裕，无需 offload。
