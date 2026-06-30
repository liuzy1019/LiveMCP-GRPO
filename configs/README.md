# configs/

训练和环境配置文件。所有 YAML 参数均带行内注释说明用途和约束。

## 训练路线与配置对应关系

| 路线 | 配置文件 | 启动脚本 | 状态 |
|------|----------|----------|------|
| OVAL GRPO | Hydra (`ppo_trainer`) + `TrainerConfig.from_env()` | `bash scripts/train_grpo.sh` | ✅ 主路线 |

训练配置由 `src/training/trainer_config.py` 统一管理（PyTorch Lightning 风格）。
Hydra 配置文件位于 `verl/verl/trainer/config/ppo_trainer.yaml`（verl 内置），
项目特有参数通过环境变量 (`OVAL_*` 前缀) 和 CLI 参数注入。

## 文件清单

| 文件 | 用途 | 状态 |
|------|------|------|
| `agent_loop.yaml` | Agent loop 注册 | ✅ |
| `ds_zero2.json` | DeepSpeed ZeRO-2 配置（JSON 格式） | ✅ |
| `live_mcp/suite_mvp.yaml` | 全量套件配置（10 domain） | ✅ |
| `live_mcp/*.yaml` | 各 domain 子进程配置（banking/calendar/crm/email/filesystem/food_delivery/issue_tracker/payments/shopping/team_chat） | ✅ |

## 正式训练核心参数

配置文件由 `src/training/trainer_config.py` 管理，支持 GPU tier 自适应默认值：

| Tier | prompt_length | response_length | max_num_seqs | micro_batch | train_batch | rollout_n |
|------|--------------|-----------------|-------------|------------|-------------|-----------|
| L20 | 12384 | 16384 | 64 | 2 | 32 | 16 |
| A100/Hopper | 16384 | 16384 | 128 | 4 | 64 | 16 |
| A10 | 10240 | 4096 | 8 | 1 | 8 | 8 |
| 其他 | 10240 | 2048 | 8 | 1 | 8 | 4 |

可通过 CLI 参数覆盖：`--model`、`--gpus`、`--total-steps`、`--batch-size`、`--rollout-n`、`--lr`、`--strategy`、`--wandb`。

## 环境变量覆盖

| 变量 | 用途 | 默认值 |
|------|------|--------|
| `OVAL_MODEL_PATH` | Policy 模型路径 | `models/Qwen3-4B` |
| `OVAL_TRAIN_FILE` | 训练数据路径 | `data/train.parquet` |
| `OVAL_VAL_FILE` | 验证数据路径 | `data/val.parquet` |
| `OVAL_TOTAL_STEPS` | 训练步数 | 300 |
| `OVAL_ROLLOUT_N` | Rollout 每组数量 | tier 自适应 |
| `OVAL_PROMPT_LENGTH` | 最大 prompt 长度 | tier 自适应 |
| `OVAL_RESPONSE_LENGTH` | 最大 response 长度 | tier 自适应 |
| `OVAL_GPU_MEM_UTIL` | GPU 显存利用率 | tier 自适应 |
| `OVAL_USE_WANDB` | 启用 WandB | 0 |
| `OVAL_WANDB_PROJECT` | WandB 项目名 | `oval-mcp-grpo` |
| `OVAL_LR` | 学习率 | `1e-6` |
| `OVAL_STRATEGY` | 分布式策略 | `fsdp` |
| `OVAL_DOMAINS` | Oval loop domain 列表 | 全部 10 个 |
| `OVAL_SUITE_PATH` | Suite 配置路径 | `configs/live_mcp/suite_mvp.yaml` |

## 注意

- `ds_zero2.json` 保持 JSON 格式（DeepSpeed 不支持 YAML）
- 所有路径使用项目根目录相对路径，禁止写死机器绝对路径
- 本目录只描述配置事实；正式训练是否完成以 checkpoints、训练日志和 GPU 环境复验结果为准
