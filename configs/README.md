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
| `livemcp_rollout.yaml` | LiveMCP rollout 注册 | ✅ |
| `live_mcp/ten_domain_suite.yaml` | 十域 subprocess suite 配置 | ✅ |
| `live_mcp/*.yaml` | 各 domain subprocess transport 与 session 配置 | ✅ |

十域 YAML 不维护手写 dependency graph 或 query template。运行时依赖图由
`scripts/build_dependency_cache.py` 按全部无序 pair 调用 Teacher 分类，并保存为绑定 schema、
Teacher model 与 classifier contract 的版本化 cache。YAML 手工边不会合入该 graph，因此
不得作为生成语义来源。

## 正式训练核心参数

配置文件由 `src/training/trainer_config.py` 管理，当前使用 A10 默认值。多 GPU tier 自适应默认值：

| Tier | prompt_length | response_length | max_num_seqs | micro_batch | train_batch | rollout_n | 状态 |
|------|--------------|-----------------|-------------|------------|-------------|-----------|------|
| A10 | 12384 | 16384 | 16 | 1 | 16 | 16 | ✅ 当前 |
| L20 | 12384 | 16384 | 64 | 2 | 32 | 16 | ✅ 当前 |
| A100/Hopper | 16384 | 16384 | 128 | 4 | 64 | 16 | ✅ 当前 |
| 其他 | 10240 | 2048 | 8 | 1 | 8 | 4 | ✅ 兜底 |

> 以上为参考默认值，训练时通过 CLI 参数或环境变量按需覆盖，不绑定特定硬件。

## 环境变量覆盖

| 变量 | 用途 | 默认值 |
|------|------|--------|
| `OVAL_MODEL_PATH` | Policy 模型路径 | `models/Qwen/Qwen3-4B` |
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
| `OVAL_SUITE_PATH` | Suite 配置路径 | `configs/live_mcp/ten_domain_suite.yaml` |

## 注意

- 所有路径使用项目根目录相对路径，禁止写死机器绝对路径
- 本目录只描述配置事实；正式训练是否完成以 checkpoints、训练日志和 GPU 环境复验结果为准
