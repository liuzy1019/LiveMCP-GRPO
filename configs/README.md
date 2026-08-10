# configs/

配置目录只描述入口和覆盖关系。项目进度见 `docs/PROJECT_STATUS.md`，训练参数的唯一代码来源是
`src/training/trainer_config.py` 和 `src/training/hyperparams.py`。

## 配置入口

| 配置 | 用途 |
|---|---|
| `livemcp_rollout.yaml` | LiveMCP rollout 注册 |
| `live_mcp/ten_domain_suite.yaml` | 十域 subprocess suite 和支持的 reward profile |
| `live_mcp/*.yaml` | 各 domain transport、session 和 server 命令 |
| `verl/verl/trainer/config/ppo_trainer.yaml` | vendored verl Hydra 基础配置 |
| `tests/per_domain/*.yaml` | 逐域 30+10 生成、PROVE rollout/reward 测试的不可变输入配置 |

正式训练入口：

```bash
bash scripts/train_grpo.sh
```

`TrainerConfig.from_env()` 生成 Hydra overrides，项目不维护第二套静态训练 YAML。

`tests/per_domain/` 只保存测试输入，不保存动态结果。文件名必须与 `domain` 一致；每次运行生成
唯一 `run_id`，结果事实写入 `generation_manifest.json` 和独立审计报告。配置不得写物理 GPU ID，
GPU 由运行命令显式传入，避免把某台机器的占用状态固化进仓库。

## 十域 suite

当前 domain：

```text
calendar
shopping
banking
email
filesystem
payments
crm
issue_tracker
team_chat
food_delivery
```

十域 YAML 不维护手写 dependency graph 或 query template。依赖图由
`livemcp-gen build-cache` 对全部无序工具 pair 调用 Teacher 分类，cache 绑定 schema、
Teacher model 和 classifier contract。

## 当前训练默认值

以下值来自 `TrainerConfig`，可被环境变量或 CLI 覆盖：

| 参数 | 默认值 |
|---|---:|
| Policy model | `models/Qwen/Qwen3-4B-Instruct-2507` |
| total steps | 350 |
| train batch | 32 |
| mini batch | 8 |
| micro batch / GPU | 2 |
| prompt length | 12,384 |
| response length | 16,384 |
| rollout n | 16 |
| rollout TP | 1 |
| max user turns | 5 |
| max assistant turns | 10 |
| learning rate | `1e-6` |
| strategy | `fsdp` |
| reward profile | `oval_full` |

`data.max_prompt_length` 不得低于 10,240。真实运行参数以启动日志和实验目录中的 resolved
overrides 为准，不能仅引用本表。

## 主要环境变量

| 变量 | 用途 |
|---|---|
| `OVAL_MODEL_PATH` | Policy 模型 |
| `OVAL_TRAIN_FILE` / `OVAL_VAL_FILE` | 数据入口 |
| `OVAL_TOTAL_STEPS` | 训练步数 |
| `OVAL_TRAIN_BATCH_SIZE` | train batch |
| `OVAL_ROLLOUT_N` | 每个 prompt 的 rollout 数 |
| `OVAL_PROMPT_LENGTH` / `OVAL_RESPONSE_LENGTH` | 序列长度 |
| `OVAL_GPU_MEM_UTIL` | rollout 显存利用率 |
| `OVAL_REWARD_PROFILE` | `prove_baseline` 或 `oval_full` |
| `OVAL_DOMAINS` | rollout domain 列表 |
| `OVAL_SUITE_PATH` | suite 路径 |
| `OVAL_USE_WANDB` / `OVAL_WANDB_PROJECT` | WandB |
| `LIVEMCP_LATA` | OVAL LATA 模式 |

## Reward profile

| Profile | Reward | Estimator |
|---|---|---|
| `prove_baseline` | PROVE 五组件任务奖励 | verl `grpo` |
| `oval_full` | PROVE 任务奖励 + OVAL 扩展 | `livemcp_grpo` |

Profile 与 estimator 的绑定在训练启动时 fail-closed，不允许静默混用。

Reward profile 不是完整实验配置。正式训练必须同时选择 experiment profile：

| Experiment profile | 作用 |
|---|---|
| `prove_local_v1` | 十域本地 PROVE baseline，冻结论文 Policy checkpoint、composition proxy、paper-shape 超参和标准 GRPO |
| `prove_reward_gray_v1` | 固定 8-row 分层 view 的 PROVE 小样本 reward 灰度 |
| `prove_reproduction_v1` | 20-domain / 13,517-row / external-abstention 严格复现；当前条件不满足时拒绝启动 |
| `oval_local_v1` | 与 `prove_local_v1` 共用模型、数据和 sampler，只改变显式声明的 OVAL 组件 |
| `oval_reward_gray_v1` | 与 PROVE 灰度使用同一 8-row view 的 OVAL 奖励对照 |

四卡 diagnostic smoke 可以覆盖 steps 或 batch，但 resolved config 必须记录 override。
正式对照不得使用 `data/train.parquet` 活动软链接，必须绑定不可变 artifact 路径和 SHA256。

## 约束

- 路径使用 repo-root-relative 形式；
- GPU、batch、micro batch、TP 和训练超参必须可注入；
- suite 只声明兼容 profile，不复制 reward 权重；
- 正式训练是否完成以实验日志、checkpoint 和真实 rollout 复验为准。
