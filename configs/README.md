# configs/

训练和环境配置文件。所有 YAML 参数均带行内注释说明用途和约束。

## 训练路线与配置对应关系

| 路线 | 配置文件 | 启动脚本 | Python 入口 |
|------|----------|----------|-------------|
| 直接 GRPO | `exp4_schemashift.yaml` | `bash scripts/run_grpo.sh` | `src/training/run_exp4.py` |
| SFT 冷启动 → GRPO | `exp4_schemashift_cold.yaml` | `MODE=cold bash scripts/run_grpo.sh` | `src/training/run_exp4.py` |

## 文件清单

| 文件 | 用途 | 状态 |
|------|------|------|
| `exp4_schemashift.yaml` | 直接 GRPO（交互式 replay + StratAdv） | ✅ |
| `exp4_schemashift_cold.yaml` | SFT 冷启动 → GRPO | ✅ |
| `sft_cold_start_4b.yaml` | Qwen3-4B SFT cold-start（格式对齐） | ✅ |
| `grpo_smoke.yaml` | GRPO smoke test（2 步验证链路） | ✅ |
| `agent_loop.yaml` | Agent loop 配置 | ✅ |
| `ds_zero2.json` | DeepSpeed ZeRO-2（JSON 格式，DS 直接消费） | ✅ |
| `live_mcp/` | Live MCP 配置（可选并行分支） | ✅ |

## E4 正式训练核心参数

配置文件：[exp4_schemashift.yaml](exp4_schemashift.yaml)

| YAML 路径 | 当前值 | 说明 |
|-----------|--------|------|
| `algorithm.schemashift.beta` | 0.25 | StratAdv 分层权重 |
| `data.max_prompt_length` | 10240 | 最大 prompt 长度 |
| `data.max_response_length` | 4096 | 最大 response 长度 |
| `rollout.group_size` | 1 | 数据侧 9 条/task |
| `rollout.max_turns` | 5 | 交互式 replay 最大轮次 |
| `trainer.total_training_steps` | 300 | 训练步数 |
| `actor.ppo_micro_batch_size_per_gpu` | 3 | 每卡 micro batch |

可通过环境变量覆盖：`N_GPUS`、`BETA`、`TOTAL_STEPS`、`SAVE_FREQ`、`TEST_FREQ`

## 注意

- `ds_zero2.json` 保持 JSON 格式（DeepSpeed 不支持 YAML）
- Live MCP 配置是可选并行分支，不影响默认 replay 训练路径
- 所有路径使用项目根目录相对路径，禁止写死机器绝对路径
