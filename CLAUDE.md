# CLAUDE.md — LiveMCP-GRPO 工作入口

本项目的权威方案文档是 `docs/OVAL-MCP.md`。进入开发前先读：

1. `docs/OVAL-MCP.md`
2. `AGENTS.md`
3. `README.md`
4. 与当前任务相关的 `data/README.md` 或输出报告

若文档之间出现冲突，处理顺序是：

```text
先核对代码/报告事实
-> 更新 docs/OVAL-MCP.md
-> 更新 AGENTS.md / README.md / data README
-> 再改实现
```

## 当前状态

- **模型**：Qwen3-4B（`models/Qwen3-4B`，BF16 ~8GB）
- **默认环境**：8×L20 44GB（conda env: `arl`）
- **SFT cold-start**：本地产物存在于 `outputs/sft_cold_start_4b/final`，训练报告在 `outputs/sft_cold_start_4b/training_report.json`
- **GRPO 训练**：E4 交互式静态 replay 路线（`schemashift_replay`）+ OVAL live MCP 路线（`schemashift_oval`）均可入口

### OVAL-MCP Phase 1 基础设施

| 组件 | 路径 | 状态 |
|------|------|------|
| **OvalMCPWorkerContext** | `src/agent_loop/oval_mcp_worker.py` | ✅ 测试通过（17/18） |
| **SchemaShiftOvalLoop** | `src/agent_loop/schemashift_oval_loop.py` | ✅ verl agent loop |
| **Oval Reward Function** | `src/reward/oval_reward_fn.py` | ✅ verl reward fn |
| **Live MCP servers** | `src/live_mcp/servers/calendar/`, `shopping/` | ✅ 启动正常 |
| **Audit + Safety + R_task** | `src/oval_mcp/verifier/`, `rewards/` | ✅ 删除+重建检测生效 |
| **饱和组跳过（§9.2-9.3）** | `src/training/schemashift_grpo_estimator.py` | ✅ 已接线（std(J)<min_group_std → advantage=0）|

已验证链路：Live MCP server → tool execution → audit event → serialization → safety verifier → reward compute → GRPO advantage（含饱和组跳过）→ smoke test 2 steps 跑通（score_range=[-0.51, 0.54]）

### OVAL-MCP Phase 2（已实现，待 GPU 训练验证）

| 组件 | 路径 | 状态 |
|------|------|------|
| **F_gamma (progress shaping)** | `src/oval_mcp/rewards/f_gamma.py` | ✅ 代码完整，单测通过 |
| **P_process (event process score)** | `src/oval_mcp/rewards/p_process.py` | ✅ 代码完整，单测通过 |
| **2D Stratified Advantage** | `src/training/schemashift_grpo_estimator.py` | ✅ 已实现 (perturbation_level × scenario_type，带 fallback chain) |
| **LATA (Length-Aware Token Allocation)** | `src/oval_mcp/training/lata.py` | ✅ 三种模式：none/sqrt_l/norm |
| **LambdaState (stall protection)** | `src/oval_mcp/training/lambda_state.py` | ✅ dual ascent + stall（连续 unsafe 冻结 λ_safe） |
| **Saturation Detection** | `src/oval_mcp/training/saturation.py` | ✅ std(J) < min_group_std → advantage=0 |
| **Batch-level GRPO fallback** | `src/training/schemashift_grpo_estimator.py` | ✅ 非 E4 数据自动降级到 batch-level z-score |
| **register_estimator verl 集成** | `src/training/register_estimator.py` | ✅ monkey-patch + non_tensor_batch 传递 + λ_safe 跨 batch 更新 |
| **oval_reward_fn 消融开关** | `src/reward/oval_reward_fn.py` | ✅ 环境变量控制（I_SHAPE/I_PROCESS/LAMBDA_*/GAMMA） |
| **M4/F/P ablation** | 训练实验 | ❌ 未跑过 GPU 训练验证 |

Phase 2 全部代码已实现并接线到 estimator 和 reward function 中，但从未在 GPU GRPO 训练中启用验证。消融实验（M4/M4+F/M4+P/M4+F+P）待执行。

## 训练路线

项目有三条训练路线：

| 路线 | Agent Loop | Reward Fn | 入口 | 用途 |
|------|-----------|-----------|------|------|
| **Direct GRPO** | `schemashift_replay` | `schemashift_reward_fn.py` | `python scripts/train_grpo.py --config configs/grpo_direct.yaml` | 交互式静态 replay |
| **Cold GRPO** | `schemashift_replay` | `schemashift_reward_fn.py` | `python scripts/train_grpo.py --config configs/grpo_cold.yaml` | SFT cold-start → replay |
| **OVAL GRPO** | `schemashift_oval` | `oval_reward_fn.py` | `bash scripts/oval_grpo_smoke.sh` | live MCP + audit + safety |

`scripts/train_grpo.py` 是 GRPO 训练入口；`OVAL GRPO` 路线使用 `--reward-fn src/reward/oval_reward_fn.py` 替换默认 reward function。

## 当前开发边界

Phase 1 已闭环：smoke test 2 steps 跑通，饱和组跳过已接线。

Phase 2 待办：在 8×L20 GPU 环境跑消融实验（M4/M4+F/M4+P/M4+F+P），验证 F_gamma 和 P_process 的实际效果。

```text
live MCP servers (calendar+shopping)
-> SchemaShiftOvalLoop (模型生成 + 真实执行 + 审计)
-> oval_reward_fn.py (R_task + C_safety + J，Phase 2 可启用 F_gamma / P_process)
-> verl GRPO training step（含饱和组跳过）
```

不要切换 RL 框架，不要从零自建大规模训练语料。

## 必守约束

- 正式 RL 基于 `verl`；`TRL` 只用于 SFT cold-start。
- 训练主源是 Toucan 派生的 replayable `EpisodeSeed`。
- 自建的是 schema perturbation、distractor、verifier、environment 和 eval set。
- 训练脚本不得写死 GPU 数、batch size、micro batch、TP size。
- 项目代码和脚本中的项目文件路径必须以项目根目录为锚点使用相对路径；不要写死 `/data/...`、`/mnt/...` 等机器绝对路径。
- 训练超参必须支持通过脚本命令行参数、环境变量或 Hydra override 注入。
- `data.max_prompt_length` 不得低于 `10240`。
- Ray 临时目录必须使用短路径（默认 `/tmp/ssgrpo_ray`），避免 AF_UNIX socket path 超过 107 bytes。
- 不确定的事实先核验或停下来对齐，不把假设写进实现。

## 常用命令

```bash
# SFT cold-start 复训（8×L20 44GB）
torchrun --nproc_per_node=8 scripts/sft_cold_start.py --config configs/sft_cold_start_4b.yaml

# 交互式静态 replay 训练（Direct GRPO）
python scripts/train_grpo.py --config configs/grpo_direct.yaml

# SFT cold-start → 交互式静态 replay 训练（Cold GRPO）
python scripts/train_grpo.py --config configs/grpo_cold.yaml

# GRPO smoke test
bash scripts/run_grpo_smoke.sh --config configs/grpo_smoke.yaml

# OVAL GRPO smoke test
bash scripts/oval_grpo_smoke.sh

# 单元测试
conda run -n arl python -m pytest tests/

# 轻量检查
conda run -n arl python -m compileall src scripts tests
git diff --check
```

## 验证

文档改动至少跑：

```bash
git diff --check
```

代码改动优先跑：

```bash
conda run -n arl python -m pytest tests/
```

只能轻量检查时至少跑：

```bash
conda run -n arl python -m compileall src scripts tests
git diff --check
```
