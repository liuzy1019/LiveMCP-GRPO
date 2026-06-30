# CLAUDE.md — LiveMCP-GRPO 工作入口

本项目的权威方案文档是 `docs/OVAL-MCP.md`。进入开发前先读：

1. `docs/OVAL-MCP.md`
2. `AGENTS.md`
3. `README.md`
4. 与当前任务相关的 `data/README.md` 或 `REVIEW.md`

若文档之间出现冲突，处理顺序是：

```text
先核对代码/报告事实
-> 更新 docs/OVAL-MCP.md
-> 更新 AGENTS.md / README.md / data README
-> 再改实现
```

## 当前状态

- **模型**：Teacher Qwen3-32B-Instruct（vLLM TP=4），Policy Qwen3-4B（`models/Qwen3-4B`）
- **默认环境**：8×L20 44GB（conda env: `arl`）
- **数据生成**：500+100 全量生成进行中，核心管线 bug 已修复（P0/P1 全部闭环）
- **GRPO 训练**：待数据生成完成后执行 smoke test
- **SFT cold-start**：已清除

### 管线状态

| 组件 | 状态 | 说明 |
|------|------|------|
| **数据生成管线** | ✅ | Parquet 序列化修复，28/28 对抗审查通过 |
| **OVAL Agent Loop** | ✅ | terminal event 修复 + budget 修复 |
| **OVAL Reward** | ✅ | success_criteria 参与计算 + 全部场景覆盖 |
| **GRPO Estimator** | ✅ | 饱和组跳过 + 2D 分层 advantage |
| **GPU 训练验证** | ⏳ | 待数据生成完成 |

### 已验证链路

```
live MCP servers (10 domains)
→ PROVE Teacher (LLM-in-the-loop, Qwen3-32B)
→ 真实 MCP 执行 → oracle trace
→ Jaccard dedup (0.70, 位置感知)
→ Parquet 序列化 (success_criteria JSON string)
→ oval_reward_fn.py (R_task + C_safety, 可选 F_gamma/P_process)
→ verl GRPO training
```

## 训练路线

项目只有一条主训练路线（OVAL GRPO）：

| 路线 | Agent Loop | Reward Fn | 入口 | 状态 |
|------|-----------|-----------|------|------|
| **OVAL GRPO** | `livemcp_oval` | `oval_reward_fn.py` | `bash scripts/train_grpo.sh` | ✅ 主路线 |

`scripts/train_grpo.py` 是 GRPO 训练 Python 入口（Hydra），`src/training/run_grpo.py` 是正式训练入口。
配置由 `src/training/trainer_config.py` (PyTorch Lightning 风格) 统一管理，支持 GPU tier 自适应默认值和 `OVAL_*` 环境变量覆盖。

## 必守约束

- 正式 RL 基于 `verl`；`TRL` 只用于 SFT cold-start。
- 训练主源是 PROVE state-machine teacher（LLM-in-the-loop）+ live MCP 执行。
- 训练脚本不得写死 GPU 数、batch size、micro batch、TP size。
- 项目代码和脚本中的项目文件路径必须以项目根目录为锚点使用相对路径；不要写死机器绝对路径。
- 训练超参必须支持通过脚本命令行参数、环境变量或 Hydra override 注入。
- `data.max_prompt_length` 不得低于 `10240`。
- Ray 临时目录必须使用短路径（默认 `/tmp/ssgrpo_ray`），避免 AF_UNIX socket path 超过 107 bytes。
- 不确定的事实先核验或停下来对齐，不把假设写进实现。
- 不要切换 RL 框架。

## 已知设计限制

| 问题 | 说明 |
|------|------|
| 非法 tool JSON 不产生 AuditEvent | 模型输出格式错误时只给 error observation，不创建审计事件。修复需要跨模块类型扩展 |
| reward 函数用最后 observation 近似 final state | 连续 tool_call 时精确值验证可能漏检（小概率，有 seen_ids fallback） |
| perturbation 仅 teacher 阶段 | PROVE 设计：只在 oracle 采集时扰动，训练环境保持干净 |

## 常用命令

```bash
# 数据生成（自动检测模型大小，选择最优并行策略）
bash scripts/generate_data.sh --model models/Qwen/Qwen3-8B --count 500
bash scripts/generate_data.sh --model models/Qwen/Qwen3-32B --count 500 --val-count 100
bash scripts/generate_data.sh --model models/Qwen/Qwen3-8B --domain calendar --count 200

# OVAL GRPO 训练
bash scripts/train_grpo.sh

# 单元测试
conda run -n arl python -m pytest tests/

# 管线端到端烟雾测试
conda run -n arl python tests/test_production_smoke.py

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
