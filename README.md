# LiveMCP-GRPO

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.10](https://img.shields.io/badge/PyTorch-2.10-red.svg)](https://pytorch.org/)
[![veRL 0.6.1](https://img.shields.io/badge/veRL-0.6.1-orange.svg)](https://github.com/volcengine/verl)

> **基于 PROVE 框架的多步 MCP 工具调用 GRPO 训练**：状态机数据合成 + 实时执行验证 + 多组件可编程奖励。
> 当前覆盖 **10 个 MCP domain、190 个工具**，在 session-scoped state isolation 下做 live-execution RL。

## 核心方法

本项目实现了 PROVE（Programmatic Rewards On Verified Environments）框架的三个组件：

| 组件 | 描述 | 对应模块 |
|------|------|----------|
| **Live MCP Environments** | 带 session-scoped 状态隔离的 MCP 服务器库，数据合成和 RL 训练共用 | `src/live_mcp/` |
| **Grounded State-Machine Data Synthesis** | 自动发现工具依赖图 → 实时状态采样 grounding → 状态机编排生成多轮对话 → 重放验证 + Jaccard 去重 | `scripts/generate_data.py` |
| **Multi-Component Programmatic Reward** | 五组件奖励：R_validity + R_coverage + R_efficiency + R_name + R_arg，无需外部 judge 模型 | `src/reward/oval_reward_fn.py` |

### 数据合成五步（PROVE §3.2）

```
Step 1. Auto-Discovered Dependency Graph（自动发现工具依赖图）
Step 2. Live-State Sampling（实时状态采样，grounded query generation）
Step 3. State-Machine Orchestrator（状态机编排器，5 组状态驱动）
Step 4. Robustness Knobs（鲁棒性注入：distractor/enum-strip/irrelevance/missing-func）
Step 5. Replay Validation & Dedup（重放验证 + Jaccard 0.70 去重）
```

---

## 🏗️ Project Structure

```
📦 livemcp-grpo/
├── src/
│   ├── live_mcp/          # Data synthesis + MCP servers（state-machine teacher, state seeder, oracle）
│   ├── agent_loop/         # verl Agent Loop（single-call + initial-state hash + final-state evidence）
│   ├── oval_mcp/           # OVAL envs + reward components
│   ├── reward/             # verl reward entry（oval_reward_fn.py: R_val + R_cov + R_eff + R_name + R_arg）
│   └── training/           # verl training components（GRPO estimator, hooks, trainer config）
├── scripts/
│   ├── generate_data.sh       # 数据生成 shell 入口（自动管理 vLLM + GPU 自适应）
│   ├── generate_data.py       # 数据生成 Python 入口（5 步状态机管线）
│   ├── train_grpo.sh          # GRPO 训练 shell 入口
│   ├── build_dependency_cache.py # 构建依赖图缓存（C(n,2) 无序 pair）
│   ├── validate_generation_pipeline.py   # 端到端管线验证
│   ├── audit_tool_semantics.py     # 实体验证
│   ├── merge_generation_shards.py # 生成分片合并（含质量门禁）
│   ├── audit_generated_data.py # 正式 Parquet 逐行生产合同审计
│   ├── serve_policy_model.sh   # 独立策略模型 vLLM 服务
│   └── bench_vllm_throughput.py # vLLM 吞吐量基准
├── configs/
│   ├── live_mcp/               # 各 domain 与 ten_domain_suite.yaml
│   └── livemcp_rollout.yaml    # LiveMCP rollout 注册
├── data/
│   ├── dependency_graphs/      # 各 domain 工具依赖图缓存
│   ├── runs/                   # 生成产出（每次运行独立子目录）
│   ├── train.parquet           # 成功生成后指向最新 run 的符号链接
│   ├── val.parquet             # 成功生成后指向最新 run 的符号链接
│   └── README.md               # 数据合同与生成说明
├── reference/                  # 参考论文（PROVE / COVERT）
├── tests/                      # pytest 测试
├── verl/                       # verl 0.6.1（vendored, editable install）
├── pyproject.toml
└── requirements.txt
```

正式训练入口为 `bash scripts/train_grpo.sh`，其唯一 Python 委托目标是
`src/training/run_grpo.py`，不保留第二套兼容训练入口。

---

## 🚀 Quick Start

### Activate the verified environment

```bash
export ARL_ENV=/mnt/data2/liuzhanyi/envs/arl
conda activate "$ARL_ENV"
export PYTHON_BIN="$ARL_ENV/bin/python"
```

也可以用统一运行时入口完成 prefix、CUDA/FlashInfer JIT 路径注入和依赖检查：

```bash
bash scripts/run_in_runtime_environment.sh --check
bash scripts/run_in_runtime_environment.sh -- bash scripts/generate_data.sh --count 50 --val-count 10
```

本机的 Conda 名称索引仍把 `arl` 指向已不存在的 `/mnt/data1/.../envs/arl`，因此当前不要使用 `conda activate arl`。生成和训练脚本均支持显式解释器变量。

### Install

```bash
pip install -e ./verl
pip install -e ".[train,rl]"
```

### Generate Training Data

```bash
# 默认使用本地 Gemma-4-31B-it
bash scripts/generate_data.sh --count 500 --val-count 100
```

### Train

```bash
bash scripts/train_grpo.sh
bash scripts/train_grpo.sh --gpus 0,1,2,3 --total-steps 300
```

### Validate

```bash
python scripts/validate_generation_pipeline.py --stages 1,2
python -m compileall src scripts tests
```

> **Hardware（当前环境）**: 8×A10 22GB（也支持 4×A10，`generate_data.sh` 自动检测）。Teacher: 本地 Gemma-4-31B-it。Policy: Qwen3-4B (vLLM local)。脚本自动检测 GPU，不绑定特定硬件。

---

## 🛠️ Tech Stack

- [veRL](https://github.com/volcengine/verl) 0.6.1 · vLLM 0.19.1 · FlashInfer · GRPO
- Teacher: 本地 Gemma-4-31B-it（对齐 PROVE 论文 Teacher）· Policy: Qwen3-4B（vLLM local serving）
- Multi-component programmatic reward: R_validity + R_coverage + R_efficiency + R_name + R_arg（论文 §3.3）
