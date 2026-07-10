# LiveMCP-GRPO

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.10](https://img.shields.io/badge/PyTorch-2.10-red.svg)](https://pytorch.org/)
[![veRL 0.6.1](https://img.shields.io/badge/veRL-0.6.1-orange.svg)](https://github.com/volcengine/verl)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **基于 PROVE 框架的多步 MCP 工具调用 GRPO 训练**：状态机数据合成 + 实时执行验证 + 多组件可编程奖励。
> 当前覆盖 **10 个 MCP domain、188 个工具**，在 session-scoped state isolation 下做 live-execution RL。

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
│   ├── train_grpo.py          # GRPO 训练 Hydra 入口
│   ├── dependency_graph.py    # 依赖图预计算/重建（n² pairwise LLM 分类）
│   ├── validate_pipeline.py   # 端到端管线验证
│   ├── verify_entities.py     # 实体验证
│   ├── merge_rollout_shards.py # Rollout 分片合并（含质量门禁）
│   ├── inspect_prompts.py     # Prompt 内容检查
│   ├── convert_external_datasets.py  # 外部数据集转换（When2Call / xLAM-Irrelevance）
│   └── bench_vllm_throughput.py # vLLM 吞吐量基准
├── configs/
│   ├── live_mcp/               # 各 domain 配置（banking/calendar/.../suite_mvp.yaml）
│   ├── agent_loop.yaml         # Agent loop 注册
│   └── ds_zero2.json           # DeepSpeed ZeRO-2 配置
├── data/
│   ├── dependency_graphs/      # 各 domain 工具依赖图缓存
│   ├── external/               # 外部数据集
│   ├── runs/                   # 生成产出（每次运行独立子目录）
│   ├── experiments/            # 实验记录（config.json + result.json）
│   ├── train.parquet           # 符号链接 → 最新 run 的 train.parquet
│   └── val.parquet             # 符号链接 → 最新 run 的 val.parquet
├── reference/                  # 参考论文（PROVE / TOUCAN）
├── tests/                      # pytest 测试
├── verl/                       # verl 0.6.1（vendored, editable install）
├── pyproject.toml
└── requirements.txt
```

---

## 🚀 Quick Start

### Install

```bash
pip install -e ./verl
pip install -e ".[train,rl]"
```

### Generate Training Data

```bash
# 默认使用本地 Gemma-4-31B-it
bash scripts/generate_data.sh --count 500 --val-count 100
# 也可指定外部 API
bash scripts/generate_data.sh --model gemini-2.5-flash --api-base https://your-proxy/v1 --count 500 --val-count 100
```

### Train

```bash
bash scripts/train_grpo.sh
bash scripts/train_grpo.sh --gpus 0,1,2,3 --total-steps 300
```

### Validate

```bash
python scripts/validate_pipeline.py
python -m compileall src scripts tests
```

> **Hardware（当前环境）**: 8×A10 22GB（也支持 4×A10，`generate_data.sh` 自动检测）。Teacher: 本地 Gemma-4-31B-it（也可通过 `--api-base` 接入外部 API）。Policy: Qwen3-4B (vLLM local)。脚本自动检测 GPU，不绑定特定硬件。

---

## 🛠️ Tech Stack

- [veRL](https://github.com/volcengine/verl) 0.6.1 · vLLM 0.19.1 · FlashInfer · GRPO
- Teacher: 本地 Gemma-4-31B-it（对齐 PROVE 论文 Teacher）· Policy: Qwen3-4B（vLLM local serving）
- Multi-component programmatic reward: R_validity + R_coverage + R_efficiency + R_name + R_arg（论文 §3.3）

---

## 📄 License

MIT
