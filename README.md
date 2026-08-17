# LiveMCP-GRPO

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![veRL 0.6.1](https://img.shields.io/badge/veRL-0.6.1-orange.svg)](https://github.com/volcengine/verl)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-yellow.svg)](LICENSE)

LiveMCP-GRPO 是面向长程、多工具任务的可执行数据生成与 GRPO 训练系统。它在有状态 MCP
环境中生成用户任务，让 Teacher 真实执行工具链，用 fresh replay 和 provenance 验证标签，
再通过可编程 reward 训练 Policy。

## 当前进度

十域 Live MCP、Teacher 五步生成、Parquet 合同、reward、Policy agent loop 和 GRPO 入口已实现。
项目当前处于逐域数据质量准入和训练前验收阶段，尚未开始正式补产或完整训练。

精确数据行数、逐域完成度、活动任务和卡点必须从当前 run manifest、artifact、
进程与 GPU 状态核实。README 不保存会随运行变化的快照。

## 系统链路

```mermaid
flowchart LR
    A[Domain Contracts] --> B[Dependency Graph]
    B --> C[Live-State Sampling]
    C --> D[Query Teacher]
    D --> E[Action FSM]
    E --> F[Live MCP]
    F --> G[Profile-Boundary Validation]
    G --> H[Fresh Replay + Provenance]
    H --> I[Plain Jaccard]
    I --> J[Parquet Contract]
    J --> K[Policy Rollout + Reward]
```

核心边界：

- domain contract 和真实 MCP trace 决定可执行性；local profile 的引用可见性合同只负责阻止
  sampler-private handle 出现在用户可见文本，Teacher
  不能扩张系统事实；
- opaque backend ID、tool name、raw arguments 和 observation 属于内部执行面，公开 query/continuation/
  terminal 只能消费 public projection；
- terminal 保留 Teacher 原始文本；确定性 private-reference / hidden-tool-name 边界在写入前 fail closed；
- artifact 同时保留审计 provenance 和 canonical public row，训练 consumer 必须重新校验 purpose、hash
  和 reward compatibility；
- 依赖图按 domain 内全部无序工具 pair 由 Teacher 分类；
- query 和参数绑定 session-scoped live state；
- distractor、enum stripping、missing-function 和 irrelevance 在 Teacher 处理前固定；
- PROVE 公开 corpus hard gates 与本地训练可消费性合同分开记录；
- Teacher、Replay 和 Policy rollout 共用同一套 MCP handler 和 schema；
- `prove_baseline` 只使用 PROVE 五组件任务奖励，`oval_full` 才启用 OVAL 扩展。

算法与合同详见 [OVAL-MCP](docs/OVAL-MCP.md)。

## 环境

Teacher 与 Policy 使用不兼容的 vLLM / Transformers 版本，必须分环境。

| 环境 | 用途 | PyTorch | vLLM | Transformers |
|---|---|---:|---:|---:|
| `arl` | Gemma-4 Teacher / GT | 2.10.0+cu128 | 0.19.1 | 5.13.0 |
| `livemcp` | Qwen3-4B-Instruct-2507 rollout / GRPO | 2.8.0+cu128 | 0.11.0 | 4.57.1 |

Teacher：

```bash
export ARL_ENV=/mnt/data2/liuzhanyi/envs/arl
conda activate "$ARL_ENV"
export PYTHON_BIN="$ARL_ENV/bin/python"
```

Policy：

```bash
export LIVEMCP_ENV=/mnt/data2/liuzhanyi/envs/livemcp
conda activate "$LIVEMCP_ENV"
export PYTHON_BIN="$LIVEMCP_ENV/bin/python"
export PYTHONNOUSERSITE=1
```

训练环境可由以下命令创建或核验：

```bash
bash scripts/setup_training_env.sh
bash scripts/setup_training_env.sh --check
```

## 常用命令

生成训练候选：

```bash
livemcp-gen run --mode full --count 500 --val-count 100 \
  --prompt-profile local_trainable_v1 \
  --semantic-gate-profile deterministic_v1
# 或直接调用
python -m src.live_mcp.corpus.cli run --mode full --count 500 --val-count 100 \
  --prompt-profile local_trainable_v1 \
  --semantic-gate-profile deterministic_v1
```

CLI 默认的 paper baseline + diagnostic gate 只产生 `paper_audit`，不能用于 rollout 或训练。

审计：

```bash
livemcp-audit data/runs/<run-id>/train.parquet data/runs/<run-id>/val.parquet
```

构建依赖图缓存：

```bash
livemcp-gen build-cache --model models/Google/Gemma-4-31B-it --api-base http://localhost:8765/v1
```

训练：

```bash
bash scripts/train_grpo.sh --gpus 0,1,2,3
```

多 seed smoke：

```bash
bash scripts/smoke_rollout_reward.sh \
  --gpus 0,1,2,3 \
  --seeds 41,42,43 \
  --steps 3 \
  --reward-profile prove_baseline
```

验证：

```bash
PYTHONNOUSERSITE=1 "$ARL_ENV/bin/python" -m pytest -q \
  tests/test_transport_contract.py
PYTHONNOUSERSITE=1 "$LIVEMCP_ENV/bin/python" -m pytest -q tests/
PYTHONNOUSERSITE=1 "$LIVEMCP_ENV/bin/python" -m compileall src scripts tests
bash -n src/live_mcp/corpus/launcher.sh scripts/train_grpo.sh
git diff --check
```

两套测试环境不可混装：native MCP transport 在 ARL 环境单独验证，Policy 全量测试在
`livemcp` 环境运行；Policy 环境没有可选 `mcp` SDK 时 transport 文件显示为 skip。

完整脚本职责和补产参数见 [scripts/README.md](scripts/README.md)。

## 目录

```text
configs/              MCP suite 与训练配置说明
data/                 活动数据、不可变 run 和数据文档
docs/                 版本化的现役算法与代码边界
scripts/              生成、合并、审计、训练入口
src/live_mcp/         环境、Teacher、Replay、数据生成
src/oval_mcp/         rollout/reward 合同
src/training/         GRPO 配置、estimator 和入口
tests/                回归与数据合同测试
verl/                 vendored verl 0.6.1
```

## 文档

| 文档 | 职责 |
|---|---|
| [当前状态](docs/PROJECT_STATUS.md) | 当前数据、验证结果、阻塞项和下一步 |
| [算法方案](docs/OVAL-MCP.md) | PROVE 对齐边界、五步生成、reward、训练和评测设计 |
| [代码架构](docs/PROVE_ARCHITECTURE.md) | 生产调用链、依赖方向和语义合同边界 |
| [域语义准入](docs/DOMAIN_SEMANTIC_AUDIT.md) | 十域事实逻辑的认证标准和当前矩阵 |
| [数据说明](data/README.md) | 数据 artifact、生成、审计、发布与消费 |
| [脚本说明](scripts/README.md) | 脚本职责、生成命令和 CI 入口 |
| [配置说明](configs/README.md) | 配置入口、默认值和环境变量覆盖方式 |


### 阅读顺序

1. 先读 `docs/PROJECT_STATUS.md`，再按需读算法、架构或域语义准入文档；
2. 运行数据任务前读 `data/README.md` 和 `scripts/README.md`；
3. 涉及当前运行时，仍需现场核实 manifest、artifact、进程和 GPU。

状态事实不得同时在多个文档中展开维护。已完成历史只从 Git 追溯，不在现役文档中重复。

## 许可

Apache License 2.0，见 [LICENSE](LICENSE)。
