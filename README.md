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

精确数据行数、逐域完成度、活动任务、卡点和下一步只在
[项目进度](docs/PROJECT_STATUS.md) 维护。不要从 README、旧 run 名或生成日志推断现役状态。

## 系统链路

```mermaid
flowchart LR
    A[Dependency Graph] --> B[Live-State Sampling]
    B --> C[State-Machine Teacher]
    C --> D[Robustness Plan]
    D --> E[Live Execution]
    E --> F[Fresh Replay + Provenance]
    F --> G[Canonical Replay + Jaccard]
    G --> H[Parquet]
    H --> I[GRPO]
```

核心边界：

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
docs/                 进度、算法、已知问题和 changelog
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
| [算法方案](docs/OVAL-MCP.md) | PROVE 对齐边界、五步生成、reward、训练和评测设计 |
| [项目进度](docs/PROJECT_STATUS.md) | 当前进度、数据目标、卡点和下一步；现役状态唯一来源 |
| [逐域认证](docs/DOMAIN_SEMANTIC_AUDIT.md) | 十域依赖图、Teacher、artifact 与 rollout 的当前认证状态 |
| [数据说明](data/README.md) | 数据 artifact、生成、审计、发布与消费 |
| [脚本说明](scripts/README.md) | 脚本职责、生成命令和 CI 入口 |
| [配置说明](configs/README.md) | 配置入口、默认值和环境变量覆盖方式 |

| [约束约定](AGENTS.md) | AI agent 工程约束、红线、Git 约定 |


### 阅读顺序

1. 新接手项目先读 `docs/PROJECT_STATUS.md`；
2. 运行数据任务前读 `data/README.md` 和 `scripts/README.md`；
3. 修改算法或合同前读 `docs/OVAL-MCP.md`；
4. 逐域认证时读 `docs/DOMAIN_SEMANTIC_AUDIT.md`。

状态事实不得同时在多个文档中展开维护。已完成历史只从 Git 追溯，不在现役文档中重复。

## 许可

Apache License 2.0，见 [LICENSE](LICENSE)。
