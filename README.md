# LiveMCP-GRPO

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![veRL 0.6.1](https://img.shields.io/badge/veRL-0.6.1-orange.svg)](https://github.com/volcengine/verl)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-yellow.svg)](LICENSE)

> Executable data synthesis and GRPO training for long-horizon, multi-tool agents in live MCP environments.

LiveMCP-GRPO 面向需要多轮规划、真实工具调用和状态变化的 agent。系统以
[PROVE](https://arxiv.org/abs/2606.03892) 的 verified-environment 思路为基础，将依赖图、
Live-State sampling、Teacher 执行、fresh replay、可编程奖励和 GRPO 训练连接成一条可审计链路。

本仓库同时提供可选的 OVAL 扩展，包括过程奖励、安全约束、进度 shaping 和长度感知 token
分配。PROVE baseline 与 OVAL 扩展使用独立 profile，不混合报告。

## 问题与动机

长程工具调用训练不只是生成一段看似合理的 tool-call 文本。训练数据必须同时满足：

1. 用户任务引用的实体在当前 session 中真实存在；
2. 多步调用之间存在可证明的参数、实体或状态依赖；
3. Teacher 的每次调用都能在真实 MCP handler 上执行；
4. 训练标签能够通过 fresh replay、provenance 和 artifact readback 重建；
5. reward 区分工具有效性、任务覆盖、调用效率、函数选择和参数正确性。

静态模板、隐藏 backend ID、仅检查最终文本或仅依赖单元测试，都不能满足这些条件。
LiveMCP-GRPO 将状态事实、执行事实和用户可见文本放在不同信任边界内，并在写入训练数据前
执行确定性验证。

## 方法

### 1. Verified MCP environments

系统内置十个有状态 domain：

`banking`、`calendar`、`crm`、`email`、`filesystem`、`food_delivery`、
`issue_tracker`、`payments`、`shopping` 和 `team_chat`。

每个 domain 由以下对象共同定义：

- MCP schema 与真实 handler；
- entity、state predicate 和 state transition contract；
- readonly probe、value binding 和 reference visibility；
- session-scoped state seeder；
- dependency relation 与 scenario chain contract。

### 2. Five-stage data synthesis

生成管线遵循五个阶段：

1. **Dependency graph**：对 domain 内工具 pair 分类，保存 immutable raw ledger，再由本地
   relation audit 生成 eligible graph。
2. **Live-State sampling**：在 fresh session 中查询真实状态，只向 Teacher 暴露 public
   projection，不暴露 sampler-private handle。
3. **Query Teacher + Action FSM**：Query Teacher 生成 grounded 用户任务，Action Teacher 在同一
   MCP 环境中执行多轮工具链。
4. **Robustness**：在 Teacher 执行前固定 distractor、enum stripping、missing-function 和
   irrelevance 条件。
5. **Replay + artifact**：执行 fresh replay、sensitive provenance、plain tool-sequence Jaccard、
   semantic boundary 和 Parquet round-trip 验证。

### 3. Programmatic reward and GRPO

`prove_baseline` 使用五组件任务奖励：

```text
R = 0.5 R_validity
  + 0.5 R_coverage
  + 0.15 R_efficiency
  + 0.2 R_name
  + 0.1 R_arg
```

`oval_full` 在 baseline 之外启用可选扩展：

- bounded process reward；
- event-log safety verifier 与 adaptive safety constraint；
- potential-based progress shaping；
- LATA length-aware token allocation。

所有 reward 都由环境状态、调用轨迹和 canonical task contract 计算，不依赖外部 judge 模型。

## 技术特点

- **事实与文本分离**：治理事实、内部执行面和用户可见面具有独立边界。
- **真实执行优先**：Teacher、fresh replay 和 Policy rollout 共用 MCP handler 与 schema。
- **引用可见性**：自然 selector 可以映射到 canonical ID，但 opaque backend ID 不进入用户文本。
- **依赖证据**：每条 canonical chain 保存 source、target、occurrence 和连续 evidence path。
- **失败可审计**：候选失败以结构化 stage、reason 和 trace evidence 保存，不用异常字符串替代归因。
- **Artifact fail closed**：purpose、schema、runtime、transition 和 reward fingerprint 在消费端重新校验。
- **双环境隔离**：Teacher 与 Policy 使用互斥的 vLLM/Transformers 依赖，避免运行时污染。
- **Profile 隔离**：论文机制审计数据不能被 rollout、reward 或训练入口误消费。

## 系统架构

```mermaid
flowchart LR
    A[Domain Contracts] --> B[Dependency Graph]
    B --> C[Live-State Sampling]
    C --> D[Query Teacher]
    D --> E[Action FSM]
    E --> F[Live MCP]
    F --> G[Local Boundary Validation]
    G --> H[Fresh Replay + Provenance]
    H --> I[Jaccard + Canonical Artifact]
    I --> J[Policy Rollout]
    J --> K[Programmatic Reward]
    K --> L[GRPO]
```

详细调用链和依赖方向见 [PROVE_ARCHITECTURE.md](docs/PROVE_ARCHITECTURE.md)。

## 项目结构

```text
LiveMCP-GRPO/
├── configs/                 # MCP suite、rollout 和训练配置
├── data/
│   └── dependency_graphs/   # 版本化 dependency cache
├── docs/                    # 算法、架构、状态和逐域准入合同
├── scripts/                 # 审计、环境、smoke 和训练入口
├── src/
│   ├── agent_loop/          # Policy 多轮 rollout
│   ├── live_mcp/            # MCP 环境、生成、replay 和 artifact
│   ├── oval_mcp/            # OVAL verifier、reward 和训练扩展
│   ├── reward/              # Reward worker 接口
│   └── training/            # GRPO estimator、配置和入口
├── tests/                   # 合同、回归和端到端结构测试
├── verl/                    # Vendored veRL 0.6.1
├── requirements.txt         # Teacher 环境依赖
└── requirements-train.txt   # Policy/GRPO 环境依赖
```

## Quick Start

### 1. Clone

```bash
git clone https://github.com/liuzy1019/LiveMCP-GRPO.git
cd LiveMCP-GRPO
```

项目要求 Linux、Python 3.11、CUDA 12.8 和 NVIDIA GPU。Teacher 与 Policy 依赖不兼容，必须使用
两个独立环境。

### 2. Teacher environment

```bash
conda create -n livemcp-teacher python=3.11 pip -y
conda activate livemcp-teacher

python -m pip install torch==2.10.0 torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install vllm==0.19.1
python -m pip install -r requirements.txt --no-build-isolation
python -m pip install -e . --no-deps
```

Teacher 模型使用本地路径或 vLLM 可识别的模型标识。Gemma-4-31B 在 A10 上推荐 TP=4。

### 3. Policy / GRPO environment

```bash
export LIVEMCP_ENV="$PWD/../envs/livemcp"
bash scripts/setup_training_env.sh
bash scripts/setup_training_env.sh --check
```

该脚本安装 `requirements-train.txt`、vendored `verl` 和当前项目，不安装 Teacher 依赖。

### 4. Build dependency cache

```bash
livemcp-gen build-cache \
  --model models/Google/Gemma-4-31B-it \
  --api-base http://localhost:8001/v1
```

### 5. Generate data

训练候选必须显式使用 local trainability profile：

```bash
livemcp-gen run \
  --mode full \
  --domain all \
  --suite configs/live_mcp/ten_domain_suite.yaml \
  --model models/Google/Gemma-4-31B-it \
  --count 100 \
  --val-count 20 \
  --prompt-profile local_trainable_v1 \
  --semantic-gate-profile deterministic_v1 \
  --preserve-candidates
```

论文机制审计使用独立 profile，产物不能训练：

```bash
livemcp-gen run \
  --mode full \
  --domain shopping \
  --count 16 \
  --val-count 4 \
  --prompt-profile paper_generation_baseline_v1 \
  --semantic-gate-profile diagnostic_only
```

### 6. Audit artifacts

```bash
livemcp-audit \
  data/runs/<run-id>/train.parquet \
  data/runs/<run-id>/val.parquet

python scripts/validate_generation_pipeline.py --stages 1,2 --domain all
python scripts/audit_prove_domains.py --domain all
```

### 7. Train

```bash
OVAL_TRAIN_FILE=data/runs/<run-id>/train.parquet \
OVAL_VAL_FILE=data/runs/<run-id>/val.parquet \
bash scripts/train_grpo.sh \
  --gpus 0,1,2,3 \
  --reward-profile prove_baseline \
  --experiment-profile custom
```

多 seed rollout/reward smoke：

```bash
bash scripts/smoke_rollout_reward.sh \
  --gpus 0,1,2,3 \
  --seeds 41,42,43 \
  --steps 3 \
  --reward-profile prove_baseline \
  --experiment-profile custom \
  --train-file data/runs/<run-id>/train.parquet \
  --val-file data/runs/<run-id>/val.parquet
```

## Profiles

| Prompt profile | Semantic gate | Artifact purpose | 用途 |
|---|---|---|---|
| `paper_generation_baseline_v1` | `diagnostic_only` | `paper_audit` | PROVE 机制审计 |
| `local_trainable_v1` | `deterministic_v1` | `training_candidate` | 本地训练候选 |

训练、rollout 和 reward 入口会拒绝 `paper_audit`，也会拒绝 profile、fingerprint 或 canonical
contract 不匹配的 artifact。

## 验证

```bash
# Teacher/native MCP
conda activate livemcp-teacher
PYTHONNOUSERSITE=1 python -m pytest -q \
  tests/test_transport_contract.py

# Policy/GRPO
PYTHONNOUSERSITE=1 "$LIVEMCP_ENV/bin/python" -m pytest -q tests/

# Static checks
bash -n src/live_mcp/corpus/launcher.sh scripts/*.sh
git diff --check
```

native MCP transport 与 Policy/GRPO 依赖必须分别验证，不能为了单环境全绿混装两套依赖。

## 文档

| 文档 | 内容 |
|---|---|
| [OVAL-MCP.md](docs/OVAL-MCP.md) | PROVE 边界、生成机制、reward 与训练合同 |
| [PROVE_ARCHITECTURE.md](docs/PROVE_ARCHITECTURE.md) | 生产调用链、模块职责与信任边界 |
| [DOMAIN_SEMANTIC_AUDIT.md](docs/DOMAIN_SEMANTIC_AUDIT.md) | 十域事实逻辑和逐域准入标准 |
| [PROJECT_STATUS.md](docs/PROJECT_STATUS.md) | 当前数据门、阻塞项和下一步 |
| [data/README.md](data/README.md) | Artifact、生成、审计与消费 |
| [scripts/README.md](scripts/README.md) | 脚本职责和运行参数 |
| [configs/README.md](configs/README.md) | 配置入口和覆盖方式 |

## 技术栈

- Training framework: [veRL](https://github.com/volcengine/verl) 0.6.1
- Teacher serving: vLLM 0.19.1 + Transformers 5
- Policy rollout: vLLM 0.11.0 + Transformers 4.57
- Policy model: Qwen3-4B
- Teacher model: Gemma-4-31B
- Environment protocol: MCP
- Configuration: Hydra / OmegaConf
- Artifact format: Parquet / PyArrow

## Acknowledgements

- [PROVE](https://arxiv.org/abs/2606.03892) for verified live-environment synthesis and programmatic reward design.
- [veRL](https://github.com/volcengine/verl) for the distributed RL training framework.
- [vLLM](https://github.com/vllm-project/vllm) for model serving and rollout inference.
- [Model Context Protocol](https://modelcontextprotocol.io/) for the tool-server protocol.

## License

Apache License 2.0. See [LICENSE](LICENSE).
