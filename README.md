# LiveMCP-GRPO

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.10](https://img.shields.io/badge/PyTorch-2.10-red.svg)](https://pytorch.org/)
[![veRL 0.6.1](https://img.shields.io/badge/veRL-0.6.1-orange.svg)](https://github.com/volcengine/verl)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-yellow.svg)](LICENSE)

> **面向长程、多工具任务的可执行数据生成与 GRPO 训练系统。**
>
> LiveMCP-GRPO 在有状态 MCP 环境中生成用户任务、执行 Teacher 工具链、重放验证轨迹，
> 并使用可编程奖励训练 Policy。

## Problem

长程工具调用数据不仅要在文本上看起来合理，还必须满足四个执行条件：

1. 参数能够从用户输入或此前 observation 中获得；
2. 工具调用在隔离的真实状态中可执行；
3. 失败、恢复和终止行为与执行事实一致；
4. 训练端能够复现生成时使用的 schema、状态和 reward contract。

仅由语言模型直接生成 tool-call JSON，无法保证这些条件。LiveMCP-GRPO 将任务生成、真实执行、
fresh replay、provenance 和训练消费校验放在同一条流水线中。

## Method

```mermaid
flowchart LR
    A[Dependency Graph] --> B[Live-State Sampling]
    B --> C[State-Machine Teacher]
    C --> D[Robustness Injection]
    D --> E[Fresh Replay + Provenance]
    E --> F[Jaccard Deduplication]
    F --> G[Parquet + GRPO]
```

### Live MCP Environments

- 每个任务使用独立 session 和确定性初始状态；
- Teacher、Replay 和 Policy rollout 共用 MCP handler；
- mutation 记录真实 state delta，异常调用回滚当前事务；
- schema、transition、initial state 和 reward 均保存可验证指纹。

### Grounded State-Machine Synthesis

- 对 domain 内全部无序工具 pair 分类并构建依赖图；
- 从 live state 中采样真实实体，约束 query 和 tool arguments；
- Teacher 在多轮状态机中选择 tool call、恢复动作或 terminal；
- distractor、enum stripping、missing function 和 irrelevance 在 Teacher 执行前固定。

### Replay-Verified Ground Truth

- 在 fresh session 中重新执行轨迹；
- 检查 sensitive-parameter provenance；
- 验证 terminal、round contract、hidden tools 和 success criteria；
- 按位置感知的 tool-call sequence Jaccard 0.70 全局去重。

### Programmatic Reward

任务奖励由五个可解释组件组成：

```text
R_task = w_val * R_validity
       + w_cov * R_coverage
       + w_eff * R_efficiency
       + w_name * R_name
       + w_arg * R_argument
```

奖励直接消费执行事件和 GT workflow，不依赖外部 judge 模型。

## Data

数据目录结构、Parquet 字段合同、生成参数和审计方法见 [data/README.md](data/README.md)。

## Quick Start

### 1. Environment

运行环境要求 Python 3.11，并需要根据本机 CUDA 安装兼容的 PyTorch 与 vLLM。
建议使用独立 Conda 环境：

```bash
git clone https://github.com/liuzy1019/LiveMCP-GRPO.git
cd LiveMCP-GRPO

conda create -n livemcp python=3.11 -y
conda activate livemcp

# 按本机 CUDA 版本安装 PyTorch 和 vLLM，然后安装项目依赖。
python -m pip install -r requirements.txt
python -m pip install -e ./verl
python -m pip install -e .
```

本地 Teacher 和 Policy 权重不随仓库分发，需要分别放入 `models/` 或通过脚本参数指定。

### 2. Generate Data

```bash
# 生成 500 条 train 和 100 条 validation 数据。
bash scripts/generate_data.sh --count 500 --val-count 100

# 限制可见 GPU 数量。
GPU_COUNT=4 bash scripts/generate_data.sh --count 500 --val-count 100
```

### 3. Validate Data

```bash
python scripts/validate_generation_pipeline.py --stages 1,2
python scripts/audit_generated_data.py data/train.parquet data/val.parquet
python -m pytest tests/
```

### 4. Train

```bash
bash scripts/train_grpo.sh
bash scripts/train_grpo.sh --gpus 0,1,2,3 --total-steps 300
```

训练脚本自动解析 GPU 数量和运行配置；GPU、batch size、micro batch 和 tensor parallel size
均可通过命令行、环境变量或 Hydra override 注入。

## Project Structure

```text
livemcp-grpo/
├── configs/                 # MCP suite 与 rollout 配置
├── data/                    # 依赖图缓存、生成合同和本地数据入口
├── scripts/                 # 数据生成、审计、合并和训练入口
├── src/
│   ├── live_mcp/            # 有状态 MCP 环境与生成状态机
│   ├── agent_loop/          # veRL multi-turn agent loop
│   ├── oval_mcp/            # verifier、reward components 与训练状态
│   ├── reward/              # veRL reward entry
│   └── training/            # GRPO 配置、estimator 与启动逻辑
├── tests/                   # 单元、合同与对抗性回归测试
└── verl/                    # vendored veRL 0.6.1 source
```

## Reproducibility

- 数据生成依赖本地 Teacher 推理，吞吐和保留率会随模型、GPU 和 seed 变化；
- dependency cache 只消除 pair-classification 成本，不消除 Teacher rollout、replay 和去重成本；
- 生成任务时应记录模型 revision、seed、schema fingerprint 和 reward fingerprint。

## Tech Stack

- Training: [veRL](https://github.com/volcengine/verl) 0.6.1 + GRPO
- Inference: vLLM 0.19.1 + FlashInfer
- Teacher: Gemma-4-31B-it
- Policy: Qwen3-4B
- Data: PyArrow / Parquet

## Acknowledgements

- [veRL](https://github.com/volcengine/verl) 提供分布式 RL 训练基础设施；
- PROVE 提供 verified environment 和多组件任务奖励的研究参考；
- Gemma 与 Qwen 系列模型用于 Teacher 和 Policy 实验。

## License

本项目使用 [Apache License 2.0](LICENSE)。vendored 依赖仍遵循其各自许可证。
