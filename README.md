# LiveMCP-GRPO

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.10](https://img.shields.io/badge/PyTorch-2.10-red.svg)](https://pytorch.org/)
[![veRL 0.6.1](https://img.shields.io/badge/veRL-0.6.1-orange.svg)](https://github.com/volcengine/verl)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> GRPO training for multi-step MCP tool-use agents on **10 domains, 188 tools**.

---

## 🏗️ Project Structure

```
📦 livemcp-grpo/
├── src/
│   ├── live_mcp/          # Data synthesis + 10 MCP servers
│   ├── agent_loop/         # verl Agent Loop
│   ├── oval_mcp/           # Reward + constrained GRPO
│   ├── reward/             # verl reward entry
│   └── training/           # verl training components
├── scripts/
│   ├── generate_data.sh       # Data generation (shell entry)
│   ├── generate_data.py       # Data generation (Python entry)
│   ├── train_grpo.sh          # GRPO training (shell entry)
│   ├── train_grpo.py          # GRPO training (Hydra entry)
│   ├── validate_pipeline.py     # End-to-end pipeline validation
│   ├── verify_entities.py     # Entity verification
│   ├── dependency_graph.py    # Dependency graph precompute/rebuild
│   ├── merge_rollout_shards.py # Rollout shard merge with quality gates
│   ├── inspect_prompts.py     # Prompt inspection
│   ├── convert_external_datasets.py  # External dataset conversion
│   └── bench_vllm_throughput.py # vLLM throughput benchmark
├── tests/                     # pytest 测试（待补充）
├── configs/
├── data/
├── verl/                   # verl 0.6.1 (vendored)
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

- [veRL](https://github.com/volcengine/verl) 0.6.1 · vLLM 0.19.1 · FlashInfer
- Teacher: local Gemma-4-31B-it (primary) or external API · Policy: Qwen3-4B

---

## 📄 License

MIT
