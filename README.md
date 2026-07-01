# LiveMCP-GRPO

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.8](https://img.shields.io/badge/PyTorch-2.8-red.svg)](https://pytorch.org/)
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
│   ├── generate_data.sh    # Data generation
│   └── train_grpo.sh       # GRPO training
├── tests/
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
pip install -e .
pip install -e ".[train,rl]"
```

### Generate Training Data

```bash
bash scripts/generate_data.sh --model models/Qwen/Qwen3-32B --count 500 --val-count 100
```

### Train

```bash
bash scripts/train_grpo.sh
bash scripts/train_grpo.sh --gpus 0,1,2,3 --total-steps 300
```

### Validate

```bash
python tests/test_all_domains.py
python -m pytest tests/
```

> **Hardware**: 8×L20 44GB. Teacher Qwen3-32B (vLLM TP=4), Policy Qwen3-4B.

---

## 🛠️ Tech Stack

- [veRL](https://github.com/volcengine/verl) 0.6.1 · vLLM 0.11.0 · FlashAttention-2 · FlashInfer
- Teacher: Qwen3-32B · Policy: Qwen3-4B

---

## 📄 License

MIT
