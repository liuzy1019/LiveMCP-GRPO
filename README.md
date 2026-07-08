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
│   ├── validate_pipeline.py   # End-to-end pipeline validation
│   ├── test_domain_integrity.py    # Domain topology & logic integrity
│   ├── test_livemcp_logic_regressions.py  # Logic regression tests
│   ├── test_runner.py         # Test orchestration
│   ├── verify_entities.py     # Entity verification
│   ├── check_data.py          # Parquet data inspection
│   └── inspect_prompts.py     # Prompt inspection
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
bash scripts/generate_data.sh --model gemini-2.5-flash --api-base https://your-proxy/v1 --count 500 --val-count 100
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

> **Hardware**: 8×L20 44GB. Teacher: Gemini via cloud API. Policy: Qwen3-4B (vLLM local).

---

## 🛠️ Tech Stack

- [veRL](https://github.com/volcengine/verl) 0.6.1 · vLLM 0.19.1 · FlashInfer
- Teacher: user-specified (recommended: Gemini via OpenAI-compatible API) · Policy: Qwen3-4B

---

## 📄 License

MIT
