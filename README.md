# LiveMCP-GRPO

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![veRL 0.6.1](https://img.shields.io/badge/veRL-0.6.1-orange.svg)](https://github.com/volcengine/verl)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-yellow.svg)](LICENSE)

> State-grounded trajectory synthesis and GRPO training for long-horizon,
> multi-tool agents in live MCP environments.

LiveMCP-GRPO connects stateful MCP servers, live-state query synthesis, teacher
execution, replay verification, programmatic rewards, and GRPO in one auditable
pipeline. It follows the verified-environment principles introduced by
[PROVE](https://arxiv.org/abs/2606.03892) and provides optional OVAL extensions
for process feedback, safety constraints, progress shaping, and length-aware
token allocation.

## Motivation

Long-horizon tool-use data must establish more than syntactic correctness:

- task entities must exist in the current environment;
- later calls must consume values or state produced by earlier calls;
- every call must execute against the real schema and handler;
- labels must be reproducible in a fresh session;
- rewards must separate validity, coverage, efficiency, tool choice, and arguments.

Static templates and text-only checks cannot prove these properties. LiveMCP-GRPO
therefore separates environment state, internal execution, and user-visible
language, and emits training artifacts only from replay-qualified evidence.

## Method

### Verified Stateful Environments

The repository includes ten stateful MCP domains: `banking`, `calendar`, `crm`,
`email`, `filesystem`, `food_delivery`, `issue_tracker`, `payments`, `shopping`,
and `team_chat`.

Each domain combines executable MCP handlers with entity, state-transition,
read-only probe, value-flow, dependency, and reference-visibility contracts.
These contracts define what the generator may claim and which references may be
shown to a user.

### Grounded Trajectory Synthesis

Data generation follows five stages:

1. **Dependency graph** records relation provenance before chain sampling.
2. **Live-state sampling** queries a fresh session and exposes a public projection.
3. **Query and action generation** creates a grounded request and executes it
   through a multi-turn state machine.
4. **Robustness injection** applies distractors, schema perturbations,
   missing-function cases, and irrelevance conditions before execution.
5. **Replay and artifact validation** checks fresh execution, sensitive
   provenance, sequence diversity, semantic boundaries, and Parquet integrity.

Opaque backend IDs, hidden tool names, raw observations, and execution arguments
remain internal. User-visible text is validated against public references.

### Programmatic Reward and GRPO

The `prove_baseline` profile uses the five-component task reward:

```text
R = 0.5 R_validity + 0.5 R_coverage + 0.15 R_efficiency
  + 0.2 R_name + 0.1 R_arg
```

The optional `oval_full` profile adds bounded process feedback, event-log safety
verification, adaptive safety constraints, progress shaping, and length-aware
token allocation. All components are computed from environment and trajectory
evidence; no external judge model is required.

## Contributions

- **Executable synthesis:** tasks and labels are grounded in session-scoped MCP state.
- **Evidence-carrying chains:** dependencies retain value-flow and transition proof.
- **Public-reference isolation:** natural selectors resolve without exposing private IDs.
- **Replay-qualified artifacts:** consumers revalidate purpose and runtime fingerprints.
- **Structured failures:** rejected candidates retain machine-readable trace evidence.
- **Profile separation:** PROVE-aligned rewards and OVAL extensions do not mix silently.

## Architecture

```mermaid
flowchart LR
    A[Domain Contracts] --> B[Dependency Graph]
    B --> C[Live-State Sampling]
    C --> D[Query Teacher]
    D --> E[Action State Machine]
    E --> F[Live MCP]
    F --> G[Boundary Validation]
    G --> H[Fresh Replay + Provenance]
    H --> I[Canonical Artifact]
    I --> J[Policy Rollout]
    J --> K[Programmatic Reward]
    K --> L[GRPO]
```

## Project Structure

```text
LiveMCP-GRPO/
├── configs/                 # MCP suites and training configuration
├── data/dependency_graphs/  # Versioned dependency caches
├── docs/                    # Algorithm and architecture contracts
├── scripts/                 # Audit, environment, and training entrypoints
├── src/
│   ├── agent_loop/          # Multi-turn policy rollout
│   ├── live_mcp/            # Environments, synthesis, replay, artifacts
│   ├── oval_mcp/            # OVAL verifier and training extensions
│   └── training/            # Reward integration and GRPO estimators
├── tests/                   # Contract and regression tests
└── verl/                    # Vendored veRL 0.6.1
```

## Quick Start

The project requires Linux, Python 3.11, CUDA 12.8, and NVIDIA GPUs. Teacher and
policy stacks use incompatible vLLM/Transformers versions and require separate
environments.

```bash
git clone https://github.com/liuzy1019/LiveMCP-GRPO.git
cd LiveMCP-GRPO

# Teacher environment
conda create -n livemcp-teacher python=3.11 pip -y
conda activate livemcp-teacher
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install vllm==0.19.1
python -m pip install -r requirements.txt --no-build-isolation
python -m pip install -e . --no-deps

# Policy / GRPO environment
export LIVEMCP_ENV="$PWD/../envs/livemcp"
bash scripts/setup_training_env.sh
```

Generate replay-qualified training candidates:

```bash
livemcp-gen run \
  --mode full \
  --domain all \
  --suite configs/live_mcp/ten_domain_suite.yaml \
  --model models/Google/Gemma-4-31B-it \
  --count 100 --val-count 20 \
  --prompt-profile local_trainable_v1 \
  --semantic-gate-profile deterministic_v1 \
  --preserve-candidates
```

Audit and train with explicit artifacts:

```bash
livemcp-audit data/runs/<run-id>/train.parquet data/runs/<run-id>/val.parquet

OVAL_TRAIN_FILE=data/runs/<run-id>/train.parquet \
OVAL_VAL_FILE=data/runs/<run-id>/val.parquet \
bash scripts/train_grpo.sh \
  --gpus 0,1,2,3 \
  --reward-profile prove_baseline \
  --experiment-profile custom
```

Use `paper_generation_baseline_v1 + diagnostic_only` for non-trainable mechanism
audits and `local_trainable_v1 + deterministic_v1` for training candidates.
Audit artifacts are rejected by training, rollout, and reward entrypoints.

## Documentation

| Document | Scope |
|---|---|
| [OVAL-MCP](docs/OVAL-MCP.md) | Algorithm, PROVE boundary, rewards, training contracts |
| [Architecture](docs/PROVE_ARCHITECTURE.md) | Call graph, module ownership, trust boundaries |
| [Data](data/README.md) | Artifact schema, generation, audit, consumption |
| [Scripts](scripts/README.md) | Operational entrypoints and arguments |
| [Configuration](configs/README.md) | Configuration files and override rules |

## Citation

```bibtex
@software{livemcp_grpo_2026,
  author = {Zhanyi Liu},
  title  = {LiveMCP-GRPO: State-Grounded Trajectory Synthesis and GRPO Training},
  year   = {2026},
  url    = {https://github.com/liuzy1019/LiveMCP-GRPO}
}
```

## Acknowledgements

Built on [PROVE](https://arxiv.org/abs/2606.03892),
[veRL](https://github.com/volcengine/verl), [vLLM](https://github.com/vllm-project/vllm),
and the [Model Context Protocol](https://modelcontextprotocol.io/).

## License

Apache License 2.0. See [LICENSE](LICENSE).
