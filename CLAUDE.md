# CLAUDE.md — LiveMCP-GRPO

Design doc: `docs/OVAL-MCP.md`. Read it before making changes.
Behavioral/data-contract changes are recorded in `docs/CHANGELOG.md`.

## Environment

### 当前可用环境

```bash
conda activate arl          # Python 3.11.15, PyTorch 2.10.0+cu128
nvidia-smi                  # 确认 GPU 可用（A10 ×8, 22GB/卡, Driver 570.195.03）
```

关键版本：

| 组件 | 版本 | 备注 |
|------|------|------|
| Python | 3.11.15 | — |
| PyTorch | 2.10.0+cu128 | — |
| CUDA (nvcc) | 12.8.93 | conda 安装 |
| vLLM | 0.19.1 | >=0.11.0 均兼容 |
| transformers | 5.13.0 | — |
| flashinfer-python | 0.6.12 | >=0.6.4 均兼容 |
| flash-attn | 未安装 | vLLM V1 默认 flashinfer attention backend |
| ray | 2.54.1 | — |
| datasets | 4.8.4 | — |
| deepspeed | 0.19.2 | — |
| peft | 0.18.1 | — |
| accelerate | 1.14.0 | — |
| trl | 0.29.1 | — |
| tensordict | 0.10.0 | — |
| xformers | 0.0.32.post1 | — |

> **vLLM 0.11.0 → 0.19.1**: 持续跟随上游。PROVE 论文未指定具体 vLLM 版本（仅写 "VERL + vLLM"），0.19.1 完全兼容。版本自动检测，脚本不强制 strict matching（见 `scripts/generate_data.sh` 中 `RECOMMENDED_VLLM_MAJOR_MINOR`）。
> **PyTorch 2.7.0 → 2.10.0**: 匹配最新 CUDA 工具链 (cu128)。
> **flash-attn**: vLLM V1 引擎默认使用 flashinfer attention backend，不依赖 flash_attn。

### 可用模型

| 模型 | 大小 | 用途 |
|------|------|------|
| `models/Google/Gemma-4-31B-it` | 31B | Teacher 生成（本地） |
| `models/Qwen/Qwen3-32B` | 32B | 大容量教师/策略模型 |
| `models/Qwen/Qwen3-8B` | 8B | 策略模型/教师 |
| `models/Qwen/Qwen3-4B` | 4B | 策略模型（默认） |
| `models/Qwen/Qwen3.5-4B` | 4B | 策略模型（多模态） |
| `models/Qwen/Qwen2.5-7B-Instruct` | 7B | 基线对比 |

### Gemma-4-31B-it vLLM 启动（`generate_data.sh` 自动管理）

Gemma-4-31B-it 模型权重 ~15.89 GiB/卡（TP=4 时），A10 22GB 显存紧凑。
**所有参数由 `generate_data.sh` 设置固定默认值**，环境变量可覆盖。

```
# ⚠️ 以下为 4×A10 验证过的最佳配置，请勿修改
VLLM_MAX_MODEL_LEN=8192        # ${VLLM_MAX_MODEL_LEN:-8192}
VLLM_MAX_NUM_SEQS=8            # ${VLLM_MAX_NUM_SEQS:-8}
VLLM_CLIENTS_PER_INSTANCE=8    # ${VLLM_CLIENTS_PER_INSTANCE:-8}
VLLM_GPU_MEMORY_UTILIZATION=0.88
VLLM_MAX_NUM_BATCHED_TOKENS=16384
```

TP 和实例数根据 `GPU_COUNT` 自动计算：
- **4×A10**：TP=4×1实例
- **8×A10**：TP=4×2实例（GPU 0-3/4-7），交错启动避开 torch.compile 峰值内存

### FlashInfer JIT 编译配置（必须）

系统 nvcc 过旧或不兼容，必须使用 conda 安装的 nvcc 12.8。flashinfer 需要 JIT 编译 sampling kernel，必须设置 `CUDA_HOME` 并补齐头文件/库：

```bash
export CUDA_HOME=/mnt/data2/liuzhanyi/envs/arl
export PATH=$CUDA_HOME/bin:$PATH
```

头文件和库符号链接已创建（一次性操作，重启后检查是否存在）：

| 链接 | 源 | 目标 |
|------|-----|------|
| CUB/Thrust/CUDA headers | `$CUDA_HOME/targets/x86_64-linux/include/*` | `$CUDA_HOME/include/` |
| curand headers | `$CUDA_HOME/lib/python3.11/site-packages/nvidia/curand/include/curand*.h` | `$CUDA_HOME/include/` |
| libcuda.so | `/usr/lib64/libcuda.so` | `$CUDA_HOME/lib64/stubs/libcuda.so` |
| libcudart.so | `$CUDA_HOME/targets/x86_64-linux/lib/libcudart.so` | `$CUDA_HOME/lib64/libcudart.so` |

### FlashInfer JIT 降级方案（如果 JIT 编译失败）

```bash
export VLLM_USE_FLASHINFER_SAMPLER=0
```

此方案回退到 PyTorch 原生采样，性能略降但保证可用。

### 环境安装步骤（从头构建）

```bash
# 1. 创建 conda 环境
conda create -n arl python=3.11 -y
conda activate arl

# 2. 安装 PyTorch (cu128)
pip install torch==2.10.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# 3. 安装 vLLM（必须先于 requirements.txt，让 vLLM 自行解析 flashinfer/compressed-tensors/xgrammar 等依赖）
pip install vllm==0.19.1

# 4. 安装项目其他依赖
pip install -r requirements.txt --no-build-isolation

# 5. 安装 verl（源码 editable）
pip install -e ./verl

# 6. 安装 nvcc（flashinfer JIT 编译需要）
conda install -c nvidia/label/cuda-12.8.0 cuda-nvcc -y

# 7. 补齐 CUDA 头文件和库符号链接（见上表）

# 8. 清理 flashinfer JIT 缓存（如果之前编译失败过）
rm -rf ~/.cache/flashinfer

# 9. 验证
export CUDA_HOME=$CONDA_PREFIX
python -c "import torch; print(torch.__version__)"
python -c "import vllm; print(vllm.__version__)"
python -c "import flashinfer; print(flashinfer.__version__)"
```

## Pipeline Status

| Component | Status | Notes |
|-----------|--------|-------|
| Data Generation | ✅ Mechanism smoke | PROVE §3.2 Teacher mechanism verified on normal, missing-function, distractor+enum, and irrelevance paths; bulk yield remains unverified |
| OVAL Agent Loop | ✅ | Single-call protocol + initial-state hash + final-state evidence |
| OVAL Reward | ✅ | Multi-component programmatic: R_val + R_cov + R_eff + R_name + R_arg（论文 §3.3） |
| GRPO Estimator | ✅ | Saturation skip + 2D stratified advantage |
| GPU Auto-Adaptation | ✅ | `scripts/gpu_config.sh` — auto-detect GPU count/memory; TP/instance calculation in `generate_data.sh` |
| Full Training Run | 🔄 | Teacher mechanism smoke 已跑通；正式批量数据与训练尚未启动 |

### Verified Pipeline

```
Step 1 — Auto-Discovered Dependency Graph
  n² pairwise LLM classification → directed graph → chains (len 2–5)
Step 2 — Live-State Sampling (Grounded Query Generation)
  Probe live MCP servers for real entities → inject sampling context into prompt
Step 3 — State-Machine Orchestrator
  Teacher LLM (Gemma-4-31B-it) drives 5-state loop:
  query → LLM processing → tool execution → recovery → continuation
Step 4 — Robustness Knobs
  Distractor injection (40%) + enum stripping (30%) + irrelevance (5%) + missing func (20%)
Step 5 — Replay Validation & Deduplication
  Fresh reset replay (error rate < 30%) → provenance check → Jaccard 0.70 dedup
       ↓
Parquet serialization (success_criteria as JSON string)
       ↓
Multi-Component Programmatic Reward (oval_reward_fn.py)
  R_validity + R_coverage + R_efficiency + R_name + R_arg  (论文 §3.3, 公式 1–5)
       ↓
verl GRPO training (350 steps, single stage)
```

## Training Route

| Route | Agent Loop | Reward Fn | Entry | Status |
|-------|-----------|-----------|-------|--------|
| OVAL GRPO | `livemcp_oval` | `oval_reward_fn.py` | `bash scripts/train_grpo.sh` | ✅ Primary |

`scripts/train_grpo.py` is the Hydra entry point; `src/training/run_grpo.py` is the official training entry.
Config managed by `src/training/trainer_config.py` (PyTorch Lightning style), with GPU tier defaults and `OVAL_*` env var overrides.

### Current Hardware / Defaults

- Teacher model: 本地 Gemma-4-31B-it (`models/Google/Gemma-4-31B-it`)
- Policy model (训练 rollout): Qwen3-4B (`models/Qwen/Qwen3-4B`)
- GPU: 8×A10 22GB

## Constraints

- Training scripts **must not hardcode** GPU count, batch size, micro batch, TP size.
- All project paths use **repo-root-relative** paths — no absolute machine paths.
- Training hyperparams injectable via CLI args, environment variables (`OVAL_*` prefix), or Hydra override.
- `data.max_prompt_length` ≥ 10240.
- Ray temp dir: short path (`/tmp/oval_ray`) to avoid AF_UNIX socket path > 107 bytes.
- SFT cold-start code has been removed. Only GRPO route exists.
- `success_criteria` value field is mixed-type (str/float/int), serialized as JSON string in parquet.
- `OracleCall(action="clarification")` must be preserved in parquet; reward side sets `allowed_terminal=["ask_clarification"]`.

## Known Design Limitations

### PROVE §3.2 对齐说明

本节只判断 Teacher 数据生成机制。只有从当前 checkout 实际生成、Replay、Parquet round-trip 并完成样本语义检查后，才能把该版本标记为可训练。

2026-07-10 当前 checkout 已完成四条 Gemma-4-31B-it 真实 smoke：normal multi-turn、missing-function、distractor+enum stripping、irrelevance。四条均完成 Parquet round-trip，Replay error rate 为 0%；这证明机制路径可运行，不代表批量 yield、分布或所有 seed 的数据质量已经成立。

| 步骤 | 对齐状态 | 说明 |
|------|----------|------|
| Step 1 — 依赖图构建 | 机制对齐 | 全量无序 pair 的 LLM 分类（含方向）→ chains len 2–5；严格 cache 必须记录并校验 pair 分类完整性 |
| Step 2 — Live-State Sampling | 机制对齐 | 通过 readonly discovery tools 探针真实实体；context 绑定 session，chain-aligned context 注入 prompt |
| Step 3 — 状态机 | 机制 smoke 已验证 | query → LLM 决策 → 执行 → recovery → continuation；真实 normal 两轮 chain 已完成 |
| Step 4 — Robustness Knobs | 机制对齐 | distractor / enum stripping / missing function 在 Teacher 处理前固定，之后使用同一配置 Replay |
| Step 5 — Replay + Dedup | 机制对齐 | fresh session、30% error 阈值、provenance check、Jaccard 0.70 |

### Robustness 注入时机

本实现按任务 seed 在 Teacher 处理前采样一次 robustness plan：distractor 加入 candidate set，enum 从 Teacher-visible schema 移除，missing function 在完整 chain/query 生成后从 Teacher schema、dependency hints 和执行层隐藏。同一 plan 贯穿 Teacher、Replay、Parquet 和 rollout，不在中途重新随机化。

### 其他已知限制

| Issue | Explanation |
|-------|-------------|
| Illegal tool JSON → no AuditEvent | Model output format errors only produce error observation, no audit event. Fix requires cross-module type extension. |
| Reward uses last observation as final state | Exact value verification may miss during consecutive tool_calls (low probability, has seen_ids fallback). |
| Teacher validator is stricter than paper text | 除 well-formed JSON 外，首轮 terminal、unknown tool、non-dict arguments、missing-function tool call 也会触发 regeneration；这是本地数据契约，实验记录中必须与论文公开事实分开。 |
| Dependency graph cache provenance | legacy cache 只有 schema hash 和 graph，不能证明全 pair 分类完整；严格 baseline 只接受带完整性元数据的新 cache。 |
| Dependency pair batch size | 当前默认每次分类 12 个 pair；这是根据 Gemma-4-31B 实跑完整率设置的推理参数，不是论文公开超参，不改变全 pair 分类语义。 |
| Oversample + recovery loop | LLM 生成存在 drop rate，当前 50% oversample + 最多 3 轮 recovery（不同 seed 偏移），保证最终产出满足目标数量。 |

## Common Commands

```bash
# ============ 数据生成 ============
# 输出到 data/runs/{MMDD_HHMM}/，自动更新 data/train.parquet 符号链接
# vLLM 由 generate_data.sh 自动启动和管理，无需手动操作
# 默认 Teacher: 本地 Gemma-4-31B-it，自动计算 TP + 动态 vLLM 参数
bash scripts/generate_data.sh --count 500 --val-count 100
# 限制 GPU 数
GPU_COUNT=4 bash scripts/generate_data.sh --count 500 --val-count 100
# 指定 run-id
bash scripts/generate_data.sh --count 500 --run-id 0709_1500
# 单域测试
bash scripts/generate_data.sh --domain calendar --count 200

# ============ GRPO 训练 ============
bash scripts/train_grpo.sh
bash scripts/train_grpo.sh --gpus 0,1,2,3 --total-steps 300
bash scripts/train_grpo.sh --wandb --wandb-project oval-mcp-grpo

# ============ 验证 ============
python scripts/validate_pipeline.py --live        # 端到端管线验证
python -m pytest tests/
python -m compileall src scripts tests
git diff --check

# ============ 维护 ============
python scripts/dependency_graph.py live --model ...       # 预计算/重建工具依赖图（live 模式）
python scripts/dependency_graph.py rebuild --source ...  # 重建依赖图缓存（offline 模式）
python scripts/inspect_prompts.py --train data/train.parquet  # 检查 prompt 内容
python scripts/verify_entities.py                       # 实体验证
python scripts/merge_rollout_shards.py                  # 合并 rollout 分片
python scripts/convert_external_datasets.py             # 转换外部数据集（when2call/xlam）
python scripts/bench_vllm_throughput.py                 # vLLM 吞吐量基准测试
```

## Logging

运行日志统一写入 `logs/`，命名格式：`MMDD_HHMM_{任务描述}.log`

```
logs/0706_1430_gen_500.log          # 7月6日 14:30 生成500条数据（主日志，tee 输出）
logs/0706_1430_vllm_instance0.log   # 同任务 vLLM 实例日志（失败时保留，成功后自动删除）
logs/0706_1500_train_grpo.log       # 7月6日 15:00 GRPO 训练
```

- 时间戳只写月日（`MMDD_HHMM`），不写年
- vLLM 日志生成成功后自动删除；失败时保留用于排查
- 测试运行日志不允许在 `logs/` 中长期堆积，实验结束后及时清理
- `.gitignore` 已忽略 `logs/*`，但保留 `.gitkeep`

## Git

```text
Remote: https://github.com/liuzy1019/LiveMCP-GRPO
Branch: main
Author: liuzy1019 <liuzy1019@buaa.edu.cn>
```

Conventional Commits: `<type>: <subject>`

| Type | Use |
|------|-----|
| feat | New feature / experiment / estimator |
| fix | Bug fix |
| docs | Documentation |
| refactor | Behavior-preserving refactor |
| test | Tests |
| chore | Config / build / deps |
| perf | Performance |

Do not push without verification. Test before commit.
