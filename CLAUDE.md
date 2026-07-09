# CLAUDE.md — LiveMCP-GRPO

Design doc: `docs/OVAL-MCP.md`. Read it before making changes.

## Environment

### 当前可用环境

```bash
conda activate arl          # Python 3.11.15, PyTorch 2.10.0+cu128
nvidia-smi                  # 确认 GPU 可用（L20 ×8, Driver 570.195.03）
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
| Data Generation | ✅ | PROVE state-machine, complete 2–5 step oracle + stratified split |
| OVAL Agent Loop | ✅ | Single-call protocol + initial-state hash + final-state evidence |
| OVAL Reward | ✅ | Ordered coverage + task-aware safety |
| GRPO Estimator | ✅ | Saturation skip + 2D stratified advantage |
| GPU Auto-Adaptation | ✅ | `scripts/gpu_config.sh` — auto-detect GPU count/memory; TP/instance calculation in `generate_data.sh` |
| Full Training Run | 🔄 | 数据生成管线已跑通，训练待启动 |

### Verified Pipeline

```
live MCP servers (10 domains: banking, calendar, crm, email, filesystem,
  food_delivery, issue_tracker, payments, shopping, team_chat)
→ PROVE Teacher (LLM-in-the-loop; preferred: Gemini via OpenAI-compatible API)
→ Real MCP execution → oracle trace
→ Jaccard dedup (0.70, position-aware)
→ Parquet serialization (success_criteria as JSON string)
→ oval_reward_fn.py (R_task + C_safety, optional F_gamma/P_process)
→ verl GRPO training
```

## Training Route

| Route | Agent Loop | Reward Fn | Entry | Status |
|-------|-----------|-----------|-------|--------|
| OVAL GRPO | `livemcp_oval` | `oval_reward_fn.py` | `bash scripts/train_grpo.sh` | ✅ Primary |

`scripts/train_grpo.py` is the Hydra entry point; `src/training/run_grpo.py` is the official training entry.
Config managed by `src/training/trainer_config.py` (PyTorch Lightning style), with GPU tier defaults and `OVAL_*` env var overrides.

### Current Hardware / Defaults

- Teacher model: 优先 Gemini（通过 `--api-base` 指定代理），本地备选 Gemma-4-31B-it
- Policy model (训练 rollout): Qwen3-4B (`models/Qwen/Qwen3-4B`)
- GPU: 8×L20 44GB

## Constraints

- Training scripts **must not hardcode** GPU count, batch size, micro batch, TP size.
- All project paths use **repo-root-relative** paths — no absolute machine paths.
- Training hyperparams injectable via CLI args, environment variables (`OVAL_*` prefix), or Hydra override.
- `data.max_prompt_length` ≥ 10240.
- Ray temp dir: short path (`/tmp/ssgrpo_ray`) to avoid AF_UNIX socket path > 107 bytes.
- SFT cold-start code has been removed. Only GRPO route exists.
- `success_criteria` value field is mixed-type (str/float/int), serialized as JSON string in parquet.
- `OracleCall(action="clarification")` must be preserved in parquet; reward side sets `allowed_terminal=["ask_clarification"]`.

## Known Design Limitations

| Issue | Explanation |
|-------|-------------|
| Illegal tool JSON → no AuditEvent | Model output format errors only produce error observation, no audit event. Fix requires cross-module type extension. |
| Reward uses last observation as final state | Exact value verification may miss during consecutive tool_calls (low probability, has seen_ids fallback). |
| Perturbation only in teacher phase | PROVE design: perturbation for teacher robustness testing, clean training environment. |
| Teacher chain-progress blocking removed | PROVE §3.2: only format validation (well-formed JSON). No post-hoc action-type blocking. Replaced silent retry with stronger chain_guidance in prompt. See `task_planner.py:decide_action`. |

## Common Commands

```bash
# ============ vLLM 推理服务 ============
export CUDA_HOME=/mnt/data2/liuzhanyi/envs/arl
export PATH=$CUDA_HOME/bin:$PATH

# vLLM 0.19.1 使用 vllm serve 命令
vllm serve models/Qwen/Qwen3-8B \
    --tensor-parallel-size 2 --max-model-len 8192 \
    --gpu-memory-utilization 0.85 --port 8001 --trust-remote-code

# 或使用封装脚本
bash scripts/start_vllm.sh models/Qwen/Qwen3-8B 8001 2 "0,1"

# ============ 数据生成 ============
# 输出到 data/runs/{MMDD_HHMM}/，自动更新 data/train.parquet 符号链接
# Teacher: Gemini via OpenAI-compatible API（推荐）
bash scripts/generate_data.sh --model gemini-2.5-flash --api-base https://your-proxy/v1 --count 500 --val-count 100
# Teacher: 本地模型
bash scripts/generate_data.sh --model models/Google/Gemma-4-31B-it --count 500 --val-count 100
# 单域测试
bash scripts/generate_data.sh --model gemini-2.5-flash --api-base https://your-proxy/v1 --domain calendar --count 200

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
python scripts/precompute_dependency_graphs.py     # 预计算工具依赖图
python scripts/rebuild_dependency_graph_cache.py   # 重建依赖图缓存
python scripts/check_data.py -f data/train.parquet # 检查 parquet 数据
python scripts/inspect_prompts.py -f data/train.parquet  # 检查 prompt 内容
python scripts/verify_entities.py                  # 实体验证
python scripts/merge_rollout_shards.py             # 合并 rollout 分片
python scripts/convert_external_datasets.py        # 转换外部数据集（when2call/xlam）
python scripts/bench_vllm_throughput.py            # vLLM 吞吐量基准测试
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
