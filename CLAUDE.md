# CLAUDE.md — LiveMCP-GRPO

Design doc: `docs/OVAL-MCP.md`. Read it before making changes.

## Environment

### 当前可用环境

```bash
conda activate arl          # Python 3.11, PyTorch 2.7.0+cu126
nvidia-smi                  # 确认 GPU 可用（L20 ×8, Driver 570.195.03）
```

关键版本：

| 组件 | 当前版本 | pyproject.toml 目标 |
|------|---------|-------------------|
| PyTorch | 2.7.0+cu126 | 2.8.0 |
| vLLM | 0.9.2 | 0.11.0 |
| flashinfer-python | 0.6.4 | 0.6.4 |
| flashinfer-cubin | 0.6.4 | 0.6.4 |
| flash-attn | 2.7.3 | 2.7.3 |
| nvcc (conda) | 12.9.86 | — |

### FlashInfer JIT 编译配置（必须）

系统 nvcc 11.8 不兼容，已通过 conda 安装 nvcc 12.9。flashinfer 0.6.4 需要 JIT 编译 sampling kernel，必须设置 `CUDA_HOME` 并补齐头文件/库：

```bash
export CUDA_HOME=/mnt/data1/zhanyiliu/liuzhanyi/anaconda3/envs/arl
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
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
```

此方案回退到 PyTorch 原生采样，性能略降但保证可用。flash-attn 2.7.3 在当前 PyTorch 2.7 下有 ABI 兼容问题（`undefined symbol: _ZN3c104cuda9SetDeviceEab`），但 vLLM V1 引擎默认使用 flashinfer attention backend，实际不依赖 flash_attn。

### 环境安装步骤（从头构建）

```bash
# 1. 创建 conda 环境
conda create -n arl python=3.11 -y
conda activate arl

# 2. 安装 PyTorch
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu126

# 3. 安装项目依赖
pip install -r requirements.txt

# 4. 安装 verl（源码 editable）
pip install -e ./verl

# 5. 安装 nvcc 12.9（flashinfer JIT 编译需要）
conda install -c nvidia/label/cuda-12.6.3 cuda-nvcc -y

# 6. 补齐 CUDA 头文件和库符号链接（见上表）

# 7. 清理 flashinfer JIT 缓存（如果之前编译失败过）
rm -rf ~/.cache/flashinfer

# 8. 验证
python -c "import torch; print(torch.__version__)"
python -c "import vllm; print(vllm.__version__)"
CUDA_HOME=$CONDA_PREFIX python -c "import flashinfer; print(flashinfer.__version__)"
```

## Pipeline Status

| Component | Status | Notes |
|-----------|--------|-------|
| Data Generation | ✅ | PROVE state-machine, complete 2–5 step oracle + stratified split |
| OVAL Agent Loop | ✅ | Single-call protocol + initial-state hash + final-state evidence |
| OVAL Reward | ✅ | Ordered coverage + task-aware safety |
| GRPO Estimator | ✅ | Saturation skip + 2D stratified advantage |
| GPU Auto-Adaptation | ✅ | Multi-tier (L20/A100/A10/Hopper/T4) |
| Full Training Run | ⏳ | Pending data generation |

### Verified Pipeline

```
live MCP servers (10 domains)
→ PROVE Teacher (LLM-in-the-loop, Qwen3-32B)
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

### Current Hardware

- Teacher model: Qwen3-32B (vLLM TP=4, GPU 4–7, 4×L20 44GB)
- Policy model: Qwen3-4B (`models/Qwen/Qwen3-4B`)
- Default environment: 8×L20 44GB

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

## Common Commands

```bash
# vLLM inference server（先设置 CUDA_HOME）
export CUDA_HOME=/mnt/data1/zhanyiliu/liuzhanyi/anaconda3/envs/arl
export PATH=$CUDA_HOME/bin:$PATH
python -m vllm.entrypoints.openai.api_server \
    --model models/Qwen/Qwen3-8B --port 8001 \
    --tensor-parallel-size 2 --max-model-len 8192 \
    --gpu-memory-utilization 0.85 --trust-remote-code

# Data generation
bash scripts/generate_data.sh --model models/Qwen/Qwen3-32B --count 500 --val-count 100
bash scripts/generate_data.sh --model models/Qwen/Qwen3-8B --domain calendar --count 200

# GRPO training
bash scripts/train_grpo.sh
bash scripts/train_grpo.sh --gpus 0,1,2,3 --total-steps 300
bash scripts/train_grpo.sh --wandb --wandb-project oval-mcp-grpo

# Validation
python -m pytest tests/
python -m compileall src scripts tests
git diff --check
```

## Logging

运行日志统一写入 `logs/`，命名格式：`MMDD_HHMM_{任务描述}.log`

```
logs/0706_1430_gen_500.log          # 7月6日 14:30 生成500条数据
logs/0706_1430_gen_500_console.log  # 同任务的 console 输出
logs/0706_1500_train_grpo.log       # 7月6日 15:00 GRPO 训练
```

- 时间戳只写月日（`MMDD_HHMM`），不写年
- 测试运行日志不允许在 `logs/` 中长期堆积，实验结束后及时清理
- `.gitignore` 已忽略 `logs/*`，但保留 `.gitkeep````

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
