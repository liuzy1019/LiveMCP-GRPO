# LiveMCP-GRPO

> State-Machine Data Synthesis + Constrained GRPO for Multi-Step MCP Tool Use

PROVE-style state-machine teacher + live MCP execution + event-sourced reward + stratified advantage, training models on 10 domains / 188 tools.

**当前方案：**
- **数据生成**：PROVE state-machine (LLM-in-the-loop, Qwen3-32B-Instruct) → `scripts/generate_data.py` → parquet
- **Policy 模型**：Qwen3-4B
- **奖励**：`J = R_task + F_gamma + P_process − λ_safe·C_safety`
- **Advantage**：2D 分层 + LATA + 饱和跳过
- **硬件**：8×A10 23GB / L20 44GB

---

## 项目结构

```
├── scripts/
│   ├── generate_data.py             # 数据生成 CLI (PROVE state-machine)
│   ├── generate_data_nightly.sh     # 夜间全量生成脚本
│   ├── train_grpo.sh                # GRPO 训练入口 (GPU 自适应)
│   └── train_runner.py              # verl 训练 runner
│
├── src/
│   ├── live_mcp/                    # MCP 环境 + 数据生成
│   │   ├── task_planner.py          #   PROVE state-machine (LLM-in-the-loop)
│   │   ├── orchestrator.py          #   任务编排 + 鲁棒性扰动
│   │   ├── api.py                   #   统一 API
│   │   ├── llm_client.py            #   LLM 客户端 (vLLM / local)
│   │   ├── dedup.py                 #   Jaccard 去重 (0.70)
│   │   ├── state_seeder.py          #   确定性状态播种 (10 domains)
│   │   └── servers/ × 10           #   MCP 子进程服务器
│   │
│   ├── agent_loop/                  # verl Agent Loop
│   │   ├── schemashift_oval_loop.py #   注册名 "schemashift_oval"
│   │   └── oval_mcp_worker.py       #   session + audit wrapper
│   │
│   ├── oval_mcp/                    # 奖励 + 约束 GRPO 算法
│   │   ├── rewards/                 #   R_task + F_gamma + P_process
│   │   ├── verifier/                #   C_safety + AuditEvent
│   │   ├── envs/                    #   DomainAdapter + AuditWrapper
│   │   └── training/                #   LambdaState + LATA + Saturation
│   │
│   ├── reward/                      # verl 奖励入口
│   │   ├── oval_reward_fn.py        #   compute_score()
│   │   └── action_parser.py         #   tool_call 解析
│   │
│   └── training/                    # verl 训练组件
│       ├── run_grpo.py              #   正式训练入口
│       ├── schemashift_grpo_estimator.py  # 2D StratAdv + LATA
│       ├── register_estimator.py    #   estimator 注册 + λ 更新
│       └── length_check.py          #   数据长度预检
│
├── configs/
│   ├── agent_loop.yaml              # Agent loop 注册
│   └── live_mcp/ × 10              # 10 domain 子进程配置
│
├── tests/
├── data/                            # 训练数据 (gitignored)
└── outputs/                         # 输出 (gitignored)
```

---

## 快速开始

### 1. 启动 vLLM Teacher

```bash
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export NVCC_APPEND_FLAGS=-allow-unsupported-compiler

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m vllm.entrypoints.openai.api_server \
  --model models/Qwen/Qwen3-32B-Instruct \
  --served-model-name Qwen3-32B-Instruct \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.85 \
  --port 8000
```

### 2. 生成训练数据

```bash
python scripts/generate_data.py \
  --model Qwen3-32B-Instruct \
  --api-base http://localhost:8000/v1
```

### 3. 启动 GRPO 训练

```bash
bash scripts/train_grpo.sh
```

---

## 管线说明

### 数据生成 (PROVE state-machine)

```
Per domain:
  1. Auto-discover tool dependency graph (live MCP probing)
  2. State machine: query generation → tool execution → continuation decisions
     (LLM-in-the-loop at every turn, against live MCP server)
  3. Replay-validate each conversation
  4. Dedup (Jaccard 0.70)
  5. Convert to parquet

Robustness knobs:
  - Distractor tools (40%)
  - Enum stripping (30%)
  - Irrelevance queries (5%)
  - Missing function (20%)
  - Execution perturbations
```

### GRPO 训练

```
parquet → vLLM rollout → Live MCP Agent Loop (real execution + audit)
  → Reward: R_task + F_gamma + P_process − λ_safe·C_safety
  → StratAdv (2D stratified advantage) + LATA + Saturation skip
  → FSDP gradient update
```

---

## 文档

| 文档 | 内容 |
|------|------|
| [REVIEW.md](REVIEW.md) | 审查报告 + 架构分析 |
| [CLAUDE.md](CLAUDE.md) | 工作入口 |
| [AGENTS.md](AGENTS.md) | AI 协作约定 |

---

## 许可

MIT License.
