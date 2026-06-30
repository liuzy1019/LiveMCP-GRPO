# LiveMCP-GRPO

> State-Machine Data Synthesis + Constrained GRPO for Multi-Step MCP Tool Use

PROVE-style state-machine teacher + live MCP execution + event-sourced reward + stratified advantage，训练模型在 10 个 domain / 188 个工具上进行多步工具调用。

**当前方案：**
- **数据生成**：PROVE state-machine (LLM-in-the-loop, Qwen3-32B-Instruct) → `scripts/generate_data.py` → parquet
- **Policy 模型**：Qwen3-4B
- **奖励**：`J = R_task + I_shape·λ_shape·F_gamma + I_process·λ_process·P_process − λ_safe·C_safety`
- **Advantage**：2D 分层 (perturbation_level × scenario_type) + LATA + 饱和跳过
- **硬件**：支持多 tier 自适应 (L20 / A100 / A10 / Hopper / T4)，详见 `scripts/gpu_config.sh`

---

## 项目结构

```
├── scripts/
│   ├── generate_data.py             # 数据生成 CLI (PROVE state-machine)
│   ├── generate_data.sh             # 统一生成脚本（自动检测模型大小+GPU拓扑+并行策略）
│   ├── train_grpo.py                # GRPO 训练 Python 入口 (Hydra)
│   ├── train_grpo.sh                # GRPO 训练 Shell 入口（PyTorch Lightning 风格配置）
│   ├── gpu_config.sh                # GPU 拓扑自动检测（共享库）
│   └── adv_review.py                # 对抗审查脚本
│
├── src/
│   ├── live_mcp/                    # MCP 环境 + 数据生成
│   │   ├── task_planner.py          #   PROVE state-machine (LLM-in-the-loop)
│   │   ├── orchestrator.py          #   任务编排 + 扰动 + 缺量守卫
│   │   ├── state_seeder.py          #   确定性状态播种（10 domains）
│   │   ├── api.py                   #   LiveMCPBranch 统一 API
│   │   ├── llm_client.py            #   LLM 客户端 (vLLM / local)
│   │   ├── dedup.py                 #   Jaccard 去重 (0.70，位置感知)
│   │   ├── config.py                #   配置管理
│   │   ├── errors.py                #   错误类型定义
│   │   ├── executor.py              #   工具执行器
│   │   ├── manager.py               #   MCP 服务器生命周期管理
│   │   ├── oracle.py                #   Oracle 程序构建
│   │   ├── reward.py                #   奖励计算（live_mcp 侧）
│   │   ├── schema_registry.py       #   Schema 注册与查询
│   │   ├── server_base.py           #   服务器基类
│   │   ├── trace.py                 #   追踪记录
│   │   ├── transport.py             #   传输层（subprocess stdio）
│   │   ├── types.py                 #   核心类型定义（LiveTask, OracleCall 等）
│   │   ├── agent_loop.py            #   Agent loop（live_mcp 侧）
│   │   └── servers/ × 10           #   MCP 子进程服务器 (banking/calendar/crm/email/
│   │                                #     filesystem/food_delivery/issue_tracker/
│   │                                #     payments/shopping/team_chat)
│   │
│   ├── agent_loop/                  # verl Agent Loop
│   │   ├── livemcp_oval_loop.py     #   LiveMCPOvalLoop（注册名 "livemcp_oval"）
│   │   └── oval_mcp_worker.py       #   session + audit 包装
│   │
│   ├── oval_mcp/                    # 奖励 + 约束 GRPO 算法（事件验证）
│   │   ├── rewards/
│   │   │   ├── task_reward.py       #   R_task 任务完成奖励
│   │   │   ├── f_gamma.py           #   F_gamma 效率塑形
│   │   │   ├── p_process.py         #   P_process 过程合理度
│   │   │   └── scalar_return.py     #   ScalarReturn 聚合
│   │   ├── verifier/
│   │   │   ├── events.py            #   AuditEvent / EventLog / TrajectoryEventLog
│   │   │   └── safety.py            #   C_safety 安全约束验证
│   │   ├── envs/
│   │   │   ├── domain_adapter.py    #   DomainAdapter
│   │   │   └── audit_wrapper.py     #   AuditWrapper
│   │   └── training/
│   │       ├── lambda_state.py      #   LambdaState λ 自适应更新
│   │       ├── lata.py              #   LATA 分层自适应温度优势
│   │       └── saturation.py        #   饱和组跳过
│   │
│   ├── reward/                      # verl 奖励入口
│   │   ├── action_parser.py         #   Action 解析器
│   │   └── oval_reward_fn.py        #   compute_score() 入口
│   │
│   ├── training/                    # verl 训练组件
│   │   ├── run_grpo.py              #   正式训练入口（OVAL Live MCP rollout）
│   │   ├── trainer_config.py        #   TrainerConfig（PyTorch Lightning 风格配置）
│   │   ├── register_estimator.py    #   estimator 注册 + λ 更新
│   │   ├── livemcp_grpo_estimator.py #  2D StratAdv + LATA estimator
│   │   ├── livemcp_advantage.py     #   Advantage 工厂
│   │   ├── livemcp_hyperparams.py   #   超参数管理
│   │   ├── livemcp_task_runner.py   #   TaskRunner
│   │   ├── hooks.py                 #   训练钩子
│   │   ├── length_check.py          #   Prompt 长度预检
│   │   └── advantage_core.py        #   Advantage 核心计算
│   │
│   └── utils.py                     # 工具函数
│
├── configs/
│   ├── agent_loop.yaml              # Agent loop 注册
│   ├── ds_zero2.json                # DeepSpeed ZeRO-2 配置
│   └── live_mcp/                    # 10 domain + suite 配置
│       ├── suite_mvp.yaml           #   全量套件（10 domain）
│       ├── banking.yaml             #   Finance: banking
│       ├── calendar.yaml            #   Productivity: calendar
│       ├── crm.yaml                 #   CRM: crm
│       ├── email.yaml               #   Productivity: email
│       ├── filesystem.yaml          #   Productivity: filesystem
│       ├── food_delivery.yaml       #   Lifestyle: food_delivery
│       ├── issue_tracker.yaml       #   CRM: issue_tracker
│       ├── payments.yaml            #   Finance: payments
│       ├── shopping.yaml            #   Commerce: shopping
│       └── team_chat.yaml           #   Social: team_chat
│
├── tests/                           # 测试（smoke + 组件 + 对抗审查 + 集成）
│   ├── test_production_smoke.py     #   管线端到端烟雾测试
│   ├── test_adversarial_pipeline.py #   对抗式审查
│   ├── test_live_mcp_10_domains.py  #   10 domain 连通性测试
│   ├── test_stratified_advantage.py #   StratAdv 测试
│   ├── test_grpo_framework_e2e.py   #   GRPO 框架端到端测试
│   ├── live_mcp/                    #   live_mcp 组件测试
│   └── ...                          #   更多组件/集成测试
│
├── data/                            # 训练数据 + 实验记录（parquet gitignored）
│   ├── train.parquet                # GRPO 训练数据
│   ├── val.parquet                  # GRPO 验证数据
│   ├── experiments/                 # 实验记录（配置+结果，跟踪入库）
│   └── README.md                    # 数据目录规范
│
├── docs/
│   └── OVAL-MCP.md                  # 权威方案文档
├── reference/                       # 参考论文 (PROVE 等)
├── verl/                            # verl 框架（vendored，editable install）
├── pyproject.toml                   # 项目元数据与依赖
├── requirements.txt                 # pip 依赖
├── AGENTS.md                        # AI 协作约定
├── CLAUDE.md                        # 工作入口 + 管线状态
└── REVIEW.md                        # 审查报告 + 修复记录
```

---

## 快速开始

### 1. 数据生成

```bash
# 统一生成（自动检测模型大小，选择最优并行策略：local 或 vLLM）
bash scripts/generate_data.sh --model models/Qwen/Qwen3-8B --count 500
bash scripts/generate_data.sh --model models/Qwen/Qwen3-32B --count 500 --val-count 100

# 单 domain 快速测试
bash scripts/generate_data.sh --model models/Qwen/Qwen3-8B --domain calendar --count 200

# 直接调用 Python CLI（高级用户）
python scripts/generate_data.py \
    --model Qwen3-32B-Instruct \
    --api-base http://localhost:8000/v1 \
    --count 500 --val-count 100
```

### 2. GRPO 训练

```bash
# PyTorch Lightning 风格配置（支持多卡、WandB、环境变量覆盖）
bash scripts/train_grpo.sh

# 指定 GPU 和步数
bash scripts/train_grpo.sh --gpus 0,1,2,3 --total-steps 300

# 启用 WandB
bash scripts/train_grpo.sh --wandb --wandb-project oval-mcp-grpo
```

---

## 管线说明

### 数据生成 (PROVE state-machine)

```
Per domain:
  1. 确定性状态播种 (state_seeder.py，10 domains，每种 domain 多个 persona/日期)
  2. LLM-in-the-loop (task_planner.py):
     teacher 看到完整 domain context（tools + live state + execution history），
     每轮决定下一步 action（tool_call / clarification / terminal）
  3. 真实 MCP 执行 (executor.py) → oracle trace
  4. 从 state delta 派生 success_criteria (state_equals / state_exists)
  5. 扰动注入 (orchestrator.py):
     - distractor tools (30%)：注入 3-8 个无关工具
     - missing function (20%)：隐藏一个必需工具
     - irrelevance query (5%)：要求 report_error 的无关查询
  6. 位置感知 Jaccard 去重 (dedup.py, 阈值 0.70)
  7. 转换为 verl 兼容 parquet (generate_data.py:
     prompt JSON string + reward_model + extra_info + scenario_type)
```

默认难度分布：complete 60% / missing 20% / minimal 20%。

### GRPO 训练

```
parquet → vLLM rollout → LiveMCPOvalLoop（真实 MCP 执行 + 审计事件采集）
  → Reward: J = R_task + I_shape·λ_shape·F_gamma + I_process·λ_process·P_process − λ_safe·C_safety
  → 2D StratAdv (perturbation_level × scenario_type) + LATA + 饱和跳过
  → FSDP gradient update
```

训练入口：`scripts/train_grpo.py` (Hydra) → `src/training/run_grpo.py` (正式入口)。
配置通过 `src/training/trainer_config.py` (PyTorch Lightning 风格) 管理，支持 CLI 参数、环境变量 (`OVAL_*` 前缀)、GPU tier 自适应默认值。

---

## 文档

| 文档 | 内容 |
|------|------|
| [CLAUDE.md](CLAUDE.md) | 工作入口 + 管线状态 + 必守约束 |
| [AGENTS.md](AGENTS.md) | AI 协作约定 + 环境事实 |
| [docs/OVAL-MCP.md](docs/OVAL-MCP.md) | 权威方案文档（奖励函数、约束 GRPO 算法、事件系统） |
| [data/README.md](data/README.md) | 数据目录规范 + Parquet Schema + 实验记录规范 |
| [configs/README.md](configs/README.md) | 配置文件说明 + 环境变量覆盖 |
| [REVIEW.md](REVIEW.md) | 审查报告 + 修复记录 |

---

## 许可

MIT License.
