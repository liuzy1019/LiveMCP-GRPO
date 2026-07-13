# AGENTS.md — LiveMCP-GRPO AI 协作约定

> 权威方案文档：`docs/OVAL-MCP.md`。架构、数据、reward、评测、阶段详情全在那里。
> 入口文档：`CLAUDE.md`。
> 本文只定义 AI agent 的行为约束和工程纪律。

## 环境

```bash
export ARL_ENV=/mnt/data2/liuzhanyi/envs/arl
conda activate "$ARL_ENV"   # named `arl` 当前错误指向已不存在的 /mnt/data1 环境
export PYTHON_BIN="$ARL_ENV/bin/python"
nvidia-smi                  # 确认 GPU 可用（A10 ×8, 22GB/卡）
```

FlashInfer JIT 编译配置见 `CLAUDE.md`。降级方案：

```bash
export VLLM_USE_FLASHINFER_SAMPLER=0
```

## 依赖约束

verl 0.6.1 从 `./verl` editable 安装。关键版本：

| Package | Version |
|---|---|
| python | 3.11 |
| torch | 2.10.0+cu128 |
| vllm | 0.19.1 |
| transformers | 5.13.0 |
| ray | 2.54.1 |

完整列表见 `requirements.txt` / `pyproject.toml`。

## 工程红线

以下操作必须先停下来确认：

- 删除不明确用途的文件或目录
- 修改 `.env`、token、密钥、CI
- 数据库 schema 或数据变更
- `git push --force`、`git rebase`、`git reset --hard`
- 安装全局依赖或修改系统配置
- 发布、部署或推生产

## 代码约束

- 训练脚本不得写死 GPU 数、batch size、micro batch、TP size
- 项目代码和脚本中的项目文件路径必须以项目根目录为锚点使用相对路径；不要写死机器绝对路径
- 训练超参必须支持通过脚本命令行参数、环境变量或 Hydra override 注入
- `data.max_prompt_length` 不得低于 `10240`
- Ray 临时目录必须使用短路径（默认 `/tmp/oval_ray`），避免 AF_UNIX socket path 超过 107 bytes
- `replace_in_file` 时 `old_string` 必须包含文件中的实际字符（制表符不要额外转义），不要基于记忆中可能被摘要的内容编辑
- 大改动前先更新方案文档，再改实现
- 不确定的事实先核验或停下来对齐，不把假设写进实现

## 数据生成管线约束

> 管线细节见 `data/README.md`，对齐 PROVE 论文 §3.2 的五步流程。核心约束如下：

- **PROVE §3.2 五步机制对齐**：依赖图构建 → Live-State 采样 → 状态机编排 → Robustness Knobs → Replay 验证 + 去重；论文未公开的 self-pair、重复节点和 Jaccard 细节必须标为本地选择，不宣称逐实现一致
- **Robustness 注入时机**：在 Teacher 处理前采样并固定。distractor 加入 Teacher candidate set，enum stripping 作用于 Teacher-visible schema，missing function 在完整 chain/query 生成后从 Teacher schema/hints/执行器中隐藏；同一配置用于 Replay 和 Parquet
- **PROVE corpus hard gates**：只把 fresh replay（schema/execution error rate ≤30%）、sensitive-parameter provenance 和 tool-call sequence Jaccard 0.70 记为论文公开过滤；不得自行增加 exact-chain、词法 capability、query/tool-name 或通用 mutation hard gate
- **Graceful give-up**：PROVE recovery 允许在当前 candidate tools 无法完成目标时直接 `report_error`；不得要求先制造一次 execution failure。无失败历史只可作为 Teacher 质量诊断，不得作为 corpus hard gate
- **多轮预算语义**：论文 `min_turns=2, max_turns=3` 表示 conversation rounds，不是 tool-action 次数。Parquet action budget 必须至少覆盖全部 ground-truth tool calls 和每轮一个 terminal；PROVE adaptive efficiency budget 只用于 reward，不作为 episode 截断上限
- **Irrelevance（5%）**：独立生成无可用工具满足的 query，并显式保存 no-tool terminal oracle
- `success_criteria` 的 `value` 字段类型混合（str/float/int），必须序列化为 JSON 字符串存入 Parquet，不能裸存 list[dict]
- `OracleCall(action="ask_clarification")` 的 `action` 字段必须保留到 parquet；legacy `clarification` 仅在 reward 读回时兼容
- 每轮数据生成改动后必须验证 `generate_data.py` → `to_parquet` → `read_parquet` → `_build_task_dict` 全链路
- 生成失败率需关注日志中的 `generate_many progress` 行和 WARNING 计数
- Replay validation 错误率阈值 30%（仅计 schema + execution 错误，不含空结果）
- Jaccard 去重阈值 0.70，基于 tool-call 序列（位置感知）
- Live-State sampling context 绑定 session；新 session 必须重新探测，同一 session 内允许缓存以减少多轮重复 probe
- Abstain 场景（clarification_required / missing_function / no_tool_or_abstention / irrelevant）的 success_criteria 为空属于预期行为，不视为数据质量问题

## 验证

提交前优先跑全量：

```bash
python -m pytest tests/
```

轻量检查：

```bash
python -m compileall src scripts tests
git diff --check
```

## Git 约定

```text
远端: https://github.com/liuzy1019/LiveMCP-GRPO
主分支: main
author: liuzy1019 <liuzy1019@buaa.edu.cn>
```

Conventional Commits：`<type>: <subject>`

| type | 用途 |
|---|---|
| feat | 新功能 / 新实验 / 新 estimator |
| fix | bug 修复 |
| docs | 文档 |
| refactor | 不改行为的重构 |
| test | 测试 |
| chore | 配置 / 构建 / 依赖 |
| perf | 性能优化 |

没有完成验证时，不要 push。

## 当前环境事实

- 基于 PROVE 框架实现论文 §3.1–§3.3 的 Teacher 生成、Live MCP 和多组件 reward 机制
- Teacher 模型：Gemma-4-31B-it（对齐论文 Teacher），也支持外部 API
- Policy 模型：Qwen3-4B，RL rollout 时通过 vLLM 本地 serving
- Reward：五组件可编程奖励（R_val + R_cov + R_eff + R_name + R_arg），无外部 judge 模型（论文 §3.3）
- OVAL GRPO 是唯一训练路线（`bash scripts/train_grpo.sh`）
- SFT cold-start 相关代码已清除
- 环境：8×A10 22GB，脚本自动检测 GPU 数（`GPU_COUNT=4` 可限制）
- Teacher 生成参数：difficulty_mix = `complete:60%, missing-required:20%, minimal:20%`；missing-function 默认目标比例 `1500/(10895+1500)≈12.1%`，由论文公开 corpus count 推导，非论文直接发布的 knob；irrelevance 5%，enum stripping 30%
- Distractor 40%——在 Teacher 处理前向 candidate set 注入 3–8 个跨域工具；Teacher、Replay metadata 和 RL rollout 使用同一可见 schema，成功调用 distractor 的轨迹不得进入 ground-truth oracle
- Oversample 50% + 最多 3 轮 recovery 保证产出数量
