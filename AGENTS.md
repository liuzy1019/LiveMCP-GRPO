# AGENTS.md — LiveMCP-GRPO AI 协作约定

> 权威方案文档：`docs/project_plan.md`。架构、数据、reward、评测、阶段详情全在那里。
> 入口文档：`CLAUDE.md`。
> 本文只定义 AI agent 的行为约束和工程纪律。

## 环境

```bash
# Python
python

# GPU 确认
nvidia-smi

# 优先 4 卡 smoke
CUDA_VISIBLE_DEVICES=0,1,2,3
```

flashinfer JIT 必须禁用：

```bash
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
```

## 依赖约束

verl 0.6.1 从 `./verl` editable 安装。关键版本：

| Package | Version |
|---|---|
| python | 3.11.15 |
| torch | 2.8.0+cu128 |
| vllm | 0.11.0 |
| deepspeed | 0.19.2 |
| flash_attn | 2.7.3 |
| transformers | 4.57.6 |
| ray | 2.54.1 |
| numpy | ≥1.26.4 (arl: 2.2.6) |
| pandas | ≥2.2.3 (arl: 3.0.2) |
| protobuf | ≥5.29.6 (arl: 7.34.1) |

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
- 项目代码和脚本中的项目文件路径必须以项目根目录为锚点使用相对路径；不要写死 `/data/...`、`/mnt/...` 等机器绝对路径
- 训练超参必须支持通过脚本命令行参数、环境变量或 Hydra override 注入
- `data.max_prompt_length` 不得低于 `10240`
- Ray 临时目录必须使用短路径（默认 `/tmp/ssgrpo_ray`），避免 AF_UNIX socket path 超过 107 bytes
- `_parse_bfcl_native_args`（如后续创建）必须保持 bounded linear parser
- 大改动前先更新 `docs/project_plan.md`，再改实现
- 不确定的事实先核验或停下来对齐，不把假设写进实现

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
远端: https://github.com/liuzy1019/SchemaShift-GRPO
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

## 已核验注意事项

- 当前 policy 模型为 Qwen3-4B（`models/Qwen3-4B`），LLM Teacher 使用 Qwen3-8B（`models/Qwen3-8B`），计划升级到 Qwen3-32B-Instruct
- OVAL GRPO 是主训练路线（`src/training/run_grpo.py`），使用 `SchemaShiftOvalLoop` + `oval_reward_fn.py`
- SFT cold-start 相关代码已清除（`scripts/sft_cold_start.py`、`configs/sft_cold_start_4b.yaml`）
- GRPO 训练默认环境为 8×L20 44GB
