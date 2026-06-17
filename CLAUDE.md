# CLAUDE.md — SchemaShift-GRPO 项目 AI 协作约定

> 给下次会话的自己看。读完这一份，能直接接着干，不要再去摸索环境、跑通逻辑、踩配置坑。
> 最后更新：2026-06-17

---

## 项目本质

面向 MCP Tool Schema 鲁棒性的 GRPO 训练优化。两个组件：
1. **Schema Augmentation**（信号源）：扰动数据增强，3 强度（none/mild/strong），基于 ~100 条同义规则。
2. **Stratified Advantage**（信号通路）：`A = strat_z + beta × global_z`，跨扰动层分层归一化，beta=0.25。

对标 [agentic-grpo-longhorizon](https://github.com/qiqihezh/agentic-grpo-longhorizon) 的 PRM-Lite（信号源）+ LATA（信号通路）双轴结构。

---

## 当前阶段（必读）

**2026-06-16 收尾状态**：E3 / E4 / E5 三个 smoke 已全部跑通到 step 1，loss 有限、ckpt 落盘、grad_norm finite。105/105 tests pass。**正式训练（>1 step）尚未跑**。

step 1 实测 timing（4×A10）：
- E3 vanilla GRPO: 55s/step, max_mem 22.4 GB
- E5 aug only: 145s/step, max_mem 18.0 GB
- E4 SchemaShift-GRPO: ~13min/step（含 final val rollout），max_mem ~21 GB（offload 后）

---

## GPU 资源约定

- 优先 **4 卡** smoke（`CUDA_VISIBLE_DEVICES=0,1,2,3`，A10 23 GB），不够再上 8 卡
- batch_size / micro_batch / TP size **不能写死**，必须根据 N_GPUS 自适应（已通过 `${N_GPUS}` env 参数化）
- 跑前先 `nvidia-smi` 看哪几张卡空，避开有人用的

---

## 环境（arl）

conda 环境路径：`/mnt/data1/zhanyiliu/liuzhanyi/anaconda3/envs/arl`。**所有训练命令优先用这个环境的 Python**，不要用系统 python。

verl 0.6.1 兼容窗口（动了会炸）：
| 包 | 版本 | 备注 |
|---|---|---|
| torch | 2.8.0+cu128 | vllm 0.11 强约束 |
| vllm | 0.11.0 | verl 0.6.1 setup.py 上限 |
| tensordict | 0.10.0 | verl 上限 |
| trl | 0.11.4 | 保留 ValueHead 符号 |
| flash_attn | 2.7.3 | verl 默认 attn_impl，必须装 |
| flashinfer-python | 0.6.4 | 保留但**禁用** JIT，见下 |
| transformers | 4.57.6 | |
| xformers | 0.0.32.post1 | vllm 0.11 强约束 |

环境备份：`docs/archive/arl.requirements.bak.txt`（pip freeze 246 行），如需回滚 `pip install -r`。

### flashinfer JIT 必须绕开

CUDA 11.8 + CCCL 1.x 编译失败（`#include <cuda/functional>` not found）。3 个 GRPO sh 默认带：
```bash
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
```
让 vLLM 走 flash_attn 后端。**永远不要**升级 flashinfer / 降级 flashinfer 到 0.3.x（pypi 没 wheel，源编译同 cccl 问题）。

---

## 数据 / 长度

| 路径 | 内容 | 行数 |
|---|---|---|
| `data/verl/exp3_grpo/` | E3 vanilla（标准 GRPO） | train 900 / val 100 |
| `data/verl/exp4_schemashift/` | E4 SchemaShift（3:3:3 group_id 完整性） | train 8100 / val 900 |
| `data/verl/exp5_aug_only/` | E5 +Aug（每 task 抽 3 条） | train 2700 / val 300 |

**`max_prompt_length` 必须 ≥ 10240**（实测 max=8809, p99=8732，留 ~1.4k buffer）。yaml 默认值已对，不要改回 2048（dataloader 会在第 1 个 batch 直接 raise）。

启动时 `src/training/length_check.py` 会自动校验：
- 实测 max > limit → RuntimeError fail-fast
- p99 > limit × 0.95 → loguru warning
- `SCHEMASHIFT_SKIP_LENGTH_CHECK=1` 跳过（仅 debug 用）

---

## 训练入口对应关系

| 实验 | 配置 | 启动脚本 | Python 入口 | 说明 |
|---|---|---|---|---|
| E3 vanilla | `configs/exp3_vanilla_grpo.yaml` | `scripts/train/grpo/run_vanilla_grpo.sh` | `src/training/run_exp3.py` | 不 patch verl |
| E4 SchemaShift | `configs/exp4_schemashift.yaml` | `scripts/train/grpo/run_schemashift.sh` | `src/training/run_exp4.py` | 注册 estimator + monkey-patch |
| E5 aug only | `configs/exp5_aug_only.yaml` | `scripts/train/grpo/run_aug_only.sh` | `src/training/run_exp3.py` | 复用 E3 入口 |

E4 注册必须在 ray actor 进程内重做（**主进程注册不跨进程传递**），通过 `SchemaShiftTaskRunner(TaskRunner)` 子类的 `run()` 入口完成，已写好不要动。

---

## 关键 BFCL 集成约定

`src/agent_loop/bfcl_agent_loop.py` 调 `bfcl_eval.execute_multi_turn_func_call`，**`test_entry_id` 必填**（BFCL 用作 instance 缓存 key）。生成方式：`f"sshift_{request_id}"`，request_id 是 vLLM 请求的 uuid4。漏传或填空串会导致跨 episode 状态污染。

`_parse_bfcl_native_args` 是 BFCL 原生格式解析器（非 JSON），有硬上限：
- input ≤ 8192 字符
- 段数 ≤ 64
- key ≤ 64 字符
- 字面量 ≤ 4096 字符
单遍 O(N) 扫描。**永远不要**改回包含 `args_part.find("=", i)` 的 O(N²) 写法（会触发 17:10:09 那次 15min cluster-wide GIL 卡死）。

---

## smoke 验收判据

跑 smoke 时不要等"reward > 0"，标准的 smoke pass 只要：
- step 1 完成
- loss 是有限数（非 NaN/inf）
- 没 traceback / OOM / Ray actor died

不符合上面三条之一的，才算 fail。GPU util 短暂为 0、日志短暂不刷新都不算 fail。45 分钟外部 timeout，到点判定。

---

## 跑 smoke 的标准命令

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 N_GPUS=4 MICRO_BATCH_PER_GPU=1 \
  TOTAL_STEPS=1 SAVE_FREQ=10 TEST_FREQ=10 VAL_BEFORE_TRAIN=False \
  WANDB_MODE=disabled \
  bash scripts/train/grpo/run_vanilla_grpo.sh
```

`VAL_BEFORE_TRAIN=False` 关键，否则 smoke 会先跑一遍 val（多 ~10 min 没必要的开销）。

---

## 测试

```bash
# 全量
/mnt/data1/zhanyiliu/liuzhanyi/anaconda3/envs/arl/bin/python -m pytest tests/

# 期望：105 passed
```

测试覆盖：
- `test_advantage` 10 — 公式 + 边界（单样本、缺层、极端 beta）
- `test_api_mapper` 9 — 函数名 + 枚举值映射
- `test_schema_perturber` 13 — 扰动生成 / 确定性 / 安全
- `test_schemashift_grpo_estimator` 11 — estimator 注册 + 分层 advantage
- `test_runtime_regressions` 19 — parquet / BFCL 解析 / reward / eval（含 4 个解析器灾难性输入测试）
- `test_enum_mapping` 14 — enum 映射合法性
- `test_register_estimator_integration` 4 — register_estimator 真实注入 non_tensor_batch
- `test_smoke_grpo_imports` 16 — 依赖契约（vllm/torch/trl/flash_attn 版本兜底）
- `test_length_check` 9 — 数据长度预检（fail-fast / warn / skip env / argv 解析）

每次提交前必须跑全量。**漏跑 → 走弯路**。

---

## 红线

下面这几件事**必须先停下来问用户**：

- 删文件 / 目录 / git 历史
- 改 .env / 密钥 / token / CI
- 数据库 schema 或数据迁移
- `git push --force` / `rebase` / `reset --hard`
- 装新全局依赖、改系统配置
- 公开发布（npm publish / 部署生产 / 发文章）

---

## 行为约定

- **基于事实**：实读文件 / 跑命令 / git 记录。不能"我以为""通常这样"
- **优先并行工具调用**（grep + read + ls 一次发出，别串行）
- **改代码必须跑测试**（最少跑一次 `pytest tests/`）
- **绝不**为让代码跑起来注释掉报错或加绕过标记，找根本原因
- 改完后写一句"改了什么 + 跑了什么验证"，不要默默结束

---

## Git Commit 规范

远端：`https://github.com/liuzy1019/SchemaShift-GRPO`，主分支 `main`，author `liuzy1019 <liuzy1019@buaa.edu.cn>`。

提交信息走 **Conventional Commits**，格式 `<type>: <subject>`，subject 用英文、祈使句、首字母小写、不加句号。可选 body 用空行隔开，写"为什么改"而不是"改了什么"。

| type | 用途 | 例 |
|---|---|---|
| `feat` | 新功能 / 新实验 / 新 estimator | `feat: add E6 regularization baseline` |
| `fix` | bug 修复 / 回归修复 | `fix: bounded parser O(N) on malformed input` |
| `docs` | 只动文档（README / CLAUDE.md / docs/） | `docs: update ablation plan with E5 results` |
| `refactor` | 不改行为的重构 | `refactor: extract bfcl_parser to src/envs/` |
| `test` | 只动 tests/ | `test: add length_check edge cases` |
| `chore` | 配置 / 构建 / 依赖 / .gitignore | `chore: pin flashinfer to 0.6.4` |
| `perf` | 性能优化 | `perf: vectorize stratified z-score` |

红线（命中先问用户，见上方红线条款）：
- `git push --force` / `git push -f`
- `git rebase` / `git reset --hard`
- 删 commit / 改 git 历史
- 推到非 `main` 的 protected 分支

常规推送流程：`pytest tests/` → `git add -A` → `git commit -m "<type>: ..."` → `git push origin main`。**漏跑 pytest 不要 push**。

---

## 当前未完成 / 下一步

| 项 | 状态 | 备注 |
|---|---|---|
| E3 / E4 / E5 step 1 smoke | ✅ | 4×A10 |
| 全量训练（300 step × 6 实验） | ⬜ | 估计 ~166 GPU-hours |
| BFCL 官方 state-based eval 集成 | ⬜ | 当前仅 response-based AST |
| E1（zero-shot）/ E6（正则化 baseline） | ⬜ | 仅文档 |
| `src/agent_loop/bfcl_agent_loop.py` 解耦 | ⬜ | 5 个文件反向 import `_parse_bfcl_native_args`，应抽到 `src/envs/bfcl_parser.py`。**仅在要接第二个环境时再做**，目前 BFCL 单一环境，ROI 低 |

---

## 项目结构速查

```
schemashift-grpo/
├── src/
│   ├── envs/                  # 环境适配（schema_perturber / api_mapper / bfcl_env）
│   ├── agent_loop/            # BFCL 多轮 agent loop（含 _parse_bfcl_native_args，需重构）
│   ├── reward/                # BFCL 正确性 reward
│   ├── eval/                  # 鲁棒性评估 + AST matching
│   └── training/              # estimator / advantage / register / run_exp3 / run_exp4 / length_check
├── configs/                   # 9 个 yaml（exp2/3/4/5 + agent_loop）
├── scripts/                   # build_parquet / generate_perturbations / train shells
├── tests/                     # 105 tests
├── docs/                      # 历史 codex review + technical report + ablation plan
├── data/verl/                 # E2/E3/E4/E5 parquet
└── verl/                      # 内置 verl fork（0.6.1）
```
