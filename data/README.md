# LiveMCP 数据

> 最后核实：2026-08-10。整体状态见 `docs/PROJECT_STATUS.md`，算法与 PROVE 边界见
> `docs/OVAL-MCP.md`，逐域准入见 `docs/DOMAIN_SEMANTIC_AUDIT.md`。

## 目录与发布规则

```text
data/
├── dependency_graphs/       # schema/Teacher/classifier-contract namespaced cache
├── runs/
│   └── <run-id>/
│       ├── train.parquet
│       ├── val.parquet
│       └── reports/
├── train.parquet            # 可选：指向已认证不可变 artifact 的活动符号链接
└── val.parquet              # 可选：同上
```

只有完成全局过滤、Parquet readback、production parser、fresh replay 和逐行审查的不可变 artifact
才能发布为活动入口。临时 shard、Teacher trace、checkpoint、survivor pool 和进程内 accepted 数量都
不是训练数据。发布活动符号链接必须显式执行，不能由一次 gray run 隐式完成。

## 当前磁盘事实

- `data/train.parquet`、`data/val.parquet` 均不存在。
- `data/runs/` 当前为空。
- 当前没有通过全局过滤、production parser、fresh replay 和逐行语义审核的正式 artifact。
- `prove_local_v1`、`oval_local_v1` 与两个 reward-gray profile 所绑定的不可变 train/val 文件缺失，
  当前会 fail closed；正式训练前必须重新建立并校验 artifact，而不是修改 hash 绕过。

从本轮代码开始，新生成行必须在 `extra_info.teacher_model_id` 保存稳定模型身份；serving URL 或 alias
不能替代该 provenance。旧行缺失该字段时审计失败，不能自动补写猜测值。

## 生成参数与论文边界

| 参数 | 当前默认/目标 | 边界 |
|---|---:|---|
| difficulty | complete 60%、missing-required 20%、minimal 20% | PROVE 公开分布 |
| distractor | 40%，3--8 个跨域工具 | PROVE 公开机制 |
| enum stripping | 30% | PROVE 公开机制 |
| irrelevance | 5% | PROVE 公开机制；当前来源是 internal proxy |
| missing-function | `1500/(10895+1500)≈12.1%` | 从公开 corpus 数量推导，不是论文 knob |
| replay error threshold | ≤30% | PROVE hard gate |
| Jaccard threshold | 0.70 | PROVE plain tool-call sequence hard gate；位置感知/增强签名仅作本地诊断 |
| checkpoint interval | 25 accepted tasks | 本地工程默认 |

PROVE 公开 hard gates 是 fresh replay、sensitive provenance 和 tool-call-sequence Jaccard。terminal、
round、hidden-tool、environment fingerprint、canonical replay、semantic quarantine 和 Parquet schema
是本地可消费性合同，必须分别报告。第三桶尚未导入 When2Call 806 条与 xLAM-Irrelevance 316 条，
不能称 strict external-source reproduction。

## 受控并行生成与验收

一组生成固定使用 4 张 GPU；8 卡机器最多同时启动两个 domain 组。每组必须使用独立、domain-scoped
run-id，不能共享 manifest；同一组 GPU 上不并发启动第二个 vLLM。不存在 `run_10_domains.sh`，不得在
文档或操作中引用它。可计算 N≥16 reward 的最小 fresh canary：

```bash
export ARL_ENV=/mnt/data2/liuzhanyi/envs/arl
PYTHONNOUSERSITE=1 "$ARL_ENV/bin/python" -m src.live_mcp.corpus.cli run \
  --mode full --domain DOMAIN --count 16 --val-count 4 \
  --prompt-profile paper_generation_baseline_v1 \
  --semantic-gate-profile diagnostic_only
```

该组合生成的行会标记为 `artifact_purpose=paper_audit`，只用于论文机制和过滤诊断，训练、
rollout、reward 入口会 fail closed。训练候选必须显式使用：

```bash
PYTHONNOUSERSITE=1 "$ARL_ENV/bin/python" -m src.live_mcp.corpus.cli run \
  --mode full --domain DOMAIN --count 16 --val-count 4 \
  --prompt-profile local_trainable_v1 \
  --semantic-gate-profile deterministic_v1
```

该组合写入 `artifact_purpose=training_candidate`；其他 profile 组合统一标记为 `experiment`。

公开 Python CLI 必须把自身的 `sys.executable` 作为 `PYTHON_BIN` 传给内部 launcher；调用者显式设置的
`PYTHON_BIN` 优先。不能依赖未激活 shell 中的 `CONDA_PREFIX`，否则绝对路径启动 ARL CLI 仍会降级到
系统 `/usr/bin/python3`。

每域按以下顺序验收：

1. schema/handler 与 dependency cache provenance；
2. raw `C(n,2)` ledger、relation warning、live-feasible chain；
3. Gemma Teacher query/action/observation/state mutation/terminal/continuation 逐行审查；
4. fresh replay、provenance、Jaccard、Parquet round-trip；
5. `validate_prove_corpus_evidence`、`validate_teacher_generation_evidence`、环境 metadata 与
   `build_reward_task` production loader；
6. Qwen3-4B-Instruct-2507 多 seed rollout/reward；
7. 记录证据并在获得明确删除授权后清理临时产物，再进入下一域。

`--domain all` 仅用于只读全局结构验证，不用于正式补产或并行启动十域 vLLM。

## Parquet 合同

顶层字段：`prompt`、`data_source`、`reward_model`、`extra_info`、`uid`、`group_id`、
`perturbation_level`、`scenario_type`。

关键 `extra_info` 至少包括 `prompt_profile`、`semantic_gate_profile`、`artifact_purpose`；consumer
会重新推导三者关系，缺失或不匹配时拒绝。其余字段包括：

- provenance：`teacher_model_id`、`prompt_profile`、`generation_method`、
  `dependency_classifier_contract_hash`；
- oracle：`oracle_calls`、`required_tools`、`dependency_edges`、`required_call_rounds`；
- 多轮：`conversation_queries`、`round_contracts`、`teacher_round_trace`、
  `teacher_attempt_trace`、terminal contract；
- robustness：`hidden_tools`、`candidate_tools`、`tool_owner_domains`；
- replay/provenance：`canonical_replay_*`、`success_criteria`、sensitive provenance evidence；
- identity/runtime：`server_schema_hashes`、`transition_fingerprints`、`initial_state_hashes`、
  `state_profiles`、reward/runtime fingerprint；
- chain：`source_chain_seed`、`chain_seed`、`query_chain_supported` 三态和对应 status。

heterogeneous nested structures 按 serializer contract 写成 JSON 字符串；消费者必须通过共享
normalizer 解析，不能按某一次 Pandas/Arrow 表现猜测类型。

## 审计命令

```bash
PYTHONNOUSERSITE=1 "$LIVEMCP_ENV/bin/python" \
  scripts/validate_generation_pipeline.py --stages 1,2 --domain all

PYTHONNOUSERSITE=1 "$LIVEMCP_ENV/bin/python" -m src.live_mcp.corpus.audit \
  data/runs/<run-id>/train.parquet data/runs/<run-id>/val.parquet

PYTHONNOUSERSITE=1 "$ARL_ENV/bin/python" -m pytest -q \
  tests/test_transport_contract.py
PYTHONNOUSERSITE=1 "$LIVEMCP_ENV/bin/python" -m pytest -q tests/
```

正式报告至少记录 candidate/accepted/final split、UID/query/Jaccard uniqueness、domain/scenario/
terminal/difficulty/toolchain、replay/provenance/parser、Teacher identity、environment/reward fingerprint，
以及逐行语义结论。行数和测试数不能替代这些证据。
