# LiveMCP 数据

> 本文只描述版本化数据合同；算法与 PROVE 边界见 `docs/OVAL-MCP.md`。
> 运行状态必须从当前 manifest、artifact 和进程核实。

## 目录与发布规则

```text
data/
├── dependency_graphs/       # schema/Teacher/classifier-contract namespaced cache
├── runs/
│   └── <run-id>/
│       ├── candidates/             # fixed-attempt 原始 shard/checkpoint
│       ├── failures/*.jsonl        # 每个被拒 candidate 的结构化证据
│       ├── teacher_trace.jsonl     # Query/Action/环境事件原始 trace
│       ├── train.parquet
│       ├── val.parquet
│       └── reports/
├── train.parquet            # 可选：已认证不可变 artifact 的发布副本
└── val.parquet              # 可选：同上
```

只有完成全局过滤、Parquet readback、production parser、fresh replay 和逐行审查的不可变 artifact
才能发布为活动入口。原始 shard、Teacher trace、checkpoint、survivor pool 和进程内 accepted 数量都
不是训练数据。launcher 仅在显式 `--publish` 时将认证后的 train/val 通过临时文件
原子复制到活动入口，不能由一次 gray run 隐式完成。

Teacher trace 与 failure JSONL 默认归档在对应 run 目录，不能依赖可独立清理的全局 `logs/`
重建失败上下文；failure record 至少保留 sampled chain、query contract、首轮实际调用/观察和最终
拒绝原因（尚未进入 Action 阶段的 query failure 除外）。

## 当前数据状态

当前 run、活动发布副本、认证数量和进程状态以 run manifest、artifact、进程和
GPU 现场状态为准，本文不复制动态快照。reward smoke 必须以 `custom`
显式传入现役不可变 train/val；严格 `prove_reproduction_v1` 在 external abstention 与
20-domain 条件缺失时 fail closed。

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

accepted-stratum 口径由 shard、merge 和 top-up 共用：先按本轮显式目标保留 irrelevance，再将
60/20/20 difficulty 配额仅分配给剩余普通候选；`--tool-required-only` 的 irrelevance 目标必须为零。
过滤后的行数或 `minimal` 标签不能反向推断 irrelevance 目标。补量 candidate 数按当前 domain
过滤前候选数与全局 Jaccard 后留存数实测估计，再由同一个 ledger 按精确 stratum deficit 分配；不得
退回固定倍数补量。

initial shard 与 top-up 的 seed namespace 统一绑定
`(base_seed, run_id, artifact_round, domain_scope, stratum, chunk_index)`；同一 run resume 必须稳定，
不同 run、difficulty 或 irrelevance stratum 不得复用候选 seed。shard 内 recovery offset 必须小于
namespace stride。哈希 identity 必须映射到 stride-aligned signed-int64 namespace，使 namespace
基址加任意合法 recovery offset 仍可由 checkpoint、failure JSONL 和 Parquet 以同一整数表示。

## 受控并行生成与验收

一组生成固定使用 4 张 GPU；8 卡机器最多同时启动两个 domain 组。每组必须使用独立、domain-scoped
run-id，不能共享 manifest；同一组 GPU 上不并发启动第二个 vLLM。不存在 `run_10_domains.sh`，不得在
文档或操作中引用它。可计算 N≥16 reward 的最小 fresh canary：

vLLM 与单次生成作业分离：作业停止时默认保留 vLLM，下一域只有在 `/v1/models` 精确匹配同一
served model 时才能复用。需要主动释放本轮新启服务时显式设置 `VLLM_SHUTDOWN_ON_EXIT=1`；该开关
不会关闭 launcher 未启动的外部服务。

launcher 必须先检查目标端口上的 exact-model service，再决定是否需要空闲 GPU；`GPU_FREE_ONLY=1`
只约束新建服务，不能阻止复用已有服务。run 结束状态不得只依赖 CLI 父进程回写，inspect/resume 必须
能够从 checkpoint、accepted artifact、failure evidence 和 production readback 幂等修复 stale
`running` manifest。

fixed-attempt diagnostic 允许所有 candidate 均被 gate 拒绝并产出 0 行 Parquet；readback/audit 必须将其
作为合法的空实验 artifact 处理并正常释放 Arrow 资源。只有该模式下，已写出且通过 readback 的
accepted artifact 才能把 transient-audit 导致的 `failed` manifest 幂等修复为 `completed`。

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

launcher 的 dependency-cache prewarm 必须显式继承本轮 `prompt_profile`；graph cache 本身按
schema/classifier contract 标识，prewarm 输出的 chain 数和随后 candidate runtime 的准入口径不得混用。

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
- chain：`source_chain_seed`、`source_chain_edges`、`chain_seed`、`realized_chain_edges` 和
  `verified_dependency_evidence`。query 是否自然表达该链不由模型自报字段裁决。

heterogeneous nested structures 按 serializer contract 写成 JSON 字符串；消费者必须通过共享
normalizer 解析，不能按某一次 Pandas/Arrow 表现猜测类型。

## 审计命令

```bash
PYTHONNOUSERSITE=1 "$LIVEMCP_ENV/bin/python" \
  scripts/validate_generation_pipeline.py --stages 1,2 --domain all \
  --prompt-profile local_trainable_v1

PYTHONNOUSERSITE=1 "$LIVEMCP_ENV/bin/python" \
  scripts/audit_prove_domains.py --domain all \
  --prompt-profile local_trainable_v1

PYTHONNOUSERSITE=1 "$LIVEMCP_ENV/bin/python" -m src.live_mcp.corpus.audit \
  data/runs/<run-id>/train.parquet data/runs/<run-id>/val.parquet

PYTHONNOUSERSITE=1 "$ARL_ENV/bin/python" -m pytest -q \
  tests/test_transport_contract.py
PYTHONNOUSERSITE=1 "$LIVEMCP_ENV/bin/python" -m pytest -q tests/
```

逐域训练候选的 dependency-chain 数量必须使用 `local_trainable_v1` 审计口径；论文公开机制审计
必须显式传入 `--prompt-profile paper_generation_baseline_v1`。审计 JSON 在每个 domain 的
`dependency.prompt_profile` 记录实际口径，不能脱离该字段比较 chain count。

正式报告至少记录 candidate/accepted/final split、UID/query/Jaccard uniqueness、domain/scenario/
terminal/difficulty/toolchain、replay/provenance/parser、Teacher identity、environment/reward fingerprint，
结构化 generation failure JSONL，以及逐行语义结论。failure JSONL 必须保留候选 seed、恢复轮、
失败阶段/原因和异常 traceback，且在人工或规则审查前保持 `unclassified`。行数和测试数不能替代
这些证据。
