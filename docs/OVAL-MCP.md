# OVAL-MCP：算法与工程合同

> 本文件是算法机制和发布边界的权威说明。代码分层见 `PROVE_ARCHITECTURE.md`，数据格式见
> `../data/README.md`，动态进度见 `PROJECT_STATUS.md`。历史过程只从 Git 追溯。

## 目标

系统在有状态 MCP 环境中生成自然用户任务，由 Teacher 真实执行工具链，经 replay、provenance 和去重形成
canonical artifact，再使用 programmatic reward 训练 Policy。结构测试、生成 yield、语义质量和训练资格必须
分别验证，不能相互替代。

## PROVE 边界

依据 Abdelaziz et al., *Synthesize and Reward -- Reinforcement Learning for Multi-Step Tool Use in Live
Environments*（arXiv:2606.03892）实现以下公开机制：

- 每个 environment 的全部无序 tool pair 由 LLM 分类为 `explicit / implicit / none`，抽取 2--5 步 chain；
- Query 前执行 readonly Live-State sampling；
- Query Teacher 与 Action/state-machine Teacher 分离，执行结果驱动 recovery 和 continuation；
- difficulty 为 complete 60%、missing-required 20%、minimal 20%；
- distractor 40%、enum stripping 30%、irrelevance 5%，并支持 missing-function；
- corpus hard gates 为 fresh replay error rate `<=30%`、sensitive provenance 和 tool-sequence Jaccard 0.70；
- reward 包含 validity、coverage、adaptive efficiency、tool name 和 argument value；
- single-stage GRPO，论文配置 group size 16、350 steps。

以下属于本地扩展，不得称为 PROVE 公开机制：

- 当前只有 10 个 domain/191 tools，不是论文的 20/343；
- strict external abstention 数据尚未导入，当前第三桶是 internal proxy；
- typed state contract、reference visibility、semantic quarantine、artifact fingerprint、round/terminal 诊断；
- local chain allocation、seed namespace、position-aware scheduling 和 OVAL reward/estimator 扩展；
- missing-function 默认比例由论文公开数据量推导，不是论文发布的 knob。

`prove_reproduction_v1` 必须在完整 domain、外部 abstention、数据规模和训练条件满足前 fail closed。当前项目
只能报告工程实现、局部认证或自定义实验，不能报告论文结果复现。

## Profile 与发布用途

| Prompt profile + semantic gate | Artifact purpose | 允许用途 |
|---|---|---|
| `paper_generation_baseline_v1 + diagnostic_only` | `paper_audit` | 论文机制审计；训练/rollout/reward 拒绝 |
| `local_trainable_v1 + deterministic_v1` | `training_candidate` | 通过正式认证后可发布 |
| 其他组合或 fixed-attempt | `experiment` | 诊断，不可发布 |

consumer 必须从字段重新推导 purpose，并验证 profile、runtime、reward 和 environment fingerprint；不能依赖目录名
或操作者记忆。paper profile 只由 PROVE hard gates 决定保留，本地 label/outcome/semantic finding 作为诊断；local
profile 额外执行确定性可消费性和安全合同。

## 五步生成

### 1. Dependency graph

每个 domain 对全部 `C(n,2)` pair 保留 immutable classifier ledger。`explicit` 表示 source observation 的具体
typed value 可以填入 target required argument；`implicit` 表示 source mutation 建立 target 必需状态；其他为
`none`。

cache 同时保存 raw graph 和 relation-audited eligible graph。raw ledger 只记录 Teacher provenance，不因本地
relation audit 被静默改写；runtime 只消费 schema、classifier contract、pair completeness 和 relation evidence
均当前的 eligible graph。explicit edge 需要 output-to-argument binding；implicit edge 在执行后通过同 seed
counterfactual replay 证明。信息相关或常见工作流顺序本身不构成 dependency。

### 2. Live-State sampling

生成前在 isolated session 中执行 readonly discovery/detail probe，形成 compact chain-aligned context。feasibility
必须按 source 本次 observation 与 target required entity/value 做 identity-level join，并区分
`discovered/state_known/state_unknown/usable`。字段缺失按 unknown fail closed。

同一 sampling epoch 可让若干 candidate 共享 deterministic baseline seed，但每条 candidate 使用独立 session；
conversation mutation 不回写 baseline。Query Teacher 可见 chain-aligned context；Action Teacher 和 Policy 只从
公开 query 或真实 tool observation 获取参数。

### 3. Query Teacher 与 Action FSM

Query Teacher 只接收 sampled chain、public/chain-aligned context、persona、reference date 和 difficulty，并只
生成自然 user message。reference date 必须由 `sampling_state_seed` 唯一派生。Action Teacher 接收 query、可见
schema、公开 conversation 和真实 execution feedback，在 FSM 中选择 tool call、recovery 或 terminal。

`source_chain_seed` 是不可改写的 query provenance，不是要求 Teacher 逐节点照抄的模板。成功场景的
`chain_seed` 必须来自首轮真实调用中可由 graph 与运行证据共同证明的至少两步路径；辅助调用单独记录。每条
realized edge 必须具有 explicit value binding 或 implicit counterfactual evidence。

首轮 state-changing capability 必须属于 `source_chain_seed` 的授权集合，并在 MCP dispatch 前检查。readonly
discovery 可以替换；continuation 使用自己的 round contract，不继承首轮授权。越权 proposal 不执行，但必须写入
结构化 failure evidence。

difficulty 只控制信息完整度，不扩张写授权：missing-required 恰好遗漏关键字段，minimal 只给目标；缺失值应通过
clarification 解决，不能执行未授权 mutation。

### 4. Robustness

`RobustnessPlan` 在 Query Teacher 前按 seed 固定，并参与 chain eligibility：

- distractor 注入 3--8 个跨域工具，Teacher/Replay/Policy 使用同一 visible schema；成功 distractor 不进入 oracle；
- enum stripping 只修改模型可见 schema，handler 仍执行原约束；
- missing-function 必须绑定 handler-audited、其余可见工具无法产生的 state effect，只允许 readonly prefix，terminal
  为 clarification/report_error；证据不足时在 Teacher 前拒绝；
- irrelevance 必须由 capability inventory 证明没有可用工具满足，不能由同一个 LLM 自报不可完成。

### 5. Replay、provenance、去重与 artifact

每条 candidate 在 freshly reset session replay。PROVE hard gates只有：

1. schema/execution error rate `<=30%`；
2. sensitive value 可追溯到 user turn 或 prior observation；
3. canonical executed plain tool sequence 的 Jaccard 小于 0.70。

replay/recertify 必须恢复每个 call 的 `server_name`、`expected_success` 和各 owner `state_profiles`。provenance
按值判断；数值可做完整数值等价归一化，不能接受部分数字串或不同数值。

local deterministic quality 在 candidate 进入 eligible pool/checkpoint/Parquet 前执行；merge 和 readback 调用同一
evaluator 只用于检测漂移。单条候选违反合同应记录 disposition 并继续同批处理；infrastructure/schema/runtime
错误必须终止 shard。

Parquet 写出后必须完成 `to_parquet -> read_parquet -> artifact validation -> build_reward_task`。ground truth、
round contract、oracle、dependency edge 和 top-level fields 必须一致；consumer 不得从另一字段重建或修补真值。

## 本地确定性合同

### Reference visibility

匹配 deterministic sampler handle 的 backend ID 是内部执行引用，不得出现在 initial query、continuation、
clarification 或 terminal。public business reference 只能由 typed binding field 显式声明；自然 selector 必须与
backend handle 分离。Action Teacher 可以从真实 observation 获取 canonical ID，但用户可见面只使用 public
projection。

private/public 集合在 terminal decision、completed trace 和 artifact consumer 三处用同一 typed contract 重算。
系统只能拒绝泄露，不能重写 Teacher 文本。ID 隔离不推断参数来源、任务 outcome 或任意自然语言含义。

### Dependency 与状态逻辑

- repeated capability 在所有有序 occurrence 中选择真正建立 target 前提的调用，再形成唯一连续 evidence path；
- creator 已建立 target 固定 postcondition、立即反转或只重复 create 时已有字段的 chain 在 Query Teacher 前排除；
- state-transition mutation 只有 `success=True && state_changed=True` 才能进入 canonical required work；
- natural selector 到 canonical ID 的 failure recovery 必须由中间 observation 中同一 record 的 alias 关系证明；
- success criteria 只从真实 state delta/observation 导出，并由 local profile 的 fresh replay 重现；
- terminal 外部副作用声明只有在 observation 明确返回该结果时才能成为确定性事实。

无法由 schema、state、query 和 trace 直接证明的自然度、偏好或开放式文本正确性只做 diagnostic/人工审核，
不得靠新增逐域正则补丁升级为通用 hard gate。

## Policy rollout 与 reward

Policy rollout 为每条轨迹建立 isolated MCP session，真实执行 schema/handler，并只从 rollout audit events 与
canonical ground truth 重算 reward。terminal contract violation 在 PROVE baseline 下是 diagnostic，不改写
工具任务得分；plain terminal compatibility 是显式本地开关，正式 baseline 默认关闭。

`prove_baseline`：

```text
R = 0.5 R_validity + 0.5 R_coverage + 0.15 R_efficiency
    + 0.2 R_name + 0.1 R_arg
```

adaptive efficiency 使用 `B = n_gt + ceil(0.5 * n_gt)`、`alpha=0.5`。no-tool task 为二值：零工具调用得 1，
否则 0。`G` 只取 canonical oracle calls，`E` 只取有 dependency evidence 的 `chain_seed`。

`oval_full` 可以加入 terminal/round/process/safety shaping，但必须使用独立 reward profile 和 estimator，不能静默
混入 PROVE baseline。

组内全同 reward 会产生零 advantage。必须按 source `group_id`、run 和 training step 统计，并结合逐轨迹证据
区分任务过易、稳定失败、reward aliasing、数据难度不足和 runtime error；不能只看平均 reward。rollout dump
必须保存 row metadata、audit events、final state 和实际 reward replay info，才能做反事实重算。

## 数据与训练门

训练候选必须完成：

1. 每域 schema/cache/live-chain/Teacher/artifact 逐层认证；
2. immutable artifact、SHA256、Teacher identity 和 environment/reward fingerprint；
3. 每个 source row 至少 16 条 current-fingerprint Policy rollout；
4. reward 分布、饱和、runtime failure 和 aliasing 审计；
5. 明确发布为 active corpus。

未完成以上门时只能运行诊断 smoke，不得启动正式 350-step GRPO。入口和命令分别见 `../scripts/README.md`、
`../data/README.md` 和 `../configs/README.md`。
