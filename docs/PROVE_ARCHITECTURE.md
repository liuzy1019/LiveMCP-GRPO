# PROVE 代码架构

> 本文件只规定代码分层、信任边界和唯一生产链路。算法合同见 `OVAL-MCP.md`。

## 分层

```text
domain contracts -> dependency graph -> live-state projection
                                      -> Query Teacher -> Action FSM -> Live MCP
                                      -> local boundary -> replay/provenance
                                      -> canonical artifact -> rollout/reward
```

| 层 | 职责 | 禁止事项 |
|---|---|---|
| `domain_contracts/`, `contracts/` | schema、实体、状态谓词、probe、值流和引用可见性事实 | 启动 server、调用 Teacher、解释自然语言 |
| `dependency_*.py` | pair provenance、eligible graph、chain policy、explicit/implicit evidence | 写逐域工具名补丁 |
| `live_state_*.py` | readonly discovery、public projection、known/unknown/usable | 把 debug state 暴露给 Teacher/Policy |
| `generation/` | Query/Action Teacher、FSM、recovery、robustness、多轮编排 | 定义 domain 状态事实 |
| `replay/` | fresh replay、criteria、sensitive provenance | 修改 canonical oracle |
| `corpus/` | candidate disposition、mix、Jaccard、merge、Parquet | 复制 reward 或另建质量规则 |
| `artifact/` | row contract、fingerprint、readback、reward-task projection | 猜测 Teacher 意图或修补缺失证据 |
| `agent_loop/`, `reward/`, `training/` | rollout、programmatic reward、GRPO | 消费未通过 artifact contract 的数据 |

底层不得 import 顶层 orchestrator。domain facts 不得散落到 generation、validator 或 reward 控制流。

## 信任边界

1. **治理事实面**：SchemaRegistry、domain contracts、dependency graph 和 live-state projection。Teacher 输出
   不能改写这些事实。
2. **内部执行面**：真实 tool name、opaque backend ID、arguments、observation 和 state delta。允许用于执行、
   replay 和 provenance。
3. **用户可见面**：query、continuation 和 terminal。local profile 只允许 public business reference 和 natural
   selector；sampler-private handle 与 hidden tool name 必须 fail closed。

terminal 保留 Teacher 原文。边界检查可以拒绝，不得通过字符串替换伪造正确回答。系统没有通用自然语言
claim prover；只有 schema、state 和 trace 能直接反证的内容才能成为确定性 hard gate。

## 唯一生产链路

```text
pair classification
  -> immutable raw ledger
  -> relation-audited eligible graph
  -> scenario/goal-coherent chain
  -> symbolic + live-state feasibility
  -> grounded Query Teacher
  -> Action FSM + live MCP
  -> dependency/ID/local-quality evidence
  -> owner-preserving fresh replay + provenance
  -> plain tool-sequence Jaccard 0.70
  -> canonical Parquet + production readback
  -> Policy rollout (N >= 16) + programmatic reward
```

CLI、launcher、resume、top-up 和 recertify 只能编排该链路，不能各自实现 mix、ID 边界、Jaccard、artifact
parser 或 semantic gate。

关键不变量：

- `source_chain_seed` 是 query provenance；`chain_seed` 是由真实调用与 dependency evidence 证明的 canonical path。
- 首轮 state-changing capability 必须属于 immutable `source_chain_seed`；readonly discovery 可替换，continuation
  不继承首轮授权。
- repeated capability 必须按真实 occurrence 和连续 evidence path 对齐，不能固定使用第一次调用。
- creator 已满足 target 固定后置条件、立即反转或冗余 create-update 的 chain 在 Query Teacher 前排除。
- natural selector 到 canonical ID 的恢复必须由中间 observation 中同一 record 的 alias 事实证明。
- fresh replay/recertify 必须保留逐调用 owner、`expected_success` 和每个 owner 的 `state_profiles`。
- local deterministic quality 使用一个共享 evaluator；candidate prewrite 首次执行，merge/readback 只检测漂移。
- artifact purpose 由 prompt profile 与 semantic gate 唯一推导，consumer 必须重新验证。
- top-up seed namespace 与 retained Jaccard ledger 必须持久化，不能降低阈值填补容量。

## Orchestrator 边界

`TaskOrchestrator` 只负责请求 chain/context、调用 Teacher/FSM、调用共享 validator，并根据结构化结果
accept/reject。禁止向 orchestrator 增加事实表、handler 状态判断、ID 正则、cache JSON 解析、Parquet 字段解释
或 reward 逻辑。

每个 domain 的差异只能通过以下声明式对象表达：

- `ToolContract`、`StatePredicate`、`StateTransition`；
- `ReferenceVisibility`、`ProbeContract`、`ValueBinding`；
- `QueryMutationAuthorizationContract`、`MissingFunctionContract`、`ScenarioChainContract`。

无法表达的业务规则先标为合同能力缺口并阻断相应 chain，不在 generation 或 merge 中增加一次性补丁。

## 变更门

共享生成逻辑变更必须完成：原失败复现、聚焦测试、相邻反例、旧 artifact 重算或 fresh replay、production
readback。单测通过、某个 domain 成功或过滤率下降都不能独立证明系统修复。
