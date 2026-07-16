# data/

本目录存放依赖图缓存和 Teacher 生成的 Parquet。生成产物不入库（见 `.gitignore`）。

---

## 目录结构

```
data/
├── dependency_graphs/              # 各 domain 工具依赖图（precompute 产出）
│   └── {domain}_{hash}.json        # 单个 domain 的拓扑依赖图
├── runs/                           # 每次生成的带版本子目录（gitignored）
│   └── {MMDD_HHMM}/                # 单次生成目录，命名与日志一致
│       ├── train.parquet
│       └── val.parquet
├── train.parquet -> runs/{MMDD_HHMM}/train.parquet   # 完整 run 验收后建立
├── val.parquet   -> runs/{MMDD_HHMM}/val.parquet     # 完整 run 验收后建立
└── README.md
```

### 命名约定

- `runs/{MMDD_HHMM}/` — 与 `logs/{MMDD_HHMM}_gen_{N}.log` 时间戳对应，方便追溯
- `data/train.parquet` 和 `data/val.parquet` 只在 train/val 均完整产出并通过生成门禁后更新；失败或零验证集 smoke 不得覆盖入口
- 每次正式生成使用独立 run 目录；只保留需要训练或复核的当前 run
- vLLM 运行日志写到 `logs/`，生成成功后自动删除；失败时保留用于排查

---

## 数据生成管线

> 重新生成后建议运行 `python scripts/validate_generation_pipeline.py --stages 1,2`，并用 Stage 3 或正式生成 artifact 核对运行链路。
> 完整设计参考 PROVE 论文 §3.2。

### 当前验收状态

当前轨迹合同为 `live-mcp-canonical-replay-trajectory-v1`。十域 strict dependency cache
均匹配当前 schema、Teacher 与 classifier contract。旧 Parquet 已清理，当前没有可训练数据。
重新生成后的每行必须同时通过生产 parser、Teacher/attempt replay、canonical replay、
环境 metadata 与公开 corpus hard gates；结构门禁仍不替代自然语言语义审查。生成后使用
`python scripts/audit_generated_data.py <train.parquet> <val.parquet>` 逐行复核。

### Step 1 — 自动发现依赖图（Auto-Discovered Dependency Graph）

```
对每个 domain，对全部无序工具对 `C(n,2)=n(n-1)/2` 各调用一次 LLM 分类器，判断关系：
  explicit（工具 A 的输出是工具 B 的必需输入）
  implicit（A 必须先执行以建立状态）
  none
→ 缓存为有向依赖图（按 tool schema、dependency semantics 和 classifier contract 索引）
→ 提取 length-2 到 length-5 的工具链，作为多步 query 的种子
```

每个 domain 的依赖图缓存在项目根目录锚定的 `data/dependency_graphs/{domain}_{schema_hash}.json`，不得随进程 cwd 改变位置。cache 同时绑定 Teacher model ID 与 classifier prompt 的 `classifier_contract_hash`；更换模型或分类合同必须 cache miss，不能复用其他 Teacher 的图。

这是运行时依赖图的唯一权威来源。十域 server YAML 只保存 subprocess/session 与工具分类
审计信息，不维护另一套手写 dependency edges 或 query templates。
严格 cache 必须保存全部 `C(n,2)` 的 `pair_classifications` ledger（包括 `none`），并由 ledger 推导 `classified_pair_count` 和 graph；loader 必须核对 pair key 集合与预期全集完全相等，并验证 graph 与 ledger 重建结果一致。旧的有序 `n(n-1)` cache、缺少 ledger 的 artifact 或仅声称 `classification_complete=true` 的计数均不能作为论文全部 pair 已分类的证据。
pair ledger 必须满足关系定义的确定性合同：`explicit` target 至少包含一个 required input；明确 readonly/non-mutating 的 source 不能作为 `implicit` source。违反合同的 Teacher 响应不允许本地改写为其他关系，只重试对应 pair；同一 schema、Teacher 与 prompt 合同下的其余有效 pair 可以从现有 cache 复用。
每个无序 pair 只分类一次，LLM 输出 source/target 决定有向关系；`none` 表示该 pair 不建立边。论文没有公开 chain traversal 是否允许重复节点，当前提取不重复节点的 simple paths，这是可审计的本地实现选择。pairwise classifier 默认每批 8 个 pair、最多 512 输出 tokens；若返回缺项，只重问未分类 pair。任一 batch 重试耗尽仍不完整时，整张 graph fail-closed，不缓存部分结果。
并行生成时，同一 `{domain}_{schema_hash}` 的冷缓存只允许一个进程执行 LLM 分类；等待进程获得文件锁后必须再次读取缓存。持久 `flock` 文件集中放在 `data/dependency_graphs/.locks/`，释放锁后不自动 unlink，避免等待进程与新进程锁住不同 inode。缓存通过同目录临时文件和 `os.replace` 原子发布，生成 shard 启动前默认由单进程完成所选 domain 的预热。进程内 graph/chains 也必须以 `(domain, schema_hash, classifier_contract_hash)` 为 key，不能只按 domain 复用。

缓存保留 LLM 对全部无序 pair 的分类结果（关系为有向 source/target 或 `none`），不再经过本地 handler denylist、domain 黑名单、entity heuristic 或强制破环改写。候选 chain 的可行性由后续 live-state、execution 与 replay 验证；handler 事实只能用于 feasibility 或运行验证，不能回写 pairwise classifier 的 cache。

### Step 2 — 实时状态采样（Live-State Sampling / Grounded Query Generation）

```
在 query 生成前, 对每个 domain 调用只读探测工具（search / list / get_unique_values）
→ 枚举真实存在的实体（账户、联系人、事件、商品等）
→ 按已选 chain 的真实 handler 前置条件筛选可用实体；当前不维护论文示例中的全局 supporting-data filter 表
→ 每个 seed/session 首次使用时重建 compact sampling context；同一未变化 session 内最多复用 k 次，写入后的 continuation 强制刷新
→ 注入 query 生成 prompt，约束 LLM 只能使用 context 中存在的实体
```

这一步约束 query 使用真实实体，并在生成前排除已知不可行 chain；最终是否可执行仍以 live execution 和 fresh replay 为准。

### Step 3 — 状态机编排器（State-Machine Orchestrator）

```
每个对话由 5 组状态驱动（论文 §3.2 Step 3）：
  1. QUERY GENERATION — 从依赖图链中采样，用实时状态 grounding，
     按信息完整度分层（complete 60% / missing 20% / minimal 20%），
     叠加 persona + reference date 条件
  2. LLM PROCESSING — Teacher LLM 被 prompt 提供 query + tool schemas，
     输出 tool call / 澄清 / 终止；validator 检查 well-formed JSON
  3. TOOL EXECUTION — tool call 分派到 live MCP server，
     结果分类为 SUCCESS / PARTIAL_SUCCESS / FAILURE
  4. RECOVERY — 失败时状态机在 retry-with-corrected-params /
     retry-with-alternative-tool / give-up 中选择
  5. CONTINUATION — 采样是否结束对话/追问/澄清，turn-decay 调度，
     min_turns=2, max_turns=3
```

首轮 user query 由一条 live-state feasible dependency chain 生成。该 chain 及其 graph hints 不传给 Teacher action prompt；Teacher 只根据 query、candidate tool schemas 和真实 execution history 决策。前置节点不拆成后续 user turn；continuation 只根据刷新后的 live state、此前 query 与真实 execution history继续同一 conversation，不为后续 round 重新构造 dependency chain。Replay 可执行不等于语义正确，正式 smoke 还必须核对 query、oracle 和 terminal。

Query Teacher 返回 `user_query`、内部 `target_capability` 和 `chain_supported`；目标必须等于 sampled chain 尾节点，且 query 不得直接给出本应由前置节点产生或发现的信息。无法自然构造该多步目标时返回 `UNSAT`。不满足时在 Query Generation 内最多重试三次，耗尽后拒绝 candidate。以上声明不写入用户 query、不传给 action planner，也不是 oracle exact-chain gate。`complete` 只要求用户已知且完成目标所需的信息充分；若某个 ID 只能由 chain 前置 creator 产生，不能强制 query 引用执行前 Current State 中的其他 ID。

首轮 Query Teacher 的 `Current State` 与 `Chain-Aligned Entities` 使用同一份 chain-specific view；不得额外暴露未通过当前 chain handler 前置条件过滤的全量实体。summary 保留实体类型、status、amount、关联 ID 及可计算余额，避免 invoice/payment ID 混用以及对不可 refund/dispute 状态生成请求。Action Teacher 不接收 chain，只按 query、schema、同轮 grounded entities 和真实 execution history 核对资源类型与状态/数值约束。

tool description 必须与真实 handler 前置条件一致。Continuation 不构造新 dependency chain，但 query generator 同时看到同轮 schemas 与刷新后的 state，不能仅凭 domain 示例猜测合法状态或参数类型。

对无参数 discovery 前置节点将发现的实体，query view 隐藏 opaque ID，仅暴露真实 name/customer/category/status/value 等 selector；validation view 继续保存完整 ID。Action Teacher 不直接读取 sampler context，ID 来源只允许 user query 或 prior tool observation。该机制不改变 Replay/Jaccard 门禁，也不强制最终 oracle exact-match sampled chain。

可行链在通过 schema/handler/live-state feasibility 后均匀采样；不再按 mutation 数量或单一工具名设置论文外优先池。失败由真实 execution、state-machine recovery 与 fresh replay 处理。

运行链路不执行状态反转、capability 同义词或实体字符串匹配过滤；PROVE 未发布这些 gate。是否保留样本由用户目标的实际执行、fresh replay、provenance 和去重决定。`chain_seed`/`dependency_edges` 的完整性仅是本项目 OVAL coverage reward 的本地结构合约。

`source_chain_seed` 表示生成 query 时使用的工作流种子；只有 oracle 实际完整覆盖该序列时才写入 `chain_seed` 并构造 OVAL dependency edges。PROVE 没有发布“必须逐节点覆盖 seed chain”的 corpus hard gate。每个 round 的用户目标仍须由成功执行完成，或以 clarification / abstention 合法终止。Missing-function 在 chain 选择后隐藏必要工具。

Recovery 的 alternative tool 不要求与 chain 节点名称相同；成功轨迹由 live execution 和 fresh replay 判定。Missing-function 可以先调用仍可见的前置工具，再因隐藏能力缺失而 clarification / abstention；不得以 `missing_function=True` 拒绝所有 tool call。若最终 abstain 且留下 chain visible prefix 之外的成功 mutation，生成阶段重试该候选，避免把未完成 workaround 的部分副作用当作 RL ground truth；这是本地质量合同，不是论文 hard gate。硬约束仍只有 hidden tool 不可见、不可执行、不可进入 oracle，以及最终 clarification / abstention。

Query Teacher 只强制最终 mutating target 的 query 原文授权片段；内部 dependency mutation 不要求逐节点证据，也不因依赖边自动获得副作用授权。无法表达为一个自然用户目标的 chain/query 返回 `UNSAT`。生成器不执行 tool-name、同义词或 exact-chain 文本 gate。完整执行 trace 不裁剪；只有被真实 execution history 标记为 exact no-progress 的同轮重复调用，才从 Parquet required oracle 和 round contract 中排除。

`live-mcp-canonical-replay-trajectory-v1` 状态机 fail-closed：未知/已关闭 session 不得静默重建；每次 handler 调用以 session state 快照为事务边界，异常或 `state_changed` 声明与真实 delta 不一致时原地回滚；成功 mutation 输出实际 `state_delta_paths`，每条 success criterion 保存对应成功调用 provenance，并持久化逐轮 query/oracle/history 与真实 execution attempt。Required-workflow 投影后还必须 fresh replay 精确导出的 tool calls、hidden tools 与 criteria，结果随行保存为 `canonical_replay_*`。这些 provenance、Teacher 和 canonical 证据是审计/可消费性元数据，不是 PROVE corpus hard gate。Global merge 会核对当前 `trajectory_schema_version`、executable schema、transition、reward 和 initial-state hashes；任何非当前合同数据必须重新生成，不能修改旧 metadata 冒充新标签。

过滤边界：fresh replay、sensitive-parameter provenance 和 Jaccard 0.70 是论文公开 corpus gates。Parquet schema、显式 terminal、round contracts、hidden-tool 不泄漏与 dependency-edge 索引是本项目 rollout/reward 的结构合同。位置感知 Jaccard 和按公开 corpus 数量反推的 missing-function 采样率属于本地工程选择，必须在实验 metadata 中记录，不能写成论文公布的实现细节。

实体初始状态中的可变字段必须逐实体复制，禁止多个实体共享同一个 `list`/`dict` 对象；否则一次合法 mutation 会产生无关资源的 state delta，并污染 success criteria。环境合同测试必须覆盖 email labels、team-chat reactions 等集合字段的对象隔离。

Reversible 最终目标优先绑定可发现的既有实体：banking 通过 `list_scheduled_transfers` 暴露可取消 transfer，payments seeder/list_webhooks 暴露可删除 webhook，shopping wishlist 暴露可移除 product；冻结账户和 settled payment 同样由 live state 提供。`create -> delete` 等 source chain 仍只是 query seed，不要求 Action Teacher执行内部 setup mutation。

Recovery 边界：论文明确包含 graceful give-up with fallback explanation。Teacher 判断当前 candidate tools 无法完成用户结果时可以直接 `report_error`，不要求先执行一个必然失败的工具调用；“无 execution failure 的 report_error”只记录诊断，不作为导出拒绝条件。

预算边界：论文 continuation 的 `min_turns=2, max_turns=3` 指 conversation rounds；一次 round 内可以包含多个顺序 tool calls。Parquet `budget` 是本项目 rollout 的 action-turn 工程合同，必须满足 `budget >= ground-truth tool-call 数 + conversation round 数`，从而至少容纳每轮一个 terminal。论文 §3.3 的 adaptive efficiency budget 按 tool-call 数计算奖励惩罚，不负责截断 episode。

真实十域 smoke 的补充约束：成功 normal conversation 必须有 2--3 个真实 user turns；follow-up 必须依赖当前 live state 和全部已完成 user/assistant rounds 并继续此前 conversation，不得由内部 chain 节点机械改写。供 continuation 使用的 compact entity summary 必须保留判断目标是否已经满足的业务字段，避免把已有 reminder、地点等结果再次请求。filesystem archive 操作只使用 live-state 中的 file 实体。

并行 shard 的初始 quota 很小时，单个候选被质量校验拒绝不等于 Teacher/MCP 全链路不可用。`shard_mode` 的显式 zero-yield 必须进入已有 recovery seed 轮次；非 shard 的首轮 zero-yield 仍保持 fail-fast，以免掩盖模型服务或 MCP 整体故障。

每一步 LLM 决策都看到完整 domain context（tools, live state, execution history），形成 teacher 轨迹。每条轨迹记录为 oracle trace，包含全部 tool call 参数和 server 响应。

训练行的初始 Policy prompt 必须包含首轮 Action Teacher 实际看到的 compact observable entity summaries（最多 15 条）。这是可见信息合同的对齐，不包含 true state snapshot、依赖 chain 或 sampler-private ID；后续状态仍通过真实 MCP observation 和多轮 user query 进入 rollout。

设置 `LIVEMCP_TEACHER_TRACE_PATH` 后，审计日志除 exact LLM messages/raw response 外，还记录 generation setup、chain/live-state selection、Query/Action candidate contract、fresh replay 的逐调用 observation、provenance 与 acceptance。设置 `LIVEMCP_TEACHER_TRACE_INCLUDE_STATE=1` 时才附加 true state；该状态只用于核查，永不进入模型输入或数据过滤。

Teacher 工具文本以 MCP schema annotations 为准：`readonly=true, mutating=false` 会被显式渲染为“不修改状态”，并覆盖工具名启发式。新建 team-chat thread 的初始 `messages=[]`；包含 `create_thread` 后再 reaction 的 query 只能要求对已存在的 root message 操作，不能假设链内自动产生 reply。

实现显式记录并转换 `query / turn / tool_execution / response / continuation` 五组状态。MCP outcome 分为 `SUCCESS / PARTIAL_SUCCESS / FAILURE`：三者都写入 response evidence，`FAILURE` 进入 recovery，`PARTIAL_SUCCESS` 作为独立 outcome 返回 Teacher 继续决策，不能折叠为普通 success。

### Step 4 — 鲁棒性注入（Robustness Knobs）

```
生成过程中以概率注入扰动，扩展训练分布（论文 §3.2 Step 4）：
  - Distractor injection（40%）：混入 3-8 个来自其他 domain 的无关工具
  - Enum stripping（30%）：从参数 schema 中移除枚举值列表，迫使模型从描述推理约束
  - Irrelevance queries（5%）：生成没有可用工具能处理的 query，训练拒绝能力
  - Missing function（目标约 12.1%）：在链选择后隐藏一个必需工具，产生 abstention/clarification 样本；比例由论文公开的 1,500 / (10,895 + 1,500) corpus count 推导
```

强制执行契约：

```text
missing-function:
  完整 query_chain 先生成确实需要 hidden tool 的 query
  Teacher 只接收移除 hidden tool 后的 candidate schemas，不接收 graph hints
  可先执行仍可见的前置工具，最终以 clarification/abstention 终止
  hidden_tools 在生成、Parquet、rollout executor 和 reward 中保持一致

Replay / provenance:
  验证完整 teacher_attempt_trace（失败、blocked、retry、alternative 均保留）
  ground-truth oracle 仅保留最终有效步骤

distractor:
  在 Teacher 生成前把 3-8 个跨 domain 无关工具加入 candidate set，并对同一候选集 replay
  schema 携带 owner domain 并路由到真实 MCP server
  ground-truth oracle 不应调用无关 distractor

irrelevance:
  impossible query 与 assistant terminal 都经过 Teacher FSM
  完成 oracle 不含成功 tool-call，并以 report_error / ask_clarification 合法终止
  失败尝试保留并接受统一 replay 30% gate，不额外要求零尝试
  不得由生成器直接硬编码 terminal oracle
```

### Step 5 — 重放验证与去重（Replay Validation and Deduplication）

```
每条完成的对话在 freshly reset 的 MCP 环境中重放：
  - 只计入 schema 级和执行级错误（不含空结果响应）
  - 错误率 > 30% 的对话丢弃
  - Provenance check：敏感参数（passwords, tokens）只能出现在可追溯到前序 user
    turn 或 tool output 的位置
  - Jaccard 去重：全部 replay/provenance 存活样本进入同一 tool-call sequence 候选池，阈值 0.70；不按 domain 豁免
  - 零工具 clarification/abstention 不回退到 query-text Jaccard，避免加入论文未发布的文本过滤
```

### 训练数据构成

```
训练集只接收当前 Live MCP 生成链路产出的两类轨迹：
  1. Multi-turn MCP conversations — 状态机生成轨迹
  2. Clarification / abstention trajectories — missing-function 与 irrelevance 变体
```

**生成策略**：全局只计算一次候选预算；shard 子进程不重复增加固定 oversample floor。`generate_many` 按 domain 增量提交任务，达到 quota 后停止为该 domain 创建新 future。不足时 recovery 只补充缺口，经 replay、provenance、Jaccard 去重和训练合约过滤后取精确 N 条。recovery 默认最多 3 轮；可通过 `--max-recovery-rounds` 显式调整，但该值属于本地工程产出参数，必须记录到实验配置，不能表述为 PROVE 公开算法参数。不同 domain 的富余样本不能代替缺口。长任务应指定 `--checkpoint-path`：每轮完成后原子保存候选 `LiveTask`、下一轮缺口请求和配置指纹；用相同配置重启时只补剩余 domain，不重复首轮。

Teacher JSON stage 使用独立输出预算，避免短 JSON 响应按通用 1024-token 上限占用 vLLM KV cache。action loop 对完全重复且无状态变化的调用只注入 no-progress 反馈，后续替代动作或 terminal 仍必须由 Teacher 产生。multi-shard merge 会输出逐域 deficit、Jaccard 前候选数、实际保留率和建议补样数；launcher 保留已生成 shard，只为缺口 domain 追加 top-up shard，再重新执行全局 Jaccard 与 train/val split。建议补样数按该域累计保留率反推并增加 20% 有限裕量，再在现有 generation client 槽位内均匀切片，避免单个 deficit domain 串行占满整个补量阶段；这些设置只影响生成预算和调度，不放宽 replay、provenance 或 Jaccard 门槛。

多 shard 边界：`--shard-mode` 子进程只保证输出通过 replay 与训练结构合同的 eligible 候选，不执行 shard-local Jaccard，也不要求每个 shard 独立满足十域比例。子进程按 eligible 行数恢复局部生成短缺；所有 shard 汇总后，global merge 才执行跨 shard Jaccard 0.70、逐域 train/val quota 与 split 隔离。最终 split 将全域规范化后的相同首轮 user query 绑定到同一 split，同时保持逐域 train/val quota，防止跨 domain 的 irrelevant/query 模板泄漏；该分组不删除候选、不改变 PROVE corpus gate。Top-up 的部分有效结果必须保留并交给下一次 global merge，只有全局 top-up 轮次耗尽后仍存在 domain 缺口才 fail-closed。

### 难度分布

| 类型 | 比例 | 说明 |
|------|------|------|
| **complete** | 60% | user query 包含全部所需信息 |
| **missing** | 20% | user query 省略一个关键参数 |
| **minimal** | 20% | user query 极其简略，需模型自行推断 |

---

## Parquet Schema

| 字段 | 类型 | 说明 |
|------|------|------|
| `prompt` | str | 仅包含初始 `system + user`；不得包含 teacher tool history |
| `data_source` | str | `"live_mcp_state_machine"` |
| `reward_model` | dict | `{"style":"rule", "ground_truth": {"task_id","oracle_calls","success_criteria","required_tools"}}` |
| `extra_info` | dict | domain, target_servers, required_tools, scenario_type, oracle_calls, hidden_tools 等 |
| `uid` | str | 等于 task_id |
| `group_id` | str | 等于 task_id（每个 task 独立一组） |
| `perturbation_level` | str | `complete` / `missing` / `minimal` |
| `scenario_type` | str | 轨迹事实标签，如 `normal_safe_success` / `tool_error_recovery` / `clarification_required` / `no_tool_or_abstention` / `partial_completion_or_abstention` / `missing_function` / `irrelevant` |
| `has_state_outcome_oracle` | bool | 是否存在非空 state/outcome success criteria；只用于标签强度分层与诊断，不是 PROVE corpus hard gate |
| `server_schema_hashes` | JSON str | 主域及所有可见 distractor owner domain 的完整 executable schema hash；rollout 逐域 fail-fast 校验 |

### P0 Baseline Integrity Fields（extra_info）

以下字段在 `extra_info` 中以 JSON string 存储（Parquet round-trip 保障）：

| 字段 | 类型 | 说明 |
|------|------|------|
| `round_contracts` | JSON `[{"round_idx":int, "required_tools":[str], "allowed_terminal_actions":[str]}]` | 每轮 reference 工具和合法终止动作。多轮数据必须存在且数量与 `conversation_queries` 一致；terminal 强制校验；reference call 只与同轮 audit event 对齐，但不作为 rollout exact-name 截断条件 |
| `dependency_edges` | JSON `[[src_idx, dst_idx], ...]` | 索引边列表，通过全链对齐（full-chain alignment）生成。src_idx/dst_idx 是 `oracle_calls` 中 tool_call 的位置（0-based）。所有边必须满足 `src_idx < dst_idx`。中间节点可同时作为前边 dst 和后边 src。chain_seeded 任务必须恰好 `len(chain_seed)-1` 条边，不完整者在 split 前被拒绝 |
| `dependency_graph_complete` | bool | **诊断字段**，不替代导出前门禁。chain_seeded 任务由 `_validate_task_training_contract` 强制校验边完整性
| `conversation_queries` | JSON `[str, ...]` | 多轮对话的每轮 user query。长度即轮数 |
| `paper_replay_valid` | bool | schema/execution error rate ≤ 30%（PROVE §3.2 Step 5） |
| `project_outcome_valid` | bool | 所有 success_criteria 在 fresh session 上满足 |
| `criteria_failed_count` | int | 实际失败 success_criteria 数量（来自 replay 统计） |
| `replay_error_rate` | float | replay 错误率 |
| `teacher_attempt_trace` | JSON | 全部 Teacher tool attempt 的 round、参数、expected/actual outcome 与原始 MCP observation；模型实际看到的投影文本由相同 projector/budget 产生，exact prompt/raw response 仍以当前 run JSONL 为准；该字段不进入 Policy prompt/reward |
| `teacher_round_trace` | JSON | 每轮 user query、完整 oracle（含中间 terminal 文本）与真实 execution history；用于输入充分性和 continuation 审计 |
| `chain_seed` | JSON `[str]` | 任务使用的工具链种子 |
| `generation_mode` | str | 固定为 `chain_seeded`；缺少 dependency-chain seed 直接拒绝 |

### Terminal Progression 规则（P0-2）

多轮 rollout 遵循以下 fail-closed 规则：

| Terminal | 行为 |
|----------|------|
| `final_answer` | `allowed_terminal_actions` 包含时推进下一轮；本轮 reference `required_tools` 缺失只记录 `round_tool_diagnostic`，不截断等价路径 |
| `report_error` | **始终终止** episode，不推进 |
| `ask_clarification` | 仅在有配对 user reply 时推进，否则终止 |
| 非法 terminal | 记录 `contract_violation` 审计事件并停止 |
| 多余 terminal | reward 拒绝（`n_actual != n_expected`） |

Rollout 缺 contract 或 contract 数量与对话轮数不一致 → `RuntimeError`（fail-closed）。
每个 tool/terminal `AuditEvent` 必须记录真实 `round_idx`；缺少 terminal、未知 terminal、少于或多于
`round_contracts` 的 terminal 都属于不完整 trajectory，不进入 PROVE 五组件计算。后续 user query
尚未注入时调用 future-round reference tool 不得获得该 GT step 的 coverage/argument match；这不把
每轮 reference tool name 升级为 rollout exact-chain hard gate。

### Entity Quality Filtering（P0-1）

数据生成时先构造 probe record，再按候选 chain 做实体可行性判断：

1. **首轮 probe**：枚举所有实体（list_/search_ 只读工具）
2. **阶段 enrichment**（仅 food_delivery）：对 restaurant 实体逐次调用 `get_menu`，合并菜单数据到 record
3. **状态过滤**：全局 `DOMAIN_ENTITY_QUALITY_FILTERS` 当前为空；按已选 chain 的 handler 前置条件判定，只有字段明确冲突才排除，字段缺失交给 live execution/replay

chain-specific 条件包括账户基数/状态、商品库存、订单 lifecycle、filesystem 类型等；它们只在对应 chain 需要时生效。`_extract_chain_context` 按字段**存在性**判断是否使用 qualified list，空 qualified 不会回退到原始实体。

### 关键约束

- `reward_model.ground_truth.success_criteria` 是 **JSON 字符串**（非 list[dict]），避免 pyarrow 混合类型崩溃
- `reward_model.ground_truth.oracle_calls` 保存 Teacher 实际成功 oracle calls 和一个显式终止动作；seed chain 仅在完整实现时写入 `chain_seed/dependency_edges`，不能假定每条 oracle 都严格等于 2–5 步 seed chain
- `prompt` 是 JSON 字符串，OVAL loop 端自动 `json.loads` 恢复
- `oracle_calls` 在 `extra_info` 中也是 JSON 字符串序列化，避免 pyarrow struct 统一化导致的字段丢失

---

## 数据生成命令

### vLLM 默认参数

`generate_data.sh` 默认使用已验证的 4×A10 正式生成档位；环境变量可显式覆盖：

当目标样本数少于配置的 GPU/client 数时，launcher 只启动 `train+val>0` 的 active shard；零配额 shard 不创建进程，也不计入失败分母。这样小规模 smoke test 的耗时和成败只反映真实生成请求。

| GPU | TP / 实例 | max_model_len | max_num_seqs | clients | enforce_eager | gpu_memory_utilization |
|-----|-----------|---------------|--------------|---------|---------------|------------------------|
| 4×A10（默认） | TP=4 × 1 | 8192 | 8 | 8 | 0 | 0.88 |

`max_num_batched_tokens=16384`，CUDA Graph 默认开启。`GPU_COUNT=8` 可显式扩展为
两个 TP=4 实例，但不再是默认行为。

所有参数均可通过环境变量覆盖：
```bash
GPU_COUNT=8 VLLM_MAX_NUM_SEQS=16 VLLM_CLIENTS_PER_INSTANCE=4 \
  bash scripts/generate_data.sh --count 500
```

### 运行命令

```bash
# 统一生成脚本（推荐，自动检测并行策略，自动管理 vLLM）
# 默认 Teacher: 本地 Gemma-4-31B-it
# 输出到 data/runs/{MMDD_HHMM}/，并自动更新 data/train.parquet 符号链接
bash scripts/generate_data.sh --count 500 --val-count 100

# 指定 run-id（默认自动取当前时间 MMDD_HHMM）
bash scripts/generate_data.sh --count 500 --run-id 0709_1500

# 单 domain 快速测试
bash scripts/generate_data.sh --domain calendar --count 200
```

生成完成后：
- 数据写入 `data/runs/{MMDD_HHMM}/train.parquet` 和 `val.parquet`
- `data/train.parquet` / `data/val.parquet` 符号链接自动更新
- 主日志写入 `logs/{MMDD_HHMM}_gen_{N}.log`
- vLLM 日志（如有）成功后自动删除，失败时保留在 `logs/`

## 训练数据读取

训练时通过环境变量指定数据路径：
```bash
export OVAL_TRAIN_FILE=data/train.parquet
export OVAL_VAL_FILE=data/val.parquet
bash scripts/train_grpo.sh
```
