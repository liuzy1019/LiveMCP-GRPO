你正在审查 LiveMCP-GRPO 项目的 PROVE Teacher 数据生成链路。

本轮只审查一个 domain：

DOMAIN = {domain_name}

目标不是泛化重构，也不是评价 Teacher 模型强弱，而是核实该 domain 从 MCP handler 到 Teacher-visible context、生成 trace、Replay 和 RL rollout 的系统语义是否一致。

所有结论必须基于实际代码、真实 schema、handler 分支、状态字段和 trace。不得用注释、README 或函数名代替代码证据。文档只能用于确认设计要求，不能证明实现已经成立。

如果当前无法运行环境，先完成静态审查，并明确标记哪些结论需要 runtime trace 才能验证。不得伪造运行结果。

==================================================
一、必须遵守的 PROVE baseline 约束
==================================================

1. PROVE §3.2 主链：

dependency graph
→ Live-State sampling
→ state-machine Teacher
→ robustness knobs
→ fresh Replay + provenance + Jaccard dedup

2. Live-State context 必须绑定 session。新 session 重新探测，同一 session 的多轮 conversation 可以复用或刷新 context。

3. Conversation continuation 是显式的第 2/3 轮 user message，不是系统偷偷执行的动作。

4. Rollout 只有在上一轮 terminal 与 round contract 匹配后，才能注入下一条 conversation query。

5. 不得自行增加以下 corpus hard gate：

- exact-chain hard gate
- query/tool-name 词法匹配 gate
- lexical capability gate
- 通用 mutation/state-delta gate
- 要求 report_error 之前必须先制造 execution failure

6. PROVE 公开 hard gates 只有：

- fresh Replay 的 schema/execution error rate ≤ 30%
- sensitive parameter provenance
- tool-call sequence Jaccard 0.70

7. Graceful give-up 是合法行为。当当前 candidate tools 无法完成目标时，Teacher 可以直接 report_error，不要求先执行失败工具。

8. 论文未公开的 traversal、continuation 概率和语义启发式必须标为本地实现选择，不能声称是论文原始算法。

==================================================
二、审查范围
==================================================

请按以下调用链逐层检查，不得跳层：

A. Tool schema
B. MCP handler
C. Session initial state / state seeding
D. Read-only discovery tools
E. Probe observation entity extraction
F. Entity quality filtering
G. Teacher-visible state projection
H. Dependency graph and chain semantics
I. Initial query generation
J. Tool execution and recovery
K. Follow-up generation
L. Replay and provenance
M. Parquet round contracts
N. RL rollout consumption

重点寻找“真实状态正确，但 Teacher-visible context 丢字段”的问题。

==================================================
三、Tool schema 与 handler 审查
==================================================

枚举该 domain 的全部用户可见工具，并为每个工具建立语义表：

| Tool | Read/Write | Required args | Handler 前置条件 | 成功后的状态变化 | 返回字段 | 失败条件 |
|---|---|---|---|---|---|---|

必须直接阅读 handler 的实际条件分支。

重点检查：

1. schema required/type/enum 是否与 handler 实际接受参数一致；
2. handler 是否使用了 schema 未声明的隐含参数；
3. schema 描述是否准确表达 handler 的状态前置条件；
4. handler 成功后修改了哪些字段；
5. discovery/get/list/search 工具是否返回后续工具需要的全部 ID 和状态字段；
6. handler 返回的 success、state_changed、observation 是否与真实效果一致；
7. 是否存在“返回 success，但目标状态没有变化”的幂等场景；
8. 是否存在依赖特殊 status、balance、stock、permission、lifecycle 的操作。

不要因为 handler 执行失败就直接判定 Teacher 错误。先判断 Teacher 是否看得到满足前置条件所需的信息。

==================================================
四、Live-State 与 Teacher-visible projection 审查
==================================================

针对每一种主要 entity type，建立字段传播矩阵：

| 字段 | 真实 state 存在 | discovery 返回 | entity extractor 保留 | quality filter 使用 | summary 保留 | Query Teacher 可见 | Follow-up Teacher 可见 |
|---|---:|---:|---:|---:|---:|---:|---:|

至少检查：

- entity ID
- display name/title
- lifecycle status
- 时间字段
- location
- balance/amount/price/stock
- owner/member/attendee
- reminders/labels/tags
- permission/availability
- handler 前置条件依赖的所有字段
- 上一轮工具操作产生的 outcome 字段

重点识别以下模式：

1. 真实 state 有字段，但 discovery 没返回；
2. discovery 返回了，但 `_extract_probe_entities` 没保留；
3. entity record 保留了，但 `_format_entity_summary` 丢失；
4. initial Query 能看到，但 follow-up refresh 看不到；
5. summary 截断或字段白名单导致 outcome 不可见；
6. quality predicate 需要字段，但 probe record 中没有该字段；
7. chain context 暴露了不可执行实体；
8. Teacher 被要求避免重复，但无法看到上一轮目标已经满足。

任何字段缺失都必须指出具体丢失层级。

示例判定：

真实 event.reminders 已含 15 分钟
→ get/list event 返回 reminders
→ entity extractor 保留 reminders
→ `_format_entity_summary` 未输出 reminders
→ Follow-up Teacher 无法知道 reminder 已存在
→ 重复请求属于 state projection 缺口，不归因于 Teacher 能力

==================================================
五、Dependency chain 语义审查
==================================================

检查该 domain 实际生成的 dependency edges 和长度 2–5 chains。

对每条代表性 chain 判断：

1. explicit edge：
   source output 是否确实提供 target 的 required input？

2. implicit edge：
   source 是否确实建立 target 成功所需状态？

3. 是否只是同 entity 类型，但不存在真实依赖？

4. chain 中后续 handler 的前置条件，是否由：
   - 初始 live state；
   - 前序工具创建的 state；
   - 前序 observation；
   明确满足？

5. chain-aligned context 是否包含执行该 chain 所需实体及字段？

6. 是否存在 handler 语义上不可执行的 chain？

不得要求 Teacher 必须严格复现 chain 才算有效。Chain 是 query/oracle synthesis seed，不是额外 corpus hard gate。

输出代表性结果：

| Chain | 边类型 | Handler 上是否可执行 | Context 是否充分 | 风险 |
|---|---|---:|---:|---|

==================================================
六、Initial Query 与 Follow-up 语义审查
==================================================

Initial Query 检查：

1. 是否表达 chain 最终目标；
2. 是否引用真实存在的实体；
3. complete/missing-required/minimal 是否符合对应信息级别；
4. 是否泄露工具名或内部执行步骤；
5. query 所需状态是否可由当前 candidate tools 完成。

Follow-up 检查：

1. 是否与上一轮共享至少一种连续性：
   - 同一用户目标；
   - 同一实体；
   - 同一事务；
   - 上一轮结果引出的自然下一步；
   - 对上一轮结果的查询、确认、调整、撤销或补充。

2. 是否只是“同一个 domain”，但换了实体、目标和业务事务；

3. 是否重复请求上一轮已经满足的状态；

4. 如果重复，Teacher-visible refreshed state 是否明确暴露了完成状态；

5. 是否错误地让用户知道内部 tool calls；

6. 是否把 assistant confirmation 写成 user follow-up；

7. 是否因 summary 缺字段而无法生成合理 continuation。

语义分类：

- `same_task_continuation`
- `same_transaction_extension`
- `same_entity_new_subgoal`
- `same_domain_unrelated_task`
- `duplicate_completed_action`
- `ambiguous_due_to_missing_state`
- `invalid_followup`

banking/payments 不能仅凭“还是银行业务”判为 continuation。必须检查实体、事务和目标的连续性。

==================================================
七、问题归因规则
==================================================

每个异常必须归入以下一种主要根因：

1. `handler_semantics`
   - handler 前置条件或状态变化不合理；
   - handler 返回值与真实状态不一致。

2. `schema_contract`
   - schema 与 handler 参数、类型、enum、描述不一致。

3. `discovery_coverage`
   - readonly 工具没有返回后续决策所需字段或实体。

4. `entity_extraction`
   - observation 有字段，但 extractor 丢失或错误分类。

5. `quality_filter`
   - 不可执行实体没有被过滤，或合法实体被错误过滤。

6. `state_projection`
   - entity record 有字段，但 Teacher-visible summary/context 丢失。

7. `dependency_graph`
   - edge 或 chain 不符合真实 handler 依赖。

8. `query_generation`
   - Teacher 已获得充分、正确状态，但 Query 仍语义错误。

9. `continuation_prompt`
   - 状态充分，但 follow-up prompt 允许同域无关目标或重复动作。

10. `teacher_behavior`
    - schema、状态、projection、prompt 均充分正确，Teacher 仍违反明确约束。

11. `replay_or_rollout_contract`
    - Teacher trace 正确，但 Replay、Parquet 或 rollout 消费错位。

12. `insufficient_evidence`
    - 静态代码不足以判定，需要 runtime trace。

只有排除 1–9 和 11 后，才能把问题归因于 `teacher_behavior`。

==================================================
八、最小测试设计
==================================================

为该 domain 设计最小、定向测试，不要一开始运行全量生成。

Test 1：State projection

- reset 一个确定 seed；
- 调 readonly discovery；
- 找到一个代表性 entity；
- 验证 handler 前置条件依赖字段存在于 observation；
- 验证 entity record 保留；
- 验证 Teacher-visible summary 保留。

Test 2：Mutation → refresh

- 执行一个合法 mutation；
- 获取真实 state；
- 重新执行 readonly discovery；
- 构造 follow-up Teacher 可见 context；
- 验证 mutation outcome 在 summary 中可见。

Test 3：Duplicate prevention

- 先完成一个操作；
- 刷新 context；
- 检查 Teacher 是否能从输入判断操作已完成；
- 再生成 follow-up；
- 不把 Teacher 输出本身作为唯一断言，先断言输入信息充分。

Test 4：Chain executability

- 从实际 dependency graph 选择代表性 chain；
- 使用 chain-aligned context；
- 顺序执行；
- 记录每一步 schema valid、execution status、observation 和 state delta。

Test 5：Follow-up continuity

至少检查三类 trace：

- 正常同任务 continuation；
- 同域但无关新任务；
- 因状态字段缺失产生的重复任务。

Test 6：Replay parity

- 使用相同 seed fresh reset；
- 重放完整 attempt trace；
- 检查 schema/execution error rate；
- 检查敏感参数 provenance；
- 不额外增加论文未公开 hard gate。

Test 7：Rollout round contract

- 初始 prompt 只包含首问；
- 上一轮 terminal 匹配 contract 后才注入 follow-up；
- 注入后 required-tools 计数按新 round 重置；
- 同一 session 的状态保持连续。

如果不能运行测试，输出准确的测试文件、fixture、调用顺序和断言设计，但不要声称测试通过。

==================================================
九、修复原则
==================================================

1. 优先修复最窄的根因层。

例如：

真实 state 和 discovery 正确
+ summary 丢字段
→ 只修复 state projection
→ 不修改状态机、采样算法或 corpus hard gate

2. 不得通过扩大 prompt 塞入完整数据库状态来掩盖字段设计问题。

3. summary 应保留：

- handler 前置条件字段；
- 用户可感知的 outcome 字段；
- continuation 判断“是否已经完成”所需字段。

4. domain-specific 字段优先使用结构化 projection，不要无限扩大通用字段白名单。

5. Prompt 防重复要求只能作为辅助。若 Teacher 输入中没有完成状态，不能靠“不要重复”解决。

6. 每次只修复当前 domain 的已证实问题，避免跨 domain 猜测式重构。

==================================================
十、输出格式
==================================================

# {domain_name} Semantic Audit

## 1. 结论

- 状态：PASS / PASS WITH RISKS / FAIL
- 最大问题：
- 是否属于 Teacher 能力问题：
- 是否影响 PROVE baseline：
- 是否需要 runtime 验证：

## 2. Tool/Handler 语义表

完整表格。

## 3. 字段传播矩阵

完整表格。

## 4. Dependency chain 审查

列出代表性 chain、可执行性和 context 充分性。

## 5. Follow-up 语义审查

对每条提供的 trace：

- 前一轮目标
- 前一轮实际状态变化
- Follow-up 请求
- 实体连续性
- 事务连续性
- Teacher 可见状态是否充分
- 分类
- 根因

## 6. Findings

按优先级排序：

### [P0/P1/P2] 标题

- 代码证据：
- 真实状态：
- Teacher-visible 状态：
- 语义后果：
- 根因分类：
- 最小修复：
- 不应修改的部分：
- 测试建议：

没有代码事实支撑时不得输出 finding。

## 7. 测试计划或测试结果

逐项列出：

- 测试输入
- session seed
- 操作顺序
- 预期状态
- Teacher-visible context
- 断言
- 是否实际运行

## 8. 最终建议

只能选择：

- 无需修改；
- 只修 schema；
- 只修 handler；
- 只修 discovery/extraction；
- 只修 state projection；
- 只修 continuation prompt；
- 修复多个已证实层；
- 证据不足，先采集 runtime trace。

禁止使用“整体重写”“增强智能”“换更强 Teacher”这类不可验证建议。
