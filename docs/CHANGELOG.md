# 版本更新记录

本文件记录 LiveMCP-GRPO 各版本的功能新增、行为变化和缺陷修复。

格式参考 [Keep a Changelog](https://keepachangelog.com/en/2.0.0/)，版本号遵循 [Semantic Versioning](https://semver.org/)。新变化先写入 `Unreleased`，形成版本时再移动到带日期的版本段。

## [Unreleased]

### Contract boundary

- 最终 corpus 取消十域均匀配额：每域保留可配置最低 train/val 覆盖，剩余名额按真实候选中
  live-state feasible、位置感知 Jaccard-unique 的 dependency-chain 容量分配。初始 shard 的均匀
  candidate exploration 仅用于估计容量，不再被描述为 PROVE 分布要求。
- Required-workflow no-op 投影增加显式执行语义：`state_transition` 型成功 no-op 从 GT 排除，
  `action_execution` 型成功动作即使净状态未变化仍保留；完整 factual trace 与 PROVE hard gates 不变。

- Teacher action guidance 明确三项本地语义合同：跨轮唯一指代不得伪造 ambiguity；多目标请求
  不得因一个子目标失败而放弃独立可行的其他目标；missing-function 只有在补充信息能够解除
  阻塞时才使用 `ask_clarification`。这些约束不加入 PROVE 公开 corpus hard gates，也不增加
  exact-chain 或通用自然语言 judge。
- Question-shaped `final_answer` 校验覆盖回答中间明确向用户索取输入的二人称问句，同时允许
  引用带问号的标题或历史文本。Email seeded state 改为只将相同规范化 subject 聚合到同一
  thread，不再按记录序号把无关主题机械分组；旧 initial-state fingerprint 数据必须重生成。
- Global merge 增加本地 trajectory-integrity gate：同一 round 中真实 execution failure
  若未被同名 capability 的后续成功重试消解，不得以 `final_answer` 声称完成；
  对带 `id` / `*_id` 目标参数的 mutating capability，成功重试必须保持相同目标身份，
  防止通过操作另一资源掩盖原失败。`report_error` / `ask_clarification` 仍是合法 recovery terminal。该规则只隔离
  “失败未解决却宣告成功”的不可训练标签，不改变 PROVE 公开 30% Replay error 阈值。
- Sensitive provenance 数值归一化支持千分位金额，并识别 `$`/`USD` 等明确货币符号；
  filesystem `readlink` live-state feasibility 只接受明确 symlink/link 类型；email 历史
  labels-aliasing 检查不再把 reply/send 新建邮件的合法空 labels 字段误判为污染。
- 轨迹合同更新为功能化名称 `live-mcp-canonical-replay-trajectory-v1`：canonical Parquet 必须持久化逐轮 Teacher query/oracle/history、全部真实 execution attempts，以及最终 required-workflow 的 fresh replay 证据；生成 readback、merge、训练预检和 rollout 共用同一证据验证器。历史数据缺少该证据，当前不得训练。
- `ToolSemantics` 收敛逐工具 operation、sensitive fields 与允许 mutation roots；entity、requirements、relevance 仍由 orchestrator 的单一解析器提供，并通过显式 callback 给 planner 消费。PROVE 五步算法和公开 replay/provenance/Jaccard hard gates 不变。
- 旧训练入口、旧 import re-export、DomainAdapter 名称前缀 fallback 和 reward 配置 fallback 纳入本轮删除范围；统计/生成/serving 的当前运行容错不属于旧兼容层。
- 删除 OracleCall dict、AuditEvent 手写缩减格式、reward 无 adapter 启发式和 `unseeded_fallback` 四条旧兼容路径；未知域/工具、缺失 adapter 或缺失 canonical safety batch 均 fail-closed。

### Removed

- 删除旧外部数据转换、prompt 检查、数据实验记录、无消费者 reward/saturation 模块和
  package re-export；历史 Parquet、旧模型别名及失效测试产物不再保留。
- 运行入口改为功能命名：`generation_runtime.py`、`merge_generation_shards.py`、
  `serve_policy_model.sh`；新增 `audit_generated_data.py` 复用生产 parser 与质量门禁逐行审计。

- 删除十域 YAML 中未被运行链路消费的手写 dependency edges、query templates 和
  domain reward profile；依赖图的唯一权威来源是绑定 schema、Teacher 和 classifier
  contract 的 `data/dependency_graphs` cache。
- 删除 suite 中未消费的 live/trace 占位配置、并行调用开关、错误停止开关和旧 reward
  weights；保留真实消费的 subprocess、session、observation budget 与 supported profiles。
- 删除未被正式 FSDP 训练入口引用的旧 `configs/ds_zero2.json`。
- 删除无调用方的旧 JSONL task save/load/summary facade、未接入 manager 的
  `InProcessTransport` 和已被 fresh replay 替代的 `OracleValidator`；保留 replay 仍使用的
  `criterion_satisfied`。
- 清理失效 dependency cache、lock、断链 parquet symlink、空 run、运行日志、Python/pytest
  cache，以及未被文档引用的派生 HTML/图片/论文文本副本。
- 删除 README/launcher 中并未由 shell 参数解析器支持的外部 `--api-base` 示例，以及指向
  不存在 LICENSE 文件的 badge。
- 删除 shell 错误提示中的机器绝对环境路径；运行时继续通过 `PYTHON_BIN` 注入解释器。

### Fixed

- Payments invoice lifecycle now rejects `pay_invoice` unless the invoice is
  genuinely unpaid (`pending` or `overdue`) and has no open dispute; a dispute
  can no longer be silently overwritten by a later payment while the dispute
  remains open.
- Calendar attendee-bearing tools now expose email-shaped JSON schemas and
  enforce the same email validation in handlers.  Names such as `"Sarah"`
  cannot be stored as deliverable attendees or queried as free/busy identities;
  the caller must first obtain an actual email address.

- Teacher launcher 的最终 Parquet 门禁改为直接调用权威
  `scripts/audit_generated_data.py`，删除重复的内联 pandas/reward parser 实现。定向 smoke
  曾在内联检查已打印 `Parquet validation PASSED` 后于解释器关闭阶段触发 SIGABRT，导致
  合格产物被错误标记为生成失败；统一入口同时避免两套质量合同继续漂移。
- Teacher generation 与独立 Policy serving 默认关闭 vLLM usage telemetry，并将
  vLLM 可写 cache/config 根目录放到 `${TMPDIR:-/tmp}`；用户 home 所在分区空间耗尽时，
  不再因写 `~/.config/vllm/usage_stats.json` 污染或中断模型启动。路径仍可由环境变量覆盖。

- 重建 calendar 当前 schema 的完整 `C(17,2)=136` pair cache；十域 Stage 1/2 验证为
  `51 passed / 0 failed`，并验证模型服务关闭后仍可严格命中磁盘缓存。

- Required-workflow 投影后对精确导出的 oracle、hidden tools 和 success criteria 再执行隔离 fresh replay；过去只验证投影前 Teacher attempt trace，存在“原始尝试可执行但最终训练标签未自动复验”的结构缺口。
- 成功但 `state_changed=false` 的 mutating no-op 不再成为 required oracle；真实尝试仍保留在 Teacher trace。Question-shaped `final_answer` 会触发格式重试并要求使用 `ask_clarification`。
- `payments.delete_webhook` 现在真实删除资源，重复删除 fail-closed；`calendar.create_recurring` 拒绝 `BYDAY` 与 DTSTART weekday 不一致的规则。calendar schema 变化使旧 dependency cache 显式失效。

- 修复多轮 reward 把未来 round 的工具调用提前计入 GT coverage/argument 匹配的问题；每个 action event 携带真实 `round_idx`，GT call 只与同轮事件匹配。
- 缺失、重复、越界或类型不匹配的 per-round terminal 现在作为不完整 trajectory fail-closed；有效 trajectory 的 PROVE 五组件公式不变，该结构门禁不是第六项 reward。
- train/val 首轮 query 隔离改为十域全局分组交换，同时保持每域 quota；Jaccard signature 增加 primary domain，跨域同名工具不再互相误删。
- 未来生成行持久化 `teacher_attempt_trace` 和 `teacher_round_trace`，旧行不能仅靠补写版本号或 fingerprint 混入当前训练。

- 候选层、首轮检查和最终 oracle guard 现在共用同一零工具 terminal 合同：`minimal` 与
  `missing` 可直接 `ask_clarification`，任何 difficulty 在当前 schema/state 已证明不可完成时
  可直接 `report_error`；不再要求 Teacher 先制造一次失败工具调用。
- Teacher query synthesis 在结构化返回 `UNSAT` 时立即放弃当前固定 chain/state 候选，交由
  pool-level oversampling 采样新链；解析失败和 target mismatch 仍可重试，PROVE 依赖图与
  corpus hard gates 不变。
- vLLM 明确报告输入加输出仅超出 context window 少量 token 时，Teacher 客户端保持完整输入
  并仅将本次 decode budget 收缩到剩余窗口后重试一次；剩余预算不足或非上下文错误继续
  fail-closed，不通过截断 Teacher 输入掩盖信息缺失。

- 修复 missing-function 零工具 `report_error` 被 reward readback 错拒的问题；
  `ask_clarification` 与 graceful `report_error` 均按同一生成合同解析，普通任务仍 fail-closed。
- 普通与 irrelevant 样本统一持久化 provenance 结果和 primary initial-state hash；
  Parquet readback 现在同时验证 replay、provenance、owner schema/transition/initial-state 合同。
- 十域 seeded state 与 handler 新建记录统一使用 seed 对应的 reference date，消除
  prompt 的“今天”与 live entities 固定在 2026-06-24 的跨层时间漂移。
- `SchemaRegistry.get_schema()` 对无 domain 的跨域同名工具不再按注册顺序返回首项；
  歧义输入 fail-closed。
- dependency validator 分开报告 raw graph paths 与静态 state-precondition feasible paths，
  并增加不修改 cache 的 producer/requirement 语义诊断；recovery 默认值统一为 3。
- 修正文档中的 PROVE reward 定义：`R_coverage` 为 dependency-ordered GT step
  coverage，`R_name` 为 per-model-call name precision；实现原本即与论文一致。

- 使用 ARL 解释器恢复 `./verl` 0.6.1 editable 安装及四个缺失 base 依赖；
  vendored verl 增加 Transformers 4/5 视觉 AutoModel 名称兼容层，保留 Gemma-4 Teacher
  所需的 Transformers 5.13.0，而不再为了 verl 导入降级 Transformers。
  `main_ppo`、项目 estimator 注册和 Gemma-4 vLLM serving 需同时通过。
- stateful request timeout 现在使整条 session quarantine；fresh Replay 将“预期失败但实际成功
  且产生 mutation”作为 trajectory outcome mismatch 直接失效，不再由论文 30% 普通
  schema/execution error rate 吸收。
- server handler 的返回 envelope、after-state 和 delta 校验纳入同一事务边界；rollout action
  audit、diagnostic 与 reward integrity 分离，基础设施异常不再伪装为模型普通零奖励。
- 环境 metadata 校验统一到生成、merge、训练预检与 rollout 共用入口，并分层绑定 schema、
  transition/seeder、observation 和 reward contract；Parquet 写盘后逐行走生产解析链。
- planner 已删除重复实体映射并直接消费 orchestrator 的权威实体/依赖解析器；解析失败不再
  静默退化为空依赖。尚未完成的 domain-local `ToolSemantics` 迁移不写成既成事实。
- 十域 tool schema 在 suite bootstrap 原子发现一次，后续 seeded session 复用进程级快照，
  state sampling 仍逐 session fresh；这移除了每条生成/replay/rollout 的十次重复 schema RPC。
- 删除十域 YAML 中未被运行时消费且不完整的 tool 分类副本；server `TOOLS` annotations 现在是
  schema 与 readonly/mutating 分类的唯一真源，Stage 2 核验全部 190 个工具的互斥分类。

- `merge_generation_shards.py` 直接脚本入口显式注入项目根目录，与 launcher 的实际调用一致
  方式一致，不再因找不到 `src` 而在 global merge 阶段失败。
- 幂等 `cd`、同数量购物车更新和重复 `split` 返回真实 `state_changed=False`，不再被事务
  边界误判为 execution error。
- Payments live-state 候选过滤排除已有 linked payment 且无法先取消的 invoice；Teacher 与
  Policy observation budget 统一由 suite 注入并在训练前核对。
- dependency classifier 删除与 pre-existing state 规则冲突的 reversible setup 示例，
  semantics 升至 v11；v10 cache 必须重建，不能继续复用。

- MCP handler 调用增加统一事务快照：异常和错误 `state_changed` 声明会回滚，成功调用记录真实 `state_delta_paths`；Parquet 保存 criterion 到成功调用的 provenance，但不把该诊断加入 PROVE hard gate。
- Reversible 目标改为优先使用 live-state 既有实体：新增 banking `list_scheduled_transfers`，并 seed 可删除 webhook 与可移除 wishlist item；不再依赖无最终效果的 create-then-delete setup。
- Global merge 增加历史 team-chat reaction alias 污染识别。对旧 500+100 逐行复核确认 20 条别名污染（train 15、val 5），且 600 条均为 stale `live-mcp-trajectory-v0`，全部阻断训练。
- 修复 state seeder 中 email `labels` 与 team-chat `reactions` 复用可变模板 list 的实体别名；单实体 mutation 不再连带修改其他实体。旧 email 轨迹中出现无关 label state criteria 的行必须隔离并用修复后环境重建。
- 修复 global merge 可能把相同 domain、相同规范化首轮 user query 分到 train 与 val 的评测泄漏；最终 split 现在按首轮输入分组隔离，不新增 query-text corpus hard gate。
- 删除 vLLM 0.19.1 已不识别的 `VLLM_ATTENTION_BACKEND` 环境变量，不再产生无效 backend 告警；attention backend 由当前 vLLM 按模型和硬件自动选择。
- 修复 shard/global merge 职责倒置：`--shard-mode` 不再以本地 Jaccard-unique 数作为成功条件，只输出 replay/contract eligible 候选；Jaccard 0.70 与逐域配额只在 global merge 执行。Top-up 不再因单个局部失败立即丢弃其他成功结果，按各域实际全局 Jaccard 保留率估计候选预算，并在 generation client 槽位内切分并行；checkpoint 反序列化同时恢复逐轮 `OracleCall` 类型。
- Teacher 与 Policy 统一使用完整 loss-aware execution envelope，补齐 Teacher history 中的 `schema_valid`。
- 环境合同按 executable owner domain 保存 `server_schema_hashes`，覆盖跨域 distractor schema drift。
- strict `prove_baseline` 不增加公式外 reward 项；round/terminal 不完整时整条 trajectory 在五组件计算前失效，有效轨迹的论文五组件得分保持不变。
- irrelevant query synthesis 与 task lifecycle 接入统一 Teacher trace；Parquet 增加 `has_state_outcome_oracle` 诊断字段。
- 将 handler 已执行的正数/非负数前置条件同步到公开 JSON schema，避免 Teacher/Policy 只能在执行失败后获知参数边界。
- 5% irrelevance、difficulty mix 与 robustness rate 仍是候选采样目标，不改成论文未要求的小批次精确 hard quota。
- `prove_baseline` 绑定 verl 原生 GRPO，并移除 tool-required 轨迹上的 terminal/identity
  formula-external reward 惩罚；`oval_full` 才启用这些项目扩展。
- 修复训练入口把 vendored `./verl` 放在项目根之前、导致 `verl/scripts` 抢占项目 `scripts`
  包的问题；项目根现在始终优先。
- 删除与当前 per-prompt rollout group 不兼容的旧 2D StratAdv 和 E4 3:3:3 预检。
  `livemcp_grpo` 对同一 prompt 做标准组内归一化，混合 perturbation/scenario 的 group
  直接拒绝；LATA 失败不再静默降级。
- 训练 launcher 删除全局 `ray stop --force` 和按进程名 KILL vLLM 的清理逻辑，退出时只清理
  当前训练调用链拥有的资源。
- reward 删除未接入生产链的 last-observation state approximation；bool 参数不再与 0/1
  数值错误匹配。

### Changed

- 标准 MCP SDK transport 增加 `structuredContent` 与完整 `content[]` envelope normalizer；单 JSON text 保持向后兼容，显式 `mcp_stdio` 配置才启用，不改变本地十域 subprocess JSON 链路。
- Teacher 与 GRPO rollout 共用 versioned loss-aware observation projector；成功/失败统一保留执行状态、错误、`state_changed`、schema validity 和业务 observation，Policy observation budget 改为配置读取，不再 raw prefix truncation。
- reward 配置增加显式 `prove_baseline` / `oval_full` profile；strict baseline 强制关闭 safety/shaping/process 扩展，训练配置快照记录最终目标公式与环境合同版本。
- 正式 DomainAdapter operation 只消费 live schema/ToolSemantics；未知工具 fail-closed，不保留名称前缀 fallback。
- Parquet 与 rollout metadata 增加 schema、observation/projection/domain contract 版本、observation budget 和 reward profile compatibility，用于训练前兼容检查。
- loss-aware projector 的极限压缩保留首尾实体 ID，不再从尾部单向删除刚创建或最近返回的实体。
- `scripts/train_grpo.sh` 统一委托正式入口 `src/training/run_grpo.py`，确保 profile 导出、LambdaState 初始化/禁用、长度预检和 estimator fail-fast 位于真实训练调用链。
- 正式入口在 LiveMCP estimator 注册失败时立即终止，不再记录错误后继续启动 Ray。
- dependency pair 合同增加保守的 typed output/reference 连续性检查；例如 `create_draft` 只返回 draft，不能满足只接受 email 的 `move_to_thread`，而 `create_subtask` 返回的 `issue_id` 仍可供后续 issue 工具使用。违规 pair 仍交给 Teacher 重判，不手工改写 graph。
- Query Teacher 单次响应增加 mutating capability 的 query 原文授权证据；只在生成阶段核对证据覆盖与原文存在性，不增加第二次 LLM judge、词法 capability gate 或 exact-chain corpus gate。
- live-state compact summary 补齐 channel/thread 等关系字段和 `created_at`；Food Delivery seeded order 与 query reference date 使用同一 seed temporal anchor，避免未来订单和不存在的 relative-date 实体。
- 完整 Teacher trace 与 RL required workflow 分离：真实调用全部保留，只有 execution history 明确标记的同轮 exact no-progress repeat 从训练 oracle、round contract 和 ground-truth call budget 排除。

- dependency graph cache 增加关系定义合同校验：拒绝 required-input 为空的 `explicit` target 和 readonly/non-mutating 的 `implicit` source；只让 Teacher 重判违规 pair，不手工改写其余缓存分类。
- dependency graph 局部修复跳过已由缓存完整覆盖的 batch，避免向 Teacher 发送空 pair 请求；修复缓存时的决策次数等于实际缺失 pair 所在请求，不再随整个 domain 的 pair 数增长。
- dependency pair 重试会把上一响应违反的关系定义反馈给 Teacher，避免在相同 prompt 上重复产生同一非法分类；最终关系仍由 Teacher 输出，代码不代判。
- `validate_generation_pipeline.py` Stage 1 同步执行关系定义合同校验，避免结构完整但语义自相矛盾的 cache 被误报为 strict pass。
- 清理当前生成/奖励链路中已不可达的 helper 与旧两级 validity 兼容参数：删除未被任何生产或测试调用的 entity/chain/irrelevance helper，移除不参与 PROVE 五项 reward 计算的 `w_struct/w_exec`、`r_positive/z_pos` 和旧 `_compute_structural_validity` 包装。Validity 只保留论文公开的 name/schema/execution 三级等权实现。

- 正式生成默认 recovery 上限提高到 6 轮，launcher 为每个 shard 写 checkpoint；单 shard 临近配额时可续跑，不再因默认三轮耗尽丢弃数小时候选池。
- 明确 PROVE continuation 的 2--3 turns 是 conversation rounds，而 rollout `budget` 是本项目 action-turn 工程合同；逐行预算至少覆盖全部 reference tool calls 和每轮 terminal，adaptive efficiency budget 仍只用于 reward。
- Teacher recovery prompt 明确 alternative tool 必须保持用户请求的同一结果；当前能力无法完成时 graceful give-up，不把不同业务 mutation 当作替代操作。该提示不新增论文外 corpus hard gate。

- Teacher action prompt 不再接收 dependency graph hints、chain progress 或 must-execute chain；missing-function 允许可见前置调用，并删除 zero-tool 与首轮 action-type hard blocking。
- PROVE chain 恢复为 query-generation seed：Teacher 不再接收剩余 chain 作为隐藏的 must-execute Oracle 清单，而只根据 query、tool schemas 与真实 execution history 决策。
- dependency pairwise classifier 收紧为论文定义：可选顺序、主题相关和“有帮助但非必需”的 read step 均分类为 `none`；当前 semantics v10 十域严格缓存已完整重建。
- PROVE 多轮语义恢复为首轮 chain-seeded query 加 live-state grounded continuation；normal success 生成 2--3 turns，后续 turn 不重新构造论文未描述的 dependency chain，clarification/missing-function/irrelevance 可提前终止。
- dependency graph build/load 保留完整 LLM pairwise 分类；handler 事实只用于下游 live-state feasibility、真实 execution 与 replay。Replay 仅豁免成功执行的空结果，不再把 `not found` execution failure 当成空结果。
- 删除按持久 mutation 数量和 filesystem `readlink` 的 chain 采样偏置，并移除 mutating-step 词法授权 gate。
- 十域生成的 recovery 轮数改为可通过 `--max-recovery-rounds` 配置（默认 6）；避免少数高重复 domain 在固定三轮后仅差少量唯一轨迹而丢弃整轮结果。
- 数据生成新增 `--checkpoint-path`：逐轮原子保存内部 `LiveTask` 候选池、已完成轮数、下一轮 domain 缺口及配置指纹；相同配置重启后从下一 recovery 轮继续，配置不一致则 fail-fast，避免末段配额不足丢弃数小时有效候选。
- dependency classifier 从每批 12 pair / 2048 tokens 调整为 8 pair / 512 tokens，避免 Gemma 在小 JSON 任务上长解码阻塞冷缓存。
- PROVE 重新对齐：移除每轮必须命中 chain 精确目标工具名的 hard gate，允许 recovery 使用可执行 alternative tool。
- missing-function 允许先执行仍可用的前置工具，再以 clarification / abstention 结束；只继续禁止 hidden tool 泄漏或执行。
- 删除 replay 前的 exact-query 预去重；Jaccard 0.70 改为对全部 surviving conversations 统一执行，不再按 domain 豁免。
- 删除零工具轨迹的 query-text Jaccard fallback；论文未发布该规则，空 tool-call sequence 不据此互相去重。
- dependency graph 按论文公式对每个无序 `C(n,2)=n(n-1)/2` pair 只分类一次，由 LLM 输出有向 source/target；移除 LLM 分类后的 domain 黑名单、entity heuristic 与强制破环，旧有序 `n(n-1)` cache 自动失效。
- 移除“oracle 必须逐节点完整覆盖 source chain”的论文外 hard gate；末端用户目标仍需真实完成，未完整覆盖时只保留 `source_chain_seed`，不生成虚假的 OVAL dependency edges。
- 修正 missing-function 比例误读：论文的 20% 属于 `missing-required` information level；默认 missing-function 目标改为按公开 corpus count 推导的 `1500/(10895+1500)≈12.1%`。
- dependency graph cache key 绑定 schema 和 dependency semantics；handler precondition 仅用于 live feasibility、execution 与 replay，不再改写或使 LLM pairwise cache 失效。
- 多 client 生成改为单层候选预算和增量 futures 调度；domain 达到 quota 后停止提交新 Teacher 请求。
- shard-local split 不再执行只能在全局成立的 domain/scenario 覆盖门禁；全局 merge 继续检查 validation domain 是否被 train 覆盖。
- banking 实体过滤从“全局排除 frozen account”改为按具体工具链状态前置条件判断。
- Teacher 候选默认只生成一次完整 conversation；失败恢复留在状态机内部，外层 recovery 按 Jaccard-unique 实际缺口动态补量。
- live entity 过滤改为 chain-specific 且 unknown-pass：只有 handler 字段明确冲突时预过滤，避免把 probe 字段缺失误判为无效状态。
- distractor 按 PROVE Step 4 改为在 Teacher 生成前加入 candidate set；enum stripping 与 missing-function 同样作用于 Teacher 阶段。
- 多 shard merge 在全局候选池执行同 domain、位置感知 Jaccard 0.70 后再划分 train/val。

### Fixed

- Query mutation evidence 不再要求作为 chain 前置步骤的 session-internal `cd` 导航成为独立用户副作用；若 `cd` 是最终目标仍要求授权，其他 mutating chain capability 的逐字授权合同保持不变。Shard recovery 与 global top-up 不再重复注入 irrelevance，避免低 yield domain 的补产把 5% 初始采样比例放大。
- 修复 CRM `add_note(entity_type=lead/contact)` 被固定当作 deal dependency、Issue Tracker `user` selector 未绑定 user、Email `archive_email -> add_label` 被误判 delete/recreate 的三处确定性场景分类错误。
- Live MCP stdio manager 将 suite 中的裸 `python` 入口绑定到当前 `sys.executable`，避免指定 ARL 父进程却用系统 Python 启动 server；server 边界异常保留 request id，真实 import/handler 错误不再表现为 10 秒假超时。
- CRM `delete_contact` 维护 lead/contact 引用完整性，不再删除仍由 converted lead 引用的 contact；对应不可执行 create/delete chain 在 live feasibility 阶段拒绝，不改写 LLM dependency cache。
- Teacher-visible live-state projection 改为 domain/entity-specific 字段合同，补齐 Filesystem permissions/size、Email read/archived、Issue Tracker workflow/assignment、Team Chat archived/reactions、Food Delivery tip 和 Payments refund progress。
- Shopping `list_orders` 现在作为 order 的 primary readonly discovery source，order status/total/created_at 不再在 extractor 中退化为 `{}`。
- 已证实的数值 handler 约束同步进入 Teacher-visible schema：Payments invoice amount、CRM deal amount、Food Delivery rating/tip。
- Continuation prompt 明确“同 domain 不等于 continuation”，要求沿上一轮任务、事务、实体或公开结果继续；保持单次 user-message generation，不增加额外 Teacher classifier 或 corpus hard gate。

- missing-function 最终 clarification/abstention 若遗留 hidden-tool 前缀之外的成功 mutation，会在生成阶段重试，避免失败 workaround 的部分副作用成为 required oracle；合法可见 prefix 仍保留。

- readonly live-state probe 现在区分主实体 ID 与外键引用；deal/lead/order/task 等来源记录不再把 `name/status/amount` 覆盖到 contact/product/account 等被引用实体。
- missing-function candidate 若首轮已用可见工具完成原始 query 并返回 `final_answer`，会在 continuation 前作废；后续无关请求的 `report_error` 不再伪装成 hidden capability 导致的 abstention。
- clarification continuation prompt 现在包含当前 domain、visible tool schemas 与前置条件，并禁止引入无关或跨域目标。
- Teacher trace 新增 round input/output boundary；设置 `LIVEMCP_TEACHER_TRACE_INCLUDE_STATE=1` 时额外记录仅供审计的真实 session state snapshot，该状态不进入 Teacher prompt、oracle 或 reward。
- Parquet `extra_info` 补充 Teacher attempt/failure 与 Replay call/error 计数；policy ID 规则恢复为允许来自 user request 或 prior tool result 的真实 ID。
- launcher 的 Parquet integrity check 对显式空 split（例如 `--val-count 0`）只校验 schema，不再调用 `df.iloc[0]` 导致已成功生成的定向 smoke 被误判失败。
- `assignee` 与 `recipient` 等承载稳定 member ID 的非 `_id` 参数现在会正确解析为 user selector；Food Delivery `list_orders.status` 补齐全部合法 order 状态 enum 和描述，未知状态不再以成功空结果混入 oracle。
- `missing_dependency` 轨迹诊断现在会扣除已由当前 tool-call 非空 ID/path selector 直接绑定的实体类型，避免把携带有效 `message_id/channel_id/thread_id` 且 replay 成功的后续写调用误标为缺依赖。
- Step 2 follow-up grounding 扩展到 complete/missing/minimal 全部 difficulty；现有实体的 ID、名称、邮箱、username 或路径必须来自同轮刷新 live state，missing/minimal 只允许遗漏任务参数，不允许伪造实体。
- chain-aligned entity summary 保留 attendees、labels、watchers 和 members 等 handler 必需 supporting data，并对过长 list 做有界摘要。
- Teacher-visible schemas 补齐 CRM deal 正金额、Food Delivery rider 状态、Calendar attendee email、Team Chat member/archived-channel 及 Filesystem move target 不存在等真实 handler 前置条件；Team Chat 已归档 channel 不再进入 `send_message` chain context。
- Food Delivery `update_order_status` schema 公开与 handler 一致的完整 lifecycle 转移图和状态 enum，避免 Teacher 生成 `in_transit` 或跨级状态；enum stripping 仅移除 enum，description 仍保留转移契约。
- `report_error` 轨迹不再误标 `normal_safe_success`：零调用为 `no_tool_or_abstention`，成功前缀后 graceful give-up 为本地 `partial_completion_or_abstention`，有执行失败历史仍为 `tool_error_recovery`。
- `validate_generation_pipeline.py` Stage 1 改为按当前 server tool-schema hash 加载 strict cache，并校验 cache version、`C(n,2)` pair ledger、graph 重建一致性和 classifier provenance；当前 schema 无 cache 时 fail-closed，不再用历史 JSON 假验收。

- success criteria 递归记录现有实体中新建/变化的嵌套字段，Calendar `respond_to_event` 新建的 response map 不再产生空 criteria；Shopping 仅在最终 cart 非空时生成 `cart_not_empty`，不再把已 checkout 清空的购物车判错。
- issue tracker 补充只读 `list_members` 和明确的 `user_id` 参数合同，使 assignee/watcher 能从真实 MCP observation grounding；live-state sampler 会用后续更完整的 member observation 补全先前从 issue 记录发现的同 ID user，并移除 Query prompt 中会污染其他 domain 的虚构姓名示例。
- 修复 Teacher 格式重试耗尽后产生空 oracle round、随后被导出器伪装成 `final_answer` contract；空 round 现在在状态机和导出合同两层 fail-closed。
- continuation user simulator 现在读取上一轮对用户可见的 assistant response，避免操作完成后才反向询问本应在操作前确认的信息；不暴露隐藏 tool execution history。
- 回退状态机内部的重复成功调用抑制及对应 prompt 约束；PROVE 仅公开会话完成后的 replay、效率奖励和 tool-call sequence Jaccard 去重，轨迹内每个 Teacher tool-call action 现在都真实执行并保留 observation。
- 修复 `--shard-mode` 总量不足时再次落入逐域 recovery 的分支错误；shard recovery 现在始终只按全局 unique 行数缺口补量，domain quota 仅在全局 merge 执行。
- 修复 continuation 后续 user round 使用首轮 chain-aligned 子集进行 action grounding，导致真实实体被误报为不存在；后续 query 与 action planner 现在共享同轮刷新 live state。
- 修复新 user round 因历史执行记录非空而跳过首动作合同、允许空 `final_answer` 吞掉 follow-up 的问题；clarification round 单独允许直接回答，follow-up round 仍进入工具执行，空 terminal 会重试。
- Teacher 对缺少用户决定型 mutation 必填值的请求走 clarification，不再从无关实体复制 amount/address 等值；附带业务背景不再被当成跨域 capability 请求。以上均为状态机输入修复，不新增 PROVE 外 corpus hard gate。
- 修复 `--shard-mode` 仍执行逐域 quota recovery/split、导致单 shard 少一域便在全局 merge 前失败；shard 现在只保证总 unique 行数，逐域 train/val 配额仅在 global merge 执行。
- 移除“普通 `report_error` 必须先出现 execution failure”的本地 hard gate；PROVE recovery 允许能力不足时直接 graceful give-up，避免为通过过滤而制造失败工具调用及 payments 配额枯竭。
- 修复多轮 reference 可执行但训练 rollout 被固定 action cap 提前截断：Parquet 记录 `minimum_action_budget`，agent loop 使用经过结构核验的逐行 budget。

- sensitive-parameter provenance 按 conversation round 注入 user query，后续 user turn 不再反向验证更早的 tool argument。
- enum stripping 递归移除嵌套 object/array schema 中的 enum，不再只处理顶层 properties。
- 删除未引用且与当前 pre-Teacher robustness 生命周期冲突的旧 `perturb.py`，以及禁用的 deterministic dependency augmentation 和手写 edge 表。
- 已完成 bound goal 后再输出 `ask_clarification`/`report_error` 的轨迹会被拒绝，避免“已列出频道却继续问是否归档”等无依据终态污染 clarification 场景。
- filesystem `tar_extract` 的 live-state 候选必须是 file，目录不再作为 archive grounding。
- `ask_clarification` 终态无论是否包含可见前置 tool calls，统一标记为 `clarification_required`，避免污染 normal 场景统计。
- reward `_build_task_dict` 不再清空 missing-function/clarification 的可见前置 oracle calls；仅 irrelevance/no-tool 强制零调用。
- recovery 中单个 domain 零产出不再中断其他缺口 domain；初始全链路零产出仍保持 fail-fast。
- 全 domain 生成在 early split 前强制满足逐 domain 配额，最终 split 也按配额选样，避免用其他 domain 的多余样本伪装十域覆盖。
- continuation 恢复 PROVE 公布的 `min_turns=2`、`max_turns=3`；不再因首轮完成 dependency chain 绕过最小轮数。
- `_chain_is_feasible` 现在消费实际 qualified entity record；filesystem 的 `cat/stat/sort/readlink` 等实体消费工具不再绕过 handler state/type precondition。
- food delivery 的 order grounding 上下文包含当前状态的合法下一状态；payments dispute invoice 状态资格与 handler 保持一致。
- capability 同义词、实体字符串匹配与状态反转授权检查从运行链路移除；PROVE 未发布这些 gate。
- food delivery `create_order.items` schema 补全为 `{name, quantity}` object array；Teacher schema formatter 保留 nested array/object 结构，不再只展示顶层 `array` 导致字符串参数进入 handler。
- food schema 变更只使 food domain cache 失效；当前有效 cache 由 schema + semantics v10 + classifier contract 统一索引。

- 下游 feasibility 按真实 handler 状态拒绝 settled payment→cancel、placed order→rider/rating 等确定不可执行链，不回写或手工清洗 raw LLM cache。
- Parquet `extra_info` 现在保留 `chain_seed` 和 `source_chain_seed`，可从数据行反向审核依赖链来源。
- 修复小 shard 已有足量数据却因局部 train/val domain 不一致而整体失败的问题。
- 更新当前可用 `arl` 环境为 `/mnt/data2/liuzhanyi/envs/arl`；训练与独立 vLLM 入口优先使用 `PYTHON_BIN`/`CONDA_PREFIX`，避免 Conda 旧名称索引回退到系统 Python 或其他 `vllm` CLI。
- 训练和独立 vLLM 入口禁用 user-site packages，避免环境内外的 Torch/Transformers 二进制混装。
- 同名工具在缺少 domain hint 且参数无法唯一判定 owner 时不再按注册顺序静默路由。
- complete-query grounding 现在识别前置 creator 产生的实体，不再要求 `create_issue -> comment_issue`、`create_event -> add_attendee` 等合法链引用执行前并不存在的 live ID。
- 移除超出 PROVE §3.2 的 outcome-criteria、scenario/terminal、unsafe 标签及轨迹内 exact-call 拒绝门禁；这些字段保留为诊断，硬过滤仍为 replay、provenance、Jaccard 和必要结构合约。
- 修复 live-state qualified summary 仅按 ID 对齐导致跨实体类型串用摘要的问题，现统一使用 `(entity_type, entity_id)`。
- query synthesis 要求状态变更工具的必填参数必须由用户请求或 live state 提供，避免 Teacher 自行补造 deal amount 等值。
- success criteria 对 generic state delta 与 domain postcondition 做语义去重，list 字段不再重复计权。
- Stage 3 读取生成 parquet 并检查实际行数与逐域配额，不再只凭子进程返回码宣称通过。
- `generate_data.sh` 仅在 train/val parquet 完整性验证通过后发布默认数据链接；失败 smoke 不再污染训练入口。
- 单进程 split 与 global shard merge 分别强制 train/val 逐域配额；单域候选不足时 fail-closed，不再用其他 domain 补位。
- 小样本 train/val 的 scenario 标签覆盖只保留诊断，不再作为 PROVE 未公开的 corpus rejection gate。

## [0.1.0] - 2026-07-11

### Added

- 为 dependency graph cache 增加完整性元数据：`expected_pair_count`、`classified_pair_count` 和 `classification_complete`。
- 为冷缓存构建增加跨进程文件锁、锁后复查和原子发布；多 shard 生成前默认单进程预热。
- 增加 round capability completion gate：当前 round 必须完成绑定目标，或以 `ask_clarification` / `report_error` 合法终止后才能结束。
- 增加 train/val 全局 fingerprint 与 task ID 交叉去重，并从剩余候选中回填目标数量。
- 增加 generation client seed stride 配置，默认 `1,000,000`，并禁止与三轮 recovery seed 区间重叠。
- 增加 Teacher 生成质量回归测试，覆盖完整链 prompt、filesystem 自然语言别名、round progression、合法 dependency predecessor、outcome gate、跨 split 去重和 seed 隔离。

### Changed

- 完整 2–5 步 `dependency chain` 现在用于生成一个 grounded user task；初始 query 必须明确授权 chain 末端的用户可观察结果。
- chain 前置节点作为同一任务的内部工作流执行，不再机械拆成“每个 user turn 一个工具节点”。Continuation 只表示基于 refreshed live state 的后续用户交互。
- 自然语言与 Unix 命令名的词法 capability 检查降为诊断；live grounding、真实执行和 fresh replay 是权威判据。
- missing-function 在完整 query 生成后隐藏必要工具；共享 entity type 不再作为能力等价证据。
- shard merge 改为先收集、质量过滤和去重全部候选，再做最终 train/val 截断。

### Fixed

- 修复 filesystem 自然语言 query 因 `ls`、`mkdir`、`find`、`touch` 等命令名未原样出现而被错误拒绝的问题。
- 修复目标写操作失败后，仅因辅助读取成功就推进 continuation 的错误。
- 修复 `verify_account → apply_loan` 被误判为缺失 dependency 的问题。
- 修复 `find → rm` 等 filesystem 合法链被误标为 `missing_dependency` 的问题；`find`、`grep`、`tree`、`du`、`df` 现按读取/发现操作处理。
- 修复 `project_outcome_valid=false` 样本仍可能进入 split 或最终 merge 的问题。
- 修复 train/val 各自截断后才检查交叉重复、导致目标 `500+100` 最终只有 `500+98` 的问题。
- 修复多 generation client seed 与 recovery seed 区间碰撞的问题。

### Removed

- 删除无仓库内调用方的 `src/training/livemcp_advantage.py` 兼容转发层；advantage 纯函数继续由 `src/training/advantage_core.py` 统一导出。
