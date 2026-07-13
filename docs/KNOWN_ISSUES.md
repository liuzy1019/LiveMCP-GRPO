# 已知问题

本文只记录当前 checkout 尚未关闭的问题。已修复或废弃方案移入 `CHANGELOG.md`，不在这里保留历史状态。

- 最后更新：2026-07-12
- `Fixing`：实现尚未完成。
- `Validating`：实现和静态回归完成，等待当前 checkout 的端到端验收。

## 审核与执行门禁

- 不得仅依据注释、docstring、设计文档或仿真脚本宣称修复完成。
- 涉及 MCP 行为时必须检查真实 handler、调用方、状态输入和输出消费方。
- 修复和全量静态回归完成前，不得启动 dependency cache 或 Teacher 数据生成。
- 只有完成 Teacher → handler → fresh replay → Parquet → merge → reward readback，才能关闭端到端问题。

## KI-001：PROVE Teacher 链修复待验收

- 状态：`Validating`
- 影响范围：Teacher action prompt、missing-function trajectory

已完成：

- graph hints 和 chain progress 已从 Teacher action 接口及 prompt 删除；
- missing-function 可执行仍可见的前置工具，不再使用 zero-tool contract；
- hidden tool 仍必须不可见、不可执行且不得进入 oracle，最终以 clarification 或 abstention 终止；
- 删除首轮 action-type blocking，仅保留格式、candidate membership 和最终轨迹合同；
- minimal clarification 的首轮与最终合同已统一；
- 删除相关死参数、词法 helper 和旧 continuation helper。

Dependency cache 路径已改为 schema + semantics v9，删除 handler hash、handler denylist 和失效的 offline rebuild 入口。十域 cache 已从空目录完成冷构建：10/10 文件的 `classification_complete=true`，pair 数均为 `n * (n - 1)`，构建锁已释放。当前 checkout 的十域各 1 条真实生成探针通过，仍需用正式批量数据验收 Teacher 语义质量。

## KI-002：最新十域数据语义质量未通过

- 状态：`Validating`
- 当前有效证据：`data/runs/prove_v14_provealigned_gray5_2/{train,val}.parquet` 与 `logs/prove_v14_provealigned_gray5_2.log`
- 运行合同：最新 5 train + 2 val 灰度逐条审查、Parquet/reward 读回和 65 条静态回归通过；等待正式 500+100 多 shard 验收。

已确认问题：

- shopping：query 未要求推荐和比较，但 oracle 执行了相关调用；
- filesystem：zip 目标混入 tar/extract/unzip 调用；
- issue_tracker：对非 required 字段过度澄清；
- email：消歧前写入并出现重复回复；
- team_chat：未落实 query 的时间范围。

后续十域各 1 条真实探针 `logs/validate_pipeline_train_20260712_140039.parquet` 已逐轮展开 `conversation_queries` 与 `round_contracts` 审查。后续 oracle 调用均有对应用户轮次，不属于首轮请求外的无授权调用。发现并修复两个新问题：

- query synthesis 对状态变更工具的必填参数约束不足，CRM 曾在用户未指定 deal amount 时自行填入 `1`；现要求必填状态参数必须来自用户授权或 live state。同 seed 定向重放已生成 `$5k` → `amount=5000` 的一致轨迹；
- generic state delta 与 domain criteria 会重复描述同一 postcondition；现按 type/server/path/value 精确去重，并避免 list 字段在 scalar/list 分支重复生成。

历史 20+10 审查曾确认：`issue_tracker_220715_75307` 的 reference 包含 10 次 tool call 和 3 个 conversation-round terminal，旧 Parquet `budget=10` 无法在 action loop 中复现至少 13 个必要动作。旧 artifact 已在修复验证后清理；现已将论文 2--3 conversation rounds 与工程 action-turn budget 分离，逐行 budget 至少为 `ground-truth tool calls + conversation rounds`，rollout 不再用较小的全局缺省值截断逐行合同。

Teacher prompt 已按论文 recovery 语义明确：仅在当前用户目标实际完成后 `final_answer`；alternative tool 必须保持同一用户结果；无等价能力时停止状态修改并 graceful `report_error`。未新增论文外 semantic judge、exact-chain、词法 capability 或通用 mutation hard gate。当前全量静态回归 56 pass；正式十域批量数据需在该修复后重跑，因此本项继续保持 `Fixing`。

后续失败灰度进一步暴露本地过度过滤：8 条 replay 存活轨迹因 `report_error` 前没有 execution failure 被删除，其中 payments 占 5 条，最终仅 2/3 唯一配额并在 split 前失败。失败 artifact 已清理；PROVE recovery 明确允许 graceful give-up，未要求先制造失败调用，该 hard gate 已移除，保留论文公开 replay/provenance/Jaccard 门禁。

Replay 只能证明 schema/execution 可复现，不能单独证明 query 语义蕴含全部调用。修复后必须重新进行十域逐条人工语义审查，不能复用本轮结论。

2026-07-12 最新 40+10 数据审查又定位并修复两处 continuation 事实链错误：后续轮 query 使用刷新 live state，但 action planner 仍使用首轮 chain 子集；以及新 user round 因全局 execution history 非空而跳过首动作合同，允许空 `final_answer` 吞掉后续目标。当前实现让 query generator 与 action planner 共享同轮 live context，并按每个 user round 重新进入 action 状态；clarification round 可直接回答，follow-up round 仍须完成新 outcome，空 terminal 会重试。未增加 semantic judge 或 corpus hard gate。原失败 seed `130772` 重放后真实执行 delivered-order cancel 并失败，fresh replay error rate 为 1/3（33%），由 PROVE 30% gate 正常过滤，未写入 parquet。

2026-07-13 对 898 条旧 checkpoint 候选做全局 Jaccard 后审查，发现一条 user round 因 Teacher JSON 重试耗尽而完全为空，却被导出器默认成 `final_answer` contract；另有 response-blind continuation 造成操作完成后才询问操作前消歧信息，以及同轮重复成功读取膨胀 oracle。空 round fail-close 与 continuation 读取上一轮公开 assistant response 已保留。曾加入的“同状态重复成功调用抑制”经代码与论文复核后确认偏离 PROVE：它会跳过真实 MCP 执行并改变轨迹，因此已回退；重复调用只作为人工质量诊断，不增加轨迹内 hard filter。受该偏移影响的 v12 正式任务已停止，不能作为最终验收数据。

## KI-003：多 shard 小样本生成过量提交

- 状态：`Validating`
- 影响范围：`scripts/generate_data.sh` 多 client 模式

已实现单层 candidate budget 和 bounded/incremental scheduling，domain 达到 quota 后停止提交新 future。仍缺当前 checkout 的 40 train + 10 val 验收。

验收标准：

- submitted futures 不超过全局 candidate budget；
- client 数量增加不重复应用固定 oversample floor；
- domain 达到 quota 后不再产生该 domain 的新 Teacher 请求；
- 最终精确生成 40+10，train/val 无 fingerprint 或 task ID 重叠。

## KI-004：shard-local 与 global split 合同待验收

- 状态：`Validating`
- 影响范围：多 client shard 导出与全局 merge

已将 shard-local 检查与 global coverage 检查拆分，并将 `chain_seed/source_chain_seed` 写入导出 provenance。仍缺当前 checkout 的多 shard Parquet 实测。

40+10 实测发现 `--shard-mode` 仍把逐域 quota 传入 `_stratified_task_split`，导致一个 shard 的 `issue_tracker=3/4` 在全局 merge 前提前失败；当时两个 shard 合计候选已足够全局需求。现修正为 shard 只按总 unique 行数 recovery 和无逐域 quota split，逐域 train/val quota 仅由 global merge fail-closed 执行。

2026-07-13 v13 灰度进一步证明 recovery 分支仍有遗漏：shard 第一轮 12/15 unique 后错误生成 filesystem/CRM/issue_tracker 逐域请求。当前实现已统一为 shard 只请求 `all` 的总 unique 缺口；旧 v13 进程已停止，修复后重新灰度。

修复后的 v14 fresh-seed 灰度首轮生成 7/7 unique，并直接完成 5+2 split，未触发逐域 shard recovery。过滤后逐条核实无空 round、伪造 terminal 或轨迹内调用裁剪；email 的两次 `archive_email` 分别由两个后续 user round 授权且参数 ID 不同。仍观察到 Teacher 将 invoice ID 当作 webhook 线索后 graceful `report_error` 的语义弱样本；该样本 replay 为 0 error，PROVE 未公开 semantic judge，因此记录为人工质量诊断，不增加 hard gate。

代码复核后补充修复：单进程 split 和 global shard merge 均按 train/val 分别计算逐域配额；某一 domain 的全局唯一候选不足时 fail-closed，不允许其他 domain 补位。小 split 的 scenario label parity 仅记录诊断，不再作为 PROVE 之外的 rejection gate。

验收标准：

- shard-local 只检查行数、字段合同和 shard 内去重；
- domain coverage、全局 Jaccard 0.70 和 train/val 去重只在 merge 后执行；
- Parquet→reward 读回保留完整 oracle、terminal、round contracts 和 provenance。

## KI-005：live-state feasibility 待十域运行验收

- 状态：`Validating`

静态规则只保留实体存在性、基数、类型和真实 handler 状态前置条件；deterministic graph augmentation、手写 edge 表和 handler cache 改图已删除。

当前核查结果：

- 修复静态前置条件重复承担 live-state 判断的问题，不再要求 cart/order/task 必须由同一 chain 中的 `create/list/get` 建立；
- 十域 437 条二元 cache 边中，静态层仅拒绝 8 条由 handler 状态转移确定不可执行的边；
- `food_delivery` 现按 `LIFECYCLE` 顺序模拟组合状态，覆盖 update→track 和 update→rate；
- `reorder` 已同时建模为“消费旧 order、创建新 order”，避免后续调用绑定旧订单；
- Stage 1/2 验证为 52 pass、0 fail、14 diagnostic warning；当前全量测试 56 pass；十域各 1 条真实生成探针通过并精确覆盖 10 个 domain。

PROVE 原文公式为 `C(n,2)=n(n-1)/2`，不是 OCR 文本中的 `n²`。当前 semantics v10 对每个无序 pair 只分类一次，由 LLM 决定有向 source/target；semantics v9 的有序 `n(n-1)` cache 自动失效。论文未公开 chain traversal 是否允许重复节点，当前仍采用 simple path。raw cache 可能存在 LLM classifier noise，例如 email `create_draft → get_email/archive_email` 与实际 drafts/emails 容器语义不符；按照论文机制不手工改写 cache，此风险由 live execution、replay 和数据审查暴露。

剩余验收：正式批量生成后逐条检查 query、oracle、状态变化和 terminal 语义。图中的孤立点与环仅作为 LLM 分类结果诊断，不作为 corpus rejection gate。

论文公开的 corpus gates 仅包括 fresh replay、sensitive-parameter provenance 和 Jaccard 0.70。Parquet schema、显式 terminal、round contracts、hidden-tool 不泄漏和 dependency-edge 索引属于 FSM/rollout/reward 的必要结构合同。Position-aware Jaccard 与按 corpus 数量反推的 missing-function 比例属于已记录的本地实验选择，不宣称为论文公开实现。
