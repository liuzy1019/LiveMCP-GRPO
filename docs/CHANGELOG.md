# 版本更新记录

本文件记录 LiveMCP-GRPO 各版本的功能新增、行为变化和缺陷修复。

格式参考 [Keep a Changelog](https://keepachangelog.com/en/2.0.0/)，版本号遵循 [Semantic Versioning](https://semver.org/)。新变化先写入 `Unreleased`，形成版本时再移动到带日期的版本段。

## [Unreleased]

### Changed

- 正式生成默认 recovery 上限提高到 6 轮，launcher 为每个 shard 写 checkpoint；单 shard 临近配额时可续跑，不再因默认三轮耗尽丢弃数小时候选池。
- 明确 PROVE continuation 的 2--3 turns 是 conversation rounds，而 rollout `budget` 是本项目 action-turn 工程合同；逐行预算至少覆盖全部 reference tool calls 和每轮 terminal，adaptive efficiency budget 仍只用于 reward。
- Teacher recovery prompt 明确 alternative tool 必须保持用户请求的同一结果；当前能力无法完成时 graceful give-up，不把不同业务 mutation 当作替代操作。该提示不新增论文外 corpus hard gate。

- Teacher action prompt 不再接收 dependency graph hints、chain progress 或 must-execute chain；missing-function 允许可见前置调用，并删除 zero-tool 与首轮 action-type hard blocking。
- PROVE chain 恢复为 query-generation seed：Teacher 不再接收剩余 chain 作为隐藏的 must-execute Oracle 清单，而只根据 query、tool schemas 与真实 execution history 决策。
- dependency pairwise classifier 收紧为论文定义：可选顺序、主题相关和“有帮助但非必需”的 read step 均分类为 `none`；当前 semantics v9 十域缓存已完整重建。
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
- dependency graph 改为逐个有序 `n(n-1)` pair 独立分类；移除 LLM 分类后的 domain 黑名单、entity heuristic 与强制破环，旧 `nC2` cache 自动失效。
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
- food schema 变更只使 food domain cache 失效；当前有效 cache 由 schema + semantics v9 统一索引。

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
