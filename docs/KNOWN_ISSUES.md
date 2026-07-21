# 已知问题

本文档只记录当前 checkout 尚未关闭的问题；已修复行为进入
`docs/CHANGELOG.md`，算法与数据合同以 `docs/OVAL-MCP.md` 和
`data/README.md` 为准。

- 最后更新：2026-07-21

## KI-008：ARL 的 vLLM/CUDA 周边依赖 metadata 漂移

- 状态：`Validating`
- 影响范围：Policy vLLM rollout / Full Training Run

ARL 已完成 `./verl` editable 安装，并通过 `verl.trainer.main_ppo`、项目 runtime
validator 和 `livemcp_grpo` estimator 注册 smoke。4×A10 Teacher vLLM 已在当前版本持续运行
1,000+200 候选生成；该事实不能替代 Policy vLLM rollout 与完整训练 smoke。

`pip check` 仍报告 vLLM 0.19.1 周边 metadata 冲突，包括
`compressed-tensors`、`depyf`、`flashinfer-python/cubin`、`llguidance`、
`nvidia-cudnn-frontend`，以及 xformers 声明的 Torch 版本差异。在真实 Policy
rollout 通过前，不为了消除 metadata 告警批量改动 GPU 依赖树。

## KI-009：正式 GT 等待跨 run 联合验收与发布

- 状态：`Validating`
- 影响范围：正式训练数据入口与语义置信度

`0717_capacity_200_50` 的逐行审计定位并修复了两项 MCP 环境缺陷：disputed invoice 可被
`pay_invoice` 覆盖，以及 calendar attendee/email 接口接受 display name。修复改变了 calendar
与 payments 的 schema/transition fingerprint；该轮 200+50 Parquet 会被 production parser
按环境身份不匹配拒绝，不能用于 rollout 或训练。

修复后的 `0717_mcpfix_1000_200` 已从当前环境重新生成并保留 920 train + 200 val；生产 Parquet
审计无 diagnostics，十域 tool-semantics 审计覆盖 190 tools / 103 mutating tools 且无失败。
该 run 未达到最初 1,000 train 精确目标，所以没有发布默认训练 symlink。第二轮保存候选完成
定向补采后，`0720_mcpfix_round2_1000_200_topup` 已形成 1,000 train + 200 val；生产 Parquet
逐行审计无 diagnostics。最终 1,200 条全局 Jaccard-unique pool 中，calendar/CRM 合计 7 条
不可实现的冻结权重精确配额由其他域合格余量吸收，各域最低覆盖及全部质量门槛保持不变。
独立 seed 的 `0721_mcpfix_round3_1000_200` 正在生成；完成后仍需对两轮执行统一 global merge、
跨 run Jaccard、隔离和逐行语义审计，不能把两个已选 split 直接拼接后训练。

重新生成后仍需持续报告而非隐去的质量信号：

- provenance 对 amount、currency、account/invoice ID 等业务字段的覆盖是本地扩展，
  必须与论文公开机制分开披露；没有论文证据前不擅自放宽现有 gate。
- Teacher action 自动纠正、follow-up 实体/意图连续性与无工具 query 多样性需要继续作为
  非 hard-gate 质量指标。

不得手工补旧字段或改 fingerprint；正式入口只发布联合验收后的当前合同数据。

## KI-010：launcher 成功退出路径需要下轮生成复核

- 状态：`Validating`
- 影响范围：生成作业退出码与自动清理

当前 shell 语法与 285 项全量单元验证通过；reward fingerprint/fail-fast 修复已由保存候选的
真实 merge 验证，没有再把环境身份失配转换成 top-up。手动恢复 merge 已通过 integrity check，
但没有走 launcher 的 symlink publish/`GEN_SUCCESS` 成功尾段；正在运行的
`0721_mcpfix_round3_1000_200` 将继续核对完整 launcher 退出路径。在最终成功或失败证据产生前
不提前关闭该项。

## KI-011：top-up 成本修复等待真实缺口复核

- 状态：`Validating`
- 影响范围：Teacher 灰度与小规模多域生成效率

`0716_semantic_fix_adv_smoke_6_0` 的三域 6-row smoke 中，初始单样本 shard 没有覆盖
email 与 issue_tracker；global merge 正确 fail-closed，但每个仅缺 2 条的 domain 随后按
top-up 下限启动 4 个各 6 条的 shard，即每域生成 24 个候选。全局 quota 已有足够候选后，
运行中的 shard 不会提前停止，造成多余 Teacher 请求。

当前实现已改为先按全局观测的 Jaccard retention 估计每域总 top-up count，再把该总数切分到
可用 client slots；不再把同一固定下限乘以 shard 数。部分成功 shard 会保留并进入下一次 global
merge，相关回归测试已通过。`0720_mcpfix_round2_1000_200` 的三轮真实 deficit top-up 已验证
成功 shard 保留、按观测 retention 估算以及全局重算流程；最终精确域配额的少量缺口在总量和
最低覆盖均满足时按冻结 capacity weights 重分配，不再触发无意义补产。该调整不改变 fresh
replay、provenance、Jaccard 与最低域覆盖，也不得通过减少质量门禁换取速度。

## 执行门禁

- 不得仅根据注释或测试数量宣称不存在所有潜在 bug。
- 涉及 MCP 行为时必须核对 Teacher-visible input、handler、状态变化、fresh replay 与下游消费方。
- PROVE corpus hard gates 与本地工程/质量指标分开记录。
- Policy rollout 和完整训练 smoke 通过前，不关闭 KI-008。
