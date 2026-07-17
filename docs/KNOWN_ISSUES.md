# 已知问题

本文档只记录当前 checkout 尚未关闭的问题；已修复行为进入
`docs/CHANGELOG.md`，算法与数据合同以 `docs/OVAL-MCP.md` 和
`data/README.md` 为准。

- 最后更新：2026-07-17

## KI-008：ARL 的 vLLM/CUDA 周边依赖 metadata 漂移

- 状态：`Validating`
- 影响范围：Policy vLLM rollout / Full Training Run

ARL 已完成 `./verl` editable 安装，并通过 `verl.trainer.main_ppo`、项目 runtime
validator 和 `livemcp_grpo` estimator 注册 smoke。Teacher vLLM 已完成 500+100
数据生成；该事实不能替代 Policy rollout 与完整训练 smoke。

`pip check` 仍报告 vLLM 0.19.1 周边 metadata 冲突，包括
`compressed-tensors`、`depyf`、`flashinfer-python/cubin`、`llguidance`、
`nvidia-cudnn-frontend`，以及 xformers 声明的 Torch 版本差异。在真实 Policy
rollout 通过前，不为了消除 metadata 告警批量改动 GPU 依赖树。

## KI-009：MCP 环境修复后等待重新生成当前合同数据

- 状态：`Open`
- 影响范围：Teacher 数据分布与语义置信度

`0717_capacity_200_50` 的逐行审计定位并修复了两项 MCP 环境缺陷：disputed invoice 可被
`pay_invoice` 覆盖，以及 calendar attendee/email 接口接受 display name。修复改变了 calendar
与 payments 的 schema/transition fingerprint；该轮 200+50 Parquet 现在会被 production parser
按环境身份不匹配拒绝，不能用于 rollout 或训练。下一步必须从当前环境重新生成，并重新执行
production parser、canonical replay、环境 metadata 和逐行自然语言/工具逻辑审计。受影响域的
strict dependency cache 也必须按新 schema key 正常重建，不能手工改旧 cache 的 fingerprint。

重新生成后仍需持续报告而非隐去的质量信号：

- provenance 对 amount、currency、account/invoice ID 等业务字段的覆盖是本地扩展，
  必须与论文公开机制分开披露；没有论文证据前不擅自放宽现有 gate。
- Teacher action 自动纠正、follow-up 实体/意图连续性与无工具 query 多样性需要继续作为
  非 hard-gate 质量指标。

不得手工补旧字段或改 fingerprint；必须从当前生成链重建。

## KI-010：launcher 成功退出路径需要下轮生成复核

- 状态：`Validating`
- 影响范围：生成作业退出码与自动清理

历史日志已清理，当前 checkout 没有可复核的原始错误产物，因此不能把旧报错归因到当前源码。
当前 shell 语法与静态/单元验证通过；下轮小规模真实 launcher
运行需显式核对 merge 返回值、generation PID 统计、integrity check、symlink publish、
`GEN_SUCCESS` 与最终退出码；未复现前不写入猜测性修复。

## KI-011：小缺口 top-up 会显著过量生成

- 状态：`Open`
- 影响范围：Teacher 灰度与小规模多域生成效率

`0716_semantic_fix_adv_smoke_6_0` 的三域 6-row smoke 中，初始单样本 shard 没有覆盖
email 与 issue_tracker；global merge 正确 fail-closed，但每个仅缺 2 条的 domain 随后按
top-up 下限启动 4 个各 6 条的 shard，即每域生成 24 个候选。全局 quota 已有足够候选后，
运行中的 shard 不会提前停止，造成多余 Teacher 请求。该问题不放宽或改变 fresh replay、
provenance、Jaccard 与逐域 quota，只影响调度成本。修复时应基于缺口和已观测保留率设置
小规模上限，并保持 top-up shard 的部分结果可合并；不得通过减少质量门禁换取速度。

## 执行门禁

- 不得仅根据注释或测试数量宣称不存在所有潜在 bug。
- 涉及 MCP 行为时必须核对 Teacher-visible input、handler、状态变化、fresh replay 与下游消费方。
- PROVE corpus hard gates 与本地工程/质量指标分开记录。
- Policy rollout 和完整训练 smoke 通过前，不关闭 KI-008。
