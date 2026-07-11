# 版本更新记录

本文件记录 LiveMCP-GRPO 各版本的功能新增、行为变化和缺陷修复。

格式参考 [Keep a Changelog](https://keepachangelog.com/en/2.0.0/)，版本号遵循 [Semantic Versioning](https://semver.org/)。新变化先写入 `Unreleased`，形成版本时再移动到带日期的版本段。

## [Unreleased]

后续尚未形成版本的功能变化记录在这里。

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
