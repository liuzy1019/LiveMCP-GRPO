# 项目状态

> 最后核实：2026-08-17（Asia/Shanghai）。本文件只维护当前状态和下一验收门。
> 稳定算法合同见 `OVAL-MCP.md`，代码边界见 `PROVE_ARCHITECTURE.md`。

## 当前结论

- 十域 Live MCP、Teacher 五步生成、canonical artifact、Policy rollout、PROVE reward 和 GRPO 入口均已实现。
- 当前没有已知可复现的系统级生成错误；这只针对已审计历史案例，不代表未知输入已经被穷尽证明。
- 最近一轮 55 条 candidate failure 中，52 条是 Teacher/轨迹违规并被正确拒绝，3 条是系统误拒并已在共享
  dependency、trace 和 failure-recovery 逻辑修复。
- 原 2 条 shopping finding 是规则误报；修复后同一批数据重算为 0 finding。
- 当前数据尚未正式发布，逐域准入仍为 `0/10`，不能开始正式 GRPO 或声称论文结果复现。

## 当前数据

仅保留两个不可变目录：

| 目录 | 用途 | 状态 |
|---|---|---|
| `data/runs/overnight_10d_32p16_20260813` | 原始十域 source run，32 train + 16 val | completed，未发布 |
| `data/runs/overnight_10d_32p16_20260813_recertified_20260817` | 当前 runtime/reward 重认证副本 | 48/48 fresh replay 通过，0 reject |

重认证产物的 production readback、deterministic local-quality、domain semantic evaluator 和用户可见
private-ID 检查均为 0 failure/diagnostic。该结果证明这 48 行在当前 runtime 下可执行、可解析，不证明样本量、
难度分布或 Policy reward 已达到训练要求。

`data/dependency_graphs/` 只保留十域当前 10 个 cache。当前没有项目生成进程、tmux/vLLM session、旧日志、
Python bytecode cache 或项目临时目录。

## 当前验证

- Policy 环境：`701 passed / 10 skipped / 1 Ray warning`。
- ARL/native MCP transport：`10 passed`。
- 十域 Stage 1+2：`51 pass / 0 fail / 34 diagnostics`。
- SchemaRegistry：191 tools，schema/handler/contract 缺失为 0。
- 重认证 artifact：train 32/32、val 16/16，production audit diagnostics 为 0。

Stage 1 diagnostics 是 raw classifier provenance 与本地 relation audit 的差异、孤立节点、受控循环和跨域
重名 tool；runtime 只消费 relation-audited eligible graph。测试和静态审计只提供结构证据，不能替代逐行数据
与 rollout 结论。

## 未通过的门

1. 当前只有单 seed、48 行，不能给出逐域 yield 或分布稳定性结论。
2. terminal 没有通用自然语言事实证明器；未被声明式合同覆盖的事实仍需逐条审查。
3. strict external abstention 来源尚未导入，第三桶仍是 internal proxy。
4. 每个 source row 尚无 `N>=16` current-fingerprint Qwen rollout group，reward 方差与 aliasing 未验证。
5. 350-step GRPO 和 benchmark 未执行。
6. worktree 含大量未提交系统改动，当前结果尚不能由一个 Git commit 唯一重建。

## 下一步

1. 对当前 48 行运行每行 `N>=16` 的 Policy rollout，并逐轨迹重算 reward。
2. 区分 Policy 能力、runtime failure、reward aliasing 和数据难度问题。
3. 完成逐域准入与 provenance/manifest 认证后，再决定是否发布 `training_candidate`。
4. 只有发布门和 rollout 门同时通过后才启动正式 GRPO。
