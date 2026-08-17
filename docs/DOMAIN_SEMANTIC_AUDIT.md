# Domain Semantic Audit

> 本文件只维护逐域准入合同和当前结论。动态数据、进程与测试状态见 `PROJECT_STATUS.md`。

## 准入合同

每个 domain 必须依次通过：

1. **Schema/handler**：公开 schema、required/type/enum、handler 前提和状态转换一致。
2. **Dependency**：`C(n,2)` ledger 完整，cache fingerprint 当前，eligible edge 有 typed relation 证据。
3. **Live chain**：source observation 能提供 target 所需实体/值，状态前提在 fresh session 可满足。
4. **Teacher**：query、调用、observation、state change、terminal 和 continuation 逐条语义一致。
5. **Artifact**：fresh replay、provenance、Jaccard、Parquet round-trip 和 production loader 通过。
6. **Policy**：每个 source row 至少 16 条 rollout，能够区分能力失败、runtime failure、reward aliasing 和饱和。

pytest、cache complete、静态 feasibility 或单条 replay 都不能单独构成准入。系统缺陷必须以原失败条件完成
“复现 -> 修复 -> 同条件验证 -> 相邻反例 -> artifact/rollout”闭环。

## 当前矩阵

正式准入为 `0/10`。十域 schema、handler、当前 dependency cache 和静态 relation audit 已通过；保留的
48 行十域 artifact 已在当前 runtime 下重认证，但单 seed、小样本且没有 `N>=16` Policy rollout，因此不能把
局部 artifact 通过提升为 domain 准入。

| Domain | 主要语义风险 | 当前结论 |
|---|---|---|
| banking | account ownership、自然 selector 到 canonical ID、余额/币种、sensitive provenance | 未准入 |
| calendar | reference date、event identity、recurrence/attendee/reminder transition | 未准入 |
| crm | lead/contact/deal 关联、stage transition、重复实体与删除前提 | 未准入 |
| email | email/thread identity、draft/send/label、append-only 行为 | 未准入 |
| filesystem | cwd/path、permission、copy/move/delete、同名路径和类型状态 | 未准入 |
| food_delivery | restaurant/menu/order identity、逐级 order lifecycle、地址与支付 provenance | 未准入 |
| issue_tracker | issue/sprint/assignee identity、workflow transition、creator 后置条件 no-op | 未准入 |
| payments | invoice/payment/refund/dispute lifecycle、金额和跨实体绑定 | 未准入 |
| shopping | product/order/return lifecycle、recommendation grounding、checkout/return authorization | 未准入 |
| team_chat | channel/message/thread identity、reaction/thread 前提、append-only 行为 | 未准入 |

逐域事实必须来自 domain contract、真实 handler 和 fresh trace，不能复制其他 domain 的正则或状态假设。

## 认证记录

每个 domain 通过后只记录以下当前证据：

- cache、schema、runtime 和 reward fingerprint；
- candidate/accepted/filtered 数及结构化失败归因；
- replay/provenance/Jaccard/Parquet/loader 结果；
- rollout seeds、group 数、reward 分布和失败模式；
- 是否准入及唯一剩余阻塞。

旧 cache 版本、调试轮次、PID、已修复案例和临时 chain 计数只从 Git/运行 artifact 追溯，不写回本文。
