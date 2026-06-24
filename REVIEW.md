# SchemaShift-GRPO 代码审查报告 — 10 Domain 全量交付

> 审查日期：2026-06-24（第七次：10 domain 工具数 100% 对齐 PROVE Table 2）
> 审查范围：10 Domain MCP Server(188 tools) + 10 DomainAdapter + 10 StateSeeder + 全量系统测试
> PROVE 基准：20 server / 343 tools → 本阶段精选实现 10 server / 188 tools（覆盖 5/6 类别 × 6 种状态范式）

## 1. Domain 矩阵 — PROVE 对齐

| # | Domain | 论文工具 | 我们的工具 | 对齐率 | 状态特征 | 安全约束 | Category |
|---|---|---|---|---|---|---|---|
| 1 | banking | 17 | **17** | 100% | Accounts, transfers, provenance | 冻结/余额/身份验证 | Finance |
| 2 | calendar | 17 | **17** | 100% | Events, recurring, attendees, timezone | identity_policy, delete+recreate | Productivity |
| 3 | shopping | 23 | **23** | 100% | Catalog, cart, checkout, reviews, coupons | 库存一致性, 空车订单 | Commerce |
| 4 | email | 17 | **17** | 100% | Inbox, drafts, threads, labels, filters | append-only, thread 一致性 | Productivity |
| 5 | filesystem | 40 | **40** | 100% | Deepest state, permissions, archives | 受保护路径, 权限升级 | Productivity |
| 6 | payments | 10 | **10** | 100% | Invoices, refunds, webhooks, disputes | 重复支付, 退款验证 | Finance |
| 7 | crm | 16 | **16** | 100% | Leads, contacts, deals, tasks, notes | 引用完整性, 身份保留 | Knowledge/CRM |
| 8 | issue_tracker | 20 | **20** | 100% | Workflow, sprints, subtasks, time tracking | 严格状态转换, 分配约束 | Knowledge/CRM |
| 9 | team_chat | 11 | **11** | 100% | Channels, messages, threads, DMs | append-only, channel 存在性 | Social |
| 10 | food_delivery | 17 | **17** | 100% | Lifecycle, tracking, rating, reorder | 阶段约束, 取消窗口 | Lifestyle |
| **合计** | **188** | **188** | **100%** | **6 种状态范式** | **14 类安全约束** | — |

> 剩余 10 个 PROVE domain（trading, marketplace, retail_chain, travel_booking, maps, social_media, iot_devices, vehicle, budget, video_meeting）与已实现的 10 个存在严重的状态范式重叠，对 reward 信号多样性贡献有限，暂不纳入。

## 2. 各 Server 完整工具矩阵

### banking — 17 tools
`list_accounts`, `get_account_info`, `get_balance`, `get_history`, `get_statement`, `transfer`, `wire_transfer`, `deposit`, `withdraw`, `bill_pay`, `schedule_transfer`, `cancel_transfer`, `freeze_account`, `unfreeze_account`, `verify_account`, `get_exchange_rate`, `apply_loan`

### calendar — 17 tools
`list_events`, `search_events`, `get_event`, `create_event`, `update_event`, `delete_event`, `create_recurring`, `add_attendee`, `remove_attendee`, `get_free_busy`, `check_conflicts`, `set_reminder`, `get_working_hours`, `change_timezone`, `respond_to_event`, `export_calendar`, `get_recurring_info`

### shopping — 23 tools
`search_products`, `get_product`, `list_categories`, `compare_products`, `get_recommendations`, `add_to_cart`, `update_cart_quantity`, `remove_from_cart`, `get_cart`, `clear_cart`, `apply_coupon`, `get_coupons`, `checkout`, `get_order`, `list_orders`, `track_order`, `return_order`, `get_return_status`, `add_review`, `get_reviews`, `add_to_wishlist`, `remove_from_wishlist`, `get_wishlist`

### email — 17 tools
`list_inbox`, `search_emails`, `get_email`, `send_email`, `create_draft`, `forward_email`, `reply_email`, `add_label`, `remove_label`, `move_to_thread`, `get_thread`, `archive_email`, `mark_read`, `mark_unread`, `create_filter`, `list_filters`, `get_attachments`

### filesystem — 40 tools
Navigation: `ls`, `cd`, `pwd` · Read: `cat`, `head`, `tail`, `wc`, `stat` · Search: `find`, `grep`, `tree` · Create: `mkdir`, `touch`, `mv`, `cp`, `rm` · Permissions: `chmod`, `chown`, `umask` · Disk: `du`, `df` · Links: `symlink`, `readlink` · Archives: `tar_create`, `tar_extract`, `zip`, `unzip` · Text: `diff`, `sort`, `uniq`, `cut`, `sed`, `awk` · Checksums: `md5sum`, `sha256sum`, `file_info`, `xxd` · Utilities: `truncate`, `split`, `join`

### payments — 10 tools
`create_invoice`, `get_invoice`, `list_invoices`, `pay_invoice`, `refund_invoice`, `cancel_payment`, `dispute_invoice`, `create_webhook`, `list_webhooks`, `delete_webhook`

### crm — 16 tools
`create_lead`, `update_lead`, `convert_lead`, `delete_lead`, `list_leads`, `create_contact`, `update_contact`, `delete_contact`, `create_deal`, `update_deal`, `list_deals`, `get_deal`, `create_task`, `list_tasks`, `complete_task`, `add_note`

### issue_tracker — 20 tools
`create_issue`, `get_issue`, `list_issues`, `update_issue`, `assign_issue`, `transition_issue`, `comment_issue`, `add_label`, `remove_label`, `add_watcher`, `remove_watcher`, `create_sprint`, `list_sprints`, `add_to_sprint`, `remove_from_sprint`, `create_subtask`, `list_subtasks`, `time_track`, `get_time_report`, `set_milestone`

### team_chat — 11 tools
`list_channels`, `create_channel`, `archive_channel`, `get_channel`, `send_message`, `send_dm`, `create_thread`, `get_thread`, `react_message`, `search_messages`, `get_user_status`

### food_delivery — 17 tools
`list_restaurants`, `search_restaurants`, `get_restaurant`, `get_menu`, `filter_by_dietary`, `get_popular_items`, `create_order`, `get_order`, `list_orders`, `update_order_status`, `cancel_order`, `get_estimated_time`, `track_rider`, `rate_order`, `add_tip`, `reorder`, `contact_support`

## 3. 架构对齐度

| 论文要求 | 对齐状态 | 说明 |
|---|---|---|
| subprocess stdio 通信 | ✅ | `SubprocessStdioTransport` |
| session-scoped state isolation | ✅ | 独立 state dict / session |
| OpenAI-compatible function schemas | ✅ | JSON Schema 格式 |
| 同一套环境用于数据合成+RL | ⚠️ | sampler 有 10 个但未接入 teacher state machine |
| 5 组件 reward | ⚠️ | R_cov+R_name+R_eff 落地，R_val/R_arg 权重声明但未完整实现 |
| Distractor injection (40%) | ⚠️ | `_apply_distractors` 追加 3 个，论文 3-8 个 |
| Enum stripping (30%) | ❌ | 未实现 |
| Irrelevance queries (5%) | ❌ | 未实现 |
| Missing-function abstention | ⚠️ | `_apply_missing_function` 简化版 |
| Replay validation | ❌ | 未实现 error rate 检查 |
| Provenance check | ⚠️ | `sensitive_params` 标注但未实现验证 |
| Jaccard dedup (0.70) | ❌ | 未实现 |
| RECOVERY→CONTINUATION state machine | ❌ | STATES 声明但未驱动 teacher |

## 4. 安全约束矩阵

| Domain | 检测的安全违规 | 触发条件 | 层级 |
|---|---|---|---|
| banking | 未授权冻结 | before/after frozen diff → `unauthorized_freeze` | DomainAdapter |
| banking | 身份不匹配 | verify_account owner 比对 → `identity_violation` | DomainAdapter |
| banking | 转账到冻结账户 | server error → execution_failed | Server |
| calendar | delete+recreate 同实体 | 内容比对 → `forbidden_transition_delete_recreate` | DomainAdapter |
| filesystem | 删除受保护路径 | `/protected/` 检测 → `deleting_protected_path` | DomainAdapter |
| filesystem | 权限升级 | chmod 新旧权限比较 → `permission_escalation` | DomainAdapter |
| filesystem | 从 root 转移所有权 | chown 检测 → `cannot_transfer_from_root` | Server |
| payments | 重复支付 | server 返回 "already paid" → `double_payment` | DomainAdapter |
| payments | 退款超出发票金额 | server 层金额比较 | Server |
| crm | 转换已丢失 lead | server 返回 "lost" → `convert_lost_lead` | DomainAdapter |
| crm | 删除被引用 contact | server 层检查 deal 引用 | Server |
| issue_tracker | 无效工作流转换 | 状态图约束 → `invalid_workflow_transition` | DomainAdapter |
| issue_tracker | 关闭未分配 issue | server 层 assignee 检查 | Server |
| team_chat | 发送到不存在 channel | server 层 channel 存在检查 → `send_to_nonexistent_channel` | DomainAdapter |
| food_delivery | 取消 preparing 订单 | 生命周期约束 → `cancel_after_preparing` | DomainAdapter |
| food_delivery | 跳跃生命周期 | 状态图约束 → `lifecycle_stage_skip` | DomainAdapter |

### Identity Policy 覆盖

```
preserve    → calendar, filesystem, crm, issue_tracker, banking
create_new  → shopping, food_delivery
append_only → email, team_chat
verify      → payments
```

## 5. 与 PROVE 论文的整体差距

| 维度 | PROVE | 我们 | 差距 |
|---|---|---|---|
| Domain 数 | 20 | 10 | 类别覆盖已达 5/6，剩余均为范式重叠 |
| 工具总数 | 343 | 188 | 单 domain 工具数 100% 对齐 |
| 训练样本 | 13,517 | 0 | 未跑 teacher 数据合成 |
| 训练模型 | Qwen3-4B | 同上 | ✅ |
| GRPO steps | 350 | 2 (smoke) | 未正式训练 |
| 奖励组件 | R_val+R_cov+R_eff+R_name+R_arg | R_cov+R_name+R_eff 主要 | R_val/R_arg 待落地 |
| 数据合成 | State machine teacher | orchestrator 框架 | 状态机未驱动 teacher |
| Robustness knobs | distract+enum+irrelevance+missing | distract(简化)+missing(简化) | enum/irrelevance 未实现 |

## 6. 文件变更清单

### 新增文件 × 7
- `src/live_mcp/servers/email/server.py` — 17 tools
- `src/live_mcp/servers/filesystem/server.py` — 40 tools
- `src/live_mcp/servers/payments/server.py` — 10 tools
- `src/live_mcp/servers/crm/server.py` — 16 tools
- `src/live_mcp/servers/issue_tracker/server.py` — 20 tools
- `src/live_mcp/servers/team_chat/server.py` — 11 tools
- `src/live_mcp/servers/food_delivery/server.py` — 17 tools

### 重写文件 × 3
- `src/live_mcp/servers/banking/server.py` — 6→17 tools
- `src/live_mcp/servers/calendar/server.py` — 4→17 tools
- `src/live_mcp/servers/shopping/server.py` — 5→23 tools

### 新增配置 × 7
- `configs/live_mcp/{email,filesystem,payments,crm,issue_tracker,team_chat,food_delivery}.yaml`

### 修改文件 × 4
- `src/live_mcp/state_seeder.py` — 10 域完整初始状态
- `src/live_mcp/sampler.py` — 10 域采样器
- `src/live_mcp/orchestrator.py` — 10 域注册
- `src/oval_mcp/envs/domain_adapter.py` — 10 域 + `get_adapter` 扩展

### 测试
- `tests/test_live_mcp_10_domains.py` — 26 项集成测试

## 7. 下一步

```text
1. ⬜ R_val/R_arg 在 TaskReward.compute() 中落地
2. ⬜ Enum stripping + Irrelevance queries 扰动注入
3. ⬜ State machine teacher 驱动 orchestrator
4. ⬜ Replay validation + provenance check
5. ⬜ 10-Domain GPU smoke test (ROLLOUT_N=9)
6. ⬜ Phase 2 消融实验 (M4/M4+F/M4+P/M4+F+P)
```

**10 Domain × 188 tools 全量交付完成。**