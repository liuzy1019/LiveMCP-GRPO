# OVAL-MCP：Live MCP 长链路工具调用 GRPO 方案

## 0. 目标

OVAL-MCP 的目标是训练模型在 live MCP 场景下完成长链路工具调用：

```text
给定用户任务、MCP tool schemas、历史工具调用与真实 MCP 返回，
模型学习选择下一步：
  tool_call | final_answer | ask_clarification | report_error
```

训练环境采用 PROVE-style live MCP 配置：

```text
MCP servers
  -> each server runs as an independent subprocess
  -> communication over stdio / MCP transport
  -> exposes OpenAI-compatible tool schemas

MCPManager
  -> start / stop / reset servers
  -> maintain session-scoped state isolation

MCPTool wrapper
  -> integrates with verl rollout
  -> routes model-generated tool_call to the target MCP server
  -> returns the real execution observation / error

Audit + verifier layer
  -> records calls, observations, errors, state checks
  -> normalizes events through DomainAdapter
  -> computes reward, cost, process signal
```

核心算法：

```text
Event-Verified Constrained GRPO for Long-Horizon MCP Tool Use
```

训练目标：

```text
maximize    E_pi[R_task(tau)]
subject to  E_pi[C_safety(tau)] <= epsilon
```

完整可选训练 surrogate：

```text
J(tau) =
  R_task(tau)
  + I_shape lambda_shape F_gamma(tau)
  + I_process lambda_process P_process(tau)
  - lambda_safe C_safety(tau)
```

其中：

- `R_task`：任务完成质量。
- `C_safety`：工具调用轨迹中的不可接受副作用。
- `F_gamma`：基于 verifier progress 的 potential shaping。
- `P_process`：有界过程信号，用于长链路信用分配。
- `lambda_safe`：安全约束的 Lagrange multiplier。
- `I_shape, I_process`：ablation 开关。Phase 1 默认 `I_shape=0/optional`，`I_process=0`。

长链路 GRPO 必须同时解决三件事：

```text
1. signal source：局部质量信号来自哪里；
2. signal path：局部信号如何传到 token / turn 梯度；
3. saturation diagnostics：组内 reward/cost 无方差时如何发现。
```

本方案中：

```text
signal source = event-sourced verifier + outcome assertions + safety event log + progress potential
signal path   = scalarized group advantage; Phase 3 adds length-aware turn/token advantage allocation
diagnostics   = reward/cost group variance + safe/unsafe mixed-group rate + unsafe success rate
```

## 1. 事实依据

本方案只依赖公开论文中可核实的事实。

### 1.1 PROVE

PROVE 论文展示了以下事实：

```text
1. live MCP execution environments 可用于多步工具调用 RL；
2. 环境支持 stateful execution 和 session-scoped isolation；
3. 数据生成可从 live sampled server state 出发；
4. replay validation 可过滤不可执行任务；
5. programmatic multi-component reward 可用于 GRPO 训练；
6. 论文报告 20 个 stateful MCP servers、343 tools、约 13K 训练样本。
```

这支持本方案采用：

```text
MCP server subprocesses over stdio
+ MCPManager lifecycle / reset
+ MCPTool wrapper for verl rollout
+ session-scoped state isolation
+ live-state sampling
+ replay validation
+ GRPO rollout on the same MCP backend
```

来源：`Synthesize and Reward -- Reinforcement Learning for Multi-Step Tool Use in Live Environments`, arXiv:2606.03892。

### 1.1.1 PROVE 环境配置到 OVAL 的映射

OVAL-MCP 不重新定义 live MCP 环境，而是在 PROVE 的环境配置上增加 reward/cost/advantage 逻辑。

```text
PROVE component                         OVAL-MCP usage
---------------------------------------------------------------------------
MCP server subprocess over stdio         actual tool execution backend
MCPManager start/stop/reset              rollout lifecycle and reproducibility
session-scoped state isolation           one session_id per rollout
MCPTool wrapper in verl                  policy action -> MCP tool call
OpenAI-compatible function schemas       action parser / validity reward
live-state sampler                       grounded task construction
auto-discovered dependency graph         coverage / progress predicates
state-machine orchestrator               seed trajectories and replay checks
robustness knobs                         distractor / enum / missing-function data
replay validation                        executable trajectory filtering
multi-component reward                   baseline reward components
```

OVAL-MCP 在此基础上新增：

```text
1. audit wrapper:
   把 model call、MCP observation、error、state check 规范化为 trajectory event log。

2. event-sourced safety cost:
   用完整事件轨迹计算 C_safety，而不是只看 final state 或 final answer。

3. constrained GRPO:
   用 lambda_safe 动态控制 E[C_safety] <= epsilon。

4. potential/process signal:
   用 verifier progress 和 bounded process score 改善长链路信用分配。

5. length-aware advantage allocation:
   防止长回复或多轮工具调用稀释局部信号。
```

### 1.2 COVERT

COVERT 论文展示了以下事实：

```text
1. RL 需要 executable environments，而不仅是离线 SFT 数据；
2. tool-use 数据可以先生成 reliable base trajectories；
3. 再通过 oracle-preserving augmentation 增加环境复杂度；
4. multi-level validation 可提升工具使用数据质量；
5. reward 可以基于 exact verifier 或 judge-assisted verifier。
```

这支持本方案采用：

```text
base task generation
+ deterministic validation
+ optional verifier-preserving augmentation
```

默认实现不使用 LLM judge，避免验证目标不稳定。

来源：`Controllable and Verifiable Tool-Use Data Synthesis for Agentic Reinforcement Learning`, arXiv:2604.09813。

### 1.3 Constrained GRPO

Constrained GRPO 论文指出：

```text
1. GRPO 可扩展到显式行为约束；
2. 约束可用 indicator cost functions 表达；
3. naive multi-component advantage 会破坏约束项的相对权重；
4. 正确方式是先 scalarize reward/cost，再进行 group-relative advantage。
```

这支持本方案采用：

```text
J_i = R_task + I_shape lambda_shape F_gamma + I_process lambda_process P_process - lambda_safe C_safety
A_i = normalize_group(J_i)
```

而不是分别 normalize `R_task`、`F`、`C_safety` 后再相加。

来源：`Constrained Group Relative Policy Optimization`, arXiv:2602.05863。

### 1.4 Potential-Based Shaping

Potential-based shaping 的理论结论是：

```text
F(s, s') = gamma Phi(s') - Phi(s)
```

这类 shaping 在标准 MDP 条件下不会改变最优策略集合，只改变学习过程中的信用分配和收敛行为。

这支持本方案不用随意手写过程 reward，而是定义 verifier progress potential：

```text
Phi(m_t) = completed_verifier_steps / total_verifier_steps
F_t = gamma Phi(m_{t+1}) - Phi(m_t)
```

来源：`Potential-Based Shaping and Q-Value Initialization are Equivalent`, arXiv:1106.5267；该文基于 Ng, Harada, Russell 1999 的 potential-based shaping 结论。

### 1.5 Agentic-GRPO-LongHorizon

`agentic-grpo-longhorizon` 项目对长链路工具调用 GRPO 的经验结论可以作为 reward 设计参照。

该项目的可借鉴逻辑是：

```text
1. 二元 outcome reward 容易导致 group reward saturation；
2. 只加过程奖励不够，过程信号还需要有效传播到长回复 token；
3. length-aware advantage allocation 可以减少长链路信号稀释；
4. 每个 reward 改动必须通过 ablation 验证，而不是只看训练 reward。
```

因此 OVAL-MCP 不直接复用 PRM-Lite 规则，而是采用同构分解：

```text
PRM-Lite-style signal source
  -> event verifier process score / progress potential / safety event cost

LATA-style signal path
  -> turn/token length-aware advantage allocation
```

区别是：`agentic-grpo-longhorizon` 的任务核心是 tau-bench 多轮工具对话质量；OVAL-MCP 的任务核心是 PROVE-style stateful MCP live execution 的可验证性、效率和安全约束。

### 1.6 PROVE-style Domain Coverage

OVAL-MCP 的目标场景对齐 PROVE 的 live MCP 设定，而不是单个 calendar domain。

PROVE 的关键环境事实：

```text
1. 20 个 stateful MCP servers；
2. 343 个 user-visible tools；
3. 每个 server 10-40 个工具；
4. session-scoped state isolation；
5. 同一组 live environments 同时用于数据合成和 RL training；
6. 数据包含 multi-turn MCP conversations、missing-function clarification、abstention/no-tool。
```

因此 OVAL-MCP 的泛化对象是 domain adapter，而不是某个具体工具集合。

每个 domain adapter 必须提供统一 verifier 接口：

```text
DomainAdapter = {
  event_normalizer,
  outcome_predicates,
  safety_predicates,
  progress_predicates,
  protected_resources,
  budget_policy
}
```

`tool_schemas`、`state_sampler`、`dependency_graph` 属于环境/数据生成层的 `DomainRuntimeSpec`，不是训练算法直接依赖的 verifier 接口。OVAL-MCP 算法只消费 `DomainAdapter` 的标准化 verifier 输出，不直接写死 calendar、shopping、filesystem 或 banking 的业务规则。

## 2. 核心问题

长链路 MCP 工具调用和普通 function calling 的差异是：

```text
工具调用不仅返回 observation，还可能改变环境状态。
```

一个任务可能最终结果正确，但过程不安全。

例子：

```text
用户：把 Alice 的会议从 3 点改到 4 点。
```

安全轨迹：

```text
search_events
get_event_detail
update_event(event_id=原会议ID, start_time=4点)
```

不安全轨迹：

```text
delete_event(原会议ID)
create_event(同标题、同参会人、4点)
```

如果只看最终状态或最终回答，两条轨迹可能都被判断为成功。

数学上，存在：

```text
observable_final_state(tau_safe) ~= observable_final_state(tau_unsafe)
outcome(tau_safe) = outcome(tau_unsafe) = 1
C_safety(tau_safe) = 0
C_safety(tau_unsafe) = 1
```

如果 reward 只依赖最终 outcome：

```text
R = f(outcome)
```

则无法区分这两条轨迹。

因此 reward 必须定义在完整执行轨迹上：

```text
R, C = f(tau)
```

其中 `tau` 包含每一步工具执行事件。

## 3. 数学建模

定义 live MCP execution problem：

```text
E = (S, T, A, P, O, V)
```

含义：

- `S`：环境状态空间。
- `T`：工具 schema 集合。
- `A`：动作空间。
- `P`：工具调用诱导的状态转移。
- `O`：工具 observation。
- `V`：verifier，包括 outcome、safety、progress。

动作空间：

```text
A = {
  tool_call(name, args),
  final_answer(text),
  ask_clarification(question),
  report_error(reason)
}
```

终止规则：

```text
final_answer / ask_clarification / report_error are terminal actions.
tool_call is non-terminal unless max_turns / budget is reached.
```

如果终止动作不属于任务的 `allowed_terminal_actions`，则 terminal predicate 失败，并进入 `PEN_missing_required_response` 或相应任务失败项。

第 `t` 步历史：

```text
h_t = (user_query, tool_schemas, a_1, o_1, ..., a_{t-1}, o_{t-1})
```

策略：

```text
a_t ~ pi_theta(a | h_t)
```

工具调用：

```text
o_t, s_t = call_tool(a_t, s_{t-1})
```

轨迹：

```text
tau = (s_0, e_1, e_2, ..., e_T, s_T)
```

若某个 MCP server 不暴露完整 inspectable state，则 `s_t` 是 audit layer 可获得的检查视图：

```text
s_t = inspectable_state_t
      or replay_check_state_t
      or predicate_observation_state_t
```

也就是说，所有进入 reward/cost 的 predicate 必须可由 MCP observation、replay validation、server-provided state/check tools 中至少一种证据验证。

每个事件：

```text
e_t = {
  h_t,
  a_t,
  o_t,
  s_{t-1},
  s_t,
  d_t,
  z_t
}
```

其中：

```text
d_t = diff_state(s_{t-1}, s_t)
z_t = audited_tool_event produced by rollout audit layer
```

`z_t` 不能只从最终 state diff 反推，也不能假设所有第三方 MCP server 原生提供标准 event log。OVAL-MCP 在 rollout 层增加 audit wrapper：

```text
audit_wrapper:
  1. 记录每个 model action：tool_call / final_answer / ask_clarification / report_error
  2. 若 action 是 tool_call，记录 MCP observation / error
  3. 若 action 是 terminal action，记录 terminal text / question / reason，且不产生 state transition
  4. 在 server 支持 get_state 时记录 pre_state / post_state / diff
  5. 调用 domain adapter 把 action、observation、diff 规范化为 event
  6. 追加到 session-scoped trajectory event log
```

严谨原因是：

```text
delete(target) -> create(similar_target)
```

可能让最终 state 看起来接近一次 update，但中间已经发生 unsafe side effect。因此 safety verifier 必须读取 trajectory event log，而不是只读最终状态。

Event schema：

```json
{
  "event_id": "evtlog_000001",
  "session_id": "sess_...",
  "step": 3,
  "action_type": "tool_call",
  "tool_name": "update_event",
  "terminal_action": null,
  "operation": "update",
  "target_type": "domain_resource_type",
  "target_id": "evt_102",
  "before_hash": "sha256:...",
  "after_hash": "sha256:...",
  "changed_fields": ["start_time", "end_time"],
  "created_ids": [],
  "deleted_ids": [],
  "duplicate_of": null,
  "provenance": "audit_wrapper"
}
```

terminal action event 使用相同 schema，但 `action_type` 为 `final_answer` / `ask_clarification` / `report_error`，`tool_name` 为空，`operation = "terminal"`，并携带 terminal text / question / reason 的 hash 或 verifier 可用摘要。

`diff_state` 仍保留，但只用于校验 `event_log` 与状态变化是否一致，不作为 safety verifier 的唯一依据。

## 4. Live MCP Rollout Backend

OVAL-MCP 使用 PROVE-style live MCP runtime。运行时由四层组成：

```text
MCPServer subprocess
  - one server per environment/domain
  - stdio / MCP transport
  - exposes tool schemas and executes calls

MCPManager
  - start / stop server processes
  - reset(seed, session_id)
  - provide schema discovery
  - enforce session-scoped isolation

MCPTool wrapper
  - called by verl rollout worker
  - routes tool_call(name, args, session_id) to MCPManager
  - returns observation or execution error

AuditVerifier
  - wraps MCPTool calls
  - records trajectory events
  - calls DomainAdapter predicates
  - computes R_task, C_safety, F_gamma, P_process
```

Runtime contract：

```text
1. list_tools(env_id) returns current OpenAI-compatible tool schemas.
2. reset(env_id, seed, session_id) creates a reproducible isolated state.
3. call_tool(env_id, session_id, name, args) executes on the live MCP server.
4. observation/error is returned exactly from MCP execution.
5. replay(env_id, seed, trace, replay_session_id) re-executes a trajectory against a fresh reset for validation.
6. get_state/session diff is used when the server exposes inspectable state.
7. if full state is unavailable, DomainAdapter must provide executable predicates from observations, check tools, or replay checks.
8. predicates without executable evidence are not allowed in reward/cost.
```

**get_state 不可用时的 predicate 可行性规则**：

```text
当 MCP server 不支持 get_state 时，state_diff 不可得。
DomainAdapter 必须从以下证据源推断 predicates：

证据源优先级（从强到弱）：
  1. observation-based：MCP call 的返回值直接包含所需信息
     适用 predicates：resolved_required_entity, completed_required_transition
     示例：create_event 返回 {"id": "evt_123", "status": "created"}

  2. check-tool-based：调用 server 提供的查询工具验证
     适用 predicates：verified_postcondition, satisfied_dependency_edge
     示例：调用 get_event(id) 确认字段已更新

  3. replay-based：通过 replay validation 对比两次执行结果
     适用 predicates：所有 predicates（最通用但最慢）
     示例：replay 后对比 observation sequence 是否一致

  4. audit-log-based：从 audit_wrapper 记录的 action/observation 序列推断
     适用 predicates：forbidden_transition（通过 action 模式匹配）
     示例：检测到 delete + create 序列 → duplicate side effect

不可推断的 predicate 处理：
  如果某个 predicate 在当前 DomainAdapter 中无法从上述任何证据源验证：
    - 该 predicate 不得进入 reward/cost 计算
    - 记录 unverifiable_predicate_count 作为诊断指标
    - C_safety 对该 predicate 取保守值 0（不惩罚不确定的违规）
    - R_coverage 对该 predicate 取保守值 0（不奖励不确定的完成）
  这确保 reward signal 的每一分都有可执行证据支撑。
```

### 4.1 MCP Rollout API

```python
class MCPManager:
    def start(self, env_id: str) -> None:
        ...

    def stop(self, env_id: str) -> None:
        ...

    def reset(self, env_id: str, seed: int, session_id: str) -> dict:
        ...

    def list_tools(self, env_id: str) -> list[dict]:
        ...

    def call_tool(self, env_id: str, session_id: str, name: str, args: dict) -> dict:
        ...

    def replay(
        self,
        env_id: str,
        seed: int,
        trace: list[dict],
        replay_session_id: str | None = None,
    ) -> dict:
        ...

    def get_state(self, env_id: str, session_id: str) -> dict | None:
        ...
```

MCPTool wrapper 暴露给 verl rollout worker：

```python
class MCPTool:
    def __call__(self, env_id: str, session_id: str, name: str, args: dict) -> dict:
        return manager.call_tool(env_id, session_id, name, args)
```

Audit wrapper 负责：

```text
pre_state = manager.get_state(env_id, session_id)
if action.type == "tool_call":
  observation_or_error = mcp_tool(env_id, session_id, name, args)
else:
  observation_or_error = None
post_state = manager.get_state(env_id, session_id)
state_diff = diff(pre_state, post_state) if both states exist else None
event = adapter.normalize_event(action, observation_or_error, state_diff)
trajectory.append(event)
```

`replay_session_id` 必须不同于原 rollout 的 `session_id`。Replay validation 只能在 fresh reset 的隔离 session 中执行，不能污染原 rollout state。

### 4.2 Domain Adapter

不同 MCP server 的业务状态不同，但训练算法只消费统一 verifier 接口：

```python
class DomainAdapter:
    def normalize_event(self, action, observation, state_diff) -> dict:
        ...

    def outcome_predicates(self, task) -> list:
        ...

    def safety_predicates(self, task) -> list:
        ...

    def progress_predicates(self, task) -> list:
        ...

    def protected_resources(self, task) -> list:
        ...

    def budget(self, task) -> int:
        ...
```

环境和数据生成层可以另外维护：

```python
class DomainRuntimeSpec:
    tool_schemas: list[dict]
    state_sampler: object
    dependency_graph: object
```

`DomainRuntimeSpec` 用于构造任务；`DomainAdapter` 用于验证轨迹。只有 `DomainAdapter` 输出的 predicates 才进入 reward/cost。算法不直接依赖 calendar、shopping、filesystem、banking 等业务字段。

## 5. 任务数据

每条任务：

```json
{
  "task_id": "domain_task_0001",
  "seed": 1001,
  "initial_state_hash": "sha256:...",
  "user_query": "...",
  "tool_schemas": [],
  "outcome_assertions": [],
  "safety_constraints": [],
  "verifier_automaton": {},
  "identity_policy": "preserve | create_new | append_only | lookup_only | domain_defined",
  "required_tool_calls": [],
  "allowed_terminal_actions": ["final_answer"],
  "budget": 5
}
```

`required_tool_calls` 表示完成任务所需的能力集合、依赖边或等价工具族，不表示唯一 reference trace。若多个工具序列都能满足同一组 predicates，verifier 必须接受这些合法替代路径。

### 5.1 Outcome Assertions

Outcome assertions 由 domain adapter 实例化，统一写成 predicates：

```text
required_resource_resolved == true
required_transition_completed == true
required_output_fields_match == true
task_required_fields_preserved == true
final_response_satisfies_task == true
```

关键规则：

```text
如果任务的 identity_policy = preserve，且 target identity 不保留，则 required_transition / coverage predicate 失败。
```

跨域注意：

```text
1. identity-preserving mutation:
   calendar / crm / issue_tracker 等任务通常要求 preserve。

2. create_new:
   shopping 下单、filesystem 新建文件、crm 新建 lead 等任务允许新 identity。

3. append_only:
   email / team_chat / social_media 等任务通常验证 append event 与 recipient/thread/channel provenance。

4. lookup_only:
   maps / search-like 查询任务不要求状态 identity preservation，但仍要求参数 provenance 与 answer correctness。
```

因此 identity failure 不是全局规则，而是 task-level predicate。

### 5.2 Safety Constraints

Safety constraints 也由 domain adapter 实例化：

```text
not forbidden_transition
not wrong_resource_mutation
not identity_or_provenance_violation
not protected_field_loss
not sensitive_param_provenance_violation
not invalid_dependency_order
```

`task_required_fields_preserved` 和 `protected_field_loss` 必须分开定义：

```text
task_required_fields_preserved:
  任务语义要求保持的字段，进入 R_task / outcome predicates。

protected_field_loss:
  明确不可接受的副作用，进入 C_safety。
```

同一底层字段可以同时影响二者，但实现必须在诊断中分别报告：前者是任务未完成，后者是安全约束违规，不能用一个 predicate 同时充当两种角色。

### 5.3 Abstention / Clarification Tasks

如果 `required_tool_calls = []`，任务不是普通 coverage 问题，而是 no-tool / abstention / clarification 问题：

```text
valid terminal actions:
  final_answer       if answerable without tools
  ask_clarification  if required information is missing
  report_error       if no available MCP tool can satisfy the request
```

这类任务的 reward 不使用 tool coverage；它验证：

```text
1. zero unnecessary tool calls；
2. terminal action belongs to allowed_terminal_actions；
3. final text / clarification / error reason satisfies task predicate。
```

### 5.4 Verifier Automaton

定义任务进度状态：

```text
m_t = {
  resolved_required_entity,
  satisfied_dependency_edge,
  completed_required_transition,
  verified_postcondition,
  produced_required_response
}
```

不同任务使用不同子集。

Automaton 不应编码唯一参考轨迹顺序。它只表达：

```text
1. 必须满足的 predicates；
2. 安全关键的 partial order；
3. forbidden transition；
4. accepting condition。
```

例如通用 partial order：

```text
resolved_required_entity before completed_required_transition
satisfied_dependency_edge before completed_required_transition
completed_required_transition before verified_postcondition
verified_postcondition before produced_required_response
```

不应强制具体工具顺序：

```text
tool_A before tool_B
```

除非该顺序本身是 safety constraint。这样 verifier 奖励的是合法计划空间，而不是脚本 reference trace。

## 6. 数据生成算法

```text
Input:
  MCP servers
  MCPManager
  tool schemas
  seed set

Per MCP environment:
  1. start server as subprocess over stdio
  2. discover tool schemas
  3. build / cache dependency graph over tool pairs
     - explicit: output of A is required input of B
     - implicit: A establishes state required by B
     - none
  4. extract length-2 to length-5 tool chains
  5. run live-state sampler through read-only discovery tools
  6. construct grounded query context from real IDs, names, categories, value ranges

Per conversation:
  1. reset(seed, session_id)
  2. sample a dependency-chain seed
  3. generate user query grounded in sampled live state
  4. run state-machine orchestrator:
     query -> deterministic oracle / teacher processing -> tool execution -> response -> continuation
  5. apply robustness knobs:
     distractor tools, enum stripping, missing-function, irrelevance/no-tool
  5a. apply execution-level perturbations（执行级扰动）:
     perturb_probability = 0.15–0.30 per tool call
     perturbation types:
       - intermittent_api_error:  返回 "Internal Server Error" 或超时，oracle 自动 retry
       - paginated_response:      返回 {"items": [...], "next_cursor": "xxx"}，迫使多轮调用
       - incomplete_intermediate: 搜索结果只返回 snippets 而不显示完整详情，迫使后续 extract 调用
       - partial_batch_failure:   批量操作中部分对象操作失败（如 update 10 条，3 条失败），
                                  迫使模型检查结果并逐个处理失败项
     当扰动发生后，oracle 必须执行恢复行为（retry / 分页 fetch / 补充 extract / 逐个处理），
     确保最终 predicate 集合与未扰动版本一致。扰动不改变任务的成功条件，
     只增加达到成功所需的平均 turn 数。
     扰动类型按 domain 适配：
       - filesystem/terminal: intermittent_api_error, partial_batch_failure
       - search/shopping:    paginated_response, incomplete_intermediate
       - calendar/crm:       intermittent_api_error, partial_batch_failure
       - email/team_chat:    paginated_response, partial_batch_failure
  6. replay completed conversation against fresh reset
  7. keep only executable traces that pass validation
  8. instantiate DomainAdapter predicates:
     outcome_assertions, safety_constraints, progress_predicates, budget_policy
```

Reference / teacher trace 不作为 imitation target，只证明任务可执行，并提供 dependency/order predicates 和 budget reference。由 trace 推导出的 predicates 必须经过 replay validation，并应表示合法计划空间，而不是单条脚本路径。

## 7. Reward、Process Signal 和 Cost

### 7.0 Formal Objective vs Training Surrogate

OVAL-MCP 的真实优化目标是 constrained objective：

```text
maximize    E_pi[R_task(tau)]
subject to  E_pi[C_safety(tau)] <= epsilon
```

训练时使用的 `J_i` 是 GRPO 的 scalarized surrogate。完整形式为：

```text
J_i =
  R_task(tau_i)
  + I_shape lambda_shape F_gamma(tau_i)
  + I_process lambda_process P_process(tau_i)
  - lambda_safe C_safety(tau_i)
```

其中：

```text
F_gamma:
  在标准 MDP + 原始 return 优化中满足 PBRS 条件时，不改变最优策略集合。
  在 GRPO group-normalized surrogate 中只作为有理论来源的 shaping signal，
  不单独宣称 policy invariance，必须通过 ablation 验证。

P_process:
  是 bounded auxiliary process signal，不保证 policy invariance。
  它只能作为学习信号和 ablation component，不能被解释为真实任务目标。

lambda_safe C_safety:
  是 Lagrangian penalty，对应安全约束。
```

Phase 1 默认：

```text
I_process = 0
I_shape = 0, or optional lightweight ablation
J_i = R_task(tau_i) - lambda_safe C_safety(tau_i)
```

因此实验报告必须同时给出：

```text
1. true metrics:
   Task Success Rate, Constraint Violation Rate, Unsafe Success Rate

2. training diagnostics:
   J distribution, F_gamma distribution, P_process distribution, lambda_safe trajectory
```

不能用 `J` 的上升直接证明真实任务目标改进。

### 7.1 Task Reward

若 `required_tool_calls = []`：

```text
R_task(tau) =
  1.0 if no tool calls and terminal predicate passes
  0.0 otherwise
```

**no-tool 任务与通用公式的优先级关系**：

```text
当 required_tool_calls = [] 时，使用上述二元定义，不使用通用公式。

理由：
  通用公式中 R_efficiency = -alpha_eff * max(0, n_model_calls - B) / max(B, 1)
  当 n_required_calls = 0 时，B = 0，R_efficiency = -alpha_eff * n_model_calls
  这会对任何工具调用产生无界惩罚（随调用次数线性增长），
  与二元定义的 "任何工具调用 → 0.0" 语义重叠但量级不一致。

  因此明确规定：
    required_tool_calls = [] 的任务完全使用二元 R_task 定义
    通用公式（R_positive/Z_pos + w_eff * R_efficiency）仅适用于 required_tool_calls != [] 的任务
    两者互斥，不存在 fallback 或混合计算
```

若任务需要工具调用：

```text
R_positive(tau) =
  w_val R_validity(tau)
  + w_cov R_coverage(tau)
  + w_name R_name(tau)
  + w_arg R_arg(tau)

Z_pos = w_val + w_cov + w_name + w_arg

R_task(tau) =
  clip(
    R_positive(tau) / Z_pos
    + w_eff R_efficiency(tau),
    -0.2,
    1.0
  )
```

其中：

```text
R_validity:
  per-call structural validity (schema compliance):
    valid tool name (exists in current tool_schemas)
    required args present with compatible JSON types
    args satisfy schema constraints (enum, format, required fields)
  execution validity (MCP runtime success):
    live MCP execution returns non-error observation

  两层分别计分：
    R_validity = w_struct * R_structural + w_exec * R_execution
    w_struct = 0.6, w_exec = 0.4

  数学理由：
    将结构有效性与执行成功性分离，避免因环境状态原因（如操作不存在的资源）
    导致格式完美的调用被全额惩罚，从而抑制合理的探索行为。
    结构有效性只依赖 schema 合规，与环境状态无关；
    执行有效性依赖 MCP server 实际返回，反映调用的环境适配度。

  recoverable execution errors:
    R_structural = 1 (schema compliant)
    R_execution = 0 (execution failed)
    可 receive process credit only if the next action correctly recovers

R_coverage:
  required workflow / outcome predicates completed in dependency order,
  including valid terminal action and final_response_satisfies_task when required;
  safety predicates are not counted as coverage

  R_coverage = completed_coverage_predicates / total_coverage_predicates

  其中：
    total_coverage_predicates = task 的 verifier automaton 中所有 non-safety predicates 数量
    completed_coverage_predicates = 截至轨迹结束时已满足的 predicates 数量

  R_coverage 是离散比例值（如 3/5 = 0.6），不是连续值。
  这与 Phi 的定义一致（两者基于相同的 predicate 集合），
  但 R_coverage 是终态值，Phi 是过程值。

R_name:
  unique selected tool names overlap required / valid tool set

  R_name = |unique_model_tool_names ∩ required_tool_set| / |required_tool_set|

  计算粒度说明：
    R_name 基于 unique tool name set 的 coverage ratio，而非 per-call match rate。
    重复调用同一工具名 N 次只计为 1 次 name match，防止重复调用膨胀 R_name。
    required_tool_set 来自 task 的 required_tool_calls 中的 unique tool names。

  if n_model_calls = 0 and required_tool_calls != []:
    R_name = 0

R_arg:
  argument values match grounded live-state entities and required values

  "aligned calls" 的定义：
    一个 model call 被称为 "aligned" 当且仅当：
      1. 它的 tool_name 匹配 required_tool_calls 中某个 entry 的 tool_name
      2. 它对应的 coverage predicate 被满足（即该调用实际推进了 workflow）
    对齐机制是基于 coverage predicate 的，不是基于 reference trace 的位置匹配。
    这允许模型用不同于 reference 的顺序完成任务，只要 predicate 被满足即可。

  R_arg = mean_{aligned calls} arg_match_score(call)
  arg_match_score(call) = |matched_arg_values| / |required_arg_values|

  only aligned calls from R_coverage are scored;
  unaligned calls do not receive argument-value credit

R_efficiency:
  adaptive excess-call penalty
```

Recommended weights follow PROVE-style balance:

```text
w_val = 0.5
w_cov = 0.5
w_eff = 0.15
w_name = 0.2
w_arg = 0.1
```

Auxiliary terms are smaller than validity/coverage so they guide learning without dominating task completion.

若 `identity_policy = preserve` 且 target identity 失败：

```text
R_coverage = 0
```

**identity violation 的双维度计数说明**：

```text
当 identity_policy = preserve 且 safety constraints 包含 identity_or_provenance_violation 时，
同一次 identity 丢失事件会同时触发：
  R_coverage = 0 → R_task 被拉低（任务完成维度）
  C_safety = 1  → J 减去 lambda_safe（安全约束维度）

这是有意的设计选择，不是 bug：
  - R_task 和 C_safety 语义不同：前者衡量任务是否完成，后者衡量是否安全
  - 但在 J 的标量空间中，identity violation 天然比其他类型的 task failure 惩罚更重

实验必须记录这种不对称性：
  在 ablation 报告中单独披露 identity_violation_penalty_magnitude：
    = R_task_loss + lambda_safe * C_safety
    = (R_task_without_identity_fail - R_task_with_identity_fail) + lambda_safe
  并与其他 failure mode 的惩罚量级对比，确保不会因过度惩罚导致模型对 identity task 过度保守。
```

范围约束：

```text
R_validity, R_coverage, R_name, R_arg in [0, 1]
R_efficiency <= 0
R_positive / Z_pos in [0, 1]
R_task in [-0.2, 1.0]
```

Efficiency follows PROVE 的 adaptive budget 思路：

```text
B = n_required_calls + ceil(beta_budget * n_required_calls)
R_efficiency = -alpha_eff * max(0, n_model_calls - B) / max(B, 1)
```

`n_required_calls` 来自 dependency-chain / replay-validated reference，不是固定常数。这样复杂任务有更多 slack，避免把必要的信息收集调用当作冗余。

`R_task` 只描述任务完成质量，不直接吞并 safety cost。允许出现：

```text
R_task(tau) > 0 and C_safety(tau) = 1
```

这类轨迹称为 unsafe success，必须在 constrained objective 中由 `C_safety` 控制，而不是在 outcome verifier 中被悄悄混掉。这样才能分别报告：

```text
Task Success Rate
Unsafe Success Rate
Constraint Violation Rate
```

Safety predicates 不进入 `R_coverage`。如果某条轨迹完成了 required workflow 但触发 forbidden event，它应表现为：

```text
R_task high
C_safety = 1
```

这样 constrained GRPO 才能显式学习“成功但不安全”的差异。

### 7.2 Safety Cost

默认使用二值 cost：

```text
C_safety(tau) = 1 if any forbidden event occurs else 0
```

forbidden event 来自 `event_log`：

```text
forbidden transition under domain adapter
wrong resource mutation
identity or provenance violation
protected field loss
sensitive parameter provenance violation
dependency/order violation
duplicate or inconsistent side effect
```

也可以扩展为分级 cost：

```text
protected field loss: 0.3
wrong resource mutation: 0.5
duplicate or inconsistent side effect: 0.7
identity/provenance violation: 1.0
```

第一版先用二值 cost，解释最清楚。

`C_safety` 只由 audited event log 与 safety predicates 决定，不由 final answer 文本决定。

若同一轨迹同时有多个 forbidden event，二值 cost 仍为 1；详细类型进入通用诊断指标：

```text
C_forbidden_transition
C_wrong_resource_mutation
C_identity_violation
C_protected_field_loss
C_sensitive_param_provenance_violation
C_ordering_violation
C_duplicate_or_inconsistent_side_effect
```

### 7.3 PRM-Lite-style Event Process Score

参照 `agentic-grpo-longhorizon` 的 PRM-Lite 思路，OVAL-MCP 也需要局部质量信号，但规则必须来自 verifier 和 event semantics，而不是手写对话风格偏好。

定义每步 process score。这里 penalty predicates 直接用负值表示：

```text
B_t = sum(triggered bonus values)
N_t = sum(triggered penalty values)   # N_t <= 0
p_t_raw = B_t + N_t

if step triggers forbidden event:
  p_t = min(p_t_raw, -abs(N_t_forbidden))
else:
  p_t = p_t_raw

P_process(tau) = clip(sum_t p_t, -p_max, p_max)
```

**forbidden event clamping 语义说明**：

```text
设计意图：forbidden event step 不得获得正 process score，且必须保留其 penalty 效果。

旧规则 p_t = min(p_t_raw, 0) 的问题：
  当 B_t > |N_t| 时，p_t_raw > 0，clamping 到 0 抹掉了 penalty 效果。
  forbidden event step 既不被奖励也不被惩罚，失去区分度。

新规则 p_t = min(p_t_raw, -abs(N_t_forbidden))：
  N_t_forbidden = 该 step 中由 forbidden event 触发的 penalty 值之和
（如 PEN_forbidden_transition_attempt = -0.08）
  这保证：
    1. forbidden event step 的 p_t <= -0.08（至少保留 forbidden penalty 本身）
    2. 若 p_t_raw 已经更负（其他 penalty 叠加），则保留更负的值
    3. 正向 bonus 不能抵消 forbidden event 的 penalty

  边界情况：
    B_t = 0.08, N_t = -0.08 (forbidden only):
      p_t_raw = 0, N_t_forbidden = -0.08
      p_t = min(0, -0.08) = -0.08  ✓ penalty 保留

    B_t = 0.13, N_t = -0.08 (forbidden only):
      p_t_raw = 0.05, N_t_forbidden = -0.08
      p_t = min(0.05, -0.08) = -0.08  ✓ bonus 不抵消 forbidden

    B_t = 0, N_t = -0.13 (forbidden + other penalty):
      p_t_raw = -0.13, N_t_forbidden = -0.08
      p_t = min(-0.13, -0.08) = -0.13  ✓ 保留更严重的惩罚

与 C_safety 的关系：
  C_safety 承担轨迹级安全约束（Lagrangian multiplier 机制）
  p_t 的 forbidden penalty 承担局部 process 信号（step-level 区分度）
  两者在不同维度工作，不构成双重惩罚：
    C_safety 影响 lambda_safe 的 dual ascent
    p_t 影响 LATA 的 turn-level 权重分配

**PEN 到 forbidden event 的映射表**：

当 step 同时触发多个 predicates 时，只有映射到 §7.2 forbidden events 的 PEN 才计入 N_t_forbidden。

```text
PEN predicate                        → 对应 forbidden event（§7.2）
─────────────────────────────────────────────────────────────────────
PEN_forbidden_transition_attempt     → forbidden transition
PEN_wrong_resource_action            → wrong resource mutation
                                       （当且仅当 action 的实际效果触发
                                        wrong_resource_mutation predicate）
PEN_unresolved_entity_action         → 不映射（process 问题，非 safety）
PEN_redundant_no_progress_action     → 不映射（效率问题，非 safety）
PEN_missing_required_response        → 不映射（任务完成问题，非 safety）
PEN_invalid_tool_schema              → 不映射（格式问题，非 safety）

不在映射表中的 PEN 不计入 N_t_forbidden，仅通过 p_t_raw = B_t + N_t 参与 p_t 计算。
```

注意：

```text
PEN_wrong_resource_action 的映射是有条件的：
  "action targets unrelated or ambiguous resource" 本身不一定是 safety 违规。
  只有当 action 的实际效果触发 wrong_resource_mutation 的 safety predicate 时，
  才计入 N_t_forbidden。DomainAdapter 负责判断该条件是否满足。

  举例：
    - 模型调用 update_event(event_id="wrong_event_999") → server 返回 "not found"
      → 不触发 wrong_resource_mutation（没有实际修改错误资源）
      → PEN_wrong_resource_action 不计入 N_t_forbidden

    - 模型调用 delete_file(path="/etc/important_config")
      → 实际删除了受保护文件
      → 触发 wrong_resource_mutation
      → PEN_wrong_resource_action 计入 N_t_forbidden
```
```

通用 bonus predicates：

```text
B_resolve_required_entity       +0.05  required entity/resource is uniquely resolved
B_satisfy_dependency_edge       +0.05  a dependency-ordered predecessor is completed
B_preserve_required_identity    +0.05  operation keeps required identity/provenance
B_complete_required_transition  +0.08  expected state transition is completed (see note on F_gamma overlap)
B_verify_postcondition          +0.04  required postcondition is checked or observed
B_recover_from_tool_error       +0.04  valid correction after non-fatal tool error
```

通用 penalty predicates，取值为负数：

```text
PEN_redundant_no_progress_action  -0.03 repeated action with no new predicate progress
PEN_unresolved_entity_action      -0.05 action requires an unresolved resource/entity
PEN_wrong_resource_action         -0.05 action targets unrelated or ambiguous resource
PEN_forbidden_transition_attempt  -0.08 action attempts a forbidden transition
PEN_missing_required_response     -0.05 task completed but response/abstention is invalid
PEN_invalid_tool_schema           -0.05 unparseable or schema-invalid call
```

**命名约定**：

```text
B_xxx  = bonus predicate（正值）
PEN_xxx = penalty predicate（负值）
P_process = 轨迹级 process score 变量名

避免使用 P_ 前缀同时表示 penalty predicate 和 P_process 变量，
以消除读者歧义。
```

**数值校准推导**：

```text
设计约束：lambda_process * P_process_max = lambda_process * p_max = 0.3 * 0.3 = 0.09
即 process signal 对 J 的最大贡献不超过 R_task 的 ~9%，确保 outcome 主导。

数值设定原则：
  1. 单步 bonus 上限 = 0.08（B_complete_required_transition）
     一个典型 5-step 任务的最大 P_process = 5 * 0.08 = 0.4 > p_max = 0.3
     因此 clip 生效，防止简单任务的 process score 过大

2. 单步 penalty 下限 = -0.08（PEN_forbidden_transition_attempt）
     与最大 bonus 对称，使得 forbidden attempt 能完全抵消一次 progress bonus

  3. 中等 bonus/penalty = ±0.05
     对应 "有意义但非关键" 的事件（resolve entity, dependency edge）
     一个 5-step 任务中 3 个中等 bonus = 0.15，约为 p_max 的一半

  4. 轻微 penalty = -0.03（redundant action）
     比中等 penalty 弱，因为冗余调用不如错误调用严重
     但累积 10 次冗余 = -0.30 = p_max，此时 clip 生效

初始校准方法：
  这些值是基于 "典型 5-step 任务" 的量级分析设定的初始值。
  Phase 2 ablation 必须验证：
    a. M4+P 相比 M4 的 group saturation rate 是否下降
    b. process score 的 std 是否足以在 group 内产生有意义的方差
    c. 若 std(P_process) < 0.01 * std(R_task)，说明数值过小，需放大
  根据 ablation 结果可按比例缩放所有 bonus/penalty 值。
```

Domain examples:

```text
calendar:
  required entity = event
  forbidden transition = delete target and recreate duplicate
  protected fields = attendees, reminders, notes

banking/payments:
  required entity = account / invoice / payment
  forbidden transition = transfer/refund without sensitive-param provenance
  protected fields = account id, amount, currency, authorization source

filesystem:
  required entity = path / inode-like identity
  forbidden transition = delete or overwrite protected unrelated file
  protected fields = permissions, owner, directory identity

email/team_chat:
  required entity = thread / channel / recipient
  forbidden transition = send before required recipient/content verification
  protected fields = recipient, thread id, labels, attachments
```

规则约束：

```text
1. process score 不得奖励 forbidden event；
2. process score 不得超过 outcome reward 的主导地位；
3. process score 必须可由 event log / verifier state 确定；
4. 每条规则必须有对应单测和 ablation；
5. F_gamma 与 P_process 信号重叠处理：
   当 M4+F+P 同时启用时，B_complete_required_transition 与 Phi 增量
   会对同一 progress event 产生双重正信号。这不违反数学约束，
   但实验必须记录 overlap_ratio = (同时触发 F>0 和 p>0 的 step 数) / total_steps，
   并在 ablation 中对比 M4+F、M4+P、M4+F+P 的边际收益。
   若 overlap_ratio > 0.8 且 M4+F+P 相比 M4+F 无显著提升，
   应考虑在 P_process 中排除已被 F_gamma 覆盖的 progress predicates。
```

推荐默认：

```text
p_max = 0.3
lambda_process = 0.3
```

这使 process signal 最大贡献约为 `0.09`，用于打破组内饱和和提供局部梯度，但不覆盖最终任务成功。

若某一步触发 forbidden event：

```text
p_t <= 0
```

即使该步同时推进了 coverage，也不能获得正 process score。coverage 由 `R_task` 表达，安全违规由 `C_safety` 和非正 process signal 表达。

## 8. Potential-Based Progress Shaping

定义 progress potential：

```text
Phi(m_t) = completed_required_states(m_t) / total_required_states
```

**required_states 的精确定义**：

```text
required_states 是 §5.4 verifier automaton 中 progress predicates 的子集：
  resolved_required_entity
  satisfied_dependency_edge
  completed_required_transition
  verified_postcondition
  produced_required_response

total_required_states = 该 task 的 verifier automaton 中上述 predicates 的总数
completed_required_states(m_t) = 截至 step t 已满足的 predicates 数量

不同 task 的 total_required_states 可能不同（简单任务 3 个，复杂任务 8 个），
但 Phi 始终归一化到 [0, 1]，因此跨 task 的 F_gamma 量级一致。

不包含在 required_states 中的：
  - safety predicates（forbidden event 检测）→ 归入 C_safety
  - terminal action predicates → 归入 R_coverage 的 terminal 部分
  - 非必须的 optional predicates

Phi 的单调性保证：
  required_states 只能被 "完成"，不能被 "撤销"。
  如果一个 action 导致已完成的 predicate 失效（如删除了已 resolve 的 entity），
  这应被 event log 检测为 forbidden event，进入 C_safety，
  而不是让 Phi 回退。Phi 是单调非递减的。
```

每步 shaping：

```text
F_t = gamma Phi(m_{t+1}) - Phi(m_t)
```

轨迹 shaping：

```text
F_gamma(tau) = sum_t gamma^t * (gamma Phi(m_{t+1}) - Phi(m_t))
```

**γ=1 时的 telescoping 性质与信用分配层次**：

```text
当 gamma = 1.0 时：
  F_gamma(tau) = sum_t (Phi(m_{t+1}) - Phi(m_t)) = Phi(m_T) - Phi(m_0)

这意味着 F_gamma 在 trajectory-level 只依赖终点和起点的 potential 差，
与中间路径无关。两条最终达到相同 Phi(m_T) 的不同轨迹，F_gamma 完全相同。

这不是 bug，而是 PBRS 的数学必然：trajectory-level shaping 只能区分
不同终点进度的轨迹，不能区分相同终点但不同路径的轨迹。

信用分配的层次划分：
  Phase 1-2（trajectory-level advantage）：
    F_gamma 的作用是让不同 Phi(m_T) 的轨迹在 group 内产生方差，
    从而让 GRPO 能区分 "完成 80% progress" vs "完成 20% progress" 的轨迹。
    这是 inter-trajectory 的信用分配，不是 intra-trajectory 的。

  Phase 3（LATA turn-level allocation）：
    F_u = gamma * Phi(m_{u+1}) - Phi(m_u) 作为 per-turn 局部信号进入 q_u，
    由 LATA 的 signed relevance 实现 intra-trajectory 的 step-level 信用分配。
    此时即使 gamma=1，F_u 在每个 turn 上是不同的（取决于该 turn 是否推进了 progress），
    因此 LATA 能区分同一轨迹内不同 turn 的贡献。

结论：
  - Phase 1-2 的 F_gamma 不声称提供 step-level 信用分配，只提供 trajectory-level progress signal
  - Step-level 信用分配完全由 Phase 3 的 LATA + F_u 承担
  - 若需要在 Phase 2 就获得 step-level 区分（不等到 Phase 3），可设 gamma < 1，
    此时早期 progress 获得更高折扣权重，但这会引入 horizon-dependent bias
```

这样过程信号来自 verifier automaton，而不是人工随意打分。

`Phi` 只能来自 progress predicates，不能包含 safety predicates 或 final task score：

```text
Phi = f(progress_predicates)
Phi excludes C_safety
Phi excludes R_task
```

否则 shaping 会和真实 reward/cost 重复计数，破坏解释性。

解释：

```text
如果一步工具调用推进了任务状态，Phi 增大，F_t 为正。
如果没有推进，F_t 约为 0。
如果导致 verifier 回退或失败，可进入 safety cost 或 task failure。
```

注意：Potential-based shaping 的理论不改变最优策略集合依赖标准 MDP 条件。对于 LLM history-based policy，这里将 `history + verifier_state` 视作扩展状态，作为工程近似。
在实际 GRPO 训练中，`F_gamma` 会先与 `R_task`、`P_process`、`C_safety` scalarize 成 `J`，再通过 group-relative advantage normalize。因此实验结论只能声称它是有理论来源的 progress shaping，不能单独声称整个 GRPO surrogate 保持最优策略不变。

数学条件：

```text
1. shaping 使用与 RL return 一致的 discount gamma；
2. Phi 只能依赖扩展状态，不依赖未来轨迹；
3. terminal / absorbing failure state 的 Phi 必须固定；
4. 若实现中不满足这些条件，只能称为 heuristic shaping，不能声称 policy invariant。
```

默认推荐：

```text
gamma = 1.0 for short finite-horizon MCP rollout tasks
Phi(success_terminal) = 1 when verifier progress predicates are all satisfied
```

**absorbing failure state 的 Phi 定义**：

```text
Phi(absorbing_failure) = Phi(m_{T-1})
```

即 absorbing failure state 继承进入该状态前的最后一步 progress potential。

数学理由：

```text
1. PBRS 理论允许 absorbing state 的 Phi 取任意固定值，不影响最优策略集合。
2. 但若设 Phi(absorbing_failure) = 0，当模型已完成部分 progress（如 Phi(m_{T-1}) = 0.6）
   后触发 forbidden event 进入 absorbing failure，shaping 会产生：
     F_T = gamma * 0 - 0.6 = -0.6
   这与 C_safety = 1 构成双重惩罚，违反本方案 "R_task、C_safety、F_gamma 分开记录、
   不重复计数" 的原则。
3. 设 Phi(absorbing_failure) = Phi(m_{T-1}) 使得 F_T = 0，
   failure 的惩罚完全由 C_safety 承担，shaping 只负责 progress 信号，职责分离清晰。
4. 这不影响 F_gamma 在正常轨迹中的 progress shaping 功能。
```

若后续使用 `gamma < 1`，必须保留上面的 `gamma^t` 折扣项。

推荐 `lambda_shape` 默认值：

```text
lambda_shape = 0.5
```

量级分析：

```text
F_gamma 范围：[0, 1]（Phi 从 0 到 1 的差）
lambda_shape * F_gamma_max = 0.5 * 1.0 = 0.5

对比：
  R_task 范围：[-0.2, 1.0]，典型成功值 ~0.7
  lambda_process * P_max = 0.3 * 0.3 = 0.09
  lambda_safe * C_safety = lambda_safe * 1 = lambda_safe（动态）

lambda_shape = 0.5 使得：
  - 完全成功轨迹（Phi(m_T)=1）的 shaping 贡献为 0.5，
    与 R_task 量级可比但不超过
  - 部分进度轨迹（Phi(m_T)=0.4）的 shaping 贡献为 0.2，
    足以在 group 内产生有意义的方差
  - 相比 lambda_process 的 0.09 上限，shaping 信号更强，
    这合理因为 progress 是比 process style 更核心的信号
```

## 9. Constrained GRPO

每个 task 采样 `G` 条 rollout：

```text
tau_1, ..., tau_G ~ pi_theta(. | q)
```

每条轨迹计算：

```text
R_task(tau_i)
C_safety(tau_i)
F_gamma(tau_i)
P_process(tau_i)
```

Scalarized return：

```text
J_i =
  R_task(tau_i)
  + I_shape lambda_shape F_gamma(tau_i)
  + I_process lambda_process P_process(tau_i)
  - lambda_safe C_safety(tau_i)
```

Group-relative advantage：

```text
A_i = (J_i - mean(J_1...J_G)) / (std(J_1...J_G) + eps)
```

如果组内 `std(J)` 低于阈值：

```text
std(J_1...J_G) < min_group_std
```

该 group 不产生 policy gradient，只记录 saturation diagnostic。用 `eps` 强行放大近零方差会制造噪声梯度。

GRPO objective：

```text
L =
- E [
  min(
    rho_i,t A_i,
    clip(rho_i,t, 1-eps_clip, 1+eps_clip) A_i
  )
]
+ beta_KL KL(pi_theta || pi_ref)
```

安全乘子更新：

```text
hat_C_batch = mean_{q in B, i in 1..G} C_safety(tau_{q,i})
lambda_safe = clip(lambda_safe + alpha_lambda (hat_C_batch - epsilon), 0, lambda_safe_max)
```

若当前 batch 违规率高于 `epsilon`，`lambda_safe` 增大。若违规率低于 `epsilon`，`lambda_safe` 按投影梯度下降变小，但不会小于 0，也不会超过 `lambda_safe_max`。

这比固定安全扣分更合理，因为它直接优化：

```text
E[C_safety] <= epsilon
```

推荐初始化与超参：

```text
lambda_safe_init = 1.0
alpha_lambda = 0.01
epsilon = 0.05
lambda_safe_max = 10.0  (防止极端情况下 lambda 爆炸)
```

数学理由：

```text
1. lambda_safe_init = 1.0：
   使训练初期 safety cost 与 R_task 量级可比（R_task ∈ [-0.2, 1.0]，C_safety ∈ {0,1}）。
   若 init = 0，训练初期模型可能学到大量 unsafe 行为后 lambda 才开始生效。

2. alpha_lambda = 0.01：
   Lagrangian dual ascent 的步长。过大导致 lambda 震荡，过小导致约束响应迟缓。
   0.01 使得在 batch_size=64、violation_rate=0.2 时，
   每步 lambda 变化约 0.01*(0.2-0.05) = 0.0015，约 667 步翻倍，节奏适中。

3. epsilon = 0.05：
   允许 5% 的 batch-level violation rate。
   这是 "几乎总是安全" 的工程标准，可根据业务需求调整。

4. lambda_safe_max = 10.0：
   上界保护。当 lambda 达到上界时，C_safety 的惩罚已是 R_task 满分的 10 倍，
   若仍无法降低 violation rate，说明问题在数据/模型能力而非 lambda 大小。
```

### 9.1 Signal Path：Length-Aware Turn/Token Advantage

仅有 `J_i` 不够。长链路 agent 中，如果把同一个轨迹 advantage 线性均摊到所有 token，局部过程信号会被长回复稀释。

Advantage 只分配给 policy 生成的 tokens：

```text
eligible tokens =
  tool_call tokens
  final_answer tokens
  ask_clarification tokens
  report_error tokens

ineligible tokens =
  MCP observations
  environment errors
  system/tool schemas
  user query
```

MCP observation 是环境返回，不是策略动作，不能进入 policy-gradient loss。

因此 OVAL-MCP 采用 LATA-style 信号通路：

```text
turn u has L_u eligible policy tokens and local event score q_u
trajectory group advantage is A_i
```

先计算局部事件质量：

```text
q_u =
  I_shape lambda_shape F_u
  + I_process lambda_process p_u
  - lambda_safe c_u
```

其中 `q_u > 0` 表示该 turn 包含正向进展，`q_u < 0` 表示该 turn 包含局部错误或安全 cost。

**c_u 的精确定义**：

```text
c_u 是 turn u 的局部 safety cost，由 event log 确定性映射得到。

定义规则：

1. 二值 C_safety 模式（Phase 1-2 推荐）：
   c_u = 1  if turn u 的 action 直接触发了 forbidden event（event log 中有对应记录）
   c_u = 0  otherwise

   一致性约束：
     C_safety(tau) = min(1, sum_u c_u)
   即：轨迹级 C_safety = 1 当且仅当存在至少一个 turn 的 c_u = 1。

2. 分级 cost 模式（可选扩展）：
   c_u = severity(forbidden_event_type_at_turn_u)
   其中 severity 来自 §7.2 的分级表（如 protected_field_loss: 0.3, identity_violation: 1.0）

   一致性约束：
     C_safety(tau) = min(1, max_u c_u)
   即：轨迹级 C_safety 取所有 turn 中最严重违规的 severity。

3. 多 turn 共同导致违规的分配规则：
   若一个 forbidden event 需要多个 turn 的 action 共同触发（如 turn 3 删除 + turn 5 重建 = duplicate）：
     c_u = severity / n_contributing_turns  对每个 contributing turn
   这保证 sum_contributing c_u = severity，不膨胀总 cost。

4. 无违规轨迹：
   若 C_safety(tau) = 0，则所有 c_u = 0。

映射实现：
  event_log 中每个 forbidden event 记录了 step（即 turn）编号。
  c_u 的计算是确定性的：遍历 event_log，对每个 forbidden event，
  将其 severity 分配到对应 turn(s)。

**二值模式与分级模式的互斥约束**：

```text
两种 C_safety 模式不可在实验中途切换，原因如下：

  二值模式（Phase 1-2 推荐）：
    C_safety = 1 if any forbidden event else 0
    c_u = 1  if turn u triggers forbidden event else 0
    lambda_safe 的 dual ascent 基于二值 violation rate

  分级模式（可选扩展）：
    C_safety = min(1, max_u c_u)
    c_u = severity ∈ {0.3, 0.5, 0.7, 1.0}
    lambda_safe 的 dual ascent 基于分级 severity-weighted rate

  两种模式下的 hat_C_batch 含义不同：
    二值模式：hat_C_batch = 违规轨迹比例
    分级模式：hat_C_batch = mean(min(1, max_u severity_u))
    二者不可直接对比。

  切换规则：
    如果 Phase 3 需要切换到分级模式（例如为了区分不同严重程度在 LATA 中的衰减力度），
    必须重新跑 Phase 1-2 的 baseline（M4 系列）使用分级模式，
    否则 M5 vs M4+F+P 的对比会因为 C_safety 定义不同而失去意义。

  推荐策略：
    Phase 1-2 固定使用二值模式（解释最清楚）。
    Phase 3 的 LATA 也使用二值模式的 c_u（c_u ∈ {0, 1}），
    仅当二值模式下 unsafe success rate 无法降低时才考虑分级模式。
```
```

权重必须与轨迹 advantage 的符号一致。否则会出现严重错误：当 `A_i < 0` 时，如果仍然用 `softplus(q_u)`，负面事件的 `q_u` 更小，反而得到更小惩罚。

因此定义 signed relevance：

```text
if A_i >= 0:
  r_u = softplus(q_u / temperature)
else:
  r_u = softplus((-q_u) / temperature)
```

**temperature 推荐值**：

```text
temperature = 0.1（推荐）

量级分析（以 lambda_safe = 1.0, lambda_process = 0.3 为例）：

  q_u 的典型范围：
    危险 turn（c_u = 1, q_u ≈ -lambda_safe = -1.0）：
      softplus(-1.0 / 0.1) = softplus(-10) ≈ 4.5e-5 → 权重接近 0

    安全正向 turn（p_u = 0.08, q_u = lambda_process * 0.08 = 0.024）：
      softplus(0.024 / 0.1) = softplus(0.24) ≈ 0.81

    强正向 turn（F_u = 0.25, q_u = lambda_shape * 0.25 = 0.125）：
      softplus(0.125 / 0.1) = softplus(1.25) ≈ 1.46

    中性 turn（q_u ≈ 0）：
      softplus(0) ≈ 0.693

  ratio(强正向 / 危险) ≈ 1.46 / 4.5e-5 ≈ 3.2e4
  危险 turn 几乎不获得梯度，安全正向 turn 获得正常梯度。

对比 temperature = 1.0（默认值）：
  危险 turn：softplus(-1.0) ≈ 0.313
  强正向 turn：softplus(0.125) ≈ 1.06
  ratio ≈ 3.4 → 危险 turn 仍获得约 30% 的正常梯度权重，衰减力度不足。

推导来源：
  PROVE 使用 multicomponent reward 时通过权重平衡各组件量级。
  本方案中 q_u 的最大范围约为 [-lambda_safe, lambda_shape + lambda_process*p_max] ≈ [-1.0, 0.59]。
  temperature = 0.1 使得 softplus 在 q_u 的有效范围内从 ~0（危险）到 ~1.5（强正向）变化，
  提供足够的区分度而不使函数饱和。
```

若该 ablation 是 `LATA-only` 且不启用任何局部质量项，则使用 `r_u = 1`，只测试 sqrt(length) allocation 的信号通路效果。若启用 `process + LATA` 或 `F/P/C-local + LATA`，才使用上面的 signed relevance。

`q_u` 只用于分配同一条轨迹内部的 token/turn 权重，不改变轨迹级 advantage 的符号：

```text
sign(A_{i,u,token}) = sign(A_i)
```

再做 length-aware allocation：

```text
define: U = {u : L_u > 0}  (eligible turn set, excluding turns with zero policy tokens)

for u in U:
  g_u = r_u / sqrt(max(L_u, 1))

mean_token(g) = (sum_{u in U} L_u * g_u) / (sum_{u in U} L_u)
token_weight_u = g_u / mean_token(g)
A_{i,u,token} = A_i * token_weight_u
```

约束：

```text
mean_token_weight_per_trajectory = 1
sum_eligible_tokens A_{i,u,token} / num_eligible_tokens = A_i
```

这样：

```text
1. 好/坏事件附近的 turn 获得更强信号；
2. 长 turn 不会被 A/L 过度惩罚；
3. 总体更新方向仍由 group scalarized J 决定。
4. 正 advantage 强化正向局部事件，负 advantage 惩罚负向局部事件。
```

若训练框架暂时只支持轨迹级 reward，可以先使用 trajectory-level `A_i`，但必须记录为弱化版本，并在 ablation 中单独命名：

```text
M4a Constrained GRPO without LATA-style allocation
M4b Constrained GRPO with length-aware turn allocation
```

### 9.2 Group Saturation Diagnostics

Constrained GRPO 仍可能因为组内无方差而没有有效梯度。每个训练 step 必须记录：

```text
std(J_i within group)
std(C_safety_i within group)
all_success_group_rate
all_failure_group_rate
all_safe_group_rate
all_unsafe_group_rate
mixed_safety_group_rate
unsafe_success_rate
```

如果 `mixed_safety_group_rate` 长期接近 0，说明安全 cost 没有在组内形成可学习对比。可选处理：

```text
1. 提高 rollout temperature；
2. 增大 group size；
3. 对同一 task 注入 unsafe temptation / distractor action space；
4. 使用 replay buffer 混入同 prompt 的 safe/unsafe historical rollouts；
5. 暂停增大 lambda_safe，先修数据采样。
```

### 9.3 Saturated Group 与 Lambda Update 的交互规则

当 group 因 `std(J) < min_group_std` 被 skip policy gradient 时，其 rollout 的处理规则：

```text
1. lambda_safe 更新：saturated group 的 rollout 参与 hat_C_batch 计算。
   理由：这些 rollout 是 valid execution，只是组内无方差不产生 policy gradient。
   它们的 safety violation 信息是真实的，必须反映在约束满足度估计中。

2. 防止 lambda 单调增大的保护机制：
   如果连续 K_stall 个 training step 满足：
     all_unsafe_group_rate > tau_unsafe_stall  (e.g., tau_unsafe_stall = 0.5)
     且 lambda_safe 持续增大
   则触发 lambda stall protection：
     lambda_safe 冻结（不再增大）
     记录 lambda_stall_triggered = true
     优先执行数据采样调整（增加 safe success 样本比例）

3. 数学保证：
   hat_C_batch = mean_{all valid rollouts in B} C_safety(tau)
   其中 "all valid rollouts" 包括 saturated groups 的 rollout，
   但不包括 invalid reset rollout（hash 不一致的）。

4. 诊断指标：
   saturated_group_unsafe_rate:  被 skip 的 group 中 C_safety=1 的比例
   lambda_stall_count:          lambda 连续增大的 step 数
   effective_gradient_group_rate: 实际产生 policy gradient 的 group 比例
```

推荐超参：

```text
K_stall = 10
tau_unsafe_stall = 0.5
```

## 10. Rollout 训练循环

```text
For each training step:
  sample task batch B

  For each task q in B:
    For k = 1..G:
      session_id = hash(run_id, step, q.task_id, k)
      manager.reset(q.env_id, q.seed, session_id)
      if manager.get_state(q.env_id, session_id) is available:
        if hash(state) != q.initial_state_hash:
          mark rollout invalid and continue

      tau_k = rollout(policy, MCPTool(manager), AuditVerifier, q, session_id)

      R_k = R_task(tau_k)
      C_k = C_safety(tau_k)
      F_k = F_gamma(tau_k)
      P_k = P_process(tau_k)
      J_k = R_k + I_shape lambda_shape F_k + I_process lambda_process P_k - lambda_safe C_k

    if std(J_1...J_G) < min_group_std:
      record saturation diagnostic
      skip policy gradient for this group
    else:
      A_k = normalize_group(J_1...J_G)
      allocate A_k to trajectory or turns/tokens according to phase
      update policy with GRPO objective

  lambda_safe update by projected dual ascent using all valid rollouts in B
```

Dual update uses the batch-level empirical violation rate:

```text
hat_C_batch = mean_{valid q,k} C_safety(tau_{q,k})
lambda_safe = clip(lambda_safe + alpha_lambda (hat_C_batch - epsilon), 0, lambda_safe_max)
```

而不是对每个 task group 单独更新一次。否则不同 task 的安全难度会造成 lambda 抖动，并放大 batch 内任务顺序的影响。

"valid rollouts" 的定义：

```text
valid rollouts = all rollouts where reset hash is consistent
               = includes saturated groups (std(J) < min_group_std)
               = excludes invalid reset rollouts (hash mismatch)
```

即：saturated group 的 rollout 不产生 policy gradient，但参与 hat_C_batch 计算。
理由见 §9.3。

如果 reset hash 不一致，该 rollout 作废，不参与 policy gradient、lambda update 或 diagnostics 的分子/分母；另行记录 invalid_reset_rate。

## 11. 评测设计

### 11.1 Phase Plan

OVAL-MCP 分三阶段实现，避免把 safety、process signal 和 token-level allocation 的收益混在一起。

Phase 1 只验证 live execution + event-sourced safety + constrained GRPO：

```text
runtime:
  PROVE-style MCP subprocess servers
  session isolation / reset / replay validation
  2-4 DomainAdapter
  audit event log

training:
  binary C_safety
  PROVE-style R_task
  trajectory-level constrained GRPO
  I_process = 0
  I_shape = 0 by default; optional lightweight F ablation

goal:
  prove R_task - lambda_safe C_safety works on live MCP rollout
```

Phase 2 单独验证 trajectory-level shaping / process signal：

```text
M4:       R_task - lambda_safe C_safety
M4+F:     R_task + lambda_shape F_gamma - lambda_safe C_safety
M4+P:     R_task + lambda_process P_process - lambda_safe C_safety
M4+F+P:   R_task + lambda_shape F_gamma + lambda_process P_process - lambda_safe C_safety
```

**M4+F vs M4 的解释性约束**：

```text
R_coverage = completed_coverage_predicates / total_coverage_predicates
F_gamma（γ=1）= Phi(m_T) = completed_required_states / total_required_states

required_states 是 progress predicates 的子集（§8），
而 total_coverage_predicates 包含所有 non-safety predicates（§7.1）。
如果 progress predicates 是 coverage predicates 的子集（通常情况下是），
两者共享部分底层信息。

因此 M4+F vs M4 的消融必须记录：
  1. corr(R_coverage_component, F_gamma)  —— 报告 R_coverage 中 progress 相关部分与 F_gamma 的相关性
  2. partial_R2(F_gamma | R_coverage)      —— F_gamma 对 J 方差的独特贡献

解释规则：
  - 若 partial_R2 < 0.05：F_gamma 的提升主要来自放大已有 progress 信号（改变 group 内方差结构），
    而非提供新信息。这不意味着 M4+F 无效，但意味着其效果可以通过调大 R_coverage 权重模拟。
    **结论：M4+F 的提升归因于 progress emphasis（放大 progress 信号在 J 中的相对权重），
    不是新的信号源。**
  - 若 partial_R2 >= 0.05：F_gamma 提供了 R_coverage 之外的独特信息，
    M4+F 的提升可归因于 shaping 的信用分配改善。

即使完全共线（partial_R2 = 0），lambda_shape = 0.5 也改变了 progress signal 在 J 中的相对权重，
可能通过改变 group 内方差结构来影响训练动力学。但此时的效果不等价于 "F_gamma 提供了新信息"。

**P_process 的有效范围与解释**：

trajectory-level P_process（Phase 2）在典型情况下对组内 J 方差的贡献 < 1%。
原因：P_process 是基于进度谓词的 bounded score，在组内不同 trajectory 之间
方差极小——大部分 trajectory 要么都完成了进度谓词（P_process ≈ 1.0），
要么都没完成（P_process ≈ 0.0），仅在边界情况下产生差异。

P_process 的实际作用：
  - 反饱和：当组内所有 trajectory 的 R_task 相同时（saturation），
    P_process 的微小差异可以提供非零方差，防止 gradient signal 消失。
  - 不提供 step-level 信用分配改善：step-level 区分能力完全由 Phase 3 的
    LATA + 局部质量信号承担。若需要在 Phase 2 获得 step-level 区分，
    可设 gamma < 1。

```

Phase 3 再验证 turn/token signal path：

```text
requirements:
  turn/token span tracking
  eligible policy-token mask
  signed relevance weighting when local quality is enabled
  sqrt(length) allocation

ablations:
  trajectory-level baseline
  LATA-only          # r_u = 1, tests sqrt(length) allocation only
  process-only       # trajectory-level P_process, no token allocation
  process + LATA     # local q_u uses P_process-derived p_u
  F/P/C-local + LATA # optional full signed relevance path

additional training diagnostics:
  prefix_overlap_ratio —— 衡量组内轨迹多样性的诊断指标，不修改算法。

  定义：
    对组内 G 条轨迹，识别所有轨迹中行为相同的共享前缀 turn（同一 turn 产生完全相同的 a_t 和 o_t）。
    令 L_shared 为共享前缀中包含的 eligible policy token 数，
    L_total 为组内所有轨迹的 eligible policy token 总数。

    prefix_overlap_ratio = L_shared / L_total

  数学含义：
    高 prefix_overlap_ratio 意味着组内多条轨迹共享大量相同前缀 token，
    这些 token 的有效样本量接近 1（而非 G），梯度估计方差接近 σ²（而非 σ²/G）。
    这不是 LATA 的缺陷——LATA 在这些 token 上的权重分配是正确的。
    问题在于 rollout 采样阶段组内轨迹多样性不足——模型的探索行为不够分散。
    这与 min_group_std 检查的整条轨迹无方差是不同的问题：
    min_group_std 检测 J_i 无方差；prefix_overlap_ratio 检测 token 级多样性的缺失。

  诊断用途：
    - 若 prefix_overlap_ratio 长期 > 0.5：说明模型对早期 turn 的探索不足，
      即使 group 没有被 min_group_std 跳过，前缀 token 的梯度估计方差依然较大。
    - 降低方法（不在 LATA 内）：
      a. 提高 rollout temperature
      b. 增大 group size G
      c. 同一 task 配多个不同初始状态的 variant
      d. 在数据生成中注入任务级随机扰动（参见 §6 step 5a）
    - 若 prefix_overlap_ratio < 0.2：当前采样多样性足够，训练正常。

  实现注意：
    - 仅对 eligible policy token 计数（排除 prompt token 和 observation token）
    - 共享前缀的判断基于 (action_tokens, observation_tokens) 的 exact match
    - 该指标仅用于训练日志中的监控曲线，不参与梯度计算
```

### 11.1.1 外部评测

对标 PROVE 论文，使用以下开源 benchmark 做 external validation（与训练时的 OVAL-MCP reward 体系独立，仅作为泛化性证据）：

```text
BFCL Multi-Turn [Patil et al., 2024; Zhong et al., 2025]:
  - 评估多步 function calling 能力
  - 四个子类别：Base Multi-Turn / Missing Function / Missing Parameter / Long Context
  - 报告 Overall MT 及各子类别 accuracy
  - 项目已有 bfcl-eval>=2025.10 依赖（pyproject.toml）

T-Eval [Chen et al., 2023]:
  - 六个维度评估 tool-use：instruction following / planning / reasoning /
    retrieval / understanding / review
  - 分 JSON 和 string 两种 prompt 格式
  - 报告各维度及 Overall 分数

可选扩展（后续评估）：
  - MCPMark [Wu et al., 2025]：127 个多步 MCP 任务，与训练环境同构，可对接
    DomainAdapter 产出 OVAL-MCP 完整指标（C_safety / F_gamma / P_process）
  - τ²-bench [Barres et al., 2025]：对话式 agent 评估，需 LLM user simulator
```

评测运行方式：
```text
1. 训练完成后，各 ablation model checkpoint 在 BFCL Multi-Turn 和 T-Eval 上跑评测
2. BFCL Multi-Turn：使用 bfcl-eval 原生评测流程，不做修改
3. T-Eval：使用官方评测脚本，不做修改
4. 报告格式：各 model × 各 ablation × 各 benchmark 子指标矩阵
```

注意：BFCL Multi-Turn 和 T-Eval 是 outcome-only 评测（只看最终 tool call 正确性），
不包含 OVAL-MCP 的 trajectory event log。因此只能在 benchmark 上验证
"训练不会让模型在标准基准上退化"，无法验证 C_safety / F_gamma / P_process 的
消融效果。如需验证 safety 信号对外部任务的影响，使用 MCPMark（对接 DomainAdapter）。

### 11.2 对照组

这里 `C_final_state` 指只看最终可观察状态的 safety proxy；`C_event_log` 指由 audited event log 计算的 `C_safety`。M2/M3 使用固定 `lambda` 做对照；M4 及之后使用动态 `lambda_safe`。

```text
M0 Outcome-only GRPO:
  J = R_task

M1 Final-state safety:
  J = R_task - lambda C_final_state

M2 Event-sourced safety:
  J = R_task - lambda C_event_log

M3 Event-sourced safety + optional potential shaping:
  J = R_task + lambda_shape F_gamma - lambda C_event_log

M4 Constrained GRPO trajectory-level:
  J = R_task - lambda_safe C_event_log
  lambda_safe dynamic update

M4+F:
  J = R_task + lambda_shape F_gamma - lambda_safe C_event_log

M4+P:
  J = R_task + lambda_process P_process - lambda_safe C_event_log

M4+F+P:
  J = R_task + lambda_shape F_gamma + lambda_process P_process - lambda_safe C_event_log

M5 Turn/token allocation:
  J = R_task + lambda_shape F_gamma + lambda_process P_process - lambda_safe C_event_log
  turn/token advantage uses sqrt(length) allocation
```

### 11.3 指标

```text
Task Success Rate
Unsafe Success Rate
Constraint Violation Rate
Forbidden Transition Rate
Wrong Resource Mutation Rate
Identity / Provenance Violation Rate
Protected Field Loss Rate
Sensitive Param Provenance Violation Rate
Missing Dependency Rate
Over-call Ratio
Group Reward Saturation Rate
Mixed Safety Group Rate

External Benchmark Scores（独立于训练 reward，仅作泛化验证）:
  BFCL Multi-Turn Overall
  BFCL MT - Base / Miss-Func / Miss-Param / Long-Ctx
  T-Eval Overall
  T-Eval - Instruct / Plan / Reason / Retrieve / Understand / Review

Prefix Overlap Ratio
```

### 11.4 核心验证

```text
M2 vs M1:
  检验 event-sourced safety 是否能发现 final-state safety 漏掉的中间副作用。

M4 vs M2:
  检验 constrained GRPO 是否更稳定控制 violation rate。

M4+F / M4+P / M4+F+P vs M4:
  分别检验 potential shaping、process signal、二者组合的贡献。

M5 vs M4+F+P:
  检验 turn/token signal path 是否比 trajectory-level signal 更有效。
```

### 11.5 数据分布

OVAL-MCP 的训练数据不能只覆盖安全成功轨迹。每个 split 必须报告：

```text
normal_safe_success
unsafe_success_forbidden_transition
wrong_resource_mutation
identity_or_provenance_violation
protected_field_loss
missing_dependency
tool_error_recovery
no_tool_or_abstention
clarification_required
distractor_tools
overcall_redundant_read
```

训练分布建议：

```text
互斥任务类型分布（总和 = 100%）：
  normal_safe_success:          35%-45%
  unsafe temptation tasks:      20%-30%
  missing dependency/recovery:  10%-15%
  no_tool/clarification:        10%-15%

正交属性（与上述类型独立叠加，不互斥）：
  distractor-heavy schemas:     30%-40% of all tasks
```

说明：distractor-heavy 是正交维度，任何类型的 task 都可以同时具有 distractor-heavy 属性。
例如一个 unsafe temptation task 可以同时是 distractor-heavy（schema 中包含大量无关工具）。
互斥类型的百分比之和应为 100%（取中值时 40+25+12.5+12.5=90%，剩余 10% 为其他边界情况）。

其中 unsafe temptation task 指：存在表面上可完成 outcome 但会触发 safety cost 的捷径，例如：

```text
calendar: delete+create instead of identity-preserving update
banking: transfer/refund without required provenance
filesystem: overwrite/copy path that loses permission or identity
shopping: create duplicate order or mutate wrong cart
email/team_chat: send/post before recipient or thread verification
issue_tracker/crm: transition wrong issue/deal or skip required workflow state
```

Domain mixing 约束：

```text
1. 训练 split 不得由单一 MCP server 主导；
2. 每个 state archetype 至少有 success、recovery、abstention、distractor 样本；
3. reward distribution 必须按 MCP server、state_archetype、scenario_type 分组记录；
4. ablation 必须报告 BFCL Multi-Turn 和 T-Eval 上的外部评测结果。
```

## 12. 工程实现

工程实现细节（目录结构、测试清单、维护约定）见 [oval_mcp_engineering.md](./oval_mcp_engineering.md)。


## 13. 严谨性检查

正式实验必须同时满足：

```text
1. rollout 使用 actual MCP execution backend；
2. 每条 trajectory 有 session_id、policy actions、tool calls、observations、errors、terminal actions、event log；
3. reward/cost 只由 task predicates、event log、state checks 计算；
4. R_task、C_safety、F_gamma、P_process 分开记录，不存在双重惩罚；
4a. Phi(absorbing_failure) = Phi(m_{T-1})，safety failure 的惩罚完全由 C_safety 承担；
4b. R_validity 分为 structural 和 execution 两层，分别计分；
5. group advantage 先 scalarize J，再 normalize；
6. lambda_safe 使用 batch-level projected dual ascent，saturated group rollout 参与 hat_C_batch；
6a. lambda_safe 有上界 lambda_safe_max，有 stall protection 机制；
7. Phase 1 使用 trajectory-level constrained GRPO，不强制启用 P_process 或 LATA；
8. Phase 2 单独消融 F_gamma 与 P_process；
9. Phase 3 的 LATA-only 与 process+LATA 分开；只有启用局部质量项时才使用 signed relevance；
10. process score 的 penalty 项保持负值，不得通过 `B - negative_penalty` 变成奖励；
11. task-required field preservation 与 protected field loss 分开记录；
12. replay validation 使用 fresh isolated session，invalid reset rollout 不进入训练统计分母；
13. ablation 覆盖 outcome-only、event-safety、constrained、shaping、process、length-aware allocation；
14. eval 报告 BFCL Multi-Turn 和 T-Eval 结果，可选扩展 MCPMark；
15. 不用训练 surrogate J 代替真实任务指标。
```

## 14. 参考文献

1. `Synthesize and Reward -- Reinforcement Learning for Multi-Step Tool Use in Live Environments`, arXiv:2606.03892.
2. `Controllable and Verifiable Tool-Use Data Synthesis for Agentic Reinforcement Learning`, arXiv:2604.09813.
3. `Constrained Group Relative Policy Optimization`, arXiv:2602.05863.
4. `Potential-Based Shaping and Q-Value Initialization are Equivalent`, arXiv:1106.5267.
5. `qiqihezh/agentic-grpo-longhorizon`, GitHub repository, used as reward-design reference for PRM-Lite-style process signal, LATA-style advantage allocation, and ablation discipline.
6. Shishir G. Patil et al. "BFCL: The Berkeley Function Calling Leaderboard." In Advances in Neural Information Processing Systems, 2024.
7. Zhiqiang Zhong et al. "BFCL Multi-Turn: Multi-Step Function Calling Evaluation." 2025.
8. Zehui Chen et al. "T-Eval: Evaluating Tool-Use Capabilities of Large Language Models." arXiv:2312.14033, 2023.
9. Zachary Barres et al. "τ²-bench: A Benchmark for Tool-Using Conversational Agents with Dual-Control Environments." 2025.
10. Fanshi Zhang, Yaoqi Ye, Jiawei Wang et al. "MCPMark: A Benchmark for Stress-Testing Realistic and Faithful MCP Agents." 2025.
