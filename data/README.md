# LiveMCP Data

本目录保存依赖图缓存和本地生成数据。Parquet、运行日志和单次生成目录默认不进入 Git。

## Layout

```text
data/
├── dependency_graphs/              # 按 domain 和 schema hash 索引的依赖图
├── runs/                           # 本地生成 run，不进入 Git
│   └── <run-id>/
│       ├── candidates/             # 可恢复 shard 候选
│       ├── train.parquet
│       ├── val.parquet
│       └── merge_report.json
├── train.parquet                   # 指向已发布 train split 的本地符号链接
└── val.parquet                     # 指向已发布 validation split 的本地符号链接
```

只有 train 和 validation 均完成生成、全局过滤与生产审计后，才允许更新活动符号链接。未完成
run 的 `survivor_pool.parquet` 和 `candidates/` 不能作为正式训练数据。

## Current Snapshot

当前联合候选集包含 2,121 条轨迹：

| Split | Rows | UID unique | Contract diagnostics | Prompt overflow |
|-------|-----:|-----------:|---------------------:|----------------:|
| Train | 1,621 | 1,621 | 0 | 0 |
| Validation | 500 | 500 | 0 | 0 |

联合候选覆盖 10 个 domain。`data/train.parquet` 和 `data/val.parquet` 当前仍指向上一轮已发布
数据；切换前还需要完成人工语义抽检和 Policy 消费 smoke。

## Generation Pipeline

### 1. Dependency Graph

对每个 domain 的全部无序工具 pair 进行一次关系分类：

- `explicit`：source 的输出为 target 提供必需输入；
- `implicit`：source 必须先执行以建立 target 所需状态；
- `none`：当前 pair 不建立依赖边。

分类结果保存完整 pair ledger、schema hash、Teacher model ID 和 classifier contract hash。缺少
pair、引用未知工具或与当前 schema 不一致的缓存会直接失效。

### 2. Live-State Sampling

生成 query 前，系统在当前 session 中调用只读 discovery tools，收集可用实体和状态。Query
Teacher 只看到与当前 chain 相关的 compact view；opaque ID 必须由用户输入或此前 tool
observation 提供，不能从隐藏状态复制。

### 3. State-Machine Teacher

每个 conversation 依次经过：

```text
query generation -> teacher decision -> tool execution -> recovery -> continuation
```

Teacher 只能使用当前可见 schema、用户请求和真实 execution history。所有 tool calls 都在有状态
MCP session 中执行，并记录 observation、terminal、state delta 和成功条件。

### 4. Robustness Injection

扰动计划在 Teacher 执行前按 task seed 固定：

- distractor tools：混入跨域无关工具；
- enum stripping：移除部分枚举提示；
- missing function：隐藏完成目标所需的能力；
- irrelevance：生成当前工具集合无法处理的请求。

Teacher、Replay 和 Parquet 使用同一份可见 schema。隐藏工具不能出现在 Teacher 输入、执行器
路由或 ground-truth oracle 中。

### 5. Replay and Deduplication

候选轨迹需要经过：

1. fresh-session replay；
2. sensitive-parameter provenance；
3. terminal、round 和 hidden-tool contract；
4. success-criteria 验证；
5. tool-call sequence Jaccard 0.70 全局去重；
6. Parquet round-trip 和训练 parser readback。

## Parquet Contract

每一行包含以下顶层字段：

| Field | Description |
|-------|-------------|
| `prompt` | Policy 初始对话和可见 MCP schema |
| `data_source` | 数据来源标识 |
| `reward_model` | reward ground truth 容器 |
| `extra_info` | oracle、环境、replay 和审计信息 |
| `uid` | 全局唯一轨迹 ID |
| `group_id` | GRPO grouping key |
| `perturbation_level` | robustness 配置摘要 |
| `scenario_type` | success、recovery、clarification 或 abstention 类型 |

`extra_info` 中的关键合同字段包括：

- `oracle_calls`、`required_tools`、`required_call_rounds`；
- `round_contracts`、`terminal_action`、`allowed_terminal_actions`；
- `hidden_tools`、`tool_owner_domains`、`candidate_tools`；
- `success_criteria`、`criterion_call_provenance`；
- `canonical_replay_*`、`paper_replay_valid`；
- `server_schema_hashes`、`transition_fingerprints`、`initial_state_hashes`；
- `reward_fingerprint`、`trajectory_schema_version`、`budget`。

`success_criteria` 和 heterogeneous nested structures 在 Parquet 中按当前 serializer contract
编码，读取后必须通过训练端 `_build_task_dict`，不能仅以 PyArrow 写入成功作为验收依据。

## Generate

```bash
# 全域生成。
bash scripts/generate_data.sh --count 500 --val-count 100

# 限制 GPU 数。
GPU_COUNT=4 bash scripts/generate_data.sh --count 500 --val-count 100

# 单域 smoke。
bash scripts/generate_data.sh --domain calendar --count 20 --val-count 5
```

默认输出目录为 `data/runs/<run-id>/`。模型路径、API endpoint、GPU 数、并发和 vLLM 参数均可
通过脚本参数或环境变量覆盖。

## Validate

```bash
# 依赖图和 MCP schema 静态验证。
python scripts/validate_generation_pipeline.py --stages 1,2

# 正式 Parquet 逐行合同审计。
python scripts/audit_generated_data.py \
  data/runs/<run-id>/train.parquet \
  data/runs/<run-id>/val.parquet

# 完整回归测试。
python -m pytest tests/
```

验收报告至少需要记录：输入候选数、task-ID 去重数、Jaccard 去重数、quarantine 数、各 domain
保留量、train/validation 数量和环境指纹。

## Consume

```python
import pandas as pd

train = pd.read_parquet("data/train.parquet")
validation = pd.read_parquet("data/val.parquet")
```

训练入口会再次校验 schema、transition、initial-state、reward fingerprint、round budget 和 prompt
长度。任何不兼容行都会 fail closed，不会静默修补旧 metadata。

## Publication Checklist

在发布到 Hugging Face 或 ModelScope 前，还需要：

- 固定 train/validation artifact 和校验和；
- 完成人工语义抽检并记录抽样方法；
- 提供 Dataset Card、字段说明和生成模型信息；
- 确认数据许可证和上游模型使用条款；
- 扫描 secret、机器路径和潜在个人信息；
- 用公开代码对最终 artifact 重跑生产审计。
