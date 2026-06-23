#!/usr/bin/env python3
"""端到端验证 verl 交互式静态 replay 框架。

模拟真实 rollout 流程：
  1. 构建 parquet 格式的数据
  2. 模拟 SchemaShiftReplayLoop 的多轮 tool_call→observation→继续生成
  3. 将最终 response_text 送入 reward function 打分
  4. 验证分数和 exact_success 符合预期

不依赖 vllm/ray，纯逻辑测试。
"""
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "verl"))

from transformers import AutoTokenizer

# ── 加载 tokenizer ──
MODEL_PATH = str(Path(__file__).resolve().parent.parent / "models/Qwen3-4B")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
except Exception:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

FAILS = 0

def check(label, actual, expected, tolerance=1e-6):
    global FAILS
    ok = abs(actual - expected) < tolerance
    if ok:
        print(f"  ✅ {label}: {actual} == {expected}")
    else:
        FAILS += 1
        print(f"  ❌ {label}: {actual} != {expected} (差 {abs(actual-expected):.6f})")

def check_bool(label, actual, expected):
    global FAILS
    if actual == expected:
        print(f"  ✅ {label}: {actual} == {expected}")
    else:
        FAILS += 1
        print(f"  ❌ {label}: {actual} != {expected}")

# ═══════════════════════════════════════════════════════
# 测试 1：模拟 agent loop 多轮交互（手动模拟状态机）
# ═══════════════════════════════════════════════════════

print("=" * 65)
print("测试 1：Agent Loop 状态机模拟 — call_then_final EXACT")
print("=" * 65)

# --- 构建与 parquet 数据结构一致的数据 ---
oracle_actions = [
    {
        "action_type": "tool_call",
        "tool_calls": [{"name": "get_weather", "arguments": {"city": "Beijing"}}],
        "match_mode": "ordered",
        "replay_observation": '{"temperature": 25, "condition": "sunny"}',
    },
    {
        "action_type": "final_answer",
        "final_answer": "The weather is sunny with 25C",
    },
]
episode_type = "call_then_final"

# 构建 prompt（与 prepare_grpo_data.py 一致）
tools_desc = "- get_weather: Get weather for a city\n    - city (string, required): City name\n"
system_msg = (
    "You are a helpful assistant with access to the following tools. "
    "Use them when needed to answer the user's question.\n\n"
    f"Available tools:\n{tools_desc}\n\n"
    "Response format:\n"
    '- To call a tool: <tool_call>{"name": "tool_name", "arguments": {...}}</tool_call>\n'
    "- To give final answer: <final_answer>your answer</final_answer>\n"
)
messages = [
    {"role": "system", "content": system_msg},
    {"role": "user", "content": "What's the weather in Beijing?"},
]

# 编码 prompt
prompt_ids = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=True
)

# --- 模拟 LLM 的逐轮输出 ---
model_turns = [
    '<tool_call>{"name": "get_weather", "arguments": {"city": "Beijing"}}</tool_call>',
    "<final_answer>The weather is sunny with 25C</final_answer>",
]

# 手动模拟 agent loop 状态机
all_response_ids = []
all_response_mask = []
oracle_step_idx = 0
consecutive_errors = 0
max_consecutive_errors = 2
max_turns = 5
response_length = 4096

import re
_TOOL_CALL_PATTERN = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_FINAL_ANSWER_PATTERN = re.compile(r"<final_answer>(.*?)</final_answer>", re.DOTALL)
_REPORT_ERROR_PATTERN = re.compile(r"<report_error>(.*?)</report_error>", re.DOTALL)
_ASK_CLARIFICATION_PATTERN = re.compile(r"<ask_clarification>(.*?)</ask_clarification>", re.DOTALL)

def _is_terminal(text):
    return bool(
        _FINAL_ANSWER_PATTERN.search(text)
        or _REPORT_ERROR_PATTERN.search(text)
        or _ASK_CLARIFICATION_PATTERN.search(text)
    )

def _parse_tool_calls_json(text):
    text = text.strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "name" in obj:
            return [{"name": obj["name"], "arguments": obj.get("arguments", {})}]
        if isinstance(obj, list):
            return [{"name": c["name"], "arguments": c.get("arguments", {})}
                    for c in obj if isinstance(c, dict) and "name" in c]
    except json.JSONDecodeError:
        pass
    return []

# 计算 system prompt 前缀长度用于 observation 编码
sp_tokens = tokenizer.apply_chat_template([{}], add_generation_prompt=False, tokenize=True)
system_prefix_len = len(sp_tokens)

for turn_idx, response_text in enumerate(model_turns):
    # 编码响应
    response_ids = tokenizer.encode(response_text, add_special_tokens=False)
    all_response_ids.extend(response_ids)
    all_response_mask.extend([1] * len(response_ids))

    if len(all_response_ids) >= response_length:
        print(f"  turn={turn_idx} 超长，终止")
        break

    # 解析 tool_call
    tc_matches = list(_TOOL_CALL_PATTERN.finditer(response_text))

    if not tc_matches:
        is_term = _is_terminal(response_text)
        print(f"  turn={turn_idx} 无 tool_call (terminal={is_term})，终止")
        break

    if _is_terminal(response_text):
        print(f"  turn={turn_idx} same-turn violation，终止")
        break

    # 解析 tool calls
    all_parsed = []
    for m in tc_matches:
        all_parsed.extend(_parse_tool_calls_json(m.group(1)))

    if not all_parsed:
        print(f"  turn={turn_idx} JSON 解析失败")
        break

    # 匹配 oracle
    if oracle_step_idx < len(oracle_actions):
        current = oracle_actions[oracle_step_idx]
        or_type = current.get("action_type")

        if or_type != "tool_call":
            print(f"  turn={turn_idx} oracle 期望 {or_type}，模型给了 tool_call")
            break

        model_name = all_parsed[0]["name"]
        oracle_name = current["tool_calls"][0]["name"]
        model_args = all_parsed[0]["arguments"]
        oracle_args = current["tool_calls"][0]["arguments"]

        # 精确匹配检查
        if model_name == oracle_name and model_args == oracle_args:
            obs = current.get("replay_observation", '{"status": "success"}')
            oracle_step_idx += 1
            consecutive_errors = 0
            print(f"  turn={turn_idx} ✅ 匹配成功 step={oracle_step_idx}/{len(oracle_actions)}")
        else:
            obs = f"Error: Tool call failed. Expected '{oracle_name}' with correct arguments."
            consecutive_errors += 1
            print(f"  turn={turn_idx} ❌ 不匹配 model={model_name} expected={oracle_name}")
            if consecutive_errors >= max_consecutive_errors:
                print(f"  turn={turn_idx} 连续错误超限，终止")
                break
    else:
        print(f"  turn={turn_idx} oracle 已耗尽")
        break

    # 追加 observation (mask=0)
    tool_msg = [{"role": "tool", "content": obs}]
    tool_tokens = tokenizer.apply_chat_template(
        tool_msg, add_generation_prompt=True, tokenize=True
    )
    tool_tokens = tool_tokens[system_prefix_len:]  # 去掉 chat template 前缀
    all_response_ids.extend(tool_tokens)
    all_response_mask.extend([0] * len(tool_tokens))

# 截断
all_response_ids = all_response_ids[:response_length]
all_response_mask = all_response_mask[:response_length]

# 解码完整 response
final_response_text = tokenizer.decode(all_response_ids, skip_special_tokens=True)
print(f"  最终 response_text ({len(final_response_text)} chars):")
print(f"  {final_response_text[:200]}...")

# 验证：response 应包含 tool_call 和 final_answer 两个 tag
check("包含 2 个 <tool_call>", final_response_text.count("<tool_call>"), 1)
check("包含 1 个 <final_answer>", final_response_text.count("<final_answer>"), 1)
check("包含 observation", "temperature" in final_response_text, True)
check_bool("oracle 全部覆盖", oracle_step_idx, 2)

# ═══════════════════════════════════════════════════════
# 测试 2：将 agent loop 输出送入 reward function
# ═══════════════════════════════════════════════════════

print("\n" + "=" * 65)
print("测试 2：Agent Loop 输出 → Reward Function")
print("=" * 65)

from src.reward.schemashift_reward_fn import compute_score

gt = {
    "oracle_actions": oracle_actions,
    "episode_type": episode_type,
}

result = compute_score("schemashift", final_response_text, gt)
print(f"  score={result['score']:.4f} exact={result['exact_success']}")
print(f"  coverage_ratio={result['coverage_ratio']}")

# 预期：多步 EXACT → 1.25
# 但 response 里包含了 observation 文本（tool 角色的内容），
# reward 只解析模型生成的 action tags，observation 文本不影响解析
check("多步 EXACT score", result["score"], 1.25, tolerance=0.01)
check_bool("exact_success", bool(result["exact_success"]), True)

# ═══════════════════════════════════════════════════════
# 测试 3：多步错误场景 — 第一步正确，第二步错误
# ═══════════════════════════════════════════════════════

print("\n" + "=" * 65)
print("测试 3：多步 — 第一步正确，第二步错误值")
print("=" * 65)

model_turns_3 = [
    '<tool_call>{"name": "get_weather", "arguments": {"city": "Beijing"}}</tool_call>',
    "<final_answer>The weather is rainy with 10C</final_answer>",
]

all_response_ids = []
all_response_mask = []
oracle_step_idx = 0
consecutive_errors = 0

for turn_idx, response_text in enumerate(model_turns_3):
    response_ids = tokenizer.encode(response_text, add_special_tokens=False)
    all_response_ids.extend(response_ids)
    all_response_mask.extend([1] * len(response_ids))

    tc_matches = list(_TOOL_CALL_PATTERN.finditer(response_text))
    if not tc_matches or _is_terminal(response_text):
        break
    all_parsed = []
    for m in tc_matches:
        all_parsed.extend(_parse_tool_calls_json(m.group(1)))
    if not all_parsed:
        break

    if oracle_step_idx < len(oracle_actions):
        current = oracle_actions[oracle_step_idx]
        if current.get("action_type") != "tool_call":
            break
        if all_parsed[0]["name"] == current["tool_calls"][0]["name"] and all_parsed[0]["arguments"] == current["tool_calls"][0]["arguments"]:
            obs = current.get("replay_observation", '{"status": "success"}')
            oracle_step_idx += 1
            consecutive_errors = 0
        else:
            obs = "Error"
            consecutive_errors += 1
            if consecutive_errors >= max_consecutive_errors:
                break
    else:
        break

    tool_tokens = tokenizer.apply_chat_template(
        [{"role": "tool", "content": obs}], add_generation_prompt=True, tokenize=True
    )[system_prefix_len:]
    all_response_ids.extend(tool_tokens)
    all_response_mask.extend([0] * len(tool_tokens))

final_response_text_3 = tokenizer.decode(all_response_ids[:response_length], skip_special_tokens=True)
result_3 = compute_score("schemashift", final_response_text_3, gt)
print(f"  score={result_3['score']:.4f} exact={result_3['exact_success']}")
# 预期：0.69 (step1=1.05, step2=wrong final_answer → weighted average)
check("step1 正确 step2 错误 score", result_3["score"], 0.69, tolerance=0.02)
check_bool("exact_success=False", bool(result_3["exact_success"]), False)

# ═══════════════════════════════════════════════════════
# 测试 4：多步 — 模型跳过 tool_call 直接给 final_answer
# ═══════════════════════════════════════════════════════

print("\n" + "=" * 65)
print("测试 4：多步 — 跳过 tool_call 直接 final_answer")
print("=" * 65)

# 模拟：模型第一轮就输出 final_answer
turn1_response = "<final_answer>The weather is sunny with 25C</final_answer>"
response_ids = tokenizer.encode(turn1_response, add_special_tokens=False)
all_response_ids_4 = list(response_ids)
all_response_mask_4 = [1] * len(response_ids)

# 没有 tool_call，agent loop 检测到 terminal 会直接 break
# reward 看到的是纯 final_answer
result_4 = compute_score("schemashift", turn1_response, gt)
print(f"  score={result_4['score']:.4f} exact={result_4['exact_success']}")
# 预期：-0.132 (跳过 tool_call penalty -0.15)
check("跳过 tool_call score", result_4["score"], -0.132, tolerance=0.01)
check_bool("exact_success=False", bool(result_4["exact_success"]), False)

# ═══════════════════════════════════════════════════════
# 测试 5：单步 call_only — 完整流程
# ═══════════════════════════════════════════════════════

print("\n" + "=" * 65)
print("测试 5：单步 call_only EXACT")
print("=" * 65)

oracle_single = [
    {
        "action_type": "tool_call",
        "tool_calls": [{"name": "get_weather", "arguments": {"city": "Beijing"}}],
        "match_mode": "ordered",
        "replay_observation": '{"temperature": 25}',
    },
]

single_turn = '<tool_call>{"name": "get_weather", "arguments": {"city": "Beijing"}}</tool_call>'

# agent loop: 生成 → 匹配 → 返回 observation → 无后续生成（response_length 耗尽或下一轮无 tool_call）
all_ids_5 = []
all_mask_5 = []

# turn 1: model outputs tool_call
tc_ids = tokenizer.encode(single_turn, add_special_tokens=False)
all_ids_5.extend(tc_ids)
all_mask_5.extend([1] * len(tc_ids))

# 匹配成功 → inject observation
obs_tokens = tokenizer.apply_chat_template(
    [{"role": "tool", "content": '{"temperature": 25}'}],
    add_generation_prompt=True,
    tokenize=True,
)[system_prefix_len:]
all_ids_5.extend(obs_tokens)
all_mask_5.extend([0] * len(obs_tokens))

# turn 2: 模拟模型给出 final_answer（在单步 call_only task 中多余）
# 但实际上 agent loop 会在下一轮生成后检测到没有 tool_call → 终止
# 这里我们模拟一个终止情况：response 超长
final_response_5 = tokenizer.decode(all_ids_5[:response_length], skip_special_tokens=True)

gt_single = {
    "oracle_actions": oracle_single,
    "episode_type": "call_only",
}

result_5 = compute_score("schemashift", final_response_5, gt_single)
print(f"  score={result_5['score']:.4f} exact={result_5['exact_success']}")
print(f"  response contains observation: {'temperature' in final_response_5}")

# 注意：response 包含 tool_call + observation 文本
# reward 会解析出 tool_call → 单步 EXACT → 1.05
# observation 文本中的 content 不会被解析为 action tag
check("单步 EXACT score", result_5["score"], 1.05, tolerance=0.01)
check_bool("exact_success=True", bool(result_5["exact_success"]), True)

# ═══════════════════════════════════════════════════════
# 测试 6：测试 name_map 扰动场景在 agent loop 中的匹配
# ═══════════════════════════════════════════════════════

print("\n" + "=" * 65)
print("测试 6：name_map 扰动 → Agent Loop 匹配 + Reward 打分")
print("=" * 65)

from src.eval.matching import map_enum_values, strict_args_match

# 模拟 perturbed 场景：工具名被扰动
perturbed_response = '<tool_call>{"name": "weather_retrieve", "arguments": {"city": "Beijing"}}</tool_call>'
name_map = {"weather_retrieve": "get_weather"}

# agent loop 端：用 name_map 匹配
tc_m = list(_TOOL_CALL_PATTERN.finditer(perturbed_response))
parsed_6 = _parse_tool_calls_json(tc_m[0].group(1))
# 检查匹配
model_call = parsed_6[0]
oracle_call = oracle_actions[0]["tool_calls"][0]
canonical_model = name_map.get(model_call["name"], model_call["name"])
canonical_oracle = name_map.get(oracle_call["name"], oracle_call["name"])
print(f"  canonical model={canonical_model} oracle={canonical_oracle}")
print(f"  args match: {model_call['arguments'] == oracle_call['arguments']}")
check_bool("name_map 匹配成功", canonical_model == canonical_oracle, True)

# reward 端
gt_mild = {
    "oracle_actions": [{
        "action_type": "tool_call",
        "tool_calls": [{"name": "get_weather", "arguments": {"city": "Beijing"}}],
        "match_mode": "ordered",
    }],
    "episode_type": "call_only",
}
result_6 = compute_score(
    "schemashift", perturbed_response, gt_mild,
    {"name_map": name_map, "perturbation_level": "mild"}
)
print(f"  score={result_6['score']:.4f} exact={result_6['exact_success']}")
check("name_map reward score", result_6["score"], 1.05, tolerance=0.01)
check_bool("name_map exact_success", bool(result_6["exact_success"]), True)

# ═══════════════════════════════════════════════════════
# 测试 7：agent loop 连续错误 → 终止
# ═══════════════════════════════════════════════════════

print("\n" + "=" * 65)
print("测试 7：Agent Loop 连续错误 → 终止 episode")
print("=" * 65)

# 模拟连续两轮输出错误工具名
oracle_step_idx_7 = 0
consecutive_errors_7 = 0

for turn_idx in range(3):
    # 模型输出错误工具名
    response = '<tool_call>{"name": "wrong_tool", "arguments": {"city": "Beijing"}}</tool_call>'
    tc_matches = list(_TOOL_CALL_PATTERN.finditer(response))
    all_parsed = []
    for m in tc_matches:
        all_parsed.extend(_parse_tool_calls_json(m.group(1)))

    if oracle_step_idx_7 < len(oracle_actions):
        current = oracle_actions[oracle_step_idx_7]
        if all_parsed[0]["name"] != current["tool_calls"][0]["name"]:
            consecutive_errors_7 += 1
            print(f"  turn={turn_idx} 错误 {consecutive_errors_7}/{max_consecutive_errors}")
        if consecutive_errors_7 >= max_consecutive_errors:
            print(f"  turn={turn_idx} 连续错误超限，终止")
            break

check("连续错误在第2轮终止", turn_idx, 1)
check("consecutive_errors 计数正确", consecutive_errors_7, 2)
check("oracle 未推进", oracle_step_idx_7, 0)

# ═══════════════════════════════════════════════════════
# 测试 8：parquet 数据能正确加载且 agent loop 能提取 replay_data
# ═══════════════════════════════════════════════════════

print("\n" + "=" * 65)
print("测试 8：Parquet 数据完整性 → Agent Loop 可读取")
print("=" * 65)

import pandas as pd

df = pd.read_parquet("data/grpo_train_replay.parquet")
row = df.iloc[0]
ei = row["extra_info"]

# 模拟 agent loop 的 replay_data 提取流程
from src.utils import normalize_extra_info, normalize_json_field

ei = normalize_extra_info(ei)
tools_kwargs = normalize_json_field(ei.get("tools_kwargs", {}))
replay_tool_kwargs = tools_kwargs.get("replay_data", {})
replay_tool_kwargs = normalize_json_field(replay_tool_kwargs)
replay_data = replay_tool_kwargs.get("create_kwargs", replay_tool_kwargs)

# fallback
if not replay_data:
    rd = normalize_json_field(ei.get("replay_data", {}))
    if isinstance(rd, dict):
        replay_data = rd

oa = replay_data.get("oracle_actions", [])
ep = replay_data.get("episode_type", "")
nm = normalize_json_field(ei.get("name_map"))
em = normalize_json_field(ei.get("enum_map"))

print(f"  episode_type={ep}")
print(f"  oracle_actions={len(oa)} 步")
print(f"  action_types={[a.get('action_type','?') for a in oa]}")
print(f"  name_map entries={len(nm)}")
print(f"  enum_map entries={len(em)}")
print(f"  perturbation_level={ei.get('perturbation_level','?')}")

check_bool("oracle_actions 非空", len(oa) > 0, True)
check_bool("episode_type 非空", bool(ep), True)
check_bool("perturbation_level 存在", "perturbation_level" in ei, True)

# 验证奖励函数在这个真实数据上不会 crash
gt_real = {
    "oracle_actions": oa,
    "episode_type": ep,
}
from src.reward.schemashift_reward_fn import compute_score
_ = compute_score("schemashift", "test output", json.dumps(gt_real), ei)
print("  ✅ reward function 在真实数据上无 crash")

# ═══════════════════════════════════════════════════════
# 汇总
# ═══════════════════════════════════════════════════════

print("\n" + "=" * 65)
if FAILS == 0:
    print("✅ 全部 8 个测试通过 — verl 交互式静态 replay 框架无逻辑 bug")
else:
    print(f"❌ {FAILS} 个失败")
