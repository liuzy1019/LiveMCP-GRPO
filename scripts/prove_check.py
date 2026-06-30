"""PROVE compliance checker for generated training data."""
import pandas as pd, json, numpy as np, sys

train_path = sys.argv[1] if len(sys.argv) > 1 else '/tmp/big200_train.parquet'
val_path = sys.argv[2] if len(sys.argv) > 2 else '/tmp/big200_val.parquet'
train = pd.read_parquet(train_path)
val = pd.read_parquet(val_path)

print("=" * 60)
print("PROVE QUALITY COMPLIANCE REPORT")
print("=" * 60)

# ── 1. Prompt structure: multi-turn conversation completeness ──
print("\n##### 1. PROMPT STRUCTURE #####")
for idx in [0, 3, 100]:
    ei = eval(train['extra_info'].iloc[idx]) if isinstance(train['extra_info'].iloc[idx], str) else train['extra_info'].iloc[idx]
    p = json.loads(train['prompt'].iloc[idx]) if isinstance(train['prompt'].iloc[idx], str) else train['prompt'].iloc[idx]
    roles = [m['role'] for m in p]
    print(f"\nRow {idx}: domain={ei['domain']} scenario={ei['scenario_type']} rounds={ei['conversation_rounds']}")
    print(f"  Pattern: {' → '.join(roles)}")
    for m in p:
        if m['role'] == 'assistant':
            has_tc = 'tool_calls' in m
            content_preview = str(m.get('content', ''))[:80]
            print(f"  assistant: tool_calls={'YES' if has_tc else 'NO'} content=\"{content_preview}\"")
    # Show first user query fully
    user_msgs = [m for m in p if m['role'] == 'user']
    if user_msgs:
        print(f"  user[0]: \"{user_msgs[0]['content']}\"")

# ── 2. Oracle calls: parse from JSON and check structure ──
print("\n\n##### 2. ORACLE CALL QUALITY #####")
total_oracle_calls = 0
empty_oracle = 0
tool_name_counts = {}
tool_call_lengths = []
actions = {}

for i in range(len(train)):
    ei_raw = train['extra_info'].iloc[i]
    ei = eval(ei_raw) if isinstance(ei_raw, str) else ei_raw
    oc_json = ei.get('oracle_calls', '[]')
    try:
        oc = json.loads(oc_json) if isinstance(oc_json, str) else oc_json
        total_oracle_calls += len(oc)
        tool_call_lengths.append(len(oc))
        if len(oc) == 0:
            empty_oracle += 1
        for call in oc:
            name = call.get('tool_name', 'unknown')
            tool_name_counts[name] = tool_name_counts.get(name, 0) + 1
            action = call.get('action', 'tool_call')
            actions[action] = actions.get(action, 0) + 1
    except:
        print(f"  ERROR parsing oracle_calls at row {i}")
        break

print(f"Total oracle calls: {total_oracle_calls}")
print(f"Empty oracle rows: {empty_oracle}/{len(train)} ({100*empty_oracle/len(train):.1f}%)")
print(f"Unique tool names: {len(tool_name_counts)}")
print(f"Oracle calls/row: min={min(tool_call_lengths)} max={max(tool_call_lengths)} avg={sum(tool_call_lengths)/len(tool_call_lengths):.2f}")
print(f"Actions: {actions}")
print(f"Top 15 tools: {sorted(tool_name_counts.items(), key=lambda x: -x[1])[:15]}")

# ── 3. PROVE perturbation knobs check ──
print("\n\n##### 3. PERTURBATION KNOBS (vs PROVE targets) #####")
train_ei = train['extra_info'].apply(lambda x: eval(x) if isinstance(x, str) else x)

has_dist = train_ei.apply(lambda x: x.get('has_distractors', False))
has_miss = train_ei.apply(lambda x: x.get('has_missing_function', False))
scenario = train_ei.apply(lambda x: x.get('scenario_type', 'unknown'))
rounds = train_ei.apply(lambda x: x.get('conversation_rounds', 0))
domains = train_ei.apply(lambda x: x.get('domain', 'unknown'))

print(f"Distractors: {has_dist.sum()}/{len(train)} ({has_dist.mean():.1%})  PROVE target: 40%")
print(f"Missing-function: {has_miss.sum()}/{len(train)} ({has_miss.mean():.1%})  PROVE target: 20%")
irrelevant = (scenario == 'irrelevant').sum()
print(f"Irrelevance: {irrelevant}/{len(train)} ({irrelevant/len(train):.1%})  PROVE target: 5%")
print(f"Rounds distribution: {rounds.value_counts().sort_index().to_dict()}  PROVE target: min=2, max=3")
print(f"Scenario distribution: {scenario.value_counts().to_dict()}")

# ── 4. Chain length (tool chain complexity) ──
print("\n\n##### 4. TOOL CHAIN ANALYSIS #####")
# Group oracle calls by domain
domain_chain_lens = {}
for i in range(len(train)):
    ei = eval(train['extra_info'].iloc[i]) if isinstance(train['extra_info'].iloc[i], str) else train['extra_info'].iloc[i]
    domain = ei.get('domain', 'unknown')
    oc_json = ei.get('oracle_calls', '[]')
    try:
        oc = json.loads(oc_json) if isinstance(oc_json, str) else oc_json
        domain_chain_lens.setdefault(domain, []).append(len(oc))
    except:
        pass

print("Chain lengths by domain:")
for d in sorted(domain_chain_lens.keys()):
    lens = domain_chain_lens[d]
    print(f"  {d:20s}: min={min(lens):2d} max={max(lens):2d} avg={sum(lens)/len(lens):5.1f}  PROVE range: 2-5")

# ── 5. Entity grounding — sample real queries with entity references ──
print("\n\n##### 5. ENTITY GROUNDING SAMPLES (first 10 queries) #####")
for idx in range(min(10, len(train))):
    ei = eval(train['extra_info'].iloc[idx]) if isinstance(train['extra_info'].iloc[idx], str) else train['extra_info'].iloc[idx]
    p = json.loads(train['prompt'].iloc[idx]) if isinstance(train['prompt'].iloc[idx], str) else train['prompt'].iloc[idx]
    user_msgs = [m for m in p if m['role'] == 'user']
    first_q = user_msgs[0]['content'] if user_msgs else ''
    oc_json = ei.get('oracle_calls', '[]')
    try:
        oc = json.loads(oc_json) if isinstance(oc_json, str) else oc_json
        tool_names = [c.get('tool_name', '?') for c in oc]
    except:
        tool_names = []
    print(f"  [{idx}] domain={ei['domain']:15s} tools={tool_names}")
    print(f"       query: \"{first_q[:200]}\"")

# ── 6. Duplicate task check ──
print("\n\n##### 6. DEDUPLICATION #####")
uids = train['uid'].tolist()
print(f"Unique UIDs: {len(set(uids))} / {len(uids)}")
print(f"Duplicate UIDs: {len(uids) - len(set(uids))}")

# Check prompt similarity via first user query
queries = []
for i in range(len(train)):
    p = json.loads(train['prompt'].iloc[i]) if isinstance(train['prompt'].iloc[i], str) else train['prompt'].iloc[i]
    user_msgs = [m for m in p if m['role'] == 'user']
    queries.append(user_msgs[0]['content'] if user_msgs else '')

unique_queries = set(queries)
print(f"Unique first-user-queries: {len(unique_queries)} / {len(queries)}")
print(f"Duplicate queries: {len(queries) - len(unique_queries)}")

# ── 7. Reward model format check ──
print("\n\n##### 7. REWARD_MODEL FORMAT #####")
rm_sample = eval(train['reward_model'].iloc[0]) if isinstance(train['reward_model'].iloc[0], str) else train['reward_model'].iloc[0]
print(f"Keys: {list(rm_sample.keys())}")
gt = rm_sample.get('ground_truth', {})
print(f"ground_truth keys: {list(gt.keys()) if isinstance(gt, dict) else 'N/A'}")
if isinstance(gt, dict):
    oc_json = gt.get('oracle_calls', '')
    try:
        oc = json.loads(oc_json) if isinstance(oc_json, str) else oc_json
        print(f"ground_truth.oracle_calls count: {len(oc)}")
        if len(oc) > 0:
            print(f"  first call: {json.dumps(oc[0], ensure_ascii=False)[:200]}")
    except:
        print(f"Could not parse oracle_calls: {str(oc_json)[:100]}")

# ── Summary ──
print("\n\n" + "=" * 60)
print("QUALITY SUMMARY")
print("=" * 60)
print(f"Train rows: {len(train)}, Val rows: {len(val)}")
print(f"Total oracle calls: {total_oracle_calls}")
print(f"Distinct tools: {len(tool_name_counts)}")
print(f"10 domains: {sorted(domains.unique())}")
print(f"Prompt format: system + user/assistant interleaved, NO tool_calls in assistant, NO tool responses")
print(f"Oracle storage: JSON string in extra_info + reward_model.ground_truth")
