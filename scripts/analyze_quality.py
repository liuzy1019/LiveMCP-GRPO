"""Analyze generated training data quality against PROVE standards."""
import pandas as pd
import json
import numpy as np
import sys

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        return super().default(obj)

train_path = sys.argv[1] if len(sys.argv) > 1 else '/tmp/big200_train.parquet'
val_path = sys.argv[2] if len(sys.argv) > 2 else '/tmp/big200_val.parquet'

train = pd.read_parquet(train_path)
val = pd.read_parquet(val_path)

print("=" * 60)
print("ROW-LEVEL INSPECTION (3 train samples)")
print("=" * 60)

for idx in [0, 50, 130]:
    ei = eval(train['extra_info'].iloc[idx]) if isinstance(train['extra_info'].iloc[idx], str) else train['extra_info'].iloc[idx]
    oc = ei['oracle_calls']
    print(f"\n--- Row {idx} | domain={ei['domain']} scenario={ei['scenario_type']} rounds={ei['conversation_rounds']} dist={ei['has_distractors']} miss={ei['has_missing_function']} perturb={ei['perturbation_level']} ---")
    print(f"  oracle_calls: {len(oc)} calls, required_tools: {ei['required_tools']}")
    if len(oc) > 0:
        c0 = oc[0]
        if isinstance(c0, dict):
            d = {k: str(v)[:80] for k, v in c0.items()}
            print(f"  call[0]: {json.dumps(d, cls=NpEncoder)}")
        else:
            print(f"  call[0] type: {type(c0)}")

print("\n" + "=" * 60)
print("PROMPT STRUCTURE ANALYSIS (first 2 rows)")
print("=" * 60)

for idx in [0, 2]:
    ei = eval(train['extra_info'].iloc[idx]) if isinstance(train['extra_info'].iloc[idx], str) else train['extra_info'].iloc[idx]
    p = train['prompt'].iloc[idx]
    msgs = json.loads(p) if isinstance(p, str) else p
    user_msgs = [m for m in msgs if m.get('role') == 'user']
    assistant_msgs = [m for m in msgs if m.get('role') == 'assistant']
    tool_msgs = [m for m in msgs if m.get('role') == 'tool']
    
    print(f"\n--- Row {idx}: domain={ei['domain']} scenario={ei['scenario_type']} rounds={ei['conversation_rounds']} ---")
    print(f"  Messages: total={len(msgs)} system={1 if msgs[0]['role']=='system' else 0} user={len(user_msgs)} assistant={len(assistant_msgs)} tool={len(tool_msgs)}")
    for i, um in enumerate(user_msgs):
        print(f"  User[{i}]: {um['content'][:120]}")
    for i, am in enumerate(assistant_msgs):
        tc = am.get('tool_calls', [])
        tc_names = [t.get('function', {}).get('name', '?') for t in tc] if tc else []
        print(f"  Assistant[{i}]: calls={tc_names}, content_len={len(am.get('content', ''))}")

print("\n" + "=" * 60)
print("REWARD_MODEL STRUCTURE (first 2 rows)")
print("=" * 60)

for idx in [0, 1]:
    rm_raw = train['reward_model'].iloc[idx]
    rm = eval(rm_raw) if isinstance(rm_raw, str) else rm_raw
    print(f"\n--- Row {idx} type={type(rm).__name__} ---")
    if isinstance(rm, dict):
        for k, v in rm.items():
            if isinstance(v, (list, np.ndarray)):
                print(f"  {k}: list(len={len(v)})")
                if len(v) > 0:
                    if isinstance(v[0], dict):
                        print(f"    [0]: keys={list(v[0].keys())[:8]}")
            else:
                print(f"  {k}: {repr(v)[:150]}")
    elif isinstance(rm, list):
        print(f"  list(len={len(rm)})")
        if len(rm) > 0 and isinstance(rm[0], dict):
            print(f"  [0]: keys={list(rm[0].keys())[:8]}")

print("\n" + "=" * 60)
print("AGGREGATE QUALITY METRICS")
print("=" * 60)

def analyze_df(df, name):
    total_dist = 0
    total_miss = 0
    scenarios = {}
    rounds_dist = {}
    oc_lens = []
    perturb_levels = {}
    domains = {}
    
    for i in range(len(df)):
        ei = eval(df['extra_info'].iloc[i]) if isinstance(df['extra_info'].iloc[i], str) else df['extra_info'].iloc[i]
        if ei.get('has_distractors'): total_dist += 1
        if ei.get('has_missing_function'): total_miss += 1
        sc = ei.get('scenario_type', 'unknown')
        scenarios[sc] = scenarios.get(sc, 0) + 1
        r = ei.get('conversation_rounds', 0)
        rounds_dist[r] = rounds_dist.get(r, 0) + 1
        oc = ei['oracle_calls']
        oc_lens.append(len(oc))
        pl = ei.get('perturbation_level', 'unknown')
        perturb_levels[pl] = perturb_levels.get(pl, 0) + 1
        d = ei.get('domain', 'unknown')
        domains[d] = domains.get(d, 0) + 1
    
    print(f"\n[{name}] Total: {len(df)}")
    print(f"  Has distractors: {total_dist}/{len(df)} ({100*total_dist/len(df):.0f}%)")
    print(f"  Has missing_function: {total_miss}/{len(df)} ({100*total_miss/len(df):.0f}%)")
    print(f"  Oracle calls/row: min={min(oc_lens)} max={max(oc_lens)} avg={sum(oc_lens)/len(oc_lens):.1f}")
    print(f"  Scenarios: {scenarios}")
    print(f"  Rounds: {rounds_dist}")
    print(f"  Perturbation: {perturb_levels}")
    print(f"  Domains: {domains}")

analyze_df(train, 'TRAIN')
analyze_df(val, 'VAL')

# -- Tool call diversity --
print("\n" + "=" * 60)
print("TOOL CALL DIVERSITY")
print("=" * 60)

all_oc_lens = []
tool_name_freq = {}
for i in range(len(train)):
    ei = eval(train['extra_info'].iloc[i]) if isinstance(train['extra_info'].iloc[i], str) else train['extra_info'].iloc[i]
    oc = ei['oracle_calls']
    all_oc_lens.append(len(oc))
    for call in oc:
        if isinstance(call, dict):
            name = call.get('name', '?')
            tool_name_freq[name] = tool_name_freq.get(name, 0) + 1

print(f"Oracle calls: total={sum(all_oc_lens)}, avg/row={sum(all_oc_lens)/len(all_oc_lens):.2f}")
print(f"Unique tools used in oracle: {len(tool_name_freq)}")

# -- PROVE-specific checks --
print("\n" + "=" * 60)
print("PROVE COMPLIANCE GAPS")
print("=" * 60)

# Check 1: Distractor ratio (PROVE uses 40%)
dist_ratio = total_dist / len(train)
print(f"1. Distractor ratio: {dist_ratio:.0%} (PROVE target: 40%)")

# Check 2: Irrelevance queries (PROVE uses 5%)
irr_train = train[train['scenario_type'] == 'irrelevant']
print(f"2. Irrelevance queries: {len(irr_train)}/{len(train)} ({100*len(irr_train)/len(train):.1f}%) (PROVE target: 5%)")

# Check 3: Missing-function ratio (PROVE includes these)
print(f"3. Missing-function rows: {total_miss}/{len(train)} ({100*total_miss/len(train):.1f}%)")

# Check 4: Conversation rounds (PROVE uses 2-3)
print(f"4. Turn distribution: {rounds_dist} (PROVE: min=2, max=3)")

# Check 5: Oracle calls per conversation (should be 2-5 tool chains)
print(f"5. Chain length (oracle calls): avg={sum(all_oc_lens)/len(all_oc_lens):.1f} (PROVE: 2-5)")

# Check 6: Sample query grounding — do queries reference real entities?
# Need to check a few prompts for entity references
print("\n6. Entity grounding check (first 5 train queries):")
for idx in range(min(5, len(train))):
    ei = eval(train['extra_info'].iloc[idx]) if isinstance(train['extra_info'].iloc[idx], str) else train['extra_info'].iloc[idx]
    p = train['prompt'].iloc[idx]
    msgs = json.loads(p) if isinstance(p, str) else p
    user_msgs = [m for m in msgs if m.get('role') == 'user']
    first_query = user_msgs[0]['content'] if user_msgs else ''
    print(f"  [{idx}] domain={ei['domain']}: \"{first_query[:150]}\"")

# Check 7: data_source — is it properly set?
print(f"\n7. Data sources in train:")
print(train['data_source'].value_counts().to_string())
print(f"\n8. Scenario types in train:")
print(train['scenario_type'].value_counts().to_string())
