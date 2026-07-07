import pandas as pd
import json
import numpy as np

train = pd.read_parquet('data/train.parquet')
val = pd.read_parquet('data/val.parquet')
all_data = pd.concat([train, val])
print('Total:', len(all_data))

# ===== Problem 1: Multi-turn =====
print('\n===== P1: Multi-turn oracle vs prompt =====')
multi = all_data[all_data['extra_info'].apply(lambda x: x.get('conversation_rounds', 1) > 1)]
print('Multi-turn tasks:', len(multi))
counts = []
for _, r in multi.iterrows():
    oc = r['extra_info'].get('oracle_calls', [])
    if isinstance(oc, str):
        oc = json.loads(oc)
    counts.append(len(oc))
if counts:
    print(f'Oracle calls range: [{min(counts)}, {max(counts)}]')
    print(f'Oracle calls mean: {np.mean(counts):.1f}, median: {np.median(counts)}')

# ===== Problem 3a: required_tools vs oracle tools mismatch =====
print('\n===== P3a: required_tools vs oracle tools mismatch =====')
tool_task_mask = all_data['scenario_type'].isin(['task_planner', 'distractor'])
tool_tasks = all_data[tool_task_mask]
mismatch = 0
for _, r in tool_tasks.iterrows():
    ei = r['extra_info']
    oc = ei.get('oracle_calls', [])
    if isinstance(oc, str):
        oc = json.loads(oc)
    required = ei.get('required_tools', [])
    if isinstance(required, str):
        required = [t.strip() for t in required.split(',') if t.strip()]
    oracle_names = set(c.get('tool_name', '') for c in oc if isinstance(c, dict))
    required_set = set(required)
    if oracle_names and required_set and oracle_names != required_set:
        mismatch += 1
print(f'Tasks with tool mismatch: {mismatch} / {len(tool_tasks)}')

# ===== Problem 3b: state_exists(path="") =====
print('\n===== P3b: state_exists with empty path =====')
empty_state = 0
total_criteria = 0
for _, r in all_data.iterrows():
    ei = r['extra_info']
    sc = ei.get('success_criteria', [])
    if isinstance(sc, str):
        sc = json.loads(sc)
    if not isinstance(sc, list):
        continue
    for c in sc:
        if isinstance(c, dict) and c.get('type') == 'state_exists' and c.get('path', '') == '':
            empty_state += 1
            break
    total_criteria += len(sc) if isinstance(sc, list) else 0
print(f'Tasks with empty state_exists: {empty_state} / {len(all_data)}')

# ===== Problem 4a: missing_function oracle fallback =====
print('\n===== P4a: missing_function required_tool_calls rebuild =====')
mf_tasks = all_data[all_data['scenario_type'] == 'missing_function']
print(f'missing_function tasks: {len(mf_tasks)}')
for _, r in mf_tasks.iterrows():
    ei = r['extra_info']
    oc = ei.get('oracle_calls', [])
    if isinstance(oc, str):
        oc = json.loads(oc)
    req = ei.get('required_tools', [])
    if isinstance(req, str):
        req = [t.strip() for t in req.split(',') if t.strip()]
    has_oracle = len(oc) > 0
    print(f'  {r["uid"]}: oracle_calls={len(oc)}, required_tools={req}, has_oracle={has_oracle}')

# ===== Problem 4b: clarification oracle locking terminal =====
print('\n===== P4b: clarification oracle in history =====')
clar = 0
for _, r in all_data.iterrows():
    ei = r['extra_info']
    oc = ei.get('oracle_calls', [])
    if isinstance(oc, str):
        oc = json.loads(oc)
    has_clar = any(isinstance(c, dict) and c.get('action') == 'clarification' for c in oc)
    if has_clar:
        clar += 1
        print(f'  {r["uid"]}: domain={ei.get("domain")}, scenario={r["scenario_type"]}, oracle_count={len(oc)}')
print(f'Total with clarification oracle: {clar}')
