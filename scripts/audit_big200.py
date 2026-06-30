"""定位 BUG 的针对性审查 — 写到 workspace 下的临时位置"""
import pandas as pd, json
from collections import Counter

train = pd.read_parquet('/tmp/big200_train.parquet')

print("=== ground_truth vs extra_info oracle_calls 一致性 ===")
for idx in [1, 3, 100]:
    rm = train['reward_model'].iloc[idx]
    rm = eval(rm) if isinstance(rm, str) else rm
    gt = rm.get('ground_truth', {}) if isinstance(rm, dict) else {}
    gt_oc_raw = gt.get('oracle_calls', '[]')
    gt_oc = json.loads(gt_oc_raw) if isinstance(gt_oc_raw, str) else gt_oc_raw

    ei_raw = train['extra_info'].iloc[idx]
    ei = eval(ei_raw) if isinstance(ei_raw, str) else ei_raw
    ei_oc_raw = ei.get('oracle_calls', '[]')
    ei_oc = json.loads(ei_oc_raw) if isinstance(ei_oc_raw, str) else ei_oc_raw

    print(f"row {idx}: gt_oc={len(gt_oc)} ei_oc={len(ei_oc)}")
    if len(gt_oc) != len(ei_oc):
        print(f"  ↑ MISMATCH: gt raw type={type(gt_oc_raw).__name__} preview={str(gt_oc_raw)[:200]}")

print("\n=== assistant content uniqueness ===")
asst_contents = set()
for i in range(len(train)):
    p = train['prompt'].iloc[i]
    msgs = json.loads(p) if isinstance(p, str) else list(p)
    for m in msgs:
        if m.get('role') == 'assistant':
            asst_contents.add(m.get('content','')[:100])
print(f"unique assistant contents: {len(asst_contents)}")
for c in list(asst_contents)[:5]:
    print(f"  - {repr(c)[:120]}")

print("\n=== row 1 full prompt (multi-turn) ===")
p = train['prompt'].iloc[1]
msgs = json.loads(p) if isinstance(p, str) else list(p)
for m in msgs[1:]:
    print(f"  [{m['role']:9s}] {m.get('content','')[:200]}")

print("\n=== duplicate queries ===")
qs = []
for i in range(len(train)):
    p = train['prompt'].iloc[i]
    msgs = json.loads(p) if isinstance(p, str) else list(p)
    user_msgs = [m for m in msgs if m.get('role') == 'user']
    qs.append((i, user_msgs[0]['content'] if user_msgs else ''))

q_count = Counter(q for _, q in qs)
for q, n in q_count.most_common(8):
    if n > 1:
        idxs = [i for i, x in qs if x == q]
        ocs_per = []
        for j in idxs:
            ei = eval(train['extra_info'].iloc[j]) if isinstance(train['extra_info'].iloc[j], str) else train['extra_info'].iloc[j]
            oc_raw = ei.get('oracle_calls', '[]')
            oc = json.loads(oc_raw) if isinstance(oc_raw, str) else oc_raw
            tool_seq = tuple(c.get('tool_name','') for c in oc)
            ocs_per.append(tool_seq)
        unique_ocs = len(set(ocs_per))
        print(f"  [{n}x at {idxs}] unique oracle seq={unique_ocs}: \"{q[:90]}\"")

print("\n=== 0-call rows scenario distribution ===")
zero_scenarios = Counter()
for i in range(len(train)):
    ei = eval(train['extra_info'].iloc[i]) if isinstance(train['extra_info'].iloc[i], str) else train['extra_info'].iloc[i]
    oc_raw = ei.get('oracle_calls', '[]')
    oc = json.loads(oc_raw) if isinstance(oc_raw, str) else oc_raw
    real = [c for c in oc if (c.get('action') or 'tool_call') != 'clarification']
    if len(real) == 0:
        zero_scenarios[(ei.get('scenario_type','?'), ei.get('domain','?'))] += 1
print(f"total 0-call rows: {sum(zero_scenarios.values())}")
for (s, d), n in zero_scenarios.most_common(15):
    print(f"  scen={s:18s} domain={d:15s}: {n}")

print("\n=== overshoot chain (>5 calls) examples ===")
over = 0
for i in range(len(train)):
    ei = eval(train['extra_info'].iloc[i]) if isinstance(train['extra_info'].iloc[i], str) else train['extra_info'].iloc[i]
    oc_raw = ei.get('oracle_calls', '[]')
    oc = json.loads(oc_raw) if isinstance(oc_raw, str) else oc_raw
    real = [c for c in oc if (c.get('action') or 'tool_call') != 'clarification']
    if len(real) > 5:
        over += 1
        if over <= 3:
            names = [c.get('tool_name','') for c in real]
            uniq = len(set(names))
            print(f"  row {i} dom={ei.get('domain')} len={len(real)} uniq={uniq} names={names}")
print(f"total >5: {over}/{len(train)}")

print("\n=== perturbation_level raw values ===")
levels = Counter()
for i in range(len(train)):
    ei = eval(train['extra_info'].iloc[i]) if isinstance(train['extra_info'].iloc[i], str) else train['extra_info'].iloc[i]
    levels[ei.get('perturbation_level', '?')] += 1
print(dict(levels))
