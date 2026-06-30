"""smoke30 产物完整审计"""
import pandas as pd, json
from collections import Counter

train = pd.read_parquet('/tmp/smoke30_train.parquet')
val = pd.read_parquet('/tmp/smoke30_val.parquet')
print(f"train={len(train)} val={len(val)}")

def parse_ei(x):
    return eval(x) if isinstance(x, str) else x

# ====== A: PROMPT STRUCTURE ======
print("\n====== A: PROMPT STRUCTURE ======")
placeholder = 0
real_xml = 0
real_clarify = 0
real_tool_result = 0
single_user = 0
multi_user = 0
for i in range(len(train)):
    p = train['prompt'].iloc[i]
    msgs = json.loads(p) if isinstance(p, str) else list(p)
    roles = [m.get('role') for m in msgs]
    n_user = roles.count('user')
    has_asst = roles.count('assistant')
    has_tool = roles.count('tool')
    
    if n_user == 1 and has_asst == 0:
        single_user += 1
    else:
        multi_user += 1
    
    for m in msgs:
        if m.get('role') == 'assistant':
            c = m.get('content','') or ''
            if c.startswith('[The assistant'):
                placeholder += 1
            elif '<tool_call>' in c:
                real_xml += 1
            elif '<ask_clarification>' in c:
                real_clarify += 1
    if has_tool > 0:
        real_tool_result += 1

print(f"  single-user (1 round): {single_user}")
print(f"  multi-user:            {multi_user}")
print(f"  placeholder asst:      {placeholder}")
print(f"  real <tool_call> XML:  {real_xml}")
print(f"  real <ask_clarify>:    {real_clarify}")
print(f"  with tool-result msg:  {real_tool_result}")

# ====== B: ORACLE QUALITY ======
print("\n====== B: ORACLE QUALITY ======")
all_ocs = []
empty_rows = 0
overshoot = 0
dup_rows_oc = 0
for i in range(len(train)):
    ei = parse_ei(train['extra_info'].iloc[i])
    oc_raw = ei.get('oracle_calls', '[]')
    oc = json.loads(oc_raw) if isinstance(oc_raw, str) else oc_raw
    real = [c for c in oc if c.get('action', 'tool_call') != 'clarification']
    all_ocs.extend(real)
    if len(real) == 0:
        empty_rows += 1
    if len(real) > 5:
        overshoot += 1
    names = [c.get('tool_name','') for c in real]
    if len(names) != len(set(names)):
        dup_rows_oc += 1

print(f"  total real calls: {len(all_ocs)}")
print(f"  empty oracle rows: {empty_rows}/{len(train)}")
print(f"  >5 calls rows: {overshoot}/{len(train)}")
print(f"  dup tool-name rows: {dup_rows_oc}/{len(train)}")

chain_lens = []
for i in range(len(train)):
    ei = parse_ei(train['extra_info'].iloc[i])
    oc_raw = ei.get('oracle_calls', '[]')
    oc = json.loads(oc_raw) if isinstance(oc_raw, str) else oc_raw
    real = [c for c in oc if c.get('action', 'tool_call') != 'clarification']
    chain_lens.append(len(real))
cl_counter = Counter(chain_lens)
print(f"  chain lengths: {dict(sorted(cl_counter.items()))}")

# ====== C: PERTURBATION LEVEL ======
print("\n====== C: PERTURBATION LEVEL (PROVE 60/20/20) ======")
levels = Counter()
for i in range(len(train)):
    ei = parse_ei(train['extra_info'].iloc[i])
    levels[ei.get('perturbation_level', '?')] += 1
print(f"  {dict(levels)}")

# ====== D: SCENARIO MIX ======
print("\n====== D: SCENARIO MIX ======")
scen = Counter()
for i in range(len(train)):
    ei = parse_ei(train['extra_info'].iloc[i])
    scen[ei.get('scenario_type', '?')] += 1
print(f"  {dict(scen)}")

# ====== E: DUPLICATE QUERIES ======
print("\n====== E: DUPLICATE QUERIES ======")
qs = []
for i in range(len(train)):
    p = train['prompt'].iloc[i]
    msgs = json.loads(p) if isinstance(p, str) else list(p)
    um = [m['content'] for m in msgs if m.get('role')=='user']
    qs.append(um[0] if um else '')
q_count = Counter(qs)
dups = {q:n for q,n in q_count.items() if n > 1}
print(f"  unique queries: {len(set(qs))}/{len(qs)}")
print(f"  duplicate queries: {len(dups)}")

# ====== F: ASSISTANT CONTENT SAMPLES ======
print("\n====== F: ASSISTANT CONTENT SAMPLES ======")
for i in [0, 1, 3]:
    if i >= len(train): break
    ei = parse_ei(train['extra_info'].iloc[i])
    p = train['prompt'].iloc[i]
    msgs = json.loads(p) if isinstance(p, str) else list(p)
    dom = ei.get('domain','?')
    scen = ei.get('scenario_type','?')
    lvl = ei.get('perturbation_level','?')
    rounds = ei.get('conversation_rounds', 0)
    print(f"\n  row {i} domain={dom} scen={scen} level={lvl} rounds={rounds}")
    for m in msgs:
        c = m.get('content','')[:150]
        print(f"    [{m.get('role'):9s}] {c}")

# ====== G: GROUND_TRUTH VALID ======
print("\n====== G: GROUND_TRUTH ======")
gt_empty = 0
for i in range(min(5, len(train))):
    rm = train['reward_model'].iloc[i]
    rm = eval(rm) if isinstance(rm, str) else rm
    gt = rm.get('ground_truth', {})
    gt_oc = json.loads(gt.get('oracle_calls','[]')) if isinstance(gt.get('oracle_calls'), str) else gt.get('oracle_calls', [])
    ei = parse_ei(train['extra_info'].iloc[i])
    ei_oc = json.loads(ei.get('oracle_calls','[]')) if isinstance(ei.get('oracle_calls'), str) else ei.get('oracle_calls', [])
    if len(gt_oc) == 0:
        gt_empty += 1
    print(f"  row {i}: gt_oc={len(gt_oc)} ei_oc={len(ei_oc)} match={len(gt_oc)==len(ei_oc)}")
print(f"  gt_empty rows (first 5): {gt_empty}")
