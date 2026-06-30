import pandas as pd, json
train = pd.read_parquet('/tmp/big200_train.parquet')

for i in range(15):
    ei = eval(train['extra_info'].iloc[i]) if isinstance(train['extra_info'].iloc[i], str) else train['extra_info'].iloc[i]
    p = json.loads(train['prompt'].iloc[i]) if isinstance(train['prompt'].iloc[i], str) else train['prompt'].iloc[i]
    q = [m for m in p if m['role']=='user'][0]['content']
    oc_json = ei.get('oracle_calls','[]')
    oc = json.loads(oc_json) if isinstance(oc_json, str) else oc_json
    names = [c.get('tool_name','?') for c in oc]
    uniq = len(set(names))
    print(f'[{i:2d}] {ei["domain"]:15s} {ei["scenario_type"]:17s} n={len(oc):2d} unq={uniq:2d} | {q[:90]}')

# Duplicate query check
queries = []
for i in range(len(train)):
    p = json.loads(train['prompt'].iloc[i]) if isinstance(train['prompt'].iloc[i], str) else train['prompt'].iloc[i]
    q = [m for m in p if m['role']=='user'][0]['content']
    queries.append(q)

uq = set(queries)
print(f'\nUnique queries: {len(uq)}/{len(queries)}')
if len(queries) > len(uq):
    from collections import Counter
    dupes = [(q, c) for q, c in Counter(queries).items() if c > 1]
    for q, c in sorted(dupes, key=lambda x: -x[1])[:10]:
        print(f'  x{c}: "{q[:120]}"')
