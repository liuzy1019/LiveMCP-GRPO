"""Debug oracle_calls format and prompt structure."""
import pandas as pd, json, numpy as np

train = pd.read_parquet('/tmp/big200_train.parquet')
ei = eval(train['extra_info'].iloc[0]) if isinstance(train['extra_info'].iloc[0], str) else train['extra_info'].iloc[0]
oc = ei['oracle_calls']
print("type(oc):", type(oc))
print("type(oc[0]):", type(oc[0]))
s0 = str(oc[0])
print("oc[0] len:", len(s0))
print("oc[0] preview:", s0[:300])
print()

# Check if it's JSON
try:
    parsed = json.loads(s0)
    print("JSON parsed OK, type:", type(parsed))
    if isinstance(parsed, list):
        print("list len:", len(parsed))
        if len(parsed) > 0:
            print("first:", json.dumps(parsed[0], ensure_ascii=False)[:300])
except:
    print("Not JSON")

# Prompt detail
p = train['prompt'].iloc[0]
msgs = json.loads(p) if isinstance(p, str) else p
print("\nPrompt messages:")
for i, m in enumerate(msgs):
    keys = list(m.keys())
    tc = m.get('tool_calls', None)
    print(f"  [{i}] role={m['role']} keys={keys}")
    if tc is not None:
        print(f"       tool_calls count: {len(tc)}")
        if len(tc) > 0:
            print(f"       first call: {json.dumps(tc[0], ensure_ascii=False)[:200]}")
