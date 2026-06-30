"""快速验证 big200 中 oracle_calls_per_round 字段是否落盘"""
import pandas as pd, json

train = pd.read_parquet('/tmp/big200_train.parquet')

print('=== columns ===')
print(train.columns.tolist())

print('\n=== extra_info keys (row 1) ===')
ei = eval(train['extra_info'].iloc[1]) if isinstance(train['extra_info'].iloc[1], str) else train['extra_info'].iloc[1]
print(sorted(ei.keys()))

print('\n=== row 1 prompt roles ===')
p = train['prompt'].iloc[1]
msgs = json.loads(p) if isinstance(p, str) else list(p)
print('roles:', [m.get('role') for m in msgs])
print('any tool role?', any(m.get('role')=='tool' for m in msgs))

print('\n=== row 1 oracle_calls preview ===')
oc_raw = ei.get('oracle_calls', '[]')
oc = json.loads(oc_raw) if isinstance(oc_raw, str) else oc_raw
for c in oc[:5]:
    print(' ', c.get('action'), '|', c.get('tool_name'), '|', c.get('arguments'))

print('\n=== assistant content row 1 ===')
for m in msgs:
    if m.get('role') == 'assistant':
        print(repr(m.get('content','')[:200]))
        break
