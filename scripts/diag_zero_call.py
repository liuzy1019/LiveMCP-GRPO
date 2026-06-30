"""定位 task_planner 0-call 任务的实际成因"""
import pandas as pd, json
from collections import Counter

train = pd.read_parquet('/tmp/big200_train.parquet')

print("=== task_planner 0-call rows 详情 ===")
for i in range(len(train)):
    ei = eval(train['extra_info'].iloc[i]) if isinstance(train['extra_info'].iloc[i], str) else train['extra_info'].iloc[i]
    oc_raw = ei.get('oracle_calls', '[]')
    oc = json.loads(oc_raw) if isinstance(oc_raw, str) else oc_raw
    real = [c for c in oc if (c.get('action') or 'tool_call') != 'clarification']
    if len(real) == 0 and ei.get('scenario_type') == 'task_planner':
        p = train['prompt'].iloc[i]
        msgs = json.loads(p) if isinstance(p, str) else list(p)
        ums = [m['content'] for m in msgs if m.get('role') == 'user']
        print(f"  row {i} domain={ei['domain']} hidden={ei.get('hidden_tools')} "
              f"required={ei.get('required_tools')[:3]} "
              f"missing_func={ei.get('has_missing_function')}")
        print(f"    query: {ums[0][:120]}")
        # 是否 missing_function 把它清空了？
