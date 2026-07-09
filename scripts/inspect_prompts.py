"""Deep prompt inspection for PROVE data."""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

parser = argparse.ArgumentParser(description="PROVE prompt inspection and quality checks.")
parser.add_argument("--train", default="data/train.parquet",
                    help="Path to train parquet (default: data/train.parquet).")
parser.add_argument("--val", default="data/val.parquet",
                    help="Path to val parquet (default: data/val.parquet).")
args = parser.parse_args()

train_path = Path(args.train)
val_path = Path(args.val)

if not train_path.exists():
    print(f"ERROR: train file not found: {train_path}", file=sys.stderr)
    sys.exit(1)
if not val_path.exists():
    print(f"ERROR: val file not found: {val_path}", file=sys.stderr)
    sys.exit(1)

t = pd.read_parquet(train_path)
v = pd.read_parquet(val_path)
print(f'Train={len(t)}  Val={len(v)}  Total={len(t)+len(v)}\n')


def get_prompt(row):
    p = row['prompt']
    return json.loads(p) if isinstance(p, str) else list(p)


def get_extra(row):
    ei = row['extra_info']
    return json.loads(ei) if isinstance(ei, str) else ei


def get_rm(row):
    rm = row['reward_model']
    return json.loads(rm) if isinstance(rm, str) else rm


def parse_oracle(gt):
    """oracle_calls may be a string or list"""
    oc = gt.get('oracle_calls', [])
    if isinstance(oc, str):
        oc = json.loads(oc)
    return oc


# ============================================================
# 1. Per-scenario sample
# ============================================================
scenarios = ['task_planner', 'missing_function', 'distractor', 'irrelevant']
for scen in scenarios:
    found = False
    for i in range(len(t)):
        ei = get_extra(t.iloc[i])
        if ei.get('scenario_type') == scen:
            print('=' * 70)
            print(f'[{scen}] Row {i}: domain={ei["domain"]} lvl={ei["perturbation_level"]} rounds={ei.get("conversation_rounds")}')
            p = get_prompt(t.iloc[i])
            print(f'Prompt: {len(p)} messages')
            for j, m in enumerate(p):
                role = m['role']
                content = m.get('content', '') or ''
                has_tc = '<tool_call>' in content if role != 'system' else False
                snippet = content[:150].replace('\n', '\\n')
                print(f'  [{j:2d}] {role:10s} len={len(content):5d} tc={has_tc}: {snippet}')
            rm = get_rm(t.iloc[i])
            gt = rm['ground_truth']
            oc = parse_oracle(gt)
            print(f'  oracle_calls: {len(oc)}')
            for o in oc[:5]:
                a = o.get('action', 'tool_call')
                tname = o.get('tool_name', '?')
                args_dict = o.get('arguments', {})
                args_str = ', '.join(f'{k}={v}' for k, v in list(args_dict.items())[:3])
                print(f'    [{a}] {tname}({args_str})')
            print(f'  required_tools: {gt.get("required_tools",[])}')
            found = True
            break

# ============================================================
# 2. Global stats
# ============================================================
print('\n' + '=' * 70)
print('GLOBAL STATS')
all_t = pd.concat([t, v])
tc_dist = Counter()
clarify_n = 0
report_err_n = 0
total_turns = 0
tool_usage = Counter()

for i in range(len(all_t)):
    p = get_prompt(all_t.iloc[i])
    for m in p:
        role = m['role']
        content = m.get('content', '') or ''
        if role == 'assistant':
            total_turns += 1
            tc = content.count('<tool_call>')
            tc_dist[tc] += 1
            if '<ask_clarification>' in content:
                clarify_n += 1
            if '<report_error>' in content:
                report_err_n += 1
            names = re.findall(r'\{[^}]*"name"\s*:\s*"([^"]+)"', content)
            for tn in names:
                tool_usage[tn] += 1

print(f'  Assistant turns: {total_turns}')
print(f'  Tool calls per turn: {dict(sorted(tc_dist.items()))}')
print(f'  Clarifications: {clarify_n}')
print(f'  Report errors: {report_err_n}')
print(f'  Unique tools used: {len(tool_usage)}')
print(f'  Top 15 tools:')
for tn, cnt in tool_usage.most_common(15):
    print(f'    {tn}: {cnt}')

# ============================================================
# 3. Check for quality issues
# ============================================================
print('\n' + '=' * 70)
print('QUALITY CHECKS')

empty_queries = 0
for i in range(len(all_t)):
    p = get_prompt(all_t.iloc[i])
    for m in p:
        if m['role'] == 'user' and not (m.get('content', '') or '').strip():
            empty_queries += 1
            break
print(f'  Empty user queries: {empty_queries}')

tool_result_len_dist = Counter()
empty_results = 0
for i in range(len(all_t)):
    p = get_prompt(all_t.iloc[i])
    for m in p:
        if m['role'] == 'tool':
            content = m.get('content', '') or ''
            if not content.strip():
                empty_results += 1
            tool_result_len_dist[len(content)] += 1
p50_elements = sorted(tool_result_len_dist.elements())
p50 = p50_elements[len(p50_elements) // 2] if p50_elements else 0
print(f'  Empty tool results: {empty_results}')
print(f'  Tool result length percentiles: p50={p50}')

domains_per_row = []
for i in range(len(all_t)):
    ei = get_extra(all_t.iloc[i])
    domains_per_row.append(ei.get('domain', '?'))
domain_counts = Counter(domains_per_row)
print(f'  Domain coverage: {dict(sorted(domain_counts.items()))}')
