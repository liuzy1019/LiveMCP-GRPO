#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["pandas", "pyarrow"]
# ///
"""Complete adversarial review — fresh run on latest generated data.
9 dimensions: outer dict, JSON serialization, argument inflation,
reward integration, R_arg viability, edge cases, HF round-trip,
chain quality (multi-step), failure analysis."""

import pandas as pd, json, sys
from collections import Counter
sys.path.insert(0, '/mnt/data2/liuzhanyi/livemcp-grpo')
from src.reward.oval_reward_fn import _build_task_dict

TRAIN = '/tmp/fresh_review_train.parquet'
VAL = '/tmp/fresh_review_val.parquet'

for label, path in [("TRAIN", TRAIN), ("VAL", VAL)]:
    try:
        df = pd.read_parquet(path)
    except FileNotFoundError:
        print(f"⚠ {label}: file not found (generation may still be running)")
        continue
        
    print(f"\n{'='*60}")
    print(f"REVIEW: {label} — {len(df)} rows, {dict(df['scenario_type'].value_counts())}")
    print(f"{'='*60}")
    errs = []

    # 1: Outer dict
    d1 = sum(1 for i in range(len(df)) if not isinstance(df.iloc[i]['reward_model'], dict) or not isinstance(df.iloc[i]['extra_info'], dict))
    print(f"  D1 outer_dict:      {'✓' if d1==0 else '✗'} ({d1})")
    [errs.append('D1') for _ in range(d1)]

    # 2: JSON string
    d2 = 0
    for i in range(len(df)):
        ei = df.iloc[i]['extra_info']
        gt = df.iloc[i]['reward_model']['ground_truth']
        if not isinstance(ei['oracle_calls'], str): d2 += 1; errs.append(f"D2[{i}]")
        if not isinstance(gt['oracle_calls'], str): d2 += 1; errs.append(f"D2[{i}]")
    print(f"  D2 json_string:     {'✓' if d2==0 else '✗'} ({d2})")

    # 3: No inflation
    d3 = 0; n_ocs = 0; arg_lens = []
    for i in range(len(df)):
        ei = df.iloc[i]['extra_info']
        ocs = json.loads(ei['oracle_calls'])
        for oc in ocs:
            n_ocs += 1
            a = oc['arguments']
            arg_lens.append(len(a))
            n = sum(1 for v in a.values() if v is None)
            if n > 0: d3 += 1; errs.append(f"D3[{i}] {oc['tool_name']}: {n} null")
    avg_args = sum(arg_lens) / len(arg_lens) if arg_lens else 0
    print(f"  D3 no_inflation:    {'✓' if d3==0 else '✗'} ({d3}/{n_ocs}, avg {avg_args:.1f} keys/call)")

    # 4: Reward integration
    d4 = 0
    for i in range(len(df)):
        rm = df.iloc[i]['reward_model']; ei = dict(df.iloc[i]['extra_info'])
        gt = rm['ground_truth']
        for k in ('oracle_calls','success_criteria','required_tools'):
            if k not in ei and k in gt: ei[k] = gt[k]
        try:
            td = _build_task_dict(ei)
            for rtc in td.get('required_tool_calls', []):
                a = rtc.get('arguments', {})
                if sum(1 for v in a.values() if v is None) > 0:
                    d4 += 1; errs.append(f"D4[{i}] {rtc['tool_name']}")
            for rk in ('task_id','required_tool_calls','success_criteria','allowed_terminal_actions'):
                if rk not in td: d4 += 1; errs.append(f"D4[{i}] missing {rk}")
        except Exception as e:
            d4 += 1; errs.append(f"D4[{i}] {e}")
    print(f"  D4 reward_integ:    {'✓' if d4==0 else '✗'} ({d4})")

    # 5: R_arg viability
    d5 = 0
    for i in range(len(df)):
        rm = df.iloc[i]['reward_model']; ei = dict(df.iloc[i]['extra_info'])
        gt = rm['ground_truth']
        for k in ('oracle_calls','success_criteria','required_tools'):
            if k not in ei and k in gt: ei[k] = gt[k]
        try:
            td = _build_task_dict(ei)
            for rtc in td.get('required_tool_calls', []):
                a = rtc.get('arguments', {})
                if a and sum(1 for v in a.values() if v is None) > 0:
                    d5 += 1; errs.append(f"D5[{i}] {rtc['tool_name']}: null in gt")
        except Exception as e:
            d5 += 1; errs.append(f"D5[{i}] {e}")
    print(f"  D5 r_arg_viable:    {'✓' if d5==0 else '✗'} ({d5})")

    # 6: Edge cases (run once on train only)
    if label == "TRAIN":
        d6 = 0
        tests = [
            ("empty", {"oracle_calls":"[]","task_id":"t1","domain":"t","required_tools":[]}, []),
            ("clarify", {"oracle_calls":'[{"action":"clarification","tool_name":"ask_clarification","arguments":{"question":"x"}}]',"task_id":"t2","domain":"t","required_tools":[]}, []),
            ("legacy", {"oracle_calls":[{"tool_name":"get","arguments":{"k":"v"}}],"task_id":"t3","domain":"t","required_tools":["get"]}, [{"tool_name":"get","arguments":{"k":"v"}}]),
            ("none", {"oracle_calls":None,"task_id":"t4","domain":"t","required_tools":[]}, []),
            ("bad_json", {"oracle_calls":"not-json{{","task_id":"t5","domain":"t","required_tools":[]}, []),
        ]
        for name, ei, expected in tests:
            try:
                td = _build_task_dict(ei)
                actual = td.get('required_tool_calls', [])
                if expected == [] and actual != []: d6 += 1; errs.append(f"D6 {name}")
                elif expected != [] and actual != expected: d6 += 1; errs.append(f"D6 {name}")
            except Exception as e: d6 += 1; errs.append(f"D6 {name} {e}")
        print(f"  D6 edge_cases:      {'✓' if d6==0 else '✗'} ({d6})")

    # 7: HF Dataset round-trip
    try:
        from datasets import Dataset
        ds = Dataset.from_parquet(path)
        d7 = 0
        for i in range(min(3, len(ds))):
            r = ds[i]
            if not isinstance(r['reward_model'], dict): d7 += 1
            if not isinstance(r['extra_info'], dict): d7 += 1
            if not isinstance(r['extra_info']['oracle_calls'], str): d7 += 1
            ocs = json.loads(r['extra_info']['oracle_calls'])
            for oc in ocs:
                if sum(1 for v in oc['arguments'].values() if v is None) > 0: d7 += 1
        print(f"  D7 hf_dataset:      {'✓' if d7==0 else '✗'} ({d7}, {len(ds)} rows)")
    except ImportError:
        print("  D7 hf_dataset:      SKIP")
    except Exception as e:
        print(f"  D7 hf_dataset:      ✗ {e}")

    # 8: Chain quality
    chain_lens = []
    domain_chains = Counter()
    multi_step_domains = set()
    for i in range(len(df)):
        ei = df.iloc[i]['extra_info']
        domain = ei.get('domain','?')
        ocs = json.loads(ei['oracle_calls'])
        real = [oc for oc in ocs if oc.get('action') != 'clarification']
        cl = len(real)
        chain_lens.append(cl)
        domain_chains[(domain, cl)] += 1
        if cl >= 2: multi_step_domains.add(domain)
    
    cl_dist = Counter(chain_lens)
    print(f"\n  D8 chain_quality:")
    print(f"     distribution: {dict(sorted(cl_dist.items()))}")
    print(f"     avg: {sum(chain_lens)/len(chain_lens):.1f}")
    print(f"     multi-step (>=2): {sum(1 for l in chain_lens if l >= 2)}/{len(chain_lens)}")
    print(f"     domains with multi-step: {sorted(multi_step_domains)}")
    
    # Show example chains
    for i in range(len(df)):
        ei = df.iloc[i]['extra_info']
        ocs = json.loads(ei['oracle_calls'])
        real = [oc for oc in ocs if oc.get('action') != 'clarification']
        if len(real) >= 3:
            domain = ei.get('domain','?')
            tools = [oc['tool_name'] for oc in real]
            query = json.loads(df.iloc[i]['prompt'])[1]['content'][:150]
            print(f"     eg [{domain}] {tools}")
            print(f"         \"{query}\"")
            if sum(1 for l in chain_lens if l >= 3) >= 2:  # show at most 2
                break

    # 9: Failure rate
    fail_count = sum(1 for l in chain_lens if l == 0)
    print(f"  D9 failure_rate:    {fail_count}/{len(chain_lens)} ({100*fail_count/len(chain_lens):.0f}% 0-call)")
    
    total = d1+d2+d3+d4+d5+(d6 if label=='TRAIN' else 0)
    print(f"\n  OVERALL {'PASS' if total==0 else 'FAIL'} (D1:{d1} D2:{d2} D3:{d3} D4:{d4} D5:{d5})")
    if errs:
        for e in errs[:10]: print(f"    err: {e}")
