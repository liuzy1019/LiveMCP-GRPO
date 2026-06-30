"""Comprehensive audit of generated parquet files against PROVE invariants."""
import pandas as pd, json, sys
from collections import Counter

def audit(parquet_paths: list[str]):
    dfs = [pd.read_parquet(p) for p in parquet_paths]
    combined = pd.concat(dfs)

    print(f"Files: {parquet_paths}")
    print(f"Total rows: {len(combined)}")
    print()

    # =====================================================================
    # 1. BASIC STATS
    # =====================================================================
    chain_lens = []
    scenarios = Counter()
    domains = Counter()
    perturbation_levels = Counter()
    n_conversation_rounds = []

    # =====================================================================
    # 2. ISSUE FLAGS
    # =====================================================================
    n_placeholder = 0          # "[The assistant" residue
    n_chain_over5 = 0          # oracle >5 real calls
    n_xml0_oc_gt0 = 0           # no <tool_call> in prompt but oracle has calls
    n_oracle_dup_tool = 0       # same tool_name in oracle (dup within task)
    n_missing_func_has_oc = 0   # scenario=missing_function but oracle_calls non-empty
    n_tool_result_mismatch = 0  # tool_result count != tool_call count (L5)
    n_tool_result_orphan = 0    # orphan tool results

    issues = []

    for i in range(len(combined)):
        ei = eval(combined['extra_info'].iloc[i]) if isinstance(combined['extra_info'].iloc[i], str) else combined['extra_info'].iloc[i]
        oc_raw = ei.get('oracle_calls', '[]')
        oc = json.loads(oc_raw) if isinstance(oc_raw, str) else oc_raw
        real_oc = [c for c in oc if c.get('action', 'tool_call') != 'clarification']

        p = combined['prompt'].iloc[i]
        msgs = json.loads(p) if isinstance(p, str) else list(p)

        # -- basic counters --
        chain_lens.append(len(real_oc))
        domains[ei.get('domain', '?')] += 1
        scenarios[ei.get('scenario_type', '?')] += 1
        perturbation_levels[ei.get('perturbation_level', '?')] += 1
        n_conversation_rounds.append(int(ei.get('conversation_rounds', 1)))

        # -- count XML tags in assistant messages --
        assistant_msgs = [m for m in msgs if m.get('role') == 'assistant']
        xml_tool_call_count = sum(
            (m.get('content', '') or '').count('<tool_call>')
            for m in assistant_msgs
        )
        tool_msgs = [m for m in msgs if m.get('role') == 'tool']
        tool_result_count = len(tool_msgs)

        # -- collect tool_call names from prompt XML --
        prompt_tool_names = []
        import re
        for m in assistant_msgs:
            content = m.get('content', '') or ''
            found = re.findall(r'"name":\s*"([^"]+)"', content)
            prompt_tool_names.extend(found)

        # -- check placeholder --
        has_ph = any('[The assistant' in (m.get('content', '') or '') for m in assistant_msgs)
        has_fa = any('<final_answer>' in (m.get('content', '') or '') for m in assistant_msgs)
        has_re = any('<report_error>' in (m.get('content', '') or '') for m in assistant_msgs)

        # -- check duplicate tool names in oracle --
        oracle_names = [c.get('tool_name') for c in real_oc]
        dup_names = {n: c for n, c in Counter(oracle_names).items() if c > 1}

        row_issues = []

        if has_ph:
            n_placeholder += 1
            row_issues.append('PLACEHOLDER')

        if len(real_oc) > 5:
            n_chain_over5 += 1
            row_issues.append(f'CHAIN={len(real_oc)}')

        if xml_tool_call_count == 0 and len(real_oc) > 0:
            n_xml0_oc_gt0 += 1
            row_issues.append('XML=0/HAS_OC')

        if dup_names:
            n_oracle_dup_tool += 1
            row_issues.append(f'DUP={dup_names}')

        if ei.get('scenario_type') == 'missing_function' and len(real_oc) > 0:
            n_missing_func_has_oc += 1
            row_issues.append('MF_HAS_OC')

        # -- check tool_result vs tool_call alignment (L5) --
        if xml_tool_call_count > 0 and tool_result_count > xml_tool_call_count:
            n_tool_result_mismatch += 1
            row_issues.append(f'T_RES>{xml_tool_call_count}')

        # -- check orphan tool results (no corresponding tool_call name) --
        if tool_result_count > 0:
            for j, tc_name in enumerate(prompt_tool_names[:tool_result_count]):
                pass  # aligned
            if tool_result_count > len(prompt_tool_names):
                n_tool_result_orphan += 1
                row_issues.append(f'ORPHAN_T={tool_result_count-len(prompt_tool_names)}')

        if row_issues:
            issues.append((i, ei.get('domain', '?'), ei.get('scenario_type', '?'),
                          ei.get('perturbation_level', '?'),
                          ei.get('conversation_rounds', '?'),
                          len(real_oc), xml_tool_call_count, tool_result_count,
                          len(prompt_tool_names[:10]), row_issues))

    # =====================================================================
    # PRINT SUMMARY
    # =====================================================================
    cl = Counter(chain_lens)
    print(f"{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    print(f"Domains:              {dict(domains)}")
    print(f"Scenarios:            {dict(scenarios)}")
    print(f"Perturbation levels:  {dict(perturbation_levels)}")
    print(f"Chain lengths: min={min(chain_lens)} max={max(chain_lens)} avg={sum(chain_lens)/len(chain_lens):.1f}")
    print(f"Chain distribution:   {dict(sorted(cl.items()))}")
    print(f"Conv rounds:          {dict(sorted(Counter(n_conversation_rounds).items()))}")
    print()
    print(f"{'='*80}")
    print(f"ISSUE COUNTS")
    print(f"{'='*80}")
    print(f"  PLACEHOLDER (prompt residue):           {n_placeholder}/{len(combined)}")
    print(f"  CHAIN > 5:                              {n_chain_over5}/{len(combined)}")
    print(f"  XML call=0 but oracle has calls:        {n_xml0_oc_gt0}/{len(combined)}")
    print(f"  Dup tool names in oracle:               {n_oracle_dup_tool}/{len(combined)}")
    print(f"  missing_func has oracle calls:          {n_missing_func_has_oc}/{len(combined)}")
    print(f"  tool_results > tool_calls (L5 viol):    {n_tool_result_mismatch}/{len(combined)}")
    print(f"  Orphan tool results:                    {n_tool_result_orphan}/{len(combined)}")
    print()

    if issues:
        print(f"{'='*80}")
        print(f"ISSUE DETAILS ({len(issues)} rows)")
        print(f"{'='*80}")
        for row in issues:
            (idx, dom, scen, lvl, rounds, oc_len, xml, t_res, t_names, flags) = row
            print(f"  [{idx:2d}] {dom:12s} {scen:16s} {lvl:10s} r={rounds} oc={oc_len} xml={xml} tr={t_res} tn={t_names} | {', '.join(flags)}")

    # =====================================================================
    # 3. SPOT CHECK: sample prompts for correctness
    # =====================================================================
    print()
    print(f"{'='*80}")
    print(f"SPOT CHECK (first 3 normal + first 2 missing_function)")
    print(f"{'='*80}")
    normal_count = 0
    mf_count = 0
    for i in range(len(combined)):
        ei = eval(combined['extra_info'].iloc[i]) if isinstance(combined['extra_info'].iloc[i], str) else combined['extra_info'].iloc[i]
        scen = ei.get('scenario_type', '?')
        oc_raw = ei.get('oracle_calls', '[]')
        oc = json.loads(oc_raw) if isinstance(oc_raw, str) else oc_raw
        real_oc = [c for c in oc if c.get('action', 'tool_call') != 'clarification']

        if scen in ('task_planner', 'distractor') and normal_count < 3:
            normal_count += 1
            p = combined['prompt'].iloc[i]
            msgs = json.loads(p) if isinstance(p, str) else list(p)
            print(f"\n--- [{i}] {ei.get('domain')} {scen} oc={len(real_oc)} ---")
            for m in msgs:
                c = str(m.get('content', ''))
                role = m.get('role', '?')
                if role == 'assistant':
                    print(f"  [{role}] {c[:200]}{'...' if len(c)>200 else ''}")
                elif role == 'tool':
                    print(f"  [{role}] {c[:120]}{'...' if len(c)>120 else ''}")
                elif role == 'user':
                    print(f"  [{role}] {c[:120]}{'...' if len(c)>120 else ''}")
        elif scen == 'missing_function' and mf_count < 2:
            mf_count += 1
            p = combined['prompt'].iloc[i]
            msgs = json.loads(p) if isinstance(p, str) else list(p)
            print(f"\n--- [{i}] {ei.get('domain')} {scen} oc={len(real_oc)} ---")
            for m in msgs:
                c = str(m.get('content', ''))
                role = m.get('role', '?')
                if role == 'assistant':
                    print(f"  [{role}] {c[:200]}{'...' if len(c)>200 else ''}")
                elif role == 'tool':
                    print(f"  [{role}] {c[:120]}{'...' if len(c)>120 else ''}")
                elif role == 'user':
                    print(f"  [{role}] {c[:120]}{'...' if len(c)>120 else ''}")

    return issues

if __name__ == '__main__':
    paths = sys.argv[1:] if len(sys.argv) > 1 else ['/tmp/smoke30c_train.parquet', '/tmp/smoke30c_val.parquet']
    audit(paths)
