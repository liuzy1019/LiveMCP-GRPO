import pandas as pd

for split, path in [('TRAIN', 'data/gen_100_50_v3/train.parquet'), ('VAL', 'data/gen_100_50_v3/val.parquet')]:
    df = pd.read_parquet(path)
    print('='*50)
    print(split, len(df), 'rows')
    print('Columns:', list(df.columns))
    for col in ['domain', 'difficulty', 'scenario']:
        if col in df.columns:
            print(col + ':', dict(df[col].value_counts()))
    for col in ['num_turns', 'conversation_rounds']:
        if col in df.columns:
            vc = dict(df[col].value_counts().sort_index())
            print(col + ':', vc, 'mean=%.2f' % df[col].mean())
    if 'oracle_program' in df.columns:
        lens = df['oracle_program'].apply(lambda x: len(x) if isinstance(x, list) else 0)
        print('oracle_program len:', dict(lens.value_counts().sort_index()), 'mean=%.2f' % lens.mean())
    for col in ['prompt', 'messages', 'conversations']:
        if col in df.columns:
            s = df[col].iloc[0]
            if isinstance(s, list):
                role = s[0].get('role') if isinstance(s[0], dict) else '?'
                print(col + '[0]: list len=' + str(len(s)) + ' role=' + str(role))
            else:
                print(col + '[0]: ' + type(s).__name__ + ' len=' + str(len(str(s))))
    rc = ['prompt', 'messages', 'conversations', 'tools', 'visible_tools',
          'oracle_program', 'success_criteria', 'task_id', 'server_name']
    print('present:', [c for c in rc if c in df.columns])
    print('MISSING:', [c for c in rc if c not in df.columns])
    if 'conversation_rounds' in df.columns:
        multi = df[df['conversation_rounds'] > 1]
        print('Multi-round:', len(multi), '/', len(df))
        if len(multi) > 0:
            row = multi.iloc[0]
            print('  sample task_id=' + str(row.get('task_id', '?')) + ' rounds=' + str(row.get('conversation_rounds', '?')))
            prog = row.get('oracle_program', [])
            if isinstance(prog, list):
                for s in prog:
                    if isinstance(s, dict):
                        print('    action=' + str(s.get('action', '?')) + ' tool=' + str(s.get('tool_name', '?')))
    print()
