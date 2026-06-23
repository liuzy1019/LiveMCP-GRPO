#!/usr/bin/env python3
"""Debug: parse multi-step call_then_call"""
import sys, re
sys.path.insert(0, '.')

s = '<tool_call>{"name":"get_weather","arguments":{"city":"Beijing"}}</tool_call>\n\n--- obs ---\n\n<tool_call>{"name":"send_msg","arguments":{"text":"Sunny"}}</tool_call>'
print(f'input contains {s.count("<tool_call>")} <tool_call> tags')

from src.reward.schemashift_reward_fn import _MULTI_TAG_PATTERN, _parse_multi_step_actions
ms = list(_MULTI_TAG_PATTERN.finditer(s))
print(f'_MULTI_TAG_PATTERN matches: {len(ms)}')
for m in ms:
    print(f'  span={m.span()} tag={m.group(1)}')

actions = _parse_multi_step_actions(s)
print(f'_parse_multi_step_actions returns {len(actions)} actions')
for a in actions:
    print(f'  type={a.action_type} name={a.tool_name} args={a.arguments}')

from src.reward.schemashift_reward_fn import compute_score

gt_cc = {
    'oracle_actions': [
        {'action_type': 'tool_call', 'tool_calls': [{'name': 'get_weather', 'arguments': {'city': 'Beijing'}}], 'match_mode': 'ordered'},
        {'action_type': 'tool_call', 'tool_calls': [{'name': 'send_msg', 'arguments': {'text': 'Sunny'}}], 'match_mode': 'ordered'},
    ],
    'episode_type': 'call_then_call',
}
r = compute_score('schemashift', s, gt_cc)
print(f'compute_score: {r["score"]}')
