#!/usr/bin/env python3
"""系统性验证所有 reward 场景的取值。"""
import sys, json
sys.path.insert(0, '.')
from src.reward.schemashift_reward_fn import compute_score

FAILS = 0
def T(label, result, exp_score=None, exp_exact=None, note=''):
    global FAILS
    s = result['score']; e = result['exact_success']
    ok_s = '' if exp_score is None else ('✅' if abs(s - exp_score) < 1e-6 else f'❌exp={exp_score:.4f}')
    ok_e = '' if exp_exact is None else ('✅' if e == exp_exact else f'❌exp={exp_exact}')
    if '❌' in ok_s or '❌' in ok_e:
        FAILS += 1
    print(f'{label:55s} score={s:+.4f} exact={e} {ok_s} {ok_e}')
    if note:
        print(f'  ↳ {note}')
    return result

gt = {
    'oracle_actions': [{'action_type': 'tool_call',
        'tool_calls': [{'name': 'get_weather', 'arguments': {'city': 'Beijing'}}],
        'match_mode': 'ordered'}],
    'episode_type': 'call_only',
}
sol_exact = '<tool_call>{"name": "get_weather", "arguments": {"city": "Beijing"}}</tool_call>'

# ═══ 单步 tool_call ═══
T('1.  tool_call EXACT', compute_score('schemashift', sol_exact, gt), 1.05, True)
# 错误工具名：ordered matching 下仍然配对，schema/keys/values 得分（只有 tool_selection=0）
# partial = (0.10+0.15+0+0.20+0.25)=0.70 → 0.3*0.70=0.21
T('2.  tool_call 错误工具名', compute_score('schemashift',
    '<tool_call>{"name":"wrong","arguments":{"city":"Beijing"}}</tool_call>', gt),
    0.21, False, 'ordered匹配仍配对→schema/keys/values得分')
T('3.  tool_call 错误参数key', compute_score('schemashift',
    '<tool_call>{"name":"get_weather","arguments":{"location":"Beijing"}}</tool_call>', gt),
    0.12, False)
T('4.  tool_call 错误参数值', compute_score('schemashift',
    '<tool_call>{"name":"get_weather","arguments":{"city":"Shanghai"}}</tool_call>', gt),
    0.225, False)
# "random text" ≥10 chars → parser fallback 为 final_answer → action_type_mismatch → 0.03
T('5.  无格式输出≥10 chars', compute_score('schemashift', 'random text', gt),
    0.03, False, '≥10 chars → plain_text_final_answer → mismatch → 0.03')
T('6.  tool_call arguments非dict', compute_score('schemashift',
    '<tool_call>{"name":"get_weather","arguments":"Beijing"}</tool_call>', gt),
    0.12, False, 'has_invalid_args + exact→False → schema_valid=0')
T('7.  oracle=tc model=final_answer', compute_score('schemashift', '<final_answer>done</final_answer>', gt),
    0.03, False)
T('8.  oracle=tc model=report_error', compute_score('schemashift', '<report_error>err</report_error>', gt),
    0.03, False)

# ═══ final_answer ═══
fa_gt = {'oracle_actions': [{'action_type': 'final_answer', 'final_answer': 'Sunny warm today'}], 'episode_type': 'call_only'}
T('9.  final_answer EXACT', compute_score('schemashift', '<final_answer>Sunny warm today</final_answer>', fa_gt), 1.05, True)
T('10. final_answer keyword subset', compute_score('schemashift', '<final_answer>Sunny warm today 25C</final_answer>', fa_gt), 0.27, False)
# "Rainy cold" vs "Sunny warm today": 0 overlap → final_answer_match=0 → partial=0.50 → 0.15
T('11. final_answer 无重叠', compute_score('schemashift', '<final_answer>Rainy cold</final_answer>', fa_gt),
    0.15, False, '0 overlap → match=0 → partial=0.3*1+0.2*1+0.5*0=0.50 → 0.15')
T('12. oracle=fa model=tool_call', compute_score('schemashift', sol_exact, fa_gt), 0.03, False)

# ═══ report_error / ask_clarification ═══
re_gt = {'oracle_actions': [{'action_type': 'report_error'}], 'episode_type': 'call_only'}
T('13. report_error match', compute_score('schemashift', '<report_error>conn failed</report_error>', re_gt), 1.05, True)
T('14. oracle=re model=tool_call', compute_score('schemashift', sol_exact, re_gt), 0.03, False)

ac_gt = {'oracle_actions': [{'action_type': 'ask_clarification'}], 'episode_type': 'call_only'}
T('15. ask_clarification match', compute_score('schemashift', '<ask_clarification>which city?</ask_clarification>', ac_gt), 1.05, True)

# ═══ 边界/异常 ═══
T('16. EXACT + extra tag', compute_score('schemashift', sol_exact + ' <final_answer>done</final_answer>', gt), 0.25, False)
T('17. ground_truth 非法JSON', compute_score('schemashift', sol_exact, 'not-json', {}), 0.0, False)
T('18. oracle_actions为空', compute_score('schemashift', sol_exact, {'oracle_actions': [], 'episode_type': 'call_only'}), 0.0, False)

# ═══ 多步 call_then_final ═══
gt_m = {
    'oracle_actions': [
        {'action_type': 'tool_call', 'tool_calls': [{'name': 'get_weather', 'arguments': {'city': 'Beijing'}}], 'match_mode': 'ordered'},
        {'action_type': 'final_answer', 'final_answer': 'Sunny'},
    ],
    'episode_type': 'call_then_final',
}
sol_m = '<tool_call>{"name":"get_weather","arguments":{"city":"Beijing"}}</tool_call>\n\n--- obs ---\n\n<final_answer>Sunny</final_answer>'
T('19. 多步 EXACT (1.05+0.2traj)', compute_score('schemashift', sol_m, gt_m), 1.25, True)
T('20. 多步 step1正确 step2错误', compute_score('schemashift', sol_m.replace('Sunny</final_answer>', 'Rainy</final_answer>'), gt_m), 0.69, False)
T('21. 多步 跳过tool_call给fa', compute_score('schemashift', '<final_answer>Sunny</final_answer>', gt_m), -0.132, False)
# same_turn: 两tag之间0字符
sol_nosep = '<tool_call>{"name":"get_weather","arguments":{"city":"Beijing"}}</tool_call><final_answer>Sunny</final_answer>'
T('22. same_turn_violation (0间隔)', compute_score('schemashift', sol_nosep, gt_m), 0.90, False)
# 3 actions, 最后两个 final_answer 间 "\n\n"<5 chars → also triggers same_turn_violation
sol_extra = sol_m + '\n\n<final_answer>Done</final_answer>'
T('23. extra step + same_turn触发', compute_score('schemashift', sol_extra, gt_m),
    0.85, False, '第2-3个tag间\\n\\n<5chars→same_turn=-0.15+efficiency=-0.05')

# ═══ name_map / 边界 ═══
T('24. name_map扰动映射', compute_score('schemashift',
    '<tool_call>{"name":"weather_retrieve","arguments":{"city":"Beijing"}}</tool_call>',
    gt, {'name_map': {'weather_retrieve': 'get_weather'}, 'perturbation_level': 'mild'}), 1.05, True)
T('25. 空输出', compute_score('schemashift', '', gt), 0.0, False)
T('26. ground_truth是JSON字符串', compute_score('schemashift', sol_exact, json.dumps(gt)), 1.05, True)

# ═══ 多步 call_then_call ═══
gt_cc = {
    'oracle_actions': [
        {'action_type': 'tool_call', 'tool_calls': [{'name': 'get_weather', 'arguments': {'city': 'Beijing'}}], 'match_mode': 'ordered'},
        {'action_type': 'tool_call', 'tool_calls': [{'name': 'send_msg', 'arguments': {'text': 'Sunny'}}], 'match_mode': 'ordered'},
    ],
    'episode_type': 'call_then_call',
}
sol_cc = '<tool_call>{"name":"get_weather","arguments":{"city":"Beijing"}}</tool_call>\n\n--- obs ---\n\n<tool_call>{"name":"send_msg","arguments":{"text":"Sunny"}}</tool_call>'
T('27. 多步 call_then_call EXACT', compute_score('schemashift', sol_cc, gt_cc), 1.25, True)
T('28. 多步 call_then_call step2错误', compute_score('schemashift', sol_cc.replace('Sunny', 'Rainy'), gt_cc), 0.72, False)

print(f'\n{"✅ 全部 28/28 通过" if FAILS == 0 else f"❌ {FAILS} 个失败"}')
