"""验证 replay 和 reward 的匹配一致性。

覆盖历史发现的 replay/reward 匹配不一致场景：
- list_numeric_string
- list_dict_key_order
- nested_list
- flat_enum_perturbed
- flat_enum_original
"""
import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'verl')

from src.agent_loop.schemashift_replay_loop import _match_tool_call
from src.reward.component_reward import ComponentReward, OracleAction, SampleMetadata

reward_fn = ComponentReward()


def test_list_numeric_string():
    """[1, '2', 3] vs ['1', '2', '3'] — 数值/字符串归一化。"""
    oracle = {'tool_calls': [{'name': 'fn', 'arguments': {'ids': [1, '2', 3]}}], 'match_mode': 'ordered'}
    model = [{'name': 'fn', 'arguments': {'ids': ['1', '2', '3']}}]
    replay = _match_tool_call(model, oracle)
    oracle_obj = OracleAction(action_type='tool_call', tool_calls=[{'name': 'fn', 'arguments': {'ids': [1, '2', 3]}}], match_mode='ordered')
    r = reward_fn.compute('<tool_call>{"name": "fn", "arguments": {"ids": ["1", "2", "3"]}}</tool_call>', oracle_obj, SampleMetadata())
    assert replay == r.exact_success, f'list_numeric_string: replay={replay} reward={r.exact_success}'
    print(f'  list_numeric_string: replay={replay} reward={r.exact_success} OK')


def test_list_dict_key_order():
    """[{"a":1,"b":2}] vs [{"b":2,"a":1}] — dict key 顺序无关。"""
    oracle = {'tool_calls': [{'name': 'fn', 'arguments': {'data': [{'a': 1, 'b': 2}]}}], 'match_mode': 'ordered'}
    model = [{'name': 'fn', 'arguments': {'data': [{'b': 2, 'a': 1}]}}]
    replay = _match_tool_call(model, oracle)
    oracle_obj = OracleAction(action_type='tool_call', tool_calls=[{'name': 'fn', 'arguments': {'data': [{'a': 1, 'b': 2}]}}], match_mode='ordered')
    r = reward_fn.compute('<tool_call>{"name": "fn", "arguments": {"data": [{"b": 2, "a": 1}]}}</tool_call>', oracle_obj, SampleMetadata())
    assert replay == r.exact_success, f'list_dict_key_order: replay={replay} reward={r.exact_success}'
    print(f'  list_dict_key_order: replay={replay} reward={r.exact_success} OK')


def test_nested_list():
    """[[1,2],[3,4]] vs [[3,4],[1,2]] — 嵌套 list 无序匹配。"""
    oracle = {'tool_calls': [{'name': 'fn', 'arguments': {'matrix': [[1, 2], [3, 4]]}}], 'match_mode': 'ordered'}
    model = [{'name': 'fn', 'arguments': {'matrix': [[3, 4], [1, 2]]}}]
    replay = _match_tool_call(model, oracle)
    oracle_obj = OracleAction(action_type='tool_call', tool_calls=[{'name': 'fn', 'arguments': {'matrix': [[1, 2], [3, 4]]}}], match_mode='ordered')
    r = reward_fn.compute('<tool_call>{"name": "fn", "arguments": {"matrix": [[3, 4], [1, 2]]}}</tool_call>', oracle_obj, SampleMetadata())
    assert replay == r.exact_success, f'nested_list: replay={replay} reward={r.exact_success}'
    print(f'  nested_list: replay={replay} reward={r.exact_success} OK')


def test_flat_enum_perturbed():
    """flat enum map: perturbed value 应被接受。"""
    flat_enum = {'centigrade': 'celsius', 'fahrenheit_scale': 'fahrenheit'}
    oracle = {'tool_calls': [{'name': 'set_temp', 'arguments': {'unit': 'celsius', 'value': '25'}}], 'match_mode': 'ordered'}
    model = [{'name': 'set_temp', 'arguments': {'unit': 'centigrade', 'value': '25'}}]
    replay = _match_tool_call(model, oracle, enum_map=flat_enum)
    oracle_obj = OracleAction(action_type='tool_call', tool_calls=[{'name': 'set_temp', 'arguments': {'unit': 'celsius', 'value': '25'}}], match_mode='ordered')
    meta = SampleMetadata(enum_map=flat_enum)
    r = reward_fn.compute('<tool_call>{"name": "set_temp", "arguments": {"unit": "centigrade", "value": "25"}}</tool_call>', oracle_obj, meta)
    assert replay == r.exact_success, f'flat_enum_perturbed: replay={replay} reward={r.exact_success}'
    print(f'  flat_enum_perturbed: replay={replay} reward={r.exact_success} OK')


def test_flat_enum_original():
    """flat enum map: original value 应被拒绝。"""
    flat_enum = {'centigrade': 'celsius', 'fahrenheit_scale': 'fahrenheit'}
    oracle = {'tool_calls': [{'name': 'set_temp', 'arguments': {'unit': 'celsius', 'value': '25'}}], 'match_mode': 'ordered'}
    model = [{'name': 'set_temp', 'arguments': {'unit': 'celsius', 'value': '25'}}]
    replay = _match_tool_call(model, oracle, enum_map=flat_enum)
    oracle_obj = OracleAction(action_type='tool_call', tool_calls=[{'name': 'set_temp', 'arguments': {'unit': 'celsius', 'value': '25'}}], match_mode='ordered')
    meta = SampleMetadata(enum_map=flat_enum)
    r = reward_fn.compute('<tool_call>{"name": "set_temp", "arguments": {"unit": "celsius", "value": "25"}}</tool_call>', oracle_obj, meta)
    assert replay == r.exact_success, f'flat_enum_original: replay={replay} reward={r.exact_success}'
    print(f'  flat_enum_original: replay={replay} reward={r.exact_success} OK')


if __name__ == '__main__':
    print('Running matching consistency tests...')
    test_list_numeric_string()
    test_list_dict_key_order()
    test_nested_list()
    test_flat_enum_perturbed()
    test_flat_enum_original()
    print('\nAll consistency tests PASSED!')
