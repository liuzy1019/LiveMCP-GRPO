"""ReplayMCPExecutor + MCPToolEnvironment + TrajectoryVerifier 集成测试。"""

import torch
import pytest

from src.data.episode_schema import (
    ActionType,
    EpisodeSeed,
    EpisodeType,
    OracleStep,
    VerifierConfig,
)
from src.envs.replay_mcp_executor import ExecutionResult, MatchConfig, ReplayMCPExecutor
from src.envs.mcp_tool_environment import EnvConfig, MCPToolEnvironment, StepInfo
from src.reward.trajectory_verifier import TrajectoryVerifier


# ---- fixtures ----


def _make_simple_episode() -> EpisodeSeed:
    """单步 tool_call episode。"""
    return EpisodeSeed(
        episode_id="test_001",
        source="toucan",
        episode_type=EpisodeType.CALL_ONLY.value,
        tools_snapshot=[{
            "name": "get_weather",
            "description": "Get weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                    "unit": {"type": "string", "description": "Temperature unit"},
                },
                "required": ["city"],
            },
        }],
        initial_messages=[
            {"role": "user", "content": "What's the weather in Beijing?"},
        ],
        oracle_trace=[
            OracleStep(
                step=0,
                action_type=ActionType.TOOL_CALL.value,
                tool_name="get_weather",
                arguments={"city": "Beijing", "unit": "celsius"},
                replay_observation='{"temperature": 25, "condition": "sunny"}',
                verifier=VerifierConfig(type="exact"),
            ),
        ],
        max_turns=3,
    )


def _make_multi_step_episode() -> EpisodeSeed:
    """多步 call_then_final episode。"""
    return EpisodeSeed(
        episode_id="test_002",
        source="toucan",
        episode_type=EpisodeType.CALL_THEN_FINAL.value,
        tools_snapshot=[{
            "name": "search_flights",
            "description": "Search flights",
            "parameters": {
                "type": "object",
                "properties": {
                    "origin": {"type": "string"},
                    "destination": {"type": "string"},
                    "date": {"type": "string"},
                },
                "required": ["origin", "destination", "date"],
            },
        }],
        initial_messages=[
            {"role": "user", "content": "Find flights from Beijing to Shanghai on 2024-01-15"},
        ],
        oracle_trace=[
            OracleStep(
                step=0,
                action_type=ActionType.TOOL_CALL.value,
                tool_name="search_flights",
                arguments={"origin": "Beijing", "destination": "Shanghai", "date": "2024-01-15"},
                replay_observation='[{"flight": "CA1234", "price": 800}]',
                verifier=VerifierConfig(type="exact"),
            ),
            OracleStep(
                step=1,
                action_type=ActionType.FINAL_ANSWER.value,
                expected_content="Found flight CA1234 for 800 CNY",
                verifier=VerifierConfig(type="exact"),
            ),
        ],
        max_turns=5,
    )


def _make_parallel_episode() -> EpisodeSeed:
    """并行 tool_call episode。"""
    return EpisodeSeed(
        episode_id="test_003",
        source="toucan",
        episode_type=EpisodeType.CALL_ONLY.value,
        tools_snapshot=[
            {"name": "get_weather", "description": "Get weather", "parameters": {}},
            {"name": "get_time", "description": "Get time", "parameters": {}},
        ],
        initial_messages=[
            {"role": "user", "content": "What's the weather and time in Tokyo?"},
        ],
        oracle_trace=[
            OracleStep(
                step=0,
                action_type=ActionType.PARALLEL_TOOL_CALL.value,
                calls=[
                    {"tool_name": "get_weather", "arguments": {"city": "Tokyo"}},
                    {"tool_name": "get_time", "arguments": {"timezone": "Asia/Tokyo"}},
                ],
                match_mode="set",
                replay_observations=[
                    '{"temperature": 20}',
                    '{"time": "14:30"}',
                ],
                verifier=VerifierConfig(type="exact"),
            ),
        ],
        max_turns=3,
    )


# ---- ReplayMCPExecutor tests ----


class TestReplayMCPExecutor:
    def test_exact_match_single_call(self):
        ep = _make_simple_episode()
        executor = ReplayMCPExecutor(ep)

        result = executor.step({
            "action_type": "tool_call",
            "tool_calls": [{"name": "get_weather", "arguments": {"city": "Beijing", "unit": "celsius"}}],
        })

        assert result.matched is True
        assert result.observation == '{"temperature": 25, "condition": "sunny"}'
        assert result.done is True  # 只有一步

    def test_name_mismatch(self):
        ep = _make_simple_episode()
        executor = ReplayMCPExecutor(ep)

        result = executor.step({
            "action_type": "tool_call",
            "tool_calls": [{"name": "wrong_tool", "arguments": {"city": "Beijing"}}],
        })

        assert result.matched is False
        assert result.done is True
        assert "tool_name mismatch" in result.match_detail

    def test_key_coverage_match(self):
        """宽松模式：只要 oracle keys 都存在就匹配。"""
        ep = _make_simple_episode()
        executor = ReplayMCPExecutor(ep)

        # 多了一个 extra key，但 oracle 的 keys 都在
        result = executor.step({
            "action_type": "tool_call",
            "tool_calls": [{"name": "get_weather", "arguments": {"city": "Beijing", "unit": "celsius", "extra": "value"}}],
        })

        assert result.matched is True

    def test_missing_key(self):
        """宽松模式：缺少 oracle key 则不匹配。"""
        ep = _make_simple_episode()
        executor = ReplayMCPExecutor(ep)

        # 缺少 unit key
        result = executor.step({
            "action_type": "tool_call",
            "tool_calls": [{"name": "get_weather", "arguments": {"city": "Beijing"}}],
        })

        assert result.matched is False
        assert "missing argument keys" in result.match_detail

    def test_name_map(self):
        """通过 name_map 映射工具名。"""
        ep = _make_simple_episode()
        config = MatchConfig(name_map={"fetch_weather": "get_weather"})
        executor = ReplayMCPExecutor(ep, match_config=config)

        result = executor.step({
            "action_type": "tool_call",
            "tool_calls": [{"name": "fetch_weather", "arguments": {"city": "Beijing", "unit": "celsius"}}],
        })

        assert result.matched is True

    def test_terminal_action(self):
        """模型输出 final_answer 时直接结束。"""
        ep = _make_simple_episode()
        executor = ReplayMCPExecutor(ep)

        result = executor.step({
            "action_type": "final_answer",
            "content": "The weather is sunny",
        })

        assert result.done is True
        assert result.matched is False  # oracle 期望 tool_call

    def test_parallel_set_match(self):
        """并行调用无序匹配。"""
        ep = _make_parallel_episode()
        executor = ReplayMCPExecutor(ep)

        # 顺序反过来
        result = executor.step({
            "action_type": "tool_call",
            "tool_calls": [
                {"name": "get_time", "arguments": {"timezone": "Asia/Tokyo"}},
                {"name": "get_weather", "arguments": {"city": "Tokyo"}},
            ],
        })

        assert result.matched is True
        assert result.done is True

    def test_parallel_count_mismatch(self):
        """并行调用数量不匹配。"""
        ep = _make_parallel_episode()
        executor = ReplayMCPExecutor(ep)

        result = executor.step({
            "action_type": "tool_call",
            "tool_calls": [
                {"name": "get_weather", "arguments": {"city": "Tokyo"}},
            ],
        })

        assert result.matched is False

    def test_multi_step_progression(self):
        """多步 episode 正确推进。"""
        ep = _make_multi_step_episode()
        executor = ReplayMCPExecutor(ep)

        # Step 0: tool_call
        r1 = executor.step({
            "action_type": "tool_call",
            "tool_calls": [{"name": "search_flights", "arguments": {"origin": "Beijing", "destination": "Shanghai", "date": "2024-01-15"}}],
        })
        assert r1.matched is True
        assert r1.done is False
        assert executor.current_step == 1

        # Step 1: final_answer
        r2 = executor.step({
            "action_type": "final_answer",
            "content": "Found flight CA1234 for 800 CNY",
        })
        assert r2.done is True
        assert r2.matched is True  # oracle 也期望 final_answer

    def test_reset(self):
        """reset 后可以重新执行。"""
        ep = _make_simple_episode()
        executor = ReplayMCPExecutor(ep)

        executor.step({"action_type": "final_answer", "content": "test"})
        assert executor.done is True

        executor.reset()
        assert executor.done is False
        assert executor.current_step == 0

    def test_already_done(self):
        """done 后再 step 返回空结果。"""
        ep = _make_simple_episode()
        executor = ReplayMCPExecutor(ep)

        executor.step({"action_type": "final_answer", "content": "test"})
        result = executor.step({"action_type": "tool_call", "tool_calls": []})
        assert result.done is True
        assert result.observation == ""


# ---- MCPToolEnvironment tests ----


class TestMCPToolEnvironment:
    def test_reset_returns_prompt(self):
        ep = _make_simple_episode()
        env = MCPToolEnvironment(EnvConfig(strict_parse=True))
        prompt = env.reset(ep)

        assert "get_weather" in prompt
        assert "Beijing" in prompt
        assert not env.done

    def test_correct_tool_call_flow(self):
        ep = _make_simple_episode()
        env = MCPToolEnvironment(EnvConfig(strict_parse=False))
        env.reset(ep)

        # 模型输出正确的 tool_call
        model_output = '<tool_call>{"name": "get_weather", "arguments": {"city": "Beijing", "unit": "celsius"}}</tool_call>'
        obs, done = env.step(model_output)

        # 单步 episode，tool_call 匹配后 step_ptr 推进到末尾 → done
        assert done is True

    def test_environment_rejects_wrong_argument_value_by_default(self):
        ep = _make_simple_episode()
        env = MCPToolEnvironment(EnvConfig(strict_parse=False))
        env.reset(ep)

        model_output = '<tool_call>{"name": "get_weather", "arguments": {"city": "Shanghai", "unit": "celsius"}}</tool_call>'
        obs, done = env.step(model_output)

        info = env.get_reward_info()
        assert done is True
        assert info["step_infos"][0].execution_result.matched is False
        assert "argument values mismatch" in info["step_infos"][0].execution_result.match_detail

    def test_multi_step_flow(self):
        ep = _make_multi_step_episode()
        env = MCPToolEnvironment(EnvConfig(strict_parse=False))
        env.reset(ep)

        # Step 0: tool_call
        model_output = '<tool_call>{"name": "search_flights", "arguments": {"origin": "Beijing", "destination": "Shanghai", "date": "2024-01-15"}}</tool_call>'
        obs, done = env.step(model_output)
        assert done is False
        assert "CA1234" in obs

        # Step 1: final_answer
        model_output = '<final_answer>Found flight CA1234 for 800 CNY</final_answer>'
        obs, done = env.step(model_output)
        assert done is True

    def test_max_turns_truncation(self):
        ep = _make_multi_step_episode()
        env = MCPToolEnvironment(EnvConfig(strict_parse=False, max_turns=1))
        env.reset(ep)

        # Step 0: tool_call（匹配成功，但 max_turns=1 会在下一步截断）
        model_output = '<tool_call>{"name": "search_flights", "arguments": {"origin": "Beijing", "destination": "Shanghai", "date": "2024-01-15"}}</tool_call>'
        obs, done = env.step(model_output)

        # 第一步匹配成功后 executor 推进，done 取决于 executor
        # 这里 executor 还没 done（还有 step 1），但 env 检查 max_turns 在下一次 step 时
        if not done:
            model_output2 = '<final_answer>test</final_answer>'
            obs2, done2 = env.step(model_output2)
            # max_turns=1，第二次 step 时 current_step=1 >= max_turns → truncated
            assert done2 is True

    def test_get_reward_info(self):
        ep = _make_simple_episode()
        env = MCPToolEnvironment(EnvConfig(strict_parse=False))
        env.reset(ep)

        model_output = '<tool_call>{"name": "get_weather", "arguments": {"city": "Beijing", "unit": "celsius"}}</tool_call>'
        env.step(model_output)

        info = env.get_reward_info()
        assert info["episode_id"] == "test_001"
        assert info["done"] is True
        assert len(info["step_infos"]) == 1
        assert info["step_infos"][0].execution_result.matched is True

    def test_unparseable_output(self):
        ep = _make_simple_episode()
        env = MCPToolEnvironment(EnvConfig(strict_parse=True))
        env.reset(ep)

        # 无标签的输出在 strict 模式下 unparseable
        model_output = "I think the weather is nice"
        obs, done = env.step(model_output)

        # unparseable → executor 收到 action_type="unparseable" → 未知类型 → done
        assert done is True

    def test_match_config_not_mutated_across_resets(self):
        """共享 EnvConfig.match_config 不应被 episode metadata 原地污染。"""
        ep_with_map = _make_simple_episode()
        ep_with_map.metadata["name_map"] = {"fetch_weather": "get_weather"}
        ep_without_map = _make_simple_episode()

        shared_match_config = MatchConfig()
        env = MCPToolEnvironment(
            EnvConfig(strict_parse=False, match_config=shared_match_config)
        )

        env.reset(ep_with_map)
        assert env.get_reward_info()["metadata"]["name_map"] == {"fetch_weather": "get_weather"}

        env.reset(ep_without_map)
        assert env.get_reward_info()["metadata"]["name_map"] == {}
        assert shared_match_config.name_map == {}


# ---- TrajectoryVerifier tests ----


class TestTrajectoryVerifier:
    def test_single_step_exact_match(self):
        """单步精确匹配应得到正 reward。"""
        ep = _make_simple_episode()
        env = MCPToolEnvironment(EnvConfig(strict_parse=False))
        env.reset(ep)

        model_output = '<tool_call>{"name": "get_weather", "arguments": {"city": "Beijing", "unit": "celsius"}}</tool_call>'
        env.step(model_output)

        reward_info = env.get_reward_info()
        response_mask = torch.ones(50)  # 模拟 50 个 response tokens

        verifier = TrajectoryVerifier()
        token_rewards = verifier.compute_rewards(reward_info, response_mask)

        assert token_rewards.shape == (50,)
        # 精确匹配应得到正 reward
        total = token_rewards.sum().item()
        assert total > 0

    def test_mismatch_low_reward(self):
        """不匹配应得到低 reward。"""
        ep = _make_simple_episode()
        env = MCPToolEnvironment(EnvConfig(strict_parse=False))
        env.reset(ep)

        model_output = '<tool_call>{"name": "wrong_tool", "arguments": {"x": 1}}</tool_call>'
        env.step(model_output)

        reward_info = env.get_reward_info()
        response_mask = torch.ones(50)

        verifier = TrajectoryVerifier()
        token_rewards = verifier.compute_rewards(reward_info, response_mask)

        total = token_rewards.sum().item()
        # 不匹配的 reward 应该很低
        assert total < 0.5

    def test_last_token_placement(self):
        """reward 应放在最后一个 response token 上。"""
        response_mask = torch.zeros(100)
        response_mask[20:80] = 1.0  # response tokens 在 20-79

        ep = _make_simple_episode()
        env = MCPToolEnvironment(EnvConfig(strict_parse=False))
        env.reset(ep)
        env.step('<tool_call>{"name": "get_weather", "arguments": {"city": "Beijing", "unit": "celsius"}}</tool_call>')

        reward_info = env.get_reward_info()
        verifier = TrajectoryVerifier(reward_placement="last_token")
        token_rewards = verifier.compute_rewards(reward_info, response_mask)

        # 只有 position 79 有值
        assert token_rewards[79].item() != 0
        assert token_rewards[:79].sum().item() == 0
        assert token_rewards[80:].sum().item() == 0

    def test_uniform_placement(self):
        """uniform 模式下 reward 均匀分配。"""
        response_mask = torch.zeros(100)
        response_mask[20:30] = 1.0  # 10 个 response tokens

        ep = _make_simple_episode()
        env = MCPToolEnvironment(EnvConfig(strict_parse=False))
        env.reset(ep)
        env.step('<tool_call>{"name": "get_weather", "arguments": {"city": "Beijing", "unit": "celsius"}}</tool_call>')

        reward_info = env.get_reward_info()
        verifier = TrajectoryVerifier(reward_placement="uniform")
        token_rewards = verifier.compute_rewards(reward_info, response_mask)

        # 10 个 token 应该有相同的值
        response_rewards = token_rewards[20:30]
        assert torch.allclose(response_rewards, response_rewards[0].expand(10))

    def test_batch_rewards(self):
        """批量计算。"""
        ep = _make_simple_episode()
        env = MCPToolEnvironment(EnvConfig(strict_parse=False))

        reward_infos = []
        for _ in range(3):
            env.reset(ep)
            env.step('<tool_call>{"name": "get_weather", "arguments": {"city": "Beijing", "unit": "celsius"}}</tool_call>')
            reward_infos.append(env.get_reward_info())

        response_masks = torch.ones(3, 50)
        verifier = TrajectoryVerifier()
        batch_rewards = verifier.compute_batch_rewards(reward_infos, response_masks)

        assert batch_rewards.shape == (3, 50)
        # 三个相同的 episode 应该得到相同的 reward
        assert torch.allclose(batch_rewards[0], batch_rewards[1])
        assert torch.allclose(batch_rewards[1], batch_rewards[2])

    def test_unmatched_extra_step_gets_no_all_exact_bonus(self):
        """没有 oracle 对照的额外 step 不应触发 all_steps_exact bonus。"""
        verifier = TrajectoryVerifier(trajectory_bonus=0.1)
        reward = verifier._compute_trajectory_reward(
            step_results=[None],
            reward_info={"total_steps": 1, "oracle_total_steps": 0},
        )
        assert reward == 0.0

    def test_truncation_penalty(self):
        """截断应有惩罚。"""
        # 手动构造 truncated reward_info
        reward_info = {
            "episode_id": "test",
            "step_infos": [],
            "total_steps": 0,
            "oracle_total_steps": 2,
            "done": True,
            "truncated": True,
            "all_matched": False,
            "perturbation_level": "none",
            "scenario_type": "call_only",
            "metadata": {"name_map": {}, "enum_map": {}},
        }

        response_mask = torch.ones(50)
        verifier = TrajectoryVerifier(truncation_penalty=-0.5)
        token_rewards = verifier.compute_rewards(reward_info, response_mask)

        total = token_rewards.sum().item()
        assert total < 0  # 截断惩罚

    def test_diagnostics(self):
        """诊断信息应包含关键字段。"""
        ep = _make_simple_episode()
        env = MCPToolEnvironment(EnvConfig(strict_parse=False))
        env.reset(ep)
        env.step('<tool_call>{"name": "get_weather", "arguments": {"city": "Beijing", "unit": "celsius"}}</tool_call>')

        reward_info = env.get_reward_info()
        verifier = TrajectoryVerifier()
        diag = verifier.get_diagnostics(reward_info)

        assert diag["episode_id"] == "test_001"
        assert diag["total_steps"] == 1
        assert diag["all_matched"] is True
        assert len(diag["steps"]) == 1
