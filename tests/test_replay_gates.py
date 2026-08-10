from types import SimpleNamespace

from src.live_mcp.replay.gates import replay_validate
from src.live_mcp.types import OracleCall


class _Manager:
    def create_session(self, **_kwargs):
        return SimpleNamespace(session_id="replay-session")

    def discover_tools(self, _session_id):
        return []

    def get_state(self, _session_id):
        return {}

    def close_session(self, _session_id):
        return None


class _Executor:
    def __init__(self, result):
        self.result = result

    def execute(self, *_args, **_kwargs):
        return self.result


def _result(*, success: bool, schema_valid: bool = True):
    return SimpleNamespace(
        success=success,
        schema_valid=schema_valid,
        execution_status="SUCCESS" if success else "FAILURE",
        state_changed=False,
        observation={},
        error_type="" if success else "precondition_failed",
        error_message="" if success else "expected precondition",
    )


def test_expected_execution_failure_is_replay_consistent() -> None:
    replay = replay_validate(
        oracle_calls=[OracleCall(
            "get_product", {}, expected_success=False,
        )],
        manager=_Manager(),
        executor=_Executor(_result(success=False)),
        seed=1,
        domain="shopping",
    )

    assert replay[:4] == (True, 0.0, 0, 1)


def test_expected_failure_that_succeeds_is_replay_error() -> None:
    replay = replay_validate(
        oracle_calls=[OracleCall(
            "get_product", {}, expected_success=False,
        )],
        manager=_Manager(),
        executor=_Executor(_result(success=True)),
        seed=1,
        domain="shopping",
    )

    assert replay[:4] == (False, 1.0, 1, 1)


def test_schema_error_remains_error_for_expected_failure() -> None:
    replay = replay_validate(
        oracle_calls=[OracleCall(
            "get_product", {}, expected_success=False,
        )],
        manager=_Manager(),
        executor=_Executor(_result(success=False, schema_valid=False)),
        seed=1,
        domain="shopping",
    )

    assert replay[:4] == (False, 1.0, 1, 1)
