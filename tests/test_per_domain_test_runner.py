from pathlib import Path

import pytest

from scripts.per_domain_test_runner import (
    CONFIG_ROOT,
    DOMAINS,
    generation_command,
    load_config,
    rollout_command,
    validate_gpus,
)


def test_all_domain_configs_share_the_formal_30_plus_10_contract() -> None:
    configs = [load_config(CONFIG_ROOT / f"{domain}.yaml") for domain in DOMAINS]

    assert [config["domain"] for config in configs] == list(DOMAINS)
    assert all(config["generation"]["count"] == 30 for config in configs)
    assert all(config["generation"]["val_count"] == 10 for config in configs)
    assert all(config["generation"]["gpu_count"] == 4 for config in configs)


def test_domain_runner_uses_only_public_generation_and_reward_entries() -> None:
    config = load_config(CONFIG_ROOT / "banking.yaml")
    generation = generation_command(config, "test_run")
    rollout = rollout_command(
        config,
        Path("/mnt/data2/liuzhanyi/livemcp-grpo/data/runs/test_run"),
        "0,1,2,3",
        "owned_test_run",
    )

    assert generation[1:5] == ["-m", "src.live_mcp.corpus.cli", "run", "--mode"]
    assert "src.live_mcp.corpus.shard" not in generation
    assert rollout[:2] == ["bash", "scripts/smoke_rollout_reward.sh"]
    assert rollout[rollout.index("--reward-profile") + 1] == "prove_baseline"
    assert rollout[rollout.index("--artifact-id") + 1] == "owned_test_run"


@pytest.mark.parametrize("value", ["0,1,2", "0,1,2,2", "0,1,2,x", "0,1,2,3,4"])
def test_domain_runner_rejects_non_four_distinct_gpus(value: str) -> None:
    with pytest.raises(ValueError, match="exactly four distinct"):
        validate_gpus(value)


def test_domain_runner_accepts_exact_four_distinct_gpus() -> None:
    assert validate_gpus("4,5,6,7") == "4,5,6,7"


def test_domain_runner_rejects_formal_config_drift(tmp_path: Path) -> None:
    source = (CONFIG_ROOT / "banking.yaml").read_text()
    path = tmp_path / "banking.yaml"
    path.write_text(source.replace("rollout_n: 16", "rollout_n: 8"))

    with pytest.raises(ValueError, match="formal rollout contract drift"):
        load_config(path)
