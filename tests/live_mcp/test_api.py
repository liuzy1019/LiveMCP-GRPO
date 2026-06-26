import pytest
from src.live_mcp.api import LiveMCPBranch, load_live_tasks, save_live_tasks


@pytest.mark.requires_llm
def test_live_mcp_branch_generates_and_loads_tasks(tmp_path):
    """LLM teacher generates tasks → save → load → verify."""
    output = tmp_path / "tasks.jsonl"
    with LiveMCPBranch.from_suite("configs/live_mcp/suite_mvp.yaml") as branch:
        tasks = branch.generate_tasks_llm(
            server_name="calendar",
            count=2,
            seed=31,
            model_path="models/Qwen3-4B",
        )
        save_live_tasks(tasks, output)
    loaded = load_live_tasks(output)
    assert len(tasks) == 2
    assert len(loaded) == 2
    assert all("calendar" in task.target_servers for task in tasks)
