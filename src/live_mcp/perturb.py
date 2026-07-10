"""
Perturbation knobs for RL rollout robustness (PROVE §3.2).

Two perturbation types:
1. distractor tools — inject irrelevant tools into the action space
2. enum stripping — hide enum values from parameter descriptions

Teacher/oracle always sees clean schemas. Perturbations only affect
the RL rollout prompt, testing model robustness to noisy schemas.

Usage in data generation:
    from src.live_mcp.perturb import Perturber

    perturber = Perturber(all_tools_pool=registry.all_tools())
    task.visible_tools, metadata = perturber.apply(task.visible_tools, seed=seed)

Usage in RL rollout:
    from src.live_mcp.perturb import Perturber, build_system_prompt

    perturber = Perturber(all_tools_pool)
    perturbed_tools, meta = perturber.apply(clean_tools, seed=seed)
    system_prompt = build_system_prompt(
        perturbed_tools, domain_desc,
        reference_date=reference_date,
    )
"""

import hashlib
import random
from typing import Any


def strip_enums_from_schemas(tools: list[dict]) -> list[dict]:
    """Return new list of tool dicts with enum values removed from input_schema."""
    result: list[dict] = []
    for tool in tools:
        t = dict(tool)
        props = t.get("input_schema", {}).get("properties", {})
        if props:
            stripped: dict[str, dict] = {}
            for k, v in props.items():
                stripped[k] = {kk: vv for kk, vv in v.items() if kk != "enum"}
            t["input_schema"] = {**t.get("input_schema", {}), "properties": stripped}
        result.append(t)
    return result


class Perturber:
    """Applies perturbation knobs to tool schemas.

    All randomness is deterministic: given the same seed and config,
    the same perturbations are produced.
    """

    def __init__(
        self,
        all_tools_pool: list[dict] | None = None,
        distractor_rate: float = 0.40,
        strip_enums_rate: float = 0.30,
    ):
        self.all_tools_pool = all_tools_pool or []
        self.distractor_rate = distractor_rate
        self.strip_enums_rate = strip_enums_rate

    def apply(
        self, clean_tools: list[dict], seed: int = 0
    ) -> tuple[list[dict], dict]:
        """Apply perturbations to tool schemas.

        Returns:
            perturbed_tools: shallow-copied and perturbed tool schemas
            metadata: {"has_distractors", "distractor_count", "strip_enums"}
        """
        rng = random.Random(seed)
        metadata: dict[str, Any] = {}
        perturbed = [dict(t) for t in clean_tools]

        # 1. Enum stripping
        if rng.random() < self.strip_enums_rate:
            perturbed = strip_enums_from_schemas(perturbed)
            metadata["strip_enums"] = True

        # 2. Distractor injection
        if self.all_tools_pool and rng.random() < self.distractor_rate:
            known = {t["name"] for t in clean_tools}
            candidates = [t for t in self.all_tools_pool if t["name"] not in known]
            if candidates:
                # Use task_id-derived seed for deterministic selection
                n = min(len(candidates), rng.randint(3, 8))
                selected = rng.sample(candidates, n)
                perturbed.extend([dict(t) for t in selected])
                metadata["has_distractors"] = True
                metadata["distractor_count"] = len(selected)

        return perturbed, metadata


def build_system_prompt(
    tools: list[dict],
    domain_desc: str,
    reference_date: str = "",
) -> str:
    """Build the system prompt from (possibly perturbed) tool schemas.

    This is the same format used in generate_data._tasks_to_rows,
    extracted here so rollout can rebuild prompts with perturbed tools.
    """
    from src.live_mcp.task_planner import _format_tools

    tools_text = _format_tools(tools)
    date_line = f"\nToday's date: {reference_date}." if reference_date else ""
    return (
        f"You are an AI assistant for the following domain:\n{domain_desc}\n\n"
        f"## Available Tools\n{tools_text}\n\n"
        f"## Response Format\n"
        f"Output exactly ONE action per turn using XML tags:\n\n"
        f'- <tool_call>{{"name": "<tool_name>", "arguments": {{...}}}}</tool_call>\n'
        f"  Call a tool with its required parameters.\n\n"
        f"- <final_answer>your answer</final_answer>\n"
        f"  When the task is fully completed.\n\n"
        f"- <report_error>brief reason</report_error>\n"
        f"  When the task cannot be completed with available tools.\n\n"
        f"- <ask_clarification>what you need to know</ask_clarification>\n"
        f"  When genuinely missing critical information and no tool can resolve it.\n\n"
        f"## Rules\n"
        f"- Call ONE tool at a time. Wait for the result before the next action.\n"
        f"- Do not output hidden reasoning, chain-of-thought, or <think> tags.\n"
        f"- Use ONLY entity IDs that appear in tool results. Never invent or guess IDs.{date_line}"
    )
