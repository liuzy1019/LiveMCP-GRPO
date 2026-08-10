"""Shared deterministic state-seeding primitives."""

from __future__ import annotations

import datetime as _datetime
import random


def _reference_datetime(seed: int) -> _datetime.datetime:
    from src.live_mcp.task_planner import reference_datetime_for_seed

    return reference_datetime_for_seed(seed)


def _seed_scoped_id(prefix: str, seed: int, idx: int, width: int = 3) -> str:
    if seed < 0:
        raise ValueError("seed must be non-negative")
    return f"{prefix}_s{seed}_{idx + 1:0{width}d}"


def _sample_entities(
    rng: random.Random,
    template_pool: list,
    target_count: int,
    id_prefix: str,
) -> list:
    indices = list(range(len(template_pool)))
    rng.shuffle(indices)
    selected = indices[:min(target_count, len(template_pool))]
    result = [template_pool[index] for index in sorted(selected)]
    while len(result) < target_count:
        base_index = rng.randint(0, len(template_pool) - 1)
        base = template_pool[base_index]
        variant_id = f"{id_prefix}_{rng.randint(100, 999):03d}"
        result.append(
            (variant_id,) + base[1:]
            if len(base) > 1
            else (variant_id, base[1])
        )
    return result
