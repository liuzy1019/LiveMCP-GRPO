"""Parquet readback validation at the artifact/runtime contract boundary."""

from __future__ import annotations

import gc
import importlib
import json
from pathlib import Path

import pandas as pd

from src.live_mcp.artifact.validation import validate_artifact_contract
from src.live_mcp.registry.environment_metadata import (
    compute_initial_state_hashes,
    normalize_state_profiles,
    validate_environment_metadata,
)
from src.utils import normalize_extra_info


def validate_parquet_readback(path: Path) -> None:
    """Run every written row through artifact and environment consumers."""
    frame = pd.read_parquet(path)
    if frame.empty:
        # Both supported environments can abort during interpreter teardown
        # when an empty PyArrow-backed frame is left for global finalization.
        # Release it while the runtime is still fully initialized.
        del frame
        gc.collect()
        return
    for row_index, row in frame.iterrows():
        try:
            extra_info = normalize_extra_info(row.get("extra_info"))
            reward_model = row.get("reward_model") or {}
            if not isinstance(reward_model, dict):
                raise RuntimeError("reward_model must be a mapping")
            task = validate_artifact_contract(
                extra_info,
                require_training=False,
                ground_truth=reward_model.get("ground_truth"),
            )
            owners_raw = extra_info.get("tool_owner_domains", {})
            if isinstance(owners_raw, str):
                owners_raw = json.loads(owners_raw)
            owners = {str(extra_info.get("domain") or "")}
            if isinstance(owners_raw, dict):
                owners.update(str(owner) for owner in owners_raw.values())
            owners.discard("")
            tools_by_owner = {
                owner: list(importlib.import_module(
                    f"src.live_mcp.servers.{owner}.server"
                ).TOOLS)
                for owner in owners
            }
            validate_environment_metadata(
                extra_info,
                current_tools_by_domain=tools_by_owner,
                required_owner_domains=owners,
                reward_profile="prove_baseline",
                runtime_max_observation_chars=int(
                    extra_info.get("max_observation_chars", 4096)
                ),
                current_initial_state_hashes=compute_initial_state_hashes(
                    owners,
                    int(extra_info["session_seed"]),
                    normalize_state_profiles(
                        extra_info.get("state_profiles"), owners
                    ),
                ),
            )
        except Exception as exc:
            raise RuntimeError(
                f"{path}: row {row_index} failed production parser readback: {exc}"
            ) from exc
        if not isinstance(task, dict) or "required_tool_calls" not in task:
            raise RuntimeError(
                f"{path}: row {row_index} produced invalid reward task"
            )


__all__ = ["validate_parquet_readback"]
