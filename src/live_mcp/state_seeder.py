"""Registry facade for deterministic ten-domain state seeders."""

from __future__ import annotations

import copy
from typing import Any, Callable

from src.live_mcp.state_seeders.banking import _banking_state
from src.live_mcp.state_seeders.calendar import _calendar_state
from src.live_mcp.state_seeders.crm import _crm_state
from src.live_mcp.state_seeders.email import _email_state
from src.live_mcp.state_seeders.filesystem import _filesystem_state
from src.live_mcp.state_seeders.food_delivery import _food_delivery_state
from src.live_mcp.state_seeders.issue_tracker import _issue_tracker_state
from src.live_mcp.state_seeders.payments import _payments_state
from src.live_mcp.state_seeders.shopping import _shopping_state
from src.live_mcp.state_seeders.team_chat import _team_chat_state


StateBuilder = Callable[[int], dict[str, Any]]


_STATE_PROFILE_BUILDERS: dict[tuple[str, str], StateBuilder] = {
    ("calendar", "baseline"): _calendar_state,
    ("shopping", "baseline"): _shopping_state,
    ("banking", "baseline"): _banking_state,
    ("email", "baseline"): _email_state,
    ("filesystem", "baseline"): _filesystem_state,
    ("payments", "baseline"): lambda seed: _payments_state(
        seed, state_profile="baseline",
    ),
    ("payments", "payments_rare_state_v1"): lambda seed: _payments_state(
        seed, state_profile="payments_rare_state_v1",
    ),
    ("crm", "baseline"): _crm_state,
    ("issue_tracker", "baseline"): _issue_tracker_state,
    ("team_chat", "baseline"): _team_chat_state,
    ("food_delivery", "baseline"): _food_delivery_state,
}


class StateSeeder:
    def seed_state(
        self,
        server_name: str,
        session_id: str,
        seed: int,
        state_profile: str = "baseline",
    ) -> dict[str, Any]:
        del session_id
        builder = _STATE_PROFILE_BUILDERS.get((server_name, state_profile))
        if builder is None:
            raise ValueError(
                f"unsupported state profile for {server_name}: {state_profile}"
            )
        return builder(seed)

    def reset_state(
        self,
        server_name: str,
        session_id: str,
        seed: int,
        state_profile: str = "baseline",
    ) -> dict[str, Any]:
        return copy.deepcopy(
            self.seed_state(server_name, session_id, seed, state_profile)
        )


def available_state_profiles(server_name: str) -> list[str]:
    return sorted(
        profile
        for domain, profile in _STATE_PROFILE_BUILDERS
        if domain == server_name
    )
