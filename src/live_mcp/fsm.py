"""Conversation finite-state machine and robustness plan for LiveMCP generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import random
from typing import Any


class FSMStateGroup(str, Enum):
    """State groups persisted for each generated conversation."""

    QUERY = "query"
    TURN = "turn"
    TOOL_EXECUTION = "tool_execution"
    RESPONSE = "response"
    CONTINUATION = "continuation"


@dataclass
class ConversationFSM:
    """Finite-state controller for one synthesized conversation.

    The FSM is part of generation control, not merely a trace recorder.  An
    illegal edge indicates an orchestrator bug and fails the candidate closed.
    """

    state: FSMStateGroup = FSMStateGroup.QUERY
    transitions: list[dict[str, Any]] = field(default_factory=list)

    def transition(
        self,
        target: FSMStateGroup,
        event: str,
        **evidence: Any,
    ) -> None:
        legal_targets = {
            FSMStateGroup.QUERY: {FSMStateGroup.TURN},
            FSMStateGroup.TURN: {
                FSMStateGroup.TURN,
                FSMStateGroup.TOOL_EXECUTION,
                FSMStateGroup.RESPONSE,
            },
            FSMStateGroup.TOOL_EXECUTION: {FSMStateGroup.RESPONSE},
            FSMStateGroup.RESPONSE: {
                FSMStateGroup.TURN,
                FSMStateGroup.TOOL_EXECUTION,
                FSMStateGroup.RESPONSE,
                FSMStateGroup.CONTINUATION,
            },
            FSMStateGroup.CONTINUATION: {
                FSMStateGroup.CONTINUATION,
                FSMStateGroup.QUERY,
            },
        }
        if target not in legal_targets[self.state]:
            raise RuntimeError(
                "Illegal conversation FSM transition: "
                f"{self.state.value} -> {target.value} ({event})"
            )
        self.transitions.append({
            "from": self.state.value,
            "to": target.value,
            "event": event,
            **evidence,
        })
        self.state = target


@dataclass
class RobustnessPlan:
    """Robustness knobs plus any factually bound perturbation contract.

    Sampling decides only whether a perturbation is requested.  A requested
    missing-function knob is not executable until generation binds
    ``hidden_tool`` and ``missing_function_evidence`` from audited contracts.
    """
    inject_distractors: bool = False
    distractor_tools: list[dict] = field(default_factory=list)
    strip_enums: bool = False
    missing_function: bool = False
    missing_function_requested: bool = False
    hidden_tool: str | None = None
    missing_function_evidence: tuple[str, ...] = ()
    missing_function_binding_failure: str = ""
    irrelevance: bool = False

    @classmethod
    def sample(
        cls,
        seed: int,
        all_tools_pool: list[dict],
        domain_tools: list[dict],
        distractor_rate: float,
        strip_enums_rate: float,
        missing_function_rate: float,
        irrelevance: bool = False,
    ) -> "RobustnessPlan":
        """Sample a deterministic robustness plan from the given seed."""
        rng = random.Random(seed)
        known_names = {t["name"] for t in domain_tools}

        # Distractor: sample 3-8 tools from other domains
        distractors: list[dict] = []
        inject_distractors = bool(all_tools_pool and rng.random() < distractor_rate)
        if inject_distractors:
            unique_candidates: dict[str, dict] = {}
            for tool in all_tools_pool:
                name = str(tool.get("name") or "")
                if name and name not in known_names and name not in unique_candidates:
                    unique_candidates[name] = tool
            candidates = list(unique_candidates.values())
            if candidates:
                n = min(len(candidates), rng.randint(3, 8))
                distractors = [dict(t) for t in rng.sample(candidates, n)]

        # Enum stripping
        strip_enums = rng.random() < strip_enums_rate

        # Missing function
        missing_function = rng.random() < missing_function_rate

        return cls(
            inject_distractors=bool(distractors),
            distractor_tools=distractors,
            strip_enums=strip_enums,
            missing_function=missing_function,
            missing_function_requested=missing_function,
            irrelevance=irrelevance,
        )


def teacher_tool_call_budget(
    configured_budget: int,
    source_chain: list[str] | None,
) -> int:
    """Return a safety ceiling that does not censor longer source chains.

    A chain node can require one discovery/detail call in addition to the
    chain-aligned action itself.  Four extra calls cover initial discovery and
    evidence delivery.  This is a local execution-safety ceiling, not a PROVE
    corpus filter; the terminal decision is accounted for separately.
    """
    base = max(1, int(configured_budget))
    if not source_chain:
        return base
    return max(base, 2 * len(source_chain) + 4)
