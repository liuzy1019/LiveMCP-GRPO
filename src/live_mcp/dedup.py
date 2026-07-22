"""Jaccard deduplication for LLM-generated task traces."""

from __future__ import annotations

from typing import Iterable

from src.live_mcp.types import LiveTask


def jaccard_similarity(a: LiveTask, b: LiveTask) -> float:
    """Jaccard similarity between two tasks' oracle tool-call sequences.

    Each task is represented as an ordered list of oracle tool names. Position
    is included so:
      * [a, b] vs [b, a]    -> distinguishable
      * [a, b] vs [a, a, b] -> distinguishable

    Arguments are intentionally ignored: two traces that execute the same tool
    sequence on different entity IDs are near-duplicates for dependency-skill
    coverage and should be filtered by the 0.70 threshold.

    Returns a float in [0.0, 1.0].
    """
    sigs_a = _call_sequence(a)
    sigs_b = _call_sequence(b)

    if not sigs_a and not sigs_b:
        # Jaccard is computed on tool-call sequences, not query text.
        # Empty sequences provide no positive similarity evidence and must not
        # collapse clarification or abstention examples.
        return 0.0
    if not sigs_a or not sigs_b:
        return 0.0

    # Position-aware sequence set: each entry tagged with its index so order and
    # repeated calls matter while argument values do not.
    set_a = {(i, tn) for i, tn in enumerate(sigs_a)}
    set_b = {(i, tn) for i, tn in enumerate(sigs_b)}

    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union)


def dedup_tasks(
    tasks: Iterable[LiveTask],
    threshold: float = 0.70,
) -> list[LiveTask]:
    """Greedy deduplication: keep first occurrence, discard subsequent similar tasks.

    Preserves insertion order.  For each task, if any previously kept task
    has Jaccard similarity >= *threshold*, it is skipped.
    """
    kept: list[LiveTask] = []
    for task in tasks:
        is_dup = False
        for kept_task in kept:
            if jaccard_similarity(task, kept_task) >= threshold:
                is_dup = True
                break
        if not is_dup:
            kept.append(task)
    return kept


# ── helpers ──────────────────────────────────────────────────────────


def _call_sequence(task: LiveTask) -> list[str]:
    """Build the ordered tool-name sequence from oracle calls.

    Uses list (not set) to preserve call order and repeat count. Argument values
    are omitted so deduplication measures the tool-call sequence.
    """
    calls = task.oracle_program.calls
    primary_domain = str(task.target_servers[0]) if task.target_servers else ""
    sigs: list[str] = []
    for call in calls:
        if call.action != "tool_call":
            continue
        owner = str(getattr(call, "server_name", "") or primary_domain)
        sigs.append(f"{owner}::{call.tool_name}")
    return sigs
