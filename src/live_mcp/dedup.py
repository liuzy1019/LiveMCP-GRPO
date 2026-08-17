"""Jaccard deduplication for LLM-generated task traces."""

from __future__ import annotations

from typing import Iterable

from src.live_mcp.types import LiveTask


def jaccard_similarity(a: LiveTask, b: LiveTask) -> float:
    """Jaccard similarity between two tasks' oracle tool-call sequences.

    Each task is represented by the set of plain oracle tool names, matching
    the published Jaccard gate over tool-call sequences. Order, repetitions,
    arguments, owners, and hidden generation metadata do not alter this gate.

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

    set_a = set(sigs_a)
    set_b = set(sigs_b)

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
    sigs: list[str] = []
    for call in calls:
        if call.action != "tool_call":
            continue
        sigs.append(str(call.tool_name))
    return sigs
