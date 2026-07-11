"""Jaccard-based deduplication for LLM teacher generated tasks.

PROVE uses Jaccard similarity threshold of 0.70 on oracle call sequences
to remove near-duplicate training tasks. This improves data diversity.
"""

from __future__ import annotations

from typing import Iterable

from src.live_mcp.types import LiveTask


def jaccard_similarity(a: LiveTask, b: LiveTask) -> float:
    """Jaccard similarity between two tasks' oracle tool-call sequences.

    PROVE deduplicates on tool-call sequences.  Each task is represented as an
    ordered list of tool names from its oracle program. Position is included so:
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
        # Both zero-tool (irrelevance / missing_function) — dedup by query
        # text similarity instead of ignoring them.  Jaccard on word sets
        # provides a reasonable diversity filter for abstention tasks.
        qa = set((a.user_prompt or "").lower().split())
        qb = set((b.user_prompt or "").lower().split())
        if not qa or not qb:
            return 0.0
        intersection = qa & qb
        union = qa | qb
        return len(intersection) / len(union)
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

    Only compares tasks within the same domain — cross-domain Jaccard is
    always 0 (different tool sets) so cross-domain comparison is wasteful
    and never triggers dedup.

    Preserves insertion order.  For each task, if any previously kept task
    *in the same domain* has Jaccard similarity >= *threshold*, it is skipped.
    """
    kept: list[LiveTask] = []
    for task in tasks:
        task_domain = task.target_servers[0] if task.target_servers else ""
        is_dup = False
        for kept_task in kept:
            kept_domain = kept_task.target_servers[0] if kept_task.target_servers else ""
            if task_domain != kept_domain:
                continue
            if jaccard_similarity(task, kept_task) >= threshold:
                is_dup = True
                break
        if not is_dup:
            kept.append(task)
    return kept


# ── helpers ──────────────────────────────────────────────────────────


def _call_sequence(task: LiveTask) -> list[str]:
    """Build the ordered tool-name sequence from oracle calls.

    Uses list (not set) to preserve call order and repeat count. Arguments are
    ignored to match PROVE's sequence-level Jaccard deduplication.
    """
    calls = task.oracle_program.calls
    has_tool_call = any(
        (call.get("action", "tool_call") if isinstance(call, dict)
         else getattr(call, "action", "tool_call")) == "tool_call"
        for call in calls
    )
    if not has_tool_call:
        # Fallback: use original oracle program from metadata for
        # missing_function tasks (oracle_program.calls was cleared
        # in _apply_missing_function but original stored in metadata).
        orig = task.metadata.get("original_oracle_program", {})
        if isinstance(orig, dict) and orig.get("calls"):
            calls = orig["calls"]

    sigs: list[str] = []
    for call in calls:
        tool_name: str = ""

        if isinstance(call, dict):
            # metadata fallback: plain dict format from to_plain()
            tool_name = call.get("tool_name", "")
        else:
            # native OracleCall dataclass
            tool_name = call.tool_name

        action = call.get("action", "tool_call") if isinstance(call, dict) else getattr(call, "action", "tool_call")
        if action != "tool_call":
            continue

        sigs.append(tool_name)
    return sigs
