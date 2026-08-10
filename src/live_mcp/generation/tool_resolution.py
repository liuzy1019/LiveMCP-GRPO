"""Tool-name resolution used by the Action Teacher loop."""

from __future__ import annotations

def fuzzy_match_tool(raw: str, valid_names: set[str]) -> str | None:
    """Try to fix a hallucinated tool name by the existing name contract."""
    raw_lower = raw.lower()
    for name in valid_names:
        if name.lower() == raw_lower:
            return name
    if raw_lower.endswith("s"):
        singular = raw_lower[:-1]
        for name in valid_names:
            if name.lower() == singular:
                return name
    for name in valid_names:
        lowered = name.lower()
        if raw_lower in lowered or lowered in raw_lower:
            return name
    raw_words = set(raw_lower.replace("_", " ").split())
    best_name, best_overlap = None, 0
    for name in valid_names:
        name_words = set(name.lower().replace("_", " ").split())
        overlap = len(raw_words & name_words)
        if overlap > best_overlap and overlap >= max(1, len(raw_words) - 1):
            best_overlap, best_name = overlap, name
    return best_name
