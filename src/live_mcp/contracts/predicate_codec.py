"""Migration codec from audited predicate tokens to typed state facts.

The codec is deliberately domain-neutral.  It lets current audited facts feed
the new simulator while those facts move to native ``StatePredicate`` values.
"""

from __future__ import annotations

from src.live_mcp.contracts.models import EntityBinding, StatePredicate


_BINARY_SUFFIXES = {
    "exists": True,
    "absent": False,
    "member": True,
    "nonmember": False,
    "already_authored": True,
    "not_already_authored": False,
    "recurring": True,
    "nonrecurring": False,
}

_GLOBAL_VALUES = {
    "cart_empty": ("cart.contents", "empty"),
    "cart_nonempty": ("cart.contents", "nonempty"),
    "coupon_applied": ("cart.coupon", True),
}


def decode_state_predicate(
    token: str,
    *,
    required_arguments: frozenset[str],
    output_fields: frozenset[str],
    phase: str,
) -> StatePredicate:
    """Decode the existing predicate vocabulary without inspecting a domain."""
    if token in _GLOBAL_VALUES:
        slot, value = _GLOBAL_VALUES[token]
        return StatePredicate(slot, EntityBinding("global", slot, "global"), value)

    head, separator, tail = token.partition(":")
    if not separator:
        return StatePredicate(
            head.replace("_", "."),
            EntityBinding("global", head, "global"),
            True,
        )

    if head.endswith("_status"):
        entity_type = head.removesuffix("_status")
        binding_name = f"{entity_type}_id"
        source = (
            "output"
            if phase == "post" and binding_name in output_fields
            else "argument"
        )
        return StatePredicate(
            f"{entity_type}.status",
            EntityBinding(entity_type, binding_name, source),
            tail,
        )

    for suffix, value in sorted(
        _BINARY_SUFFIXES.items(), key=lambda item: len(item[0]), reverse=True,
    ):
        marker = f"_{suffix}"
        if not head.endswith(marker):
            continue
        entity_type = head.removesuffix(marker)
        slot_name = suffix
        if suffix in {"exists", "absent"}:
            slot_name = "exists"
        elif suffix in {"member", "nonmember"}:
            slot_name = "membership"
        elif suffix in {"already_authored", "not_already_authored"}:
            slot_name = "authored"
        elif suffix in {"recurring", "nonrecurring"}:
            slot_name = "recurring"
        source = "output" if phase == "post" and tail in output_fields else "argument"
        return StatePredicate(
            f"{entity_type}.{slot_name}",
            EntityBinding(entity_type, tail, source),
            value,
        )

    source = "output" if phase == "post" and tail in output_fields else "argument"
    entity_type = tail.removesuffix("_id") if tail.endswith("_id") else "value"
    return StatePredicate(
        head.replace("_", "."),
        EntityBinding(entity_type, tail, source),
        True,
    )
