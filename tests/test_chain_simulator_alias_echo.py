from src.live_mcp.contracts.chain_simulator import (
    simulate_symbolic_chain,
    symbolic_step_bindings,
)
from src.live_mcp.contracts.factory import build_contract_registry
from src.live_mcp.contracts.models import ToolContract
from src.live_mcp.servers.filesystem.server import TOOLS as FILESYSTEM_TOOLS


def _contract(
    domain: str, name: str, *, arguments: set[str], outputs: set[str],
) -> ToolContract:
    return ToolContract(
        domain=domain,
        name=name,
        readonly=True,
        mutating=False,
        arguments=frozenset(arguments),
        output_fields=frozenset(outputs),
    )


def test_alias_echo_is_not_available_as_observation_derived_value() -> None:
    source = _contract(
        "shopping", "apply_coupon", arguments={"code"}, outputs={"coupon"},
    )
    target = _contract(
        "shopping", "inspect_coupon", arguments={"code"}, outputs=set(),
    )

    steps = symbolic_step_bindings("shopping", [source, target])

    assert steps[1]["argument:code"] == "live:shopping:code"


def test_novel_alias_output_remains_available_downstream() -> None:
    source = _contract(
        "shopping", "discover_coupon", arguments=set(), outputs={"coupon"},
    )
    target = _contract(
        "shopping", "apply_coupon", arguments={"code"}, outputs=set(),
    )

    steps = symbolic_step_bindings("shopping", [source, target])

    assert steps[1]["argument:code"] == "step:0:coupon"


def test_filesystem_copy_move_preserve_type_for_downstream_checks() -> None:
    registry = build_contract_registry({"filesystem": FILESYSTEM_TOOLS})

    _, invalid = simulate_symbolic_chain(
        registry,
        "filesystem",
        ["touch", "cp", "mv", "readlink"],
    )
    _, valid = simulate_symbolic_chain(
        registry,
        "filesystem",
        ["symlink", "readlink"],
    )

    assert invalid
    assert invalid[0].tool_name == "readlink"
    assert invalid[0].predicate.slot == "filesystem.type"
    assert valid == ()


def test_symbolic_binding_uses_nearest_compatible_output() -> None:
    registry = build_contract_registry({"filesystem": FILESYSTEM_TOOLS})
    contracts = [
        registry.get("filesystem", name)
        for name in ("touch", "cp", "mv", "stat")
    ]

    steps = symbolic_step_bindings("filesystem", contracts)

    assert steps[2]["argument:source"] == "step:1:target"
    assert steps[3]["argument:path"] == "step:2:target"
