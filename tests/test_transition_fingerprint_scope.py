from pathlib import Path

from src.live_mcp.registry.environment_metadata import _semantic_symbols
from src.live_mcp.state_seeder import available_state_profiles


def test_non_recursive_symbol_fingerprint_excludes_other_builders(
    tmp_path: Path,
) -> None:
    source = tmp_path / "dispatch.py"
    source.write_text(
        "def a():\n    return 1\n\n"
        "def b():\n    return 2\n\n"
        "class Dispatch:\n"
        "    def run(self):\n"
        "        return BUILDERS['a']()\n\n"
        "BUILDERS = {'a': a, 'b': b}\n",
        encoding="utf-8",
    )
    recursive = _semantic_symbols(source, {"Dispatch"})
    scoped = _semantic_symbols(
        source, {"Dispatch"}, follow_dependencies=False,
    )
    assert "Constant(value=2)" in recursive
    assert "Constant(value=2)" not in scoped


def test_state_profiles_are_scoped_per_domain() -> None:
    assert available_state_profiles("banking") == ["baseline"]
    assert available_state_profiles("payments") == [
        "baseline",
        "payments_rare_state_v1",
    ]
    assert available_state_profiles("unknown") == []
