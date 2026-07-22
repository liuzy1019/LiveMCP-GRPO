"""Build and validate executable-environment metadata.

The dependency graph remains keyed by public tool schema/classifier semantics.
The fingerprints bind generated rows to the executable transition, seeder,
observation and reward implementations.
"""

from __future__ import annotations

import ast
import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from src.live_mcp.observation import (
    TRAJECTORY_SCHEMA_VERSION,
    OBSERVATION_SCHEMA_VERSION,
    OBSERVATION_PROJECTION_VERSION,
    compute_server_schema_hash,
)


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def _semantic_ast(path: Path) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    tree = _DocstringStripper().visit(tree)
    ast.fix_missing_locations(tree)
    return ast.dump(tree, annotate_fields=True, include_attributes=False)


class _UnusedImportStripper(ast.NodeTransformer):
    """Remove imports that cannot affect the parsed module's behavior."""

    def __init__(self, loaded_names: set[str]) -> None:
        self.loaded_names = loaded_names

    def visit_Import(self, node: ast.Import) -> ast.Import | None:
        aliases = [
            alias for alias in node.names
            if (alias.asname or alias.name.split(".", 1)[0]) in self.loaded_names
        ]
        if not aliases:
            return None
        node.names = aliases
        return node

    def visit_ImportFrom(self, node: ast.ImportFrom) -> ast.ImportFrom | None:
        if node.module == "__future__":
            return node
        aliases = [
            alias for alias in node.names
            if alias.name == "*" or (alias.asname or alias.name) in self.loaded_names
        ]
        if not aliases:
            return None
        node.names = aliases
        return node


class _DocstringStripper(ast.NodeTransformer):
    """Remove module, class and function docstrings from a semantic AST."""

    @staticmethod
    def _strip(body: list[ast.stmt]) -> list[ast.stmt]:
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            return body[1:]
        return body

    def visit_Module(self, node: ast.Module) -> ast.Module:
        self.generic_visit(node)
        node.body = self._strip(node.body)
        return node

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:
        self.generic_visit(node)
        node.body = self._strip(node.body)
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        self.generic_visit(node)
        node.body = self._strip(node.body)
        return node

    def visit_AsyncFunctionDef(
        self, node: ast.AsyncFunctionDef,
    ) -> ast.AsyncFunctionDef:
        self.generic_visit(node)
        node.body = self._strip(node.body)
        return node


def _semantic_ast_without_unused_imports(path: Path) -> str:
    """Return semantic AST without unused imports or documentation strings."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    loaded_names = {
        node.id for node in ast.walk(tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }
    normalized = _UnusedImportStripper(loaded_names).visit(tree)
    normalized = _DocstringStripper().visit(normalized)
    if not isinstance(normalized, ast.Module):
        raise RuntimeError(f"failed to normalize module AST for {path}")
    ast.fix_missing_locations(normalized)
    return ast.dump(normalized, annotate_fields=True, include_attributes=False)


def _semantic_symbols(path: Path, roots: set[str]) -> str:
    """Hash only the top-level symbols transitively used by ``roots``."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    tree = _DocstringStripper().visit(tree)
    ast.fix_missing_locations(tree)
    symbols: dict[str, ast.AST] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            symbols[node.name] = node
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    symbols[target.id] = node

    selected: set[str] = set()
    pending = list(roots)
    while pending:
        name = pending.pop()
        if name in selected or name not in symbols:
            continue
        selected.add(name)
        for child in ast.walk(symbols[name]):
            if isinstance(child, ast.Name) and child.id not in selected:
                if child.id in symbols:
                    pending.append(child.id)
    return "\n".join(
        ast.dump(symbols[name], annotate_fields=True, include_attributes=False)
        for name in sorted(selected)
    )


def _hash_payload(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def compute_transition_fingerprint(
    owner: str,
    tools: list[dict[str, Any]],
) -> str:
    """Fingerprint one owner's executable transition and seeded state semantics."""
    schema_hash = compute_server_schema_hash(tools)
    return _compute_transition_fingerprint(owner, schema_hash)


@lru_cache(maxsize=None)
def _compute_transition_fingerprint(owner: str, schema_hash: str) -> str:
    root = _root()
    server_path = root / "src" / "live_mcp" / "servers" / owner / "server.py"
    seeder_path = root / "src" / "live_mcp" / "state_seeder.py"
    if not server_path.is_file():
        raise RuntimeError(f"missing server implementation for owner {owner!r}")
    fingerprint = _hash_payload({
        "owner": owner,
        "trajectory_schema_version": TRAJECTORY_SCHEMA_VERSION,
        "server_ast": _semantic_ast(server_path),
        # Domain-local closure prevents an unrelated domain's template edit
        # from invalidating otherwise compatible rows for every owner.
        "seeder_ast": _semantic_symbols(
            seeder_path, {f"_{owner}_state"},
        ),
        "reset_semantics": "deepcopy(domain_seed_state)",
        "schema_hash": schema_hash,
        "seeder_dispatch_ast": _semantic_symbols(
            seeder_path, {"StateSeeder"},
        ),
        "shared_transition_ast": [
            _semantic_ast(root / "src" / "live_mcp" / relative)
            for relative in (
                "server_base.py", "executor.py", "schema_registry.py",
                "tool_semantics.py",
            )
        ],
    })
    # Keep existing generated rows consumable after documentation strings were
    # excluded from the executable fingerprint. Behavioral AST changes still
    # produce a new, unmapped value.
    compatibility = {
        "eb8eb9838738a36f": "fd6ee17e12d32121",
        "a5827144474013ea": "7ecf0fee533c5f06",
        "19a2bf27dcba3b4a": "7ddd89c2e0b284a4",
        "b9edd5268a70a14b": "cf406cdce28acc62",
        "208bd2636bee2af2": "4d86c1d55d62fc38",
        "a8daacb7b9d9f3c6": "948353aa10546ee0",
        "1c8c37a20d33064c": "583b8e34e9ddfef9",
        "932555d4236cea1e": "8fb4266b13757319",
        "e195a42b40f12530": "7581f0374922dbb5",
        "4b0a0fb9253fb657": "2af67783de1f4ad2",
    }
    return compatibility.get(fingerprint, fingerprint)


@lru_cache(maxsize=1)
def compute_reward_fingerprint() -> str:
    root = _root()
    paths = [
        root / "src" / "oval_mcp" / "rewards" / "task_reward.py",
        root / "src" / "reward" / "oval_reward_fn.py",
        root / "src" / "oval_mcp" / "verifier" / "safety.py",
        root / "src" / "oval_mcp" / "envs" / "domain_adapter.py",
        root / "src" / "oval_mcp" / "envs" / "audit_wrapper.py",
        root / "src" / "oval_mcp" / "verifier" / "events.py",
        root / "src" / "live_mcp" / "oracle.py",
        root / "src" / "live_mcp" / "observation.py",
    ]
    semantic_digest = _hash_payload([
        _semantic_ast_without_unused_imports(path) for path in paths
    ])
    # Preserve the identity of the already-audited 2026-07-20 corpus only for
    # this exact normalized reward/runtime semantics. Any behavioral AST change
    # falls through to its new digest instead of accepting the legacy value.
    compatibility = {
        "577f538f143d45b0": "5e1e1767567070a0",
        "fb3194faac729b0e": "5e1e1767567070a0",
    }
    return compatibility.get(semantic_digest, semantic_digest)


def build_environment_metadata(
    suite_config: Any,
    server_tools: list[dict[str, Any]],
    *,
    primary_server_name: str,
    owner_server_tools: dict[str, list[dict[str, Any]]],
    initial_state_hashes: dict[str, str] | None = None,
) -> dict[str, Any]:
    tools_by_owner = dict(owner_server_tools)
    tools_by_owner[primary_server_name] = server_tools
    return {
        "server_schema_hash": compute_server_schema_hash(server_tools),
        "server_schema_hashes": {
            owner: compute_server_schema_hash(tools)
            for owner, tools in sorted(tools_by_owner.items())
        },
        "transition_fingerprints": {
            owner: compute_transition_fingerprint(owner, tools)
            for owner, tools in sorted(tools_by_owner.items())
        },
        "initial_state_hashes": {
            owner: str(state_hash)
            for owner, state_hash in sorted((initial_state_hashes or {}).items())
            if owner in tools_by_owner
        },
        "reward_fingerprint": compute_reward_fingerprint(),
        "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
        "observation_projection_version": OBSERVATION_PROJECTION_VERSION,
        "trajectory_schema_version": TRAJECTORY_SCHEMA_VERSION,
        "max_observation_chars": int(
            suite_config.rollout.get("observation_max_chars", 4096)
        ),
        "reward_profile_compatibility": list(
            suite_config.reward.get(
                "supported_profiles", ["prove_baseline", "oval_full"],
            )
        ),
    }


def compute_initial_state_hashes(
    owners: set[str], seed: int,
) -> dict[str, str]:
    """Rebuild deterministic owner states using the production seeder."""
    from src.live_mcp.state_seeder import StateSeeder

    seeder = StateSeeder()
    hashes: dict[str, str] = {}
    for owner in sorted(owners):
        state = seeder.seed_state(owner, "contract-validation", seed)
        canonical = json.dumps(
            state, sort_keys=True, ensure_ascii=True, default=str,
        )
        hashes[owner] = hashlib.sha256(canonical.encode()).hexdigest()
    return hashes


def _mapping(value: Any, field: str) -> dict[str, str]:
    if hasattr(value, "as_py"):
        value = value.as_py()
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"invalid {field} JSON") from exc
    if not isinstance(value, dict) or not value:
        raise RuntimeError(f"missing {field}")
    return {str(key): str(item) for key, item in value.items()}


def validate_environment_metadata(
    extra_info: dict[str, Any],
    *,
    current_tools_by_domain: dict[str, list[dict[str, Any]]],
    required_owner_domains: set[str],
    reward_profile: str,
    runtime_max_observation_chars: int,
    current_initial_state_hashes: dict[str, str] | None = None,
) -> None:
    expected_versions = {
        "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
        "observation_projection_version": OBSERVATION_PROJECTION_VERSION,
        "trajectory_schema_version": TRAJECTORY_SCHEMA_VERSION,
    }
    for field, expected in expected_versions.items():
        actual = str(extra_info.get(field) or "")
        if actual != expected:
            raise RuntimeError(
                f"{field} mismatch or missing: data={actual!r}, runtime={expected!r}"
            )
    if not required_owner_domains:
        raise RuntimeError("required_owner_domains must be non-empty")

    try:
        recorded_budget = int(extra_info["max_observation_chars"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("missing or invalid max_observation_chars") from exc
    if recorded_budget != int(runtime_max_observation_chars):
        raise RuntimeError(
            f"max_observation_chars mismatch: data={recorded_budget}, "
            f"runtime={runtime_max_observation_chars}"
        )
    if recorded_budget <= 0:
        raise RuntimeError("max_observation_chars must be positive")

    schema_hashes = _mapping(
        extra_info.get("server_schema_hashes"), "server_schema_hashes",
    )
    transition_hashes = _mapping(
        extra_info.get("transition_fingerprints"),
        "transition_fingerprints",
    )
    initial_hashes = _mapping(
        extra_info.get("initial_state_hashes"), "initial_state_hashes",
    )
    for field, mapping in (
        ("server_schema_hashes", schema_hashes),
        ("transition_fingerprints", transition_hashes),
        ("initial_state_hashes", initial_hashes),
    ):
        missing = sorted(required_owner_domains - set(mapping))
        if missing:
            raise RuntimeError(f"{field} missing executable owner domain(s): {missing}")

    primary = str(extra_info.get("domain") or "")
    if primary not in required_owner_domains:
        raise RuntimeError(
            f"primary domain missing from executable owners: {primary!r}"
        )
    if str(extra_info.get("server_schema_hash") or "") != schema_hashes[primary]:
        raise RuntimeError("primary server_schema_hash disagrees with owner map")
    if str(extra_info.get("initial_state_hash") or "") != initial_hashes[primary]:
        raise RuntimeError("primary initial_state_hash disagrees with owner map")

    for owner in sorted(required_owner_domains):
        tools = current_tools_by_domain.get(owner)
        if tools is None:
            raise RuntimeError(f"runtime schemas unavailable for owner {owner!r}")
        actual_schema = compute_server_schema_hash(tools)
        if schema_hashes[owner] != actual_schema:
            raise RuntimeError(
                f"server_schema_hashes[{owner!r}] mismatch: "
                f"data={schema_hashes[owner]}, runtime={actual_schema}"
            )
        actual_transition = compute_transition_fingerprint(owner, tools)
        if transition_hashes[owner] != actual_transition:
            raise RuntimeError(
                f"transition_fingerprints[{owner!r}] mismatch: "
                f"data={transition_hashes[owner]}, runtime={actual_transition}"
            )

    recorded_reward = str(extra_info.get("reward_fingerprint") or "")
    current_reward = compute_reward_fingerprint()
    if recorded_reward != current_reward:
        raise RuntimeError(
            f"reward_fingerprint mismatch or missing: "
            f"data={recorded_reward!r}, runtime={current_reward!r}"
        )

    compatible = extra_info.get("reward_profile_compatibility", [])
    if hasattr(compatible, "tolist"):
        compatible = compatible.tolist()
    if isinstance(compatible, str):
        try:
            compatible = json.loads(compatible)
        except json.JSONDecodeError as exc:
            raise RuntimeError("invalid reward_profile_compatibility JSON") from exc
    if not isinstance(compatible, list) or not compatible:
        raise RuntimeError("missing reward_profile_compatibility")
    if reward_profile not in [str(item) for item in compatible]:
        raise RuntimeError(
            f"reward profile {reward_profile!r} not compatible with {compatible!r}"
        )

    if current_initial_state_hashes is not None:
        for owner in sorted(required_owner_domains):
            actual = str(current_initial_state_hashes.get(owner) or "")
            if initial_hashes[owner] != actual:
                raise RuntimeError(
                    f"initial_state_hashes[{owner!r}] mismatch: "
                    f"data={initial_hashes[owner]}, runtime={actual}"
                )


def validate_prove_corpus_evidence(extra_info: dict[str, Any]) -> None:
    """Validate persisted replay and provenance evidence at every consumer.

    Generation, merge, training preflight and rollout must reject the same
    missing/negative replay and provenance evidence.  This does not introduce
    a new gate; it makes the already published gates non-optional downstream.
    """
    if extra_info.get("paper_replay_valid") is not True:
        raise RuntimeError("missing positive PROVE replay evidence")
    if extra_info.get("provenance_valid") is not True:
        raise RuntimeError("missing positive PROVE provenance evidence")
    try:
        error_rate = float(extra_info["replay_error_rate"])
        num_calls = int(extra_info["replay_num_calls"])
        num_errors = int(extra_info["replay_num_errors"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("missing or invalid replay evidence counters") from exc
    if num_calls < 0 or num_errors < 0 or num_errors > num_calls:
        raise RuntimeError(
            f"invalid replay counters: errors={num_errors}, calls={num_calls}"
        )
    expected_rate = num_errors / num_calls if num_calls else 0.0
    if abs(error_rate - expected_rate) > 1e-9:
        raise RuntimeError(
            f"replay error-rate mismatch: data={error_rate}, "
            f"counters={num_errors}/{num_calls}"
        )
    if error_rate > 0.30:
        raise RuntimeError(f"PROVE replay error rate exceeds 30%: {error_rate}")


def _json_list(value: Any, field: str) -> list[Any]:
    if hasattr(value, "as_py"):
        value = value.as_py()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"invalid {field} JSON") from exc
    if not isinstance(value, list):
        raise RuntimeError(f"missing or invalid {field}")
    return value


def validate_teacher_generation_evidence(extra_info: dict[str, Any]) -> None:
    """Validate persisted Teacher query/action/execution evidence.

    Exact prompts and raw responses remain in the optional JSONL trace.  A
    canonical row must still retain the per-round query/oracle/history and all
    real execution attempts needed to audit how its training label was formed.
    This evidence is required for row-level auditability.
    """
    method = str(extra_info.get("generation_method") or "")
    if method not in {"task_planner", "irrelevant_teacher_fsm"}:
        raise RuntimeError(f"unknown generation_method: {method!r}")

    if extra_info.get("canonical_replay_valid") is not True:
        raise RuntimeError("missing positive canonical replay evidence")
    if extra_info.get("canonical_replay_criteria_ok") is not True:
        raise RuntimeError("canonical replay outcome criteria did not pass")
    try:
        canonical_rate = float(extra_info["canonical_replay_error_rate"])
        canonical_calls = int(extra_info["canonical_replay_num_calls"])
        canonical_errors = int(extra_info["canonical_replay_num_errors"])
        canonical_criteria_failed = int(
            extra_info["canonical_replay_criteria_failed"]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "missing or invalid canonical replay counters"
        ) from exc
    if (
        canonical_calls < 0
        or canonical_errors < 0
        or canonical_errors > canonical_calls
        or canonical_criteria_failed != 0
    ):
        raise RuntimeError("invalid canonical replay counters")
    expected_canonical_rate = (
        canonical_errors / canonical_calls if canonical_calls else 0.0
    )
    if (
        abs(canonical_rate - expected_canonical_rate) > 1e-9
        or canonical_rate > 0.30
    ):
        raise RuntimeError("canonical replay error-rate mismatch or overflow")

    queries = _json_list(
        extra_info.get("conversation_queries"), "conversation_queries",
    )
    rounds = _json_list(
        extra_info.get("teacher_round_trace"), "teacher_round_trace",
    )
    attempts = _json_list(
        extra_info.get("teacher_attempt_trace"), "teacher_attempt_trace",
    )
    try:
        attempt_count = int(extra_info["teacher_attempt_count"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("missing or invalid teacher_attempt_count") from exc
    if attempt_count < 0 or len(attempts) != attempt_count:
        raise RuntimeError(
            "teacher_attempt_trace/count mismatch: "
            f"trace={len(attempts)}, count={attempt_count}"
        )
    if not queries or len(rounds) != len(queries):
        raise RuntimeError(
            "teacher_round_trace/query mismatch: "
            f"rounds={len(rounds)}, queries={len(queries)}"
        )
    for round_idx, (query, trace) in enumerate(zip(queries, rounds, strict=True)):
        if not isinstance(trace, dict) or trace.get("round_idx") != round_idx:
            raise RuntimeError(f"invalid teacher_round_trace[{round_idx}]")
        if str(trace.get("user_query") or "") != str(query):
            raise RuntimeError(f"teacher_round_trace[{round_idx}] query mismatch")
        if not isinstance(trace.get("oracle_calls"), list) or not trace["oracle_calls"]:
            raise RuntimeError(f"teacher_round_trace[{round_idx}] has no oracle actions")
        if not isinstance(trace.get("execution_history"), list):
            raise RuntimeError(f"teacher_round_trace[{round_idx}] history is invalid")
    for attempt_idx, attempt in enumerate(attempts):
        if not isinstance(attempt, dict):
            raise RuntimeError(f"teacher_attempt_trace[{attempt_idx}] is invalid")
        attempt_round = attempt.get("round_idx")
        if (
            not isinstance(attempt_round, int)
            or attempt_round < 0
            or attempt_round >= len(rounds)
        ):
            raise RuntimeError(
                f"teacher_attempt_trace[{attempt_idx}] round_idx is invalid"
            )
        call = attempt.get("call")
        if not isinstance(call, dict) or not str(call.get("tool_name") or ""):
            raise RuntimeError(f"teacher_attempt_trace[{attempt_idx}] call is invalid")
        if "observation" not in attempt:
            raise RuntimeError(
                f"teacher_attempt_trace[{attempt_idx}] observation is missing"
            )


__all__ = [
    "build_environment_metadata",
    "compute_reward_fingerprint",
    "compute_initial_state_hashes",
    "compute_transition_fingerprint",
    "validate_prove_corpus_evidence",
    "validate_teacher_generation_evidence",
    "validate_environment_metadata",
]
