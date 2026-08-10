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

from src.live_mcp.config import project_root
from src.live_mcp.protocol.observation import (
    TRAJECTORY_SCHEMA_VERSION,
    OBSERVATION_SCHEMA_VERSION,
    OBSERVATION_PROJECTION_VERSION,
    compute_server_schema_hash,
)


def _root() -> Path:
    return project_root()


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


def _semantic_symbols(
    path: Path,
    roots: set[str],
    *,
    follow_dependencies: bool = True,
) -> str:
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
        if follow_dependencies:
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
    from src.live_mcp.state_seeder import available_state_profiles

    root = _root()
    server_path = root / "src" / "live_mcp" / "servers" / owner / "server.py"
    seeder_path = root / "src" / "live_mcp" / "state_seeder.py"
    domain_seeder_path = (
        root / "src" / "live_mcp" / "state_seeders" / f"{owner}.py"
    )
    if not server_path.is_file():
        raise RuntimeError(f"missing server implementation for owner {owner!r}")
    fingerprint = _hash_payload({
        "owner": owner,
        "trajectory_schema_version": TRAJECTORY_SCHEMA_VERSION,
        "server_ast": _semantic_ast(server_path),
        # Each domain module contains its templates and builder together, so
        # an unrelated domain edit does not invalidate this owner while every
        # executable domain-local state fact remains fingerprinted.
        "seeder_ast": _semantic_ast(domain_seeder_path),
        "reset_semantics": "deepcopy(domain_seed_state)",
        "schema_hash": schema_hash,
        "seeder_dispatch_ast": _semantic_symbols(
            seeder_path, {"StateSeeder"}, follow_dependencies=False,
        ),
        "available_state_profiles": available_state_profiles(owner),
        "shared_transition_ast": [
            _semantic_ast(root / "src" / "live_mcp" / relative)
            for relative in (
                "server_base.py", "executor.py",
                "state_seeders/common.py",
                "registry/schemas.py",
                "registry/tool_semantics.py",
            )
        ],
    })
    return fingerprint


@lru_cache(maxsize=1)
def compute_reward_fingerprint(
    reward_profile: str = "prove_baseline",
) -> str:
    if reward_profile not in {"prove_baseline", "oval_full"}:
        raise ValueError(f"unknown reward profile: {reward_profile!r}")
    root = _root()
    paths = [
        root / "src" / "oval_mcp" / "rewards" / "task_reward.py",
        root / "src" / "reward" / "oval_reward_fn.py",
    ]
    if reward_profile == "oval_full":
        paths.append(root / "src" / "oval_mcp" / "verifier" / "safety.py")
    paths.extend([
        root / "src" / "oval_mcp" / "envs" / "domain_adapter.py",
        root / "src" / "oval_mcp" / "envs" / "audit_wrapper.py",
        root / "src" / "oval_mcp" / "verifier" / "events.py",
        root / "src" / "live_mcp" / "oracle.py",
        root / "src" / "live_mcp" / "protocol" / "observation.py",
    ])
    return _hash_payload([
        _semantic_ast_without_unused_imports(path) for path in paths
    ])


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
    state_profiles = state_profiles_for_suite(
        suite_config, set(tools_by_owner)
    )
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
        "state_profiles": state_profiles,
        # The scalar field is the canonical PROVE corpus identity; the mapping
        # binds every supported training profile explicitly.
        "reward_fingerprint": compute_reward_fingerprint("prove_baseline"),
        "reward_profile_fingerprints": {
            profile: compute_reward_fingerprint(profile)
            for profile in ("prove_baseline", "oval_full")
        },
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
    owners: set[str],
    seed: int,
    state_profiles: dict[str, str] | None = None,
) -> dict[str, str]:
    """Rebuild deterministic owner states using the production seeder."""
    from src.live_mcp.state_seeder import StateSeeder

    seeder = StateSeeder()
    hashes: dict[str, str] = {}
    resolved_profiles = normalize_state_profiles(
        state_profiles or {owner: "baseline" for owner in owners}, owners,
    )
    for owner in sorted(owners):
        state = seeder.seed_state(
            owner,
            "contract-validation",
            seed,
            resolved_profiles[owner],
        )
        canonical = json.dumps(
            state, sort_keys=True, ensure_ascii=True, default=str,
        )
        hashes[owner] = hashlib.sha256(canonical.encode()).hexdigest()
    return hashes


def state_profiles_for_suite(
    suite_config: Any,
    owners: set[str] | None = None,
) -> dict[str, str]:
    """Return explicit per-owner profiles, defaulting to baseline."""
    requested_owners = set(owners or ())
    profiles = {
        str(cfg.name): str(cfg.session.get("state_profile", "baseline"))
        for cfg in suite_config.servers
        if cfg.enabled and (not requested_owners or cfg.name in requested_owners)
    }
    missing = sorted(requested_owners - set(profiles))
    if missing:
        raise RuntimeError(f"suite missing state profile owner(s): {missing}")
    return dict(sorted(profiles.items()))


def normalize_state_profiles(
    value: Any,
    owners: set[str],
) -> dict[str, str]:
    """Normalize explicit per-owner state-profile metadata."""
    if hasattr(value, "as_py"):
        value = value.as_py()
    if value in (None, ""):
        raise RuntimeError("missing state_profiles")
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError as exc:
            raise RuntimeError("invalid state_profiles JSON") from exc
        if not isinstance(decoded, dict):
            raise RuntimeError("state_profiles must be a mapping")
        raw = decoded
    elif isinstance(value, dict):
        raw = value
    else:
        raise RuntimeError("state_profiles must be a mapping")
    if set(raw) != owners:
        raise RuntimeError(
            "state_profiles must exactly cover owner domains: "
            f"profiles={sorted(raw)}, owners={sorted(owners)}"
        )
    profiles = {owner: str(raw[owner]) for owner in sorted(owners)}
    if any(not profile for profile in profiles.values()):
        raise RuntimeError("state_profiles values must be non-empty")
    return profiles


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
    state_profiles = normalize_state_profiles(
        extra_info.get("state_profiles"), required_owner_domains
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

    profile_fingerprints = extra_info.get("reward_profile_fingerprints", {})
    if hasattr(profile_fingerprints, "as_py"):
        profile_fingerprints = profile_fingerprints.as_py()
    if isinstance(profile_fingerprints, str):
        try:
            profile_fingerprints = json.loads(profile_fingerprints)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "invalid reward_profile_fingerprints JSON"
            ) from exc
    if profile_fingerprints and not isinstance(profile_fingerprints, dict):
        raise RuntimeError("reward_profile_fingerprints must be a mapping")
    recorded_reward = str(
        (profile_fingerprints or {}).get(reward_profile)
        or extra_info.get("reward_fingerprint")
        or ""
    )
    current_reward = compute_reward_fingerprint(reward_profile)
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


def validate_tool_owner_contract(extra_info: dict[str, Any]) -> dict[str, str]:
    """Bind every policy-visible bare tool name to exactly one server owner."""
    visible_raw = _json_list(
        extra_info.get("visible_tool_names"), "visible_tool_names",
    )
    if any(
        not isinstance(value, str) or not value.strip()
        for value in visible_raw
    ):
        raise RuntimeError("visible_tool_names must contain non-empty strings")
    visible_names = [value.strip() for value in visible_raw]
    if not visible_names:
        raise RuntimeError("visible_tool_names must not be empty")
    if len(visible_names) != len(set(visible_names)):
        raise RuntimeError("policy-visible tool names must be globally unique")

    owners_raw = extra_info.get("tool_owner_domains")
    if hasattr(owners_raw, "as_py"):
        owners_raw = owners_raw.as_py()
    if isinstance(owners_raw, str):
        try:
            owners_raw = json.loads(owners_raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError("invalid tool_owner_domains JSON") from exc
    if not isinstance(owners_raw, dict):
        raise RuntimeError("missing or invalid tool_owner_domains")
    if any(
        not isinstance(name, str) or not name.strip()
        or not isinstance(owner, str) or not owner.strip()
        for name, owner in owners_raw.items()
    ):
        raise RuntimeError(
            "tool_owner_domains keys and owners must be non-empty strings"
        )
    owners = {
        name.strip(): owner.strip() for name, owner in owners_raw.items()
    }
    missing = sorted(set(visible_names) - set(owners))
    extra = sorted(set(owners) - set(visible_names))
    if missing or extra:
        raise RuntimeError(
            "tool_owner_domains must exactly cover visible_tool_names: "
            f"missing={missing}, extra={extra}"
        )
    return owners


def validate_semantic_gate_evidence(extra_info: dict[str, Any]) -> None:
    """Re-evaluate the persisted local semantic disposition at every consumer.

    Per OVAL-MCP.md §5, semantic quarantine is a LOCAL contract: only a
    provable contradiction (hard gate) under the deterministic profile aborts
    consumption.  Subjective-naturalness diagnostics (hard_gate=False) are
    recorded but never block the train/rollout pipeline.
    """
    from src.live_mcp.corpus.semantic_core import resolve_semantic_gate_profile
    from src.live_mcp.corpus.semantic_quarantine import (
        evaluate_semantic_quarantine,
    )

    try:
        profile = resolve_semantic_gate_profile(extra_info)
    except ValueError as exc:
        raise RuntimeError(f"invalid semantic_gate_profile: {exc}") from exc

    issue = evaluate_semantic_quarantine(extra_info)
    if issue is not None and issue.hard_gate and profile == "deterministic_v1":
        raise RuntimeError(issue.quality_issue)


def validate_training_artifact_evidence(extra_info: dict[str, Any]) -> None:
    """Reject audit/experiment rows at the rollout and training boundary."""
    from src.live_mcp.corpus.semantic_core import validate_artifact_purpose

    try:
        validate_artifact_purpose(extra_info, require_training=True)
    except ValueError as exc:
        raise RuntimeError(f"non-training artifact: {exc}") from exc


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
    teacher_model_id = str(extra_info.get("teacher_model_id") or "").strip()
    if not teacher_model_id:
        raise RuntimeError("missing teacher_model_id provenance")
    validate_tool_owner_contract(extra_info)
    from src.live_mcp.artifact.dependency_contract import (
        validate_dependency_artifact,
    )

    validate_dependency_artifact(extra_info)

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
    "normalize_state_profiles",
    "state_profiles_for_suite",
    "validate_prove_corpus_evidence",
    "validate_semantic_gate_evidence",
    "validate_training_artifact_evidence",
    "validate_teacher_generation_evidence",
    "validate_environment_metadata",
]
